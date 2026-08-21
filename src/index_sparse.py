"""
=============================================================================
PHASE 2 (part 3 of 3) — Sparse index (BM25)
=============================================================================

WHAT YOU'RE BUILDING
    artifacts/chunks.jsonl -> a BM25 lexical index.

WHY A SECOND INDEX AT ALL
    You just built a dense index that understands meaning. Why add a 1994-era
    keyword algorithm on top?

    Because embeddings are lossy in exactly the place this corpus is dense with
    information. Ask an embedding model to distinguish:

        "NB-1"   vs  "NB-2"
        "E101"   vs  "E204"   vs  "E317"
        "36-month" vs "48-month"

    These are near-identical strings with unrelated meanings, and they compress
    to nearly the same vector. Meanwhile BM25 treats "e204" as an atom that
    either appears in a chunk or does not — no interpolation, no semantics, no
    confusion. On identifier lookups it is not merely competitive with dense
    retrieval; it is strictly better.

    This is not a hypothetical concern on this corpus: 53 of the 100 questions
    in eval/golden_set.json hinge on telling NB-1 from NB-2, and the two product
    spec sheets are near-identical documents differing mainly in their numbers.

    The reverse is also true (BM25 scores 0 on a paraphrase with no shared
    words), which is why Phase 3 fuses them rather than picking one.

WHAT TO IMPLEMENT
    Two stubs: tokenize(), search_bm25().

RUN IT
    uv run python -m src.index_sparse
    uv run python -m src.index_sparse --self-check
    uv run python -m src.index_sparse --search "E204"

DEPENDS ON: Phase 1 output only. NO SERVER NEEDED — you can do this offline.
=============================================================================
"""

from __future__ import annotations

import pickle
import re
import sys

from rank_bm25 import BM25Okapi

from src.config import Config, load_config
from src.ingest import load_chunks

from pprint import pprint

# =============================================================================
# TODO(Phase 2) #5 — tokenize
# =============================================================================


def tokenize(text: str) -> list[str]:
    """Split text into the terms BM25 will match on.

    ---------------------------------------------------------------------------
    THIS FUNCTION IS THE WHOLE BALL GAME
    ---------------------------------------------------------------------------
    BM25 can only match tokens that your tokenizer produced. So the tokenizer
    decides what BM25 is capable of finding, and the obvious implementation
    destroys the exact advantage you came here for.

    Watch what the standard `re.findall(r"\\w+", text.lower())` does:

        "NB-1"     -> ["nb", "1"]        <- the important one
        "36-month" -> ["36", "month"]
        "99.2%"    -> ["99", "2"]
        "$18,500"  -> ["18", "500"]
        "E204"     -> ["e204"]           (survives, by luck of having no punctuation)
        "IP65"     -> ["ip65"]           (survives, same reason)

    Look at the first line. "NB-1" became the terms "nb" and "1". "nb" appears
    in every chunk of both product documents and so carries almost no IDF
    weight; "1" is a bare digit colliding with every table row in the corpus.
    You have just made NB-1 and NB-2 — the two documents 53 of the 100 golden
    questions are about — nearly indistinguishable to the one retriever whose
    entire purpose was precision.

    Note that E204 and IP65 survive intact, because they happen to contain no
    punctuation. Do not let that fool you into thinking the naive regex is fine;
    it is correct by accident on those two and wrong on the case that matters.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Lowercase. (BM25 is case-sensitive by default; "IP65" and "ip65" would
         be different terms.)

      2. Extract tokens with a regex that KEEPS internal punctuation inside
         identifier-shaped runs. A workable pattern:

             r"[a-z0-9]+(?:[-_./%][a-z0-9]+)*"

         Read it as: an alphanumeric run, optionally followed by more runs glued
         on by a single -, _, ., / or %. That preserves "nb-1", "36-month",
         "99.2", "v1.5", "e204", "ip65" while still splitting on whitespace.

      3. NOTICE WHAT THIS PATTERN STILL GETS WRONG. "$18,500" comes out as
         ["18", "500"] — the comma is not in the glue set. Adding "," would fix
         the price but would also glue "1,2" in ordinary prose. There is no
         clean answer; there is only a documented one. Decide, and write the
         decision in a comment.

      4. DECIDE the other ambiguity this creates:
           * "nb-1" is now one token, so a user typing "NB 1" or "nb1" will NOT
             match it. Acceptable? (For a product code in a spec-sheet corpus:
             probably yes.)
           * Consider emitting BOTH the glued token and its parts — index
             "nb-1" AND "nb" AND "1". This costs index size and dilutes IDF, but
             makes the retriever forgiving of how the user types the code.

      5. Consider stripping Markdown table pipes and stopwords. Careful with
         stopwords on this corpus: "V" (volts) and "A" (amps) look like noise to
         a generic stopword list.

      6. Whatever you decide: use THIS SAME FUNCTION for both indexing and
         querying. A mismatch between the two tokenizers produces an index that
         silently never matches anything, and it is a genuinely hard bug to see
         because every step looks like it worked.

    Returns: list[str] of terms.
    """
    # TODO(Phase 2): implement per the recipe above.
    
    text = text.lower()
    # Commas are deliberately separators: preserving them would glue ordinary
    # prose such as "1,2" into one term.
    return re.findall(r"[a-z0-9]+(?:[-_./%][a-z0-9]+)*", text)


# =============================================================================
# TODO(Phase 2) #6 — search_bm25
# =============================================================================


def search_bm25(bm25: BM25Okapi, chunk_ids: list[str], query: str, k: int
                ) -> list[tuple[str, float]]:
    """Score every chunk against the query and return the top k.

    ---------------------------------------------------------------------------
    THE FORMULA (you don't implement it — rank_bm25 does — but you should be
    able to explain every symbol in a design review)
    ---------------------------------------------------------------------------

        score(D, Q) = SUM over terms t in Q of

                                       tf(t,D) * (k1 + 1)
              IDF(t)  *  ------------------------------------------------
                         tf(t,D) + k1 * (1 - b + b * |D| / avgdl)

      tf(t,D)   how many times term t appears in chunk D.

      IDF(t)    INVERSE DOCUMENT FREQUENCY, roughly log(N / df(t)) where N is
                the number of chunks and df(t) is how many contain t. This is
                the part that makes the algorithm smart: a term appearing in
                every chunk ("the", "robot") gets an IDF near zero and
                contributes nothing, while "e-204" appearing in one chunk out
                of 45 gets a large IDF and dominates the score. Rare terms are
                informative terms. You get this for free — but only if your
                tokenizer kept "e-204" as one token.

      k1        TERM FREQUENCY SATURATION (default 1.5). Without it, a chunk
                mentioning "payload" ten times would score 10x one mentioning it
                once. That is wrong — the second mention tells you much less
                than the first. The formula's shape means score rises steeply
                for the first occurrences then flattens toward an asymptote.
                k1=0 ignores frequency entirely; large k1 approaches linear.

      b         LENGTH NORMALIZATION (default 0.75), with |D| the chunk's length
                and avgdl the average across the corpus. A long chunk is more
                likely to contain any given term by chance, so its score is
                discounted. b=0 disables the correction; b=1 applies it fully.
                THIS IS WHY tiny chunks are dangerous — see the comment in
                ingest.py::merge_small_sections.

      Note what is absent: any notion of meaning, word order, or synonyms. BM25
      scores "How rapid is the courier?" against "Top speed | 24 km/h" at
      exactly zero. That blind spot is the dense index's job.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. tokens = tokenize(query)          <- the SAME tokenizer as indexing
      2. scores = bm25.get_scores(tokens)  <- numpy array, one score per chunk,
                                              in the order the corpus was added
      3. Pair each score with chunk_ids[i], sort descending, take k.
      4. Return [(chunk_id, float(score)), ...].

      Edge case worth handling: if NO query term appears anywhere in the corpus,
      every score is 0.0 and the "top k" is an arbitrary tie. Returning them is
      harmless (Phase 3's fusion will handle it), but be aware that a BM25 score
      of 0 means "no lexical evidence at all", not "weak evidence".

      Unlike cosine, BM25 scores are UNBOUNDED and corpus-dependent — a score of
      8.3 means nothing on its own and is not comparable to another corpus's
      8.3. That non-comparability is precisely why Phase 3 fuses by RANK rather
      than by score.
    """
    # TODO(Phase 2): implement per the recipe above.
    
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    
    # print(scores, chunk_ids)
    
    chunkToScore = list( (chunk_id, float(score)) for chunk_id, score in zip(chunk_ids, scores))
    
    chunkToScore.sort(key=lambda x: x[1], reverse=True)

    # pprint(chunkToScore)
    return chunkToScore
    # raise NotImplementedError("Phase 2: search_bm25")


# =============================================================================
# GIVEN: build / save / load
# =============================================================================


def build(cfg: Config) -> None:
    """Tokenize every chunk, build BM25Okapi, pickle it alongside the id list."""
    chunks = load_chunks(cfg.path("chunks"))
    corpus_tokens = [tokenize(c.text) for c in chunks]

    bm25 = BM25Okapi(corpus_tokens)
    chunk_ids = [c.chunk_id for c in chunks]

    out = cfg.path("bm25_index")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Pickle, not JSON: BM25Okapi precomputes IDF tables and document lengths at
    # construction time, and we want to persist that work rather than redo it.
    # (Pickle is fine for a local artifact you generate yourself. Never unpickle
    # a file you did not produce — it executes arbitrary code on load.)
    with out.open("wb") as fh:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, fh)

    vocab = {t for toks in corpus_tokens for t in toks}
    avg_len = sum(len(t) for t in corpus_tokens) / len(corpus_tokens)
    print(f"indexed {len(chunks)} chunks -> {out.name}")
    print(f"  vocabulary: {len(vocab)} unique terms")
    print(f"  avgdl (average chunk length in tokens): {avg_len:.1f}")

    # Print a few identifier-shaped terms so you can SEE whether your tokenizer
    # preserved them. If "e-204" is missing here, your tokenizer shattered it.
    interesting = sorted(t for t in vocab if any(ch.isdigit() for ch in t) and "-" in t)
    print(f"  identifier-shaped terms kept intact ({len(interesting)}): "
          f"{interesting[:12]}")


def load_sparse(cfg: Config) -> tuple[BM25Okapi, list[str]]:
    p = cfg.path("bm25_index")
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run `python -m src.index_sparse` first."
        )
    with p.open("rb") as fh:
        d = pickle.load(fh)
    return d["bm25"], d["chunk_ids"]


def search(cfg: Config, query: str, k: int) -> list[tuple[str, float]]:
    """Sparse retrieval entry point used by Phase 3."""
    bm25, chunk_ids = load_sparse(cfg)
    return search_bm25(bm25, chunk_ids, query, k)


# =============================================================================
# GIVEN: self-check
# =============================================================================


def self_check(cfg: Config) -> None:
    chunks = {c.chunk_id: c for c in load_chunks(cfg.path("chunks"))}
    bm25, chunk_ids = load_sparse(cfg)

    # --- 1. reload integrity ------------------------------------------------
    assert len(chunk_ids) == len(chunks), (
        f"index has {len(chunk_ids)} entries but chunks.jsonl has {len(chunks)}. "
        "Rebuild the index."
    )
    print(f"  [1/4] ok  {len(chunk_ids)} chunks reloaded")

    # --- 2. the tokenizer preserves identifiers ----------------------------
    toks = tokenize("Error E204 on the NB-1 under a 36-month term, IP65, 99.2%")
    assert "nb-1" in toks, (
        f"tokenize() shattered 'NB-1' -> {toks}. This is the case that matters: "
        "53 of the 100 golden questions depend on telling NB-1 from NB-2. See "
        "the docstring."
    )
    assert "36-month" in toks, f"tokenize() shattered '36-month' -> {toks}"
    assert "e204" in toks and "ip65" in toks, f"lost a plain identifier: {toks}"
    print(f"  [2/4] ok  identifiers survive tokenization: {toks}")

    # --- 3. exact-identifier lookup ----------------------------------------
    # This is the query that justifies the whole file. Compare its dense rank
    # in Phase 3 to see how much BM25 is doing for you. (E204 is one of the four
    # error codes in faq-troubleshooting.md.)
    hits = search(cfg, "E204", 3)
    assert hits and hits[0][1] > 0, "'E204' scored 0 everywhere — check tokenize()"
    top_doc = chunks[hits[0][0]].doc
    assert top_doc == "faq-troubleshooting.md", (
        f"'E204' ranked {top_doc} first, expected faq-troubleshooting.md. "
        f"Top 3: {[(cid, round(s, 2)) for cid, s in hits]}"
    )
    print(f"  [3/4] ok  'E204' -> {hits[0][0]} (score {hits[0][1]:.2f})")

    # --- 4. BM25's known blind spot ----------------------------------------
    # We ASSERT the weakness, so it is documented in the code rather than
    # discovered in Phase 6.
    #
    # Note we measure it by RANK, not by score. Comparing this query's score to
    # the E204 query's score above would be meaningless — BM25 scores are
    # unbounded and query-dependent, which is exactly the point made in
    # search_bm25's docstring and exactly why Phase 3 fuses by rank.
    target = next(c for c in chunks.values()
                  if c.doc == "product-nb1.md" and "Top speed" in c.body)
    para_q = "how quickly does the courier travel"
    ranked = [cid for cid, _ in search(cfg, para_q, len(chunk_ids))]
    rank = ranked.index(target.chunk_id) + 1
    assert rank > 5, (
        f"expected BM25 to rank the answer chunk poorly for a paraphrase, but "
        f"it came {rank}. (Not a failure of your code — just means this check "
        "no longer demonstrates anything; pick a query with less vocabulary "
        "overlap.)"
    )
    print(f"  [4/4] ok  {para_q!r} ranks the chunk that literally answers it "
          f"({target.chunk_id}, 'Top speed | 24 km/h') at #{rank} of "
          f"{len(chunk_ids)} — the query shares no content words with it. "
          "THIS is the gap the dense index exists to fill.")

    print("\nPHASE 2 (sparse) SELF-CHECK PASSED")


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    cfg = load_config()
    try:
        if "--self-check" in sys.argv:
            self_check(cfg)
        elif "--search" in sys.argv:
            query = sys.argv[sys.argv.index("--search") + 1]
            chunks = {c.chunk_id: c for c in load_chunks(cfg.path("chunks"))}
            print(f"query: {query!r}  tokens: {tokenize(query)}\n")
            for rank, (cid, score) in enumerate(search(cfg, query, 5), start=1):
                c = chunks[cid]
                print(f"{rank}. {score:7.3f}  {cid}  ({' > '.join(c.heading_path)})")
                print(f"     {c.body[:110].replace(chr(10), ' ')} ...")
        else:
            build(cfg)
            print("\nNext: uv run python -m src.index_sparse --self-check")
    except FileNotFoundError as exc:
        print(f"Missing prerequisite: {exc}")
        return 1
    except NotImplementedError as exc:
        print(f"Not implemented yet: {exc}")
        print("\nPhase 2 (sparse) stubs, in order:")
        print("  5. tokenize()     src/index_sparse.py")
        print("  6. search_bm25()  src/index_sparse.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
