"""
=============================================================================
PHASE 3 — Hybrid retrieval with Reciprocal Rank Fusion
=============================================================================

WHAT YOU'RE BUILDING
    Two ranked lists (dense, BM25) -> one ranked list that is better than
    either.

THE PROBLEM YOU CANNOT SOLVE THE OBVIOUS WAY
    Your first instinct is a weighted sum:

        final = 0.5 * cosine_score + 0.5 * bm25_score

    This does not work, and it is worth understanding exactly why:

      * cosine scores live in [-1, 1] and, in practice, cluster tightly around
        0.4-0.8 for anything vaguely on-topic.
      * BM25 scores are UNBOUNDED. They depend on corpus size, IDF of the query
        terms, and chunk lengths. A rare-term query might score 14.0; a common-
        term query on the same corpus might top out at 1.2.

    Adding these is adding degrees Celsius to kilometres. Worse, the imbalance
    is not even constant — it changes per query, so no fixed weight fixes it.

    You could min-max normalize each list per query. People do. But that makes
    scores depend on the worst result in the batch, so adding one bad candidate
    silently reshuffles the good ones.

THE ANSWER: throw the scores away and keep only the ORDER.
    That is Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).
    It is ten lines long, has one parameter, needs no training or calibration,
    and routinely beats carefully tuned score blending. It is one of the best
    effort-to-payoff ratios in information retrieval.

WHAT TO IMPLEMENT
    Two stubs: fuse_rrf(), hybrid_retrieve().

RUN IT
    uv run python -m src.retrieve "What is error code E204?"
    uv run python -m src.retrieve "How long does the NB-1 take to fast charge?"

YOUR PHASE 3 DELIVERABLE
    The CLI prints all three rankings side by side and flags "RESCUE" whenever
    a chunk that dense retrieval buried gets pulled into the top-k by BM25.
    Find one such query and write it down — it is the evidence in your README
    that hybrid retrieval earned its complexity. Candidates on this corpus:
    "E204", "IP65", "36-month", "MTBF", "NB-2 payload".

DEPENDS ON: Phase 1 + Phase 2.
=============================================================================
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from src.config import Config, load_config
from src.index_dense import search as dense_search
from src.index_sparse import search as sparse_search
from src.ingest import Chunk, load_chunks

from pprint import pprint

# =============================================================================
# GIVEN: result type
# =============================================================================


@dataclass
class RetrievedChunk:
    """One fused result, carrying provenance from BOTH arms.

    Keeping the per-arm ranks (rather than just the final score) is a deliberate
    debugging affordance: when a retrieval goes wrong you immediately see
    WHICH retriever was responsible. Phase 6's ablation table is built on
    exactly this information.

    `dense_rank` / `sparse_rank` are 1-indexed, or None if the chunk did not
    appear in that arm's candidate pool at all.
    """

    chunk: Chunk
    rrf_score: float
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


# =============================================================================
# TODO(Phase 3) #1 — fuse_rrf
# =============================================================================


def fuse_rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    ---------------------------------------------------------------------------
    THE FORMULA
    ---------------------------------------------------------------------------

        RRF(d) = SUM over ranked lists L of    1 / (k + rank_L(d))

    where rank_L(d) is d's 1-INDEXED position in list L, and d contributes
    nothing for a list it does not appear in.

    Worked example with k=60. A chunk at rank 1 in dense and rank 8 in sparse:

        1/(60+1) + 1/(60+8) = 0.01639 + 0.01471 = 0.03110

    A chunk at rank 2 in dense and absent from sparse:

        1/(60+2)            = 0.01613

    So appearing decently in BOTH lists beats appearing excellently in one.
    That is the entire behavior you want from a fusion rule, achieved without
    a single tunable weight.

    ---------------------------------------------------------------------------
    WHAT k DOES
    ---------------------------------------------------------------------------
    k is a DAMPING constant. Look at the gap between consecutive ranks:

        k=0   -> 1/1, 1/2, 1/3 ...     rank 1 is worth 2x rank 2. One list's
                                       top hit can dominate everything.
        k=60  -> 1/61, 1/62, 1/63 ...  ranks 1 and 2 differ by 0.03%. Positions
                                       matter, but only gently; agreement
                                       ACROSS lists matters more.

    Large k says "I trust the consensus more than I trust any single
    retriever's confidence." That is the right prior when your retrievers have
    complementary blind spots — which is exactly the situation you engineered
    in Phase 2. 60 is the paper's value and is a fine default; try k=1 in the
    CLI afterwards and watch the ranking get twitchier.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. scores: dict[str, float] = {}
      2. for each ranking list:
             for rank, doc_id in enumerate(ranking, start=1):     # 1-INDEXED
                 scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
         Using enumerate(..., start=0) is the classic bug here: it makes rank 0
         and rank 1 differ, changing every score subtly rather than breaking
         loudly.
      3. Return sorted(scores.items(), key=lambda kv: -kv[1]) — descending.
      4. Ties: chunks appearing at the same ranks in the same lists get
         identical scores. Python's sort is stable, so they come out in
         insertion order (i.e. dense-first, since that's the first list you
         pass). Fine — just know it is arbitrary, not meaningful.

    Note this function knows nothing about embeddings, BM25, or even documents.
    It fuses ORDERED LISTS OF IDS. You could add a third arm (a reranker, a
    title-match heuristic) by appending one more list, with no other changes.

    Returns: [(doc_id, rrf_score), ...] sorted by score descending.
    """
    # TODO(Phase 3): implement per the recipe above.

    # pprint(rankings)

    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (
                k + rank
            )  # so if the score is not there to begin w default to zero

    return sorted(
        scores.items(), key=lambda x: -x[1]
    )  # descending sort on the score for each doc


#     raise NotImplementedError("Phase 3: fuse_rrf")


# =============================================================================
# TODO(Phase 3) #2 — hybrid_retrieve
# =============================================================================


def hybrid_retrieve(
    cfg: Config, query: str, k: int | None = None, overfetch: int | None = None
) -> list[RetrievedChunk]:
    """Run both retrievers, fuse with RRF, return the top k as RetrievedChunks.

    ---------------------------------------------------------------------------
    WHY YOU OVER-FETCH BEFORE FUSING
    ---------------------------------------------------------------------------
    Suppose you want k=8 final results and you ask each arm for its top 8.
    A chunk ranked #12 by dense but #2 by BM25 — precisely the "BM25 rescue"
    case this whole phase exists to capture — never enters the dense candidate
    pool, so fusion never sees the disagreement it was built to resolve.

    So each arm fetches k * overfetch candidates (config default: 8 * 3 = 24),
    fusion considers the union, and only then do you trim to k. Retrieval is
    cheap; the expensive step is generation, and that only ever sees k.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Default k and overfetch from cfg.retrieval when None.
      2. n = k * overfetch
      3. dense  = dense_search(cfg, query, n)    -> [(chunk_id, cosine), ...]
         sparse = sparse_search(cfg, query, n)   -> [(chunk_id, bm25), ...]
         (Both are already written and imported at the top of this file.)
      4. Build lookup dicts so you can recover per-arm rank and score:
             dense_rank = {cid: i for i, (cid, _) in enumerate(dense, start=1)}
             dense_score = dict(dense)
         ... and the same for sparse.
      5. fused = fuse_rrf([[cid for cid, _ in dense],
                           [cid for cid, _ in sparse]],
                          k=cfg.retrieval.rrf_k)
      6. Load chunk bodies: `{c.chunk_id: c for c in load_chunks(cfg.path("chunks"))}`
         and assemble RetrievedChunk objects for the top k of `fused`, filling in
         dense_rank / sparse_rank / scores via .get() (None when absent).
      7. Return the list, already ordered by rrf_score descending.

    PERFORMANCE NOTE, and a deliberate simplification: this reloads the chunks,
    the vector matrix and the BM25 pickle on every call. At 45 chunks that is
    milliseconds and the clarity is worth it. In Phase 6 you will call this 100
    times per pipeline and may want to hoist the loading into a class — but
    measure first. Guessing at what is slow is how you end up optimizing the
    wrong thing.
    """
    # TODO(Phase 3): implement per the recipe above.

    if k == None:
        k = cfg.retrieval.top_k

    if overfetch == None:
        overfetch = cfg.retrieval.overfetch

    n = k * overfetch

    # now actually call both dense and sparse, to get the ranks
    # then pass a list of lists containing both to rrf

    dense = dense_search(cfg, query, n)  # already sorted by scores
    sparse = sparse_search(cfg, query, n)

    # pprint(dense)

    dense_rank = {cid: i for i, (cid, _) in enumerate(dense, start=1)}
    dense_score = dict(dense)
    # pprint(dense_score)

    sparse_rank = {cid: i for i, (cid, _) in enumerate(sparse, start=1)}
    sparse_score = dict(sparse)
    # print(type(sparse_rank))

    rankings = [[cid for cid, _ in dense], [cid for cid, _ in sparse]]
    # pprint(rankings)

    fused = fuse_rrf(rankings, k=cfg.retrieval.rrf_k)

    chunks = {c.chunk_id: c for c in load_chunks(cfg.path("chunks"))}

    # pprint(rrf)
    # pprint(chunks)

    retrievedChunks: list[RetrievedChunk] = []

    for fuse in fused:
        chunkName = fuse[0]
        retrievedChunks.append(
            RetrievedChunk(
                chunk=chunks[chunkName],
                rrf_score=fuse[1],
                dense_rank=dense_rank.get(chunkName),
                dense_score=dense_score.get(chunkName),
                sparse_rank=sparse_rank.get(chunkName),
                sparse_score=sparse_score.get(chunkName),
            )
        )

    # pprint(retrievedChunks[:2])

    return retrievedChunks
    # raise NotImplementedError("Phase 3: hybrid_retrieve")


# =============================================================================
# GIVEN: CLI — the comparison view
# =============================================================================


def _fmt_rank(r: int | None) -> str:
    return f"{r:>3}" if r is not None else "  -"


def compare(cfg: Config, query: str, k: int = 5) -> None:
    """Print dense / sparse / hybrid side by side and flag rescues.

    THIS OUTPUT IS THE POINT OF PHASE 3. Read it carefully on a few queries.
    """
    n = k * cfg.retrieval.overfetch
    chunks = {c.chunk_id: c for c in load_chunks(cfg.path("chunks"))}

    dense = dense_search(cfg, query, n)
    sparse = sparse_search(cfg, query, n)
    hybrid = hybrid_retrieve(cfg, query, k=k)

    def label(cid: str) -> str:
        c = chunks[cid]
        path = " > ".join(c.heading_path[-2:]) if c.heading_path else "(preamble)"
        return f"{cid:<28} {path[:44]}"

    print(f"\nquery: {query!r}")
    print(f"(each arm fetched {n} candidates; showing top {k})\n")

    print("DENSE  (cosine similarity, semantic)")
    for i, (cid, s) in enumerate(dense[:k], start=1):
        print(f"  {i}. {s:6.3f}  {label(cid)}")

    print("\nSPARSE (BM25, lexical)")
    for i, (cid, s) in enumerate(sparse[:k], start=1):
        print(f"  {i}. {s:6.3f}  {label(cid)}")

    print(f"\nHYBRID (RRF, k={cfg.retrieval.rrf_k})")
    print(f"     {'rrf':>7}  {'dense':>5} {'sparse':>6}  chunk")
    rescues: list[str] = []
    for i, r in enumerate(hybrid, start=1):
        marks = []
        # A "rescue" = a chunk the other arm would have missed at this cutoff.
        if (r.sparse_rank is not None and r.sparse_rank <= 3) and (
            r.dense_rank is None or r.dense_rank > k
        ):
            marks.append("<- BM25 RESCUE")
            rescues.append(r.chunk_id)
        if (r.dense_rank is not None and r.dense_rank <= 3) and (
            r.sparse_rank is None or r.sparse_rank > k
        ):
            marks.append("<- dense rescue")
        print(
            f"  {i}. {r.rrf_score:7.5f}  {_fmt_rank(r.dense_rank)} "
            f"{_fmt_rank(r.sparse_rank):>6}  {label(r.chunk_id)} {' '.join(marks)}"
        )

    if rescues:
        print(f"\n  *** BM25 RESCUE on this query: {rescues}")
        print(
            "  A chunk BM25 ranked in its top 3 was outside dense's top "
            f"{k}. This is your README evidence that hybrid beats dense-only."
        )
    else:
        print(
            "\n  (no BM25 rescue on this query — try an exact identifier like "
            "'E204', 'IP65', or '36-month')"
        )


def main() -> int:
    cfg = load_config()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        print('usage: python -m src.retrieve "your query here"')
        return 1

    try:
        compare(cfg, args[0], k=cfg.retrieval.top_k if "--full" in sys.argv else 5)
    except FileNotFoundError as exc:
        print(f"Missing prerequisite: {exc}")
        return 1
    except NotImplementedError as exc:
        print(f"Not implemented yet: {exc}")
        print("\nPhase 3 stubs, in order:")
        print("  1. fuse_rrf()          src/retrieve.py")
        print("  2. hybrid_retrieve()   src/retrieve.py")
        print("\nfuse_rrf is pure logic with no dependencies — you can test it")
        print("in a REPL before wiring anything up.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
