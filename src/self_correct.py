"""
=============================================================================
PHASE 5 — Self-correction / verification loop
=============================================================================

WHAT YOU'RE BUILDING
    A draft answer -> a verified answer + a faithfulness score.

THE PROBLEM
    Phase 4 asked the model nicely to only use the sources. Models do not
    reliably obey. They interpolate, they round, they merge two chunks into a
    claim neither one makes, and on a FICTIONAL corpus like this one they
    happily fall back on plausible-sounding invention.

    Prompting is a request. Verification is a check. You need both.

THE APPROACH: check each sentence against the sources, twice, cheaply
    For every sentence in the draft, compute against every retrieved chunk:

        cosine similarity   of their embeddings      -> catches paraphrase
        Jaccard overlap     of their token sets      -> catches copied facts

    A sentence counts as SUPPORTED if EITHER passes its threshold. The OR is
    the design, not laziness:

      * Cosine alone fails on this corpus's bread and butter. "The NB-1 costs
        $18,500." and "The NB-1 costs $19,500." are ~0.99 cosine apart — the
        embedding barely notices the digit that makes one of them false.
      * Jaccard alone fails on any real paraphrase, since it needs literal
        shared tokens.

    Together they cover each other's blind spot. Neither is clever; both are
    fast, free, deterministic and debuggable, which matters more than clever
    when the check runs on every answer you generate.

    BE HONEST ABOUT THE CEILING. This is a SIMILARITY check, not an entailment
    check. It cannot catch a sentence that reuses the source's exact vocabulary
    while inverting its meaning ("the robot is NOT suspended in snow" scores
    high against "suspended in snow"). A proper solution runs a natural
    language inference (NLI) model, or an LLM-as-judge, and costs 100x more per
    check. Write that limitation into your README rather than overselling the
    metric.

WHAT TO IMPLEMENT
    Four stubs: split_sentences, jaccard, verify_sentence, self_correct.

RUN IT
    uv run python -m src.self_correct --self-check     # offline, no server
    uv run python scripts/calibrate.py                 # pick your thresholds
    uv run python -m src.pipeline "..." --verify

DEPENDS ON: Phases 1-4.
=============================================================================
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np

from src.config import Config, load_config
from src.context import NOT_FOUND
from src.embed import Embedder
from src.generate import Generator
from src.index_sparse import tokenize
from src.retrieve import RetrievedChunk

# =============================================================================
# GIVEN: result types
# =============================================================================


@dataclass
class SentenceVerdict:
    sentence: str
    supported: bool
    best_sim: float
    best_jaccard: float
    best_chunk_id: str | None
    # Which check carried it: "cosine", "jaccard", "both", or "" if unsupported.
    # Worth recording — if Jaccard is doing all the work, your embedding model
    # or its prefixes are probably misconfigured.
    passed_by: str = ""


@dataclass
class VerificationResult:
    answer: str
    verdicts: list[SentenceVerdict] = field(default_factory=list)
    iterations: int = 0
    faithfulness: float = 1.0

    @property
    def unsupported(self) -> list[SentenceVerdict]:
        return [v for v in self.verdicts if not v.supported]


# =============================================================================
# TODO(Phase 5) #1 — split_sentences
# =============================================================================


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentence-sized claims.

    ---------------------------------------------------------------------------
    WHY THIS IS NOT text.split(".")
    ---------------------------------------------------------------------------
    Look at what your corpus is made of, and what an answer about it looks like:

        "The NB-1 has a fleet uptime of 99.2% and an MTBF of 12,000 hours."
        "Fast charge (0-80%) takes 45 min; list price is $18,500."
        "Scheduled maintenance is every 500 operating hours or 2,000 km."

    Splitting on "." shatters "99.2%" into two fragments, one of which is the
    orphan "2% and an MTBF of 12,000 hours". You then verify that fragment
    against the sources, it fails, and your self-correction loop starts
    rewriting a perfectly good answer to fix a bug in your tokenizer.

    Sentence splitting is the kind of task that looks trivial and is not. Real
    systems use spaCy or NLTK's Punkt. We do it with a regex because pulling a
    250MB NLP model in to split four sentences is the wrong trade — but you
    should know that is the trade you are making.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Split on sentence-ending punctuation ONLY when it is followed by
         whitespace AND the next non-space character starts a new sentence.
         A workable pattern:

             re.split(r"(?<=[.!?])\\s+(?=[A-Z\\[])", text.strip())

         Read the pieces:
           (?<=[.!?])   look BEHIND: the previous char ended a sentence
           \\s+          consume the space (so it is not kept in either half)
           (?=[A-Z\\[])  look AHEAD: next char is a capital, or "[" for a
                        sentence that opens with a citation marker

         The lookahead is what saves you: in "99.2% and", the "." is followed
         by a digit, so no split. In "12,000 hours. Scheduled", the "." is
         followed by a space and a capital, so it splits. Decimals and
         thousands separators survive for free.

      2. Handle semicolons if you want finer-grained claims. "X is 45 min;
         price is $18,500." contains TWO verifiable facts, and verifying them
         separately gives you a more meaningful faithfulness score. Your call —
         document it.

      3. Strip each piece and drop empties.

      4. Known limitation to leave a comment about: this splits on "e.g. The"
         and on "approx. Two". Neither appears in this corpus. Note it rather
         than building an abbreviation list you cannot finish.

    Returns: list[str], each a trimmed sentence. Empty list for empty input.
    """
    # TODO(Phase 5): implement per the recipe above.
    raise NotImplementedError("Phase 5: split_sentences")


# =============================================================================
# TODO(Phase 5) #2 — jaccard
# =============================================================================


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets.

    ---------------------------------------------------------------------------
    THE FORMULA
    ---------------------------------------------------------------------------

        J(A, B) = |A intersect B| / |A union B|        in [0, 1]

    1.0 means identical sets, 0.0 means disjoint.

    ---------------------------------------------------------------------------
    WHY THIS METRIC, AND WHAT ITS SHAPE COSTS YOU
    ---------------------------------------------------------------------------
    We use it as the "did the model copy a real fact" detector, because it
    reacts to literal tokens in a way cosine does not.

    But notice the denominator: the UNION. A short sentence checked against a
    long chunk is penalized for every token the chunk has and the sentence does
    not. Verifying

        "The payload is 25 kg."                       (~5 tokens)

    against a 60-token spec-table chunk gives at best 5/60 = 0.083, even though
    every single word of the sentence appears in the chunk. So a Jaccard
    threshold of, say, 0.30 would reject a perfectly grounded sentence.

    Two ways out, and you should pick one deliberately:
      (a) Keep symmetric Jaccard and set the threshold LOW (~0.05-0.15). Simple,
          but the right threshold now depends on your chunk sizes.
      (b) Use CONTAINMENT instead: |A intersect B| / |A|, i.e. "what fraction of
          the SENTENCE's tokens appear in the chunk?" That is much closer to the
          question you actually care about, and it is chunk-length independent.

    (b) is the better metric for this job. It is not what "Jaccard" means, so if
    you implement (b), rename the function and say so. Do not quietly redefine a
    standard term — the next reader will assume the textbook definition.

    scripts/calibrate.py will show you the real distribution either way. Choose
    from the data, not from this comment.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Guard the empty case: if not a or not b, return 0.0. (An empty union
         is a ZeroDivisionError; an empty sentence is not "perfectly similar".)
      2. return len(a & b) / len(a | b)
    """
    # TODO(Phase 5): implement per the recipe above.
    raise NotImplementedError("Phase 5: jaccard")


# =============================================================================
# TODO(Phase 5) #3 — verify_sentence
# =============================================================================


def verify_sentence(
    sentence: str,
    sent_vec: np.ndarray,
    chunk_vecs: np.ndarray,
    chunk_token_sets: list[set[str]],
    chunk_ids: list[str],
    sim_t: float,
    jac_t: float,
) -> SentenceVerdict:
    """Check one sentence against every retrieved chunk.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Cosine against every chunk in one shot:
             sims = chunk_vecs @ sent_vec
         Both sides are already L2-normalized by embed.py, so this IS cosine
         (see the derivation in embed.py::l2_normalize).

      2. Jaccard against every chunk:
             sent_tokens = set(tokenize(sentence))
             jacs = [jaccard(sent_tokens, ct) for ct in chunk_token_sets]
         Reuse index_sparse.tokenize (already imported). Using a DIFFERENT
         tokenizer here than the one your BM25 index uses would mean your
         verification and your retrieval disagree about what a word is.

      3. Find the best chunk under EACH measure independently — the chunk with
         the highest cosine is not necessarily the one with the highest overlap,
         and either one supporting the sentence is enough.

      4. supported = (best_sim >= sim_t) OR (best_jac >= jac_t)

      5. Set `passed_by` to "cosine", "jaccard", "both", or "".

      6. best_chunk_id: report the chunk from whichever measure actually carried
         the decision (prefer the cosine winner when both pass). This is what
         gets shown to a human debugging a bad verdict, so make it the useful
         one, not just chunk_ids[0].

      7. Return a SentenceVerdict. Cast numpy scalars with float() — they do not
         survive json.dumps in Phase 6.
    """
    # TODO(Phase 5): implement per the recipe above.
    raise NotImplementedError("Phase 5: verify_sentence")


# =============================================================================
# GIVEN: whole-answer verification
# =============================================================================


def verify_answer(
    cfg: Config,
    answer: str,
    sources: list[RetrievedChunk],
    embedder: Embedder,
) -> VerificationResult:
    """Verify every sentence of an answer. Computes faithfulness.

        faithfulness = supported_sentences / total_sentences

    A NOTE ON WHAT THIS NUMBER IS AND ISN'T. Faithfulness measures whether the
    answer is grounded in the retrieved context. It says NOTHING about whether
    the answer is correct or even responsive:

      * An answer can be perfectly faithful to a chunk that does not answer the
        question.
      * A refusal is trivially faithful (see below) — it makes no claims.

    So faithfulness is GAMEABLE ON ITS OWN: a system that refuses everything
    scores 1.0. That is exactly why Phase 6 reports it next to key-fact recall
    and false-refusal rate. Never show this metric alone.
    """
    if answer.strip() == NOT_FOUND:
        # A refusal asserts nothing, so there is nothing to be unfaithful about.
        # Scoring it 1.0 is a convention, not a measurement — and it is precisely
        # the loophole described above. Phase 6 must exclude or flag these.
        return VerificationResult(answer=answer, verdicts=[], iterations=0,
                                  faithfulness=1.0)

    sentences = split_sentences(answer)
    if not sentences or not sources:
        return VerificationResult(answer=answer, verdicts=[], iterations=0,
                                  faithfulness=0.0 if sentences else 1.0)

    # Sentences get the QUERY prefix, chunks get the PASSAGE prefix. A sentence
    # being checked against passages plays the query role, so this mirrors how
    # the vectors were trained. See embed.py on asymmetric models.
    sent_vecs = embedder.embed(sentences, is_query=True)
    chunk_vecs = embedder.embed([s.chunk.body for s in sources], is_query=False)
    chunk_token_sets = [set(tokenize(s.chunk.body)) for s in sources]
    chunk_ids = [s.chunk_id for s in sources]

    verdicts = [
        verify_sentence(
            sent, sent_vecs[i], chunk_vecs, chunk_token_sets, chunk_ids,
            cfg.verification.sim_threshold, cfg.verification.jaccard_threshold,
        )
        for i, sent in enumerate(sentences)
    ]
    supported = sum(1 for v in verdicts if v.supported)
    return VerificationResult(
        answer=answer,
        verdicts=verdicts,
        iterations=0,
        faithfulness=supported / len(verdicts),
    )


# =============================================================================
# GIVEN: the rewrite prompt
# =============================================================================

REWRITE_PROMPT = """Your previous answer contained claims that are not supported \
by the sources.

Previous answer:
{draft}

Unsupported claims:
{unsupported}

Rewrite the answer so that every remaining claim is directly supported by the \
sources below. Delete any claim you cannot support — do not soften it, do not \
hedge it, do not rephrase it. If removing the unsupported claims leaves nothing \
to say, reply with exactly: {not_found}

Sources:
{context}

Question: {question}"""

# Why "delete, don't hedge": a model told to fix an unsupported claim will
# otherwise turn "The CEO is Jane Smith" into "The CEO may be Jane Smith" —
# which passes a human skim, still fails verification, and is arguably worse
# because it launders an invention into an uncertainty. Be explicit.


# =============================================================================
# TODO(Phase 5) #4 — self_correct
# =============================================================================


def self_correct(
    cfg: Config,
    question: str,
    context: str,
    draft: str,
    sources: list[RetrievedChunk],
    generator: Generator,
    embedder: Embedder,
) -> VerificationResult:
    """Verify, and if anything fails, rewrite and re-verify. Bounded.

    ---------------------------------------------------------------------------
    THE LOOP
    ---------------------------------------------------------------------------
      1. result = verify_answer(cfg, draft, sources, embedder)
      2. iteration = 0
      3. while result.unsupported and iteration < cfg.verification.max_iterations:
             a. Build the rewrite prompt. REWRITE_PROMPT is above; fill
                `unsupported` with a bulleted list of the failing sentences —
                naming them specifically is what makes the rewrite targeted
                rather than a reroll.
             b. Call generator.complete([...]) with the same SYSTEM_PROMPT from
                context.py, so the citation and refusal rules still apply.
             c. iteration += 1
             d. result = verify_answer(cfg, new_answer, sources, embedder)
      4. result.iterations = iteration
      5. return result

    ---------------------------------------------------------------------------
    WHY THE ITERATION CAP IS NOT OPTIONAL
    ---------------------------------------------------------------------------
    Without it this loop can run forever, and it will. The failure mode is not
    exotic: a true sentence that your similarity check cannot verify — a
    correct arithmetic result, say, whose digits appear nowhere in any chunk
    (golden question q083 computes 12,400 x 0.88 = 10,912; "10,912" is not in
    the corpus). The model rewrites it, the rewrite is still unverifiable, and
    you have built a machine that burns GPU forever on a correct answer.

    So: cap it, and ACCEPT the answer that comes out of the last iteration even
    if it still has unsupported sentences. Report the faithfulness honestly
    rather than looping until the number looks good. `max_iterations: 2` in
    config.yaml is a starting point.

    ---------------------------------------------------------------------------
    A DESIGN CHOICE TO MAKE EXPLICITLY
    ---------------------------------------------------------------------------
    When the loop ends with sentences still unsupported, you have three options:
      (a) return the answer as-is, with a low faithfulness score  <- suggested
      (b) mechanically DELETE the unsupported sentences from the answer
      (c) discard everything and return NOT_FOUND

    (b) is tempting and usually wrong: deleting a sentence can strand the
    pronoun in the next one, and it hides the failure from your metrics.
    (c) trades hallucination for over-refusal, which Phase 6 will measure as a
    false-refusal rate.

    (a) keeps the system honest and pushes the decision to the caller. Whichever
    you pick, Phase 6 measures the consequence — so pick one, measure it, and
    write down what it cost.
    """
    # TODO(Phase 5): implement per the recipe above.
    raise NotImplementedError("Phase 5: self_correct")


# =============================================================================
# GIVEN: self-check (offline — no server needed)
# =============================================================================


def self_check(cfg: Config) -> None:
    # --- 1. sentence splitting survives decimals ---------------------------
    text = ("The NB-1 has a fleet uptime of 99.2% and an MTBF of 12,000 hours. "
            "Fast charge takes 45 min. The list price is $18,500.")
    sents = split_sentences(text)
    assert len(sents) == 3, f"expected 3 sentences, got {len(sents)}: {sents}"
    assert "99.2%" in sents[0], f"decimal was shattered: {sents[0]!r}"
    assert "$18,500" in sents[2], f"lost the price: {sents[2]!r}"
    print(f"  [1/4] ok  split into {len(sents)} sentences, decimals intact")

    assert split_sentences("") == [], "empty input should give an empty list"
    assert len(split_sentences("One sentence with no trailing period")) == 1
    print("  [2/4] ok  empty and single-sentence inputs handled")

    # --- 2. jaccard ---------------------------------------------------------
    assert jaccard(set("abc"), set("abc")) == 1.0
    assert jaccard(set("abc"), set("xyz")) == 0.0
    assert abs(jaccard({"a", "b"}, {"b", "c"}) - 1 / 3) < 1e-9, "|A&B|/|A|B|"
    assert jaccard(set(), {"a"}) == 0.0, "empty set must not divide by zero"
    print("  [3/4] ok  jaccard: identical=1.0, disjoint=0.0, {a,b}/{b,c}=1/3")

    # --- 3. verify_sentence's OR logic (synthetic vectors, no server) -------
    # Chunk 0 is a paraphrase (high cosine, low overlap).
    # Chunk 1 shares the literal number (low cosine, high overlap).
    sent = "The list price is $18,500 per unit."
    sent_vec = np.array([1.0, 0.0], dtype=np.float32)
    chunk_vecs = np.array([[0.99, 0.14], [0.0, 1.0]], dtype=np.float32)
    chunk_vecs /= np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
    token_sets = [set(tokenize("the cost of one unit")),
                  set(tokenize("List price | $18,500 per unit"))]

    v = verify_sentence(sent, sent_vec, chunk_vecs, token_sets,
                        ["c0", "c1"], sim_t=0.9, jac_t=0.99)
    assert v.supported and v.passed_by == "cosine", (
        f"cosine alone should carry this: {v}"
    )
    v = verify_sentence(sent, sent_vec, chunk_vecs, token_sets,
                        ["c0", "c1"], sim_t=0.999, jac_t=0.2)
    assert v.supported and v.passed_by == "jaccard", (
        f"jaccard alone should carry this: {v}"
    )
    v = verify_sentence(sent, sent_vec, chunk_vecs, token_sets,
                        ["c0", "c1"], sim_t=0.999, jac_t=0.99)
    assert not v.supported, f"neither check passes, should be unsupported: {v}"
    print("  [4/4] ok  OR logic: cosine-only, jaccard-only, and neither")

    print("\nPHASE 5 SELF-CHECK PASSED")
    print("\nNext: uv run python scripts/calibrate.py — do NOT hand-pick your")
    print("thresholds. The self-check above used synthetic values.")


if __name__ == "__main__":
    cfg = load_config()
    try:
        if "--self-check" in sys.argv:
            self_check(cfg)
        else:
            print(__doc__)
            print("usage: python -m src.self_correct --self-check")
    except NotImplementedError as exc:
        print(f"Not implemented yet: {exc}")
        print("\nPhase 5 stubs, in order:")
        print("  1. split_sentences()  src/self_correct.py")
        print("  2. jaccard()          src/self_correct.py")
        print("  3. verify_sentence()  src/self_correct.py")
        print("  4. self_correct()     src/self_correct.py")
        raise SystemExit(1) from None
