"""
=============================================================================
PHASE 4 (part 3 of 3) — End-to-end pipeline
=============================================================================

WHAT YOU'RE BUILDING
    question -> retrieve -> build context -> generate -> cited answer

    This is the file where the project becomes a RAG system rather than a set
    of components. After this, Phase 5 makes it more careful and Phase 6 tells
    you whether any of it actually worked.

WHAT TO IMPLEMENT
    One stub: should_abstain(). It is short, and it is the most interesting
    design decision in Phase 4 — see its docstring.

RUN IT
    uv run python -m src.pipeline "What is the NB-1 payload capacity?"
    uv run python -m src.pipeline "Who is the CEO of Nimbus Robotics?"
    uv run python -m src.pipeline --verbose "How do I file an incident report?"

    # Phase 5: run the verification loop on this one answer
    uv run python -m src.pipeline --verify --verbose "What does the NB-2 cost?"

    # Phase 6: the ablation modes, one query at a time
    uv run python -m src.pipeline --mode=dense "What is error code E204?"
    uv run python -m src.pipeline --mode=none  "What is error code E204?"

    The second one is question q091 from eval/golden_set.json, a labelled
    NEGATIVE: the corpus genuinely does not name a CEO. A correct system says
    "Not found in sources." An incorrect one invents a plausible name, which is
    exactly the failure this project exists to measure.

DEPENDS ON: Phases 1-3 and the rest of Phase 4.
=============================================================================
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from src.config import Config, load_config
from src.context import NOT_FOUND, build_context, build_messages
from src.embed import Embedder
from src.generate import Generator, parse_citations
from src.index_dense import search as dense_search
from src.ingest import load_chunks
from src.retrieve import RetrievedChunk, hybrid_retrieve
from src.self_correct import self_correct

# =============================================================================
# GIVEN: result type
# =============================================================================


@dataclass
class QueryResult:
    """Everything one query produced. Phase 6 serializes this to metrics.json.

    Note the two fields at the bottom. They are Phase 5's outputs, declared now
    and left as None until src/self_correct.py exists. Declaring them early
    means Phase 5 does not change this dataclass's shape, so nothing that
    consumes a QueryResult (the CLI, the API in Phase 7, the eval harness in
    Phase 6) has to be rewritten when verification lands.
    """

    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    sources: list[RetrievedChunk] = field(default_factory=list, repr=False)
    abstained: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # --- Phase 5 fields: populated by self_correct.py, None until then -----
    iterations: int | None = None
    faithfulness: float | None = None

    @property
    def cited_docs(self) -> list[str]:
        """Source documents actually cited — the unit Phase 6 scores against,
        since golden_set.json labels `expected_sources` at document level."""
        by_id = {s.chunk_id: s.chunk.source for s in self.sources}
        return list(dict.fromkeys(by_id[c] for c in self.citations if c in by_id))


# =============================================================================
# TODO(Phase 4) #3 — should_abstain
# =============================================================================


def should_abstain(retrieved: list[RetrievedChunk], min_score: float) -> bool:
    """Decide whether to refuse BEFORE spending a generation call.

    ---------------------------------------------------------------------------
    THE DESIGN QUESTION THIS FUNCTION EXISTS TO POSE
    ---------------------------------------------------------------------------
    You have two independent places to handle an unanswerable question, and they
    fail in opposite directions.

    (a) RETRIEVAL-CONFIDENCE GATING — this function. If the best retrieved chunk
        scores below a floor, nothing relevant was found, so refuse immediately.
          + Cheap: no LLM call at all, so refusals are ~50x faster than answers.
          + Deterministic: same input, same decision, always.
          - The threshold is corpus-, embedder-, and query-dependent. A value
            tuned on this corpus means nothing on another.
          - It is blind to the actual question. Retrieval always returns SOME
            top-k; "Who is the CEO?" still pulls company-overview.md with a
            respectable score, because that chunk is about the company. High
            retrieval confidence does not imply the answer is present.

    (b) PROMPTED ABSTENTION — rule 3 of the system prompt in context.py. The
        model sees the sources and judges for itself.
          + Question-aware: it can tell that company-overview.md discusses the
            company without naming a CEO. This is the case (a) cannot see.
          - Costs a full generation.
          - Unreliable in exactly the way you're trying to fix. A model asked
            to refuse will sometimes confabulate anyway, and will sometimes
            over-refuse a question it could have answered.

    They are complementary, not redundant. (a) catches "nothing relevant exists
    in the corpus at all"; (b) catches "relevant material exists but doesn't
    contain this specific fact". A serious system runs both. Phase 5 adds a
    third, post-hoc layer: verify the generated claims against the sources.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. If `retrieved` is empty -> return True. (Nothing to answer from.)
      2. Otherwise compare retrieved[0].rrf_score against min_score and return
         True when it is strictly below.

    That's the code. The work is in choosing min_score, and there is a trap:

      RRF SCORES ARE NOT CONFIDENCE. An RRF score is built from ranks alone —
      1/(60+1) + 1/(60+1) = 0.0328 is the score of a chunk ranked first by both
      arms, and it gets that score whether the query was a perfect match or
      total nonsense. Every query produces a rank-1 chunk. So the RRF score of
      the top hit is very nearly CONSTANT across queries, which makes it a
      terrible thing to threshold.

      Confirm this yourself before tuning: run several unanswerable questions
      through `python -m src.pipeline --verbose` and compare the top RRF score
      to a few answerable ones. If the two distributions overlap almost
      completely — and they will — then:

        * config.yaml ships min_score: 0.0, i.e. this gate is OFF by default,
          and prompted abstention (b) does the work. That is the honest default.
        * If you want a real retrieval gate, threshold the raw COSINE score
          instead (retrieved[0].dense_score), which does vary with query
          relevance. You would need to extend this function's signature.
        * Either way, Phase 6 measures the consequence: rejection rate on the 10
          `negative` questions, traded against false refusals on the other 90.
          Do not tune this by vibes when you have a labelled test set.

    Write down which you chose and what it cost you. "We measured, the gate
    didn't help, we left it off" is a stronger answer in a design review than a
    threshold nobody can justify.
    """
    # TODO(Phase 4): implement per the recipe above.
    if len(retrieved) == 0:
        return True

    if retrieved[0].rrf_score < min_score:
        return True
    # raise NotImplementedError("Phase 4: should_abstain")


# =============================================================================
# GIVEN: retrieval modes
#
# Phase 4 only ever needs "hybrid". The other two exist so that Phase 6 can run
# an ABLATION — the same pipeline with one component removed — which is the only
# way to attribute a result to a component rather than just describe a system.
# =============================================================================

RETRIEVAL_MODES = ("hybrid", "dense", "none")


def _retrieve(cfg: Config, question: str, mode: str) -> list[RetrievedChunk]:
    if mode == "hybrid":
        return hybrid_retrieve(cfg, question)

    chunks = {c.chunk_id: c for c in load_chunks(cfg.path("chunks"))}

    if mode == "dense":
        # Pipeline A in the ablation: no BM25, no fusion. The rrf_score field is
        # reused to carry the cosine score so downstream code needs no changes.
        hits = dense_search(cfg, question, cfg.retrieval.top_k)
        return [
            RetrievedChunk(chunk=chunks[cid], rrf_score=s, dense_rank=i,
                           dense_score=s)
            for i, (cid, s) in enumerate(hits, start=1)
        ]

    if mode == "none":
        # Pipeline 0: NO RETRIEVAL AT ALL — hand the model the entire corpus.
        #
        # This is only possible because the corpus is ~2,900 tokens and fits in
        # the context window. It is the honest upper bound: if your retrieval
        # pipeline cannot beat "paste everything", then retrieval is subtracting
        # value, not adding it. Most RAG write-ups omit this baseline, and the
        # reason they omit it is not flattering.
        #
        # Note it will blow past generation.context_token_budget. That is the
        # point of the baseline; build_context will trim to the budget, so raise
        # the budget in config.yaml if you want a true no-retrieval run.
        return [
            RetrievedChunk(chunk=c, rrf_score=1.0)
            for c in sorted(chunks.values(), key=lambda c: c.chunk_id)
        ]

    raise ValueError(f"unknown retrieval mode {mode!r}, expected one of {RETRIEVAL_MODES}")


# =============================================================================
# GIVEN: the pipeline
# =============================================================================


def query(
    cfg: Config,
    question: str,
    *,
    generator: Generator | None = None,
    embedder: Embedder | None = None,
    verbose: bool = False,
    mode: str = "hybrid",
    verify: bool | None = None,
) -> QueryResult:
    """Run one question end to end.

    `generator` and `embedder` are injectable so Phase 6 can reuse warm clients
    across 100 questions instead of reconnecting (and re-paying warmup) each
    time. `mode` selects the retrieval ablation; `verify` overrides
    cfg.verification.enabled for a single call.
    """
    timings: dict[str, float] = {}
    owns_generator = generator is None
    gen = generator or Generator(cfg)
    do_verify = cfg.verification.enabled if verify is None else verify

    try:
        # --- retrieve ------------------------------------------------------
        t0 = time.perf_counter()
        retrieved = _retrieve(cfg, question, mode)
        timings["retrieve"] = time.perf_counter() - t0

        if verbose:
            print(f"\nretrieved {len(retrieved)} chunks:")
            for i, r in enumerate(retrieved, start=1):
                print(f"  {i}. rrf={r.rrf_score:.5f} dense={r.dense_rank} "
                      f"sparse={r.sparse_rank}  {r.chunk_id}")

        # --- abstention gate -----------------------------------------------
        if should_abstain(retrieved, cfg.retrieval.min_score):
            timings["total"] = sum(timings.values())
            return QueryResult(
                question=question,
                answer=NOT_FOUND,
                sources=retrieved,
                abstained=True,
                timings=timings,
            )

        # --- build context -------------------------------------------------
        t0 = time.perf_counter()
        context, used = build_context(retrieved, cfg.generation.context_token_budget)
        timings["context"] = time.perf_counter() - t0

        if verbose:
            print(f"\ncontext: {len(used)}/{len(retrieved)} chunks fit the "
                  f"{cfg.generation.context_token_budget}-token budget")
            print("-" * 70)
            print(context)
            print("-" * 70)

        # --- generate ------------------------------------------------------
        t0 = time.perf_counter()
        out = gen.complete(build_messages(question, context))
        timings["generate"] = time.perf_counter() - t0
        answer = out.text

        # --- verify (Phase 5) ----------------------------------------------
        iterations: int | None = None
        faithfulness: float | None = None
        if do_verify:
            t0 = time.perf_counter()
            emb = embedder or Embedder(cfg)
            try:
                vr = self_correct(cfg, question, context, answer, used, gen, emb)
            finally:
                if embedder is None:
                    emb.close()
            timings["verify"] = time.perf_counter() - t0
            answer, iterations, faithfulness = vr.answer, vr.iterations, vr.faithfulness
            if verbose and vr.unsupported:
                print(f"\n{len(vr.unsupported)} sentence(s) still unsupported "
                      f"after {vr.iterations} rewrite(s):")
                for v in vr.unsupported:
                    print(f"  sim={v.best_sim:.3f} jac={v.best_jaccard:.3f}  "
                          f"{v.sentence[:80]}")

        # --- parse citations -----------------------------------------------
        # NB: map against `used`, not `retrieved`. [S1] refers to the first
        # chunk that made it into the context, which is not necessarily the
        # first chunk retrieved. Parse AFTER verification, since a rewrite
        # renumbers nothing but can drop citations along with their sentences.
        cited, invalid = parse_citations(answer, used)

        timings["total"] = sum(timings.values())
        return QueryResult(
            question=question,
            answer=answer,
            citations=cited,
            invalid_citations=invalid,
            sources=used,
            abstained=answer.strip() == NOT_FOUND,
            timings=timings,
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            iterations=iterations,
            faithfulness=faithfulness,
        )
    finally:
        if owns_generator:
            gen.close()


# =============================================================================
# GIVEN: CLI
# =============================================================================


def _print_result(res: QueryResult) -> None:
    print("\n" + "=" * 70)
    print(f"Q: {res.question}")
    print("=" * 70)
    print(res.answer)
    print()

    if res.abstained:
        print("ABSTAINED — the system declined to answer.")
    if res.citations:
        by_id = {s.chunk_id: s for s in res.sources}
        print("citations:")
        for cid in res.citations:
            s = by_id[cid]
            path = " > ".join(s.chunk.heading_path)
            print(f"  {cid:<28} {s.chunk.source}  ({path})")
    elif not res.abstained:
        # Worth flagging: an answer with no citations is unverifiable, which
        # means Phase 5 cannot check it and Phase 6 cannot score its sources.
        print("WARNING: answered with NO citations. The prompt asked for them.")

    if res.invalid_citations:
        print(f"HALLUCINATION SIGNAL: cited non-existent sources "
              f"{res.invalid_citations} (only {len(res.sources)} were provided)")

    print("\nlatency: " + "  ".join(
        f"{k}={v:.2f}s" for k, v in res.timings.items()
    ))
    if res.prompt_tokens:
        print(f"tokens:  {res.prompt_tokens} in / {res.completion_tokens} out")

    if res.faithfulness is None:
        print("faithfulness: not measured — pass --verify (or set "
              "verification.enabled in config.yaml) to run the Phase 5 loop")
    else:
        print(f"faithfulness: {res.faithfulness:.2f} "
              f"({res.iterations} verification iteration(s))")


def main() -> int:
    cfg = load_config()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        print('usage: python -m src.pipeline "your question"'
              ' [--verbose] [--verify] [--mode=hybrid|dense|none]')
        return 1

    try:
        gen = Generator(cfg)
        try:
            # Warm up first, or the timing breakdown below measures model load.
            w = gen.warmup()
            if w > 0.5:
                print(f"(warmed up the model in {w:.1f}s — excluded from timings)")
            mode = "hybrid"
            for m in RETRIEVAL_MODES:
                if f"--mode={m}" in sys.argv:
                    mode = m
            res = query(
                cfg,
                args[0],
                generator=gen,
                verbose="--verbose" in sys.argv,
                mode=mode,
                verify=True if "--verify" in sys.argv else None,
            )
        finally:
            gen.close()
        _print_result(res)
    except FileNotFoundError as exc:
        print(f"Missing prerequisite: {exc}")
        return 1
    except NotImplementedError as exc:
        print(f"Not implemented yet: {exc}")
        print("\nRemaining stubs for a working end-to-end pipeline:")
        print("  Phase 1  split_by_heading, merge_small_sections, make_chunk_id")
        print("  Phase 2  l2_normalize, numpy_search, build_faiss, faiss_search,")
        print("           tokenize, search_bm25")
        print("  Phase 3  fuse_rrf, hybrid_retrieve")
        print("  Phase 4  build_context, parse_citations, should_abstain")
        print("\nSee PHASES.md for the order and the concepts behind each one.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
