"""
=============================================================================
PHASE 6 — Evaluation & ablation
=============================================================================

WHAT YOU'RE BUILDING
    eval/golden_set.json -> metrics.json + a table you can put in a README.

WHY THIS IS THE PHASE THAT MATTERS
    Everything before this was construction. "I built a hybrid RAG pipeline with
    self-correction" is a DESCRIPTION. It is not a result, and nobody senior
    will read it as one.

    A result looks like: "hybrid retrieval raised context recall@8 from 0.71 to
    0.89, driven entirely by the 24 identifier questions; verification raised
    faithfulness from 0.82 to 0.94 at the cost of 3 extra false refusals and
    1.9s of p50 latency."

    The difference between those two sentences is this file.

THE ABLATION
    An ablation removes ONE component and re-measures. Without it you cannot
    attribute an outcome to a cause — you only know the whole thing works.

      Pipeline 0  no retrieval, entire corpus in the prompt
      Pipeline A  dense retrieval only
      Pipeline B  hybrid (dense + BM25 + RRF)
      Pipeline C  hybrid + Phase 5 verification

    Pipeline 0 is the one people leave out. This corpus is ~2,900 tokens and
    FITS IN THE CONTEXT WINDOW, so "paste everything" is a legitimate strategy
    and an honest upper bound on answer quality. If B does not beat 0 on
    accuracy, your retrieval is subtracting value and the interesting finding is
    that you did not need RAG at this scale. That is a perfectly good result —
    report it.

    Expected (but verify, don't assume): B > A on recall/precision; C >= B on
    faithfulness; 0 strong on accuracy but worst on latency and cost, and
    unable to cite precisely.

USING THE GOLDEN SET AS-IS
    eval/golden_set.json already has 100 labelled questions. Do NOT author a new
    test set.

    Its `expected_sources` are labelled at DOCUMENT level, not chunk level, so
    all retrieval metrics here are computed over source documents. That is not a
    compromise — it is the more robust unit, because re-chunking would
    invalidate chunk-level labels and you WILL re-chunk.

WHAT TO IMPLEMENT
    Four stubs: key_fact_recall, context_precision_recall, mrr, percentile.

RUN IT
    uv run python scripts/run_eval.py --self-check    # metrics only, offline
    uv run python scripts/run_eval.py --n 20          # stratified sample
    uv run python scripts/run_eval.py --all           # all 100 (slow!)
    uv run python scripts/run_eval.py --n 10 --pipelines B,C

DEPENDS ON: Phases 1-5.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, Config, load_config  # noqa: E402
from src.context import NOT_FOUND  # noqa: E402
from src.embed import Embedder  # noqa: E402
from src.generate import Generator  # noqa: E402
from src.pipeline import QueryResult, query  # noqa: E402

# =============================================================================
# GIVEN: the pipelines under test
# =============================================================================

MIN_NEGATIVES = 5

PIPELINES: dict[str, dict] = {
    "0": {"label": "no-RAG (full corpus)", "mode": "none", "verify": False},
    "A": {"label": "dense only", "mode": "dense", "verify": False},
    "B": {"label": "hybrid (RRF)", "mode": "hybrid", "verify": False},
    "C": {"label": "hybrid + verify", "mode": "hybrid", "verify": True},
}


# =============================================================================
# TODO(Phase 6) #1 — key_fact_recall
# =============================================================================


def key_fact_recall(answer: str, key_facts: list[str]) -> float:
    """Fraction of the expected key facts that appear in the answer.

    ---------------------------------------------------------------------------
    WHY SUBSTRING MATCHING, AND WHEN IT LIES TO YOU
    ---------------------------------------------------------------------------
    The golden set's `key_facts` are literal corpus substrings ("25 kg",
    "12,000 hours", "suspended in snow"). Checking whether they appear in the
    answer is crude, free, deterministic and reproducible — which beats an
    LLM-judge for a metric you will compute thousands of times while iterating.

    Know its two failure directions:

      FALSE NEGATIVES — the answer says "25 kilograms" or "twenty-five kg" and
        you score it wrong. Normalization (below) buys back some of this; the
        rest is the price of a cheap metric.

      FALSE POSITIVES — the fact appears, but negated or attributed to the wrong
        product. "The NB-1 does NOT have a 60 kg payload" contains "60 kg" and
        scores as a hit. Substring matching has no idea what a sentence means.

    Report this metric as "key-fact recall", never as "accuracy". The name
    should tell the reader what was actually measured.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. If key_facts is empty, return 1.0. (Negative questions have no key
         facts; vacuously satisfied. Guard the division either way.)

      2. Normalize BOTH sides identically before comparing:
           * casefold()
           * collapse all whitespace runs to a single space —
             `" ".join(s.split())`. This is what "whitespace-insensitive" in
             the golden set's own description means, and it matters because the
             corpus wraps lines mid-fact.
         DO NOT strip punctuation: "12,000" and "99.2%" depend on it, and
         stripping it would make "12000" and "12,000" collide with unrelated
         numbers.

      3. Count how many normalized key_facts appear in the normalized answer,
         divide by len(key_facts).

      4. Consider (and document) whether a partial numeric match should count.
         "1,200" is a substring of "12,000" — no it isn't, but "200" IS a
         substring of "12,000". If a key fact is short and numeric, substring
         matching can fire on a coincidence. Worth a comment; worth checking
         your corpus for cases before deciding it's fine.

    Returns: float in [0, 1].
    """
    # TODO(Phase 6): implement per the recipe above.
    raise NotImplementedError("Phase 6: key_fact_recall")


# =============================================================================
# TODO(Phase 6) #2 — context_precision_recall
# =============================================================================


def context_precision_recall(
    retrieved_sources: list[str], expected_sources: list[str]
) -> tuple[float, float]:
    """Retrieval quality at the document level.

    ---------------------------------------------------------------------------
    THE TWO NUMBERS
    ---------------------------------------------------------------------------

        precision@k = |retrieved AND expected| / |retrieved|
                      "of what I fetched, how much was useful?"

        recall@k    = |retrieved AND expected| / |expected|
                      "of what I needed, how much did I fetch?"

    ---------------------------------------------------------------------------
    RECALL IS THE ONE THAT MATTERS MOST, AND HERE IS WHY
    ---------------------------------------------------------------------------
    Context recall is the CEILING ON YOUR ENTIRE SYSTEM. A fact that was never
    retrieved cannot be generated, cannot be cited, and cannot be verified. No
    amount of prompt engineering, no better model, and no self-correction loop
    recovers it. When end-to-end accuracy is bad, check recall FIRST — most of
    the time the generator is being blamed for the retriever's miss.

    Precision matters too, but it degrades gracefully: extra irrelevant chunks
    cost you tokens, latency, and some attention dilution (see "lost in the
    middle" in context.py), not correctness outright.

    ---------------------------------------------------------------------------
    A TRAP SPECIFIC TO THIS SETUP
    ---------------------------------------------------------------------------
    You retrieve k=8 CHUNKS, but you score against DOCUMENTS, and 8 chunks
    frequently come from only 3-4 distinct documents. So `retrieved_sources`
    must be DEDUPED before you compute precision, or you will divide by 8 when
    the real denominator is 4 and report a precision that is roughly half the
    truth. Dedupe, then measure.

    Also: a question with 3 expected sources cannot exceed precision 3/8 = 0.375
    even with perfect retrieval, because the other 5 slots are necessarily
    "wrong". Precision@k is bounded by |expected|/k. Do not read a low number as
    a failure without checking that ceiling — mention it when you report.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Guard: if expected_sources is empty (negative questions), return
         (0.0, 1.0) or flag it — but do NOT include negatives in the averages.
         The caller already excludes them; make the guard explicit anyway.
      2. Build sets of both, deduped.
      3. hits = retrieved_set & expected_set
      4. precision = len(hits) / len(retrieved_set)   (guard empty)
         recall    = len(hits) / len(expected_set)
      5. Return (precision, recall).

    Returns: (precision, recall), each in [0, 1].
    """
    # TODO(Phase 6): implement per the recipe above.
    raise NotImplementedError("Phase 6: context_precision_recall")


# =============================================================================
# TODO(Phase 6) #3 — mrr
# =============================================================================


def mrr(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Reciprocal rank of the first relevant document.

    ---------------------------------------------------------------------------
    WHY YOU NEED THIS ON TOP OF PRECISION@K
    ---------------------------------------------------------------------------
    Precision@8 cannot tell these two retrievals apart:

        run 1:  [RIGHT, wrong, wrong, wrong, wrong, wrong, wrong, wrong]
        run 2:  [wrong, wrong, wrong, wrong, wrong, wrong, wrong, RIGHT]

    Both score 1/8. But run 1 put the answer where the model will actually
    attend to it, and run 2 buried it at the far end of the context. Precision@k
    is RANK-BLIND; that is its defining weakness.

        RR   = 1 / rank_of_first_relevant        (1-indexed; 0 if none found)
        MRR  = mean of RR over all queries

    Run 1 scores 1.0, run 2 scores 0.125. Now they are distinguishable.

    Note the shape: MRR only looks at the FIRST hit. A retrieval that puts one
    relevant doc at rank 1 and misses two others scores a perfect 1.0. So MRR
    and recall answer genuinely different questions, and reporting one without
    the other is how retrieval evaluations mislead people. Report both.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Dedupe retrieved_sources while PRESERVING ORDER — `dict.fromkeys()`.
         Order is the entire input to this metric; a set() destroys it.
      2. Walk the deduped list with enumerate(..., start=1). 1-INDEXED: rank 0
         would give a division by zero, and rank 1 must score 1.0.
      3. Return 1.0 / rank at the first element present in expected_sources.
      4. Return 0.0 if no relevant document was retrieved at all. (0.0 is the
         convention — it is the limit of 1/rank as rank goes to infinity.)
    """
    # TODO(Phase 6): implement per the recipe above.
    raise NotImplementedError("Phase 6: mrr")


# =============================================================================
# TODO(Phase 6) #4 — percentile
# =============================================================================


def percentile(values: list[float], p: float) -> float:
    """The p-th percentile of `values`, with p in [0, 1].

    ---------------------------------------------------------------------------
    WHY NOT JUST REPORT THE MEAN LATENCY
    ---------------------------------------------------------------------------
    Because the mean hides the tail, and the tail is what users experience. A
    pipeline with p50=1.2s and p95=14s feels broken even though its mean might
    be a respectable 2.1s. One slow request in twenty is one slow request in
    twenty; averaging it away does not make it stop happening.

    Always report p50 AND p95. p50 tells you the typical experience; p95 tells
    you the bad one. (p99 needs far more than 100 samples to mean anything, so
    do not report it here — with n=20 you have no p99, you have "the max".)

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT — and the ambiguity you must resolve
    ---------------------------------------------------------------------------
    "The 95th percentile" is under-specified. With 20 sorted samples, which one
    is p95? Index 18? 19? Something interpolated between them? There are
    several standard answers and they disagree at small n.

    Use the NEAREST-RANK method — the simplest defensible one:

        1. Guard the empty list (return 0.0).
        2. s = sorted(values)
        3. idx = ceil(p * len(s)) - 1, then clamp to [0, len(s)-1]
        4. return s[idx]

    This always returns an ACTUAL OBSERVED VALUE, which is a real advantage for
    latency: "p95 = 8.4s" means some request genuinely took 8.4s, not that a
    weighted average of two requests came to 8.4s.

    numpy.percentile() uses linear interpolation by default and will give you
    slightly different numbers. Neither is wrong. Pick one, and say which in
    your README — an unlabelled percentile is a number nobody can reproduce.
    """
    # TODO(Phase 6): implement per the recipe above.
    raise NotImplementedError("Phase 6: percentile")


# =============================================================================
# GIVEN: scoring one result
# =============================================================================


def score_one(res: QueryResult, q: dict, mode: str) -> dict:
    """Turn one QueryResult + its golden entry into a row of raw numbers."""
    retrieved_docs = list(dict.fromkeys(s.chunk.source for s in res.sources))
    rejected = res.answer.strip() == NOT_FOUND

    row: dict = {
        "id": q["id"],
        "category": q["category"],
        "difficulty": q["difficulty"],
        "should_reject": q["should_reject"],
        "rejected": rejected,
        "latency": res.timings.get("total", 0.0),
        "faithfulness": res.faithfulness,
        "iterations": res.iterations,
        "invalid_citations": len(res.invalid_citations),
        "n_citations": len(res.citations),
        "answer": res.answer,
    }

    if q["should_reject"]:
        # For a negative question the ONLY thing that matters is whether the
        # system refused. Key facts and expected sources are empty by design.
        return row

    row["key_fact_recall"] = key_fact_recall(res.answer, q["key_facts"])

    if mode == "none":
        # Pipeline 0 "retrieves" the entire corpus, so recall is trivially 1.0
        # and precision is just |expected|/|corpus|. Reporting those numbers
        # alongside the real pipelines would be actively misleading, so we
        # record None and the table prints "n/a".
        row["precision"] = None
        row["recall"] = None
        row["mrr"] = None
    else:
        p, r = context_precision_recall(retrieved_docs, q["expected_sources"])
        row["precision"] = p
        row["recall"] = r
        row["mrr"] = mrr(retrieved_docs, q["expected_sources"])
    return row


def aggregate(rows: list[dict]) -> dict:
    """Collapse per-question rows into the summary metrics for one pipeline."""
    pos = [r for r in rows if not r["should_reject"]]
    neg = [r for r in rows if r["should_reject"]]

    def mean(xs: list) -> float | None:
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    lat = [r["latency"] for r in rows]
    return {
        "n": len(rows),
        "n_negative": len(neg),
        "key_fact_recall": mean([r.get("key_fact_recall") for r in pos]),
        "precision_at_k": mean([r.get("precision") for r in pos]),
        "recall_at_k": mean([r.get("recall") for r in pos]),
        "mrr": mean([r.get("mrr") for r in pos]),
        # Two sides of the same coin. Reporting rejection_rate alone is
        # gameable: a system that refuses everything scores 1.0. Always show the
        # false-refusal rate next to it.
        "rejection_rate": (sum(r["rejected"] for r in neg) / len(neg)) if neg else None,
        "false_refusal_rate": (sum(r["rejected"] for r in pos) / len(pos)) if pos else None,
        "faithfulness": mean([r["faithfulness"] for r in rows]),
        "invalid_citation_rate": mean([float(r["invalid_citations"] > 0) for r in rows]),
        "p50_latency": percentile(lat, 0.50),
        "p95_latency": percentile(lat, 0.95),
    }


# =============================================================================
# GIVEN: sampling, running, reporting
# =============================================================================


def stratified_sample(questions: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n questions keeping the category mix, and ALWAYS include negatives.

    A uniform random sample of 20 from 100 would pull ~2 negatives, which is far
    too few to say anything about rejection rate. Stratifying is not statistical
    politeness here; it is what makes the small-sample run interpretable.
    """
    # Floor on negatives. Below ~5 the rejection rate moves in 20-point steps
    # and the number stops meaning anything.
    if n >= len(questions):
        return questions
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_cat[q["category"]].append(q)

    out: list[dict] = []
    for cat, qs in sorted(by_cat.items()):
        take = max(1, round(n * len(qs) / len(questions)))
        if cat == "negative":
            take = min(len(qs), max(take, MIN_NEGATIVES))
        out.extend(rng.sample(qs, min(take, len(qs))))

    # Rounding up per category overshoots n. Trim the non-negatives back down,
    # never the negatives — so the returned size can EXCEED n for very small n
    # (the floor wins). The caller prints the real count rather than pretending
    # it honoured --n.
    if len(out) > n:
        neg = [q for q in out if q["category"] == "negative"]
        rest = [q for q in out if q["category"] != "negative"]
        out = neg + rest[: max(0, n - len(neg))]
    return out


def run_pipeline(cfg: Config, key: str, questions: list[dict],
                 gen: Generator, emb: Embedder) -> tuple[list[dict], list[dict]]:
    spec = PIPELINES[key]
    rows, raw = [], []
    print(f"\n--- pipeline {key}: {spec['label']} "
          f"(mode={spec['mode']}, verify={spec['verify']}) ---")
    for i, q in enumerate(questions, start=1):
        t0 = time.perf_counter()
        res = query(cfg, q["question"], generator=gen, embedder=emb,
                    mode=spec["mode"], verify=spec["verify"])
        rows.append(score_one(res, q, spec["mode"]))
        raw.append({"id": q["id"], "question": q["question"],
                    "answer": res.answer, "citations": res.citations})
        print(f"  {i:3d}/{len(questions)}  {q['id']}  "
              f"{time.perf_counter() - t0:5.1f}s  "
              f"{'REJECT' if rows[-1]['rejected'] else '      '}  "
              f"{q['question'][:52]}")
    return rows, raw


def _fmt(v, pct: bool = True) -> str:
    if v is None:
        return "  n/a"
    return f"{v:5.1%}" if pct else f"{v:5.2f}"


def print_table(summary: dict[str, dict]) -> None:
    keys = list(summary)
    print("\n" + "=" * 78)
    print("ABLATION")
    print("=" * 78)
    rows = [
        ("key-fact recall",   "key_fact_recall",       True),
        ("context prec@k",    "precision_at_k",        True),
        ("context recall@k",  "recall_at_k",           True),
        ("MRR",               "mrr",                   False),
        ("rejection (neg)",   "rejection_rate",        True),
        ("false refusal",     "false_refusal_rate",    True),
        ("faithfulness",      "faithfulness",          True),
        ("bad-citation rate", "invalid_citation_rate", True),
        ("p50 latency (s)",   "p50_latency",           False),
        ("p95 latency (s)",   "p95_latency",           False),
    ]
    print(f"{'metric':<20}" + "".join(f"{k:>13}" for k in keys))
    print(f"{'':<20}" + "".join(
        f"{PIPELINES[k]['label'][:12]:>13}" for k in keys))
    print("-" * 78)
    for label, field, pct in rows:
        print(f"{label:<20}" + "".join(
            f"{_fmt(summary[k][field], pct):>13}" for k in keys))
    print("=" * 78)

    print("\nMarkdown (paste into your README):\n")
    print("| metric | " + " | ".join(
        f"{k} — {PIPELINES[k]['label']}" for k in keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for label, field, pct in rows:
        print(f"| {label} | " + " | ".join(
            _fmt(summary[k][field], pct).strip() for k in keys) + " |")


def self_check() -> None:
    """Offline checks on the four metric stubs. No server, no indexes needed."""
    # --- key_fact_recall ----------------------------------------------------
    ans = "The NB-1 carries 25 kg and has an MTBF of 12,000 hours."
    assert key_fact_recall(ans, ["25 kg", "12,000 hours"]) == 1.0
    assert key_fact_recall(ans, ["25 kg", "60 kg"]) == 0.5
    assert key_fact_recall(ans, []) == 1.0, "no key facts is vacuously satisfied"
    # Whitespace-insensitive, as the golden set's own description requires.
    assert key_fact_recall("payload is 25    kg", ["25 kg"]) == 1.0, (
        "normalize whitespace on BOTH sides before comparing"
    )
    assert key_fact_recall("Payload: 25 KG", ["25 kg"]) == 1.0, "casefold both sides"
    print("  [1/4] ok  key_fact_recall: exact, partial, empty, whitespace, case")

    # --- context_precision_recall ------------------------------------------
    p, r = context_precision_recall(
        ["corpus/a.md", "corpus/b.md"], ["corpus/a.md"])
    assert (p, r) == (0.5, 1.0), f"expected (0.5, 1.0), got ({p}, {r})"
    # Duplicates must not inflate the denominator — see the docstring.
    p, r = context_precision_recall(
        ["corpus/a.md", "corpus/a.md", "corpus/a.md"], ["corpus/a.md"])
    assert (p, r) == (1.0, 1.0), (
        f"expected (1.0, 1.0), got ({p}, {r}) — dedupe retrieved_sources first"
    )
    p, r = context_precision_recall(["corpus/x.md"], ["corpus/a.md"])
    assert (p, r) == (0.0, 0.0)
    print("  [2/4] ok  context_precision_recall: partial, deduped, miss")

    # --- mrr ----------------------------------------------------------------
    assert mrr(["a", "b", "c"], ["a"]) == 1.0, "rank 1 must score 1.0 (1-indexed)"
    assert mrr(["a", "b", "c"], ["c"]) == 1 / 3
    assert mrr(["a", "b"], ["z"]) == 0.0, "no hit scores 0.0"
    assert mrr(["a", "a", "b"], ["b"]) == 0.5, (
        "dedupe must PRESERVE ORDER — b is 2nd distinct doc, so 1/2"
    )
    print("  [3/4] ok  mrr: rank 1, rank 3, no hit, order-preserving dedupe")

    # --- percentile ---------------------------------------------------------
    xs = [float(i) for i in range(1, 21)]  # 1..20
    assert percentile(xs, 0.50) == 10.0, "nearest-rank p50 of 1..20 is 10"
    assert percentile(xs, 0.95) == 19.0, "nearest-rank p95 of 1..20 is 19"
    assert percentile(xs, 1.0) == 20.0
    assert percentile([], 0.5) == 0.0, "empty input must not raise"
    assert percentile([7.0], 0.95) == 7.0
    for v in (percentile(xs, 0.5), percentile(xs, 0.95)):
        assert v in xs, "nearest-rank must return an OBSERVED value"
    print("  [4/4] ok  percentile: nearest-rank, empty, single, observed-only")

    print("\nPHASE 6 SELF-CHECK PASSED")
    print("\nThese cover the metric functions only. The ablation itself is")
    print("validated by running it: uv run python scripts/run_eval.py --n 20")


def main() -> int:
    if "--self-check" in sys.argv:
        self_check()
        return 0

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="stratified sample size")
    ap.add_argument("--all", action="store_true", help="run all 100 questions")
    ap.add_argument("--pipelines", default="0,A,B,C")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    golden = json.loads((ROOT / cfg.eval.golden_set).read_text())
    questions = golden["questions"]
    if not args.all:
        questions = stratified_sample(questions, args.n, args.seed)

    keys = [k.strip() for k in args.pipelines.split(",") if k.strip()]
    for k in keys:
        if k not in PIPELINES:
            print(f"unknown pipeline {k!r}; choose from {list(PIPELINES)}")
            return 1

    total = len(questions) * len(keys)
    if not args.all and len(questions) != args.n:
        print(f"note: sampled {len(questions)}, not {args.n} — the stratifier "
              f"keeps at least {MIN_NEGATIVES} negative questions.")
    print(f"{len(questions)} questions x {len(keys)} pipelines = {total} runs")
    print("Running SERIALLY — the backing llama-server uses --parallel 1, so")
    print("concurrency would only add queueing noise to your latency numbers.")
    print(f"At ~8s/answer this is roughly {total * 8 / 60:.0f} minutes; "
          "pipeline C is slower (extra rewrite calls).")
    print()

    gen = Generator(cfg)
    emb = Embedder(cfg)
    try:
        # Warm up ONCE, outside the timed section. Without this the first
        # question of the first pipeline absorbs the model load and single-
        # handedly ruins your p95.
        w = gen.warmup()
        print(f"warmed up in {w:.1f}s (excluded from all timings)")

        summary, details, raws = {}, {}, {}
        for k in keys:
            rows, raw = run_pipeline(cfg, k, questions, gen, emb)
            summary[k], details[k], raws[k] = aggregate(rows), rows, raw
    finally:
        gen.close()
        emb.close()

    print_table(summary)

    out = ROOT / cfg.eval.output
    out.write_text(json.dumps({
        "config": {
            "chat_model": cfg.models.chat,
            "embedding_model": cfg.models.embedding,
            "top_k": cfg.retrieval.top_k,
            "rrf_k": cfg.retrieval.rrf_k,
            "sim_threshold": cfg.verification.sim_threshold,
            "jaccard_threshold": cfg.verification.jaccard_threshold,
        },
        "n_questions": len(questions),
        "question_ids": [q["id"] for q in questions],
        "summary": summary,
        "per_question": details,
        "answers": raws,
    }, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")

    print("\nBEFORE YOU BELIEVE THIS TABLE:")
    print(f"  * n={len(questions)}"
          f"{' (of 100)' if not args.all else ''}. A 2-3 point difference "
          "between pipelines is noise. Report only gaps that survive that.")
    print("  * api-reference.md backs just 1 of the 100 questions, so anything "
          "you conclude about it is a sample of one.")
    print("  * Check the per-question rows in metrics.json for the questions "
          "where B lost to A. Those are where you learn something.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"\nMissing prerequisite: {exc}")
        raise SystemExit(1) from None
    except NotImplementedError as exc:
        print(f"\nNot implemented yet: {exc}")
        print("\nPhase 6 needs every earlier phase finished. Its own stubs, in")
        print("scripts/run_eval.py:")
        print("  1. key_fact_recall()")
        print("  2. context_precision_recall()")
        print("  3. mrr()")
        print("  4. percentile()")
        raise SystemExit(1) from None
