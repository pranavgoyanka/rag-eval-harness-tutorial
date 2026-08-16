"""
=============================================================================
PHASE 5 (part 2) — Threshold calibration
=============================================================================

WHAT THIS IS FOR
    src/self_correct.py has two magic numbers:

        verification.sim_threshold      cosine cutoff
        verification.jaccard_threshold  token-overlap cutoff

    Every faithfulness number you report in Phase 6 is downstream of those two
    values. Picking them by intuition means your headline metric is a function
    of a guess — and it will be the first thing anyone competent asks about.

    So: MEASURE them.

HOW YOU GET LABELLED DATA WITHOUT LABELLING ANYTHING
    You need two populations of (cosine, jaccard) scores: sentences that SHOULD
    verify, and sentences that SHOULD NOT.

      POSITIVES — the `expected_answer` fields in eval/golden_set.json, checked
        against the chunks retrieved for their OWN question. These are correct
        answers written from the corpus, so by construction they are grounded.

      NEGATIVES — the same golden answers, checked against the chunks retrieved
        for a DIFFERENT, unrelated question. The sentence is well-formed, fluent
        and on-topic for the corpus; it simply is not supported by THAT context.
        That is exactly the failure mode you are trying to catch, and it costs
        you zero hand-labelling.

    This mismatched-pairing trick is worth remembering. It generates hard
    negatives — plausible corpus-flavoured text — rather than the easy negatives
    you'd get from random strings.

    Its limitation, which you should state when you report the thresholds: real
    hallucinations are usually MORE similar to the context than a mismatched
    answer is (the model is riffing on what it just read). So thresholds tuned
    here are, if anything, optimistic. Phase 6's adversarial check is the
    counterweight.

RUN IT
    uv run python scripts/calibrate.py            # 12 questions
    uv run python scripts/calibrate.py --n 30     # more data, slower

REQUIRES: Phases 1-3 built, embeddings server up, Phase 5 stubs implemented.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_config  # noqa: E402
from src.embed import Embedder  # noqa: E402
from src.index_sparse import tokenize  # noqa: E402
from src.retrieve import hybrid_retrieve  # noqa: E402
from src.self_correct import jaccard, split_sentences  # noqa: E402

# =============================================================================
# TODO(Phase 5) #5 — suggest_thresholds
# =============================================================================


def suggest_thresholds(
    positives: list[float], negatives: list[float]
) -> tuple[float, float, float]:
    """Pick a cutoff separating two score populations.

    ---------------------------------------------------------------------------
    THE CONCEPT: this is a binary classifier, so treat it like one
    ---------------------------------------------------------------------------
    A threshold turns a continuous score into a yes/no decision, and every
    choice trades two errors against each other:

        FALSE NEGATIVE — a grounded sentence scores below the threshold, gets
          flagged unsupported, and the loop rewrites a correct answer. Costs you
          latency and can degrade a good answer.

        FALSE POSITIVE — a hallucinated sentence scores above the threshold and
          sails through. Costs you the entire point of the exercise.

    They are not equally bad here. The whole reason this project exists is to
    catch ungrounded claims, so bias toward catching them — accept some false
    negatives to reduce false positives. That is a judgement about what the
    system is FOR, not a mathematical fact, and it belongs in your README.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Sweep candidate thresholds across the observed range. A simple sweep
         over ~200 evenly spaced values between min and max of all scores is
         plenty; there is no need to be clever.

      2. For each candidate t, count:
             tp = positives >= t        fn = positives <  t
             fp = negatives >= t        tn = negatives <  t

      3. Score each candidate. Options, in increasing order of appropriateness:
           * ACCURACY (tp+tn)/total — do NOT use it. Your two populations are
             different sizes, and accuracy quietly rewards just predicting the
             majority class.
           * YOUDEN'S J = sensitivity + specificity - 1
                        = tp/(tp+fn) + tn/(tn+fp) - 1
             Balanced, size-independent, one line. Good default.
           * F-beta with beta < 1, which weights precision over recall — the
             formal way to encode "false positives are worse here".

         Start with Youden's J. If the resulting threshold lets too many
         negatives through for your taste, switch to F-beta and say why.

      4. Return (best_threshold, sensitivity_at_best, specificity_at_best) so
         the caller can print what the choice actually buys.

    ---------------------------------------------------------------------------
    WHAT TO DO IF THE DISTRIBUTIONS OVERLAP HEAVILY
    ---------------------------------------------------------------------------
    They might. If the best Youden's J is near 0, that means this score does not
    separate grounded from ungrounded text on your corpus — and NO threshold
    will fix that. The honest responses are to lean on the other signal, or to
    report that the check is weak, NOT to pick the least-bad number and present
    it as calibrated. A metric built on a non-separating score is worse than no
    metric, because it looks like evidence.
    """
    # TODO(Phase 5): implement per the recipe above.
    raise NotImplementedError("Phase 5: suggest_thresholds")


# =============================================================================
# GIVEN: data collection and reporting
# =============================================================================


def percentiles(xs: list[float]) -> str:
    if not xs:
        return "(no data)"
    s = sorted(xs)

    def p(q: float) -> float:
        return s[min(len(s) - 1, int(q * len(s)))]

    return (f"min={s[0]:.3f}  p10={p(.10):.3f}  p25={p(.25):.3f}  "
            f"p50={p(.50):.3f}  p75={p(.75):.3f}  p90={p(.90):.3f}  "
            f"max={s[-1]:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="questions to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    golden = json.loads((ROOT / cfg.eval.golden_set).read_text())
    answerable = [q for q in golden["questions"] if not q["should_reject"]]

    rng = random.Random(args.seed)
    sample = rng.sample(answerable, min(args.n, len(answerable)))

    print(f"calibrating on {len(sample)} answerable questions "
          f"(seed={args.seed})\n")

    # Retrieve once per question, and remember the context for each.
    contexts: list[tuple[dict, list]] = []
    for i, q in enumerate(sample, start=1):
        print(f"  retrieving {i}/{len(sample)}: {q['id']}", end="\r")
        contexts.append((q, hybrid_retrieve(cfg, q["question"])))
    print(" " * 60, end="\r")

    emb = Embedder(cfg)
    pos_sim: list[float] = []
    pos_jac: list[float] = []
    neg_sim: list[float] = []
    neg_jac: list[float] = []

    try:
        for idx, (q, sources) in enumerate(contexts):
            sents = split_sentences(q["expected_answer"])
            if not sents:
                continue

            # The MISMATCHED context: a different question's retrieved chunks.
            mis_q, mis_sources = contexts[(idx + len(contexts) // 2) % len(contexts)]
            if mis_q["id"] == q["id"]:
                continue

            sent_vecs = emb.embed(sents, is_query=True)
            for label, srcs, sims_out, jacs_out in (
                ("pos", sources, pos_sim, pos_jac),
                ("neg", mis_sources, neg_sim, neg_jac),
            ):
                chunk_vecs = emb.embed([s.chunk.body for s in srcs], is_query=False)
                token_sets = [set(tokenize(s.chunk.body)) for s in srcs]
                for j, sent in enumerate(sents):
                    sims = chunk_vecs @ sent_vecs[j]
                    st = set(tokenize(sent))
                    jacs = [jaccard(st, ts) for ts in token_sets]
                    sims_out.append(float(max(sims)))
                    jacs_out.append(max(jacs))
    finally:
        emb.close()

    print("=" * 72)
    print(f"COSINE   positives (n={len(pos_sim)})")
    print(f"         {percentiles(pos_sim)}")
    print(f"         negatives (n={len(neg_sim)})")
    print(f"         {percentiles(neg_sim)}")
    print()
    print(f"JACCARD  positives (n={len(pos_jac)})")
    print(f"         {percentiles(pos_jac)}")
    print(f"         negatives (n={len(neg_jac)})")
    print(f"         {percentiles(neg_jac)}")
    print("=" * 72)

    sim_t, sim_sens, sim_spec = suggest_thresholds(pos_sim, neg_sim)
    jac_t, jac_sens, jac_spec = suggest_thresholds(pos_jac, neg_jac)

    print("\nSUGGESTED THRESHOLDS")
    print(f"  sim_threshold:     {sim_t:.3f}   "
          f"(catches {sim_sens:.0%} of grounded, rejects {sim_spec:.0%} of not)")
    print(f"  jaccard_threshold: {jac_t:.3f}   "
          f"(catches {jac_sens:.0%} of grounded, rejects {jac_spec:.0%} of not)")
    print("\nPut these in config.yaml under `verification:`.")
    print("\nREAD THE DISTRIBUTIONS ABOVE BEFORE YOU TRUST THESE NUMBERS.")
    print("If the positive and negative ranges overlap heavily, no threshold")
    print("separates them and the suggestion is meaningless — say so in your")
    print("README instead of shipping a number that looks calibrated.")
    print("\nRemember self_correct.py uses OR: a sentence passes if EITHER check")
    print("passes, so the combined false-positive rate is worse than either")
    print("column above. Measure the pair, not just the parts.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"\nMissing prerequisite: {exc}")
        raise SystemExit(1) from None
    except NotImplementedError as exc:
        print(f"\nNot implemented yet: {exc}")
        print("\nCalibration needs Phases 1-3 built, plus split_sentences() and")
        print("jaccard() in src/self_correct.py and suggest_thresholds() here.")
        raise SystemExit(1) from None
