# PHASES.md — the build guide

This repo is a **scaffold, not a finished system**. All the plumbing is written;
the conceptually interesting functions are stubs that raise
`NotImplementedError`. Each stub's docstring contains the concept, the formula,
the failure modes, and a numbered recipe.

You fill in 14 functions. In doing so you build a hybrid-retrieval RAG pipeline
with a real evaluation harness, and — more to the point — you end up able to
defend every design decision in it.

**Ground rules**

- Phases are strictly ordered. Phase N never imports from Phase N+1.
- No pytest. Every module has a `--self-check` that asserts its own contract.
  Read the self-check before implementing: **it is the spec**.
- Phases 1 and part of 2 need no server. Phases 2–4 need the LLM server.

---

## Status

| Phase | Topic | Code | Stubs |
|---|---|---|---|
| 0 | Environment & preflight | ✅ written | — |
| 1 | Ingestion & chunking | ✅ scaffolded | 3 |
| 2 | Dual indexing (dense + sparse) | ✅ scaffolded | 6 |
| 3 | Hybrid retrieval + RRF | ✅ scaffolded | 2 |
| 4 | Context + generation | ✅ scaffolded | 3 |
| 5 | Self-correction loop | ✅ scaffolded | 5 |
| 6 | Evaluation & ablation | ✅ scaffolded | 4 |
| 7 | API + UI | ✅ written | — |
| 8 | Packaging | ✅ template | — |

**23 stubs total.** Phase 7 has none on purpose (see below), and Phase 8 is a
README with marked blanks for your own results.

---

## Phase 0 — Environment

**One manual prerequisite.** The chat server is running, but it does not serve
embeddings: `/v1/embeddings` returns `501 not_supported_error`. A chat model
cannot serve embeddings at all — you need a second model registered on the
llama-swap router, launched with `--embeddings`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python -m src.config --check
```

`--check` probes all three endpoints and, if embeddings are missing, prints the
exact llama-swap config block to add (`nomic-ai/nomic-embed-text-v1.5-GGUF`,
`--embeddings --pooling mean`). Your router reports `max_instances: 4`, so the
embedding model can stay resident alongside `qwen3.8-27b`.

**Note on `base_url`.** If you are working inside a container, `localhost:8101`
is the *container*, not your host. `config.yaml` defaults to
`http://host.docker.internal:8101/v1`. Override with `export RAG_BASE_URL=...`.

**DoD:** `--check` prints `chat: ok` and `embeddings: ok (dim=768)`.

---

## Phase 1 — Ingestion & chunking

**File:** `src/ingest.py` · **Needs no server.**

**What you learn.** Chunking is the highest-leverage decision in RAG and the
one most often made by accident. A retriever can only return what you indexed;
if a chunk splits a table from its header, no reranker recovers it.

**A sizing note specific to this corpus.** The usual advice is 300–500 token
chunks. This corpus is ~2,900 tokens across 8 documents — that advice would give
you roughly *one chunk per file*, retrieval would be trivially perfect, and your
Phase 6 numbers would be meaningless. We split on Markdown heading sections
instead (~40–250 tokens). Notice that the "standard" answer was wrong here
because of a property of the data, not because the advice is bad.

**Stubs:** `split_by_heading` · `merge_small_sections` · `make_chunk_id`

**Key concept — contextual chunk headers.** The body of the NB-1 spec table is
`| Payload capacity | 25 kg |` with the words "NB-1" and "Courier" nowhere in
it. Embedded as-is it is indistinguishable from the NB-2's nearly identical
table. Prepending the heading breadcrumb fixes this, and it is the single
highest-leverage line of code in the phase.

```bash
uv run python -m src.ingest              # build + histogram
uv run python -m src.ingest --self-check # 4 assertions
```

**DoD:** `artifacts/chunks.jsonl` exists, ~35 chunks (a reference solution
produces exactly 35, ~78 tokens mean, 2,731 tokens total), histogram inside
40–250 tokens, self-check green. Read the histogram — one fat bar on the right
means your splitter isn't splitting.

---

## Phase 2 — Dual indexing

**Files:** `src/embed.py`, `src/index_dense.py`, `src/index_sparse.py`
**Stubs:** `l2_normalize` · `numpy_search` · `build_faiss` · `faiss_search` ·
`tokenize` · `search_bm25`

You build **two** indexes because they fail in opposite directions:

|  | Dense (embeddings) | Sparse (BM25) |
|---|---|---|
| Matches on | meaning | exact terms |
| "how fast is the courier" → "Top speed \| 24 km/h" | ✅ | ✗ (no shared words) |
| "NB-1" vs "NB-2" | ✗ (near-identical vectors) | ✅ (different atoms) |
| Needs a model | yes | no |
| Score range | cosine ∈ [−1, 1] | unbounded |

**The dense side.** You implement search twice: once in ~6 lines of numpy, once
in FAISS, then the self-check asserts they return identical results. The point
is to see that "vector database" at this scale is `matrix @ query` — and to be
able to say *at what n* that stops being true, and what `IndexFlatIP` trades
away versus `HNSW`/`IVF`.

**The sparse side.** `tokenize()` is where the phase is won or lost. The obvious
`\w+` regex turns `NB-1` into `["nb", "1"]` — "nb" appears in every chunk of both
product docs and carries no IDF weight, "1" is a bare digit colliding with every
table row. You have just made NB-1 and NB-2 indistinguishable to the one
retriever whose entire purpose was precision, and **53 of the 100 golden
questions turn on that distinction**. The self-check asserts
`"nb-1" in tokenize("... NB-1 ...")`.

(Watch out: `E204` and `IP65` survive the naive regex, since they contain no
punctuation. Don't let that fool you into thinking it's fine — it's correct by
accident on those and wrong on the case that matters.)

```bash
uv run python -m src.embed        --self-check
uv run python -m src.index_dense  && uv run python -m src.index_dense  --self-check
uv run python -m src.index_sparse && uv run python -m src.index_sparse --self-check
```

**DoD:** both indexes reload with matching counts; numpy and FAISS agree on
top-k; `"E204"` returns the `faq-troubleshooting.md` chunk at rank 1.

---

## Phase 3 — Hybrid retrieval + RRF

**File:** `src/retrieve.py` · **Stubs:** `fuse_rrf` · `hybrid_retrieve`

**The problem.** You cannot combine the two rankings with a weighted sum.
Cosine lives in [−1, 1] and clusters around 0.4–0.8; BM25 is unbounded and its
scale shifts *per query* depending on term rarity. Adding them is adding Celsius
to kilometres, and no fixed weight fixes it.

**The answer.** Throw the scores away, keep the order:

```
RRF(d) = Σ over ranked lists L of  1 / (k + rank_L(d))     # rank is 1-indexed
```

Ten lines, one parameter, no training, routinely beats tuned score blending.
`k=60` damps rank dominance so *agreement across retrievers* outweighs any one
retriever's confidence.

**Your deliverable for this phase.** The CLI prints dense, sparse, and hybrid
rankings side by side and flags `<- BM25 RESCUE` when a chunk that dense buried
gets pulled into the top-k by BM25. Find one and write it down — it is the
evidence in your README that hybrid earned its complexity.

```bash
uv run python -m src.retrieve "What is error code E204?"
uv run python -m src.retrieve "How long does the NB-1 take to fast charge?"
uv run python -m src.retrieve "What ingress protection rating does the NB-1 have?"
```

**DoD:** a recorded query where the right chunk is outside dense top-5 but
inside BM25 top-3.

---

## Phase 4 — Context + generation

**Files:** `src/context.py`, `src/generate.py`, `src/pipeline.py`
**Stubs:** `build_context` · `parse_citations` · `should_abstain`

**Concepts.**

- **Lost in the middle** — model accuracy on a fact is U-shaped in its position
  within a long context. Stuffing marginal chunks pushes good ones toward the
  low-attention middle, so a context budget is an accuracy decision, not just a
  cost one.
- **Citations as a hallucination detector** — if you supply 6 sources and the
  model writes `[S9]`, that marker refers to nothing. Counting out-of-range
  citations costs zero extra model calls.
- **Two places to refuse.** Retrieval gating (cheap, deterministic, blind to the
  question) versus prompted abstention (question-aware, costs a generation,
  unreliable). `should_abstain`'s docstring works through why RRF scores are a
  *bad* thing to threshold — every query has a rank-1 chunk, so the top RRF
  score is nearly constant regardless of relevance. Verify that yourself.

**Environment gotchas already handled for you:** `qwen3.8-27b` is a reasoning
model (`enable_thinking: false` is passed via `chat_template_kwargs`), and
llama-swap loads models on demand, so `Generator.warmup()` exists and the CLI
calls it before timing anything.

```bash
uv run python -m src.generate --self-check          # citation parsing, no server
uv run python -m src.pipeline "What is the NB-1 payload capacity?"
uv run python -m src.pipeline "Who is the CEO of Nimbus Robotics?"
uv run python -m src.pipeline --verbose "How do I file an incident report?"
```

**DoD:** cited answers on 5 questions; `Not found in sources.` on q091 (the
corpus genuinely never names a CEO); per-stage latency printed.

---
## Phase 5 — Self-correction loop

**Files:** `src/self_correct.py`, `scripts/calibrate.py`
**Stubs:** `split_sentences` · `jaccard` · `verify_sentence` · `self_correct` ·
`suggest_thresholds`

Phase 4 *asked* the model to stay grounded. Models don't reliably obey.
Prompting is a request; verification is a check. You need both.

**The mechanism.** Split the draft into sentences. For each, compute against
every retrieved chunk both a **cosine similarity** and a **Jaccard token
overlap**. A sentence is supported if **either** passes. The OR is the design:

- Cosine alone fails on this corpus's bread and butter — "costs $18,500" and
  "costs $19,500" sit at ~0.99 cosine, and the embedding barely notices the
  digit that makes one of them false.
- Jaccard alone fails on any genuine paraphrase.

Unsupported sentences trigger one targeted rewrite, capped by `max_iterations`.
`faithfulness = supported / total`.

**Concepts.**

- **Sentence splitting is not `text.split(".")`.** This corpus is full of
  `99.2%`, `$18,500`, `12,000 hours`. Naive splitting shatters them, the
  fragments fail verification, and your loop starts rewriting correct answers to
  fix a bug in your tokenizer.
- **Jaccard's denominator is the union**, so a 5-token sentence checked against
  a 60-token chunk tops out around 0.08 even when perfectly grounded. Read the
  note in `jaccard`'s docstring — *containment* (`|A∩B|/|A|`) is probably the
  metric you actually want, and if you switch, rename the function.
- **The cap is not optional.** Golden question q083 computes 12,400 × 0.88 =
  10,912; "10,912" appears nowhere in the corpus, so a *correct* answer is
  permanently unverifiable. Without a cap you've built a machine that burns GPU
  forever on a right answer.
- **Faithfulness is gameable alone.** A refusal asserts nothing, so it scores
  1.0. A system that refuses everything is perfectly faithful. This is exactly
  why Phase 6 reports it next to key-fact recall and false-refusal rate.

**Calibrate, don't guess.** `scripts/calibrate.py` builds two labelled score
populations without any hand-labelling: golden answers checked against *their
own* retrieved chunks (positives) versus the *same* answers checked against a
**different question's** chunks (negatives). That mismatched-pairing trick
yields hard negatives — fluent, on-topic corpus text that simply isn't supported
by that context. It then sweeps thresholds and reports sensitivity/specificity.

```bash
uv run python -m src.self_correct --self-check   # offline
uv run python scripts/calibrate.py --n 20        # measure your thresholds
uv run python -m src.pipeline --verify --verbose "What does the NB-2 cost?"
```

**DoD:** self-check green; thresholds chosen from the calibration output and
written into `config.yaml`; the loop provably terminates. Then the adversarial
test: paste a false claim into a draft and confirm it gets rewritten or dropped.

**If the distributions overlap heavily,** no threshold separates them and the
honest move is to say so — not to pick the least-bad number and present it as
calibrated. A metric built on a non-separating score is worse than no metric,
because it looks like evidence.

---

## Phase 6 — Evaluation & ablation

**File:** `scripts/run_eval.py`
**Stubs:** `key_fact_recall` · `context_precision_recall` · `mrr` · `percentile`

Everything before this was construction. "I built a hybrid RAG pipeline" is a
description. A *result* looks like: "hybrid raised context recall@8 from 0.71 to
0.89, driven entirely by the identifier questions; verification raised
faithfulness from 0.82 to 0.94 at the cost of 3 false refusals and 1.9s of p50."
This file is the difference between those two sentences.

**The four pipelines.**

| | Pipeline | Why it's in the table |
|---|---|---|
| **0** | No retrieval — entire corpus in the prompt | The corpus is ~2,900 tokens, so it *fits*. The honest upper bound. If B can't beat it, retrieval is subtracting value at this scale — and that's a real finding, not a failure. Most write-ups omit this baseline; the reason is not flattering. |
| **A** | Dense only | |
| **B** | Hybrid (dense + BM25 + RRF) | `A → B` isolates what BM25 + fusion bought you |
| **C** | Hybrid + Phase 5 verification | `B → C` isolates the verification loop |

**Use `eval/golden_set.json` as-is.** 100 questions already labelled. Its
`expected_sources` are **document**-level, so all retrieval metrics are computed
over documents — which is the more robust unit anyway, since re-chunking would
invalidate chunk-level labels and you *will* re-chunk.

**Metric concepts.**

- **Context recall@k is the ceiling on the whole system.** A fact never
  retrieved cannot be generated, cited, or verified. When end-to-end accuracy is
  bad, check recall *first* — usually the generator is being blamed for the
  retriever's miss.
- **Precision@k is bounded by `|expected|/k`.** A question with 3 expected
  sources cannot exceed 3/8 even with perfect retrieval. Don't read a low number
  as failure without checking that ceiling.
- **Dedupe before measuring precision.** You retrieve 8 *chunks* but score
  *documents*, and 8 chunks often come from 3–4 docs. Divide by 8 and you report
  roughly half the truth.
- **MRR exists because precision@k is rank-blind.** `[RIGHT,·,·,·,·,·,·,·]` and
  `[·,·,·,·,·,·,·,RIGHT]` both score 1/8; MRR scores them 1.0 and 0.125. But MRR
  only sees the *first* hit, so it and recall answer different questions —
  report both or mislead.
- **p50 and p95, never the mean.** The mean hides the tail and the tail is what
  users feel. "p95" is genuinely ambiguous at small n; the stub specifies
  nearest-rank so every number is an actually-observed value. Say which method
  you used — an unlabelled percentile is unreproducible.
- **Rejection rate is gameable alone**, so it's always reported beside
  false-refusal rate.

```bash
uv run python scripts/run_eval.py --n 20            # stratified sample
uv run python scripts/run_eval.py --all             # all 100, slow
uv run python scripts/run_eval.py --n 10 --pipelines B,C
```

Sampling is **stratified** and always keeps ≥5 negatives — a uniform sample of
20 would pull ~2, far too few to say anything about rejection. Runs are
**serial**: the backend uses `--parallel 1`, so concurrency adds queueing noise
to latency and nothing else. Budget roughly 8s/answer × questions × pipelines.

**DoD:** a four-column table with real numbers; `metrics.json` written; the
thesis checked (B > A on recall/precision, C ≥ B on faithfulness). If it
doesn't hold, debug retrieval and thresholds *before* concluding the thesis is
wrong.

**Two caveats to record.** `api-reference.md` backs only **1** of the 100
questions, so per-document conclusions about it are a sample of one. And with
n=100 (n=10 negatives), a 2–3 point gap is noise — report only what survives
that. The most informative thing in `metrics.json` is the per-question rows
where B *lost* to A.

---

## Phase 7 — API + UI

**Files:** `src/api.py`, `static/index.html` · **No stubs, deliberately.**

Every other phase leaves the interesting function blank. This one doesn't,
because there's no new RAG concept here — it's a web server in front of code you
already wrote. Making you retype FastAPI boilerplate would teach you FastAPI.

Two decisions in the file are worth reading:

- **Warm clients live for the process lifetime**, opened in the lifespan
  handler. Construct a `Generator` per request and the first request after any
  idle period pays model-load time — and your "why is it sometimes slow?"
  investigation ends there.
- **The response includes every retrieved chunk, not just cited ones**, because
  this is a debugging tool: seeing what was retrieved and *not* cited is how you
  diagnose a bad answer. A user-facing product would return citations only —
  and if you ever point this at a corpus with access controls, note that it
  hands the caller raw text for anything the retriever surfaced. Real systems
  filter by permission *before* retrieval, not after.

```bash
uv run uvicorn src.api:app --port 8000
```

The UI shows the answer, per-stage latency, faithfulness, and every chunk with
its dense rank, BM25 rank and RRF score — the same view as the Phase 3 CLI.

**DoD:** a browser query returns all fields; the second query is faster (warm).

---

## Phase 8 — Packaging

`README.md` is written with two clearly marked `TODO` blocks. You fill them:

1. **"Why hybrid: the evidence"** — paste the retrieval comparison for your
   Phase 3 BM25-rescue query.
2. **"Results"** — paste the ablation table. `run_eval.py` prints a
   ready-to-paste Markdown version.

Then write 3–5 sentences of **interpretation** under the table. The table is
data; the interpretation is the part that shows judgement. Say what surprised
you, and name the differences small enough to be noise. A table without
interpretation reads as a dump; a table whose author tells you which rows not to
trust reads as competence.

Remaining: a LICENSE, a fresh-clone quickstart test, and a demo GIF (record 2–3
queries including one unanswerable).

---

# Glossary

**Chunk** — the unit you index and retrieve. Not the same as a document.

**Dense retrieval** — matching by embedding similarity. Catches paraphrase,
blurs near-identical identifiers.

**Sparse / lexical retrieval** — matching by term overlap (BM25). The mirror
image: exact on identifiers, blind to paraphrase.

**Embedding** — a map from text to a point in ℝᵈ where nearby means similar in
meaning.

**Cosine similarity** — `(a·b)/(‖a‖‖b‖)`, in [−1, 1]. Angle, not distance.

**L2 normalization** — scaling a vector to length 1. Afterwards the plain dot
product *is* the cosine, which is why `IndexFlatIP` works.

**Inner product (IP) index vs L2 index** — for unit vectors they rank
identically (`‖a−b‖² = 2 − 2a·b`). IP is preferred because the returned score is
directly interpretable as a cosine.

**Exact vs ANN search** — `IndexFlatIP` compares against every vector: always
correct, O(n·d). `HNSW`/`IVF` search a subset: sublinear, may miss true
neighbours. You trade recall for latency, and only when latency is a problem.

**BM25** — `Σ_t IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|D|/avgdl))`.

**IDF** — inverse document frequency, ≈ `log(N/df(t))`. Rare terms score high;
terms in every chunk contribute nothing. This is what makes BM25 smart.

**k1** — term-frequency saturation (default 1.5). The tenth mention of a word
tells you much less than the first, so the curve flattens.

**b** — length normalization (default 0.75). Long chunks are discounted; short
chunks get inflated scores, which is why tiny chunks are dangerous.

**RRF (Reciprocal Rank Fusion)** — `Σ_L 1/(k + rank_L(d))`. Combines rankings
using ordinal information only, sidestepping incomparable score scales.

**Over-fetch** — retrieving k·m candidates per arm before fusing, so a chunk one
arm buried can still be rescued by the other.

**Contextual chunk header** — prepending the heading breadcrumb to a chunk so it
remains interpretable in isolation.

**Lost in the middle** — accuracy on a fact is U-shaped in its position within a
long context; middle is worst.

**Grounding** — the constraint that every claim trace to a supplied source.

**Abstention** — refusing to answer. Measured by rejection rate on negatives,
and always reported against the false-refusal rate on answerable questions.

**Faithfulness** — fraction of generated claims supported by retrieved context.
Distinct from correctness: an answer can be faithful to a wrong source, or
correct while citing nothing.

**Context precision@k** — fraction of retrieved chunks that are relevant.

**Context recall@k** — fraction of relevant material that was retrieved. *The
ceiling on the whole system.*

**MRR (Mean Reciprocal Rank)** — mean of `1/rank` of the first relevant result.
Rank-sensitive in a way precision@k is not.

**Jaccard similarity** — `|A ∩ B| / |A ∪ B|` over token sets. Crude, but catches
verbatim numbers and IDs that cosine smooths over.

**Ablation** — removing one component and re-measuring, to attribute the gain.
Without it, "we built a hybrid RAG pipeline" is a description, not a result.

**p50 / p95** — median and 95th-percentile latency. Report both; the mean hides
the tail.
