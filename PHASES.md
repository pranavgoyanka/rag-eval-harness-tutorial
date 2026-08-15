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
| 5 | Self-correction loop | 📋 designed below | — |
| 6 | Evaluation & ablation | 📋 designed below | — |
| 7 | API + UI | 📋 designed below | — |
| 8 | Packaging | 📋 designed below | — |

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

# Phases 5–8 — designed, not yet built

No code exists for these. This is the design so you know where it's going.

## Phase 5 — Self-correction loop

**New file:** `src/self_correct.py`. Wires into the `iterations` and
`faithfulness` fields already declared on `QueryResult`.

The idea: after generating a draft, check each sentence against the retrieved
chunks and rewrite or drop the unsupported ones.

1. Split the answer into sentences.
2. For each sentence, compute against every retrieved chunk:
   - **cosine similarity** of their embeddings, and
   - **Jaccard token overlap** — `|A ∩ B| / |A ∪ B|` over `tokenize()`'d terms.
3. A sentence is **supported** if `cosine ≥ SIM_T` **OR** `jaccard ≥ JAC_T`.
   The OR matters: cosine catches paraphrase, Jaccard catches the case cosine is
   worst at — numbers and identifiers copied verbatim ("`$18,500`", "`E204`"),
   which is most of this corpus.
4. Unsupported sentences → one rewrite pass with the offending sentence named,
   capped at `max_iterations` so the loop always terminates.
5. `faithfulness = supported_sentences / total_sentences`.

**Why deterministic checks instead of an NLI model or an LLM judge?** It is
fast, free, reproducible, and debuggable. It is also *weaker* — it cannot catch
a sentence that copies the right words while inverting the meaning. Name that
limitation in your README rather than overselling the metric.

**Calibration, not guessing.** `scripts/calibrate.py` runs ~10 known-good
answers and prints the (cosine, jaccard) distribution per sentence. Choose
thresholds from where the supported and unsupported populations separate. Two
thresholds picked by vibes will quietly decide every faithfulness number you
report.

**Adversarial check:** inject a false claim into a draft and confirm the loop
rewrites or drops it.

## Phase 6 — Evaluation & ablation

**New file:** `scripts/run_eval.py`. Output: `metrics.json` + a printed table.

**Use `eval/golden_set.json` as-is — do not author a new test set.** It already
has 100 labelled questions (factual 45, procedural 15, multi_hop 12,
comparison 10, negative 10, numeric 8; easy 55 / medium 29 / hard 16).

**Its `expected_sources` are labelled at DOCUMENT level**, not chunk level. So
compute context precision/recall over source documents. This deletes the
chunk-id labelling work you might otherwise have budgeted for — and it is the
more robust unit anyway, since re-chunking would invalidate chunk-level labels.

**Four pipelines**, not three:

| | Pipeline | Why it's in the table |
|---|---|---|
| **0** | No retrieval — stuff the entire corpus | The corpus is only ~2,900 tokens, so it *fits in context*. This is the honest upper bound. If RAG doesn't beat it, your retrieval is subtracting value. Most write-ups omit this baseline because it's unflattering. |
| **A** | Dense only | |
| **B** | Hybrid (dense + BM25 + RRF) | Expect B > A on recall, driven by the identifier questions |
| **C** | Hybrid + Phase 5 verification | Expect C ≥ B on faithfulness, possibly worse on answer recall (verification drops true-but-unverifiable sentences) |

**Metrics**

- **Key-fact recall** — fraction of `key_facts` present in the answer
  (whitespace-insensitive substring). Cheap and surprisingly effective.
- **Context precision@k** — of the k retrieved chunks, what fraction come from
  `expected_sources`.
- **Context recall@k** — of the `expected_sources`, what fraction appear in the
  retrieved set. *This is the ceiling on everything downstream:* a fact not
  retrieved cannot be generated.
- **MRR** — mean of `1/rank_of_first_relevant_chunk`. Precision@k treats rank 1
  and rank 8 alike; MRR does not.
- **Rejection rate** — on the 10 negatives, did it emit `NOT_FOUND`? Report the
  false-refusal rate on the other 90 alongside it, or the metric is gameable by
  refusing everything.
- **Faithfulness** — from Phase 5.
- **p50 / p95 latency** — report both. The mean hides the tail, and the tail is
  what users feel. Warm up first, and run **serially** (`--parallel 1` on the
  backend means concurrency buys nothing but noise).

**Two caveats to record in the README.** `api-reference.md` backs only **1** of
the 100 questions, so per-document recall on it is statistically meaningless —
don't read anything into that row. And with n=100 (n=10 for negatives), a
few-point difference between pipelines is noise; report the differences that
survive that.

**Sanity check before you write it up:** B > A on recall/precision, C ≥ B on
faithfulness. If not, debug retrieval and thresholds *before* concluding the
thesis is wrong.

## Phase 7 — API + UI

**New files:** `src/api.py`, `static/index.html`. Add `uv add fastapi uvicorn`
(already declared as the `api` optional-dependency group in `pyproject.toml`).

`POST /query` → `QueryResult` as JSON; `GET /health`; static mount.
The UI shows the answer, citation chips, faithfulness, and the latency
breakdown. Hold one warm `Generator` for the process lifetime — reconnecting
per request re-pays warmup.

## Phase 8 — Packaging

Rewrite `README.md`: problem statement, architecture diagram, the **BM25 rescue
example from Phase 3**, the ablation table from Phase 6, quickstart, and 3–4
design-tradeoff bullets (FlatIP vs HNSW; deterministic verification vs NLI;
document-level vs chunk-level eval labels; why the no-RAG baseline is in the
table). Fresh-clone test. Demo GIF with one unanswerable question.

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
