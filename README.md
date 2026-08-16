# RAG + Eval Harness

A hybrid-retrieval RAG pipeline built from primitives — no LangChain, no vector
database service — with a **real evaluation harness** measuring what each
component actually contributes.

The interesting part is not the pipeline. It's the ablation table: four
configurations of the same system, scored on 100 labelled questions, so that
every design decision is attached to a number rather than a plausible story.

> **This repository is a teaching scaffold.** The plumbing is written; the
> conceptually interesting functions are stubs with detailed recipes in their
> docstrings. See **[PHASES.md](PHASES.md)** for the build order, the concepts,
> and the glossary. If you want the finished system, you build it — that is the
> point.

---

## The problem

Ask a language model a question about your internal documentation and it will
answer confidently whether or not it knows. Retrieval-augmented generation is
the standard fix: fetch relevant passages, put them in the prompt, instruct the
model to use only those. But "put RAG on it" hides four decisions that determine
whether the result works:

1. **How you chunk** sets a hard ceiling on what can ever be retrieved.
2. **How you retrieve** determines whether exact identifiers (`NB-1`, `E204`)
   survive — embeddings are bad at precisely the strings that matter most in
   technical docs.
3. **How you handle absence** decides whether "we don't document that" comes
   back as a refusal or as a fluent invention.
4. **How you measure** decides whether you know any of the above.

This project makes each decision explicit, then measures it.

---

## Architecture

```
  corpus/*.md
      │
      ▼
 ┌──────────────────────────────────────────────────────────┐
 │ INGEST (Phase 1)                                         │
 │   split on Markdown headings, never mid-table            │
 │   merge sections under 40 tokens                         │
 │   prepend heading breadcrumb  ← "contextual chunk header" │
 └──────────────────────────────────────────────────────────┘
      │  artifacts/chunks.jsonl
      ├─────────────────────────┬──────────────────────────┐
      ▼                         ▼                          │
 ┌─────────────────┐    ┌──────────────────┐               │
 │ DENSE (Phase 2) │    │ SPARSE (Phase 2) │               │
 │ embeddings via  │    │ BM25Okapi        │               │
 │ llama.cpp       │    │ identifier-safe  │               │
 │ FAISS IndexFlatIP│   │ tokenizer        │               │
 │ = cosine top-k  │    │ = exact-term     │               │
 └─────────────────┘    └──────────────────┘               │
      │  top k·3               │  top k·3                  │
      └───────────┬────────────┘                           │
                  ▼                                        │
        ┌───────────────────────┐                          │
        │ RRF FUSION (Phase 3)  │   score(d)=Σ 1/(60+rank) │
        │ rank-only, no score   │                          │
        │ blending              │                          │
        └───────────────────────┘                          │
                  │  top k                                 │
                  ▼                                        │
        ┌───────────────────────┐                          │
        │ CONTEXT (Phase 4)     │  numbered [S1..Sn],       │
        │ token budget, order   │  breadcrumbs, budget trim │
        └───────────────────────┘                          │
                  ▼                                        │
        ┌───────────────────────┐                          │
        │ GENERATE (Phase 4)    │  grounded prompt,         │
        │ qwen3.8-27b, temp 0   │  cite or refuse           │
        └───────────────────────┘                          │
                  ▼                                        │
        ┌───────────────────────┐                          │
        │ VERIFY (Phase 5)      │  per sentence:            │
        │ cosine OR Jaccard     │  supported? rewrite?      │
        │ bounded rewrite loop  │  → faithfulness           │
        └───────────────────────┘                          │
                  ▼                                        ▼
             cited answer                        ┌────────────────────┐
                  │                              │ EVAL (Phase 6)     │
                  └─────────────────────────────▶│ 100 golden Qs      │
                                                 │ 4-way ablation     │
                                                 │ → metrics.json     │
                                                 └────────────────────┘
```

---

## Quickstart

**Prerequisite.** An OpenAI-compatible server with **both** a chat model and an
embedding model. This project was built against
[llama.cpp](https://github.com/ggml-org/llama.cpp) behind
[llama-swap](https://github.com/mostlygeek/llama-swap). A plain chat server
returns `501` on `/v1/embeddings` — you need a second model launched with
`--embeddings --pooling mean` (e.g. `nomic-ai/nomic-embed-text-v1.5-GGUF`).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Preflight: probes /v1/models, /v1/chat/completions, /v1/embeddings.
# On failure it prints the exact server config you're missing.
uv run python -m src.config --check

# Build the indexes
uv run python -m src.ingest
uv run python -m src.index_dense
uv run python -m src.index_sparse

# Ask something
uv run python -m src.pipeline "What is the NB-1 payload capacity?"
uv run python -m src.pipeline "Who is the CEO of Nimbus Robotics?"   # refuses

# Inspect retrieval (dense vs BM25 vs fused, side by side)
uv run python -m src.retrieve "What is error code E204?"

# Evaluate
uv run python scripts/calibrate.py            # measure verification thresholds
uv run python scripts/run_eval.py --n 20      # 4-way ablation

# Browser UI
uv run uvicorn src.api:app --port 8000
```

Config lives in [`config.yaml`](config.yaml). Point it at a different server
with `export RAG_BASE_URL=http://your-host:8101/v1`.

---

## Why hybrid retrieval: the evidence

<!-- ══════════════════════════════════════════════════════════════════════ -->
<!-- FILL THIS IN AFTER PHASE 3.                                            -->
<!--                                                                        -->
<!-- Run:  uv run python -m src.retrieve "<query>"                          -->
<!-- The CLI flags "<- BM25 RESCUE" when a chunk BM25 ranked in its top 3   -->
<!-- was outside dense's top-k. Paste that output here.                     -->
<!--                                                                        -->
<!-- Candidates on this corpus: "E204", "IP65", "36-month", "MTBF",         -->
<!-- "NB-2 payload". One real example beats three paragraphs of theory.     -->
<!-- ══════════════════════════════════════════════════════════════════════ -->

> **TODO:** paste the retrieval comparison for one query where BM25 rescued a
> chunk dense retrieval buried, with a sentence explaining why the embedding
> missed it.

The mechanism is straightforward: embeddings compress meaning, and near-identical
strings with unrelated meanings — `NB-1` vs `NB-2`, `E101` vs `E204` — compress
to nearly the same vector. BM25 treats them as distinct atoms. **53 of the 100
golden questions turn on telling NB-1 from NB-2**, so this is not a corner case
on this corpus; it is the main case.

The reverse blind spot is real too, and asserted in the Phase 2 self-check: for
*"how quickly does the courier travel"*, BM25 ranks the chunk that literally
contains `Top speed | 24 km/h` at **#15 of 35**, because the query shares no
content words with it. Neither retriever is sufficient alone.

---

## Results

<!-- ══════════════════════════════════════════════════════════════════════ -->
<!-- FILL THIS IN AFTER PHASE 6.                                            -->
<!--                                                                        -->
<!-- Run:  uv run python scripts/run_eval.py --all                          -->
<!-- It prints a ready-to-paste Markdown table under "Markdown (paste into  -->
<!-- your README)". Replace the placeholder table below with it.            -->
<!--                                                                        -->
<!-- Then write 3-5 sentences of INTERPRETATION underneath. The table is    -->
<!-- data; the interpretation is the part that shows judgement. Say what    -->
<!-- surprised you, and name the differences small enough to be noise.      -->
<!-- ══════════════════════════════════════════════════════════════════════ -->

> **TODO:** replace with real output from `scripts/run_eval.py`.

| metric | 0 — no-RAG (full corpus) | A — dense only | B — hybrid (RRF) | C — hybrid + verify |
|---|---|---|---|---|
| key-fact recall | — | — | — | — |
| context prec@k | n/a | — | — | — |
| context recall@k | n/a | — | — | — |
| MRR | n/a | — | — | — |
| rejection (negatives) | — | — | — | — |
| false refusal | — | — | — | — |
| faithfulness | — | — | — | — |
| p50 / p95 latency | — | — | — | — |

**The four pipelines.** `0` skips retrieval entirely and puts the whole corpus in
the prompt — possible because the corpus is only ~2,900 tokens, and included
because it is the honest upper bound. If hybrid retrieval cannot beat "paste
everything", retrieval is subtracting value at this scale, and that is a finding
worth reporting rather than hiding. `A` → `B` isolates the contribution of BM25 +
RRF. `B` → `C` isolates the verification loop.

**Caveats to state alongside any numbers.** With n=100 (n=10 for negatives), a
2–3 point gap is noise. `api-reference.md` backs exactly 1 of the 100 questions,
so nothing can be concluded about retrieval on that document. Rejection rate is
gameable on its own — a system refusing everything scores 1.0 — so it is only
meaningful next to the false-refusal rate.

---

## Design decisions & tradeoffs

**Heading-section chunks (~40–250 tokens), not the usual 300–500.**
The conventional advice would produce roughly one chunk per document on an
11 KB corpus, making retrieval trivially perfect and the evaluation meaningless.
Chunk size is a property of your data, not a constant you copy from a blog post.
Splitting on headings also guarantees Markdown tables stay intact — asserted in
the ingest self-check, and load-bearing here since most facts live in tables.

**`IndexFlatIP` (exact) over HNSW/IVF (approximate).**
At 35 chunks, exact search is microseconds and approximate indexes would be
resume-driven design. The scaffold implements dense search in numpy *first*,
then in FAISS, and asserts the two agree — because knowing that a vector store
at this scale is one matrix multiply is more useful than having used one. ANN
becomes the right call somewhere around 10⁵–10⁶ vectors, where the O(n·d) scan
stops fitting the latency budget; the tradeoff you accept there is recall.

**Reciprocal Rank Fusion over weighted score blending.**
Cosine lives in [−1, 1]; BM25 is unbounded and its scale shifts per query with
term rarity. Summing them is summing incompatible units, and no fixed weight
fixes it because the imbalance is not constant. RRF discards magnitudes and
fuses on rank alone: ten lines, one parameter, no calibration, no training.

**Deterministic verification over an NLI model or LLM-judge.**
Sentence-level cosine **OR** Jaccard is fast, free, reproducible and debuggable.
It is also **strictly weaker**: it is a similarity check, not an entailment
check, so it cannot catch a sentence that reuses the source's vocabulary while
inverting its meaning. Thresholds are measured by `scripts/calibrate.py` against
grounded and mismatched sentence populations, not hand-picked — otherwise the
headline faithfulness number is a function of a guess.

**Document-level evaluation labels, not chunk-level.**
The golden set labels `expected_sources` per document. Keeping the metrics at
that granularity means re-chunking does not invalidate the labels — and you will
re-chunk. Chunk-level labels are more precise and more brittle; here the
brittleness costs more than the precision buys.

**Two independent places to refuse.**
A retrieval-confidence gate (cheap, deterministic, blind to the question) and
prompted abstention (question-aware, costs a generation, unreliable). They catch
different things: the gate catches "nothing relevant exists"; the prompt catches
"relevant material exists but doesn't contain this fact". The gate ships
disabled by default because RRF scores are built from ranks alone and are nearly
constant across queries — a poor thing to threshold. See
`pipeline.py::should_abstain`.

---

## The corpus and golden set

`corpus/` holds 8 Markdown documents about a fictional company, Nimbus Robotics
(sidewalk delivery robots). Fictional on purpose: a model cannot fall back on
pretraining, so every unsupported claim is unambiguously a hallucination.

`eval/golden_set.json` holds 100 questions. Each entry:

| Field | Description |
|---|---|
| `id` | Stable identifier (`q001`–`q100`) |
| `question` | The user question |
| `expected_answer` | Reference answer, or `"not_answerable"` for negatives |
| `expected_value` | Canonical final value for exact/normalized matching |
| `expected_sources` | Corpus files supporting the answer (repo-root relative) |
| `key_facts` | Literal corpus substrings a correct answer should reflect |
| `category` | `factual` · `procedural` · `comparison` · `multi_hop` · `numeric` · `negative` |
| `difficulty` | `easy` · `medium` · `hard` |
| `should_reject` | `true` only for negatives |

Categories: factual 45, procedural 15, multi_hop 12, comparison 10, negative 10,
numeric 8. Difficulty: easy 55, medium 29, hard 16.

The 10 negatives are deliberately *near misses* — "NB-3", "48-month lease", "the
CEO" — chosen to sit next to real corpus content so that refusing them requires
actually checking rather than noticing an unfamiliar topic.

---

## Layout

```
├── config.yaml            all tunables, commented
├── PHASES.md              build guide, concepts, glossary
├── corpus/                8 Markdown documents
├── eval/golden_set.json   100 labelled questions
├── src/
│   ├── config.py          config + `--check` preflight        Phase 0
│   ├── ingest.py          chunking                            Phase 1
│   ├── embed.py           embedding client + cache            Phase 2
│   ├── index_dense.py     numpy + FAISS vector search         Phase 2
│   ├── index_sparse.py    BM25                                Phase 2
│   ├── retrieve.py        RRF fusion, comparison CLI          Phase 3
│   ├── context.py         prompt + context budgeting          Phase 4
│   ├── generate.py        chat client, citation parsing       Phase 4
│   ├── pipeline.py        orchestration + ablation modes      Phase 4
│   ├── self_correct.py    sentence verification loop          Phase 5
│   └── api.py             FastAPI                             Phase 7
├── scripts/
│   ├── calibrate.py       threshold calibration               Phase 5
│   └── run_eval.py        4-way ablation → metrics.json       Phase 6
└── static/index.html      browser UI                          Phase 7
```

Build artifacts (`artifacts/`) are generated and gitignored.

---

## Limitations

- **Corpus scale.** 35 chunks. Conclusions about retrieval here do not
  automatically transfer to 35,000, and the honest upper-bound baseline
  (pipeline 0) only exists *because* the corpus fits in a context window.
- **Verification is similarity, not entailment.** It cannot detect a negated
  claim built from the source's own vocabulary.
- **Key-fact recall is substring matching.** It scores "twenty-five kg" wrong
  and "does NOT have a 60 kg payload" right. Cheap and reproducible, but it is a
  proxy for correctness, not correctness.
- **Single generator model, single embedding model.** Every number is
  conditional on that pair.
- **Sample size.** 100 questions, 10 of them negative. Small differences are
  noise.

---

## Demo

<!-- TODO(Phase 8): record 2-3 queries including one unanswerable, and one
     showing the retrieval comparison CLI. Drop the GIF in docs/ and embed it
     here. -->
