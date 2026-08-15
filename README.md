# RAG + Eval Harness

A minimal end-to-end RAG evaluation setup: a small Markdown knowledge base (the
corpus, about a fictional company "Nimbus Robotics") plus a **golden set of 100
questions** with reference answers and source attribution, ready to drive a
retrieve → generate → evaluate pipeline.

## Layout

```
rag-eval-harness/
├── README.md
├── corpus/                      # knowledge base (chunk + index these)
│   ├── company-overview.md
│   ├── product-nb1.md
│   ├── product-nb2.md
│   ├── fleet-operations.md
│   ├── safety-protocols.md
│   ├── pricing.md
│   ├── api-reference.md
│   └── faq-troubleshooting.md
└── eval/
    └── golden_set.json          # 100 questions with expected answers
```

## Golden set schema

Each entry in `eval/golden_set.json` → `questions[]`:

| Field               | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| `id`                | Stable identifier (`q001`–`q100`)                                  |
| `question`          | The user question                                                  |
| `expected_answer`   | Reference answer, or `"not_answerable"` for negative questions     |
| `expected_value`    | Canonical final value for exact/normalized matching                |
| `expected_sources`  | Corpus files (paths relative to repo root) that support the answer |
| `key_facts`         | Literal corpus substrings (whitespace-insensitive) a correct answer should reflect; for numeric questions these are the input values from the corpus |
| `category`          | `factual`, `procedural`, `comparison`, `multi_hop`, `numeric`, `negative` |
| `difficulty`        | `easy`, `medium`, `hard`                                           |
| `should_reject`     | `true` only for `negative` questions (model should refuse)         |

Category counts: factual 45, procedural 15, comparison 10, multi_hop 12,
numeric 8, negative 10.

## Suggested harness flow

1. **Chunk** the corpus (e.g., split by Markdown heading, ~300–500 tokens).
2. **Index** chunks with your embedder of choice (any vector store).
3. **Retrieve** top-k chunks for each question.
4. **Generate** an answer (system prompt should instruct the model to answer
   only from context and say so when the answer is not in the context).
5. **Score** against the golden set.

## Suggested metrics

- **Retrieval**: context precision / recall — does the retrieved set cover
  `expected_sources`?
- **Answer correctness**: fraction of `key_facts` present in the generated
  answer (exact-match or LLM-judge).
- **Unanswerable rejection rate**: for `negative` questions, did the model
  refuse instead of hallucinating?
- **Faithfulness / hallucination**: LLM-judge check that every claim is
  supported by the retrieved context.
- **Latency / cost**: per-question wall time and token usage.

## Notes

- The corpus is fictional; all facts are internally consistent, so every
  non-negative question has a single correct answer derivable from the docs.
- `negative` questions are intentionally close to real topics (e.g., "NB-3",
  "48-month lease") to test that the system rejects plausible-but-absent facts.
