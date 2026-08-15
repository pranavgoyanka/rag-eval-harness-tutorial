"""
=============================================================================
PHASE 4 (part 1 of 3) — Context assembly & prompting
=============================================================================

WHAT YOU'RE BUILDING
    Retrieved chunks -> the text block that goes into the model's prompt.

WHY THIS ISN'T JUST "\\n".join(chunks)
    Three real constraints:

      1. BUDGET. Context is not free. It costs latency (every token is
         processed before the first output token appears) and, on a hosted
         model, money. More importantly it costs ACCURACY — see (2).

      2. POSITION. Models do not attend uniformly across a long context. The
         well-documented "lost in the middle" effect (Liu et al., 2023) is that
         accuracy on a fact placed in the middle of a long context is markedly
         worse than the same fact at the beginning or end — the curve is
         U-shaped. So ORDER matters, and stuffing in marginal chunks actively
         pushes your good chunks toward the low-attention middle.

      3. ATTRIBUTION. Every chunk needs a stable, visible label ([S1], [S2])
         so the model can cite it and so YOU can mechanically verify the
         citation later. Without labels, "the model said 25 kg" and "the model
         read 25 kg from a source" are indistinguishable — and that distinction
         is the entire subject of Phase 5.

WHAT TO IMPLEMENT
    One stub: build_context().

    The SYSTEM PROMPT below is given, deliberately. Prompt wording is not the
    lesson of this project, and a sloppy prompt would contaminate every metric
    you collect in Phase 6. Read it, understand each clause, change it later if
    you want — but change it BEFORE you record baseline numbers, not after.

DEPENDS ON: Phase 3.
=============================================================================
"""

from __future__ import annotations

from src.ingest import count_tokens
from src.retrieve import RetrievedChunk

# =============================================================================
# GIVEN: the sentinel for "I don't know"
# =============================================================================

# The exact string the model must emit when the context does not support an
# answer. It is a module constant, not a literal sprinkled through the code,
# because THREE separate things have to agree on it: the prompt below,
# pipeline.py's abstention path, and Phase 6's scoring of the 10 `negative`
# questions in eval/golden_set.json.
#
# (Note that golden_set.json records those as `expected_answer: "not_answerable"`.
# Phase 6 maps between the two — the golden set is a fixed artifact and the
# system's refusal wording is ours to choose.)
NOT_FOUND = "Not found in sources."


# =============================================================================
# GIVEN: the system prompt
# =============================================================================

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered sources provided.

Rules:
1. Every factual claim in your answer MUST come from the sources. Do not use
   prior knowledge, and do not infer facts that are not stated.
2. Cite the source for each claim inline, like [S1] or [S2][S3]. Cite the
   source you actually used.
3. If the sources do not contain the answer, reply with exactly:
   {NOT_FOUND}
   Do not apologise, do not speculate, and do not offer related information.
4. If the sources only PARTIALLY answer the question, state what they do
   support and explicitly say which part is unsupported.
5. Be concise. Answer the question that was asked; do not summarise the
   sources.
6. When the question asks for a specific value, give the value first, then any
   necessary qualification."""

# Notes on why each rule is there — worth reading before you edit them:
#   Rule 1 is the grounding constraint. Without it the model happily answers
#     from pretraining, and on a FICTIONAL corpus like this one that produces
#     confident, plausible, entirely invented answers.
#   Rule 2 is what makes Phase 5 possible at all. You cannot verify a claim
#     against its source if the model never said which source it used.
#   Rule 3 must specify EXACT wording. "Say you don't know" produces fifteen
#     different phrasings and an unscoreable eval.
#   Rule 4 exists because the failure mode of Rule 3 is over-refusal — a model
#     told to refuse when unsure starts refusing questions it could answer.
#     Watch this tension in your Phase 6 numbers; it is a real tradeoff, not a
#     prompt-wording problem.
#   Rule 6 makes `expected_value` matching in Phase 6 far less noisy.


USER_TEMPLATE = """Sources:
{context}

Question: {question}"""


# =============================================================================
# TODO(Phase 4) #1 — build_context
# =============================================================================


def build_context(
    chunks: list[RetrievedChunk], token_budget: int
) -> tuple[str, list[RetrievedChunk]]:
    """Format retrieved chunks into a numbered source block within a budget.

    ---------------------------------------------------------------------------
    TARGET OUTPUT FORMAT
    ---------------------------------------------------------------------------

        [S1] corpus/product-nb1.md — NB-1 "Courier" > General specifications
        | Payload capacity | 25 kg |
        | Top speed | 24 km/h |

        [S2] corpus/product-nb2.md — NB-2 "Hauler" > General specifications
        | Payload capacity | 60 kg |

    Each entry: the marker, the source path, the heading breadcrumb, then the
    chunk body. The path and breadcrumb are there so the model can say
    "according to the NB-1 spec sheet" rather than "according to [S1]", and so
    that a human reading the transcript can find the source without grep.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Walk `chunks` in the order given (already sorted by RRF score, best
         first — do not re-sort).
      2. Track a running token total using count_tokens() from ingest.py.
         Reuse it rather than inventing a second counter; two different token
         estimates in one codebase is a bug waiting to happen.
      3. Stop adding chunks once the next one would exceed `token_budget`.
         DECISION FOR YOU: stop entirely, or skip the oversized chunk and try
         the next (smaller) one? Skipping packs the budget more fully but
         reorders relevance. Pick one and say why in a comment.
      4. Number the INCLUDED chunks from 1. The numbering must match what you
         return — this is the subtle part. If you number by position in the
         input but drop chunk #3, then [S4] in the model's answer maps to the
         wrong chunk and every citation downstream is silently misattributed.
         Number as you append, never before filtering.
      5. Always include at least one chunk if any were passed, even if it alone
         exceeds the budget. Returning an empty context makes the model refuse
         every question and is much harder to debug than a slightly-over-budget
         prompt.
      6. Return (context_string, included_chunks) — the second element is the
         [S1]..[Sn] mapping that generate.py::parse_citations needs.

    ---------------------------------------------------------------------------
    ORDERING — a thing to actually try
    ---------------------------------------------------------------------------
    The natural implementation puts the best chunk first, and that's a fine
    default. But given the U-shaped attention curve described at the top of this
    file, some systems place the top-ranked chunk LAST, closest to the question,
    on the theory that recency wins. Others interleave: 1st, 3rd, 5th, ...,
    6th, 4th, 2nd, so the two best chunks bracket the weak ones.

    You now have a golden set of 100 questions. In Phase 6 you can settle this
    empirically for your corpus instead of arguing about it. Leave a note here
    with what you measured.
    """
    # TODO(Phase 4): implement per the recipe above.
    raise NotImplementedError("Phase 4: build_context")


# =============================================================================
# GIVEN: message assembly
# =============================================================================


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    """Assemble the OpenAI-style message list.

    Note the split: static instructions in the SYSTEM message, per-query data in
    the USER message. This is not cosmetic — it keeps the system prompt byte-
    identical across every request, which is what lets a server reuse a cached
    KV prefix. On a local llama.cpp server that is a real latency win once you
    are running 100 eval questions back to back.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]


if __name__ == "__main__":
    print("SYSTEM PROMPT")
    print("=" * 70)
    print(SYSTEM_PROMPT)
    print()
    print(f"refusal sentinel: {NOT_FOUND!r}")
    print("\nThis module has one stub: build_context(). Run the full pipeline with")
    print("  uv run python -m src.pipeline \"your question\"")
