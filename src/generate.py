"""
=============================================================================
PHASE 4 (part 2 of 3) — Generation
=============================================================================

WHAT YOU'RE BUILDING
    messages -> the model's answer, plus the citations it used.

WHAT'S GIVEN
    The whole HTTP client. Calling an OpenAI-compatible endpoint is not the
    lesson here, and there are two environment-specific details baked in that
    would otherwise cost you an afternoon:

      * enable_thinking. qwen3.8-27b is a REASONING model: left to its own
        devices it emits a long chain of thought into a `reasoning_content`
        field before answering. That triples your latency and pollutes citation
        parsing. config.yaml turns it off via chat_template_kwargs.

      * warmup(). llama-swap loads models on demand. Your FIRST request pays
        multi-second model-load time; every subsequent one does not. If you
        measure without warming up, your p50 latency is a lie and your "second
        query is faster!" observation is measuring disk I/O, not your pipeline.

WHAT TO IMPLEMENT
    One stub: parse_citations().

DEPENDS ON: Phase 4 part 1 (context.py).
=============================================================================
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from src.config import Config, load_config
from src.context import NOT_FOUND, SYSTEM_PROMPT
from src.retrieve import RetrievedChunk

# =============================================================================
# GIVEN: result type
# =============================================================================


@dataclass
class Generation:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


# =============================================================================
# TODO(Phase 4) #2 — parse_citations
# =============================================================================

CITATION_RE = re.compile(r"\[S(\d+)\]")


def parse_citations(
    answer: str, sources: list[RetrievedChunk]
) -> tuple[list[str], list[int]]:
    """Extract [S1]-style markers and map them back to chunk ids.

    ---------------------------------------------------------------------------
    WHY THIS MATTERS MORE THAN IT LOOKS
    ---------------------------------------------------------------------------
    A citation is the only mechanical link between a generated claim and the
    evidence for it. Everything in Phase 5 (verification) and half of Phase 6
    (faithfulness) is built on top of this function.

    And it produces a hallucination signal for free. If you gave the model 6
    sources and it writes "[S9]", that marker refers to nothing. The model did
    not miscount — it produced a citation-shaped token because citations were
    expected there, which is a direct symptom of ungrounded generation. Counting
    out-of-range citations is one of the cheapest hallucination detectors you
    will ever build, and it needs no second model call.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Find all matches of CITATION_RE (already defined above).
      2. Convert each captured group to int. These are 1-INDEXED (S1 is the
         first source), so `sources[n - 1]` is the chunk. Off-by-one here
         misattributes every single citation while looking completely fine.
      3. Partition into:
           * valid   -> 1 <= n <= len(sources): collect sources[n-1].chunk_id
           * invalid -> anything else (0, negative, or > len(sources))
      4. DEDUPE the valid chunk ids while PRESERVING first-appearance order.
         `dict.fromkeys(seq)` does this in one expression. Order matters because
         the first citation usually points at the primary source, and you will
         want that when displaying results.
      5. Return (valid_chunk_ids, invalid_indices).

    Edge case to handle deliberately: an answer of exactly NOT_FOUND has zero
    citations, and that is CORRECT, not a failure. Do not treat "no citations"
    as an error condition — pipeline.py distinguishes the two cases.
    """
    # TODO(Phase 4): implement per the recipe above.

    # print(answer)
    cite = re.findall(CITATION_RE, answer)
    # print(cite)

    cite_int = []

    for c in cite:
        cite_int.append(int(c))

    valid: list[str] = []
    invalid: list[int] = []

    for c in cite_int:
        if c <= 0 or c > len(sources):
            invalid.append(c)
        else:
            valid.append(sources[c - 1].chunk_id)

    valid = list((dict.fromkeys(valid)).keys())
    invalid = list((dict.fromkeys(invalid)).keys())

    return valid, invalid
    raise NotImplementedError("Phase 4: parse_citations")


# =============================================================================
# GIVEN: the chat client
# =============================================================================


class Generator:
    """OpenAI-compatible chat client for the local llama.cpp server."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.server.base_url,
            timeout=cfg.server.timeout,
            headers={"Authorization": f"Bearer {cfg.server.api_key}"},
        )
        self._warm = False

    def _body(self, messages: list[dict[str, str]], **overrides: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.cfg.models.chat,
            "messages": messages,
            # temperature 0 = greedy decoding. Non-negotiable for an eval
            # harness: with sampling on, rerunning the same question gives a
            # different answer, and you can no longer attribute a metric change
            # to the change you made.
            "temperature": self.cfg.generation.temperature,
            "max_tokens": self.cfg.generation.max_tokens,
        }
        if not self.cfg.generation.enable_thinking:
            # llama.cpp passes this through to the chat template. See module
            # docstring — without it qwen3.8 reasons before every answer.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        body.update(overrides)
        return body

    def warmup(self) -> float:
        """Force the model to load. Call once before timing anything.

        Returns seconds spent. On a cold llama-swap router expect several
        seconds; on a warm one, well under one.
        """
        if self._warm:
            return 0.0
        t0 = time.perf_counter()
        self._client.post(
            "/chat/completions",
            json=self._body([{"role": "user", "content": "ok"}], max_tokens=1),
        )
        self._warm = True
        return time.perf_counter() - t0

    def complete(self, messages: list[dict[str, str]], **overrides: Any) -> Generation:
        t0 = time.perf_counter()
        try:
            resp = self._client.post(
                "/chat/completions", json=self._body(messages, **overrides)
            )
            resp.raise_for_status()
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Could not reach {self.cfg.server.base_url}: {exc}\n"
                "Run `python -m src.config --check` to diagnose."
            ) from exc
        dt = time.perf_counter() - t0

        data = resp.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage", {})

        # If thinking slipped through (someone flipped enable_thinking, or the
        # template ignored the kwarg), say so loudly rather than silently
        # returning a weirdly-shaped answer.
        if msg.get("reasoning_content"):
            print(
                "  note: model returned reasoning_content "
                f"({len(msg['reasoning_content'])} chars). Set "
                "generation.enable_thinking: false in config.yaml to disable.",
                file=sys.stderr,
            )

        return Generation(
            text=(msg.get("content") or "").strip(),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_s=dt,
            raw=data,
        )

    def close(self) -> None:
        self._client.close()


# =============================================================================
# GIVEN: self-check
# =============================================================================


def self_check(cfg: Config) -> None:
    # --- 1. citation parsing (no server needed) ----------------------------
    fake = [RetrievedChunk(chunk=c, rrf_score=0.0) for c in _FakeChunks()]
    ids, bad = parse_citations("The payload is 25 kg [S1] and speed 24 km/h [S2].", fake)
    assert ids == ["a::000", "b::001"], f"expected both chunk ids, got {ids}"
    assert bad == [], f"expected no invalid citations, got {bad}"
    print("  [1/4] ok  in-range citations map to chunk ids")

    ids, bad = parse_citations("Confidently wrong [S9].", fake)
    assert ids == [] and bad == [9], f"expected ([], [9]), got ({ids}, {bad})"
    print("  [2/4] ok  out-of-range citation flagged as a hallucination signal")

    ids, _ = parse_citations("Both [S2] and again [S2] and [S1].", fake)
    assert ids == ["b::001", "a::000"], f"dedupe/order wrong: {ids}"
    print("  [3/4] ok  citations deduped in first-appearance order")

    ids, bad = parse_citations(NOT_FOUND, fake)
    assert ids == [] and bad == [], "refusal should parse cleanly with no citations"
    print("  [4/4] ok  refusal parses as zero citations, not an error")

    print("\nPHASE 4 (generate) SELF-CHECK PASSED")


class _FakeChunks:
    """Two minimal stand-ins so the citation checks need no index or server."""

    def __iter__(self):
        from src.ingest import Chunk

        yield Chunk("a::000", "a.md", "corpus/a.md", ["A"], "text a", "text a", 2)
        yield Chunk("b::001", "b.md", "corpus/b.md", ["B"], "text b", "text b", 2)


if __name__ == "__main__":
    cfg = load_config()
    try:
        if "--self-check" in sys.argv:
            self_check(cfg)
        else:
            g = Generator(cfg)
            print(f"warmup: {g.warmup():.2f}s")
            out = g.complete(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Sources:\n\nQuestion: What is 2+2?"},
                ]
            )
            print(f"latency: {out.latency_s:.2f}s  "
                  f"tokens: {out.prompt_tokens} in / {out.completion_tokens} out")
            print(f"answer: {out.text!r}")
            print(f"\n(expecting {NOT_FOUND!r} — there were no sources)")
            g.close()
    except NotImplementedError as exc:
        print(f"Not implemented yet: {exc}")
        print("\nFill in parse_citations() in src/generate.py.")
        raise SystemExit(1) from None
