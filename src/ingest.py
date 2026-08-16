"""
=============================================================================
PHASE 1 — Ingestion & Chunking
=============================================================================

WHAT YOU'RE BUILDING
    corpus/*.md  ->  artifacts/chunks.jsonl

WHY THIS PHASE MATTERS MORE THAN IT LOOKS
    Chunking is the highest-leverage and least-glamorous decision in RAG.
    A retriever can only return what you put in the index. If a chunk splits a
    table away from its header, no embedding model and no reranker on earth
    will recover the fact that "25 kg" was a payload capacity. Every later
    phase is downstream of this file.

    Concretely, the tradeoff is:
      * Chunks too LARGE  -> good recall (the answer is in there somewhere),
        bad precision (you burn context tokens on irrelevant text, and the
        chunk's embedding is an average of several topics, so it matches
        everything weakly and nothing strongly).
      * Chunks too SMALL  -> good precision, bad recall (the answer gets split
        across two chunks and you retrieve only one; or the chunk loses the
        context that made it interpretable).

    "Right" is empirical. Phase 6 is where you get to measure it.

WHAT TO IMPLEMENT
    Three stubs, in this order:
      1. split_by_heading()      <- the interesting one
      2. merge_small_sections()
      3. make_chunk_id()

RUN IT
    uv run python -m src.ingest              # build chunks.jsonl + histogram
    uv run python -m src.ingest --self-check # 4 assertions on the output

THIS PHASE NEEDS NO SERVER. You can do all of it offline.
=============================================================================
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pprint
from copy import copy

from src.config import ROOT, Config, load_config

# =============================================================================
# GIVEN: data model
# =============================================================================


@dataclass
class Section:
    """A contiguous slice of one Markdown document, bounded by headings.

    IMPORTANT CONVENTION (your split_by_heading must honor it):
      `lines` INCLUDES the heading line itself. So a section that starts at
      "## Charging" has that string as lines[0]. Keeping the heading in the body
      makes the text-preservation assertion in the self-check meaningful — every
      line of the source file must survive into exactly one section — and it
      means the heading text is also searchable by BM25 in Phase 2.

      `heading_path` is the *derived* breadcrumb, e.g.
          ["NB-1 \"Courier\" — Product Specifications", "Charging"]
      built from the enclosing `#`, then `##`, then `###` headings.
    """

    doc: str  # e.g. "product-nb1.md"
    heading_path: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip()

    @property
    def token_count(self) -> int:
        return count_tokens(self.body)


@dataclass
class Chunk:
    """One indexable unit. This is the row format of artifacts/chunks.jsonl."""

    chunk_id: str  # stable id, e.g. "product-nb1.md::003"
    doc: str  # source filename, used for doc-level eval in Phase 6
    source: str  # repo-relative path, matches golden_set.json expected_sources
    heading_path: list[str]
    text: str  # what actually gets embedded / BM25-indexed
    body: str  # raw section text WITHOUT the breadcrumb prefix
    token_count: int


# =============================================================================
# GIVEN: token counting
# =============================================================================


def count_tokens(text: str) -> int:
    """Approximate token count: whitespace words x 1.3.

    This is a PROXY, not a real tokenizer. A real count would come from the
    model's own tokenizer (`tiktoken` for OpenAI models, the GGUF's vocab for
    llama.cpp). We use the proxy because:

      1. We only need *consistency* to compare chunk sizes to each other and to
         a budget, not calibrated accuracy.
      2. Pulling in a tokenizer means pulling in a model-specific dependency for
         a decision (chunk boundaries) that shouldn't depend on the generator.

    The 1.3 multiplier comes from the rough English rule of thumb that a token
    is ~0.75 words. It will UNDER-count on this corpus, which is dense with
    numbers, units and IDs ("$18,500", "−10 °C", "E204") that tokenize into
    several tokens each. Being off by 30% is fine for budgeting; being off by
    30% would not be fine if you were packing a context window to the brim.
    """
    return int(len(text.split()) * 1.3)


# =============================================================================
# GIVEN: corpus loading
# =============================================================================


def read_corpus(cfg: Config) -> list[tuple[str, str]]:
    """Return [(filename, text), ...] for every .md file in the corpus, sorted.

    Sorted so that chunk ids are deterministic across machines and runs.
    """
    corpus_dir = ROOT / cfg.chunking.corpus_dir
    files = sorted(corpus_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found in {corpus_dir}")
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


# =============================================================================
# TODO(Phase 1) #1 — split_by_heading
# =============================================================================


def split_by_heading(doc_name: str, md_text: str, heading_levels: list[int]) -> list[Section]:
    """Split one Markdown document into Sections at heading boundaries.

    ---------------------------------------------------------------------------
    THE CONCEPT: "contextual chunk headers"
    ---------------------------------------------------------------------------
    Look at what a raw chunk from product-nb1.md looks like if you split on
    "## General specifications" and keep only the body:

        | Payload capacity | 25 kg |
        | Top speed | 24 km/h |
        | Range (single charge) | 80 km |

    Now embed that. The vector encodes "a table about capacity and speed" — with
    NO trace of the words "NB-1" or "Courier" anywhere in it. A user asking
    "what is the NB-1 payload capacity?" gets a mediocre match, and the nearly
    identical table in product-nb2.md is just as good a match. You have made the
    two documents that matter most on this corpus indistinguishable.

    The fix is to carry the heading breadcrumb into the chunk text:

        NB-1 "Courier" — Product Specifications > General specifications
        | Payload capacity | 25 kg |
        ...

    That single line is the highest-leverage thing in this entire phase. Build
    the breadcrumb here; `build_chunks()` (already written) does the prepending.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      1. Iterate over md_text.splitlines().

      2. Track whether you are inside a fenced code block. A fence is a line
         whose stripped form starts with ``` or ~~~; each one TOGGLES the state.
         While inside a fence, a line starting with '#' is a Python comment or a
         shell prompt, NOT a heading. Splitting there corrupts the code block.
         (This corpus has no fences today — implement it anyway. The moment
         someone adds a curl example to api-reference.md, an unguarded splitter
         silently starts producing garbage.)

      3. Outside a fence, detect a heading with a regex like ^(#{1,6})\\s+(.*)$.
         `len(group(1))` is the level, `group(2).strip()` is the title.

      4. If the level is in `heading_levels`, START A NEW SECTION. Otherwise
         treat the line as ordinary body text.

      5. Maintain the breadcrumb as a stack. On a heading of level L:
             truncate the stack to L-1 entries, then append this title.
         So `##` under `#` yields a 2-element path; a second `##` REPLACES the
         first one rather than nesting under it.

      6. Remember the convention: the heading line goes into the new section's
         `lines`, not thrown away.

      7. Handle the preamble — text before the first heading (this corpus has
         none, since every file opens with `#`, but don't crash on it). Give it
         an empty heading_path.

      8. Drop sections whose body is empty after stripping.

    ---------------------------------------------------------------------------
    WHAT *NOT* TO DO
    ---------------------------------------------------------------------------
      Do not split inside a Markdown table. You get this for free by splitting
      only on headings (a table row never starts with '#'), which is exactly why
      heading-splitting beats fixed-size character windows on this corpus. The
      self-check asserts it, so if you later add a character-window splitter,
      the assert will catch you.

    Returns: list[Section] in document order.
    """
    # TODO(Phase 1): implement per the recipe above.
    
    # print(md_text.splitlines())
    
    inside_fence = False
    breadcrumb = [] 
    sections: list[Section] = []
    
    for line in md_text.splitlines():
        if line.startswith("```") or line.startswith("~~~"):
            inside_fence = not inside_fence
            continue
        
        # 3. Outside a fence, detect a heading with a regex like ^(#{1,6})\\s+(.*)$.
        #          `len(group(1))` is the level, `group(2).strip()` is the title.
        
        if not inside_fence: #outside fence
            level = -1
            title = line
            
            if re.match(r"^(#{1,6})\s+(.*)$", line):    
                groups = line.split(" ", maxsplit=1)
                level = len(groups[0])
                title = groups[1].strip()
                # print("found a heading!", level, title)
            
            # 4. If the level is in `heading_levels`, START A NEW SECTION. Otherwise
            #     treat the line as ordinary body text.

            if heading_levels.count(level):
                # start section
                # level won't be -1
                
                L_MINUS_1 = level - 1
                while len(breadcrumb) > L_MINUS_1:
                    breadcrumb.pop()
                
                breadcrumb.append(title)
                
                section = Section(
                    doc=doc_name,
                    heading_path=breadcrumb.copy(),
                )
                sections.append(section)
                sections[-1].lines.append(line)
            else:
                # treat as normal
                # level will be -1
                # title is the line itself
                sections[-1].lines.append(title)
          

        # 5. Maintain the breadcrumb as a stack. On a heading of level L:
        #         truncate the stack to L-1 entries, then append this title.
        #     So `##` under `#` yields a 2-element path; a second `##` REPLACES the
        #     first one rather than nesting under it.

        # 6. Remember the convention: the heading line goes into the new section's
        #     `lines`, not thrown away.

        # 7. Handle the preamble — text before the first heading (this corpus has
        #     none, since every file opens with `#`, but don't crash on it). Give it
        #     an empty heading_path.

    # 8. Drop sections whose body is empty after stripping.
    
    for section in sections:
        if (len(section.lines)) == 1:
            sections.remove(section)    
    # pprint(sections)
    return sections
    # raise NotImplementedError("Phase 1: split_by_heading")


# =============================================================================
# TODO(Phase 1) #2 — merge_small_sections
# =============================================================================


def merge_small_sections(
    sections: list[Section], min_tokens: int, max_tokens: int
) -> list[Section]:
    """Greedily merge undersized adjacent sections.

    ---------------------------------------------------------------------------
    WHY
    ---------------------------------------------------------------------------
    Several sections in this corpus are tiny — "## Charging" plus a two-row
    table is about 25 tokens. A chunk that short is a bad index entry for two
    reasons:

      * Dense side: an embedding of 25 tokens is dominated by whatever few words
        are present. It matches erratically — sometimes ranking #1 for an
        unrelated query, sometimes missing its own topic.
      * Sparse side: BM25 divides by document length, so very short documents
        get an inflated score for any term they happen to contain. Tiny chunks
        become spurious top hits.

    Merging trades a little precision for a lot of stability.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      Walk the list front to back, keeping a `current` section:
        * If current.token_count >= min_tokens, emit it and move on.
        * Otherwise try to absorb the NEXT section into it, but only if
          combined tokens <= max_tokens.
        * Repeat until current is big enough or the next merge would overflow.
        * Never merge across documents. (You are called per-document, so this
          is automatic — but if you refactor, remember it. A chunk containing
          the tail of pricing.md and the head of safety-protocols.md would
          corrupt doc-level source attribution in Phase 6.)

      When merging A into B:
        * lines: A.lines + B.lines
        * heading_path: keep A's (the merged chunk lives under the earlier
          heading). Alternatively keep the common prefix of both — either is
          defensible; pick one and write down why in your README.

      Edge case: the LAST section may still be under min_tokens with nothing to
      merge into. Emit it anyway rather than dropping content. The self-check
      allows exactly this case.

    Returns: list[Section], length <= len(sections).
    """
    # TODO(Phase 1): implement per the recipe above.
    
    if len(sections) < 1:
        return sections
    
    def getSectionTokens(section: Section):
        lines = section.lines
        text = " ".join(section.lines)
        tokens = count_tokens(text)
        return tokens

                    # r,l
    # [S1 S1.5 S2 S3 S4]
        # [10 2  3  4  6]
        # min = 5
        # max = 10
    
    mergedSections = []
    
    #  * If current.token_count >= min_tokens, emit it and move on.
    #     #  * Otherwise try to absorb the NEXT section into it, but only if
    #     #    combined tokens <= max_tokens.
    
    # currentCount = getSectionTokens(rSections[l]) # 6
    # min = 5, max = 11
    # [6, 4, 3, 10, 2]
    # [6, 7, 10, 2]
    #     l ,r
    # l 6
    # l 4
    # l 7
    # l 9
    # l -> 0 -> 10 
    
    def mergeTwoSections(a, b):
        temp: Section = copy(a)
        temp.lines.extend(copy(b).lines)
        return temp
    
    l = 0
    r = 1
    limit = len(sections)
    
    currentSection: Section = copy(sections[l])
    
    # while (r < limit):
    #     if (getSectionTokens(currentSection) >= min_tokens):
    #         mergedSections.append(currentSection)
    #         l = r
    #         r += 1
    #         currentSection = copy(sections[l])
    #     elif getSectionTokens(currentSection) + getSectionTokens(sections[r]) > max_tokens:
    #         mergedSections.append(currentSection)
    #         currentSection = copy(sections[r])
    #         l = r
    #         r += 1
    #     else:
    #         currentSection = mergeTwoSections(currentSection, sections[r])
    #         r += 1
    # mergedSections.append(currentSection)
    # # pprint(sections)
    # pprint(mergedSections)
    # return mergedSections
    if len(sections) < 1:
        return sections

    def fresh(section: Section) -> Section:
        s = copy(section)
        s.lines = list(section.lines)   # break aliasing with the input
        return s

    counts = [count_tokens(" ".join(s.lines)) for s in sections]

    merged_sections: list[Section] = []
    current = fresh(sections[0])
    current_tokens = counts[0]

    r = 1
    while r < len(sections):            # was: r < limit  <- dropped last section
        if current_tokens >= min_tokens:
            merged_sections.append(current)
            current = fresh(sections[r])
            current_tokens = counts[r]
        elif current_tokens + counts[r] > max_tokens:
            merged_sections.append(current)
            current = fresh(sections[r])
            current_tokens = counts[r]
        else:
            current.lines.extend(sections[r].lines)
            current_tokens += counts[r]
        r += 1

    merged_sections.append(current)     # last section may be under min; emit anyway
    # pprint(merged_sections)
    return merged_sections           

    # raise NotImplementedError("Phase 1: merge_small_sections")


# =============================================================================
# TODO(Phase 1) #3 — make_chunk_id
# =============================================================================


def make_chunk_id(doc_name: str, ordinal: int) -> str:
    """Build a stable identifier for a chunk, e.g. "product-nb1.md::003".

    ---------------------------------------------------------------------------
    WHY "STABLE" IS A REQUIREMENT, NOT A NICETY
    ---------------------------------------------------------------------------
    Chunk ids are the join key between three things that get rebuilt at
    different times: the dense index, the BM25 index, and (from Phase 3 on) the
    fused results you inspect and label by hand.

    If you used a hash of the chunk TEXT, then fixing a typo in the corpus would
    renumber that chunk and silently invalidate any note you wrote about it.
    If you used a plain global counter, inserting a document at the top would
    renumber EVERY chunk in the corpus.

    Scoping the ordinal to the document, and zero-padding it, means an edit to
    pricing.md can only ever disturb pricing.md's ids.

    ---------------------------------------------------------------------------
    WHAT TO IMPLEMENT
    ---------------------------------------------------------------------------
      Return f"{doc_name}::{ordinal:03d}". That's it — one line.
      (It's a stub because the *reasoning* above is the lesson, not the code.)
    """
    # TODO(Phase 1): implement.
    return f"{doc_name}::{ordinal:03d}"
    # raise NotImplementedError("Phase 1: make_chunk_id")


# =============================================================================
# GIVEN: assembly, I/O, reporting
# =============================================================================


def build_chunks(cfg: Config) -> list[Chunk]:
    """Full pipeline: corpus -> sections -> merged sections -> Chunks."""
    chunks: list[Chunk] = []
    for doc_name, text in read_corpus(cfg):
        sections = split_by_heading(doc_name, text, cfg.chunking.heading_levels)
        sections = merge_small_sections(
            sections, cfg.chunking.min_tokens, cfg.chunking.max_tokens
        )
        for ordinal, sec in enumerate(sections):
            breadcrumb = " > ".join(sec.heading_path)
            if cfg.chunking.prepend_heading_path and breadcrumb:
                # The contextual chunk header. See split_by_heading's docstring.
                text_for_index = f"{breadcrumb}\n{sec.body}"
            else:
                text_for_index = sec.body
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_name, ordinal),
                    doc=doc_name,
                    # Matches the `expected_sources` format in eval/golden_set.json
                    # ("corpus/product-nb1.md"). Phase 6 depends on this lining up.
                    source=f"{cfg.chunking.corpus_dir}/{doc_name}",
                    heading_path=sec.heading_path,
                    text=text_for_index,
                    body=sec.body,
                    token_count=count_tokens(text_for_index),
                )
            )
    return chunks


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write JSONL — one JSON object per line.

    JSONL rather than a single JSON array because it streams: you can `head`,
    `wc -l`, and `grep` it, and Phase 2 can read it line by line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks.jsonl back. Used by Phase 2 and 3 — do not change the schema
    without rebuilding the indexes."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.ingest` first (Phase 1)."
        )
    with path.open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]


def print_histogram(chunks: list[Chunk], cfg: Config) -> None:
    """ASCII histogram of chunk token counts.

    LOOK AT THIS OUTPUT. It is the fastest way to spot a broken splitter:
      * one fat bar at the far right   -> you are producing one chunk per file
      * a pile at the far left         -> merging isn't working
      * a long tail past max_tokens    -> a section needs sub-splitting
    """
    counts = [c.token_count for c in chunks]
    lo, hi = min(counts), max(counts)
    nbins = 10
    width = max(1, (hi - lo) // nbins + 1)
    bins: dict[int, int] = {}
    for n in counts:
        bins[(n - lo) // width] = bins.get((n - lo) // width, 0) + 1

    print(f"\nchunks: {len(chunks)}   tokens: min={lo} max={hi} "
          f"mean={sum(counts) / len(counts):.0f} total={sum(counts)}")
    print(f"target range from config: {cfg.chunking.min_tokens}-{cfg.chunking.max_tokens}\n")
    for b in range(nbins):
        start, end = lo + b * width, lo + (b + 1) * width - 1
        n = bins.get(b, 0)
        print(f"  {start:4d}-{end:4d} | {'#' * n}{'' if n else ''} {n}")

    per_doc: dict[str, int] = {}
    for c in chunks:
        per_doc[c.doc] = per_doc.get(c.doc, 0) + 1
    print("\nchunks per document:")
    for doc, n in sorted(per_doc.items()):
        print(f"  {n:3d}  {doc}")


# =============================================================================
# GIVEN: self-check (this project's substitute for a test suite)
# =============================================================================

_TABLE_LINE = re.compile(r"^\s*\|")


def self_check(cfg: Config) -> None:
    """Four assertions on artifacts/chunks.jsonl.

    Read these before you implement — they ARE the spec for the stubs above.
    """
    chunks = load_chunks(cfg.path("chunks"))
    assert chunks, "no chunks were produced"

    # --- 1. Chunk ids are unique -------------------------------------------
    ids = [c.chunk_id for c in chunks]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"duplicate chunk_ids: {sorted(dupes)}"
    print(f"  [1/4] ok  {len(ids)} unique chunk ids")

    # --- 2. Size bounds -----------------------------------------------------
    # Undersized chunks are allowed ONLY as the last chunk of a document (there
    # was nothing left to merge into). Anything else means merging is broken.
    last_of_doc = {c.doc: c.chunk_id for c in chunks}  # dict keeps last write
    for c in chunks:
        assert c.token_count <= cfg.chunking.max_tokens, (
            f"{c.chunk_id} is {c.token_count} tokens, over max_tokens="
            f"{cfg.chunking.max_tokens}. Needs sub-splitting."
        )
        if c.token_count < cfg.chunking.min_tokens:
            assert last_of_doc[c.doc] == c.chunk_id, (
                f"{c.chunk_id} is {c.token_count} tokens (under min_tokens="
                f"{cfg.chunking.min_tokens}) but is not the last chunk of "
                f"{c.doc} — merge_small_sections should have absorbed it."
            )
    print(f"  [2/4] ok  all chunks within {cfg.chunking.min_tokens}-"
          f"{cfg.chunking.max_tokens} tokens (last-of-doc exempt)")

    # --- 3. Text preservation ----------------------------------------------
    # Every non-blank line of every source file must appear in some chunk body.
    # This catches the classic bug where a splitter eats heading lines, or drops
    # the tail of a document after the final heading.
    for doc_name, text in read_corpus(cfg):
        bodies = "\n".join(c.body for c in chunks if c.doc == doc_name)
        for line in text.splitlines():
            if line.strip():
                assert line.strip() in bodies, (
                    f"line lost from {doc_name}: {line.strip()[:60]!r}"
                )
    print("  [3/4] ok  every source line survives into a chunk")

    # --- 4. Table integrity -------------------------------------------------
    # A contiguous run of '|' lines is a Markdown table. It must live entirely
    # inside ONE chunk — a header row separated from its data rows is unusable.
    for doc_name, text in read_corpus(cfg):
        doc_chunks = [c for c in chunks if c.doc == doc_name]
        block: list[str] = []
        blocks: list[list[str]] = []
        for line in text.splitlines():
            if _TABLE_LINE.match(line):
                block.append(line.strip())
            elif block:
                blocks.append(block)
                block = []
        if block:
            blocks.append(block)
        for blk in blocks:
            assert any(all(row in c.body for row in blk) for c in doc_chunks), (
                f"table split across chunks in {doc_name}: "
                f"{blk[0][:50]!r} ... {blk[-1][:50]!r}"
            )
    print("  [4/4] ok  no Markdown table is split across chunks")

    print("\nPHASE 1 SELF-CHECK PASSED")


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    cfg = load_config()

    try:
        if "--self-check" in sys.argv:
            self_check(cfg)
            return 0
        chunks = build_chunks(cfg)
    except FileNotFoundError as exc:
        print(f"Missing prerequisite: {exc}")
        return 1
    except NotImplementedError as exc:
        print(f"Not implemented yet: {exc}")
        print("\nPhase 1 has three stubs to fill in, in this order:")
        print("  1. split_by_heading()      src/ingest.py")
        print("  2. merge_small_sections()  src/ingest.py")
        print("  3. make_chunk_id()         src/ingest.py")
        print("\nEach docstring contains the full recipe. The spec is enforced by")
        print("  uv run python -m src.ingest --self-check")
        return 1

    out = cfg.path("chunks")
    write_chunks(chunks, out)
    print(f"wrote {len(chunks)} chunks -> {out.relative_to(ROOT)}")
    print_histogram(chunks, cfg)

    print("\n--- sample chunks " + "-" * 55)
    for c in chunks[:3]:
        print(f"\n[{c.chunk_id}]  {c.token_count} tokens  path={c.heading_path}")
        preview = c.text if len(c.text) < 400 else c.text[:400] + " ..."
        print("  " + preview.replace("\n", "\n  "))

    print("\nNext: uv run python -m src.ingest --self-check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
