"""P3 — post-pass that prunes the ``## Sources`` block of a research report.

Why this exists (bug observed 2026-07-09): when the planner LLM falls back
to a cheap model — or when the researcher subgraph returns thin/mixed
evidence — the ``draft_md`` ``## Sources`` block contains entries for URLs
that were never fetched. ``citations.verify_citations`` correctly strips
the *URL* but leaves the ``[N]`` slot label behind, so the final report
ends up with:

    ## Sources
    [1] https://thetechbriefs.com/...   <- content farm
    [2]                                <- EMPTY, dangling
    [3] https://glm45.org/?
    [7]                                <- EMPTY, dangling

Findings that referenced ``[2]`` or ``[7]`` are left pointing at nothing.
This module sits between :func:`citations.verify_citations` and the
markdown write. It is pure (no I/O, no LLM) so it is unit-testable.

Algorithm
---------
1. Locate the ``## Sources`` (or ``## References`` for lecture mode)
   heading — the *last* top-level ``##`` heading wins.
2. Parse every numbered source line. Each line is expected to look like
   ``[N] <text-or-url>``. Lines that have no URL/text at all (after
   trim) are *dangling*.
3. Build a remap of original ``[N]`` index → new compact index ``M``
   (only for kept sources). Dangling slots are dropped.
4. Drop dangling source lines; renumber the kept sources 1..M.
5. Renumber **every** ``[N]`` reference in the body of the report to the
   new compact index. We use a word-boundary-anchored regex so URLs and
   numbers in prose are not touched.

This module deliberately does **not** try to detect ``[n]`` inside
JSON/code/section heading IDs. The synthesizer writes prose, and we
treat its output as prose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# --- Regexes ---------------------------------------------------------------

# Match a single numbered source line at the start of a line:
#   [1] https://...
#   [2] https://arxiv.org/...
#   [3] <display text>
#   [2]                       <- dangling (no content after the label)
_SOURCE_LINE_RE = re.compile(
    r"""^[ \t]*\[(\d+)\][ \t]*(.*?)$""",
    re.MULTILINE,
)

# Match a citation ref like [3] in prose. Word-boundary by digits only,
# not by chars — so "...extracted [3] tokens..." still triggers but
# "...23..." does not. We allow optional leading whitespace so the
# pattern can be substituted while preserving indentation.
_CITATION_REF_RE = re.compile(r"\[(\d+)\]")

# Match a top-level `## Sources` or `## References` heading line. End of
# line — the block runs from this line to the next `## ` heading or EOF.
# We capture the heading line itself so we can rewrite the label.
_SOURCES_HEADING_RE = re.compile(
    r"^[ \t]*##[ \t]+(?:Sources|References)\s*$",
    re.MULTILINE,
)
_NEXT_HEADING_RE = re.compile(r"^[ \t]*##[ \t]+\S", re.MULTILINE)


# --- Result type -----------------------------------------------------------

@dataclass
class SourcesSanitizationResult:
    """Outcome of :func:`sanitize_sources_block`."""

    cleaned_text: str
    dropped_count: int = 0
    renumbered: dict[int, int] = field(default_factory=dict)
    # remap[old_N] = new_M. Sources that had no body (dangling) are absent
    # from this dict — they were dropped.
    sources_block_found: bool = False


# --- Helpers ---------------------------------------------------------------

def _find_sources_block(text: str) -> tuple[int, int, str] | None:
    """Return (block_start, block_end, heading_text) for the Sources block,
    or ``None`` if no such block exists.

    The block is defined as everything from the matched ``## Sources``
    heading line to the start of the NEXT ``##`` heading (or EOF,
    whichever comes first). When multiple ``## Sources`` headings exist
    (rare; e.g. an appendix mentioning sources), we pick the LAST one — the
    primary reference list is always at the bottom of the report.
    """
    matches = list(_SOURCES_HEADING_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    start = last.start()
    # Search for next ## heading AFTER this one.
    after = text[last.end():]
    next_match = _NEXT_HEADING_RE.search(after)
    if next_match:
        end = last.end() + next_match.start()
    else:
        end = len(text)
    return start, end, last.group(0)


def _is_dangling_line(body: str) -> bool:
    """A source line is 'dangling' if its label is followed by nothing
    that looks like a URL or display text (i.e. after trim, empty).
    """
    return not body.strip()


def _build_remap(kept: list[tuple[int, str]]) -> dict[int, int]:
    """Build the old-index → new-index remap for kept source lines.

    Renumbering is compact: ``[1], [3], [5]`` becomes ``[1], [2], [3]``.
    A source that already lives at the right index keeps its index (so
    the diff is minimal when nothing was removed).
    """
    return {old: new for new, (old, _body) in enumerate(kept, start=1)}


def _renumber_body(body: str, remap: dict[int, int],
                    skipped: set[int]) -> str:
    """Rewrite ``[N]`` references in the body of the report using remap.

    Source indices in ``skipped`` (the dangling slots) are stripped
    entirely — leaving them as ``[2]`` would re-introduce the dangling
    ref bug we just fixed. Other refs not in ``remap`` are left intact
    (defensive — would only happen if the synthesizer cited an index that
    isn't in the Sources block at all, which is a different bug).
    """
    def _sub(m: re.Match) -> str:
        n = int(m.group(1))
        if n in skipped:
            return ""  # drop the dangling ref
        if n in remap:
            return f"[{remap[n]}]"
        return m.group(0)

    return _CITATION_REF_RE.sub(_sub, body)


# --- Public entry point ----------------------------------------------------

def sanitize_sources_block(report_text: str) -> SourcesSanitizationResult:
    """Drop dangling source lines and renumber the rest.

    See module docstring for the algorithm. Pure function — no I/O.
    Safe to call even when the report has no ``## Sources`` block (no-op
    in that case). Defensive against malformed input: lines that do not
    match the source-line shape are left untouched.
    """
    block = _find_sources_block(report_text)
    if block is None:
        return SourcesSanitizationResult(cleaned_text=report_text)

    block_start, block_end, heading_text = block
    before = report_text[:block_start]
    block_text = report_text[block_start:block_end]
    after = report_text[block_end:]

    # Walk the block line-by-line. Keep only numbered source lines;
    # preserve any prose / blank lines around them so the block stays
    # well-formed.
    kept_sources: list[tuple[int, str]] = []
    skipped: set[int] = set()
    dropped = 0
    new_block_lines: list[str] = []
    # We track per-line: original label + body content for renumbering.
    for line in block_text.splitlines(keepends=True):
        m = _SOURCE_LINE_RE.match(line.rstrip("\r\n"))
        if not m:
            new_block_lines.append(line)
            continue
        old_n = int(m.group(1))
        body = m.group(2)
        if _is_dangling_line(body):
            skipped.add(old_n)
            dropped += 1
            # Drop the line entirely (no replacement).
            continue
        kept_sources.append((old_n, body))
        new_block_lines.append(line)  # rewritten later when we know the remap

    if dropped == 0:
        # Nothing to do — keep the report as-is. Avoids spurious
        # renumbering churn on clean reports.
        return SourcesSanitizationResult(cleaned_text=report_text)

    remap = _build_remap(kept_sources)

    # Rewrite each kept source line: same body, new index label.
    rewritten: list[str] = []
    kept_iter = iter(kept_sources)
    for line in new_block_lines:
        m = _SOURCE_LINE_RE.match(line.rstrip("\r\n"))
        if not m:
            rewritten.append(line)
            continue
        old_n, body = next(kept_iter)
        new_n = remap[old_n]
        # Preserve trailing newline.
        nl = "\n" if line.endswith("\n") else ""
        rewritten.append(f"[{new_n}] {body}{nl}")
    new_block_text = "".join(rewritten)

    # Renumber refs in the body (everything outside the Sources block).
    # Note: renumbering the before-segment and after-segment
    # independently is equivalent to renumbering `before + after` as a
    # single string and avoids fragile slice arithmetic on equal-prefix
    # substrings.
    renumbered_before = _renumber_body(before, remap, skipped)
    renumbered_after = _renumber_body(after, remap, skipped)

    cleaned = renumbered_before + new_block_text + renumbered_after

    return SourcesSanitizationResult(
        cleaned_text=cleaned,
        dropped_count=dropped,
        renumbered=remap,
        sources_block_found=True,
    )


__all__ = [
    "sanitize_sources_block",
    "SourcesSanitizationResult",
]
