"""Report assembly (v4 spec §2.8): TL;DR → sections → Analysis → appendices → Sources.

The helpers moved here unchanged from compose.py. ``assemble_draft``
reproduces compose's sectioned assembly (lecture passes
``tldr_heading="Executive TL;DR"``), but not its flat path (``tldr``/``short``,
or a brief with one sub-question): ``flat=True`` adds ``## Findings``, and the
flat writer emits its own ``## TL;DR`` and ``## Key Findings``. So the flat
path must not move onto ``assemble_draft`` before WP-C removes the flat
writer's own headings, or those reports get ``## Findings`` above the model's
headings. WP-C implements ``normalize_cite_markers`` (identity until then).
"""
from __future__ import annotations

import re

CITE_RE = re.compile(r"\[(\d{1,3})\]")


def strip_fences(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^```(?:markdown|md)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    return t


_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s*")
_EMPHASIS_MARKER_RE = re.compile(r"^\*{1,2}|\*{1,2}$")


def _normalize_heading_text(s: str) -> str:
    """Normalize a heading candidate for duplicate comparison: strip
    markdown heading markers and bold/italic markers, then casefold."""
    s = _HEADING_PREFIX_RE.sub("", s.strip())
    s = _EMPHASIS_MARKER_RE.sub("", s.strip())
    return s.strip().rstrip(":.-").casefold()


def strip_duplicate_heading(md: str, title: str) -> str:
    """Drop a leading line that just repeats the section title.

    Section writers are told not to repeat the heading (the assembler
    renders it separately), but weaker proxy-routed models often
    disregard that instruction and open the section with
    ``## <title>`` or ``**<title>**`` anyway — producing a visibly
    duplicated heading in the rendered report. This is a structural
    guard rather than relying on prompt compliance: it removes at most
    ONE leading line, and only when it is a near-exact echo of the
    title, so a section that legitimately opens with a sentence
    sharing a few words with the title is left untouched.
    """
    if not title:
        return md
    lines = md.splitlines()
    if not lines:
        return md
    target = _normalize_heading_text(title)
    if not target or _normalize_heading_text(lines[0]) != target:
        return md
    rest = lines[1:]
    while rest and not rest[0].strip():
        rest.pop(0)
    return "\n".join(rest)


def build_sources_block(body: str, fetched: list[dict]) -> str:
    """## Sources listing every source id cited in the body, id order."""
    cited = sorted({int(m) for m in CITE_RE.findall(body)})
    by_id: dict[int, dict] = {i + 1: f for i, f in enumerate(fetched)}
    lines = ["", "## Sources", ""]
    for i in cited:
        f = by_id.get(i)
        if not f:
            continue
        title = (f.get("title") or "").strip() or f.get("url", "")
        lines.append(f"{i}. [{title[:90]}]({f.get('url')})")
    if len(lines) == 3:
        return ""
    return "\n".join(lines) + "\n"


def normalize_cite_markers(md: str) -> str:
    """Identity until WP-C (``【n】``, ``［n］``, ``〔n〕``, ``【n†…】``, ``[n, m]`` → ``[n][m]``)."""
    return md


def assemble_draft(*, tldr: str | None, sections: list[dict], analysis_md: str | None,
                   appendices: list[str] | None = None, fetched: list[dict], flat: bool,
                   tldr_heading: str = "TL;DR") -> str:
    """``## <tldr_heading>`` → sections (flat: ``## Findings`` + body) →
    ``## Analysis`` → appendices → ``## Sources``.

    Each appendix is a markdown block with its own heading. Blank ``tldr``,
    ``analysis_md`` and appendices are omitted; Sources lists the ids cited
    anywhere above it, appendices included.
    """
    if flat:
        body = "## Findings\n\n" + "\n\n".join(s["md"] for s in sections)
    else:
        body = "\n\n".join(f"## {s['title']}\n\n{s['md']}" for s in sections)
    if tldr and tldr.strip():
        body = f"## {tldr_heading}\n\n{tldr.strip()}\n\n{body}"
    if analysis_md and analysis_md.strip():
        body = f"{body}\n\n## Analysis\n\n{analysis_md.strip()}"
    for appendix in appendices or []:
        if appendix.strip():
            body = f"{body}\n\n{appendix.strip()}"
    return body + "\n" + build_sources_block(body, fetched)
