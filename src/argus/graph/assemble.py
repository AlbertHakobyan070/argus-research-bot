"""Report assembly (v4 spec §2.8): TL;DR → sections → Analysis → appendices → Sources.

``assemble_draft`` lays the report out (lecture passes
``tldr_heading="Executive TL;DR"``); the flat lengths (``tldr``/``short``, or a
brief with one sub-question) put their single body under ``## Findings`` — the
flat writer emits no headings of its own (WP-C). The TL;DR is never written by
an LLM: report_builder builds it after the cuts and passes it in.
``normalize_cite_markers`` rewrites the citation forms models emit (CJK
brackets, lists, ranges, footnotes, ``[Source n]``) to ``[n][m]…`` and never
touches code, links or URLs; compose applies it to every writer reply before
extraction and assembly. ``ANALYSIS_SECTION`` names the analysis: its heading
and its findings' section.
"""
from __future__ import annotations

import re

CITE_RE = re.compile(r"\[(\d{1,3})\]")
ANALYSIS_SECTION = "Analysis"


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


_N = r"\d{1,3}(?!\d)"
_ITEM = _N + r"(?:\s*[-–—]\s*" + _N + r")?"                                  # 3, or a range 1-3
_SEP = r"\s*[,，;；、]\s*"
_GROUP = _ITEM + r"(?:" + _SEP + _ITEM + r")*"
_CJK = re.compile(r"[【［〔]\s*(" + _GROUP + r")[^】］〕\n]*[】］〕]")       # 【4】 ［3］ 〔5〕 【2†L1-L4】 【1-3】
_FOOTNOTE = re.compile(r"\[\^(" + _N + r")\]")                             # [^2]
_SOURCE = re.compile(r"\[\s*sources?\s*:?\s*(" + _GROUP + r")\s*\]", re.I)  # [Source 3] [Source: 3] [Sources 3, 4]
_BRACKET = re.compile(r"\[\s*(" + _ITEM + _SEP + _GROUP + r"|" + _N + r"\s*[-–—]\s*" + _N + r")\s*\]")
_RANGE_MAX = 20     # a wider "range" is not a citation (a span of years, a version range)
# Never prose: a code span (any backtick run), a link or image destination, a bare URL,
# a footnote definition line.
_KEEP = re.compile(r"(?<!`)(`+)(?!`)(?:(?!\n[ \t]*\n).)+?(?<!`)\1(?!`)|\]\([^)\n]*\)|https?://\S+"
                   r"|^ {0,3}\[\^\d{1,3}\]:[^\n]*", re.S | re.M)
_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")
_LIST_ITEM = re.compile(r"\s*(?:[-*+]|\d{1,9}[.)])(?:\s|$)")


def _ids(m: re.Match) -> str:
    """The match's ids as ``[n][m]…``; the match as written when an id is 0 or
    a range is reversed or wider than ``_RANGE_MAX``."""
    ids: list[int] = []
    for item in re.split(_SEP, m.group(1)):
        ends = [int(x) for x in re.split(r"\s*[-–—]\s*", item)]
        lo, hi = ends[0], ends[-1]
        if not 0 < lo <= hi < lo + _RANGE_MAX:
            return m.group(0)
        ids.extend(range(lo, hi + 1))
    return "".join(f"[{i}]" for i in ids)


def _normalize_prose(text: str) -> str:
    out, last = [], 0
    for m in _KEEP.finditer(text):
        out += [_normalize_markers(text[last:m.start()]), m.group(0)]
        last = m.end()
    return "".join(out) + _normalize_markers(text[last:])


def _normalize_markers(text: str) -> str:
    for pattern in (_FOOTNOTE, _SOURCE, _CJK, _BRACKET):
        text = pattern.sub(_ids, text)
    return text


def _code_lines(lines: list[str]) -> list[bool]:
    """Which lines are code: a fence (``` or ~~~, to its closing fence or the
    end), or an indented block after a blank line that is not a list's
    continuation (CommonMark, simplified)."""
    code = [False] * len(lines)
    fence: tuple[str, int] | None = None
    indented = False
    before = ""                                 # the last non-blank prose line
    for k, line in enumerate(lines):
        bare = line.rstrip("\r\n")
        if fence:
            code[k] = True
            m = _FENCE.fullmatch(bare)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= fence[1] and not m.group(2).strip():
                fence, before = None, bare
            continue
        m = _FENCE.fullmatch(bare)
        if m and not (m.group(1)[0] == "`" and "`" in m.group(2)):
            fence, code[k] = (m.group(1)[0], len(m.group(1))), True
            continue
        if not bare.strip():
            continue
        if re.match(r"(?: {4}|\t)", bare) and (indented or (k == 0 or not lines[k - 1].strip())
                                                and not _LIST_ITEM.match(before) and not before[:1].isspace()):
            code[k] = indented = True
            continue
        indented, before = False, bare
    return code


def normalize_cite_markers(md: str) -> str:
    """``【n】``, ``［n］``, ``〔n〕``, ``【n†…】``, ``[n, m]`` / ``[n; m]``, ranges
    ``[1-3]`` / ``[1–3]`` (at most 20 wide, in a list or CJK brackets too),
    ``[^n]`` and ``[Source n]`` / ``[Source: n]`` → ``[n][m]…`` (spec §2.8).
    Never touched: code (fences, an unterminated one included, indented
    blocks, spans), link and image destinations and their bracketed text, bare
    URLs, footnote definitions, and an id list holding a 0; ``[n]`` already is
    the form."""
    lines = md.splitlines(keepends=True)
    out, prose = [], []
    for line, is_code in zip(lines, _code_lines(lines)):
        if is_code:
            out += [_normalize_prose("".join(prose)), line]
            prose = []
        else:
            prose.append(line)
    return "".join(out) + _normalize_prose("".join(prose))


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
        body = f"{body}\n\n## {ANALYSIS_SECTION}\n\n{analysis_md.strip()}"
    for appendix in appendices or []:
        if appendix.strip():
            body = f"{body}\n\n{appendix.strip()}"
    return body + "\n" + build_sources_block(body, fetched)
