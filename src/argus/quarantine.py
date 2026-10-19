"""Cut-not-flag (v4 spec §2.8 B1, WP-C): the sentence splitter, the locator
and the whole-sentence cutter behind report_builder's cuts.

A markdown body is read as blocks: paragraphs and list items (with their
continuation lines) hold the cuttable sentences; headings, code fences,
tables, block quotes and rules are never cut. ``locate(text, md)`` finds a
finding's verbatim ``sentence`` by normalised containment, else by the best
``difflib`` ratio ≥ 0.85 over a window of 1–3 adjacent sentences (the probe's
condensed f6 needs the window: 0.94 over two sentences, 0.68 over either
alone). ``cut(md, indices)`` removes whole sentences, drops a paragraph or
item left empty and a sub-heading left without content; an untouched body is
returned byte-identical. ``apply_cuts`` turns the panel's flags and the
unknown-id rule into finding statuses (kept / cut / unlocated), the cut
bodies and the two appendix listings.

The v3 string quarantine is gone with report_builder's legacy branch: it cut
by substring, left orphan fragments (baseline b3r) and cut the TL;DR
(baseline b1). Pure functions, no I/O.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# = graph.assemble.CITE_RE; not imported: the graph package imports this module.
_CITE = re.compile(r"\[(\d{1,3})\]")
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s")
_ITEM = re.compile(r"^(\s*(?:[-*+•]|\d+[.)])\s+)")
_FIXED_LINE = re.compile(r"^\s*[|>]|^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$")      # tables, quotes, rules
_END = re.compile(r"[.!?]+[\"'”’)\]]*(?:\s*\[\d{1,3}\])*")                    # terminator, closers, trailing [n]
_OPENERS = "\"'“‘([*_`"
# ponytail: an English abbreviation list and single-letter initials guard the "."
# split ("plan B. Then" stays one sentence); a language-aware splitter if reports
# in other languages appear.
_ABBREV = frozenset("e.g i.e vs al fig figs no nos dr mr mrs ms prof st jr sr approx cf eq u.s u.k".split())
_WINDOW = 3
_MIN_RATIO = 0.85


@dataclass
class _Block:
    kind: str                       # "text" (paragraph / list item) | "heading" | "fixed" | "blank"
    lines: list[str]
    prefix: str = ""                # a list item's marker
    level: int = 0                  # a heading's level
    sentences: list[str] = field(default_factory=list)


def _abbreviation(before: str) -> bool:
    word = re.search(r"([A-Za-z][A-Za-z.]*)$", before)
    return word is not None and (len(word.group(1)) == 1 or word.group(1).lower() in _ABBREV)


def _split(text: str) -> list[str]:
    """Sentences of one block's text: a terminator (with closing quotes or
    brackets and trailing ``[n]`` markers) followed by whitespace and a capital,
    a digit, a quote or markup."""
    out, start = [], 0
    for m in _END.finditer(text):
        ws = re.match(r"\s+", text[m.end():])
        if not ws or m.end() + ws.end() >= len(text):
            continue
        nxt = text[m.end() + ws.end()]
        if not (nxt.isupper() or nxt.isdigit() or nxt in _OPENERS):
            continue
        if m.group(0) == "." and _abbreviation(text[:m.start()]):
            continue
        out.append(text[start:m.end()].strip())
        start = m.end() + ws.end()
    tail = text[start:].strip()
    return [*out, tail] if tail else out


def _blocks(md: str) -> list[_Block]:
    blocks: list[_Block] = []
    in_fence = False
    for line in md.split("\n"):
        if in_fence:
            blocks[-1].lines.append(line)
            in_fence = not _FENCE.match(line)
        elif _FENCE.match(line):
            blocks.append(_Block("fixed", [line]))
            in_fence = True
        elif not line.strip():
            blocks.append(_Block("blank", [line]))
        elif m := _HEADING.match(line):
            blocks.append(_Block("heading", [line], level=len(m.group(1))))
        elif _FIXED_LINE.match(line):
            blocks.append(_Block("fixed", [line]))
        elif m := _ITEM.match(line):
            blocks.append(_Block("text", [line], prefix=m.group(1)))
        elif blocks and blocks[-1].kind == "text":
            blocks[-1].lines.append(line)           # a paragraph's next line, a list item's continuation
        else:
            blocks.append(_Block("text", [line]))
    for b in blocks:
        if b.kind == "text":
            b.sentences = _split("\n".join([b.lines[0][len(b.prefix):], *b.lines[1:]]).strip())
    return blocks


def sentences(md: str) -> list[str]:
    """The cuttable sentences of ``md``, in order (list items without their
    marker); headings, code, tables and quotes hold none."""
    return [s for b in _blocks(md) for s in b.sentences]


def _norm(text: str) -> str:
    text = _CITE.sub(" ", text.lower())
    text = re.sub(r"[`*_>#]", " ", text)
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def locate(text: str, md: str) -> list[int] | None:
    """Indices into ``sentences(md)`` of the whole sentences that state
    ``text``: normalised containment (spanning sentences if it must), else
    the best window of 1–3 adjacent sentences at a ``difflib`` ratio ≥ 0.85
    (the smaller window on a tie); None when neither finds it."""
    needle = _norm(text)
    if not needle:
        return None
    sents = [_norm(s) for s in sentences(md)]
    hay, starts = "", []
    for s in sents:
        starts.append(len(hay) + 1)
        hay += " " + s
    at = (hay + " ").find(f" {needle} ")
    if at >= 0:
        lo, hi = at + 1, at + 1 + len(needle)
        return [i for i, (st, s) in enumerate(zip(starts, sents)) if st < hi and st + len(s) > lo]
    # ponytail: every window of every flagged finding through difflib — O(sentences × flags);
    # an index of candidate windows if reports grow far past a few hundred sentences.
    best, where = 0.0, None
    for size in range(1, _WINDOW + 1):
        for i in range(len(sents) - size + 1):
            ratio = SequenceMatcher(None, needle, " ".join(sents[i:i + size]), autojunk=False).ratio()
            if ratio > best:
                best, where = ratio, list(range(i, i + size))
    return where if best >= _MIN_RATIO else None


def cut(md: str, indices: Iterable[int]) -> str:
    """``md`` without the sentences at ``indices`` (into ``sentences(md)``):
    a touched paragraph or item is rebuilt from its kept sentences, one left
    empty is dropped, and so is a heading left with nothing under it. Blank
    runs outside code collapse to one. No index → ``md`` unchanged."""
    drop = set(indices)
    if not drop:
        return md
    blocks = _blocks(md)
    removed: set[int] = set()
    k = 0
    for bi, b in enumerate(blocks):
        if b.kind != "text":
            continue
        gone = {j for j in range(len(b.sentences)) if k + j in drop}
        k += len(b.sentences)
        if not gone:
            continue
        kept = [s for j, s in enumerate(b.sentences) if j not in gone]
        if kept:
            b.lines = [b.prefix + " ".join(kept)]
        else:
            removed.add(bi)
    for bi in reversed(range(len(blocks))):                 # bottom-up: sub-headings before their parents
        if blocks[bi].kind != "heading":
            continue
        end = next((j for j in range(bi + 1, len(blocks))
                    if blocks[j].kind == "heading" and blocks[j].level <= blocks[bi].level), len(blocks))
        span = range(bi + 1, end)
        if any(j in removed for j in span) and not any(j not in removed and blocks[j].kind != "blank"
                                                       for j in span):
            removed.add(bi)
    out: list[str] = []
    blank = True                                            # also drops leading blank lines
    for bi, b in enumerate(blocks):
        if bi in removed or (b.kind == "blank" and blank):
            continue
        blank = b.kind == "blank"
        out.extend(b.lines)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _strip_unknown(text: str, n_sources: int) -> str:
    return re.sub(r"\s*\[(\d{1,3})\]", lambda m: m.group(0) if 0 < int(m.group(1)) <= n_sources else "", text)


def apply_cuts(findings: list[dict], mds: Mapping[str, str], flagged: Mapping[str, str], n_sources: int
               ) -> tuple[list[dict], dict[str, str], list[dict], list[dict]]:
    """The cuts of one report. ``mds``: section title → markdown ("Analysis"
    included); ``flagged``: the panel's flagged finding id → reason. A finding
    whose ``source_ids`` all fall outside 1..``n_sources`` is cut too, reason
    "unknown source id" (no judge needed). Each cut finding is located by its
    ``sentence`` (its claim for a legacy finding without one): status ``cut``
    when found, else ``unlocated`` (still in the body). Returns (findings with
    their status, the cut bodies, the cut rows, the unlocated rows); a row is
    ``{section, text, reason}`` with unknown ``[n]`` markers stripped, and one
    cut span is one row whatever the number of findings behind it."""
    out: list[dict] = []
    plan: dict[str, set[int]] = {}
    spans: dict[tuple[str, tuple[int, ...]], list[str]] = {}
    unlocated: list[dict] = []
    for finding in findings:
        f = {**finding, "status": "kept"}
        out.append(f)
        ids = [int(i) for i in f.get("source_ids") or []]
        if ids and all(not 0 < i <= n_sources for i in ids):
            reason = f"unknown source id (cites {', '.join(map(str, ids))})"
        elif f.get("id") in flagged:
            reason = flagged[f["id"]]
        else:
            continue
        section = f.get("section") or ""
        md = mds.get(section)
        idx = locate(f.get("sentence") or f.get("claim") or "", md) if md else None
        if idx is None:
            f["status"] = "unlocated"
            unlocated.append({"section": section, "reason": reason,
                              "text": _strip_unknown(f.get("claim") or f.get("sentence") or "", n_sources)})
            continue
        f["status"] = "cut"
        plan.setdefault(section, set()).update(idx)
        reasons = spans.setdefault((section, tuple(idx)), [])
        if reason not in reasons:
            reasons.append(reason)
    rows = []
    for (section, idx), reasons in spans.items():
        texts = sentences(mds[section])
        rows.append({"section": section, "text": _strip_unknown(" ".join(texts[i] for i in idx), n_sources),
                     "reason": "; ".join(reasons)})
    return out, {t: cut(md, plan.get(t, ())) for t, md in mds.items()}, rows, unlocated


def render_cut_appendix(rows: list[dict]) -> str:
    """The "## ⚠ Flagged claims (removed from the body)" appendix ("" without rows)."""
    if not rows:
        return ""
    return "\n\n".join([
        "## ⚠ Flagged claims (removed from the body)",
        "_These sentences were cut from the report: the grounding judge found them unsupported by the "
        "sources they cite, or they cite no known source._",
        "\n".join(f"- **{r['section']}:** {' '.join(r['text'].split())} — {r['reason']}" for r in rows)])


def render_unlocated_appendix(rows: list[dict]) -> str:
    """The "## ⚠ Flagged but not located (still in body)" appendix ("" without rows)."""
    if not rows:
        return ""
    return "\n\n".join([
        "## ⚠ Flagged but not located (still in body)",
        "_The grounding judge flagged these findings, but their sentences could not be located in the "
        "report, so whatever states them is still in the body._",
        "\n".join(f"- **{r['section']}:** {' '.join(r['text'].split())} — {r['reason']}" for r in rows)])


__all__ = ["sentences", "locate", "cut", "apply_cuts", "render_cut_appendix", "render_unlocated_appendix"]
