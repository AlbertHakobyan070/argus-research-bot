"""Cut-not-flag (v4 spec §2.8 B1, WP-C): the sentence splitter, the locator
and the whole-sentence cutter behind report_builder's cuts.

A markdown body is read as blocks: paragraphs and list items (with their
continuation lines) hold the cuttable sentences; headings, code fences,
tables, block quotes and rules are never cut. ``locate(text, md)`` finds a
finding's verbatim ``sentence`` as whole normalised sentences (every
occurrence), else by normalised containment, else by the best ``difflib``
ratio ≥ 0.85 over a window of 1–3 adjacent sentences (the probe's condensed f6
needs the window: 0.94 over two sentences, 0.68 over either alone);
``contains`` is the containment alone (the TL;DR's rule). ``cut(md, indices)``
removes whole sentences, drops a paragraph or item left empty and a
sub-heading left without content; an untouched body is returned
byte-identical. ``apply_cuts`` turns the panel's flags, the unknown-id rule
and the stricter verdict for twins (``twins``, spec R1: no negation between
them and the same numeric values, then compose's dedupe ratio on
``claim_norm``, or ≥ 2 values and a shared source) into finding
statuses (kept / cut / unlocated) — a finding stated in a cut sentence goes
with it unless the cut body still states it — the cut bodies and the two
appendix listings (judge reasons with their square brackets escaped);
``strip_unknown_ids`` drops an unknown id cited beside known ones, in the
sentence's own block.

The v3 string quarantine is gone with report_builder's legacy branch: it cut
by substring, left orphan fragments (baseline b3r) and cut the TL;DR
(baseline b1). Pure functions, no I/O.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from difflib import SequenceMatcher

# = graph.assemble.CITE_RE; not imported: the graph package imports this module.
_CITE = re.compile(r"\[(\d{1,3})\]")
# = graph.research._QUOTE_FOLD and _NEGATORS: quote verification's fold (quote marks and "*" out, dashes
# unified, "_" a space) and its negator tokens (the fold deletes apostrophes: don't → dont); not imported, as _CITE.
_FOLD = str.maketrans({**dict.fromkeys("\"'`“”„‟‘’‚‛«»‹›*"), **dict.fromkeys("–—‐‑−", "-"), "_": " "})
_NEGATORS = frozenset("not never cannot cant dont doesnt isnt arent wasnt werent wont wouldnt shouldnt "
                      "couldnt didnt hasnt havent hadnt aint mustnt neednt mightnt shant oughtnt without"
                      .split())
# a numeric value (spec R1): digits not attached to a name ("GPT-4", "H2", "v0.2" are names), "." and ","
# only between digits; its unit never counts ("30%" = "30 percent", "70B" = "70 billion", "120 ms" = "120ms")
_VALUE = re.compile(r"(?<![\w.])(?<!\w-)\d+(?:[.,]\d+)*")
_THOUSANDS = re.compile(r",(?=\d{3}(?!\d))")          # a comma before exactly three digits
_DECIMAL = re.compile(r"\d+(?:\.\d+)?")
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
_TWIN_RATIO = 0.85             # = compose._DUP_RATIO: the dedupe rule's near-duplicate


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


def _contained(needle: str, sents: list[str]) -> list[int] | None:
    """The sentences the first occurrence of ``needle`` spans in the joined ``sents``."""
    hay, starts = "", []
    for s in sents:
        starts.append(len(hay) + 1)
        hay += " " + s
    at = (hay + " ").find(f" {needle} ")
    if at < 0:
        return None
    lo, hi = at + 1, at + 1 + len(needle)
    return [i for i, (st, s) in enumerate(zip(starts, sents)) if st < hi and st + len(s) > lo]


def locate(text: str, md: str) -> list[int] | None:
    """Indices into ``sentences(md)`` of the whole sentences that state
    ``text``: every run of whole sentences equal to ``text``'s once
    normalised, else normalised containment (its first occurrence, spanning
    sentences if it must), else the best window of 1–3 adjacent sentences at a
    ``difflib`` ratio ≥ 0.85 (the smaller window on a tie); None when none
    finds it."""
    needle = _norm(text)
    if not needle:
        return None
    sents = [_norm(s) for s in sentences(md)]
    parts = [p for p in (_norm(s) for s in sentences(text)) if p]
    exact = [j for i in range(len(sents) - len(parts) + 1) if sents[i:i + len(parts)] == parts
             for j in range(i, i + len(parts))]
    if exact:
        return sorted(set(exact))
    where = _contained(needle, sents)
    if where is not None:
        return where
    # ponytail: every window of every flagged finding through difflib — O(sentences × flags);
    # an index of candidate windows if reports grow far past a few hundred sentences.
    best = 0.0
    for size in range(1, _WINDOW + 1):
        for i in range(len(sents) - size + 1):
            ratio = SequenceMatcher(None, needle, " ".join(sents[i:i + size]), autojunk=False).ratio()
            if ratio > best:
                best, where = ratio, list(range(i, i + size))
    return where if best >= _MIN_RATIO else None


def contains(text: str, md: str) -> bool:
    """``text`` is in ``md`` by normalised containment — the locator without
    its fuzzy window (the TL;DR's rule, spec R2)."""
    needle = _norm(text)
    return bool(needle) and _contained(needle, [_norm(s) for s in sentences(md)]) is not None


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


def strip_unknown_ids(md: str, n_sources: int) -> tuple[str, list[int]]:
    """``md`` with every unknown ``[n]`` (outside 1..``n_sources``) stripped
    from each sentence that also cites a known id (spec B1), and the ids
    dropped; a sentence citing unknown ids only is left as it is, and so is
    every table, heading or code line (they hold no sentence)."""
    dropped: list[int] = []
    blocks = []
    for b in _blocks(md):
        text, at = "\n".join(b.lines), 0
        for s in b.sentences:                       # a sentence is a substring of its own block, in order
            at = text.index(s, at)
            ids = [int(i) for i in _CITE.findall(s)]
            unknown = [i for i in ids if not 0 < i <= n_sources]
            if unknown and len(unknown) < len(ids):
                new = _strip_unknown(s, n_sources)
                text, s = text[:at] + new + text[at + len(s):], new
                dropped += [i for i in unknown if i not in dropped]
            at += len(s)
        blocks.append(text)
    return "\n".join(blocks), dropped


def claim_norm(text: str) -> str:
    """A claim as compose's dedupe pass and the twin rule compare it (one
    normalisation; = bench.metrics._norm): NFKC, casefold, ``[n]`` markers out,
    word tokens."""
    text = _CITE.sub(" ", unicodedata.normalize("NFKC", text).casefold())
    return " ".join(re.findall(r"\w+", text))


def _values(text: str) -> set[Decimal | str]:
    """The numeric values of ``text`` (spec R1), ``[n]`` markers aside:
    thousands separators out, compared by value ("$1,200" = "$1200", "2.50x"
    = "2.5x"); a run that is no decimal ("3.10.2", "2,5") compares as written."""
    text = _CITE.sub(" ", unicodedata.normalize("NFKC", text)).translate(_FOLD)
    runs = (_THOUSANDS.sub("", v) for v in _VALUE.findall(text))
    return {Decimal(v) if _DECIMAL.fullmatch(v) else v for v in runs}


def _negated(text: str) -> bool:
    """``text`` holds one of quote verification's negator tokens."""
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).translate(_FOLD).casefold())
    return not _NEGATORS.isdisjoint(words)


def twins(a: str, b: str, a_ids: Iterable[int], b_ids: Iterable[int]) -> bool:
    """``a`` and ``b`` state one claim (spec R1): neither negates what the
    other asserts (exactly one holding a negator token: never twins) and
    either (i) their ``claim_norm`` ratio is ≥ 0.85 (the dedupe rule) with the
    same numeric values, or (ii) they carry the same set of ≥ 2 distinct
    numeric values and share a cited source (``a_ids`` / ``b_ids``). A
    different value makes a different claim."""
    values = _values(a)
    if _negated(a) != _negated(b) or values != _values(b):
        return False
    x, y = claim_norm(a), claim_norm(b)
    if x and y and SequenceMatcher(None, x, y, autojunk=False).ratio() >= _TWIN_RATIO:
        return True
    return len(values) >= 2 and not {int(i) for i in a_ids}.isdisjoint(int(i) for i in b_ids)


def _twin(f: dict, cut_findings: list[dict]) -> str | None:
    """The id of a cut finding whose claim this one's ``twins`` (by their ``source_ids``)."""
    return next((str(c.get("id")) for c in cut_findings
                 if twins(f.get("claim") or "", c.get("claim") or "", f.get("source_ids") or (),
                          c.get("source_ids") or ())), None)


def apply_cuts(findings: list[dict], mds: Mapping[str, str], flagged: Mapping[str, str], n_sources: int
               ) -> tuple[list[dict], dict[str, str], list[dict], list[dict]]:
    """The cuts of one report. ``mds``: section title → markdown ("Analysis"
    included); ``flagged``: the panel's flagged finding id → reason. A finding
    whose ``source_ids`` all fall outside 1..``n_sources`` is cut too, reason
    "unknown source id" (no judge needed); so is a kept finding whose claim
    ``twins`` one of those cut ones — the stricter verdict, reason "near-duplicate
    of cut f<n>" (spec R1). Each cut finding is located by its ``sentence`` (its claim
    for a legacy finding without one): status ``cut`` when found, else
    ``unlocated`` (still in the body); then a kept finding whose sentence lies
    inside a cut span and no longer occurs in the cut body is cut with it,
    reason "stated in a cut sentence (f<n>)" (on that span's row).
    Returns (findings with their status, the cut bodies, the cut rows, the
    unlocated rows); a row is ``{section, text, reason}`` with unknown ``[n]``
    markers stripped, one row per section and text whatever the number of
    findings behind it."""
    out = [{**f, "status": "kept"} for f in findings]
    reasons: dict[int, str] = {}
    for k, f in enumerate(out):
        ids = [int(i) for i in f.get("source_ids") or []]
        if ids and all(not 0 < i <= n_sources for i in ids):
            reasons[k] = f"unknown source id (cites {', '.join(map(str, ids))})"
        elif f.get("id") in flagged:
            reasons[k] = flagged[f["id"]]
    cut_findings = [out[k] for k in reasons]
    # ponytail: every kept claim against every cut one through difflib, O(findings × cuts) over one
    # report (tens to a few hundred findings); an index if reports grow far past that.
    for k, f in enumerate(out):
        if k not in reasons and (twin := _twin(f, cut_findings)):
            reasons[k] = f"near-duplicate of cut {twin}"

    def where(f: dict) -> list[int] | None:
        md = mds.get(f.get("section") or "")
        return locate(f.get("sentence") or f.get("claim") or "", md) if md else None

    plan: dict[str, set[int]] = {}
    spans: dict[tuple[str, str], list[str]] = {}            # (section, row text) → reasons
    owner: dict[tuple[str, int], tuple[str, tuple[str, str]]] = {}    # a cut sentence → (finding id, its row)
    unlocated: list[dict] = []
    for k, reason in sorted(reasons.items()):
        f, section = out[k], out[k].get("section") or ""
        idx = where(f)
        if idx is None:
            f["status"] = "unlocated"
            unlocated.append({"section": section, "reason": reason,
                              "text": _strip_unknown(f.get("claim") or f.get("sentence") or "", n_sources)})
            continue
        f["status"] = "cut"
        plan.setdefault(section, set()).update(idx)
        first = [idx[0]]                        # its first run of sentences: one occurrence's text
        for i in idx[1:]:
            if i != first[-1] + 1:
                break
            first.append(i)
        texts = sentences(mds[section])
        key = (section, _strip_unknown(" ".join(texts[i] for i in first), n_sources))
        rows = spans.setdefault(key, [])
        if reason not in rows:
            rows.append(reason)
        for i in idx:
            owner.setdefault((section, i), (str(f.get("id")), key))
    bodies = {t: cut(md, plan.get(t, ())) for t, md in mds.items()}
    for k, f in enumerate(out):
        section = f.get("section") or ""
        if k in reasons or not plan.get(section):
            continue
        idx = where(f)
        if idx and set(idx) <= plan[section] and not contains(f.get("sentence") or f.get("claim") or "",
                                                              bodies[section]):
            fid, key = owner[(section, idx[0])]
            f["status"] = "cut"
            reason = f"stated in a cut sentence ({fid})"
            if reason not in spans[key]:
                spans[key].append(reason)
    rows = [{"section": s, "text": text, "reason": "; ".join(r)} for (s, text), r in spans.items()]
    return out, bodies, rows, unlocated


def _reason(text: str) -> str:
    """A judge's reason as report text: square brackets escaped (spec B1) —
    a reason quoting a printed list "[42, 42]" is no citation."""
    return text.replace("[", r"\[").replace("]", r"\]")


def render_cut_appendix(rows: list[dict]) -> str:
    """The "## ⚠ Flagged claims (removed from the body)" appendix ("" without rows)."""
    if not rows:
        return ""
    return "\n\n".join([
        "## ⚠ Flagged claims (removed from the body)",
        "_These sentences were cut from the report: the grounding judge found them unsupported by the "
        "sources they cite, or they cite no known source._",
        "\n".join(f"- **{r['section']}:** {' '.join(r['text'].split())} — {_reason(r['reason'])}" for r in rows)])


def render_unlocated_appendix(rows: list[dict]) -> str:
    """The "## ⚠ Flagged but not located (still in body)" appendix ("" without rows)."""
    if not rows:
        return ""
    return "\n\n".join([
        "## ⚠ Flagged but not located (still in body)",
        "_The grounding judge flagged these findings, but their sentences could not be located in the "
        "report, so whatever states them is still in the body._",
        "\n".join(f"- **{r['section']}:** {' '.join(r['text'].split())} — {_reason(r['reason'])}" for r in rows)])


__all__ = ["sentences", "locate", "contains", "cut", "apply_cuts", "strip_unknown_ids", "claim_norm", "twins",
           "render_cut_appendix", "render_unlocated_appendix"]
