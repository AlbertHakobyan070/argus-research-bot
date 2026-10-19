"""Outline node (v4 spec §3 WP-I, §2.8): the report's sections and the
evidence notes allocated to each (``OutlineSection.note_ids``).

Fresh mode (no outline yet): one ``outline.plan`` call gets the brief, the
archetype's section shape as a hint inside the length's section budget
("length wins"; a lecture keeps its Part I–IV names and applies the shape
inside the parts), the evidence strength per sub-question and the index of
the usable notes, and returns sections with ``note_ids``. Extend mode (an
outline exists and usable notes arrived that no section holds — after a gap
wave or a user Extend): the existing sections keep their titles, the new notes
are assigned to them, and at most one new section may open, only for a
sub-question whose coverage was below ``coverage_min_strong`` before the wave.
``outline["changed_sections"]`` names the sections whose note set changed
(every section in fresh mode).

The deterministic repair follows either mode and the degradation: unknown
note ids dropped, a note in at most 2 sections, every usable note allocated
(to the section sharing most sub-questions with it; ties rotate), and a
section with fewer than ``section_min_notes`` notes merged into its neighbour
— the previous section, else the next — never dropped (R11). ``outline.plan``
degrades to ``default_outline`` (fresh) or to the existing sections (extend),
allocated the same way. Flat lengths and one-sub-question briefs need no
outline. compose.py re-exports ``outline_node`` (the graph imports it there).
"""
from __future__ import annotations

import logging
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from .. import llm
from ..engines.urls import domain
from . import run_config
from .archetypes import TEMPLATES
from .jsonx import parse_json_obj
from .research import _round, compute_coverage
from .state import OutlineSection, ResearchBrief

logger = logging.getLogger("argus.outline")

# Per-mode structural budgets (sections beyond TL;DR/Sources).
_MODE_SECTIONS = {"tldr": 0, "short": 0, "medium": 4, "long": 6,
                  "lecture": 7}
_MAX_SECTIONS_PER_NOTE = 2
_INDEX_CHARS = 9000        # note index in a prompt; notes past it are allocated by the repair

OUTLINE_SYSTEM = """You plan the section structure of a research report and
allocate the evidence notes to its sections.

You get the research brief (sub-questions + success criteria), the
evidence strength per sub-question, a suggested section shape and an
index of the evidence notes. Produce a section outline:

- Sections must collectively cover the sub-questions that HAVE evidence.
- Each section gets: a title, a one-line focus, the sub-question indices
  it covers, and the ids of the notes its writer should use.
- Allocate every note; a note goes in ONE section, at most two when it
  truly serves both.
- Do NOT create sections for TL;DR or Sources — those are added
  automatically.
- Prefer thematic titles over restating sub-questions.

Return ONLY JSON:
{"sections": [{"title": "...", "focus": "...", "sub_qs": [0, 2], "note_ids": [1, 4, 7]}, ...]}
"""

LECTURE_HINT = """
This is a LECTURE-format report: name the sections as parts, in order:
"Part I — <Background theme>", "Part II — <Current state theme>" (Part II
may be split into two sections), "Part III — <Open problems theme>",
"Part IV — Practice and exercises".
"""

EXTEND_SYSTEM = """You extend the section outline of a research report with
newly gathered evidence notes.

The existing sections keep their titles. Assign each new note to the
existing section it supports best (two sections at most). Only when new
notes fit no existing section and cover a sub-question marked STARVED may
you open ONE new section for them.

Return ONLY JSON:
{"assign": [{"note_id": 12, "section": "<existing section title>"}, ...],
 "new_section": {"title": "...", "focus": "...", "sub_qs": [3], "note_ids": [12, 15]} or null}
"""


def _evidence_summary_lines(brief: ResearchBrief,
                            evidence: list[dict]) -> str:
    lines = []
    for i, sq in enumerate(brief.sub_questions):
        notes = [n for n in evidence if i in (n.get("sub_qs") or [])]
        strong = sum(1 for n in notes if int(n.get("relevance") or 0) >= 3)
        lines.append(f"({i}) {sq.q} — {len(notes)} notes, {strong} strong")
    return "\n".join(lines)


def default_outline(brief: ResearchBrief, length: str) -> list[OutlineSection]:
    """Deterministic outline: group sub-questions into N sections."""
    n_target = max(2, min(_MODE_SECTIONS.get(length, 4),
                          len(brief.sub_questions)))
    idxs = list(range(len(brief.sub_questions)))
    chunks: list[list[int]] = [[] for _ in range(n_target)]
    for pos, i in enumerate(idxs):
        chunks[pos % n_target].append(i)
    out = []
    for c, chunk in enumerate(chunks):
        if not chunk:
            continue
        title = brief.sub_questions[chunk[0]].q.rstrip("?")[:70]
        if length == "lecture":
            numeral = ["I", "II", "III", "IV", "V", "VI", "VII"][c]
            title = f"Part {numeral} — {title}"
        out.append(OutlineSection(title=title, focus="", sub_qs=chunk))
    return out


# ---------------------------------------------------------------------------
# Usable notes and the deterministic repair
# ---------------------------------------------------------------------------

def usable_notes(evidence: list[dict], fetched: list[dict], min_relevance: int) -> list[dict]:
    """The notes a section writer can use, in evidence order: a claim whose
    quote did not fail verification, and relevance ≥ ``min_relevance`` — an
    appended local file or a pinned source at any relevance."""
    def pinned(n: dict) -> bool:
        i = int(n.get("source_id") or 0)
        item = fetched[i - 1] if 0 < i <= len(fetched) else {}
        return str(n.get("source_url") or "").startswith("file:///") or bool(item.get("pinned"))
    return [n for n in evidence
            if any(c.get("quote_verified") is not False for c in n.get("claims") or [])
            and (int(n.get("relevance") or 0) >= min_relevance or pinned(n))]


def _ints(values, ok=lambda i: True) -> list[int]:
    out: list[int] = []
    for v in values if isinstance(values, list) else []:
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if ok(i) and i not in out:
            out.append(i)
    return out


def _keep_known(sections: list[OutlineSection], known: set[int]) -> None:
    """Drop unknown and repeated ids; a note keeps its first two sections."""
    count: Counter = Counter()
    for s in sections:
        ids = []
        for i in s.note_ids:
            if i in known and i not in ids and count[i] < _MAX_SECTIONS_PER_NOTE:
                ids.append(i)
                count[i] += 1
        s.note_ids = ids


def _allocate(sections: list[OutlineSection], notes: list[dict]) -> None:
    """Each note no section holds → the section sharing most sub-questions
    with it; ties rotate across the tied sections, in order."""
    held = {i for s in sections for i in s.note_ids}
    turn = 0
    for n in notes:
        sid = int(n["source_id"])
        if sid in held:
            continue
        subs = set(n.get("sub_qs") or [])
        overlap = [len(subs & set(s.sub_qs)) for s in sections]
        best = [s for s, o in zip(sections, overlap) if o == max(overlap)]
        best[turn % len(best)].note_ids.append(sid)
        turn += 1
        held.add(sid)


def _merge_into(dst: OutlineSection, src: OutlineSection) -> None:
    dst.sub_qs = list(dict.fromkeys([*dst.sub_qs, *src.sub_qs]))
    dst.note_ids = list(dict.fromkeys([*dst.note_ids, *src.note_ids]))
    dst.focus = "; ".join(f for f in (dst.focus, src.focus) if f)


def _merge_starved(sections: list[OutlineSection], min_notes: int) -> None:
    """A section with fewer than ``min_notes`` notes merges into the previous
    section (the next one, for the first): its sub-questions, notes and focus
    move, its title goes. A lone section stays as it is."""
    while len(sections) > 1:
        k = next((i for i, s in enumerate(sections) if len(s.note_ids) < min_notes), None)
        if k is None:
            return
        src = sections.pop(k)
        _merge_into(sections[k - 1] if k > 0 else sections[0], src)


def _repair(sections: list[OutlineSection], brief: ResearchBrief, notes: list[dict],
            min_notes: int) -> None:
    _keep_known(sections, {int(n["source_id"]) for n in notes})
    # Every sub-question is owned by some section (v3 rule): orphans join the last.
    owned = {i for s in sections for i in s.sub_qs}
    orphans = [i for i in range(len(brief.sub_questions)) if i not in owned]
    if orphans:
        sections[-1].sub_qs = [*sections[-1].sub_qs, *orphans]
    _allocate(sections, notes)
    _merge_starved(sections, min_notes)


# ---------------------------------------------------------------------------
# Prompts and replies
# ---------------------------------------------------------------------------

def _note_index(notes: list[dict]) -> str:
    """``[id] domain · relevance r · sub-questions k — first claim`` lines,
    up to ``_INDEX_CHARS``."""
    lines: list[str] = []
    total = 0
    for n in notes:
        claim = " ".join(str(((n.get("claims") or [{}])[0]).get("text") or "").split())[:100]
        tags = ", ".join(str(i) for i in n.get("sub_qs") or []) or "-"
        line = (f"[{n['source_id']}] {domain(n.get('source_url') or '') or 'local'} · relevance "
                f"{n.get('relevance')} · sub-questions {tags} — {claim}")
        if total + len(line) > _INDEX_CHARS:
            lines.append(f"(… {len(notes) - len(lines)} more notes — they are allocated automatically)")
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines) or "(no usable notes)"


def _shape_hint(archetype: str, length: str, budget: int) -> str:
    shape = "\n".join(f"- {title}: {focus}" for title, focus in TEMPLATES[archetype]["sectioned"])
    if length == "lecture":
        return (f"{LECTURE_HINT}\nInside the parts, cover this {archetype} shape "
                f"(at most {budget} sections in all):\n{shape}\n")
    return (f"\nSuggested shape for a {archetype} report — adapt it to the evidence, "
            f"at most {budget} sections:\n{shape}\n")


def _parse_sections(text: str, n_subs: int, budget: int) -> list[OutlineSection]:
    data = parse_json_obj(text, require_any=("sections",))
    raw = data.get("sections")
    if not isinstance(raw, list):
        raise ValueError(f"'sections' is {type(raw).__name__}, not a list")
    out: list[OutlineSection] = []
    for s in raw:
        if len(out) == budget:
            break
        if not isinstance(s, dict) or not str(s.get("title") or "").strip():
            continue
        out.append(OutlineSection(title=str(s["title"]).strip()[:100], focus=str(s.get("focus") or "")[:200],
                                  sub_qs=_ints(s.get("sub_qs"), lambda i: 0 <= i < n_subs),
                                  note_ids=_ints(s.get("note_ids"))))
    if not out:
        raise ValueError("no usable section in the reply")
    return out


def _parse_extend(text: str, titles: dict[str, str], new_ids: set[int], n_subs: int
                  ) -> tuple[list[tuple[int, str]], OutlineSection | None]:
    """``([(note id, existing title)], proposed new section | None)``: ids
    outside the new notes and titles outside the outline are left out."""
    data = parse_json_obj(text, require_any=("assign", "new_section"))
    assign = data.get("assign") or []
    if not isinstance(assign, list):
        raise ValueError(f"'assign' is {type(assign).__name__}, not a list")
    pairs = []
    for a in assign:
        if not isinstance(a, dict):
            continue
        ids = _ints([a.get("note_id")], lambda i: i in new_ids)
        title = titles.get(str(a.get("section") or "").strip().casefold())
        if ids and title:
            pairs.append((ids[0], title))
    ns = data.get("new_section")
    new = None
    if isinstance(ns, dict) and str(ns.get("title") or "").strip():
        new = OutlineSection(title=str(ns["title"]).strip()[:100], focus=str(ns.get("focus") or "")[:200],
                             sub_qs=_ints(ns.get("sub_qs"), lambda i: 0 <= i < n_subs),
                             note_ids=_ints(ns.get("note_ids"), lambda i: i in new_ids))
    return pairs, new


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

def _emit(runtime: Runtime | None, kind: str, data: dict) -> None:
    if runtime is not None:
        runtime.stream_writer({"kind": kind, "node": "outline", "data": data})


def _degrade_line(e: llm.EngineDegraded, rnd: int, runtime: Runtime | None) -> str:
    logger.warning("outline.plan degraded (%s)", e)
    _emit(runtime, "degrade", {"role": e.role, "degradation": e.degradation, "reason": str(e)})
    return f"[r{rnd}] degrade: outline.plan → {e.degradation} ({e})"


def _fresh(state, rc: run_config.RunConfig, brief: ResearchBrief, notes: list[dict], budget: int,
           runtime: Runtime | None) -> tuple[list[OutlineSection], list[dict], list[str]]:
    length = rc.length
    human = (
        f"Topic: {state.get('user_request')}\n"
        f"Mode: {length} (max {budget} sections)\n\n"
        f"Sub-questions and evidence strength:\n"
        f"{_evidence_summary_lines(brief, state.get('evidence') or [])}\n\n"
        f"Success criteria:\n"
        + "\n".join(f"- {c}" for c in brief.success_criteria)
        + _shape_hint(brief.archetype, length, budget)
        + f"\nEvidence notes ([id] domain · relevance · sub-questions — first claim):\n{_note_index(notes)}"
        + "\n\nReturn the outline JSON."
    )
    try:
        res = llm.complete("outline.plan", [SystemMessage(content=OUTLINE_SYSTEM), HumanMessage(content=human)],
                           engines=rc.engines, temperature=0.2, max_tokens=1500,
                           validate=lambda text: _parse_sections(text, len(brief.sub_questions), budget))
    except llm.EngineDegraded as e:
        return (default_outline(brief, length), [a.model_dump() for a in e.attempts],
                [_degrade_line(e, _round(state), runtime)])
    return res.parsed, [r.model_dump() for r in res.records], []


def _extend(state, rc: run_config.RunConfig, brief: ResearchBrief, sections: list[OutlineSection],
            new: list[dict], runtime: Runtime | None
            ) -> tuple[OutlineSection | None, list[dict], list[str]]:
    """Assign the new notes (in place) and return the accepted new section."""
    new_ids = {int(n["source_id"]) for n in new}
    before = compute_coverage(brief, [n for n in state.get("evidence") or [] if int(n["source_id"]) not in new_ids])
    floor = rc.research.coverage_min_strong
    starved = {int(k) for k, v in before.items() if v < floor}
    existing = "\n".join(f'- "{s.title}" (sub-questions {", ".join(map(str, s.sub_qs)) or "-"}; '
                         f'notes {", ".join(map(str, s.note_ids)) or "-"}): {s.focus}' for s in sections)
    subs = "\n".join(f"({k}) {sq.q} — {before[str(k)]} strong notes before this wave"
                     + (" — STARVED" if k in starved else "") for k, sq in enumerate(brief.sub_questions))
    human = (f"Topic: {state.get('user_request')}\n\nExisting sections:\n{existing}\n\n"
             f"Sub-questions:\n{subs}\n\n"
             f"New evidence notes ([id] domain · relevance · sub-questions — first claim):\n{_note_index(new)}"
             "\n\nReturn the assignment JSON.")
    titles = {s.title.casefold(): s.title for s in sections}
    try:
        res = llm.complete("outline.plan", [SystemMessage(content=EXTEND_SYSTEM), HumanMessage(content=human)],
                           engines=rc.engines, temperature=0.2, max_tokens=900,
                           validate=lambda text: _parse_extend(text, titles, new_ids, len(brief.sub_questions)))
    except llm.EngineDegraded as e:
        return None, [a.model_dump() for a in e.attempts], [_degrade_line(e, _round(state), runtime)]
    pairs, proposed = res.parsed
    by_title = {s.title: s for s in sections}
    for i, title in pairs:
        by_title[title].note_ids.append(i)
    accepted = (proposed if proposed is not None and proposed.note_ids
                and proposed.title.casefold() not in titles and set(proposed.sub_qs) & starved else None)
    return accepted, [r.model_dump() for r in res.records], []


def outline_node(state, runtime: Runtime | None = None) -> dict:
    """Plan the report sections and their notes (none for flat lengths), or
    extend the existing outline with notes that arrived since."""
    rc = run_config.require(state)
    brief = ResearchBrief.model_validate(state.get("brief") or {})
    budget = _MODE_SECTIONS[rc.length]
    if budget == 0 or len(brief.sub_questions) <= 1:
        return {"outline": {"sections": [], "changed_sections": []},
                "messages": [{"role": "assistant", "content": "🧭 Flat mode — no outline needed."}]}

    notes = usable_notes(state.get("evidence") or [], state.get("fetched") or [],
                         rc.research.writer_min_relevance)
    min_notes = rc.compose.section_min_notes
    existing = (state.get("outline") or {}).get("sections") or []
    if not existing:
        sections, calls, errors = _fresh(state, rc, brief, notes, budget, runtime)
        _repair(sections, brief, notes, min_notes)
        changed = [s.title for s in sections]
        message = f"🧭 Outline: {len(sections)} sections."
    else:
        sections = [OutlineSection.model_validate(s) for s in existing]
        held = {i for s in sections for i in s.note_ids}
        new = [n for n in notes if int(n["source_id"]) not in held]
        if not new:
            return {"outline": {"sections": existing, "changed_sections": []},
                    "messages": [{"role": "assistant", "content": "🧭 Outline unchanged — no new notes."}]}
        before = {s.title: set(s.note_ids) for s in sections}
        added, calls, errors = _extend(state, rc, brief, sections, new, runtime)
        if added is not None:
            sections.append(added)
        _keep_known(sections, {int(n["source_id"]) for n in notes})
        _allocate(sections, notes)
        if added is not None and len(added.note_ids) < min_notes:
            sections.remove(added)
            _merge_into(sections[-1], added)
        changed = [s.title for s in sections if set(s.note_ids) != before.get(s.title)]
        message = f"🧭 Outline extended: {len(new)} new notes; changed: {', '.join(changed) or 'none'}."
    return {
        "outline": {"sections": [s.model_dump() for s in sections], "changed_sections": changed},
        "model_calls": calls,
        "errors": errors,
        "messages": [{"role": "assistant", "content": message}],
    }


__all__ = ["outline_node", "default_outline", "usable_notes", "OUTLINE_SYSTEM", "EXTEND_SYSTEM",
           "LECTURE_HINT"]
