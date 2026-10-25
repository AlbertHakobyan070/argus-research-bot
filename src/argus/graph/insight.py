"""Analyst node (v4 spec §3 WP-I, §2.8 R12/R13): what the evidence means.

``analyst_node`` runs after compose (WP-C wires it into the graph): one
``compose.analyst`` call (plain markdown; degrade → ``omit_analysis``) reads
the per-section findings (claim, [ids], section), the deterministic
``stance_map``, the coverage and an index of the cited sources (title,
registrable domain, authority, published) — never the section bodies — and
writes the archetype's analysis headings (``archetypes``). Post-processing is
deterministic: footnote definitions are removed (a ``cite:`` line each, as
compose does), text before the first fixed heading is dropped (recorded), a
reply with none of them — or with nothing under them — is the
``omit_analysis`` degradation, takeaway bullets
without a citation move to the open questions as "(uncited)", the ``short``
variant is compacted (at most 4 takeaways and 3 open questions, no
Implications, the agree/disagree section only when the stance map shows a
conflict; what it trims is recorded), findings are extracted over the
analysis by compose's ``extract_findings`` (section "Analysis"; the labels are
not claims; a skipped extraction is a declared ``skip:`` line) and every takeaway
gets its ``confidence_label`` with the reasons. No TL;DR here: report_builder
builds it after the cuts (WP-C).
"""
from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from .. import llm
from ..engines.roles import DEGRADABLE_ROLES
from ..engines.urls import registrable_domain
from . import run_config
from .archetypes import AGREEMENT, IMPLICATIONS, OPEN_QUESTIONS, SCENARIOS, TAKEAWAYS, TEMPLATES
from .assemble import (ANALYSIS_SECTION, CITE_RE, assemble_draft, normalize_cite_markers, strip_fences,
                       strip_footnote_definitions)
from .compose import _footnote_lines, extract_findings
from .research import _round
from .state import ResearchBrief

logger = logging.getLogger("argus.insight")

ANALYSIS_LABEL = ("_Analysis — interpretation by Argus from the cited evidence; confidence labels are "
                  "computed from source independence, quote verification and source authority._")
_STANCES = ("supports", "contradicts", "mixed", "background")
_SHORT_TAKEAWAYS = 4
_SHORT_OPEN_QUESTIONS = 3
_FINDINGS_CHARS = 12000
# ponytail: a hand list of English function words for the R13 overlap test; a language-aware
# list when reports in other languages appear.
_STOP = frozenset("""a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each few for from further
had has have having he her here hers him his how i if in into is it its itself just me more most my no nor
not now of off on once only or other our ours out over own same she should so some such than that the
their them then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your""".split())

ANALYST_SYSTEM = """You write the ANALYSIS part of a grounded research report:
what the evidence means — not a summary of it.

You get the report's findings (claims with their [n] source ids, grouped
by section), a stance map per sub-question (how many sources support,
contradict, are mixed or background, and which sources contradict), the
evidence coverage per sub-question and an index of the cited sources
(title, domain, authority 0-3, publication date).

Rules:
1. Reason only from the findings and the stance map. Every factual
   statement carries the [n] ids of the findings it rests on. Introduce no
   number, date or name that is not in the findings.
2. Mark interpretation and speculation as such ("this suggests",
   "if … then …").
3. Write exactly the ### headings you are given, in that order, with
   nothing before the first one. No TL;DR, no Sources list, no JSON, no
   code fences.
4. Under "Key takeaways" write bullets ("- "), each one sentence that
   answers part of the question and ends with its citations, e.g.
   "- … [2][5]".
"""

_GUIDE = {
    TAKEAWAYS: "3-6 bullets, each ending with its [n] citations",
    AGREEMENT: "where the sources converge and where they conflict, citing both sides",
    SCENARIOS: ("base, upside and downside cases with their drivers, the leading indicators to "
                "watch, and what would change this view"),
    IMPLICATIONS: "what follows for the reader's decision or understanding",
    OPEN_QUESTIONS: "what the evidence cannot settle: thin coverage, conflicts, missing data",
}
_SHORT_GUIDE = {
    TAKEAWAYS: f"at most {_SHORT_TAKEAWAYS} bullets, each ending with its [n] citations",
    OPEN_QUESTIONS: f"at most {_SHORT_OPEN_QUESTIONS} bullets: what the evidence cannot settle",
}


# ---------------------------------------------------------------------------
# Stance map and the R13 confidence label
# ---------------------------------------------------------------------------

def stance_map(brief: ResearchBrief, evidence: list[dict]) -> dict:
    """Per sub-question ``str(k)``: how many notes support, contradict, are
    mixed or background, and the ``contradicting_sources`` (ids, ascending)."""
    out: dict[str, dict] = {}
    for k in range(len(brief.sub_questions)):
        row: dict = {s: 0 for s in _STANCES}
        against: set[int] = set()
        for n in evidence:
            if k not in (n.get("sub_qs") or []):
                continue
            stance = n.get("stance") if n.get("stance") in _STANCES else "background"
            row[stance] += 1
            if stance == "contradicts":
                against.add(int(n["source_id"]))
        row["contradicting_sources"] = sorted(against)
        out[str(k)] = row
    return out


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", CITE_RE.sub(" ", text or "").lower()) if w not in _STOP}


def confidence_label(source_ids, sources, notes, takeaway_text, stance_map) -> tuple[str, list[str]]:
    """The R13 label of a takeaway citing ``source_ids`` (1-based ids into
    ``sources``, the fetched list; ``notes`` are the evidence notes):
    D = distinct registrable domains among the cited sources of authority ≥ 1;
    V = a cited note of authority ≥ 1 holds a ``quote_verified`` claim sharing
    ≥ 2 non-stop words with the takeaway; A = the highest cited authority.
    high: D ≥ 2, V and A ≥ 2; medium: D ≥ 2 and (V or A ≥ 2), or A ≥ 2 and V;
    else low. A note contradicting on the cited notes' sub-questions
    (``stance_map``) caps the label at medium and adds "contradicted by [n]".
    An id without a note is not a source."""
    by_id = {int(n["source_id"]): n for n in notes}
    cited = [by_id[i] for i in dict.fromkeys(source_ids) if i in by_id and 0 < i <= len(sources)]
    if not cited:
        return "low", ["no known source cited"]
    words = _words(takeaway_text)
    credible = [n for n in cited if int(n.get("authority", 1)) >= 1]     # authority 0 counts for neither D nor V
    d = len({registrable_domain(sources[int(n["source_id"]) - 1].get("url") or "") for n in credible})
    v = any(c.get("quote_verified") is True and len(words & _words(c.get("text") or "")) >= 2
            for n in credible for c in n.get("claims") or [])
    a = max(int(n.get("authority", 1)) for n in cited)
    if d >= 2 and v and a >= 2:
        label = "high"
    elif (d >= 2 and (v or a >= 2)) or (a >= 2 and v):
        label = "medium"
    else:
        label = "low"
    reasons = [f"{d} independent source{'' if d == 1 else 's'}",
               "quote verified" if v else "no verified quote", f"authority {a}/3"]
    subs = {k for n in cited for k in n.get("sub_qs") or []}
    against = sorted({s for k in subs for s in (stance_map.get(str(k)) or {}).get("contradicting_sources") or []})
    if against:
        label = "medium" if label == "high" else label
        reasons.append("contradicted by " + "".join(f"[{s}]" for s in against))
    return label, reasons


# ---------------------------------------------------------------------------
# The analyst's input
# ---------------------------------------------------------------------------

def _headings(archetype: str, short: bool, conflict: bool) -> list[str]:
    heads = list(TEMPLATES[archetype]["analyst_sections"])
    if short:
        heads = [h for h in heads if h != IMPLICATIONS and (h != AGREEMENT or conflict)]
    return heads


def _findings_block(findings: list[dict]) -> str:
    groups: dict[str, list[str]] = {}
    for f in findings:
        ids = "".join(f"[{i}]" for i in f.get("source_ids") or [])
        claim = " ".join(str(f.get("claim") or "").split())[:300]
        groups.setdefault(f.get("section") or "Report", []).append(f"- {claim} {ids}".rstrip())
    lines: list[str] = []
    total = shown = 0
    for section, items in groups.items():
        lines.append(f"## {section}")
        for item in items:
            if total + len(item) > _FINDINGS_CHARS:
                lines.append(f"(… {len(findings) - shown} more findings not shown)")
                return "\n".join(lines)
            lines.append(item)
            total += len(item) + 1
            shown += 1
    return "\n".join(lines)


def _stance_lines(brief: ResearchBrief, smap: dict) -> str:
    out = []
    for k, sq in enumerate(brief.sub_questions):
        row = smap.get(str(k)) or {}
        against = "".join(f"[{s}]" for s in row.get("contradicting_sources") or [])
        out.append(f"({k}) {sq.q} — supports {row.get('supports', 0)}, contradicts {row.get('contradicts', 0)}"
                   + (f" (sources {against})" if against else "")
                   + f", mixed {row.get('mixed', 0)}, background {row.get('background', 0)}")
    return "\n".join(out) or "(no sub-questions)"


def _source_index(ids: set[int], notes_by_id: dict[int, dict], fetched: list[dict]) -> str:
    lines = []
    for i in sorted(ids):
        n = notes_by_id.get(i)
        if n is None or not 0 < i <= len(fetched):
            continue
        item = fetched[i - 1]
        url = item.get("url") or n.get("source_url") or ""
        lines.append(f"[{i}] {(item.get('title') or n.get('title') or '(untitled)')[:90]} — "
                     f"{registrable_domain(url) or 'local'} · authority {n.get('authority', 1)} · "
                     f"published {(item.get('published') or '')[:10] or 'unknown'}")
    return "\n".join(lines) or "(none)"


def _prompt(state, brief: ResearchBrief, findings: list[dict], smap: dict, evidence: list[dict],
            fetched: list[dict], heads: list[str], short: bool) -> str:
    guide = {**_GUIDE, **(_SHORT_GUIDE if short else {})}
    emphasis = f" — {TEMPLATES[brief.archetype]['flat']['analyst_emphasis']}" if short else ""
    cited = {i for f in findings for i in f.get("source_ids") or []}
    cited |= {s for row in smap.values() for s in row["contradicting_sources"]}
    coverage = " · ".join(f"({k}) {v}" for k, v in (state.get("coverage") or {}).items()) or "(none)"
    return "\n".join([
        f"Question: {state.get('user_request')}",
        f"Report type: {brief.archetype}{emphasis}",
        "Success criteria:", *([f"- {c}" for c in brief.success_criteria] or ["- (none)"]), "",
        "Write these sections:", *[f"### {h} — {guide[h]}" for h in heads], "",
        "FINDINGS BY SECTION:", _findings_block(findings), "",
        "STANCE MAP (per sub-question):", _stance_lines(brief, smap), "",
        "COVERAGE (strong notes per sub-question):", coverage, "",
        "SOURCE INDEX:", _source_index(cited, {int(n["source_id"]): n for n in evidence}, fetched), "",
        "Write the analysis markdown now."])


# ---------------------------------------------------------------------------
# The analyst's output
# ---------------------------------------------------------------------------

def _key(title: str) -> str:
    title = re.sub(r"^\s*\d+[.)]\s*", "", title)                    # "1. Key takeaways", "2) …"
    return " ".join(re.sub(r"[^\w\s]", " ", title.replace("&", " and ")).lower().split())


_FIXED = (TAKEAWAYS, AGREEMENT, SCENARIOS, IMPLICATIONS, OPEN_QUESTIONS)
_KNOWN = {**{_key(h): h for h in _FIXED}, "takeaways": TAKEAWAYS}
_HEADING = re.compile(r"^#{2,4}\s+(.+?)\s*#*\s*$")
_BOLD_LINE = re.compile(r"^\s*(?:\*\*|__)(.+?)(?:\*\*|__)\s*:?\s*$")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+(.*\S)\s*$")


def _heading(line: str) -> str | None:
    """The heading a line opens: a fixed heading's canonical name in any
    variant (level 2-4, numbered, case, "&", a trailing colon, or a line of
    only bold text), another markdown heading's own title, else None."""
    m = _HEADING.match(line)
    if m:
        title = m.group(1).strip().strip("*_").strip()
        return _KNOWN.get(_key(title), title)
    m = _BOLD_LINE.match(line)
    return _KNOWN.get(_key(m.group(1))) if m else None


def _split(md: str) -> tuple[list[str], list[list]]:
    """(the lines before the first fixed heading — other headings included —,
    [[heading, body lines], …] from it on); a repeated fixed heading continues
    its first part."""
    before: list[str] = []
    parts: list[list] = []
    current = before
    for line in md.splitlines():
        head = _heading(line)
        if head is None or (not parts and head not in _FIXED):
            current.append(line)
            continue
        part = next((p for p in parts if p[0] == head and head in _FIXED), None)
        if part is None:
            part = [head, []]
            parts.append(part)
        current = part[1]
    return before, parts


def _bullets(lines: list[str]) -> tuple[list[str], list[str]]:
    """(bullet texts, the other non-blank lines); a line right after a bullet
    continues it."""
    bullets: list[str] = []
    other: list[str] = []
    in_bullet = False
    for line in lines:
        m = _BULLET.match(line)
        if m:
            bullets.append(m.group(1))
            in_bullet = True
        elif line.strip() and in_bullet:
            bullets[-1] += " " + line.strip()
        else:
            in_bullet = False
            if line.strip():
                other.append(line)
    return bullets, other


def _trim(lines: list[str]) -> list[str]:
    return "\n".join(lines).strip().splitlines()


def _render(parts: list[list], takeaways: list[dict], *, labelled: bool) -> str:
    blocks = [ANALYSIS_LABEL] if labelled else []
    for head, lines in parts:
        body = "\n".join(lines).strip()
        if head == TAKEAWAYS:
            items = [f"- {t['text']}" + (f" — confidence **{t['confidence']}** ({', '.join(t['reasons'])})"
                                         if labelled else "") for t in takeaways]
            body = "\n\n".join(x for x in (body, "\n".join(items)) if x)
        if body:
            blocks.append(f"### {head}\n\n{body}")
    return "\n\n".join(blocks)


def _has(parts: list[list], head: str) -> bool:
    return any(h == head and "\n".join(lines).strip() for h, lines in parts)


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

def _emit(runtime: Runtime | None, kind: str, data: dict) -> None:
    if runtime is not None:
        runtime.stream_writer({"kind": kind, "node": "analyst", "data": data})


def analyst_node(state, runtime: Runtime | None = None) -> dict:
    """The analysis of the composed report (``analysis``, the findings with
    the analysis findings, an interim ``draft_md``); ``analysis`` None for
    ``tldr``, with ``compose.analyst`` off, without findings, or degraded.
    Section "Analysis" findings are this node's: stale ones are replaced."""
    rc = run_config.require(state)
    findings = list(state.get("findings") or [])
    base = [f for f in findings if f.get("section") != ANALYSIS_SECTION]

    def omitted(**extra) -> dict:
        out = {"analysis": None, **extra}
        if len(base) != len(findings):
            out["findings"] = base
        return out

    def degraded(reason: str, calls: list[dict]) -> dict:      # the declared degradation, visibly
        degradation = DEGRADABLE_ROLES["compose.analyst"]
        logger.warning("compose.analyst degraded (%s); analysis omitted", reason)
        _emit(runtime, "degrade", {"role": "compose.analyst", "degradation": degradation, "reason": reason})
        return omitted(model_calls=calls,
                       errors=[f"[r{_round(state)}] degrade: compose.analyst → {degradation} ({reason})"])

    if rc.length == "tldr" or not rc.compose.analyst or not base:
        return omitted()
    brief = ResearchBrief.model_validate(state.get("brief") or {})
    evidence = state.get("evidence") or []
    fetched = state.get("fetched") or []
    smap = stance_map(brief, evidence)
    short = rc.length == "short"
    heads = _headings(brief.archetype, short, any(row["contradicts"] for row in smap.values()))
    try:
        res = llm.complete("compose.analyst", [
            SystemMessage(content=ANALYST_SYSTEM),
            HumanMessage(content=_prompt(state, brief, base, smap, evidence, fetched, heads, short))],
            engines=rc.engines, temperature=0.3, max_tokens=900 if short else 1800)
    except llm.EngineDegraded as e:
        return degraded(str(e), [a.model_dump() for a in e.attempts])
    calls = [r.model_dump() for r in res.records]
    text, defs = strip_footnote_definitions(normalize_cite_markers(strip_fences(res.text)))
    if not text.strip():
        return degraded("empty reply", calls)
    before, parts = _split(text)
    if not parts:
        return degraded("no analysis headings", calls)

    errors: list[str] = _footnote_lines(_round(state), ANALYSIS_SECTION, defs)     # as compose records them
    if "\n".join(before).strip():
        errors.append("analysis: dropped text before the first heading")
    bullets: list[str] = []
    take = next((p for p in parts if p[0] == TAKEAWAYS), None)
    if take is None:
        errors.append("analysis: takeaways heading missing")
    else:
        bullets, take[1] = _bullets(take[1])
        uncited = [b for b in bullets if not CITE_RE.search(b)]
        bullets = [b for b in bullets if CITE_RE.search(b)]
        if not bullets:
            errors.append("analysis: no cited takeaways")
        if uncited:
            questions = next((p for p in parts if p[0] == OPEN_QUESTIONS), None)
            if questions is None:
                questions = [OPEN_QUESTIONS, []]
                parts.append(questions)
            questions[1] = [*_trim(questions[1]), *(f"- (uncited) {b}" for b in uncited)]
    if short:                                   # the flat short variant (spec §3 WP-I)
        parts = [p for p in parts if p[0] in heads or p[0] not in _FIXED]
        n_take, bullets = len(bullets), bullets[:_SHORT_TAKEAWAYS]
        n_open = kept_open = 0
        for p in parts:
            if p[0] == OPEN_QUESTIONS:
                questions, other = _bullets(p[1])
                n_open, kept_open = len(questions), min(len(questions), _SHORT_OPEN_QUESTIONS)
                p[1] = [*other, *(f"- {q}" for q in questions[:kept_open])]
        if len(bullets) < n_take or kept_open < n_open:
            errors.append(f"analysis: short variant kept {len(bullets)} of {n_take} takeaways, "
                          f"{kept_open} of {n_open} open questions")
    takeaways = []
    for b in bullets:
        ids = list(dict.fromkeys(int(m) for m in CITE_RE.findall(b)))
        label, reasons = confidence_label(ids, fetched, evidence, b, smap)
        takeaways.append({"text": b, "source_ids": ids, "confidence": label, "reasons": reasons})

    plain = _render(parts, takeaways, labelled=False)
    if not plain.strip():                       # fixed headings, empty bodies: a label-only analysis
        return degraded("no analysis content", calls)
    extracted: list[dict] = []
    if CITE_RE.search(plain):
        extracted, ecalls, eerrors = extract_findings(ANALYSIS_SECTION, plain, fetched, engines=rc.engines,
                                                      rnd=_round(state))
        calls.extend(ecalls)
        errors.extend(eerrors)
    md = _render(parts, takeaways, labelled=True)
    analysis = {"md": md, "takeaways": takeaways, "stance_map": smap,
                "has_disagreement_section": _has(parts, AGREEMENT), "has_scenarios": _has(parts, SCENARIOS),
                "archetype": brief.archetype}
    _emit(runtime, "analysis", {"takeaways": len(takeaways), "has_disagreement": analysis["has_disagreement_section"],
                                "has_scenarios": analysis["has_scenarios"]})
    draft = assemble_draft(tldr=None, sections=state.get("sections") or [], analysis_md=md, fetched=fetched,
                           flat=not (state.get("outline") or {}).get("sections"))
    return {
        "analysis": analysis,
        "findings": [{**f, "id": f"f{i}"} for i, f in enumerate([*base, *extracted])],
        "draft_md": draft,
        "model_calls": calls,
        "errors": errors,
        "messages": [{"role": "assistant",
                      "content": f"📊 Analysis: {len(takeaways)} takeaways, {len(extracted)} findings."}],
    }


__all__ = ["analyst_node", "stance_map", "confidence_label", "ANALYST_SYSTEM", "ANALYSIS_LABEL",
           "ANALYSIS_SECTION"]
