"""v4 compose (spec §3 WP-C, §2.8): section writers on their allocated notes,
the reuse rule, the R11 merge of a skipped first write, the dedupe pass and
the findings extractor.

Sectioned lengths: one ``compose.section`` call per section, which sees only
the notes the outline allocated to it (``note_ids``) plus the pinned ones,
filtered by ``writer_min_relevance`` and ``quote_verified is not False``
(``outline.usable_notes``) — never the whole evidence. Compose writes
``panel_verdict.revise_sections`` on a panel rewrite (``next == "compose"``),
else ``outline.changed_sections`` (fresh, gap wave, extend, user revise), and
always a section not written before; every other section is reused as it was
(md, ``note_ids``, findings, re-ided in outline order). A skipped first write
merges its notes into its neighbour (the previous section, else the next),
which is written once more with them; when that re-write is skipped too, the
neighbour keeps its version and the notes are reported as unwritten. A
skipped re-write keeps the previous version; every section skipped raises.
Flat lengths: one ``compose.flat`` call (``raise``) writes the body that
``assemble_draft`` puts under ``## Findings``.

Every writer reply goes through ``normalize_cite_markers`` before the
``compose.findings_extract`` pass (one call per written section; each finding
carries its verbatim ``sentence``) and the assembly. The dedupe pass cuts a
later near-duplicate of an earlier section's finding from a section written
in this run. No TL;DR here: report_builder builds it after the cuts.
User revision notes reach the writers newest first, then the panel's notes
on a panel rewrite (8 at most).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from .. import llm, quarantine
from . import run_config
from .assemble import CITE_RE, assemble_draft, normalize_cite_markers, strip_duplicate_heading, strip_fences
from .grounding import check_grounding, grounding_warning_prompt
from .jsonx import parse_json_obj
from .outline import OUTLINE_SYSTEM, _ints, default_outline, outline_node, usable_notes  # noqa: F401  (re-export: moved to outline.py, WP-I)
from .research import _round
from .state import OutlineSection, ResearchBrief
from .synthesis_modes import get_mode

logger = logging.getLogger("argus.compose")

_MODE_SECTION_TOKENS = {"medium": 1200, "long": 1800, "lecture": 2200}
_MIN_CHARS = 40                 # a shorter section reply is an empty write (a skip)
_MAX_REVISION_NOTES = 8
_DUP_RATIO = 0.85               # = bench.metrics.cross_section_duplicate_ratio's threshold
_ANALYSIS = "Analysis"          # = insight.ANALYSIS_SECTION (insight imports this module)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SECTION_SYSTEM = """You write ONE section of a grounded research report.

Rules:
1. Use ONLY the evidence notes provided. If the evidence is thin for a
   point, say so — never bridge gaps with your own knowledge.
2. Cite with [n] immediately after each factual claim, where n is the
   source id shown on the evidence note. Multiple ids allowed: [2][5].
3. Every number, date, and name must come from a claim/quote in the
   evidence.
4. Where sources conflict, present both sides with their citations.
5. Write tight, information-dense prose. ### sub-headings allowed.
6. Output RAW MARKDOWN for the section BODY ONLY. Do NOT start with a
   heading, title, or bolded restatement of the section name — it is
   rendered separately; begin directly with the first sentence of
   prose. No JSON, no code fences, no Sources list (added
   automatically).
"""

FLAT_SYSTEM = """You write a complete short grounded research report in
markdown.

Rules:
1. Use ONLY the evidence notes provided; never bridge gaps with your own
   knowledge. If evidence is thin, say so.
2. Cite with [n] after each factual claim, n = the source id on the
   evidence note. Every number/date/name must come from the evidence.
3. Write the findings as short paragraphs or a numbered list, each point
   1-2 sentences with its citations. No headings (added automatically), no
   summary paragraph, no Sources section (added automatically).
4. Output RAW MARKDOWN only — no JSON, no code fences.
"""

TLDR_SYSTEM = """Write a single short paragraph (80-400 chars) answering
the user's question from the evidence notes only. Cite with [n] inline
(n = source id). If the evidence can't answer, say so plainly. Output
raw text only."""

EXTRACT_SYSTEM = """Extract the factual findings from ONE section of a
research report. A finding = one claim with the source ids it cites.

Return ONLY JSON:
{"findings": [{"claim": "...", "sentence": "...", "source_ids": [2, 5],
               "confidence": "high|medium|low"}]}

Rules:
- claim: the fact, condensed from the section — never invented.
- sentence: the section's own sentence (or adjacent sentences) stating
  it, copied VERBATIM with its [n] markers: character for character, no
  rewording, no shortening.
- source_ids: the [n] numbers cited by that sentence.
- Skip sentences with no citation.
"""


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def _is_local_note(n: dict) -> bool:
    """User-appended vault material (file:///) — pinned, never crowded out."""
    return str(n.get("source_url") or "").startswith("file:///")


def render_notes(evidence: list[dict], note_ids, *, max_chars: int = 7000,
                 pinned: frozenset[int] = frozenset()) -> str:
    """The notes with a source id in ``note_ids`` (None: every note), plus
    the local and ``pinned`` ones, as ``[n]``-labelled blocks for a writer
    prompt: pinned first, then by relevance; a claim whose quote failed
    verification is left out. Stops at ``max_chars``."""
    def is_pinned(n: dict) -> bool:
        return _is_local_note(n) or int(n.get("source_id") or 0) in pinned

    wanted = None if note_ids is None else set(note_ids)
    picked = [n for n in evidence if wanted is None or int(n.get("source_id") or 0) in wanted or is_pinned(n)]
    picked.sort(key=lambda n: (is_pinned(n), int(n.get("relevance") or 0)), reverse=True)
    blocks: list[str] = []
    total = 0
    for n in picked:
        claims = "\n".join(
            f"   • {c.get('text')} (conf: {c.get('confidence', 'medium')})"
            + (f'\n     quote: "{c.get("quote")}"' if c.get("quote") else "")
            for c in n.get("claims") or [] if c.get("quote_verified") is not False)
        block = (f"[{n.get('source_id')}] {n.get('title') or '(untitled)'} — {n.get('source_url')}\n"
                 f"   relevance {n.get('relevance')}/5, stance: {n.get('stance')}\n{claims}")
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks) or "(no evidence notes)"


def _pinned_ids(notes: list[dict], fetched: list[dict]) -> frozenset[int]:
    out = set()
    for n in notes:
        i = int(n["source_id"])
        if _is_local_note(n) or (0 < i <= len(fetched) and fetched[i - 1].get("pinned")):
            out.add(i)
    return frozenset(out)


def _revision_block(notes: list[str]) -> str:
    if not notes:
        return ""
    return "\n\nREVISION NOTES (must address):\n" + "\n".join(f"- {n}" for n in notes)


# ---------------------------------------------------------------------------
# The findings extractor
# ---------------------------------------------------------------------------

def _parse_findings(text: str) -> list[dict]:
    data = parse_json_obj(text, require_any=("findings",))
    raw = data.get("findings")
    if not isinstance(raw, list):
        raise ValueError(f"'findings' is {type(raw).__name__}, not a list")
    return [f for f in raw if isinstance(f, dict)]


def extract_findings(title: str, md: str, fetched: list[dict], *, engines, rnd: int
                     ) -> tuple[list[dict], list[dict], list[str]]:
    """The findings of one section (or of the analysis): ``(findings, model
    call records, errors)``. One ``compose.findings_extract`` call; each
    finding keeps the verbatim ``sentence`` (markers normalised) and cites
    the ids in it — the model's ``source_ids`` only when the sentence has
    none. A finding citing nothing is left out; one citing only unknown ids
    is kept (report_builder cuts it). A skipped call is the declared
    ``[r<k>] skip: compose.findings_extract <title> (…)`` line."""
    by_id = {i + 1: f.get("url") or "" for i, f in enumerate(fetched)}
    try:
        res = llm.complete("compose.findings_extract", [
            SystemMessage(content=EXTRACT_SYSTEM),
            HumanMessage(content=f"Section: {title}\n\n{md}\n\nReturn the findings JSON.")],
            engines=engines, temperature=0.1, max_tokens=2000, validate=_parse_findings, item=title)
    except llm.EngineSkipped as e:
        logger.warning("compose.findings_extract skipped for %r (%s)", title, e)
        return [], [a.model_dump() for a in e.attempts], [f"[r{rnd}] skip: compose.findings_extract {title} ({e})"]
    out: list[dict] = []
    for fd in res.parsed:
        claim = str(fd.get("claim") or "").strip()
        if not claim:
            continue
        sentence = normalize_cite_markers(str(fd.get("sentence") or "")).strip()
        ids = list(dict.fromkeys(int(i) for i in CITE_RE.findall(sentence))) or _ints(fd.get("source_ids"))
        if not ids:
            continue
        confidence = fd.get("confidence") if fd.get("confidence") in ("high", "medium", "low") else "medium"
        out.append({"claim": claim[:600], "citation_urls": [by_id[i] for i in ids if by_id.get(i)],
                    "confidence": confidence, "section": title, "source_ids": ids, "sentence": sentence,
                    "status": "kept"})
    return out, [r.model_dump() for r in res.records], []


# ---------------------------------------------------------------------------
# Validated assessment — honest metrics, not LLM self-grading.
# ---------------------------------------------------------------------------

def build_assessment(sections: list[dict], findings: list[dict],
                     coverage: dict, brief: ResearchBrief) -> dict:
    per_section: list[dict] = []
    claims_per_section: dict[str, int] = {}
    for s in sections:
        title = s.get("title", "")
        n_claims = sum(1 for f in findings if f.get("section") == title)
        confs = [f.get("confidence") for f in findings
                 if f.get("section") == title]
        high = sum(1 for c in confs if c == "high")
        conf = ("high" if confs and high >= len(confs) / 2
                else "low" if n_claims == 0 else "medium")
        gaps = [brief.sub_questions[i].q
                for i in (s.get("sub_qs") or [])
                if str(i) in coverage and coverage[str(i)] < 2
                and i < len(brief.sub_questions)]
        per_section.append({
            "name": title, "confidence": conf,
            "open_challenges": [f"thin evidence: {g}" for g in gaps[:3]],
            "claim_count": n_claims,
        })
        claims_per_section[title] = n_claims
    n_findings = len(findings)
    n_cites = sum(len(f.get("citation_urls") or []) for f in findings)
    return {
        "sections": per_section,
        "overall_relevancy": ("high" if n_findings >= 6 else
                              "medium" if n_findings >= 2 else "low"),
        "knowledge_density": {
            "claims_per_section": claims_per_section,
            "citations_per_claim": round(n_cites / n_findings, 2)
            if n_findings else 0.0,
            "conflict_markers": 0,   # not computed in v3 — always 0
        },
        "reviewer_unsupported": [],
        "reviewer_fabrication_flags": [],
    }


def _no_evidence_md(topic: str, *, length: str, fetched_count: int,
                    reason: str) -> str:
    mode = get_mode(length)
    return (
        f"# {topic}\n\n"
        "⚠️ **Synthesis could not produce findings.**\n\n"
        "- **Class:** `synthesis_no_evidence`\n"
        f"- **Mode:** {mode.label}\n"
        f"- **Fetched evidence:** {fetched_count} item(s)\n"
        f"- **Reason:** {reason or '(no reason given)'}\n\n"
        "This is a classified synthesis failure (not a successful "
        "report). The pipeline could not derive at least one grounded, "
        "cited finding from the digested evidence.\n\n"
        "**Try:**\n"
        "1. `/research <narrower scope>` — fewer sub-topics, more "
        "concrete entity.\n"
        "2. Retry in a minute — transient proxy errors auto-recover.\n"
        "3. If it persists, the topic may genuinely have thin public "
        "coverage.\n"
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _emit(runtime: Runtime | None, kind: str, data: dict) -> None:
    if runtime is not None:
        runtime.stream_writer({"kind": kind, "node": "compose", "data": data})


class _Ctx:
    """What every writer of one compose run shares."""

    def __init__(self, state, rc: run_config.RunConfig, brief: ResearchBrief, notes: list[dict],
                 revision_notes: list[str], n_sections: int):
        self.state, self.rc, self.brief, self.notes = state, rc, brief, notes
        self.fetched = state.get("fetched") or []
        self.pinned = _pinned_ids(notes, self.fetched)
        self.revision_notes = revision_notes
        self.n_sections = n_sections
        self.rnd = _round(state)
        self.grounding_warn = grounding_warning_prompt(check_grounding(state), state.get("user_request") or "")


def _write_section(ctx: _Ctx, section: OutlineSection, note_ids: list[int]
                   ) -> tuple[str | None, list[dict], str | None]:
    """One ``compose.section`` write: ``(md | None, records, skip line | None)``."""
    mode = get_mode(ctx.rc.length)
    title = section.title
    covered = "\n".join(f"- {ctx.brief.sub_questions[i].q}" for i in section.sub_qs
                        if i < len(ctx.brief.sub_questions))
    human = (
        f"Report topic: {ctx.state.get('user_request')}\n"
        f"Section: {title}\n"
        f"Focus: {section.focus or '(general)'}\n"
        f"Sub-questions to answer here:\n{covered or '- (general coverage)'}\n"
        f"Target: ~{max(400, mode.target_chars_max // max(1, ctx.n_sections))} chars."
        f"{_revision_block(ctx.revision_notes)}\n\n"
        f"EVIDENCE NOTES:\n{render_notes(ctx.notes, note_ids, pinned=ctx.pinned)}\n\n"
        "Write the section body markdown now."
    )
    try:
        res = llm.complete("compose.section", [SystemMessage(content=SECTION_SYSTEM + ctx.grounding_warn),
                                               HumanMessage(content=human)],
                           engines=ctx.rc.engines, temperature=mode.temperature,
                           max_tokens=_MODE_SECTION_TOKENS.get(ctx.rc.length, 1200), item=title)
    except llm.EngineSkipped as e:
        logger.warning("compose.section skipped for %r (%s)", title, e)
        return None, [a.model_dump() for a in e.attempts], f"[r{ctx.rnd}] skip: compose.section {title} ({e})"
    records = [r.model_dump() for r in res.records]
    md = normalize_cite_markers(strip_duplicate_heading(strip_fences(res.text), title)).strip()
    if len(md) < _MIN_CHARS:
        records[-1].update(ok=False, error="empty reply")
        return None, records, f"[r{ctx.rnd}] skip: compose.section {title} (empty reply)"
    return md, records, None


# ---------------------------------------------------------------------------
# The dedupe pass
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """= bench.metrics._norm: NFKC, casefold, [n] markers out, word tokens."""
    text = CITE_RE.sub(" ", unicodedata.normalize("NFKC", text).casefold())
    return " ".join(re.findall(r"\w+", text))


def _dedupe(order: list[str], mds: dict[str, str], found: dict[str, list[dict]], written: set[str]) -> None:
    """In place: a finding of a section written in this run whose claim
    nearly repeats (ratio ≥ 0.85) a finding of an earlier section is cut from
    that section's md (its whole sentences) and dropped — unless its
    sentences also state another finding of that section, or cannot be
    located. The first occurrence wins; a reused section is never touched."""
    seen: list[str] = []
    for title in order:
        mine = found.get(title) or []
        if title in written:
            # ponytail: every claim against every earlier one through difflib, O(n²) over one
            # report's findings (tens to a few hundred); an index if reports grow far past that.
            dup = [any(SequenceMatcher(None, _norm(f["claim"]), s, autojunk=False).ratio() >= _DUP_RATIO
                       for s in seen) for f in mine]
            where = [quarantine.locate(f.get("sentence") or f["claim"], mds[title]) or [] for f in mine]
            keep_at = {i for k, idx in enumerate(where) if not dup[k] for i in idx}
            gone = {k for k, idx in enumerate(where) if dup[k] and idx and not set(idx) & keep_at}
            if gone:
                mds[title] = quarantine.cut(mds[title], {i for k in gone for i in where[k]})
                found[title] = [f for k, f in enumerate(mine) if k not in gone]
        seen.extend(_norm(f["claim"]) for f in found.get(title) or [])


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

def _extras(state, brief: ResearchBrief, rc: run_config.RunConfig, sections: list[dict], findings: list[dict],
            draft: str) -> dict:
    """The v3 title-page assessment and the lecture appendix (mode-dependent)."""
    mode = get_mode(rc.length)
    fetched = state.get("fetched") or []
    out: dict = {}
    if mode.include_validated_assessment:
        out["validated_assessment"] = build_assessment(sections, findings, state.get("coverage") or {}, brief)
    if mode.include_appendix:
        cites = sum(len(f.get("citation_urls") or []) for f in findings)
        out["lecture_appendix"] = {
            "methodology": (
                f"{len(state.get('queries') or [])} live queries across "
                f"Exa/DDGS/arXiv/GitHub; {len(fetched)} sources fetched; "
                f"{len(state.get('evidence') or [])} digested into evidence notes over "
                f"{state.get('research_rounds') or 1} wave(s)."),
            "tool_calls": [f"{q.get('provider')}: {q.get('query')}" for q in (state.get("queries") or [])][:30],
            "density_metrics": {
                "total_chars": len(draft), "total_findings": len(findings), "total_citations": cites,
                "unique_sources": len(fetched),
                "avg_citations_per_finding": round(cites / len(findings), 2) if findings else 0.0,
            },
        }
    return out


def _records(calls: list[dict]) -> list[llm.CallRecord]:
    """The attempt records an all-skipped EngineError carries (as research's does)."""
    return [llm.CallRecord(**c) for c in calls]


def _extract_all(ctx: _Ctx, written: dict[str, str], pool_size: int
                 ) -> tuple[dict[str, list[dict]], list[dict], list[str], bool]:
    """Extraction over every written section, on the pool: (findings by
    section, records, errors, every attempted extraction skipped)."""
    found: dict[str, list[dict]] = {}
    calls: list[dict] = []
    errors: list[str] = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="argus-extract") as ex:
        futs = {t: ex.submit(extract_findings, t, md, ctx.fetched, engines=ctx.rc.engines, rnd=ctx.rnd)
                for t, md in written.items()}
        for title, fut in futs.items():
            fs, recs, errs = fut.result()
            found[title] = fs
            calls.extend(recs)
            errors.extend(errs)
            skipped += bool(errs)
    return found, calls, errors, bool(written) and skipped == len(written)


def _flat(state, ctx: _Ctx, pv: dict, runtime: Runtime | None) -> dict:
    rc = ctx.rc
    prior = [s for s in state.get("sections") or [] if s.get("title") == "Report"]
    if pv.get("next") == "compose" and "Report" not in (pv.get("revise_sections") or []) and prior:
        section = prior[0]                       # a panel rewrite of other sections only (the analysis)
        findings = [{**f, "status": "kept"} for f in state.get("findings") or [] if f.get("section") == "Report"]
        calls, errors = [], []
        _emit(runtime, "section", {"title": "Report", "ok": True, "chars": len(section["md"]), "reused": True})
    else:
        mode = get_mode(rc.length)
        system = (TLDR_SYSTEM if rc.length == "tldr" else FLAT_SYSTEM) + ctx.grounding_warn
        human = (f"Question: {state.get('user_request')}\n"
                 f"Target length: {mode.target_chars_min}-{mode.target_chars_max} chars."
                 f"{_revision_block(ctx.revision_notes)}\n\n"
                 f"EVIDENCE NOTES:\n{render_notes(ctx.notes, None, max_chars=8000, pinned=ctx.pinned)}\n\n"
                 "Write the markdown now.")
        res = llm.complete("compose.flat", [SystemMessage(content=system), HumanMessage(content=human)],
                           engines=rc.engines, temperature=mode.temperature, max_tokens=mode.max_tokens)
        calls = [r.model_dump() for r in res.records]
        md = normalize_cite_markers(strip_fences(res.text)).strip()
        if not md:
            raise llm.EngineError("role compose.flat: empty reply", role="compose.flat", attempts=res.records)
        section = {"title": "Report", "md": md, "note_ids": [int(n["source_id"]) for n in ctx.notes]}
        found, ecalls, errors, all_skipped = _extract_all(ctx, {"Report": md}, 1)
        calls.extend(ecalls)
        findings = found["Report"]
        if all_skipped:
            raise llm.EngineError(f"compose: the findings extraction was skipped ({errors[0]})",
                                  role="compose.findings_extract", attempts=_records(ecalls))
        _emit(runtime, "section", {"title": "Report", "ok": True, "chars": len(md), "reused": False})
    findings = [{**f, "id": f"f{i}"} for i, f in enumerate(findings)]
    draft = assemble_draft(tldr=None, sections=[section], analysis_md=None, fetched=ctx.fetched, flat=True)
    return {
        "draft_md": draft, "sections": [section], "findings": findings,
        "compose_runs": int(state.get("compose_runs") or 0) + 1,
        "model_calls": calls, "errors": errors,
        "messages": [{"role": "assistant", "content": f"🧠 Composed flat report, {len(findings)} findings."}],
        **_extras(state, ctx.brief, rc, [section], findings, draft),
    }


def compose_node(state, runtime: Runtime | None = None) -> dict:
    """Write (or rewrite) the report sections from their allocated notes."""
    rc = run_config.require(state)
    brief = ResearchBrief.model_validate(state.get("brief") or {})
    fetched = state.get("fetched") or []
    runs = int(state.get("compose_runs") or 0) + 1
    notes = usable_notes(state.get("evidence") or [], fetched, rc.research.writer_min_relevance)
    if not notes:
        reason = (f"No digested evidence note reached relevance >= {rc.research.writer_min_relevance} "
                  "with at least one verified claim.")
        return {"draft_md": _no_evidence_md(state.get("user_request") or "", length=rc.length,
                                            fetched_count=len(fetched), reason=reason),
                "findings": [], "sections": [], "compose_runs": runs,
                "messages": [{"role": "assistant", "content": "⚠️ Compose: no usable evidence."}]}

    pv = state.get("panel_verdict") or {}
    panel_rewrite = pv.get("next") == "compose"
    revision_notes = [*reversed(state.get("revision_notes") or []),
                      *((pv.get("notes") or []) if panel_rewrite else [])][:_MAX_REVISION_NOTES]
    outline = [OutlineSection.model_validate(s) for s in (state.get("outline") or {}).get("sections") or []]
    ctx = _Ctx(state, rc, brief, notes, revision_notes, len(outline))
    if not outline:
        return _flat(state, ctx, pv, runtime)

    prior = {s["title"]: s for s in state.get("sections") or []}
    if panel_rewrite:
        targets = set(pv.get("revise_sections") or [])
    else:
        changed = (state.get("outline") or {}).get("changed_sections")
        targets = {s.title for s in outline} if changed is None else set(changed)
    order = [s.title for s in outline]
    by_title = {s.title: s for s in outline}
    to_write = [s for s in outline if s.title in targets or s.title not in prior]
    pool_size = max(1, min(rc.compose.concurrency, len(outline)))

    calls: list[dict] = []
    errors: list[str] = []
    note_ids = {s.title: list(s.note_ids) for s in outline}
    mds: dict[str, str] = {}
    written: set[str] = set()
    failed: set[str] = set()                                     # a write of this run that was skipped

    def run(batch: list[tuple[OutlineSection, list[int]]]) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        with ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="argus-compose") as ex:
            for (section, _), (md, recs, err) in zip(batch, ex.map(lambda b: _write_section(ctx, *b), batch)):
                calls.extend(recs)
                if err:
                    errors.append(err)
                out[section.title] = md
        return out

    for title, md in run([(s, note_ids[s.title]) for s in to_write]).items():
        if md is not None:
            mds[title] = md
            written.add(title)
        elif title in prior:                                     # a skipped re-write keeps the previous version
            failed.add(title)
    for title in order:
        if title in prior and title not in mds:
            mds[title] = prior[title]["md"]
            note_ids[title] = list(prior[title].get("note_ids") or [])
    # R11: a skipped first write merges into its neighbour (previous, else next), written once more.
    lost = [s.title for s in to_write if s.title not in mds]
    if lost and not mds:
        raise llm.EngineError("compose: every section was skipped", role="compose.section", attempts=_records(calls))
    merged: dict[str, list[str]] = {}
    for title in lost:
        k = order.index(title)
        alive = [t for t in order if t in mds]
        before = [t for t in alive if order.index(t) < k]
        merged.setdefault(before[-1] if before else alive[0], []).append(title)
    if merged:
        batch = [(by_title[t], list(dict.fromkeys([*note_ids[t], *(i for s in src for i in note_ids[s])])))
                 for t, src in merged.items()]
        for (section, ids), md in zip(batch, run(batch).values()):
            title = section.title
            if md is not None:
                mds[title], note_ids[title] = md, ids
                written.add(title)
                failed.discard(title)
            else:
                for src in merged[title]:
                    errors.append(f'[r{ctx.rnd}] compose: section "{src}" dropped — notes '
                                  f'{", ".join(map(str, note_ids[src])) or "none"} not written '
                                  f'(its merge into "{title}" was skipped)')
    order = [t for t in order if t in mds]

    found, ecalls, eerrors, all_skipped = _extract_all(ctx, {t: mds[t] for t in order if t in written}, pool_size)
    calls.extend(ecalls)
    errors.extend(eerrors)
    for title in order:
        if title not in written:
            found[title] = [{**f, "status": "kept"} for f in state.get("findings") or [] if f.get("section") == title]
    if all_skipped and not any(found.values()):
        raise llm.EngineError(f"compose: every findings extraction was skipped ({eerrors[0]})",
                              role="compose.findings_extract", attempts=_records(ecalls))
    if rc.compose.dedupe_pass:
        _dedupe(order, mds, found, written)

    sections = [{"title": t, "md": mds[t], "note_ids": note_ids[t]} for t in order]
    findings = [{**f, "id": f"f{i}"} for i, f in enumerate(f for t in order for f in found[t])]
    for title in [s.title for s in outline]:
        if title in mds:
            _emit(runtime, "section", {"title": title, "ok": title not in failed, "chars": len(mds[title]),
                                       "reused": title not in written})
        else:
            _emit(runtime, "section", {"title": title, "ok": False, "chars": 0, "reused": False})
    draft = assemble_draft(tldr=None, sections=sections, analysis_md=None, fetched=fetched, flat=False)
    mode = get_mode(rc.length)
    return {
        "draft_md": draft, "sections": sections, "findings": findings, "compose_runs": runs,
        "model_calls": calls, "errors": errors,
        "messages": [{"role": "assistant",
                      "content": (f"🧠 Composed {len(sections)} sections ({len(written)} written), "
                                  f"{len(findings)} findings ({mode.label}).")}],
        **_extras(state, brief, rc, sections, findings, draft),
    }


__all__ = ["outline_node", "compose_node", "render_notes", "extract_findings", "build_assessment",
           "default_outline", "SECTION_SYSTEM", "FLAT_SYSTEM", "TLDR_SYSTEM", "OUTLINE_SYSTEM", "EXTRACT_SYSTEM"]
