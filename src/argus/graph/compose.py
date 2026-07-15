"""v3 compose stage — outline → parallel section writers → assembly.

STORM-style: an outline is planned first, then each section is written
by its own strong-tier call that sees ONLY that section's evidence
notes. Writers emit PLAIN MARKDOWN — never markdown-inside-JSON (the v2
fragility that produced the 2026-07-12 empty-report bug is structurally
gone). Findings for the citation/quarantine machinery are extracted per
section afterwards in small JSON passes, with claims citing source ids
(short ints — far more weak-model-robust than long URLs).

Revision modes:
- panel revise  → rewrite only ``panel_verdict.revise_sections`` with
  the judges' notes;
- user revise   → rewrite every section with the user's notes
  (``revision_notes``).
"""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .. import llm
from .grounding import check_grounding, grounding_warning_prompt
from .jsonx import parse_json_obj, salvage_objects
from .state import OutlineSection, ResearchBrief
from .synthesis_modes import get_mode, is_valid as is_valid_length

logger = logging.getLogger("argus.compose")

_COMPOSE_CONCURRENCY = int(os.environ.get("ARGUS_COMPOSE_CONCURRENCY", "3"))

# Per-mode structural budgets (sections beyond TL;DR/Sources).
_MODE_SECTIONS = {"tldr": 0, "short": 0, "medium": 4, "long": 6,
                  "lecture": 7}
_MODE_SECTION_TOKENS = {"medium": 1200, "long": 1800, "lecture": 2200}


# ---------------------------------------------------------------------------
# Outline
# ---------------------------------------------------------------------------

OUTLINE_SYSTEM = """You plan the section structure of a research report.

You get the research brief (sub-questions + success criteria) and a
summary of the gathered evidence. Produce a section outline:

- Sections must collectively cover the sub-questions that HAVE evidence.
- Each section gets: a title, a one-line focus, and the sub-question
  indices it covers.
- Do NOT create sections for TL;DR or Sources — those are added
  automatically.
- Prefer thematic titles over restating sub-questions.

Return ONLY JSON:
{"sections": [{"title": "...", "focus": "...", "sub_qs": [0, 2]}, ...]}
"""

LECTURE_HINT = """
This is a LECTURE-format report: name the sections as parts, in order:
"Part I — <Background theme>", "Part II — <Current state theme>" (Part II
may be split into two sections), "Part III — <Open problems theme>",
"Part IV — Practice and exercises".
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


def outline_node(state) -> dict:
    """Plan the report sections (skipped structurally for tldr/short)."""
    length = state.get("length") or "short"
    if not is_valid_length(length):
        length = "short"
    brief = ResearchBrief.model_validate(state.get("brief") or {})
    evidence = state.get("evidence") or []

    if _MODE_SECTIONS.get(length, 0) == 0 or len(brief.sub_questions) <= 1:
        # Flat modes need no outline call.
        return {"outline": {"sections": []},
                "messages": [{"role": "assistant",
                              "content": "🧭 Flat mode — no outline needed."}]}

    human = (
        f"Topic: {state.get('user_request')}\n"
        f"Mode: {length} (max {_MODE_SECTIONS[length]} sections)\n\n"
        f"Sub-questions and evidence strength:\n"
        f"{_evidence_summary_lines(brief, evidence)}\n\n"
        f"Success criteria:\n"
        + "\n".join(f"- {c}" for c in brief.success_criteria)
        + (LECTURE_HINT if length == "lecture" else "")
        + "\n\nReturn the outline JSON."
    )
    sections: list[OutlineSection] = []
    calls: list[dict] = []
    errors: list[str] = []
    try:
        chat = llm.chat_for_tier("strong", temperature=0.2, max_tokens=700)
        resp = llm.invoke_with_retry(chat, [
            SystemMessage(content=OUTLINE_SYSTEM),
            HumanMessage(content=human[:8000]),
        ])
        calls.append(llm.record_from_response(
            "strong", llm.resolve_tier("strong"), resp).model_dump())
        data = parse_json_obj(resp.content, require_any=("sections",))
        n_subs = len(brief.sub_questions)
        for s in (data.get("sections") or [])[:_MODE_SECTIONS[length]]:
            if not isinstance(s, dict) or not (s.get("title") or "").strip():
                continue
            subs = [int(i) for i in (s.get("sub_qs") or [])
                    if isinstance(i, (int, float)) and 0 <= int(i) < n_subs]
            sections.append(OutlineSection(
                title=str(s["title"]).strip()[:100],
                focus=str(s.get("focus") or "")[:200],
                sub_qs=subs))
    except Exception as e:
        logger.warning("outline failed (%s); deterministic fallback", e)
        errors.append(f"outline: LLM failed ({e}); deterministic fallback")
    if not sections:
        sections = default_outline(brief, length)
    # Every sub-question with evidence must be owned by some section.
    owned = {i for s in sections for i in s.sub_qs}
    orphans = [i for i in range(len(brief.sub_questions)) if i not in owned]
    if orphans and sections:
        sections[-1].sub_qs = list(sections[-1].sub_qs) + orphans

    return {
        "outline": {"sections": [s.model_dump() for s in sections]},
        "model_calls": calls,
        "errors": errors,
        "messages": [{"role": "assistant",
                      "content": (f"🧭 Outline: {len(sections)} sections.")}],
    }


# ---------------------------------------------------------------------------
# Section writers
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
3. Structure: ## TL;DR (short paragraph), ## Key Findings (numbered,
   each 1-2 sentences with citations). Do NOT write a Sources section —
   it is added automatically.
4. Output RAW MARKDOWN only — no JSON, no code fences.
"""

TLDR_SYSTEM = """Write a single short paragraph (80-400 chars) answering
the user's question from the evidence notes only. Cite with [n] inline
(n = source id). If the evidence can't answer, say so plainly. Output
raw text only."""


def _is_local_note(n: dict) -> bool:
    """User-appended vault material (file:///) — pinned, never crowded out."""
    return str(n.get("source_url") or "").startswith("file:///")


def render_notes(evidence: list[dict], sub_qs: list[int] | None,
                 *, max_chars: int = 7000) -> str:
    """Render evidence notes (optionally filtered by sub-question) as
    [n]-labelled blocks for a writer prompt. Pinned (user-appended
    local) notes first, then strong notes."""
    picked: list[dict] = []
    for n in evidence:
        if sub_qs is not None and not _is_local_note(n) \
                and not (set(n.get("sub_qs") or []) & set(sub_qs)):
            continue
        picked.append(n)
    if not picked and sub_qs is not None:
        picked = list(evidence)  # graceful: don't starve a section
    picked.sort(key=lambda n: (_is_local_note(n),
                               int(n.get("relevance") or 0)), reverse=True)
    blocks: list[str] = []
    total = 0
    for n in picked:
        claims = "\n".join(
            f"   • {c.get('text')} (conf: {c.get('confidence','medium')})"
            + (f'\n     quote: "{c.get("quote")}"' if c.get("quote") else "")
            for c in (n.get("claims") or []))
        block = (f"[{n.get('source_id')}] {n.get('title') or '(untitled)'} "
                 f"— {n.get('source_url')}\n"
                 f"   relevance {n.get('relevance')}/5, "
                 f"stance: {n.get('stance')}\n{claims}")
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks) or "(no evidence notes)"


def _write_section(section: dict, brief: ResearchBrief, state: dict,
                   notes_block: str, *, length: str,
                   extra_notes: list[str]) -> tuple[str, dict | None, str]:
    """Write one section body. Returns (markdown, model_call, error)."""
    mode = get_mode(length)
    grounding_warn = grounding_warning_prompt(
        check_grounding(state), state.get("user_request") or "")
    notes_extra = ""
    if extra_notes:
        notes_extra = ("\n\nREVISION NOTES (must address):\n"
                       + "\n".join(f"- {n}" for n in extra_notes[:8]))
    covered = "\n".join(
        f"- {brief.sub_questions[i].q}"
        for i in (section.get("sub_qs") or [])
        if i < len(brief.sub_questions))
    human = (
        f"Report topic: {state.get('user_request')}\n"
        f"Section: {section.get('title')}\n"
        f"Focus: {section.get('focus') or '(general)'}\n"
        f"Sub-questions to answer here:\n{covered or '- (general coverage)'}\n"
        f"Target: ~{max(400, mode.target_chars_max // max(1, len((state.get('outline') or {}).get('sections') or [1])))} chars."
        f"{notes_extra}\n\n"
        f"EVIDENCE NOTES:\n{notes_block}\n\n"
        "Write the section body markdown now."
    )
    try:
        chat = llm.chat_for_tier(
            "strong", temperature=mode.temperature,
            max_tokens=_MODE_SECTION_TOKENS.get(length, 1200))
        resp = llm.invoke_with_retry(chat, [
            SystemMessage(content=SECTION_SYSTEM + grounding_warn),
            HumanMessage(content=human[:13000]),
        ])
        call = llm.record_from_response(
            "strong", llm.resolve_tier("strong"), resp).model_dump()
        md = _strip_fences(resp.content)
        md = _strip_duplicate_heading(md, section.get("title", ""))
        if len(md.strip()) < 40:
            return "", call, (f"compose: section '{section.get('title')}' "
                              "came back empty")
        return md.strip(), call, ""
    except Exception as e:
        return "", None, (f"compose: section '{section.get('title')}' "
                          f"failed ({e})")


def _strip_fences(text: str) -> str:
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


def _strip_duplicate_heading(md: str, title: str) -> str:
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


def _write_flat(state: dict, brief: ResearchBrief, *, length: str,
                extra_notes: list[str]) -> tuple[str, list[dict], list[str]]:
    """tldr/short path: one writer call for the whole body."""
    mode = get_mode(length)
    evidence = state.get("evidence") or []
    notes_block = render_notes(evidence, None, max_chars=8000)
    grounding_warn = grounding_warning_prompt(
        check_grounding(state), state.get("user_request") or "")
    system = TLDR_SYSTEM if length == "tldr" else FLAT_SYSTEM
    notes_extra = ""
    if extra_notes:
        notes_extra = ("\n\nREVISION NOTES (must address):\n"
                       + "\n".join(f"- {n}" for n in extra_notes[:8]))
    human = (
        f"Question: {state.get('user_request')}\n"
        f"Target length: {mode.target_chars_min}-{mode.target_chars_max} "
        f"chars.{notes_extra}\n\n"
        f"EVIDENCE NOTES:\n{notes_block}\n\nWrite the markdown now."
    )
    try:
        chat = llm.chat_for_tier("strong", temperature=mode.temperature,
                                 max_tokens=mode.max_tokens)
        resp = llm.invoke_with_retry(chat, [
            SystemMessage(content=system + grounding_warn),
            HumanMessage(content=human[:13000]),
        ])
        call = llm.record_from_response(
            "strong", llm.resolve_tier("strong"), resp).model_dump()
        return _strip_fences(resp.content), [call], []
    except Exception as e:
        return "", [], [f"compose: flat write failed ({e})"]


# ---------------------------------------------------------------------------
# TL;DR + assembly
# ---------------------------------------------------------------------------

def _write_tldr(state: dict, body: str) -> tuple[str, list[dict], list[str]]:
    human = (f"Question: {state.get('user_request')}\n\n"
             f"REPORT BODY:\n{body[:9000]}\n\n"
             "Write a 2-4 sentence TL;DR of this report. Keep the [n] "
             "citations of the claims you compress. Raw text only.")
    try:
        chat = llm.chat_for_tier("cheap", temperature=0.2, max_tokens=350)
        resp = llm.invoke_with_retry(chat, [
            SystemMessage(content="You summarize research reports faithfully."),
            HumanMessage(content=human),
        ])
        call = llm.record_from_response(
            "cheap", llm.resolve_tier("cheap"), resp).model_dump()
        return _strip_fences(resp.content), [call], []
    except Exception as e:
        return "", [], [f"compose: tldr write failed ({e})"]


_CITE_RE = re.compile(r"\[(\d{1,3})\]")


def build_sources_block(body: str, fetched: list[dict]) -> str:
    """## Sources listing every source id cited in the body, id order."""
    cited = sorted({int(m) for m in _CITE_RE.findall(body)})
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


# ---------------------------------------------------------------------------
# Findings extraction (feeds citation verification + quarantine)
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """Extract the factual findings from ONE section of a
research report. A finding = one claim with the source ids it cites.

Return ONLY JSON:
{"findings": [{"claim": "...", "source_ids": [2, 5],
               "confidence": "high|medium|low"}]}

Rules: claims must be copied/condensed from the section (not invented);
source_ids are the [n] citation numbers attached to the claim; skip
sentences with no citation.
"""


def _extract_findings(section_title: str, section_md: str,
                      fetched: list[dict]
                      ) -> tuple[list[dict], dict | None, str]:
    by_id = {i + 1: f.get("url", "") for i, f in enumerate(fetched)}
    try:
        chat = llm.chat_for_tier("cheap", temperature=0.1, max_tokens=900)
        resp = llm.invoke_with_retry(chat, [
            SystemMessage(content=EXTRACT_SYSTEM),
            HumanMessage(content=(f"Section: {section_title}\n\n"
                                  f"{section_md[:8000]}\n\n"
                                  "Return the findings JSON.")),
        ])
        call = llm.record_from_response(
            "cheap", llm.resolve_tier("cheap"), resp).model_dump()
    except Exception as e:
        return [], None, f"findings: '{section_title}' failed ({e})"
    raw_findings: list[dict] = []
    try:
        data = parse_json_obj(resp.content, require_any=("findings",))
        raw_findings = list(data.get("findings") or [])
    except Exception:
        raw_findings = salvage_objects(resp.content,
                                       required_keys=("claim",))
    out: list[dict] = []
    for fd in raw_findings:
        if not isinstance(fd, dict) or not (fd.get("claim") or "").strip():
            continue
        urls = []
        for sid in fd.get("source_ids") or []:
            try:
                u = by_id.get(int(sid), "")
            except (TypeError, ValueError):
                continue
            if u:
                urls.append(u)
        # Legacy shape tolerance: citation_urls straight from the model.
        for u in fd.get("citation_urls") or []:
            if isinstance(u, str) and u.startswith("http"):
                urls.append(u)
        if not urls:
            continue
        conf = fd.get("confidence", "medium")
        if conf not in ("high", "medium", "low"):
            conf = "medium"
        out.append({"claim": str(fd["claim"])[:600],
                    "citation_urls": list(dict.fromkeys(urls)),
                    "confidence": conf, "section": section_title})
    return out, call, ""


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
            "conflict_markers": 0,
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
# The compose node
# ---------------------------------------------------------------------------

def compose_node(state) -> dict:
    """Write (or revise) the report from the evidence notes."""
    length = state.get("length") or "short"
    if not is_valid_length(length):
        length = "short"
    brief = ResearchBrief.model_validate(state.get("brief") or {})
    evidence = state.get("evidence") or []
    fetched = state.get("fetched") or []
    outline = (state.get("outline") or {}).get("sections") or []
    prior_sections = {s.get("title"): s for s in (state.get("sections") or [])}
    panel = state.get("panel_verdict") or {}
    revise_sections = set(panel.get("revise_sections") or [])
    user_notes = list(state.get("revision_notes") or [])

    calls: list[dict] = []
    errors: list[str] = []

    usable = [n for n in evidence if (n.get("claims") or [])
              and (int(n.get("relevance") or 0) >= 2 or _is_local_note(n))]
    if not usable:
        reason = ("No digested evidence note reached relevance >= 2 with "
                  "at least one claim.")
        return {
            "draft_md": _no_evidence_md(
                state.get("user_request") or "", length=length,
                fetched_count=len(fetched), reason=reason),
            "findings": [],
            "sections": [],
            "synthesis_outcome": "no_evidence",
            "synthesis_no_evidence_reason": reason,
            "messages": [{"role": "assistant",
                          "content": "⚠️ Compose: no usable evidence."}],
        }

    # ---- flat modes: one writer call --------------------------------
    if not outline:
        body, wcalls, werrs = _write_flat(state, brief, length=length,
                                          extra_notes=user_notes)
        calls.extend(wcalls)
        errors.extend(werrs)
        if not body.strip():
            errors.append("compose: flat writer returned nothing")
            body = "_(writer failed — see errors)_"
        sections = [{"title": "Report", "md": body}]
        findings, fcall, ferr = _extract_findings("Report", body, fetched)
        if fcall:
            calls.append(fcall)
        if ferr:
            errors.append(ferr)
        for i, f in enumerate(findings):
            f["id"] = f"f{i}"
        draft = body + "\n" + build_sources_block(body, fetched)
        out: dict[str, Any] = {
            "draft_md": draft,
            "sections": sections,
            "findings": findings,
            "synthesis_outcome": "ok",
            "model_calls": calls,
            "messages": [{"role": "assistant",
                          "content": (f"🧠 Composed flat report, "
                                      f"{len(findings)} findings.")}],
        }
        if errors:
            out["errors"] = errors
        return out

    # ---- sectioned modes: parallel section writers ------------------
    is_panel_revision = bool(revise_sections) and bool(prior_sections)
    is_user_revision = bool(state.get("revise_rounds")) and bool(
        prior_sections) and not revise_sections

    def _work(section: dict) -> tuple[str, str, dict | None, str]:
        title = section.get("title", "")
        if is_panel_revision and title not in revise_sections:
            prior = prior_sections.get(title)
            if prior and prior.get("md"):
                return title, prior["md"], None, ""   # keep untouched
        extra = list(user_notes) if (is_user_revision or not
                                     is_panel_revision) else []
        if is_panel_revision and title in revise_sections:
            extra = [n for n in (panel.get("notes") or [])][:8]
        notes_block = render_notes(evidence, section.get("sub_qs"))
        md, call, err = _write_section(section, brief, state, notes_block,
                                       length=length, extra_notes=extra)
        return title, md, call, err

    sections_out: list[dict] = []
    workers = max(1, min(_COMPOSE_CONCURRENCY, len(outline)))
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="argus-compose") as ex:
        for title, md, call, err in ex.map(_work, outline):
            if call:
                calls.append(call)
            if err:
                errors.append(err)
            if md:
                sections_out.append({"title": title, "md": md})

    if not sections_out:
        reason = "Every section writer failed."
        return {
            "draft_md": _no_evidence_md(
                state.get("user_request") or "", length=length,
                fetched_count=len(fetched), reason=reason),
            "findings": [],
            "sections": [],
            "synthesis_outcome": "synthesis_error",
            "synthesis_error": reason,
            "model_calls": calls,
            "errors": errors,
            "messages": [{"role": "assistant",
                          "content": "⚠️ Compose: all section writers "
                                     "failed."}],
        }

    body = "\n\n".join(f"## {s['title']}\n\n{s['md']}"
                       for s in sections_out)

    tldr, tcalls, terrs = _write_tldr(state, body)
    calls.extend(tcalls)
    errors.extend(terrs)
    tldr_head = "## Executive TL;DR" if length == "lecture" else "## TL;DR"
    if tldr.strip():
        body = f"{tldr_head}\n\n{tldr.strip()}\n\n{body}"

    # Findings per section (parallel, cheap).
    findings: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="argus-extract") as ex:
        futs = [ex.submit(_extract_findings, s["title"], s["md"], fetched)
                for s in sections_out]
        for f in futs:
            sec_findings, call, err = f.result()
            if call:
                calls.append(call)
            if err:
                errors.append(err)
            findings.extend(sec_findings)
    for i, f in enumerate(findings):
        f["id"] = f"f{i}"

    draft = body + "\n" + build_sources_block(body, fetched)
    mode = get_mode(length)
    out = {
        "draft_md": draft,
        "sections": sections_out,
        "findings": findings,
        "synthesis_outcome": "ok",
        "model_calls": calls,
        "messages": [{"role": "assistant",
                      "content": (f"🧠 Composed {len(sections_out)} sections, "
                                  f"{len(findings)} findings "
                                  f"({mode.label}).")}],
    }
    if mode.include_validated_assessment:
        out["validated_assessment"] = build_assessment(
            sections_out, findings, state.get("coverage") or {}, brief)
    if mode.include_appendix:
        out["lecture_appendix"] = {
            "methodology": (
                f"{len(state.get('queries') or [])} live queries across "
                f"Exa/DDGS/arXiv/GitHub; {len(fetched)} sources fetched; "
                f"{len(evidence)} digested into evidence notes over "
                f"{state.get('research_rounds') or 1} wave(s)."),
            "tool_calls": [f"{q.get('provider')}: {q.get('query')}"
                           for q in (state.get("queries") or [])][:30],
            "density_metrics": {
                "total_chars": len(draft),
                "total_findings": len(findings),
                "total_citations": sum(len(f.get("citation_urls") or [])
                                       for f in findings),
                "unique_sources": len(fetched),
                "avg_citations_per_finding": round(
                    sum(len(f.get("citation_urls") or [])
                        for f in findings) / len(findings), 2)
                if findings else 0.0,
            },
        }
    if errors:
        out["errors"] = errors
    return out


__all__ = ["outline_node", "compose_node", "render_notes",
           "build_sources_block", "build_assessment", "default_outline",
           "SECTION_SYSTEM", "FLAT_SYSTEM", "OUTLINE_SYSTEM",
           "EXTRACT_SYSTEM"]
