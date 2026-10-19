"""Argus LangGraph node implementations — shared nodes.

The research pipeline lives in dedicated modules (brief.py, scout.py,
research.py, outline.py, compose.py, insight.py, panel.py). This module
keeps the nodes that frame it: intake, report_builder, deliver + the
extend/revise preps, and the quick-answer path. Each node is a function
(state) -> partial state update; no black-box prebuilt agents.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from .. import llm, quarantine
from ..citations import SourceRegistry, sanitize_report, verify_citations
from ..engines.registry import summarize_calls
from ..sources_block import sanitize_sources_block
from ..tools import markdown_to_pdf
from . import precision, run_config
from .assemble import assemble_draft
from .evidence_export import DEGRADATION_LINE, write_evidence
from .grounding import check_grounding, grounding_banner
from .research import _round
from .state import ArgusState

logger = logging.getLogger("argus.nodes")

_ANALYSIS = "Analysis"        # = insight.ANALYSIS_SECTION


# ---------------------------------------------------------------------------
# intake
# ---------------------------------------------------------------------------

INTAKE_SYSTEM = """You are the intake classifier for Argus, a research bot.

Given the user's raw request, decide:
- mode: "quick" for short factual / definitional questions; "deep" for
  anything that wants a report with sources.
- refined_query: a clean version of the user's request to feed the planner.

Return ONLY valid JSON matching the schema:
{"mode": "quick" | "deep", "refined_query": "..."}
"""


def _parse_intake(text: str) -> tuple[str, str]:
    data = _parse_json_obj(text, require_any=("mode",))
    if data.get("mode") not in ("quick", "deep"):
        raise ValueError(f"mode {data.get('mode')!r} is neither 'quick' nor 'deep'")
    return data["mode"], str(data.get("refined_query") or "").strip()


def intake_node(state: ArgusState, runtime: Runtime | None = None) -> dict:
    """Classify the request as quick vs deep and clean it up
    (``intake.classify``; degrade → ``default_classification``: deep, the raw
    request)."""
    rc = run_config.require(state)
    raw = state.get("user_request") or ""
    message = {"role": "user", "content": raw, "ts": _ts()}
    try:
        res = llm.complete("intake.classify", [SystemMessage(content=INTAKE_SYSTEM),
                                               HumanMessage(content=raw[:4000])],
                           engines=rc.engines, temperature=0.0, max_tokens=200, validate=_parse_intake)
    except llm.EngineDegraded as e:
        logger.warning("intake.classify degraded (%s)", e)
        if runtime is not None:
            runtime.stream_writer({"kind": "degrade", "node": "intake",
                                   "data": {"role": e.role, "degradation": e.degradation, "reason": str(e)}})
        return {"mode": "deep", "user_request": raw, "model_calls": [a.model_dump() for a in e.attempts],
                "errors": [f"[r{_round(state)}] degrade: intake.classify → {e.degradation} ({e})"],
                "messages": [message]}
    mode, refined = res.parsed
    return {"mode": mode, "user_request": refined or raw, "model_calls": [r.model_dump() for r in res.records],
            "messages": [message]}


# ---------------------------------------------------------------------------
# report_builder
# ---------------------------------------------------------------------------

def _integrity(md: str, fetched: list[dict]) -> str:
    """The v3 citation-integrity passes: URLs that resolve to no fetched
    source are stripped (``citations``), dangling ``[N]`` Sources slots
    dropped (``sources_block``). They raise on failure — never skipped."""
    registry = SourceRegistry()
    registry.add_from_fetched(fetched)
    verified = verify_citations(md, registry)
    sanitized = sanitize_report(verified.verified_report)
    if verified.removed_citations or sanitized.removed:
        logger.info("citation integrity: stripped %d unregistered URL(s), %d sanitized URL(s)",
                    len(verified.removed_citations), len(sanitized.removed))
    block = sanitize_sources_block(sanitized.cleaned_text)
    if block.dropped_count:
        logger.info("sources block: dropped %d dangling slot(s)", block.dropped_count)
        return block.cleaned_text
    return sanitized.cleaned_text


def report_builder_node(state: ArgusState) -> dict:
    """The deliverable (v4 spec §3 WP-C).

    1. Cut-not-flag: the panel's flagged findings and every finding citing
       only unknown source ids are cut from their section or the analysis
       (``quarantine.apply_cuts``: whole sentences; a section left empty is
       dropped); a flag that cannot be located stays in the body and is
       listed as such. Findings come back with their status.
    2. The deterministic TL;DR over what was kept (``tldr_md``), then
       ``assemble_draft``: TL;DR → sections → Analysis → appendices (cut,
       unlocated, uncited specifics) → Sources over all of them. Without
       sections (no usable evidence) the draft is the body.
    3. Title block with the engines / recency / research lines; the
       grounding banner and the "⚠ Not fact-checked" / "⚠ Degraded run"
       banners above the body; the degradation notes by round after
       Sources; the citation-integrity passes.
    4. report.md, report.pdf (ReportLab; a failure is an ``[r<k>] pdf:``
       line and a degradation note, and report.md is written again with
       it), evidence.json and metadata.json; then the report-preview gate.
    """
    from ..config import get_settings
    from .report_builder_helpers import (
        TLDR_LABEL, build_sidecar_metadata, build_tldr, degraded_run, info_lines, merge_reviewer_into_assessment,
        not_fact_checked, now_iso, render_degradation_notes, render_lecture_appendix, render_title_block,
        sidecar_to_json)
    from .synthesis_modes import get_mode

    rc = run_config.require(state)
    mode = get_mode(rc.length)
    rnd = _round(state)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic = re.sub(r"[^A-Za-z0-9_-]+", "_", state["user_request"])[:50] or "report"
    folder = get_settings().reports_root / f"{stamp}_{topic}_{rc.length}"
    folder.mkdir(parents=True, exist_ok=True)
    md_path, pdf_path, evidence_path = folder / "report.md", folder / "report.pdf", folder / "evidence.json"
    fetched = state.get("fetched") or []
    pv = state.get("panel_verdict") or {}
    model_calls = state.get("model_calls") or []
    new_errors: list[str] = []

    # 1. the cuts
    sections = state.get("sections") or []
    analysis = state.get("analysis")
    mds = {s["title"]: s["md"] for s in sections}
    if analysis:
        mds[_ANALYSIS] = analysis["md"]
    reasons = (pv.get("judge_verdicts") or {}).get("flag_reasons") or {}
    flagged = {i: reasons.get(i) or "not supported by its sources" for i in pv.get("flagged_finding_ids") or []}
    findings, cut_mds, cut_rows, unlocated = quarantine.apply_cuts(state.get("findings") or [], mds, flagged,
                                                                   len(fetched))
    appendices = [quarantine.render_cut_appendix(cut_rows), quarantine.render_unlocated_appendix(unlocated)]

    # 2. the TL;DR and the body
    tldr_md = None
    tldr = {"source": "omitted", "items": []}
    if sections:
        kept_sections = [{**s, "md": cut_mds[s["title"]]} for s in sections if cut_mds[s["title"]].strip()]
        analysis_md = cut_mds.get(_ANALYSIS) or ""
        items, source = build_tldr(analysis=analysis, analysis_md=analysis_md, sections=kept_sections,
                                   findings=findings)
        tldr = {"source": source, "items": items}
        if items:
            tldr_md = TLDR_LABEL + "\n\n" + "\n".join(f"- {i}" for i in items)
        else:
            new_errors.append(f"[r{rnd}] tldr: omitted — no kept takeaway or finding")
        if rc.panel.precision_mode == "deterministic":
            appendices.append(precision.render_appendix(precision.scan(
                "\n\n".join([*(s["md"] for s in kept_sections), analysis_md]))))
        body = assemble_draft(tldr=tldr_md, sections=kept_sections, analysis_md=analysis_md or None,
                              appendices=appendices, fetched=fetched,
                              flat=not (state.get("outline") or {}).get("sections"),
                              tldr_heading="Executive TL;DR" if rc.length == "lecture" else "TL;DR")
    else:
        body = "\n\n".join(x for x in [(state.get("draft_md") or "# (empty draft)").rstrip(), *appendices] if x)

    # 3. the title block, the banners, the integrity passes
    kept = [f for f in findings if f.get("status") == "kept"]
    validated = merge_reviewer_into_assessment(state.get("validated_assessment") or {}, state.get("review_verdict"))
    title_block = render_title_block(
        topic=state["user_request"], length=rc.length, length_label=mode.label, stamp=now_iso(),
        n_findings=len(kept), n_sources=len(fetched), revision_rounds=int(state.get("revision_rounds") or 0),
        validated_assessment=validated,
        info_lines=info_lines(state=state, rc=rc, panel=pv,
                              reroute_rate=summarize_calls(model_calls)["reroute_rate"]))
    unchecked = not_fact_checked(rc, pv, findings)
    degraded_items = degraded_run(model_calls, [*(state.get("errors") or []), *new_errors])
    banners = "".join(f"> ⚠ {b}\n\n" for b in (
        f"Not fact-checked — {unchecked}" if unchecked else None,
        f"Degraded run — {degraded_items}" if degraded_items else None) if b)
    content = _integrity(title_block + grounding_banner(check_grounding(state)) + banners + body, fetched)
    footer = f"\n\n---\n\n_Generated by Argus • {now_iso()} • mode: {mode.label}_\n"
    appendix = render_lecture_appendix(state.get("lecture_appendix") or {}, density_metrics={},
                                       total_chars=len(body)) if mode.include_appendix else ""

    def render() -> str:
        notes = render_degradation_notes([*(state.get("errors") or []), *new_errors], rnd)
        return content.rstrip() + ("\n\n" + notes if notes else "") + footer + appendix

    # 4. the files
    md_text = render()
    md_path.write_text(md_text, encoding="utf-8")
    try:
        # the human-readable topic (not the slug): PDF metadata only — the
        # body's own "# {topic}" line is the one visible title.
        markdown_to_pdf(md_text, str(pdf_path), title=state["user_request"])
        pdf = str(pdf_path)
    except Exception as e:      # ReportLab has no fallback: recorded, and report.md still ships
        logger.warning("PDF render failed: %s", e)
        new_errors.append(f"[r{rnd}] pdf: {type(e).__name__}: {e}")
        md_path.write_text(render(), encoding="utf-8")
        pdf = None

    errors = [*(state.get("errors") or []), *new_errors]
    spend = state.get("search_spend") or {}
    engines = summarize_calls(model_calls, errors, exa_usd=spend.get("exa_usd"))
    degraded = bool(degraded_items) or int(pv.get("judged") or 0) < int(pv.get("total") or 0)
    write_evidence(evidence_path, state, findings=findings, errors=errors, degraded=degraded, tldr=tldr,
                   engines=engines, generated_at=now_iso())
    rc_dump = rc.model_dump(mode="json")
    rc_dump["engines"].pop("roles", None)
    meta = build_sidecar_metadata(
        topic=state["user_request"], thread_id=state.get("thread_id"), user_id=state.get("user_id"),
        length=rc.length, length_label=mode.label, stamp=stamp, n_findings=len(kept), n_sources=len(fetched),
        revision_rounds=int(state.get("revision_rounds") or 0), validated_assessment=validated,
        lecture_appendix=state.get("lecture_appendix"), model_calls=model_calls, engines=engines,
        run_config=rc_dump, evidence_path=str(evidence_path),
        degradations=[e for e in errors if DEGRADATION_LINE.match(e)], degraded=degraded,
        exa_usd=spend.get("exa_usd"))
    (folder / "metadata.json").write_text(sidecar_to_json(meta), encoding="utf-8")

    paths = {"folder": str(folder), "md": str(md_path), "pdf": pdf, "evidence": str(evidence_path)}
    result: dict[str, Any] = {
        "report_paths": paths,
        "findings": findings,
        "tldr_md": tldr_md,
        "hitl": {"pending": True, "kind": "report_preview", "ctx": {"paths": dict(paths)}},
        "messages": [{"role": "assistant",
                      "content": f"📝 Report ready ({mode.label}). Awaiting your sign-off."}],
    }
    if new_errors:
        result["errors"] = new_errors
    return result


# ---------------------------------------------------------------------------
# deliver + Phase 2 HITL "extend research" loop
# ---------------------------------------------------------------------------

MAX_EXTEND_ROUNDS = 2


MAX_REVISE_ROUNDS = 2


def _will_extend(state: ArgusState) -> bool:
    """True when the user asked to extend and we're under the round cap."""
    return bool(state.get("extend_requested")) and \
        int(state.get("extend_rounds") or 0) < MAX_EXTEND_ROUNDS


def _will_revise(state: ArgusState) -> bool:
    """True when the user asked to revise at the preview gate and we're
    under the user-revise round cap (distinct from the reviewer's own
    reflexion budget)."""
    return bool(state.get("revision_requested")) and \
        int(state.get("revise_rounds") or 0) < MAX_REVISE_ROUNDS


def deliver_node(state: ArgusState) -> dict:
    """Records the paths; the Telegram bot layer does the actual send.

    If the user asked to extend or revise at the preview gate (and we're
    under the respective round cap), this is a no-op pass-through —
    ``route_after_deliver`` sends the run back through ``extend_prep`` or
    ``revise_prep`` instead of finalising.
    """
    if _will_extend(state):
        return {"messages": [{"role": "assistant",
                              "content": "↪️ Extending research…"}]}
    if _will_revise(state):
        return {"messages": [{"role": "assistant",
                              "content": "🔁 Revising report…"}]}
    paths = state.get("report_paths") or {}
    return {
        "hitl": {"pending": False},
        "messages": [{"role": "assistant", "content": (
            f"✅ Delivered. Folder: {paths.get('folder')}")}],
    }


def route_after_deliver(state: ArgusState) -> str:
    """Loop back to deepen research, revise the report, or end the run.

    Extend wins over revise if somehow both are set (deepening subsumes
    a re-synthesis)."""
    if _will_extend(state):
        return "extend"
    if _will_revise(state):
        return "revise"
    return "end"


def revise_prep_node(state: ArgusState) -> dict:
    """Prepare a user-requested revision pass (report-preview gate).

    Clears the request flag, bumps the user-revise counter, resets the
    round's counters (``revision_rounds``, ``compose_runs``) and the stale
    panel verdict, and marks every outline section changed, so ``compose``
    rewrites every section with the user's notes (``revision_notes``, which
    the bot appended before resuming; the writers read them newest first).
    """
    rnd = int(state.get("revise_rounds") or 0) + 1
    out = {
        "revision_requested": False,
        "revise_rounds": rnd,
        "revision_rounds": 0,   # a fresh panel budget for the revision round
        "compose_runs": 0,      # per-round counter (spec §2.4)
        "panel_verdict": None,  # no stale targeted rewrite
        "messages": [{"role": "assistant",
                      "content": f"🔁 Applying your revision (round {rnd})…"}],
    }
    outline = state.get("outline") or {}
    if outline.get("sections"):
        out["outline"] = {**outline, "changed_sections": [s["title"] for s in outline["sections"]]}
    return out


def extend_prep_node(state: ArgusState) -> dict:
    """Prepare a user-requested Extend pass (report-preview gate or a
    plain /continue): clear the flag, bump ``extend_rounds``, reset the
    round's counters (``revision_rounds``, ``compose_runs``) and the stale
    panel verdict — counters and flags only, like ``revise_prep``. v4:
    ``research_node`` runs the Extend pass itself (a fresh allowance,
    ``research.extend_querygen``, the widened search wave), so no search
    happens here; the run rejoins at ``research`` without re-triggering the
    plan gate.

    ``append_only``: /continue after /append ingests EXACTLY the user's
    appended sources (already merged into ``state["sources"]`` by the
    bot before the resume) without running fresh searches.
    """
    rnd = int(state.get("extend_rounds") or 0) + 1
    base = {
        "extend_requested": False,
        "extend_rounds": rnd,
        "revision_rounds": 0,
        "compose_runs": 0,
        "panel_verdict": None,
    }
    if state.get("append_only"):
        # NB: append_only stays True — research_node consumes it (it
        # must see the flag to skip fresh search waves) and clears it.
        srcs = state.get("sources") or []
        return {
            **base,
            "sources": srcs,
            "messages": [{"role": "assistant",
                          "content": (f"➕ Continuing with appended sources "
                                      f"only (round {rnd}) — "
                                      f"{len(srcs)} total.")}],
        }
    return {
        **base,
        "messages": [{"role": "assistant",
                      "content": f"🔎 Extending research (round {rnd})…"}],
    }


# ---------------------------------------------------------------------------
# quick_answer
# ---------------------------------------------------------------------------

QUICK_SYSTEM = ("You are Argus, a research assistant. Answer the user's question "
                "concisely and accurately. If you don't know, say so. No fake citations.")


def quick_answer_node(state: ArgusState) -> dict:
    """Cheap single-shot answer (``quick.answer``, ``raise``): grounded in the
    model's training only (no fabrication guarantees)."""
    rc = run_config.require(state)
    res = llm.complete("quick.answer", [SystemMessage(content=QUICK_SYSTEM),
                                        HumanMessage(content=state["user_request"][:4000])],
                       engines=rc.engines, temperature=0.3, max_tokens=600)
    return {
        "quick_answer": res.text,
        "model_calls": [r.model_dump() for r in res.records],
        "messages": [{"role": "assistant", "content": res.text}],
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Robust JSON parsing lives in jsonx.py (v3); legacy names kept for
# existing imports/tests.
from .jsonx import (  # noqa: E402
    parse_json_obj as _parse_json_obj,
    repair_json as _repair_json,
    salvage_findings as _salvage_findings,
)


def _ts() -> str:
    return datetime.now().astimezone().isoformat()
