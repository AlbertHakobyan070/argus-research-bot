"""Argus LangGraph node implementations — shared nodes.

v3: the research pipeline lives in dedicated modules (brief.py,
scout.py, research.py, compose.py, panel.py). This module keeps the
nodes that frame the pipeline: intake, report_builder, deliver + the
extend/revise preps, and the quick-answer path. Each node is a function
(state) -> partial state update; no black-box prebuilt agents.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .. import llm
from ..tools import markdown_to_pdf
from .state import ArgusState, ResearchBrief

logger = logging.getLogger("argus.nodes")


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


def intake_node(state: ArgusState) -> dict:
    """Classify request as quick vs deep and clean it up."""
    raw = state.get("user_request") or ""
    chat = llm.chat_for_tier("cheap", temperature=0.0, max_tokens=200)
    resp = llm.invoke_with_retry(chat, [
        SystemMessage(content=INTAKE_SYSTEM),
        HumanMessage(content=raw[:4000]),
    ])
    rec = llm.record_from_response("cheap", "cheap", resp)
    try:
        parsed = _parse_json_obj(resp.content)
    except Exception:
        parsed = {"mode": "deep" if len(raw) > 80 else "quick",
                  "refined_query": raw.strip()}
    mode = parsed.get("mode", "deep")
    if mode not in ("quick", "deep"):
        mode = "deep"
    return {
        "mode": mode,
        "user_request": parsed.get("refined_query", raw).strip() or raw,
        "model_calls": [rec.model_dump()],
        "messages": [{"role": "user", "content": raw, "ts": _ts()}],
    }


# ---------------------------------------------------------------------------
# report_builder
# ---------------------------------------------------------------------------

def report_builder_node(state: ArgusState) -> dict:
    """T7 — mode-aware report builder.

    Responsibilities:
    1. Pick the markdown template based on ``state["length"]``
       (delegated to ``_draft_md_from_findings`` if the synthesizer
       failed and returned no draft_md).
    2. Prepend a T7 title-page block (mode, date, quality summary,
       per-section confidence) so the user can see what they're getting
       before scrolling.
    3. Append a lecture appendix (methodology + tool calls + density)
       when ``state["length"] == "lecture"``.
    4. Merge reviewer-flagged unsupported_claims / fabrication_flags
       into the validated_assessment so the title page reflects the
       adversarial pass.
    5. Write ``metadata.json`` sidecar with the same fields + the full
       validated_assessment + lecture_appendix blobs.
    6. Render markdown -> PDF via ReportLab (fast, no browser) with
       fallback to intel-stack Chromium for the styled HTML route.
    7. Trigger the report-preview HITL gate.
    """
    from ..config import get_settings
    from .synthesis_modes import get_mode
    from .report_builder_helpers import (
        build_sidecar_metadata, merge_reviewer_into_assessment,
        now_iso, render_lecture_appendix, render_title_block,
        sidecar_to_json,
    )
    s = get_settings()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    length = state.get("length") or "short"
    mode = get_mode(length)
    topic = re.sub(r"[^A-Za-z0-9_-]+", "_", state["user_request"])[:50] or "report"
    folder = s.reports_root / f"{stamp}_{topic}_{length}"
    folder.mkdir(parents=True, exist_ok=True)
    md_path = folder / "report.md"
    pdf_path = folder / "report.pdf"

    # Merge reviewer verdict into the validated_assessment so the
    # title-page summary is honest about adversarial findings.
    validated = state.get("validated_assessment") or {}
    validated = merge_reviewer_into_assessment(
        validated, state.get("review_verdict"),
    )

    # Compose the final markdown: title block + body + (optional) appendix.
    n_findings = len(state.get("findings") or [])
    n_sources = len(state.get("fetched") or [])
    revision_rounds = int(state.get("revision_rounds") or 0)

    # P4 - quarantine reviewer-flagged fabrication claims BEFORE we
    # compose the body so the title-block flags report honest numbers.
    #
    # Two paths:
    #   (a) Structured (default, 2026-07-09): route findings by
    #       ``finding_id`` through ``post_synthesis.quarantine_by_id``.
    #       This is the structural fix for the §4.1 fragmentation
    #       bug - when the same flagged claim appears in multiple
    #       sections, the structured path routes it exactly once
    #       instead of partially matching substrings and producing
    #       orphaned prose fragments.
    #   (b) Legacy string-quarantine fallback: when findings lack
    #       ids (legacy checkpoint replay) or the structured path
    #       raises, fall back to the substring-matching legacy
    #       ``quarantine_flagged_claims``. Removed when the legacy
    #       module is retired (post 5.4 of the argus-async handoff).
    review_verdict = state.get("review_verdict") or {}
    body = state.get("draft_md") or "# (empty draft)"
    used_structured_quarantine = False
    if review_verdict:
        # The reviewer may flag findings by id (new structured
        # contract) OR by claim text (legacy substring contract). We
        # support both - preferring id-based when available.
        flagged_ids = review_verdict.get("flagged_finding_ids") or []
        flagged_claim_text = (
            list(review_verdict.get("unsupported_claims") or [])
            + list(review_verdict.get("fabrication_flags") or [])
        )
        try:
            from ..post_synthesis import (
                assign_finding_ids,
                quarantine_by_id as _quarantine_by_id,
                render_quarantine_appendix as _render_quarantine_appendix,
            )
            findings_records = assign_finding_ids(state.get("findings") or [])
            if findings_records and flagged_ids:
                kept, q, d, unmatched = _quarantine_by_id(
                    findings_records, flagged_ids, mode="move",
                )
                n_quarantined = len(q)
                n_still_unmatched = len(unmatched)
                # Append the structured appendix. The body itself is
                # left intact (the synthesizer already wrote it from
                # the surviving findings); the appendix is the only
                # place where flagged claims appear.
                if q:
                    body = body + _render_quarantine_appendix(q)
                used_structured_quarantine = True
                logger.info(
                    "quarantine(structured): moved %d finding(s) to "
                    "appendix (%d unmatched); kept %d",
                    n_quarantined, n_still_unmatched, len(kept),
                )
        except Exception as _e:
            logger.warning(
                "structured quarantine skipped (%s); falling back to "
                "legacy string-quarantine", _e,
            )

        if not used_structured_quarantine and flagged_claim_text:
            try:
                from ..quarantine import quarantine_flagged_claims as                     _quarantine_flagged_claims
                _qres = _quarantine_flagged_claims(
                    body,
                    review_verdict.get("unsupported_claims") or [],
                    review_verdict.get("fabrication_flags") or [],
                    mode="move",
                )
                n_quarantined = len(_qres.quarantined)
                n_still_unmatched = len(_qres.still_unmatched)
                body = _qres.cleaned_text
                if n_quarantined:
                    logger.info(
                        "quarantine(legacy): moved %d flagged "
                        "sentence(s) to appendix (%d unmatched)",
                        n_quarantined, n_still_unmatched,
                    )
            except Exception as _e:
                logger.warning(
                    "quarantine pass skipped (%s); body unchanged", _e,
                )


    title_block = render_title_block(
        topic=state["user_request"],
        length=length,
        length_label=mode.label,
        stamp=now_iso(),
        n_findings=n_findings,
        n_sources=n_sources,
        revision_rounds=revision_rounds,
        validated_assessment=validated,
    )
    appendix = ""
    if mode.include_appendix:
        appendix = render_lecture_appendix(
            state.get("lecture_appendix") or {},
            density_metrics={},
            total_chars=len(body),
        )

    # P5 — grounding banner prepended ABOVE the body so the reader sees
    # the warning BEFORE the asserted content. Empty string when
    # grounded - the helper handles that.
    try:
        from .grounding import (
            check_grounding as _check_grounding,
            grounding_banner as _grounding_banner,
        )
        _grounding_banner = _grounding_banner(_check_grounding(state))
    except Exception as _e:
        logger.warning("grounding banner skipped (%s); report unchanged", _e)
        _grounding_banner = ""

    md_text = (
        title_block
        + _grounding_banner
        + body
        + "\n\n---\n\n_Generated by Argus • "
        + now_iso()
        + f" • mode: {mode.label}_\n"
        + appendix
    )

    # Citation integrity pass (lifted from NVIDIA AI-Q Blueprint).
    # Every URL in the final report must resolve to a URL that was
    # actually fetched by a researcher tool. Anything else is stripped,
    # along with its inline citation, and recorded in the audit trail.
    # This is the structural fix for the 2026-07-08 hallucinated-URL bug.
    try:
        from ..citations import (
            SourceRegistry as _CitationRegistry,
            verify_citations as _verify_citations,
            sanitize_report as _sanitize_report,
        )
        from ..sources_block import (
            sanitize_sources_block as _sanitize_sources_block,
        )
        _reg = _CitationRegistry()
        _reg.add_from_fetched(state.get("fetched") or [])
        _verified = _verify_citations(md_text, _reg)
        _sanitized = _sanitize_report(_verified.verified_report)
        md_text = _sanitized.cleaned_text
        # P3 — drop dangling [N] source slots left behind by
        # verify_citations (which only strips the URL, not the label).
        # Also renumbers body refs so nothing points at an empty slot.
        try:
            _src_block = _sanitize_sources_block(md_text)
            if _src_block.dropped_count:
                md_text = _src_block.cleaned_text
                logger.info(
                    "sources block: dropped %d dangling slot(s), "
                    "renumbered %d ref(s)",
                    _src_block.dropped_count,
                    len(_src_block.renumbered),
                )
        except Exception as _e:
            logger.warning(
                "sources-block post-pass skipped (%s); report unchanged",
                _e,
            )
        if _verified.removed_citations or _sanitized.removed:
            logger.info(
                "citation integrity: stripped %d unregistered URL(s), %d sanitized URL(s)",
                len(_verified.removed_citations),
                len(_sanitized.removed),
            )
    except Exception as _e:
        # Citation module failure must NOT block report delivery; degrade
        # gracefully and log. The user's bot should always produce a report.
        logger.warning("citation integrity pass skipped (%s); report unchanged", _e)

    md_path.write_text(md_text, encoding="utf-8")

    pdf_ok = False
    try:
        # markdown_to_pdf uses playwright via the intel-stack helper. That
        # helper imports playwright at runtime, so if it isn't in the
        # argus venv we delegate to the intel-stack venv python.
        try:
            # NB: pass the raw human-readable topic (not the filesystem
            # slug) — the renderer uses this only for PDF metadata /
            # title-page treatment, never as a second literal heading
            # (the body's own "# {topic}" line, written by
            # render_title_block, is the single source of the visible
            # title — see the 2026-07-16 double-title bug fix).
            markdown_to_pdf(md_text, str(pdf_path),
                            title=state["user_request"])
            pdf_ok = pdf_path.exists()
        except (ModuleNotFoundError, ImportError) as e:
            # Fallback: render via intel-stack's python + the _common helper.
            logger.info("argus venv missing playwright; delegating PDF render")
            import importlib.util
            import subprocess
            helper_path = (Path(os.environ.get(
                "INTEL_STACK_DIR",
                r"A:\Hermes\Agents\intel-stack\scripts")) / "_common.py")
            helper_path.parent.mkdir(parents=True, exist_ok=True)
            # We need a small wrapper: write the md to disk, call _common.
            tmp_py = folder / "_render_pdf.py"
            tmp_py.write_text(
                "import sys, os\n"
                f"sys.path.insert(0, r'{helper_path.parent}')\n"
                "os.environ.pop('PYTHONPATH', None)\n"
                f"from _common import markdown_to_pdf\n"
                f"markdown_to_pdf(open(r'{md_path}','r',encoding='utf-8').read(),\n"
                f"                  r'{pdf_path}', title=r'{topic}')\n",
                encoding="utf-8",
            )
            intel_py = os.environ.get(
                "INTEL_PYTHON",
                r"A:\Hermes\Agents\intel-stack\venv\Scripts\python.exe",
            )
            res = subprocess.run(
                [intel_py, str(tmp_py)],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "PYTHONPATH": ""},
            )
            if res.returncode != 0:
                logger.warning("PDF fallback failed: %s", res.stderr[-500:])
            pdf_ok = pdf_path.exists()
    except Exception as e:
        logger.warning("PDF render failed: %s", e)

    meta = build_sidecar_metadata(
        topic=state["user_request"],
        thread_id=state.get("thread_id"),
        user_id=state.get("user_id"),
        length=length,
        length_label=mode.label,
        stamp=stamp,
        n_findings=n_findings,
        n_sources=n_sources,
        revision_rounds=revision_rounds,
        validated_assessment=validated,
        lecture_appendix=state.get("lecture_appendix"),
        model_calls=state.get("model_calls") or [],
    )
    (folder / "metadata.json").write_text(
        sidecar_to_json(meta), encoding="utf-8",
    )
    return {
        "report_paths": {
            "folder": str(folder),
            "md": str(md_path),
            "pdf": str(pdf_path) if pdf_ok else None,
        },
        "hitl": {"pending": True, "kind": "report_preview",
                 "ctx": {"paths": {
                     "folder": str(folder),
                     "md": str(md_path),
                     "pdf": str(pdf_path) if pdf_ok else None,
                 }}},
        "messages": [{"role": "assistant",
                      "content": f"📝 Report ready ({mode.label}). "
                                 "Awaiting your sign-off."}],
    }


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
    panel's revision budget, and clears any stale panel verdict so
    ``compose`` runs in user-revision mode (rewrite every section with
    the user's notes) rather than a leftover targeted panel revision.
    The user's feedback is already in ``revision_notes`` (the bot
    appended it before resuming).
    """
    rnd = int(state.get("revise_rounds") or 0) + 1
    return {
        "revision_requested": False,
        "revise_rounds": rnd,
        "revision_rounds": 0,   # fresh panel budget for the revision
        "panel_verdict": None,  # v3: drop stale targeted-revision state
        "messages": [{"role": "assistant",
                      "content": f"🔁 Applying your revision (round {rnd})…"}],
    }


def extend_prep_node(state: ArgusState) -> dict:
    """Broaden the search and gather MORE sources for another deep pass.

    v3: runs one widened query wave (follow-up-style queries over ALL
    sub-questions) and merges the hits into ``sources``. The run then
    rejoins at ``research``, which fetches + digests only the NEW
    sources — the wave never re-triggers the plan gate.

    ``append_only``: /continue after /append ingests EXACTLY the user's
    appended sources (already merged into ``state["sources"]`` by the
    bot before the resume) without running fresh searches.
    """
    rnd = int(state.get("extend_rounds") or 0) + 1
    base = {
        "extend_requested": False,
        "extend_rounds": rnd,
        "revision_rounds": 0,
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
    from .research import followup_queries
    from .search_providers import run_query_wave

    brief = ResearchBrief.model_validate(state.get("brief") or {})
    gaps = list(range(len(brief.sub_questions)))
    queries, calls = followup_queries(brief, gaps,
                                      list(state.get("queries") or []))
    hits, errors = run_query_wave(queries)
    sources = list(state.get("sources") or [])
    known = {s.get("url") for s in sources}
    added = 0
    for h in hits:
        if h.get("url") and h["url"] not in known:
            sources.append(h)
            known.add(h["url"])
            added += 1
    out: dict[str, Any] = {
        **base,
        "sources": sources,
        "queries": list(state.get("queries") or []) + queries,
        "model_calls": calls,
        "messages": [{"role": "assistant",
                      "content": (f"🔎 Extending research (round {rnd}) — "
                                  f"+{added} new candidate source(s), "
                                  f"{len(sources)} total.")}],
    }
    if errors:
        out["errors"] = errors
    return out


# ---------------------------------------------------------------------------
# quick_answer
# ---------------------------------------------------------------------------

def quick_answer_node(state: ArgusState) -> dict:
    """Cheap single-shot answer. Still routes through FreeLLMAPI, so it
    is grounded in the model's training (no fabrication guarantees)."""
    chat = llm.chat_for_tier("cheap", temperature=0.3, max_tokens=600)
    resp = llm.invoke_with_retry(chat, [
        SystemMessage(content=(
            "You are Argus, a research assistant. Answer the user's question "
            "concisely and accurately. If you don't know, say so. No fake citations."
        )),
        HumanMessage(content=state["user_request"][:4000]),
    ])
    rec = llm.record_from_response("cheap", llm.resolve_tier("cheap"), resp)
    return {
        "quick_answer": resp.content,
        "model_calls": [rec.model_dump()],
        "messages": [{"role": "assistant", "content": resp.content}],
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