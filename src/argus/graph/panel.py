"""v3 review panel — three parallel judges, deterministic merge.

The tripartite judgment mechanism: three judges with DIFFERENT jobs (and
different model tiers/families where the proxy allows) evaluate the
draft independently, so no single model's blind spots decide the
verdict:

- **grounding** (judge tier): does every finding trace to evidence?
  Produces flagged finding ids — the exact contract the quarantine
  machinery consumes.
- **coverage** (strong tier, family-diverse from judge via
  ``pick_strong_and_judge``): does the report answer the brief's
  sub-questions and success criteria? Names weak sections.
- **precision** (cheap tier): scans prose for uncited numbers / dates /
  names — the fabrication surface.

Merge is DETERMINISTIC (no LLM): grounding failures always force a
revision (bounded); otherwise 2-of-3 revise votes do. The merged verdict
is written both as ``panel_verdict`` (v3) and ``review_verdict``
(v2-compatible — report_builder + bot progress read it unchanged).
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage

from .. import llm
from .jsonx import parse_json_obj
from .state import PanelVerdict

logger = logging.getLogger("argus.panel")

GROUNDING_SYSTEM = """You are an adversarial grounding auditor for a
research report. You get the report's findings (claims with citation
URLs and ids) and the evidence notes (per-source claims with quotes)
they must trace to.

Flag a finding when:
- its claim is NOT supported by any claim/quote of a source it cites;
- it contains a number/date/name absent from the cited evidence;
- it cites a source whose notes contradict it (unless marked as a
  conflict).

Be strict but fair: condensed wording is fine, invented specifics are
not. Return ONLY JSON:
{"verdict": "pass|revise",
 "flagged_finding_ids": ["f2", ...],
 "unsupported_claims": ["exact claim text", ...],
 "notes": ["actionable note", ...]}
Verdict is "revise" iff at least one finding is flagged.
"""

COVERAGE_SYSTEM = """You audit a research report for COVERAGE against its
brief. You get the brief (sub-questions + success criteria) and the
report's section titles with short excerpts.

Judge: does the report substantively address each sub-question and
success criterion? Name sections that are thin, off-topic, or missing.

Return ONLY JSON:
{"verdict": "pass|revise",
 "weak_sections": ["exact section title", ...],
 "missing": ["what's not covered", ...],
 "notes": ["actionable note", ...]}
Verdict is "revise" only for MATERIAL gaps (an unanswered sub-question),
not stylistic wishes.
"""

PRECISION_SYSTEM = """You scan a research report for fabrication risk:
sentences stating specific numbers, dates, model names, or benchmarks
WITHOUT a [n] citation marker nearby.

Return ONLY JSON:
{"verdict": "pass|revise",
 "suspicious": ["the exact suspicious sentence", ...],
 "notes": ["actionable note", ...]}
Verdict is "revise" only if 3+ specific uncited facts appear.
"""


def _run_judge(name: str, tier: str, system: str, human: str,
               *, model_override: str | None = None
               ) -> tuple[str, dict, dict | None, str]:
    """Run one judge. Returns (name, parsed, model_call, error)."""
    try:
        chat = llm.chat_for_tier(tier, temperature=0.1, max_tokens=900,
                                 model_override=model_override)
        resp = llm.invoke_with_retry(chat, [
            SystemMessage(content=system),
            HumanMessage(content=human[:13000]),
        ])
        call = llm.record_from_response(
            tier, model_override or llm.resolve_tier(tier), resp
        ).model_dump()
        data = parse_json_obj(resp.content, require_any=("verdict",))
        return name, data, call, ""
    except Exception as e:
        logger.warning("panel: %s judge failed (%s)", name, e)
        return name, {}, None, f"panel: {name} judge failed ({e})"


def _sections_of_findings(findings: list[dict],
                          flagged_ids: list[str]) -> list[str]:
    flagged = set(flagged_ids)
    return sorted({f.get("section") or "" for f in findings
                   if f.get("id") in flagged and f.get("section")})


def panel_node(state) -> dict:
    """Run the three judges in parallel and merge deterministically."""
    findings = state.get("findings") or []
    evidence = state.get("evidence") or []
    sections = state.get("sections") or []
    brief = state.get("brief") or {}
    draft = state.get("draft_md") or ""

    # A no-evidence / error draft skips review — report_builder surfaces
    # the classified failure; judging a failure block is noise.
    if state.get("synthesis_outcome") in ("no_evidence", "synthesis_error") \
            or not findings:
        verdict = PanelVerdict(verdict="pass")
        return {
            "panel_verdict": verdict.model_dump(),
            "review_verdict": _to_review_verdict(verdict),
            "messages": [{"role": "assistant",
                          "content": "🔬 Panel skipped (no findings to "
                                     "audit)."}],
        }

    # ---- judge inputs ------------------------------------------------
    findings_json = json.dumps(
        [{"id": f.get("id"), "claim": f.get("claim"),
          "citation_urls": f.get("citation_urls"),
          "section": f.get("section")} for f in findings[:25]],
        indent=1)[:6000]
    ev_lines = []
    for n in evidence:
        claims = "; ".join(
            f"{c.get('text')}" + (f" [q: {c.get('quote')[:80]}]"
                                  if c.get("quote") else "")
            for c in (n.get("claims") or [])[:4])
        ev_lines.append(f"[{n.get('source_id')}] {n.get('source_url')}: "
                        f"{claims}")
    ev_block = "\n".join(ev_lines)[:6500]

    sec_lines = []
    for s in sections:
        sec_lines.append(f"### {s.get('title')}\n"
                         f"{(s.get('md') or '')[:400]}")
    sec_block = "\n\n".join(sec_lines)[:6000]
    brief_block = json.dumps({
        "sub_questions": [
            (sq.get("q") if isinstance(sq, dict) else str(sq))
            for sq in (brief.get("sub_questions") or [])],
        "success_criteria": brief.get("success_criteria") or [],
    }, indent=1)[:2500]

    grounding_human = (f"FINDINGS:\n{findings_json}\n\n"
                       f"EVIDENCE NOTES:\n{ev_block}\n\n"
                       "Return the audit JSON.")
    coverage_human = (f"BRIEF:\n{brief_block}\n\n"
                      f"REPORT SECTIONS:\n{sec_block}\n\n"
                      "Return the coverage JSON.")
    precision_human = (f"REPORT:\n{draft[:9000]}\n\n"
                       "Return the precision JSON.")

    # Family diversity: judge model differs from strong where possible.
    try:
        _strong, _judge = llm.pick_strong_and_judge()
    except Exception:
        _strong = _judge = None

    jobs = [
        ("grounding", "judge", GROUNDING_SYSTEM, grounding_human, _judge),
        ("coverage", "strong", COVERAGE_SYSTEM, coverage_human, _strong),
        ("precision", "cheap", PRECISION_SYSTEM, precision_human, None),
    ]
    results: dict[str, dict] = {}
    calls: list[dict] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=3,
                            thread_name_prefix="argus-panel") as ex:
        futs = [ex.submit(_run_judge, n, t, s, h, model_override=m)
                for (n, t, s, h, m) in jobs]
        for f in futs:
            name, data, call, err = f.result()
            results[name] = data
            if call:
                calls.append(call)
            if err:
                errors.append(err)

    # ---- deterministic merge ------------------------------------------
    g = results.get("grounding") or {}
    c = results.get("coverage") or {}
    p = results.get("precision") or {}
    judge_verdicts = {
        "grounding": g.get("verdict", "pass"),
        "coverage": c.get("verdict", "pass"),
        "precision": p.get("verdict", "pass"),
    }
    known_ids = {f.get("id") for f in findings}
    flagged_ids = [i for i in (g.get("flagged_finding_ids") or [])
                   if i in known_ids]
    revise_votes = sum(1 for v in judge_verdicts.values() if v == "revise")
    grounding_fails = judge_verdicts["grounding"] == "revise" and flagged_ids
    verdict = "revise" if (grounding_fails or revise_votes >= 2) else "pass"

    notes: list[str] = []
    notes.extend(str(n) for n in (g.get("notes") or [])[:5])
    notes.extend(str(n) for n in (c.get("notes") or [])[:4])
    notes.extend(f"missing: {m}" for m in (c.get("missing") or [])[:3])
    notes.extend(str(n) for n in (p.get("notes") or [])[:3])

    section_titles = {s.get("title") for s in sections}
    revise_secs = [t for t in (c.get("weak_sections") or [])
                   if t in section_titles]
    revise_secs += [t for t in _sections_of_findings(findings, flagged_ids)
                    if t in section_titles and t not in revise_secs]
    if verdict == "revise" and not revise_secs:
        # Judges didn't localize — revise everything rather than nothing.
        revise_secs = sorted(section_titles - {""})

    rounds = int(state.get("revision_rounds") or 0) + (
        1 if verdict == "revise" else 0)
    pv = PanelVerdict(
        verdict=verdict,
        judge_verdicts=judge_verdicts,
        notes=notes,
        flagged_finding_ids=flagged_ids,
        unsupported_claims=[str(x) for x in
                            (g.get("unsupported_claims") or [])[:8]],
        fabrication_flags=[str(x) for x in (p.get("suspicious") or [])[:8]],
        revise_sections=revise_secs,
    )
    out = {
        "panel_verdict": pv.model_dump(),
        "review_verdict": _to_review_verdict(pv),
        "revision_notes": (list(state.get("revision_notes") or [])
                           + notes if verdict == "revise" else
                           list(state.get("revision_notes") or [])),
        "revision_rounds": rounds,
        "model_calls": calls,
        "messages": [{"role": "assistant",
                      "content": (f"🔬 Panel: {verdict} "
                                  f"(grounding={judge_verdicts['grounding']}, "
                                  f"coverage={judge_verdicts['coverage']}, "
                                  f"precision={judge_verdicts['precision']})")}],
    }
    if errors:
        out["errors"] = errors
    return out


def _to_review_verdict(pv: PanelVerdict) -> dict:
    """v2 ReviewVerdict-compatible dict (report_builder + bot contract)."""
    return {
        "verdict": pv.verdict,
        "notes": list(pv.notes),
        "unsupported_claims": list(pv.unsupported_claims),
        "fabrication_flags": list(pv.fabrication_flags),
        "flagged_finding_ids": list(pv.flagged_finding_ids),
    }


def route_after_panel(state) -> str:
    v = (state.get("panel_verdict") or {}).get("verdict")
    rounds = int(state.get("revision_rounds") or 0)
    max_rounds = int(os.environ.get("ARGUS_MAX_REVISIONS", "3"))
    if v == "pass" or rounds >= max_rounds:
        return "report_builder"
    return "compose"


__all__ = ["panel_node", "route_after_panel", "GROUNDING_SYSTEM",
           "COVERAGE_SYSTEM", "PRECISION_SYSTEM"]
