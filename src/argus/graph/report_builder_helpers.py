"""T7 — report_builder helpers (title page, validated assessment, appendix).

These functions are pure (no I/O, no LLM calls) so they can be unit
tested without spinning up the full graph. ``report_builder_node``
calls them to compose the final markdown before writing it to disk.

Why separate from nodes.py
--------------------------
The title-page + validated-assessment block is content that
``report_builder_node`` writes *to the markdown* (not just to
state). It has to render cleanly in:

- the markdown itself (read back by Telegram via ``report:send``),
- the PDF (via the intel-stack Chromium renderer), and
- the ``metadata.json`` sidecar (so a future audit can reconstruct
  what the user was promised vs what was delivered).

Keeping the formatter here lets us:
- snapshot-test the rendered markdown for each length mode,
- snapshot-test the sidecar JSON,
- have ``test_bot.py`` assert the length label is visible in the
  title block (so a regression where the title page is dropped would
  fail loudly).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any


def _section_label(name: str) -> str:
    """Trim noisy prefixes so the title page line stays tight.

    E.g. ``"Part I — Background"`` -> ``"Part I"`` so the line
    ``Part I — Background: confidence=high`` doesn't repeat "Background".
    """
    if "—" in name:
        return name.split("—", 1)[0].strip().rstrip(":")
    if "-" in name:
        return name.split("-", 1)[0].strip().rstrip(":")
    return name


def _confidence_emoji(level: str) -> str:
    return {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}.get(
        level.lower(), "[?]",
    )


def render_title_block(*, topic: str, length: str, length_label: str,
                        stamp: str, n_findings: int, n_sources: int,
                        revision_rounds: int,
                        validated_assessment: dict | None) -> str:
    """Render the markdown title-page / metadata block.

    The block is prepended to ``draft_md`` so a reader opening the
    file sees a clean header before any prose. In Telegram we strip
    this block before sending the body so the user doesn't see the
    metadata twice; the PDF keeps it as a "title page".

    Parameters
    ----------
    topic
        Raw user request; the document's subject.
    length
        Canonical key (``"lecture"`` etc.).
    length_label
        Human label (``"Lecture"``).
    stamp
        ISO timestamp string for the title block.
    n_findings
        Number of ``Finding`` items the synthesizer produced.
    n_sources
        Number of unique fetched sources.
    revision_rounds
        How many reviewer-driven revisions the run went through.
    validated_assessment
        Optional dict from the synthesizer; we surface only the
        overall_relevancy + a one-line per-section confidence summary.
    """
    parts: list[str] = []
    parts.append(f"# {topic}")
    parts.append("")
    parts.append(
        f"_Mode: **{length_label}** • {stamp} • "
        f"findings: {n_findings} • sources: {n_sources} • "
        f"revisions: {revision_rounds}_"
    )
    parts.append("")
    parts.append("> **Quality summary**")
    parts.append(">")
    if validated_assessment:
        overall = validated_assessment.get("overall_relevancy", "medium")
        parts.append(f"> Overall relevancy: {_confidence_emoji(overall)} {overall}")
        density = validated_assessment.get("knowledge_density") or {}
        cpc = density.get("citations_per_claim")
        if cpc is not None:
            parts.append(f"> Citations per claim: {cpc}")
        cm = density.get("conflict_markers")
        if cm is not None and cm > 0:
            parts.append(f"> Conflict markers: {cm}")
        for sec in validated_assessment.get("sections") or []:
            name = sec.get("name") or "?"
            conf = sec.get("confidence", "medium")
            oc = sec.get("open_challenges") or []
            label = _section_label(name)
            parts.append(f"> - {label}: {_confidence_emoji(conf)} {conf}"
                         + (f" — {len(oc)} open challenge(s)"
                            if oc else ""))
        reviewer_uns = validated_assessment.get("reviewer_unsupported") or []
        reviewer_fab = validated_assessment.get("reviewer_fabrication_flags") or []
        if reviewer_uns:
            parts.append(f"> Reviewer flagged {len(reviewer_uns)} "
                         f"unsupported claim(s).")
        if reviewer_fab:
            parts.append(f"> Reviewer flagged {len(reviewer_fab)} "
                         f"fabrication risk(s).")
    else:
        parts.append("> _Validated assessment not generated "
                     "(short / tldr / medium modes)._")
    parts.append("")
    parts.append("---")
    parts.append("")
    return "\n".join(parts)


def render_lecture_appendix(appendix: dict, *, density_metrics: dict,
                             total_chars: int) -> str:
    """Render the lecture-mode appendix block (methodology + density).

    Returns markdown suitable for appending after the main report.
    Empty string if appendix is missing.
    """
    if not appendix:
        return ""
    parts: list[str] = []
    parts.append("")
    parts.append("## Appendix")
    parts.append("")
    methodology = appendix.get("methodology") or ""
    if methodology:
        parts.append("### Methodology")
        parts.append("")
        parts.append(methodology)
        parts.append("")
    tool_calls = appendix.get("tool_calls") or []
    if tool_calls:
        parts.append("### Tool calls log")
        parts.append("")
        for tc in tool_calls:
            parts.append(f"- {tc}")
        parts.append("")
    density = density_metrics or appendix.get("density_metrics") or {}
    parts.append("### Density metrics")
    parts.append("")
    parts.append("| Metric | Value |")
    parts.append("|---|---|")
    if total_chars:
        parts.append(f"| Total characters | {total_chars} |")
    if density.get("total_findings") is not None:
        parts.append(f"| Total findings | {density.get('total_findings')} |")
    if density.get("total_citations") is not None:
        parts.append(f"| Total citations | {density.get('total_citations')} |")
    if density.get("unique_sources") is not None:
        parts.append(f"| Unique sources | {density.get('unique_sources')} |")
    if density.get("avg_citations_per_finding") is not None:
        parts.append(f"| Avg citations / finding | "
                     f"{density.get('avg_citations_per_finding')} |")
    if density.get("claims_per_section"):
        cps = density["claims_per_section"]
        cps_str = ", ".join(f"{k}={v}" for k, v in cps.items())
        parts.append(f"| Claims / section | {cps_str} |")
    parts.append("")
    return "\n".join(parts)


def merge_reviewer_into_assessment(validated: dict,
                                    review_verdict: dict | None) -> dict:
    """Merge reviewer's unsupported_claims + fabrication_flags into the
    validated_assessment dict so the title-page summary reflects the
    adversarial pass.

    Returns a new dict (does not mutate the input). If no review
    verdict is available, returns ``validated`` unchanged.
    """
    if not review_verdict:
        return validated
    out = dict(validated or {})
    out["reviewer_unsupported"] = list(
        out.get("reviewer_unsupported") or []
    ) + list(review_verdict.get("unsupported_claims") or [])
    out["reviewer_fabrication_flags"] = list(
        out.get("reviewer_fabrication_flags") or []
    ) + list(review_verdict.get("fabrication_flags") or [])
    return out


def build_sidecar_metadata(*, topic: str, thread_id: str | None,
                            user_id: int | None, length: str,
                            length_label: str, stamp: str,
                            n_findings: int, n_sources: int,
                            revision_rounds: int,
                            validated_assessment: dict | None,
                            lecture_appendix: dict | None,
                            model_calls: list[dict]) -> dict[str, Any]:
    """Build the ``metadata.json`` sidecar dict.

    Two consumers:
    - the kanban reviewer auditing the run,
    - a future Argus dashboard that wants to compare mode coverage
      across runs.

    Includes the full validated_assessment + lecture_appendix blocks
    so an offline reviewer can reconstruct everything without the MD.
    """
    return {
        "topic": topic,
        "thread_id": thread_id,
        "user_id": user_id,
        "length": length,
        "length_label": length_label,
        "stamp": stamp,
        "n_findings": n_findings,
        "n_sources": n_sources,
        "revision_rounds": revision_rounds,
        "validated_assessment": validated_assessment or {},
        "lecture_appendix": lecture_appendix or {},
        "model_calls": model_calls,
    }


def sidecar_to_json(meta: dict[str, Any]) -> str:
    return json.dumps(meta, indent=2, ensure_ascii=False)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


__all__ = [
    "render_title_block",
    "render_lecture_appendix",
    "merge_reviewer_into_assessment",
    "build_sidecar_metadata",
    "sidecar_to_json",
    "now_iso",
]