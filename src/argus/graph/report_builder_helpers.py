"""report_builder helpers: title page, validated assessment, appendix (T7)
and the v4 trust surface (WP-C): the title-block info lines, the "⚠ Not
fact-checked" / "⚠ Degraded run" banners, the deterministic TL;DR, the
degradation notes by round and the metadata.json sidecar.

These functions are pure (no I/O, no LLM calls) so they can be unit
tested without spinning up the full graph. ``report_builder_node``
calls them to compose the final markdown before writing it to disk.

Why separate from nodes.py
--------------------------
The title-page + validated-assessment block is content that
``report_builder_node`` writes *to the markdown* (not just to
state). It has to render cleanly in:

- the markdown itself (read back by Telegram via ``report:send``),
- the PDF (via ReportLab, ``tools.markdown_to_pdf``), and
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
import re
from collections import Counter
from datetime import datetime
from typing import Any

from .. import quarantine
from .assemble import ANALYSIS_SECTION, CITE_RE
from .panel import failure_reason

TLDR_LABEL = "_TL;DR — compiled from the report's own cited, judged findings_"
_TLDR_ITEMS = 4
# The per-item roles (spec §2.3: invoked once per source, section or judging batch) and what a
# skipped item is, for the degraded-run banner. A single-call role's skip (the coverage review)
# is a degradation note only.
_ITEM_ROLES = {
    "research.digest": "sources could not be digested",
    "compose.section": "sections could not be written",
    "compose.findings_extract": "finding extractions failed",
    "panel.grounding": "grounding batches could not be judged",
}
_SKIP_LINE = re.compile(r"^(?:\[r\d+\]\s*)?skip:\s*([a-z0-9_.]+)\s")
# The degradation notes: the bench's skip / degrade / legacy lines, plus the other visible losses and
# the cite: repairs (spec B1; notes, never counted as degradations).
_NOTE_LINE = re.compile(r"^(?:\[r(\d+)\]\s*)?(?:skip|degrade|legacy|pdf|tldr|cancelled|compose|cite):")
_RECENCY = {"d": "past day", "w": "past week", "m": "past month", "y": "past year"}


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
                        validated_assessment: dict | None,
                        info_lines: list[str] = ()) -> str:
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
    info_lines
        v4 lines under the mode line (``info_lines()``: engines, recency,
        research counters).
    """
    parts: list[str] = []
    parts.append(f"# {topic}")
    parts.append("")
    parts.append(
        f"_Mode: **{length_label}** • {stamp} • "
        f"findings: {n_findings} • sources: {n_sources} • "
        f"revisions: {revision_rounds}_"
    )
    for line in info_lines:
        parts.append("")
        parts.append(line)
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
                            model_calls: list[dict],
                            engines: dict,
                            run_config: dict | None = None,
                            evidence_path: str | None = None,
                            degradations: list[str] = (),
                            degraded: bool = False,
                            exa_usd: float | None = None) -> dict[str, Any]:
    """Build the ``metadata.json`` sidecar dict.

    Two consumers:
    - the kanban reviewer auditing the run,
    - a future Argus dashboard that wants to compare mode coverage
      across runs.

    Includes the full validated_assessment + lecture_appendix blocks
    so an offline reviewer can reconstruct everything without the MD.
    v4: ``engines`` is ``registry.summarize_calls()``; ``run_config`` the
    RunConfig without ``engines.roles``; ``degradations`` the skip /
    degrade / legacy lines; ``degraded`` whether a banner marks the run.
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
        "engines": engines,
        "run_config": run_config,
        "evidence_path": evidence_path,
        "degradations": list(degradations),
        "degraded": degraded,
        "exa_usd": exa_usd,
    }


def info_lines(*, state, rc, panel: dict, reroute_rate: float | None) -> list[str]:
    """The title block's v4 lines: engines, recency (when applied), research counters."""
    reroute = "n/a" if reroute_rate is None else f"{reroute_rate * 100:.0f} %"
    lines = [f"_Engines: {rc.engines.profile} · reroute {reroute} · judge diversity "
             f"{'⚠' if panel.get('judge_not_diverse') else 'ok'} · judged {int(panel.get('judged') or 0)}/"
             f"{int(panel.get('total') or 0)} · warnings {len(rc.engines.warnings)}_"]
    recency = (state.get("search_spend") or {}).get("recency")
    if recency:
        since = f" (sources published since {recency['start_date']})" if recency.get("start_date") else ""
        lines.append(f"_Recency: {_RECENCY.get(recency.get('window'), recency.get('window'))}{since}_")
    coverage = [int(v) for v in (state.get("coverage") or {}).values()]
    cov = (f"coverage min {min(coverage)} / mean {round(sum(coverage) / len(coverage), 1):g}"
           if coverage else "coverage n/a")
    lines.append(f"_Research: {len(state.get('fetched') or [])} sources · {cov} · gap waves "
                 f"{int(state.get('gap_waves') or 0)} · compose runs {int(state.get('compose_runs') or 0)}_")
    return lines


def unjudged_sections(parts: list[tuple[str, str]], findings: list[dict], errors: list[str]) -> list[str]:
    """§2.3: ``section "<title>" was not fact-checked (<reason>)`` for every
    (title, final markdown) — the sections and the Analysis — that cites
    ``[n]`` but has no finding: the reason of its last ``skip:
    compose.findings_extract <title>`` line, else "no findings extracted"."""
    have = {f.get("section") for f in findings}
    out = []
    for title, md in parts:
        if title in have or not CITE_RE.search(md):
            continue
        skip = re.compile(rf"^(?:\[r\d+\]\s*)?skip: compose\.findings_extract {re.escape(title)} \(")
        lines = [e for e in errors if skip.match(e)]
        out.append(f'section "{title}" was not fact-checked '
                   f'({failure_reason(lines[-1]) if lines else "no findings extracted"})')
    return out


def not_fact_checked(rc, panel: dict, findings: list[dict], unjudged: list[str] = ()) -> str | None:
    """The "⚠ Not fact-checked" banner text, or None: the grounding judge is
    off, failed outright or left some findings unjudged (spec §2.8 R1), and
    the ``unjudged`` sections (``unjudged_sections``), each named (§2.3)."""
    if not (rc.panel.enabled and rc.panel.grounding):
        return "the grounding judge is off" if findings else None
    parts = []
    judged, total = int(panel.get("judged") or 0), int(panel.get("total") or 0)
    if judged < total:
        reasons = ", ".join(dict.fromkeys(e.split(": ", 1)[1] for e in panel.get("skipped_judges") or []
                                          if e.startswith("grounding") and ": " in e)) or "unknown"
        parts.append(f"the grounding judge failed ({reasons})" if judged == 0
                     else f"{total - judged} of {total} findings were not judged ({reasons})")
    return "; ".join([*parts, *unjudged]) or None


def degraded_run(model_calls: list[dict], errors: list[str], panel: dict) -> str | None:
    """The "⚠ Degraded run" banner text, or None: every per-item role that
    skipped half or more of its items (spec §2.3), e.g. "14 of 28 sources
    could not be digested (429)". Items = its ok calls plus its skip lines;
    the grounding batches are counted over the last panel pass only — the
    panel verdict's skipped batches of ``judge_verdicts["grounding_batches"]``."""
    ok = Counter(c.get("role") for c in model_calls
                 if c.get("ok", True) and c.get("retry_reason") != "diversity_rerun")
    skips: dict[str, list[str]] = {}
    for line in errors:
        m = _SKIP_LINE.match(line)
        if m and m.group(1) in _ITEM_ROLES and m.group(1) != "panel.grounding":
            skips.setdefault(m.group(1), []).append(failure_reason(line))
    items = {role: (len(skips.get(role) or []), ok[role]) for role in _ITEM_ROLES}
    grounding = [e.split(": ", 1)[1] for e in panel.get("skipped_judges") or [] if e.startswith("grounding batch")]
    if grounding:
        skips["panel.grounding"] = grounding
        items["panel.grounding"] = (len(grounding), (panel.get("judge_verdicts") or {})["grounding_batches"]
                                    - len(grounding))
    parts = []
    for role, what in _ITEM_ROLES.items():
        k, n_ok = items[role]
        if k and 2 * k >= k + n_ok:
            parts.append(f"{k} of {k + n_ok} {what} ({', '.join(dict.fromkeys(skips[role]))})")
    return "; ".join(parts) or None


def build_tldr(*, analysis: dict | None, analysis_md: str, sections: list[dict], findings: list[dict]
               ) -> tuple[list[str], str]:
    """R2, after the cuts: ``(items, source)``. With an analysis, its first
    ≤ 4 kept takeaways with their labels: still in the (cut) analysis, and
    stated by analysis findings that are all ``kept`` — a finding whose
    sentence is in the takeaway (or the takeaway in it), or whose claim
    ``twins`` it (by their source ids); never one ``cut`` or ``unlocated``,
    nor a takeaway no finding states. Without one, or with no takeaway kept,
    the first kept finding sentence still in each section (≤ 4). Both by
    normalised containment in the cut markdown, never a fuzzy match; else
    none ("omitted")."""
    if analysis:
        mine = [f for f in findings if f.get("section") == ANALYSIS_SECTION]

        def states(f: dict, t: dict) -> bool:
            s, text = f.get("sentence") or "", t["text"]
            return (quarantine.contains(s, text) or quarantine.contains(text, s)
                    or quarantine.twins(f.get("claim") or "", text, f.get("source_ids") or (),
                                        t.get("source_ids") or ()))

        def kept(t: dict) -> bool:
            own = [f for f in mine if states(f, t)]
            return bool(own) and all(f.get("status") == "kept" for f in own)

        items = [f"{t['text']} — confidence **{t['confidence']}**" for t in analysis.get("takeaways") or []
                 if quarantine.contains(t["text"], analysis_md) and kept(t)][:_TLDR_ITEMS]
        if items:
            return items, "takeaways"
    items = []
    for s in sections:
        first = next((f.get("sentence") or f.get("claim") for f in findings
                      if f.get("section") == s["title"] and f.get("status") == "kept"
                      and quarantine.contains(f.get("sentence") or f.get("claim") or "", s["md"])), None)
        if first:
            items.append(" ".join(first.split()))
    items = items[:_TLDR_ITEMS]
    return items, "sections" if items else "omitted"


def render_degradation_notes(errors: list[str], rnd: int) -> str:
    """"## Degradation notes": this round's lines in full (an untagged line
    takes this round's tag), earlier rounds collapsed to a count; "" when
    there are none."""
    rows = []
    for line in errors:
        m = _NOTE_LINE.match(line)
        if m:
            rows.append((int(m.group(1) or rnd), line if m.group(1) else f"[r{rnd}] {line}"))
    if not rows:
        return ""
    lines = ["## Degradation notes", ""] + [f"- {text}" for r, text in rows if r == rnd]
    earlier = Counter(r for r, _ in rows if r != rnd)
    if earlier:
        lines.append("- earlier rounds: " + ", ".join(
            f"r{r} — {n} note{'' if n == 1 else 's'}" for r, n in sorted(earlier.items())))
    return "\n".join(lines)


def sidecar_to_json(meta: dict[str, Any]) -> str:
    return json.dumps(meta, indent=2, ensure_ascii=False)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


__all__ = [
    "TLDR_LABEL",
    "info_lines",
    "not_fact_checked",
    "unjudged_sections",
    "degraded_run",
    "build_tldr",
    "render_degradation_notes",
    "render_title_block",
    "render_lecture_appendix",
    "merge_reviewer_into_assessment",
    "build_sidecar_metadata",
    "sidecar_to_json",
    "now_iso",
]