"""Structured post-synthesis helpers (finding-id-keyed quarantine).

This module is the structural fix for the §4.1 bug: **quarantine
produces fragmented prose.** The legacy ``src/argus/quarantine.py``
operates on a flat ``draft_md`` string and substring-matches
reviewer-flagged claims. When the flagged claim text recurs across
multiple sections (TL;DR + Background + Current state, etc.), only the
first sentence moves and the others are partially moved (heading without
body) or left as orphaned fragments (body without heading). The fix is
to operate on ``finding_id`` — a stable per-run identifier assigned by
the synthesizer.

What this module provides
-------------------------
- :func:`assign_finding_ids` — raw finding dicts -> :class:`FindingRecord`
  with stable ids.
- :func:`quarantine_by_id` — quarantine decisions keyed by
  ``finding_id`` (no substring matching on prose).
- :func:`render_quarantine_appendix` — the ``## ⚠️`` appendix for the
  quarantined findings.

``report_builder_node`` calls all three. The legacy string-based
``quarantine.py`` stays for reviewer verdicts that flag claims by text
instead of by finding id (e.g. legacy checkpoints replayed after a bot
upgrade).

All functions are pure / I/O-free and unit-testable without spinning
up the graph or the LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Domain type — the minimal finding shape the structured quarantine uses.
# ---------------------------------------------------------------------------

@dataclass
class FindingRecord:
    """Minimal projection of :class:`argus.graph.state.Finding` used by
    the structured post-synthesis passes. Decoupled from the pydantic
    model so the post-passes don't depend on the full Argus state.

    ``id`` is REQUIRED here (the structured path rejects findings
    without an id — they cannot be routed). The synthesizer is
    responsible for assigning ids; the fallback below assigns them
    deterministically from position so a legacy finding list is still
    processable.
    """
    id: str
    claim: str
    citation_urls: list[str]
    confidence: str = "medium"
    section: str = ""


# ---------------------------------------------------------------------------
# Fallback id assignment for legacy findings.
# ---------------------------------------------------------------------------

def assign_finding_ids(findings: Iterable[dict]) -> list[FindingRecord]:
    """Convert raw finding dicts (from ``state["findings"]``) into
    :class:`FindingRecord` with stable ids.

    If a finding already has ``id`` we honour it. Otherwise we assign
    ``"f0"``, ``"f1"``, ... based on position so the rest of the
    pipeline can address findings by id even if the synthesizer
    didn't produce one.
    """
    out: list[FindingRecord] = []
    for i, f in enumerate(findings):
        if not isinstance(f, dict):
            continue
        fid = (f.get("id") or f"f{i}").strip() or f"f{i}"
        out.append(FindingRecord(
            id=fid,
            claim=(f.get("claim") or "").strip(),
            citation_urls=list(f.get("citation_urls") or []),
            confidence=f.get("confidence") or "medium",
            section=(f.get("section") or "").strip(),
        ))
    return out


# ---------------------------------------------------------------------------
# §4.1 fix — quarantine by finding_id, not substring on prose.
# ---------------------------------------------------------------------------

def quarantine_by_id(
    findings: list[FindingRecord],
    flagged_ids: Iterable[str],
    *,
    mode: str = "move",
) -> tuple[list[FindingRecord], list[FindingRecord], list[FindingRecord], list[str]]:
    """Apply quarantine decisions to a list of findings.

    Parameters
    ----------
    findings
        All findings from the synthesizer (already id-assigned).
    flagged_ids
        Iterable of ``finding_id`` strings flagged by the reviewer
        (either ``unsupported_claims`` or ``fabrication_flags`` — both
        are treated equivalently; the reviewer distinguished them by
        severity, we don't care).
    mode
        ``"move"`` (default): keep the finding's claim but move it to
        the ``## ⚠️`` appendix. ``"remove"``: drop entirely.

    Returns
    -------
    (kept, quarantined, dropped, still_unmatched)

        - ``kept`` = findings that stay in the body
        - ``quarantined`` = findings moved to the appendix (mode="move")
        - ``dropped`` = findings removed entirely (mode="remove")
        - ``still_unmatched`` = flag strings that did not match any
          finding id (logged so the title block can warn)

    Why this is the structural fix
    ------------------------------
    The legacy string-based quarantine split prose on
    ``[.!?]\\s`` and substring-matched flagged spans. When the same
    span recurred in multiple sections (TL;DR + Background + Current
    state), only the first sentence moved; other copies either lost
    their heading (heading-without-body) or lost their trailing
    punctuation (body-without-heading) — producing the orphan
    fragments observed in the 03:30 long-mode run. By matching on
    finding_id instead of prose, we route each finding exactly once
    regardless of how many prose sections the synthesizer repeated
    it across.
    """
    if mode not in ("move", "remove"):
        raise ValueError(f"mode must be 'move' or 'remove', got {mode!r}")

    flag_set = {f for f in (flagged_ids or []) if f}
    kept: list[FindingRecord] = []
    quarantined: list[FindingRecord] = []
    dropped: list[FindingRecord] = []
    still_unmatched: list[str] = []

    for f in findings:
        if f.id in flag_set:
            if mode == "move":
                quarantined.append(f)
            else:
                dropped.append(f)
        else:
            kept.append(f)

    # Surface flag strings we couldn't route — usually means the
    # reviewer's verdict referenced a finding that wasn't in the final
    # set (e.g. removed during a revision round).
    seen_ids = {f.id for f in findings}
    for flag in flag_set:
        if flag not in seen_ids:
            still_unmatched.append(flag)

    return kept, quarantined, dropped, still_unmatched


# ---------------------------------------------------------------------------
# Appendix renderer — the quarantined findings as markdown.
# ---------------------------------------------------------------------------

def render_quarantine_appendix(findings: list[FindingRecord]) -> str:
    """Render the ``## ⚠️ Unverified / flagged claims`` appendix from
    the quarantined findings (structured). Replaces the legacy
    string-quarantine appendix.

    Its text says the findings "were removed from the body". That held
    when the body was rebuilt from the kept findings; v3's
    report_builder appends this to compose's unchanged draft, so the
    flagged claims are still in the body as well."""
    if not findings:
        return ""
    parts = ["", "---", "",
             "## ⚠️ Unverified / flagged claims", "",
             "_The following findings were flagged by the reviewer as "
             "unsupported or potentially fabricated. They were removed "
             "from the body and preserved here so the review trail is "
             "auditable. Do not rely on these claims without independent "
             "verification._", ""]
    for f in findings:
        cites = " ".join(f"[{i+1}]" for i in range(len(f.citation_urls)))
        line = f.claim
        if cites:
            line = f"{line} {cites}"
        parts.append(f"- (id={f.id}) {line}")
    parts.append("")
    return "\n".join(parts)


__all__ = [
    "FindingRecord",
    "assign_finding_ids",
    "quarantine_by_id",
    "render_quarantine_appendix",
]
