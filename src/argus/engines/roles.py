"""Engine role ids and failure-policy classes (v4 spec §2.1) — contract, no behaviour.

Unknown role anywhere → ``ValueError`` (never silent); ``schema.py``
validates role ids and ``on_exhausted`` policies against these tables.
"""
from __future__ import annotations

from typing import Literal

Tier = Literal["cheap", "strong", "judge"]

ROLES: tuple[str, ...] = (
    "intake.classify", "brief.draft", "scout.querygen", "research.rerank", "research.digest",
    "research.followup_querygen", "outline.plan", "compose.section", "compose.flat", "compose.tldr",
    "compose.findings_extract", "compose.analyst", "panel.grounding", "panel.coverage", "panel.precision",
    "quick.answer",
)
LEGACY_ROLES = ("legacy.cheap", "legacy.strong", "legacy.judge")     # Phases 1–2 only (wrappers); removed by WP-H
ROLE_TIERS: dict[str, Tier] = {
    "intake.classify": "cheap", "brief.draft": "strong", "scout.querygen": "cheap", "research.rerank": "cheap",
    "research.digest": "cheap", "research.followup_querygen": "cheap", "outline.plan": "strong",
    "compose.section": "strong", "compose.flat": "strong", "compose.tldr": "cheap", "compose.findings_extract": "cheap",
    "compose.analyst": "strong", "panel.grounding": "judge", "panel.coverage": "strong", "panel.precision": "cheap",
    "quick.answer": "cheap"}
JSON_ROLES = frozenset({"intake.classify", "brief.draft", "scout.querygen", "research.rerank", "research.digest",
                        "research.followup_querygen", "outline.plan", "compose.findings_extract", "panel.grounding",
                        "panel.coverage", "panel.precision"})
# Failure policy classes (validated in schema.py):
SKIPPABLE_ROLES = frozenset({"research.digest", "compose.findings_extract", "research.rerank", "compose.section",
                             "panel.grounding", "panel.coverage", "panel.precision"})  # on_exhausted="skip_item": drop that item (source / section / judge)
DEGRADABLE_ROLES: dict[str, str] = {                               # on_exhausted="degrade": the NAMED deterministic degradation
    "intake.classify": "default_classification",                     # mode=deep, refined_query = the raw request
    "scout.querygen": "subquestion_text_queries",                    # today's fallback_queries()
    "outline.plan": "one_section_per_subquestion_roundrobin_notes",  # today's default_outline() + round-robin note allocation
    "research.followup_querygen": "skip_followup_wave",
    "compose.tldr": "omit_tldr",
    "compose.analyst": "omit_analysis",
}
MUST_RAISE_ROLES = frozenset({"brief.draft", "compose.flat", "quick.answer"})   # on_exhausted must be "raise"
