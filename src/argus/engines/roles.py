"""Engine role ids and failure-policy classes (v4 spec §2.1) — contract, no behaviour.

Unknown role anywhere → ``ValueError`` (never silent); ``schema.py``
validates role ids and ``on_exhausted`` policies against these tables.
"""
from __future__ import annotations

from typing import Literal

Tier = Literal["cheap", "strong", "judge"]

ROLES: tuple[str, ...] = (
    "intake.classify", "brief.draft", "scout.querygen", "research.rerank", "research.digest",
    "research.followup_querygen", "research.extend_querygen", "outline.plan", "compose.section", "compose.flat",
    "compose.findings_extract", "compose.analyst", "panel.grounding", "panel.coverage", "quick.answer",
    "media.tag", "compose.calc",
)
ROLE_TIERS: dict[str, Tier] = {
    "intake.classify": "cheap", "brief.draft": "strong", "scout.querygen": "cheap", "research.rerank": "cheap",
    "research.digest": "cheap", "research.followup_querygen": "cheap", "research.extend_querygen": "cheap",
    "outline.plan": "strong", "compose.section": "strong", "compose.flat": "strong", "compose.findings_extract": "cheap",
    "compose.analyst": "strong", "panel.grounding": "judge", "panel.coverage": "strong", "quick.answer": "cheap",
    "media.tag": "cheap", "compose.calc": "strong"}
JSON_ROLES = frozenset({"intake.classify", "brief.draft", "scout.querygen", "research.rerank", "research.digest",
                        "research.followup_querygen", "research.extend_querygen", "outline.plan",
                        "compose.findings_extract", "panel.grounding", "panel.coverage", "media.tag",
                        "compose.calc"})
# Failure policy classes (validated in schema.py):
SKIPPABLE_ROLES = frozenset({"research.digest", "compose.findings_extract", "compose.section",
                             "panel.grounding", "panel.coverage"})  # on_exhausted="skip_item": drop that item (source / section / judge batch)
ALL_SKIPPED_RAISES = frozenset({"research.digest", "compose.section", "compose.findings_extract"})   # the node raises EngineError instead of shipping a hollow report (rule in §2.3)
DEGRADABLE_ROLES: dict[str, str] = {                               # on_exhausted="degrade": the NAMED deterministic degradation
    "intake.classify": "default_classification",                     # mode=deep, refined_query = the raw request
    "scout.querygen": "subquestion_text_queries",                    # today's fallback_queries()
    "research.rerank": "keyword_triage",                             # keyword-overlap triage for that wave
    "outline.plan": "one_section_per_subquestion_roundrobin_notes",  # today's default_outline() + round-robin note allocation
    "research.followup_querygen": "skip_followup_wave",
    "research.extend_querygen": "skip_extend_wave",                  # a user Extend that degrades says so in the preview
    "compose.analyst": "omit_analysis",
    "compose.calc": "omit_worked_numbers",                            # graph/calc.py: no "Worked numbers" appendix
}
MUST_RAISE_ROLES = frozenset({"brief.draft", "compose.flat", "quick.answer",   # on_exhausted must be "raise"
                              "media.tag"})                                      # no tags rather than invented ones
