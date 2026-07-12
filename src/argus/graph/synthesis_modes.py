"""T7 — Synthesis length modes.

A single source of truth for the 5 output-length modes the user can
pick at plan-approval HITL time (``len:tldr`` ... ``len:lecture``).
Both the LLM prompt and the report-template builder consult this
module, so changing a mode's target length, target finding count, or
template name only requires touching one place.

Why a dedicated module (not a dict inside nodes.py)
---------------------------------------------------
The synthesizer node and the report_builder node both need to read the
*same* per-mode contract: what max_tokens to ask the LLM for, how many
findings we want, which markdown template to apply, whether to append a
validated_assessment block. Centralising the contract makes it easy
to:
- assert in tests that ``len:lecture`` produces more findings than
  ``len:short`` (a regression that "just bump max_tokens" would mask);
- add a new mode (e.g. ``len:executive_brief``) by editing one dict
  entry rather than grepping nodes.py;
- surface the contract to the planner / HITL keyboard layer so the
  keyboard labels and the synthesizer agree on what the user is buying.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# Re-exported from state.py for convenience — modules that only need
# the mode contract shouldn't have to import the TypedDict module.
Length = Literal["tldr", "short", "medium", "long", "lecture"]


@dataclass(frozen=True)
class SynthesisMode:
    """Per-length-mode contract.

    Attributes
    ----------
    key
        Canonical length key (``"tldr"`` etc.). Matches the callback
        data after ``len:`` and the value carried in ``state["length"]``.
    label
        Human-readable label for the HITL button and progress messages.
    target_chars_min / target_chars_max
        Soft character budget for the rendered markdown report. The
        synthesizer LLM is prompted to land inside this range; the
        report_builder can warn if the actual draft is far outside it
        (e.g. lecture came back at 1500 chars -> probably short mode
        was meant).
    target_findings
        How many ``Finding`` items the synthesizer should aim for.
        Acts as a structural prompt hint, *not* a hard cap.
    max_tokens
        ``max_tokens`` for the synthesizer LLM call. Sized to roughly
        cover the upper target char count plus JSON overhead.
    temperature
        ``temperature`` for the synthesizer LLM call. Lower for short
        factual modes, slightly higher for lecture (more prose-y).
    template
        Which markdown template to apply to the synthesizer output.
        ``"minimal"``      - single TL;DR paragraph
        ``"flat"``         - current behaviour: TL;DR + numbered findings
        ``"sectioned"``    - adds ## Background / ## Current state / ##
                             Open problems / ## Sources headings
        ``"lecture"``      - full Parts I-IV + References + Appendix
    include_validated_assessment
        If True, the synthesizer LLM is also asked to return a
        ``validated_assessment`` block (per-section confidence +
        open_challenges) and the report_builder surfaces it on the
        title page. Long + lecture always include this; shorter modes
        skip it (Albert's bar: validated assessment is a deep-mode
        affordance).
    include_appendix
        If True, the report_builder appends a methodology / tool-calls
        / density-metrics appendix. Lecture only - it has room.
    """

    key: Length
    label: str
    target_chars_min: int
    target_chars_max: int
    target_findings: int
    max_tokens: int
    temperature: float
    template: Literal["minimal", "flat", "sectioned", "lecture"]
    include_validated_assessment: bool = False
    include_appendix: bool = False


# ---------------------------------------------------------------------------
# SYNTHESIS_MODES - the only place the 5 modes are defined.
# ---------------------------------------------------------------------------
# Order matters: short -> long. Albert's HITL keyboard reads labels in
# this order; "len:tldr" -> "len:lecture" goes shortest -> longest.
# ---------------------------------------------------------------------------

SYNTHESIS_MODES: dict[Length, SynthesisMode] = {
    "tldr": SynthesisMode(
        key="tldr",
        label="TL;DR",
        target_chars_min=80,
        target_chars_max=400,
        target_findings=0,         # no per-finding structure - single paragraph
        max_tokens=400,
        temperature=0.2,
        template="minimal",
    ),
    "short": SynthesisMode(
        key="short",
        # 2026-07-12: the default mode was too thin (4 findings, ~900
        # chars). Bumped to a fuller default while staying "short":
        # 6 findings, ~600-1600 chars, ~2000 tokens. Medium/Long/Lecture
        # remain the deep-dive options at the plan gate.
        label="Short",
        target_chars_min=600,
        target_chars_max=1600,
        target_findings=6,
        max_tokens=2000,
        temperature=0.3,
        template="flat",
    ),
    "medium": SynthesisMode(
        key="medium",
        label="Medium",
        target_chars_min=3000,
        target_chars_max=6500,
        target_findings=8,
        max_tokens=3500,
        temperature=0.3,
        template="sectioned",
    ),
    "long": SynthesisMode(
        key="long",
        label="Long",
        target_chars_min=10000,
        target_chars_max=16000,
        target_findings=12,
        max_tokens=7000,
        temperature=0.4,
        template="sectioned",
        include_validated_assessment=True,
    ),
    "lecture": SynthesisMode(
        key="lecture",
        label="Lecture",
        target_chars_min=14000,
        target_chars_max=24000,
        target_findings=15,
        max_tokens=9000,
        temperature=0.4,
        template="lecture",
        include_validated_assessment=True,
        include_appendix=True,
    ),
}


def get_mode(length: str | None) -> SynthesisMode:
    """Look up the contract for a length key. Unknown / None -> short.

    Returning the ``short`` mode for an unknown key (rather than
    raising) is intentional: ``state["length"]`` can be missing if the
    pre-T7 SQLite checkpoint is loaded and we want the *behaviour* to
    match the prior ship rather than 500-ing the bot.
    """
    if length in SYNTHESIS_MODES:
        return SYNTHESIS_MODES[length]  # type: ignore[index]
    return SYNTHESIS_MODES["short"]


def is_valid(length: str | None) -> bool:
    return length in SYNTHESIS_MODES


__all__ = [
    "Length",
    "SynthesisMode",
    "SYNTHESIS_MODES",
    "get_mode",
    "is_valid",
]