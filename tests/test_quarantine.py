"""Tests for src/argus/quarantine.py (P4: enforce reviewer flags).

Run with:
    PYTHONPATH='' ./venv/Scripts/python.exe -m pytest tests/test_quarantine.py -q
"""
from __future__ import annotations

import pytest

from argus.quarantine import (
    QuarantineResult,
    quarantine_flagged_claims,
)


# --- Synthesis on empty inputs -------------------------------------------

def test_empty_draft_is_noop():
    out = quarantine_flagged_claims(
        "", ["some claim"], ["some flag"],
    )
    assert out.cleaned_text == ""
    assert out.quarantined == []
    assert out.still_unmatched == ["some claim", "some flag"]


def test_empty_claims_is_noop():
    body = "# X\n\nSome prose here. More prose.\n"
    out = quarantine_flagged_claims(body, [], [])
    assert out.cleaned_text == body
    assert out.quarantined == []


def test_default_mode_is_move():
    """Default 'move' must preserve the claim in an appendix."""
    body = "# X\n\nClaim A is true. Other prose. Claim B is also true.\n"
    out = quarantine_flagged_claims(
        body, ["Claim A is true"], [],
    )
    assert "Claim A is true" not in out.cleaned_text.split("## \u26a0\ufe0f")[0], (
        "matched sentence must NOT remain in the body before the appendix"
    )
    # But it IS in the appendix:
    assert "Claim A is true" in out.cleaned_text
    # And the OTHER sentence (not matched) survives in the body.
    assert "Other prose" in out.cleaned_text
    assert "Claim B is also true" in out.cleaned_text


# --- Move mode (default) --------------------------------------------------

def test_move_mode_appends_flagged_claims_appendix():
    body = (
        "# GLM 5.2\n\n"
        "GLM-5.2 is the latest flagship LLM from Z.ai [8] with a 1M-token context window [8]. "
        "It builds on GLM-4 architecture [5]. "
        "Some other unrelated sentence here.\n"
    )
    out = quarantine_flagged_claims(
        body,
        unsupported_claims=["latest flagship LLM from Z.ai"],
        fabrication_flags=["1M-token context window"],
        mode="move",
    )
    # Body: the matched sentences are GONE from the prose.
    pre_appendix = out.cleaned_text.split("## \u26a0\ufe0f")[0]
    assert "flag-ship LLM from Z.ai" not in pre_appendix
    assert "1M-token context window" not in pre_appendix
    # Surviving sentence is in the body.
    assert "It builds on GLM-4 architecture" in pre_appendix
    assert "Some other unrelated sentence" in pre_appendix
    # Both flagged claims ended up in the appendix.
    assert "## \u26a0\ufe0f Unverified / flagged claims" in out.cleaned_text
    appendix = out.cleaned_text.split("## \u26a0\ufe0f", 1)[1]
    assert "latest flagship LLM from Z.ai" in appendix
    assert "1M-token context window" in appendix


def test_move_mode_keeps_citation_markers_in_body():
    """The flagging is by claim text only \u2014 we leave any [n] ref alone
    in the body, because the citation net (sources_block) will clean
    those up downstream. We just want the asserted prose OUT."""
    body = (
        "# X\n\n"
        "GLM-5.2 is the latest flagship LLM from Z.ai [8]. "
        "Other sentence with [1].\n"
    )
    out = quarantine_flagged_claims(
        body, ["latest flagship LLM from Z.ai"], [], mode="move",
    )
    pre_appendix = out.cleaned_text.split("## \u26a0\ufe0f")[0]
    assert "latest flagship LLM" not in pre_appendix
    # The other sentence + its [1] ref survives untouched.
    assert "Other sentence with [1]" in pre_appendix


# --- Remove mode ---------------------------------------------------------

def test_remove_mode_drops_matched_sentences():
    body = (
        "# X\n\n"
        "Sentence one here. Sentence two with the lie. Sentence three.\n"
    )
    out = quarantine_flagged_claims(
        body, ["Sentence two with the lie"], [], mode="remove",
    )
    # The flagged sentence is gone.
    assert "Sentence two with the lie" not in out.cleaned_text
    # Surrounding sentences survive.
    assert "Sentence one here" in out.cleaned_text
    assert "Sentence three" in out.cleaned_text
    # No appendix in remove mode.
    assert "## \u26a0\ufe0f" not in out.cleaned_text


# --- Case sensitivity + whitespace ---------------------------------------

def test_case_insensitive_match():
    body = "# X\n\nSentence with the lie. Other sentence.\n"
    # Flag uses ALL CAPS for the asserted claim; body sentence is
    # lowercase. We expect a case-insensitive match (LLMs vary widely
    # here).
    out = quarantine_flagged_claims(
        body, ["SENTENCE WITH THE LIE"], [], mode="move",
    )
    pre_appendix = out.cleaned_text.split("## \u26a0\ufe0f")[0]
    assert "Sentence with the lie" not in pre_appendix


def test_still_unmatched_returned():
    """A claim that doesn't match ANY sentence in the body must surface
    in ``still_unmatched`` so the report builder can flag it in the
    title block ('3 flagged claims, 1 not found')."""
    body = "# X\n\nThis is a totally different sentence.\n"
    out = quarantine_flagged_claims(
        body,
        ["Claim about unicorns"],
        ["Claim about mars"],
        mode="move",
    )
    assert "Claim about unicorns" in out.still_unmatched
    assert "Claim about mars" in out.still_unmatched
    # Body is unchanged (nothing to move).
    assert "totally different" in out.cleaned_text


# --- Real GLM 5.2 scenario -----------------------------------------------

def test_glm52_scenario_quarantines_specific_claim():
    """Reproduce the GLM 5.2 bug end-to-end through the helper."""
    draft = (
        "# GLM 5.2\n\n"
        "GLM-5.2 is the latest flagship LLM from Z.ai (zai-org), "
        "designed for long-horizon tasks with a solid 1M-token context "
        "window [8]. The model's architecture builds on GLM-4 with "
        "improved MoE routing. Some independent coverage from "
        "thetechbriefs.com summarised the release.\n\n"
        "## Sources\n\n"
        "[1] https://thetechbriefs.com/thudm-releases-glm-4\n"
        "[8] https://build.nvidia.com/z-ai/glm-5.2/modelcard\n"
    )
    out = quarantine_flagged_claims(
        draft,
        unsupported_claims=["GLM-5.2 is the latest flagship LLM from Z.ai"],
        fabrication_flags=["1M-token context window"],
        mode="move",
    )
    pre_appendix = out.cleaned_text.split("## \u26a0\ufe0f")[0]
    # The guilty sentence is gone from the body (its first sentence
    # contained BOTH flagged strings, so the WHOLE sentence moved).
    assert "latest flagship LLM from Z.ai" not in pre_appendix
    assert "1M-token context window" not in pre_appendix
    # The remaining sentences survive.
    assert "improved MoE routing" in pre_appendix
    assert "Some independent coverage" in pre_appendix
    assert "## Sources" in pre_appendix  # not touched
    # Audit trail: 2 items in ``quarantined``.
    assert len(out.quarantined) == 1, (
        f"expected 1 quarantined sentence (the first sentence contained "
        f"both flagged spans); got {len(out.quarantined)}"
    )


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        quarantine_flagged_claims(
            "# X\n\nSome text.\n", ["claim"], [], mode="bogus",
        )


def test_returns_quarantine_result_type():
    out = quarantine_flagged_claims("# X\n\nSome text.\n", [], [])
    assert isinstance(out, QuarantineResult)
    assert out.cleaned_text == "# X\n\nSome text.\n"
    assert out.quarantined == []
    assert out.still_unmatched == []
