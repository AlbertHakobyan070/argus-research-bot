"""Tests for src/argus/graph/grounding.py (P5: grounding check).

Run with:
    PYTHONPATH=./' ./venv/Scripts/python.exe -m pytest tests/test_grounding.py -q
"""
from __future__ import annotations

import pytest

from argus.graph.grounding import (
    DEFAULT_GROUNDING_THRESHOLD,
    GroundingResult,
    check_grounding,
    extract_entity_tokens,
    grounding_banner,
    grounding_warning_prompt,
)


# --- Token extraction -------------------------------------------------

def test_extract_tokens_dedupes_and_orders():
    tokens = extract_entity_tokens("GLM 5.2 GLM GLM 4 4 4 4")
    assert "glm" in tokens
    # First mention comes first, dupes dropped.
    assert tokens.index("glm") == 0


def test_extract_tokens_filters_short_and_stopwords():
    """Common glue words and length-mode names must NOT be entities."""
    tokens = extract_entity_tokens("research the model is approach")
    # All in the stop set; result is empty.
    assert tokens == []


def test_extract_tokens_preserves_multi_word_signals():
    """Each meaningful word comes through."""
    tokens = extract_entity_tokens("metacognitive reinforcement learning transformers")
    assert tokens == ["metacognitive", "reinforcement", "learning", "transformers"]


# --- Grounding scoring -----------------------------------------------

def test_check_grounding_with_no_user_request_is_grounded():
    """No entity to look for => trivially grounded (no warning)."""
    r = check_grounding({"user_request": "", "fetched": []})
    assert r.grounded is True
    assert r.entity_tokens == []


def test_check_grounding_skips_low_credibility_items():
    """A farm item below the floor must NOT count as a mention.
    The grounding signal is only as good as the credibility filter."""
    state = {
        "user_request": "GLM 5.2",
        "fetched": [
            # Two farms mentioning GLM 5.2 - excluded from credible count.
            {"title": "GLM 5.2!!! tech brief",
             "credibility_score": 0.18},
            {"title": "GLM 5.2 - hottest model 2026",
             "credibility_score": 0.12},
        ],
    }
    r = check_grounding(state)
    # Only 0 credible items, 0 mentions, NOT grounded.
    assert r.n_credible == 0
    assert r.mention_count == 0
    assert r.grounded is False


def test_check_grounding_grounds_well_known_topic():
    """Transformer architecture: many credible arxiv + .edu mentions."""
    state = {
        "user_request": "transformer attention mechanism",
        "fetched": [
            {"title": "Attention Is All You Need - revisited",
             "credibility_score": 0.9},
            {"title": "Transformer architecture survey",
             "credibility_score": 0.85},
            {"title": "Multi-head attention explained",
             "credibility_score": 0.8},
            {"title": "Sequence modeling with transformers",
             "credibility_score": 0.8},
        ],
    }
    r = check_grounding(state)
    assert r.n_credible == 4
    assert r.mention_count >= 3
    assert r.grounded is True


def test_check_grounding_obscure_entity_low_grounding():
    """P5 reproducer: GLM 5.2 with 1 direct credible mention."""
    state = {
        "user_request": "GLM 5.2",
        "fetched": [
            {"title": "GLM 5.2 technical overview",
             "excerpt": "GLM 5.2 release notes",
             "credibility_score": 0.8},  # mentions -> ground count
            {"title": "AI news brief",
             "excerpt": "weekly coverage",
             "credibility_score": 0.8},  # no GLM 5.2 -> noise
        ],
    }
    r = check_grounding(state)
    assert r.n_credible == 2
    assert r.mention_count == 1
    assert r.grounded is False
    assert "glm" in r.entity_tokens


def test_check_grounding_threshold_can_be_lowered():
    """Threshold is a hyperparameter; lower it and the same state grounds."""
    state = {
        "user_request": "GLM 5.2",
        "fetched": [
            {"title": "GLM 5.2 technical overview",
             "credibility_score": 0.8},
        ],
    }
    r_default = check_grounding(state)
    r_lower = check_grounding(state, threshold=1)
    assert r_default.grounded is False
    assert r_lower.grounded is True


def test_check_grounding_accepts_object_items():
    """Tolerates FetchedItem objects (not just dicts)."""
    from argus.graph.state import FetchedItem
    item = FetchedItem(
        url="https://arxiv.org/abs/2506.00001",
        title="Attention mechanism overview",
        credibility_score=0.9,
    )
    r = check_grounding({
        "user_request": "attention mechanism",
        "fetched": [item],
    })
    assert r.grounded is False  # threshold=3, only 1 mention
    assert r.n_credible == 1


def test_check_grounding_collects_sample_titles():
    """Sample titles surface in GroundingResult for the warning UI."""
    state = {
        "user_request": "transformer survey",
        "fetched": [
            {"title": "A Survey of Transformers",
             "credibility_score": 0.9},
            {"title": "Transformer-X paper",
             "credibility_score": 0.85},
            {"title": "Transformers in vision",
             "credibility_score": 0.85},
            {"title": "Other",
             "credibility_score": 0.85},
        ],
    }
    r = check_grounding(state, threshold=3)
    # Grounded: 3 mentions of "transformer/survey".
    assert r.grounded is True
    # Samples capped at 3.
    assert len(r.sample_evidence_titles) == 3


# --- Warning prompt + banner -----------------------------------------

def test_warning_prompt_empty_when_grounded():
    """The caller does NOT prepend anything when grounded."""
    r = GroundingResult(
        grounded=True, entity_tokens=["transformer"],
        mention_count=5, n_credible=5, threshold=3,
    )
    out = grounding_warning_prompt(r, "transformer survey")
    assert out == ""


def test_warning_prompt_non_empty_when_not_grounded():
    """When NOT grounded, the warning block lists the entity + a
    conservative-action list."""
    r = GroundingResult(
        grounded=False, entity_tokens=["glm"],
        mention_count=1, n_credible=2, threshold=3,
    )
    out = grounding_warning_prompt(r, "GLM 5.2")
    assert "GROUNDING WARNING" in out
    assert "GLM 5.2" in out
    assert "strictly conservative" in out.lower()
    assert "insufficient direct evidence" in out.lower()
    assert "1" in out and "2" in out  # mention_count + n_credible


def test_banner_empty_when_grounded():
    r = GroundingResult(
        grounded=True, entity_tokens=["x"],
        mention_count=5, n_credible=5, threshold=3,
    )
    assert grounding_banner(r) == ""


def test_banner_non_empty_with_counts_when_not_grounded():
    r = GroundingResult(
        grounded=False, entity_tokens=["unicorns"],
        mention_count=0, n_credible=2, threshold=3,
    )
    out = grounding_banner(r)
    assert "Limited direct evidence" in out
    # Counts are visible.
    assert "0" in out and "2" in out


# --- Behavioural / threshold defaults --------------------------------

def test_default_threshold_constant_matches_documented():
    """DEFAULT_GROUNDING_THRESHOLD=2 after the 2026-07-12 depth rebalance
    (was 3). Threshold-3 over-fired the strict-conservatism warning on
    normal topics; 2 still catches genuinely thin entities (0-1 credible
    mentions) while letting well-covered topics synthesize freely."""
    assert DEFAULT_GROUNDING_THRESHOLD == 2


def test_empty_state_grounds_trivially():
    """An empty fetched list with a real query = no evidence, NOT
    grounded. The warning fires so the synthesizer is told the web
    returned nothing."""
    r = check_grounding({"user_request": "GLM 5.2", "fetched": []})
    assert r.grounded is False
    assert r.n_credible == 0
    assert r.mention_count == 0
