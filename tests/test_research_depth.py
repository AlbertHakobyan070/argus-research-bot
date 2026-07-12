"""Depth rebalance guards (2026-07-12).

Locks in the fix for "reports too shallow and too filtered": legit
reference/curated/primary domains must clear the credibility floor
(they used to score ~0.11 and get cut), content farms must still be
filtered (fabrication guard intact), grounding must not over-fire, and
the default 'short' mode must not be skeletal.
"""
from __future__ import annotations

import pytest

from argus.graph.credibility import (
    CREDIBILITY_FLOOR, DomainTrust, credibility_score, score_fetched,
)
from argus.graph.grounding import DEFAULT_GROUNDING_THRESHOLD, check_grounding
from argus.graph.state import FetchedItem
from argus.graph.synthesis_modes import get_mode


def _item(url: str, title: str = "retrieval augmented generation") -> FetchedItem:
    return FetchedItem(url=url, title=title, excerpt="RAG small models",
                       markdown_path="x.md")


# ---------------------------------------------------------------------------
# legit sources clear the floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://arxiv.org/abs/2410.12345",
    "https://github.com/org/rag",
    "https://huggingface.co/blog/rag",
    "https://paperswithcode.com/task/rag",
    "https://stackoverflow.com/questions/123/rag",
])
def test_legit_domains_clear_the_floor(url):
    s = credibility_score(_item(url), user_request="retrieval augmented generation")
    assert s >= CREDIBILITY_FLOOR, (
        f"{url} scored {s:.2f} < floor {CREDIBILITY_FLOOR} — legit sources "
        "must not be filtered as noise (this was the over-filtering bug)")


def test_wikipedia_is_no_longer_scored_as_noise():
    # The exact regression: Wikipedia scored 0.11 pre-rebalance.
    s = credibility_score(
        _item("https://en.wikipedia.org/wiki/Retrieval-augmented_generation"),
        user_request="retrieval augmented generation small models")
    assert s >= 0.5, f"Wikipedia should be a solidly-credible source, got {s:.2f}"


# ---------------------------------------------------------------------------
# fabrication / content-farm guard still intact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://thetechbriefs.com/best-rag-models-2025",
    "https://glm45.org/glm-5.2-specs",
    "https://ml-briefs.com/top-10-rag",
    "https://seo-content.ai/rag",
])
def test_content_farms_still_dropped(url):
    s = credibility_score(_item(url), user_request="rag")
    assert s < CREDIBILITY_FLOOR, (
        f"content farm {url} scored {s:.2f} >= floor — the anti-fabrication "
        "guard must stay intact after the depth rebalance")


def test_tier_ordering_preserved():
    e = {d["domain"]: d["score"] for d in DomainTrust.entries()}
    # primary > trusted > (neutral default) > low still holds.
    assert DomainTrust.score_for("https://arxiv.org/abs/1") > \
        DomainTrust.score_for("https://github.com/a/b")
    assert DomainTrust.score_for("https://github.com/a/b") > \
        DomainTrust.score_for("https://glm45.org/x")


# ---------------------------------------------------------------------------
# grounding no longer over-fires on a normally-covered topic
# ---------------------------------------------------------------------------


def test_grounding_holds_on_well_covered_topic():
    """Three credible on-topic sources = grounded (no strict-conservatism
    warning). Pre-rebalance these scored below the floor and the topic was
    wrongly flagged 'insufficient evidence'."""
    fetched = score_fetched([
        _item("https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
              "retrieval augmented generation overview"),
        _item("https://arxiv.org/abs/2410.00001",
              "retrieval augmented generation for small models"),
        _item("https://github.com/org/rag",
              "retrieval augmented generation toolkit"),
    ], user_request="retrieval augmented generation")
    res = check_grounding({"user_request": "retrieval augmented generation",
                           "fetched": [f.model_dump() for f in fetched]})
    assert res.n_credible >= 3, f"expected >=3 credible, got {res.n_credible}"
    assert res.grounded, (
        "a topic with 3 credible on-topic sources must be grounded — the "
        "over-conservative gate was the cause of shallow hedged reports")


def test_grounding_still_flags_thin_topic():
    """A single content-farm source about a nonexistent entity stays
    ungrounded — the GLM-5.2-style guard survives."""
    fetched = score_fetched([
        _item("https://glm45.org/glm-5.2", "glm 5.2 specs"),
    ], user_request="glm 5.2 benchmark scores")
    res = check_grounding({"user_request": "glm 5.2 benchmark scores",
                           "fetched": [f.model_dump() for f in fetched]})
    assert not res.grounded, "a lone content farm must not ground a rare entity"


# ---------------------------------------------------------------------------
# default depth is no longer skeletal
# ---------------------------------------------------------------------------


def test_short_mode_is_not_skeletal():
    m = get_mode("short")
    assert m.target_findings >= 6, f"short bumped to >=6 findings, got {m.target_findings}"
    assert m.max_tokens >= 1800
    assert m.target_chars_max >= 1400


def test_depth_ordering_across_modes_preserved():
    modes = [get_mode(k) for k in ("tldr", "short", "medium", "long", "lecture")]
    findings = [m.target_findings for m in modes]
    # non-decreasing depth as modes get longer (tldr=0 is fine)
    assert findings == sorted(findings), findings
    assert get_mode("lecture").target_findings > get_mode("short").target_findings
