"""Tests for the credibility_node + DomainTrust static list (handoff action #2).

Run with:
    PYTHONPATH='' ./venv/Scripts/python.exe -m pytest tests/test_credibility.py -q

The credibility_node sits between fetcher and filter. It assigns each
FetchedItem a `credibility_score` in [0,1] based on three signals:
  (a) domain trust      — static DomainTrust list + tld analysis
  (b) URL pattern       — arxiv / .edu / .gov boost, content-farm penalty
  (c) title relevance   — token overlap with user_request

Items below 0.4 are *tagged* (a `credibility_flag` field) but NOT dropped;
the downstream filter_node still does its own keep_topN.
"""
from __future__ import annotations

from argus.graph.credibility import (
    DomainTrust,
    credibility_score,
    score_fetched,
)
from argus.graph.nodes import credibility_node
from argus.graph.state import FetchedItem


# ---------------------------------------------------------------------------
# (a) Domain trust + URL pattern — high-quality sources
# ---------------------------------------------------------------------------

def test_arxiv_paper_scores_high():
    """An arxiv.org/abs/ URL with a relevant title should score >= 0.75."""
    item = FetchedItem(
        url="https://arxiv.org/abs/2506.18096",
        title="Deep Research Agents: A Systematic Examination And Roadmap",
        excerpt="arxiv preprint on agent architectures",
    )
    score = credibility_score(item, user_request="deep research agents")
    assert score >= 0.75, f"arxiv paper should score high, got {score}"
    assert score <= 1.0


def test_edu_domain_scores_high():
    """A .edu URL on a relevant topic should score >= 0.7."""
    item = FetchedItem(
        url="https://cs.stanford.edu/~example/papers/transformers-survey.pdf",
        title="A Survey of Transformer Architectures",
        excerpt="stanford technical report",
    )
    score = credibility_score(item, user_request="transformer architectures")
    assert score >= 0.7, f".edu source should score high, got {score}"


def test_gov_domain_scores_high():
    """A .gov URL should score >= 0.7."""
    item = FetchedItem(
        url="https://www.nist.gov/publications/ai-risk-framework",
        title="AI Risk Management Framework",
        excerpt="official NIST guidance",
    )
    score = credibility_score(item, user_request="ai risk framework")
    assert score >= 0.7, f".gov source should score high, got {score}"


# ---------------------------------------------------------------------------
# (a) Mid-tier domain trust — known-quality blogs
# ---------------------------------------------------------------------------

def test_towardsdatascience_scores_mid():
    """towardsdatascience.com is a real Medium-publication blog: trusted
    but not primary. Should land in the 0.45-0.8 band."""
    item = FetchedItem(
        url="https://towardsdatascience.com/attention-is-all-you-need-explained",
        title="Attention Is All You Need — Explained",
        excerpt="walkthrough of the transformer paper",
    )
    score = credibility_score(item, user_request="transformer attention")
    assert 0.45 <= score <= 0.85, (
        f"towardsdatascience should be mid-trust, got {score}"
    )


# ---------------------------------------------------------------------------
# (a) Content-farm URL patterns — low scores
# ---------------------------------------------------------------------------

def test_content_farm_urls_score_low():
    """Known content-farm URL patterns should score < 0.4."""
    farms = [
        "https://best-seo-blog-2024.com/transformers-explained",
        "https://top-10-ai-tools.click/best-models-2024",
        "https://freetips.gq/attention-mechanism-guide",
        "https://blogspot-content.example.com/random-article-12345",
    ]
    for url in farms:
        item = FetchedItem(url=url, title="Some Article", excerpt="")
        score = credibility_score(item, user_request="transformer attention")
        assert score < 0.4, f"content farm {url} should score low, got {score}"


def test_suspicious_tld_scores_low():
    """.xyz / .click / .gq / .top are cheap TLDs heavily abused by spam.
    Without other strong signals, the URL alone should land < 0.4."""
    item = FetchedItem(
        url="https://random-article.xyz/some-post",
        title="Some Article",
        excerpt="",
    )
    score = credibility_score(item, user_request="anything")
    assert score < 0.4, f"suspicious TLD should score low, got {score}"


# ---------------------------------------------------------------------------
# (c) Title relevance penalty — domain-trusted URL but off-topic title
# ---------------------------------------------------------------------------

def test_title_relevance_penalty():
    """A trusted arxiv URL whose title does NOT overlap the user_request
    should be penalised below the pure-domain score."""
    user_req = "graph neural networks for molecular property prediction"
    # arxiv.org without any keyword overlap
    on_topic = credibility_score(
        FetchedItem(
            url="https://arxiv.org/abs/2506.00001",
            title="Graph Neural Networks for Molecular Property Prediction",
            excerpt="",
        ),
        user_request=user_req,
    )
    off_topic = credibility_score(
        FetchedItem(
            url="https://arxiv.org/abs/2506.00002",
            title="Cooking Recipes for Beginners",
            excerpt="",
        ),
        user_request=user_req,
    )
    assert on_topic > off_topic + 0.2, (
        f"on-topic ({on_topic}) should beat off-topic ({off_topic}) by ≥0.2"
    )


# ---------------------------------------------------------------------------
# Threshold filtering sanity — arxiv sources survive
# ---------------------------------------------------------------------------

def test_threshold_preserves_most_arxiv_sources():
    """After credibility scoring, ≥50% of arxiv.org items must remain
    above the 0.4 floor (i.e. the floor doesn't accidentally nuke the
    primary source class)."""
    items = [
        FetchedItem(
            url=f"https://arxiv.org/abs/2506.{i:05d}",
            title=f"Research Paper {i}",
            excerpt="",
        )
        for i in range(20)
    ]
    scored = score_fetched(items, user_request="research paper")
    above = sum(1 for s in scored if s.credibility_score >= 0.4)
    assert above >= len(items) * 0.5, (
        f"only {above}/{len(items)} arxiv sources above 0.4 — floor too aggressive"
    )


# ---------------------------------------------------------------------------
# Graph integration — credibility_node is a no-op on empty fetched list
# ---------------------------------------------------------------------------

def test_credibility_node_no_op_when_fetched_empty():
    """If fetcher returned nothing, credibility_node should pass through
    cleanly (empty list, no errors, no model calls)."""
    state = {
        "user_request": "test query",
        "fetched": [],
        "errors": [],
    }
    out = credibility_node(state)
    assert out["fetched"] == []
    assert "errors" not in out or out.get("errors") == []


def test_credibility_node_tags_low_items_but_keeps_them():
    """Items below the 0.4 floor must remain in fetched (not dropped);
    they should carry a `credibility_flag` so the downstream filter_node
    can decide what to do."""
    item = FetchedItem(
        url="https://random-article.xyz/spammy-post",
        title="Some Article",
        excerpt="",
    )
    state = {
        "user_request": "transformer architectures",
        "fetched": [item.model_dump()],
        "errors": [],
    }
    out = credibility_node(state)
    assert len(out["fetched"]) == 1, "credibility_node must not drop items"
    flagged = out["fetched"][0]
    assert "credibility_score" in flagged
    assert flagged["credibility_score"] < 0.4
    assert flagged.get("credibility_flag") in {
        "low_credibility", None,
    }  # flag field present, value either low_credibility or None
    # if a flag is set, it must indicate low credibility
    if flagged.get("credibility_flag"):
        assert "low" in flagged["credibility_flag"].lower()


def test_score_is_bounded_zero_to_one():
    """Every score must lie in [0, 1] for any URL/title combo."""
    cases = [
        ("https://arxiv.org/abs/2506.18096", "Deep Research Agents Survey", "deep research"),
        ("https://random-article.xyz/x", "totally unrelated", "machine learning"),
        ("", "no url at all", "anything"),
    ]
    for url, title, req in cases:
        item = FetchedItem(url=url, title=title, excerpt="")
        score = credibility_score(item, user_request=req)
        assert 0.0 <= score <= 1.0, f"score {score} out of bounds for {url!r}"


def test_domain_trust_dataclass_export():
    """DomainTrust must be a public, importable symbol (used by tests
    and by anyone tuning the trust list later)."""
    assert DomainTrust is not None
    # at minimum we should have a few well-known entries
    entries = DomainTrust.entries()
    domains = {d["domain"] for d in entries}
    assert "arxiv.org" in domains
    assert "nist.gov" in domains


# ---------------------------------------------------------------------------
# P2 fix — curated "low" tier + structural heuristics
# ---------------------------------------------------------------------------

def test_curated_low_hosts_are_low_tier():
    """thetechbriefs.com and the other curated offenders must be
    recognised as 'low' tier — not 'neutral' (which was the bug)."""
    from argus.graph.credibility import DomainTrust
    for url in [
        "https://thetechbriefs.com/thudm-releases-glm-4",
        "https://glm45.org/?",
        "https://ai-news-briefs.com/some-post",
        "https://buy-me-a-coffee.com/farm-content",
    ]:
        tier = DomainTrust.tier_for(url)
        assert tier == "low", f"curated farm {url} should be low, got {tier}"


def test_structural_low_hosts_get_low_tier():
    """Hosts with 3+ hyphens or a single-char + cheap TLD get bumped
    from 'neutral' to 'low' by the structural heuristic."""
    from argus.graph.credibility import DomainTrust
    cases = [
        ("https://seo-content-farm-quick-tips-blog.click/x", "low"),
        ("https://a.xyz/post", "low"),
        ("https://b.gq/article", "low"),
        ("https://x.top/x", "low"),
    ]
    for url, expected in cases:
        tier = DomainTrust.tier_for(url)
        assert tier == expected, f"structural-farm {url} should be {expected}, got {tier}"


def test_structural_low_does_not_downgrade_curated_primary():
    """A primary host (arxiv.org) that *also* matches a structural pattern
    (e.g. arxiv.org) keeps its curated tier — the heuristic only nudges
    neutral -> low, never downgrades primary/trusted."""
    from argus.graph.credibility import DomainTrust
    assert DomainTrust.tier_for("https://arxiv.org/abs/2402.12345") == "primary"
    assert DomainTrust.tier_for("https://github.com/x/y") == "trusted"


def test_domain_table_has_low_entries():
    """The P2 fix requires actual 'low' entries in the table; previously
    there were none, so content farms fell through to 'neutral' (0.25)."""
    from argus.graph.credibility import DomainTrust
    entries = DomainTrust.entries()
    tiers = {d["tier"] for d in entries}
    assert "low" in tiers, "_DOMAIN_TABLE must contain at least one 'low' entry"
    # And the curated thetechbriefs.com must be the 'low' entry (regression).
    thetechbriefs = next((d for d in entries
                          if d["domain"] == "thetechbriefs.com"), None)
    assert thetechbriefs is not None
    assert thetechbriefs["tier"] == "low"


# --- filter_node: credibility floor enforcement (P2 part 2) ---------------

def test_filter_node_drops_below_floor_items():
    """P2 fix: filter_node must drop items below CREDIBILITY_FLOOR.

    GLM 5.2 reproduced here: a fetcher mix of an arxiv paper, a thetechbriefs
    farm, and a neutral blog. After fix, the farm must be gone from the
    report's fetched list."""
    from argus.graph.nodes import filter_node
    items = [
        FetchedItem(  # arxiv — well above floor
            url="https://arxiv.org/abs/2406.12793",
            title="GLM-130B paper",
            excerpt="paper on the GLM family",
            markdown_path="/tmp/p1.md",
            relevance_score=0.8,
            credibility_score=0.85,
        ),
        FetchedItem(  # content farm — below floor
            url="https://thetechbriefs.com/thudm-releases-glm-4",
            title="GLM-4 release coverage",
            excerpt="tech brief article",
            markdown_path="/tmp/p2.md",
            relevance_score=0.6,
            credibility_score=0.18,  # < floor (0.4)
        ),
        FetchedItem(  # neutral blog — above floor
            url="https://towardsdatascience.com/some-post",
            title="GLM walkthrough",
            excerpt="tds article",
            markdown_path="/tmp/p3.md",
            relevance_score=0.5,
            credibility_score=0.70,
        ),
    ]
    state = {
        "user_request": "GLM 5.2",
        "fetched": [it.model_dump() for it in items],
        "plan": {"must_have_keywords": ["GLM"]},
        "errors": [],
    }
    out = filter_node(state)
    urls = [f["url"] for f in out["fetched"]]
    assert "https://thetechbriefs.com/thudm-releases-glm-4" not in urls, (
        "filter_node let through the content farm — P2 floor not enforced"
    )
    assert "https://arxiv.org/abs/2406.12793" in urls
    assert "https://towardsdatascience.com/some-post" in urls


def test_filter_node_safety_net_keeps_best_when_all_below_floor():
    """If every fetched item scores below the floor, filter_node must NOT
    emit an empty list — that's the Phase-1 'don't empty the report'
    principle. The kept set should still contain the best-ranked items."""
    from argus.graph.nodes import filter_node
    from argus.graph.credibility import CREDIBILITY_FLOOR
    items = [
        FetchedItem(
            url=f"https://thetechbriefs.com/post-{i}",
            title="GLM release coverage",
            excerpt="farm content",
            markdown_path=f"/tmp/farm-{i}.md",
            relevance_score=0.5 - (i * 0.05),  # still above 0 below floor
            credibility_score=CREDIBILITY_FLOOR - 0.05,  # just below floor
        )
        for i in range(3)
    ]
    state = {
        "user_request": "GLM 5.2",
        "fetched": [it.model_dump() for it in items],
        "plan": {"must_have_keywords": ["GLM"]},
        "errors": [],
    }
    out = filter_node(state)
    assert len(out["fetched"]) >= 1, (
        "filter_node safety net failed: emitted empty report when all "
        "items below floor"
    )
    # Items must still come from the ranked set (relevance-ordered).
    rels = [f["relevance_score"] for f in out["fetched"]]
    assert rels == sorted(rels, reverse=True), (
        "safety net should preserve the (relevance, credibility) ordering"
    )