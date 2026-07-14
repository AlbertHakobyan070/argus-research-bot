"""Tests for credibility scoring + DomainTrust static list.

Run with:
    PYTHONPATH='' ./venv/Scripts/python.exe -m pytest tests/test_credibility.py -q

v3: scoring runs inside research_node (post-fetch) and as a triage
prior (pre-fetch). Each FetchedItem gets a `credibility_score` in [0,1]
from three signals:
  (a) domain trust      — static DomainTrust list + tld analysis
  (b) URL pattern       — arxiv / .edu / .gov boost, content-farm penalty
  (c) title relevance   — token overlap with user_request

Items below the floor are *tagged* (a `credibility_flag` field) but NOT
dropped; the research triage demotes them softly instead.
"""
from __future__ import annotations

from argus.graph.credibility import (
    CREDIBILITY_FLOOR,
    DomainTrust,
    credibility_score,
    is_arxiv_year_suspicious,
    score_fetched,
)
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
# score_fetched — the scoring pass research_node runs post-fetch
# ---------------------------------------------------------------------------

def test_score_fetched_empty_list_is_noop():
    assert score_fetched([], user_request="test query") == []


def test_score_fetched_tags_low_items_but_keeps_them():
    """Items below the floor must remain (not dropped); they carry a
    `credibility_flag` so triage can demote them softly."""
    item = FetchedItem(
        url="https://random-article.xyz/spammy-post",
        title="Some Article",
        excerpt="",
    )
    scored = score_fetched([item], user_request="transformer architectures")
    assert len(scored) == 1, "score_fetched must not drop items"
    flagged = scored[0]
    assert flagged.credibility_score is not None
    assert flagged.credibility_score < 0.4
    if flagged.credibility_flag:
        assert "low" in flagged.credibility_flag.lower()


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


# --- research triage: credibility floor demotion (v3) ----------------------

def test_triage_demotes_content_farm_below_trusted_sources():
    """The P2 farm bug, v3 shape: with a cap smaller than the candidate
    pool, triage must pick the arxiv paper + neutral blog over the
    below-floor content farm."""
    from argus.graph.research import triage
    from argus.graph.state import ResearchBrief, SubQuestion
    brief = ResearchBrief(
        sub_questions=[SubQuestion(q="What is the GLM model family?")],
        must_have_keywords=["GLM"],
    )
    sources = [
        {"url": "https://thetechbriefs.com/thudm-releases-glm-4",
         "title": "GLM-4 release coverage", "snippet": "GLM tech brief",
         "sub_qs": [0]},
        {"url": "https://arxiv.org/abs/2406.12793",
         "title": "GLM-130B paper", "snippet": "paper on the GLM family",
         "sub_qs": [0]},
        {"url": "https://towardsdatascience.com/some-post",
         "title": "GLM walkthrough", "snippet": "tds article on GLM",
         "sub_qs": [0]},
    ]
    picked = triage(sources, set(), brief, cap=2)
    urls = [s["url"] for s in picked]
    assert "https://thetechbriefs.com/thudm-releases-glm-4" not in urls, (
        "triage picked the content farm over trusted sources"
    )
    assert "https://arxiv.org/abs/2406.12793" in urls


def test_triage_safety_net_keeps_best_when_all_below_floor():
    """If every candidate scores below the floor, triage must still pick
    sources — 'don't empty the report' principle, v3 shape."""
    from argus.graph.research import triage
    from argus.graph.state import ResearchBrief, SubQuestion
    brief = ResearchBrief(
        sub_questions=[SubQuestion(q="What is GLM 5.2?")],
        must_have_keywords=["GLM"],
    )
    sources = [
        {"url": f"https://thetechbriefs.com/post-{i}",
         "title": "GLM release coverage", "snippet": "farm content on GLM",
         "sub_qs": [0]}
        for i in range(3)
    ]
    picked = triage(sources, set(), brief, cap=2)
    assert len(picked) >= 1, (
        "triage emitted nothing when all items were below the floor"
    )


# ---------------------------------------------------------------------------
# (d) arxiv year-fabrication heuristic (P2.5, 2026-07-10)
# ---------------------------------------------------------------------------

def test_arxiv_year_suspicious_flags_fabricated_future_year():
    """`2606.32032` in July 2026 is current-year, but the 5-digit
    sequence at month-6 hasn't been reached yet — fabricate flag."""
    susp, year = is_arxiv_year_suspicious(
        "https://arxiv.org/abs/2606.32032", current_year=2026,
    )
    assert susp is True
    assert year == 2026


def test_arxiv_year_suspicious_flags_future_year():
    """YY > current_yy is plain-fabricated."""
    susp, year = is_arxiv_year_suspicious(
        "https://arxiv.org/abs/3301.12345", current_year=2026,
    )
    assert susp is True
    assert year == 2033


def test_arxiv_year_suspicious_allows_legitimate_past_papers():
    """A valid 2024 paper with 5-digit ID and MM >= 7 must NOT flag."""
    susp, year = is_arxiv_year_suspicious(
        "https://arxiv.org/abs/2407.01234", current_year=2026,
    )
    assert susp is False
    assert year == 2024


def test_arxiv_year_suspicious_allows_current_year_with_four_digit_seq():
    """A current-year URL with a 4-digit sequence is fine even at month <=6."""
    susp, year = is_arxiv_year_suspicious(
        "https://arxiv.org/abs/2603.1234", current_year=2026,
    )
    assert susp is False
    assert year == 2026


def test_arxiv_year_suspicious_skips_non_arxiv_hosts():
    """`/abs/YYMM.NNNNN` on google.com is not an arxiv absolute."""
    susp, year = is_arxiv_year_suspicious(
        "https://google.com/abs/9999.99999", current_year=2026,
    )
    assert susp is False
    assert year is None


def test_arxiv_year_suspicious_skips_arxiv_browse_pages():
    """`/list/cs.LG/recent` doesn't match the /abs/ pattern."""
    susp, year = is_arxiv_year_suspicious(
        "https://arxiv.org/list/cs.LG/recent", current_year=2026,
    )
    assert susp is False
    assert year is None


def test_arxiv_year_suspicious_flags_pre_2015_format_attempts():
    """arxiv new-format YY.NNNNN started 2015; YY<15 is fabricated."""
    susp, year = is_arxiv_year_suspicious(
        "https://arxiv.org/abs/9101.00001", current_year=2026,
    )
    assert susp is True
    assert year == 2091  # parsed as 2091, which is also future


def test_arxiv_year_suspicious_flags_invalid_month():
    """MM=13 doesn't exist."""
    susp, year = is_arxiv_year_suspicious(
        "https://arxiv.org/abs/2613.12345", current_year=2026,
    )
    assert susp is True
    assert year == 2026


def test_arxiv_year_suspicious_allows_version_suffix():
    """`v7` suffix is preserved but not blocking."""
    susp, year = is_arxiv_year_suspicious(
        "https://arxiv.org/abs/1706.03762v7", current_year=2026,
    )
    assert susp is False
    assert year == 2017


def test_arxiv_year_suspicious_handles_empty_path():
    """Empty / no path returns (False, None), no exception."""
    susp, year = is_arxiv_year_suspicious(
        "https://arxiv.org/abs/", current_year=2026,
    )
    assert susp is False
    assert year is None


# End-to-end: a fabricated arxiv URL must surface in score_fetched
# with credibility_flag="fabricated_path" AND score below floor.

def test_score_fetched_marks_fabricated_arxiv_with_flag_and_below_floor():
    """Fabricated arxiv URL -> flagged AND dropped below the floor so
    filter_node's P2 enforcement handles it uniformly."""
    item = FetchedItem(
        url="https://arxiv.org/abs/2606.32032",
        title="Some metacognitive RL paper",
        excerpt="non-credible abstract",
    )
    out = score_fetched([item], user_request="metacognitive RL")
    assert len(out) == 1
    scored = out[0]
    assert scored.credibility_flag == "fabricated_path"
    assert scored.credibility_score < CREDIBILITY_FLOOR


def test_score_fetched_does_not_flag_legitimate_arxiv():
    """A legitimate 2024 arxiv URL stays un-flagged."""
    item = FetchedItem(
        url="https://arxiv.org/abs/2401.01234",
        title="Deep Research Agents: A Systematic Examination And Roadmap",
        excerpt="arxiv preprint on agent architectures",
    )
    out = score_fetched([item], user_request="deep research agents")
    assert out[0].credibility_flag != "fabricated_path"
