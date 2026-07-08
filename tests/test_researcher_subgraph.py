"""Tests for researcher subgraph (handoff action #1, T8).

All HTTP is mocked — these are hermetic. The arxiv subs use the real
parsers (``_arxiv_search``, ``_arxiv_search_raw``) but with httpx
responses generated in-test, so they're fast and deterministic.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.argus.graph.researcher_subgraph import (
    arxiv_sub,
    github_sub,
    merge_research,
    web_sub,
    supervisor_node,
    build_researcher_subgraph,
    run_researcher_subgraph,
    _route_planned_source,
)
from src.argus.graph.state import PlannedSource, ResearchPlan


# ---------------------------------------------------------------------------
# Routing tests (pure function)
# ---------------------------------------------------------------------------

class TestRoutePlannedSource:
    def test_paper_routes_to_arxiv(self):
        assert _route_planned_source(
            PlannedSource(kind="paper", query="transformer")) == "arxiv"

    def test_repo_routes_to_github(self):
        assert _route_planned_source(
            PlannedSource(kind="repo")) == "github"

    def test_blog_news_doc_routes_to_web(self):
        for k in ("blog", "news", "official_doc", "search_result"):
            assert _route_planned_source(PlannedSource(kind=k)) == "web"

    def test_unknown_kind_defaults_to_web(self):
        assert _route_planned_source(
            PlannedSource(kind="search_result")) == "web"


# ---------------------------------------------------------------------------
# supervisor_node: no-op, just sanity
# ---------------------------------------------------------------------------

def test_supervisor_node_is_noop():
    assert supervisor_node({}) == {}
    assert supervisor_node({"plan": {"x": 1}}) == {}


# ---------------------------------------------------------------------------
# arxiv_sub: mocked httpx, returns the parsed entries
# ---------------------------------------------------------------------------

_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2401.01234v1</id>
    <title>Test Paper on Quantum Error Correction</title>
    <summary>Short summary about QEC.</summary>
  </entry>
  <entry>
    <id>https://arxiv.org/abs/2401.05678v2</id>
    <title>Another Paper</title>
    <summary>Another summary.</summary>
  </entry>
</feed>
"""


@pytest.mark.asyncio
async def test_arxiv_sub_returns_parsed_entries(monkeypatch):
    """Mock the sync httpx.Client used by _arxiv_search."""
    fake_response = MagicMock()
    fake_response.text = _ARXIV_ATOM
    fake_response.status_code = 200
    fake_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get = MagicMock(return_value=fake_response)
    monkeypatch.setattr("httpx.Client", lambda **kw: mock_client)

    out = await arxiv_sub({
        "user_request": "quantum error correction",
        "plan": ResearchPlan(
            must_have_keywords=["quantum", "error"],
            sub_questions=["how does QEC work?"]).model_dump(),
    })
    assert len(out["sub_results"]) == 1
    sr = out["sub_results"][0]
    assert sr["sub_kind"] == "arxiv"
    assert sr["error"] == ""
    urls = [s["url"] for s in sr["sources"]]
    assert "https://arxiv.org/abs/2401.01234v1" in urls
    assert "https://arxiv.org/abs/2401.05678v2" in urls


@pytest.mark.asyncio
async def test_arxiv_sub_handles_empty_keywords_with_fallback(monkeypatch):
    fake_response = MagicMock()
    fake_response.text = _ARXIV_ATOM
    fake_response.status_code = 200
    fake_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get = MagicMock(return_value=fake_response)
    monkeypatch.setattr("httpx.Client", lambda **kw: mock_client)

    # Empty keywords but user_request set → fallback should kick in.
    out = await arxiv_sub({
        "user_request": "metacognitive reinforcement learning",
        "plan": ResearchPlan().model_dump(),
    })
    sr = out["sub_results"][0]
    assert sr["sub_kind"] == "arxiv"
    assert len(sr["sources"]) >= 1


@pytest.mark.asyncio
async def test_arxiv_sub_does_not_crash_on_http_error(monkeypatch):
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get = MagicMock(side_effect=Exception("network"))
    monkeypatch.setattr("httpx.Client", lambda **kw: mock_client)

    out = await arxiv_sub({
        "user_request": "anything",
        "plan": ResearchPlan(
            must_have_keywords=["x"]).model_dump(),
    })
    sr = out["sub_results"][0]
    assert sr["sub_kind"] == "arxiv"
    assert sr["error"].startswith("arxiv_sub failed")


# ---------------------------------------------------------------------------
# github_sub: mocked httpx
# ---------------------------------------------------------------------------

_GH_PAYLOAD = {
    "items": [
        {
            "full_name": "anthropics/claude-code",
            "name": "claude-code",
            "html_url": "https://github.com/anthropics/claude-code",
            "description": "Anthropic's CLI for Claude.",
        },
        {
            "full_name": "openai/codex",
            "name": "codex",
            "html_url": "https://github.com/openai/codex",
            "description": "Lightweight coding agent.",
        },
    ]
}


@pytest.mark.asyncio
async def test_github_sub_returns_search_results():
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json = MagicMock(return_value=_GH_PAYLOAD)

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=fake_response)
        mock_client_cls.return_value = mock_client

        out = await github_sub({
            "user_request": "agent frameworks",
            "plan": ResearchPlan(
                must_have_keywords=["agent"]).model_dump(),
        })
    sr = out["sub_results"][0]
    assert sr["sub_kind"] == "github"
    assert sr["error"] == ""
    urls = [s["url"] for s in sr["sources"]]
    assert "https://github.com/anthropics/claude-code" in urls
    assert "https://github.com/openai/codex" in urls
    assert all(s["kind"] == "repo" for s in sr["sources"])


@pytest.mark.asyncio
async def test_github_sub_handles_rate_limit():
    fake_response = MagicMock()
    fake_response.status_code = 403

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=fake_response)
        mock_client_cls.return_value = mock_client

        out = await github_sub({
            "user_request": "x",
            "plan": ResearchPlan(must_have_keywords=["x"]).model_dump(),
        })
    sr = out["sub_results"][0]
    assert "rate-limited" in sr["error"]
    assert sr["sources"] == []


@pytest.mark.asyncio
async def test_github_sub_handles_empty_query():
    out = await github_sub({"user_request": "", "plan": {}})
    sr = out["sub_results"][0]
    assert sr["sources"] == []
    assert sr["error"] == ""


# ---------------------------------------------------------------------------
# web_sub: mocked ddgs_search
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_sub_uses_ddgs_search():
    fake_results = [
        {"url": "https://blog.example.com/post1",
         "title": "A Post About Agents",
         "snippet": "discussion of multi-agent loops",
         "source": "ddgs", "kind": "blog"},
        {"url": "https://news.example.com/x",
         "title": "News",
         "snippet": "news snippet",
         "source": "ddgs", "kind": "news"},
    ]
    with patch("src.argus.tools.ddgs_search", return_value=fake_results,
                create=True):
        # Need the import to succeed — patch the module path that
        # web_sub actually imports from.
        import src.argus.tools as tools_mod  # noqa: F401
        with patch.object(tools_mod, "ddgs_search",
                           return_value=fake_results, create=True):
            out = await web_sub({
                "user_request": "agents",
                "plan": ResearchPlan(
                    must_have_keywords=["agent"]).model_dump(),
            })
    sr = out["sub_results"][0]
    assert sr["sub_kind"] == "web"
    urls = [s["url"] for s in sr["sources"]]
    assert "https://blog.example.com/post1" in urls


@pytest.mark.asyncio
async def test_web_sub_handles_no_ddgs():
    """If ddgs_search isn't installed (action #4 still pending), no crash."""
    with patch.dict("sys.modules", {"src.argus.tools": MagicMock(
        spec=["ddgs_search"],
    )}):
        # Simulate the ddgs_search attr being absent.
        with patch("src.argus.tools.ddgs_search", side_effect=ImportError,
                    create=True):
            out = await web_sub({
                "user_request": "x",
                "plan": ResearchPlan(must_have_keywords=["x"]).model_dump(),
            })
    sr = out["sub_results"][0]
    assert sr["sub_kind"] == "web"


# ---------------------------------------------------------------------------
# merge_research: dedup, cap, pre-seed priority
# ---------------------------------------------------------------------------

def test_merge_research_dedupes_by_url():
    state = {
        "pre_seeded_sources": [],
        "sub_results": [
            {"sub_kind": "arxiv", "sources": [
                {"url": "https://arxiv.org/abs/1", "title": "a"}],
             "error": ""},
            {"sub_kind": "github", "sources": [
                {"url": "https://arxiv.org/abs/1", "title": "dup-a"},  # dup
                {"url": "https://github.com/x/y", "title": "g"}],
             "error": ""},
        ],
    }
    out = merge_research(state)
    urls = [s["url"] for s in out["final_sources"]]
    assert urls.count("https://arxiv.org/abs/1") == 1
    assert "https://github.com/x/y" in urls
    # First-seen wins (arxiv sub ordered first).
    assert out["final_sources"][0]["title"] == "a"


def test_merge_research_caps_at_18():
    many = [{"url": f"https://x.com/{i}", "title": f"t{i}"}
            for i in range(30)]
    state = {
        "pre_seeded_sources": [],
        "sub_results": [
            {"sub_kind": "arxiv", "sources": many, "error": ""},
            {"sub_kind": "github", "sources": [], "error": ""},
            {"sub_kind": "web", "sources": [], "error": ""},
        ],
    }
    out = merge_research(state)
    assert len(out["final_sources"]) == 18


def test_merge_research_preserves_pre_seeded_priority():
    state = {
        "pre_seeded_sources": [
            {"url": "https://demo.example.com/x", "title": "demo"},
        ],
        "sub_results": [
            {"sub_kind": "arxiv", "sources": [
                {"url": "https://arxiv.org/abs/9", "title": "arxiv"}],
             "error": ""},
        ],
    }
    out = merge_research(state)
    assert out["final_sources"][0]["title"] == "demo"
    assert out["final_sources"][1]["title"] == "arxiv"


def test_merge_research_collects_sub_errors():
    state = {
        "pre_seeded_sources": [],
        "sub_results": [
            {"sub_kind": "arxiv", "sources": [], "error": "arxiv died"},
            {"sub_kind": "github", "sources": [], "error": "github 403"},
            {"sub_kind": "web", "sources": [], "error": ""},
        ],
    }
    out = merge_research(state)
    errs = out["errors"]  # type: ignore[index]
    assert "arxiv died" in errs
    assert "github 403" in errs
    # web sub had no error → still no entry.
    assert len(errs) == 2


# ---------------------------------------------------------------------------
# Full subgraph integration: parallel dispatch + merge
# ---------------------------------------------------------------------------

def test_full_subgraph_runs_in_parallel(monkeypatch):
    """arxiv + github + web run, merge dedupes and caps."""
    import asyncio

    async def fake_arxiv(state):
        return {"sub_results": [{"sub_kind": "arxiv",
                                  "sources": [{"url": "https://arxiv.org/a",
                                                "title": "a"}],
                                  "error": ""}]}

    async def fake_github(state):
        return {"sub_results": [{"sub_kind": "github",
                                  "sources": [{"url": "https://github.com/x/y",
                                                "title": "g"}],
                                  "error": ""}]}

    async def fake_web(state):
        return {"sub_results": [{"sub_kind": "web",
                                  "sources": [{"url": "https://blog.example.com/p",
                                                "title": "w"}],
                                  "error": ""}]}

    monkeypatch.setattr("src.argus.graph.researcher_subgraph.arxiv_sub",
                         fake_arxiv)
    monkeypatch.setattr("src.argus.graph.researcher_subgraph.github_sub",
                         fake_github)
    monkeypatch.setattr("src.argus.graph.researcher_subgraph.web_sub",
                         fake_web)

    sg = build_researcher_subgraph()
    out = asyncio.run(sg.ainvoke({
        "user_request": "topic",
        "plan": ResearchPlan(must_have_keywords=["x"]).model_dump(),
        "pre_seeded_sources": [],
        "sub_results": [],
        "final_sources": [],
    }))
    urls = [s["url"] for s in out["final_sources"]]
    assert "https://arxiv.org/a" in urls
    assert "https://github.com/x/y" in urls
    assert "https://blog.example.com/p" in urls
    assert out["errors"] == []


# ---------------------------------------------------------------------------
# run_researcher_subgraph: end-to-end wrapper contract
# ---------------------------------------------------------------------------

def test_run_researcher_subgraph_returns_legacy_shape(monkeypatch):
    async def fake_arxiv(state):
        return {"sub_results": [{"sub_kind": "arxiv",
                                  "sources": [{"url": "https://arxiv.org/zz",
                                                "title": "z",
                                                "kind": "paper"}],
                                  "error": ""}]}
    async def fake_github(state):
        return {"sub_results": [{"sub_kind": "github", "sources": [],
                                  "error": "github 403"}]}
    async def fake_web(state):
        return {"sub_results": [{"sub_kind": "web", "sources": [],
                                  "error": ""}]}
    monkeypatch.setattr("src.argus.graph.researcher_subgraph.arxiv_sub",
                         fake_arxiv)
    monkeypatch.setattr("src.argus.graph.researcher_subgraph.github_sub",
                         fake_github)
    monkeypatch.setattr("src.argus.graph.researcher_subgraph.web_sub",
                         fake_web)

    diff = run_researcher_subgraph({
        "user_request": "anything",
        "plan": ResearchPlan(must_have_keywords=["x"]).model_dump(),
        "thread_id": "t1",
    })
    # Legacy researcher_node contract:
    assert "sources" in diff
    assert "messages" in diff
    assert diff["sources"][0]["url"] == "https://arxiv.org/zz"
    assert "github 403" in diff["errors"]
    assert diff["messages"][0]["role"] == "assistant"


def test_run_researcher_subgraph_pre_seeds_pass_through(monkeypatch):
    async def noop(state):
        return {"sub_results": [{"sub_kind": "arxiv", "sources": [],
                                  "error": ""}]}
    async def noop_g(state):
        return {"sub_results": [{"sub_kind": "github", "sources": [],
                                  "error": ""}]}
    async def noop_w(state):
        return {"sub_results": [{"sub_kind": "web", "sources": [],
                                  "error": ""}]}
    monkeypatch.setattr("src.argus.graph.researcher_subgraph.arxiv_sub", noop)
    monkeypatch.setattr("src.argus.graph.researcher_subgraph.github_sub", noop_g)
    monkeypatch.setattr("src.argus.graph.researcher_subgraph.web_sub", noop_w)

    diff = run_researcher_subgraph({
        "user_request": "x",
        "plan": {},
        "thread_id": "t1",
        "sources": [{"url": "https://demo.example.com/p",
                      "title": "demo", "kind": "search_result"}],
    })
    assert diff["sources"][0]["url"] == "https://demo.example.com/p"