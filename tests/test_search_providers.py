"""v3 search-provider tests (successor of test_researcher_subgraph.py).

All HTTP is mocked. Verifies: provider hit shapes, wave merge/dedupe,
error isolation (one provider's failure never poisons the wave), Exa
enablement/budget downgrade, and the scout's deterministic fallbacks.

Run with:
    PYTHONPATH='' ./venv/Scripts/python.exe -m pytest tests/test_search_providers.py -q
"""
from __future__ import annotations

import pytest

from argus.graph import search_providers as sp
from argus.graph import scout as scout_mod
from argus.graph.state import ResearchBrief, SubQuestion


# ---------------------------------------------------------------------------
# Exa enablement + budget
# ---------------------------------------------------------------------------

def test_exa_disabled_without_key(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    assert sp.exa_enabled() is False
    hits, err = sp.exa_search("anything")
    assert hits == []
    assert "EXA_API_KEY" in err


def test_exa_enabled_with_key(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    assert sp.exa_enabled() is True


def test_wave_downgrades_exa_to_ddgs_without_key(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    seen: list[str] = []

    def fake_ddgs(query, **kw):
        seen.append(query)
        return ([{"url": "https://x.example", "title": "t",
                  "snippet": "s", "kind": "blog", "provider": "ddgs",
                  "sub_qs": kw.get("sub_qs") or [], "text": "",
                  "published": "", "source": "ddgs", "summary": "s"}], "")

    monkeypatch.setattr(sp, "ddgs_hits", fake_ddgs)
    hits, errors = sp.run_query_wave(
        [{"query": "q1", "provider": "exa", "sub_qs": [0]}])
    assert seen == ["q1"], "exa query must downgrade to ddgs without a key"
    assert len(hits) == 1
    assert errors == []


def test_wave_respects_exa_budget(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    exa_calls: list[str] = []
    ddgs_calls: list[str] = []

    def fake_exa(query, **kw):
        exa_calls.append(query)
        return ([], "")

    def fake_ddgs(query, **kw):
        ddgs_calls.append(query)
        return ([], "")

    monkeypatch.setattr(sp, "exa_search", fake_exa)
    monkeypatch.setattr(sp, "ddgs_hits", fake_ddgs)
    queries = [{"query": f"q{i}", "provider": "exa", "sub_qs": [i]}
               for i in range(4)]
    sp.run_query_wave(queries, exa_budget=2)
    assert len(exa_calls) == 2
    assert len(ddgs_calls) == 2


# ---------------------------------------------------------------------------
# Wave merge semantics
# ---------------------------------------------------------------------------

def _hit(url, sub_qs=None, text=""):
    return {"url": url, "title": "t", "snippet": "s", "kind": "blog",
            "provider": "ddgs", "sub_qs": sub_qs or [], "text": text,
            "published": "", "source": "ddgs", "summary": "s"}


def test_wave_dedupes_by_url_and_merges_subq_tags(monkeypatch):
    def fake_ddgs(query, **kw):
        return ([_hit("https://same.example", sub_qs=kw.get("sub_qs"))], "")

    monkeypatch.setattr(sp, "ddgs_hits", fake_ddgs)
    hits, _ = sp.run_query_wave([
        {"query": "a", "provider": "ddgs", "sub_qs": [0]},
        {"query": "b", "provider": "ddgs", "sub_qs": [1]},
    ])
    assert len(hits) == 1
    assert sorted(hits[0]["sub_qs"]) == [0, 1]


def test_wave_error_isolation(monkeypatch):
    """One provider's failure must surface as an error string while the
    other providers' hits ship — the v2 failure-isolation property."""
    def fake_ddgs(query, **kw):
        return ([_hit("https://ok.example")], "")

    def fake_arxiv(query, **kw):
        return ([], "arxiv: ConnectError: boom")

    monkeypatch.setattr(sp, "ddgs_hits", fake_ddgs)
    monkeypatch.setattr(sp, "arxiv_hits", fake_arxiv)
    hits, errors = sp.run_query_wave([
        {"query": "a", "provider": "ddgs", "sub_qs": [0]},
        {"query": "b", "provider": "arxiv", "sub_qs": [1]},
    ])
    assert [h["url"] for h in hits] == ["https://ok.example"]
    assert any("arxiv" in e for e in errors)


def test_wave_prefers_hit_with_text_on_dupe(monkeypatch):
    calls = {"n": 0}

    def fake_ddgs(query, **kw):
        calls["n"] += 1
        text = "full page text " * 30 if calls["n"] == 2 else ""
        return ([_hit("https://same.example", sub_qs=kw.get("sub_qs"),
                      text=text)], "")

    monkeypatch.setattr(sp, "ddgs_hits", fake_ddgs)
    hits, _ = sp.run_query_wave([
        {"query": "a", "provider": "ddgs", "sub_qs": [0]},
        {"query": "b", "provider": "ddgs", "sub_qs": [1]},
    ])
    assert len(hits) == 1
    assert hits[0]["text"], "the text-bearing duplicate must win"


# ---------------------------------------------------------------------------
# Scout query planning
# ---------------------------------------------------------------------------

def _brief() -> ResearchBrief:
    return ResearchBrief(
        sub_questions=[
            SubQuestion(q="What attention architectures exist?",
                        kind="paper"),
            SubQuestion(q="Which repos implement flash attention?",
                        kind="repo"),
            SubQuestion(q="What are current benchmark results?",
                        kind="web"),
        ],
        must_have_keywords=["attention", "transformer"],
        summary="s")


def test_fallback_queries_cover_every_subquestion(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    queries = scout_mod.fallback_queries(_brief())
    covered = {i for q in queries for i in q["sub_qs"]}
    assert covered == {0, 1, 2}


def test_fallback_queries_route_kinds_to_providers(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    queries = scout_mod.fallback_queries(_brief())
    by_first_subq = {q["sub_qs"][0]: q["provider"] for q in queries[::-1]}
    assert by_first_subq[0] == "arxiv"
    assert by_first_subq[1] == "github"
    assert by_first_subq[2] == "ddgs"


def test_scout_node_merges_preseeded_first(monkeypatch):
    monkeypatch.setattr(scout_mod, "plan_queries",
                        lambda brief: ([{"query": "q", "provider": "ddgs",
                                         "sub_qs": [0]}], [], []))
    monkeypatch.setattr(scout_mod, "run_query_wave",
                        lambda queries: ([_hit("https://new.example")], []))
    out = scout_mod.scout_node({
        "brief": _brief().model_dump(),
        "sources": [{"url": "https://seeded.example", "title": "seed"}],
    })
    urls = [s["url"] for s in out["sources"]]
    assert urls[0] == "https://seeded.example"
    assert "https://new.example" in urls
    assert out["plan"]["planned_sources"], "plan gets executed queries"


def test_scout_node_caps_sources(monkeypatch):
    monkeypatch.setattr(scout_mod, "plan_queries",
                        lambda brief: ([{"query": "q", "provider": "ddgs",
                                         "sub_qs": [0]}], [], []))
    many = [_hit(f"https://h{i}.example") for i in range(80)]
    monkeypatch.setattr(scout_mod, "run_query_wave",
                        lambda queries: (many, []))
    out = scout_mod.scout_node({"brief": _brief().model_dump()})
    assert len(out["sources"]) == scout_mod.SCOUT_SOURCE_CAP
