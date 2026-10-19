"""Search (v4 spec §2.8 ``search.run_wave``) and the scout.

All HTTP is mocked. Verifies: provider hit shapes, the per-mode Exa budget
through the SpendLedger (over-budget queries run on the web provider, one
``search:`` line per wave), ``exa_text_pages`` / ``exa_usd``, the serialised
and capped arXiv rung, ``github_mode``, canonical dedupe, error isolation, a
DDGS failure as an errors line, and the scout's ``scout.querygen`` degrade.

Run with:
    PYTHONPATH='' ./venv/Scripts/python.exe -m pytest tests/test_search_providers.py -q
"""
from __future__ import annotations

import threading
import time

import pytest

from argus import llm, tools
from argus.engines import registry, search
from argus.engines.ledger import SpendLedger
from argus.engines.llm_openai import DriverResponse
from argus.engines.schema import ProviderSpec
from argus.graph import scout as scout_mod
from argus.graph import search_providers as sp
from argus.graph.run_config import RunConfig
from argus.graph.state import ResearchBrief, SubQuestion

from helpers import make_run_config


def _rc(length="medium", **overrides) -> RunConfig:
    return RunConfig.model_validate(make_run_config(length, **overrides))


def _hit(url, sub_qs=None, text="", provider="ddgs"):
    return {"url": url, "title": "t", "snippet": "s", "kind": "blog", "provider": provider,
            "sub_qs": list(sub_qs or []), "text": text, "published": "", "source": provider,
            "summary": "s"}


@pytest.fixture
def fake_providers(monkeypatch):
    """Every provider function replaced by a recorder; returns the call log."""
    log: list[tuple[str, str, dict]] = []

    def make(name):
        def fn(query, **kw):
            log.append((name, query, kw))
            return [_hit(f"https://{name}.example/{query.replace(' ', '-')}", kw.get("sub_qs"),
                         provider=name)], ""
        return fn

    for name, attr in (("exa", "exa_search"), ("ddgs", "ddgs_hits"),
                       ("arxiv", "arxiv_hits"), ("github", "github_hits")):
        monkeypatch.setattr(sp, attr, make(name))
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(search, "_sleep", lambda s: None)
    return log


def _ran(log, name):
    return [q for n, q, _ in log if n == name]


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


def test_exa_queries_run_on_ddgs_without_a_key(fake_providers, monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    rc = _rc()
    hits, errors = search.run_wave([{"query": "q1", "provider": "exa", "sub_qs": [0]}],
                                   cfg=rc.search, ledger=SpendLedger.for_run(rc, None), recency=None,
                                   mode="first")
    assert _ran(fake_providers, "ddgs") == ["q1"] and _ran(fake_providers, "exa") == []
    assert len(hits) == 1 and errors == [], "Exa not configured is a state, not a failure"


def test_exa_budget_per_mode_through_the_ledger(fake_providers):
    rc = _rc()
    ledger = SpendLedger({"exa_search_first": 2})
    queries = [{"query": f"q{i}", "provider": "exa", "sub_qs": [i]} for i in range(4)]
    _, errors = search.run_wave(queries, cfg=rc.search, ledger=ledger, recency=None, mode="first")
    assert sorted(_ran(fake_providers, "exa")) == ["q0", "q1"], "the budget goes in query order"
    assert sorted(_ran(fake_providers, "ddgs")) == ["q2", "q3"]
    assert errors == ["search: exa budget (first pass) spent — 2 queries ran on ddgs"]
    assert ledger.snapshot()["exa_searches"] == 2


def test_a_gap_wave_spends_at_most_two_exa_searches(fake_providers):
    rc = _rc()
    ledger = SpendLedger({"exa_search_gap": 6})
    queries = [{"query": f"g{i}", "provider": "exa", "sub_qs": [i]} for i in range(4)]
    _, errors = search.run_wave(queries, cfg=rc.search, ledger=ledger, recency=None, mode="gap")
    assert len(_ran(fake_providers, "exa")) == 2 and len(_ran(fake_providers, "ddgs")) == 2
    assert ledger.remaining("exa_search_gap") == 4, "the reserve for later gap waves is kept"
    assert errors == ["search: exa budget (gap wave) spent — 2 queries ran on ddgs"]


def test_exa_text_pages_and_cost_are_counted(monkeypatch):
    rc = _rc()
    monkeypatch.setenv("EXA_API_KEY", "test-key")

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"url": "https://a.example/1", "title": "a", "text": "t" * 500},
                                {"url": "https://a.example/2", "title": "b", "text": "u" * 500},
                                {"url": "https://a.example/3", "title": "c", "text": ""}],
                    "costDollars": {"total": 0.009}}

    sent = []

    class _Client:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json=None, headers=None):
            sent.append(json)
            return _Resp()

    monkeypatch.setattr(sp.httpx, "Client", _Client)
    ledger = SpendLedger.for_run(rc, None)
    hits, errors = search.run_wave([{"query": "q", "provider": "exa", "sub_qs": [0]}], cfg=rc.search,
                                   ledger=ledger, recency=None, mode="first")
    assert errors == [] and len(hits) == 3
    snap = ledger.snapshot()
    assert (snap["exa_searches"], snap["exa_text_pages"]) == (1, 2), (
        "the hits that came with inline text (counted; the live check found them not billed)")
    assert snap["exa_usd"] == pytest.approx(0.009)
    body = sent[0]
    assert body["type"] == rc.search.exa.search_type and body["numResults"] == rc.search.exa.num_results
    assert body["contents"]["text"]["maxCharacters"] == rc.search.exa.text_max_chars


# ---------------------------------------------------------------------------
# arXiv, GitHub, DDGS
# ---------------------------------------------------------------------------

def test_arxiv_is_serialised_with_a_min_interval_and_capped_per_run(fake_providers, monkeypatch):
    rc = _rc(**{"search.arxiv.max_queries_per_run": 2})
    in_flight, peak, sleeps = [0], [0], []
    lock = threading.Lock()

    def arxiv(query, **kw):
        with lock:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
        time.sleep(0.02)
        with lock:
            in_flight[0] -= 1
        fake_providers.append(("arxiv", query, kw))
        return [_hit(f"https://arxiv.org/abs/2401.0000{query[-1]}", kw.get("sub_qs"), provider="arxiv")], ""

    monkeypatch.setattr(sp, "arxiv_hits", arxiv)
    monkeypatch.setattr(search, "_sleep", sleeps.append)
    monkeypatch.setattr(search, "_arxiv_last", [0.0])
    ledger = SpendLedger.for_run(rc, None)
    queries = [{"query": f"p{i}", "provider": "arxiv", "sub_qs": [0]} for i in range(4)]
    _, errors = search.run_wave(queries, cfg=rc.search, ledger=ledger, recency=None, mode="first")
    assert len(_ran(fake_providers, "arxiv")) == 2 and peak[0] == 1, "one arXiv request at a time"
    assert sleeps and all(s <= rc.search.arxiv.min_interval_s for s in sleeps)
    assert len(_ran(fake_providers, "exa")) == 2, "over-cap paper queries run on the web provider"
    assert errors == ["search: arxiv cap (2 per run) reached — 2 queries ran on exa"]
    assert ledger.snapshot()["arxiv_queries"] == 2


@pytest.mark.parametrize("mode,expect", [("exa_domain", "exa"), ("repo_search", "github"), ("off", "exa")])
def test_github_mode_routes_repo_queries(fake_providers, mode, expect):
    rc = _rc(**{"search.github_mode": mode})
    ledger = SpendLedger.for_run(rc, None)
    search.run_wave([{"query": "flash attention repo", "provider": "github", "sub_qs": [1]}],
                    cfg=rc.search, ledger=ledger, recency=None, mode="first")
    (name, _q, kw), = fake_providers
    assert name == expect
    if mode == "exa_domain":
        assert kw["include_domains"] == ["github.com"]
    if mode == "off":
        assert not kw.get("include_domains")
    assert ledger.snapshot()["github_queries"] == (1 if expect == "github" else 0)


def test_ddgs_failure_is_an_errors_line(monkeypatch):
    class _Boom:
        def text(self, *a, **kw):
            raise RuntimeError("duckduckgo blocked us")

    monkeypatch.setattr(tools, "DDGS", lambda: _Boom())
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    rc = _rc()
    ledger = SpendLedger.for_run(rc, None)
    hits, errors = search.run_wave([{"query": "x", "provider": "ddgs", "sub_qs": [0]}], cfg=rc.search,
                                   ledger=ledger, recency=None, mode="first")
    assert hits == []
    assert errors == ["ddgs: RuntimeError: duckduckgo blocked us"]
    assert ledger.snapshot()["ddgs_queries"] == 1


# ---------------------------------------------------------------------------
# Wave merge semantics
# ---------------------------------------------------------------------------

def test_wave_dedupes_by_canonical_url_and_merges_subq_tags(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    urls = iter(["https://www.Same.example/a/?utm_source=x", "http://same.example/a#frag"])
    monkeypatch.setattr(sp, "ddgs_hits", lambda query, **kw: ([_hit(next(urls), kw.get("sub_qs"))], ""))
    rc = _rc()
    hits, _ = search.run_wave([{"query": "a", "provider": "ddgs", "sub_qs": [0]},
                               {"query": "b", "provider": "ddgs", "sub_qs": [1]}],
                              cfg=rc.search, ledger=SpendLedger.for_run(rc, None), recency=None, mode="first")
    assert len(hits) == 1
    assert sorted(hits[0]["sub_qs"]) == [0, 1]
    assert hits[0]["canonical_url"] == "https://same.example/a"


def test_wave_prefers_hit_with_text_on_dupe(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    texts = iter(["", "full page text " * 30])
    monkeypatch.setattr(sp, "ddgs_hits", lambda query, **kw: (
        [_hit("https://same.example", kw.get("sub_qs"), text=next(texts))], ""))
    rc = _rc()
    hits, _ = search.run_wave([{"query": "a", "provider": "ddgs", "sub_qs": [0]},
                               {"query": "b", "provider": "ddgs", "sub_qs": [1]}],
                              cfg=rc.search, ledger=SpendLedger.for_run(rc, None), recency=None, mode="first")
    assert len(hits) == 1
    assert hits[0]["text"], "the text-bearing duplicate must win"
    assert sorted(hits[0]["sub_qs"]) == [0, 1]


def test_wave_error_isolation(monkeypatch):
    """One provider's failure surfaces as an error string while the other
    providers' hits ship — the v2 failure-isolation property."""
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(sp, "ddgs_hits", lambda query, **kw: ([_hit("https://ok.example")], ""))
    monkeypatch.setattr(sp, "arxiv_hits", lambda query, **kw: ([], "arxiv: ConnectError: boom"))
    monkeypatch.setattr(search, "_sleep", lambda s: None)
    rc = _rc()
    hits, errors = search.run_wave([{"query": "a", "provider": "ddgs", "sub_qs": [0]},
                                    {"query": "b", "provider": "arxiv", "sub_qs": [1]}],
                                   cfg=rc.search, ledger=SpendLedger.for_run(rc, None), recency=None,
                                   mode="first")
    assert [h["url"] for h in hits] == ["https://ok.example"]
    assert errors == ["arxiv: ConnectError: boom"]


def test_run_query_wave_is_a_thin_alias():
    assert sp.run_query_wave is search.run_wave


# ---------------------------------------------------------------------------
# Scout query planning
# ---------------------------------------------------------------------------

def _brief() -> ResearchBrief:
    return ResearchBrief(
        sub_questions=[
            SubQuestion(q="What attention architectures exist?", kind="paper"),
            SubQuestion(q="Which repos implement flash attention?", kind="repo"),
            SubQuestion(q="What are current benchmark results?", kind="web"),
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


def _resp(text: str) -> DriverResponse:
    return DriverResponse(text=text, served_model="m", routed_via="", finish_reason="stop",
                          usage={"prompt_tokens": 1, "completion_tokens": 1}, headers={}, raw=None)


def test_scout_querygen_degrade_records_the_line_and_the_attempts(monkeypatch):
    """scout.querygen exhausted → the declared degradation (today's
    sub-question text queries), recorded in errors with the facade message
    and every attempt kept in model_calls."""
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(registry, "provider", lambda name: ProviderSpec(name=name, base_url="http://fake/v1"))

    class _Driver:
        def complete(self, engine, provider, messages, **kw):
            return _resp("Let me think about the queries first…")       # never JSON

    monkeypatch.setattr(llm, "_driver", _Driver())
    rc = _rc()
    queries, calls, errors = scout_mod.plan_queries(_brief(), rc=rc, rnd=1)
    assert queries == scout_mod.fallback_queries(_brief())
    assert len(calls) == 2 and all(c["role"] == "scout.querygen" and not c["ok"] for c in calls)
    assert calls[-1]["error"].startswith("unparseable: ValueError: ")
    assert errors == ["[r1] degrade: scout.querygen → subquestion_text_queries (role scout.querygen: "
                      f"every engine failed after 2 attempts (last: {calls[-1]['error']}))"]


def test_scout_querygen_keeps_the_llm_queries(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(registry, "provider", lambda name: ProviderSpec(name=name, base_url="http://fake/v1"))

    class _Driver:
        def complete(self, engine, provider, messages, **kw):
            return _resp('{"queries": [{"sub_q": 0, "q": "attention variants survey", "provider": "arxiv"},'
                         ' {"sub_q": 2, "q": "llm benchmark 2026", "provider": "web"},'
                         ' {"sub_q": 9, "q": "out of range", "provider": "web"}]}')

    monkeypatch.setattr(llm, "_driver", _Driver())
    queries, calls, errors = scout_mod.plan_queries(_brief(), rc=_rc(), rnd=1)
    assert errors == [] and [c["role"] for c in calls] == ["scout.querygen"]
    assert {"query": "attention variants survey", "provider": "arxiv", "sub_qs": [0]} in queries
    assert {"query": "llm benchmark 2026", "provider": "ddgs", "sub_qs": [2]} in queries
    assert {i for q in queries for i in q["sub_qs"]} == {0, 1, 2}, "a skipped sub-question gets its own query"


def test_scout_node_merges_preseeded_first_and_writes_spend(monkeypatch):
    monkeypatch.setattr(scout_mod, "plan_queries",
                        lambda brief, **kw: ([{"query": "q", "provider": "ddgs", "sub_qs": [0]}], [], []))
    seen = {}

    def wave(queries, **kw):
        seen.update(kw)
        kw["ledger"].count("ddgs_queries")
        return [_hit("https://new.example"), _hit("https://www.seeded.example/?utm_source=x")], []

    monkeypatch.setattr(scout_mod, "run_query_wave", wave)
    out = scout_mod.scout_node({
        "brief": _brief().model_dump(), "run_config": make_run_config("medium"),
        "sources": [{"url": "https://seeded.example", "title": "seed"}],
    })
    urls = [s["url"] for s in out["sources"]]
    assert urls == ["https://seeded.example", "https://new.example"], "seeded first, canonical dedupe"
    assert out["plan"]["planned_sources"], "plan gets executed queries"
    assert seen["mode"] == "first" and seen["recency"] is None
    assert out["search_spend"]["ddgs_queries"] == 1


def test_scout_node_caps_sources(monkeypatch):
    monkeypatch.setattr(scout_mod, "plan_queries",
                        lambda brief, **kw: ([{"query": "q", "provider": "ddgs", "sub_qs": [0]}], [], []))
    many = [_hit(f"https://h{i}.example") for i in range(80)]
    monkeypatch.setattr(scout_mod, "run_query_wave", lambda queries, **kw: (many, []))
    out = scout_mod.scout_node({"brief": _brief().model_dump(), "run_config": make_run_config("medium")})
    assert len(out["sources"]) == scout_mod.SCOUT_SOURCE_CAP
