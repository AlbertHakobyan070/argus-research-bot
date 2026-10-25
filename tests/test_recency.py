"""Recency (v4 spec §2.8, §3 WP-B): only for time-sensitive briefs (or an
explicit ``run_config.recency``); mapped onto Exa ``startPublishedDate``,
DDGS ``timelimit`` and arXiv ``sortBy=submittedDate``; undated hits dropped
and counted in ``dropped_undated``. Asserted against fake clients — never
live."""
from __future__ import annotations

from datetime import date

import pytest

from argus import tools
from argus.engines import search
from argus.engines.ledger import SpendLedger
from argus.graph import search_providers as sp
from argus.graph.run_config import RunConfig
from argus.graph.state import ResearchBrief

from helpers import make_run_config

TODAY = date(2026, 10, 19)
ATOM = ('<entry><title>A recent paper</title><id>http://arxiv.org/abs/2610.00001v1</id>'
        '<published>2026-10-01T00:00:00Z</published><summary>Abstract.</summary></entry>')


def _rc(**overrides) -> RunConfig:
    return RunConfig.model_validate(make_run_config("medium", **overrides))


class _Resp:
    def __init__(self, *, json_body=None, text=""):
        self._json, self.text, self.status_code = json_body, text, 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _FakeHTTP:
    """Stands in for ``httpx.Client`` in search_providers: records every call."""

    def __init__(self, exa_results):
        self.exa_results, self.calls = exa_results, []

    def __call__(self, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        self.calls.append(("POST", url, json))
        return _Resp(json_body={"results": self.exa_results, "costDollars": {"total": 0.007}})

    def get(self, url, params=None, headers=None):
        self.calls.append(("GET", url, params))
        return _Resp(text=ATOM)

    def sent(self, method):
        return [c[2] for c in self.calls if c[0] == method]


@pytest.fixture
def providers(monkeypatch):
    fake = _FakeHTTP([
        {"url": "https://news.example/a", "title": "A", "text": "x" * 400, "publishedDate": "2026-10-10"},
        {"url": "https://news.example/undated", "title": "U", "text": "y" * 400},
        {"url": "https://news.example/b", "title": "B", "text": "z" * 400, "publishedDate": "2026-10-12"},
    ])
    monkeypatch.setattr(sp.httpx, "Client", fake)
    ddgs_kwargs: dict = {}

    class _DDGS:
        def text(self, query, **kw):
            ddgs_kwargs.update(kw)
            return [{"href": "https://web.example/1", "title": "w", "body": "b"}]

    monkeypatch.setattr(tools, "DDGS", lambda: _DDGS())
    monkeypatch.setenv("EXA_API_KEY", "test-key")
    monkeypatch.setattr(search, "_sleep", lambda s: None)
    return fake, ddgs_kwargs


QUERIES = [{"query": "q exa", "provider": "exa", "sub_qs": [0]},
           {"query": "q web", "provider": "ddgs", "sub_qs": [1]},
           {"query": "q paper", "provider": "arxiv", "sub_qs": [2]}]


def test_recency_maps_onto_each_provider_and_drops_undated(providers):
    fake, ddgs_kwargs = providers
    rc = _rc()
    ledger = SpendLedger.for_run(rc, None)
    hits, errors = search.run_wave(QUERIES, cfg=rc.search, ledger=ledger,
                                   recency={"window": "m", "start_date": "2026-09-19"}, mode="first")
    assert errors == []
    (exa_body,) = fake.sent("POST")
    assert exa_body["startPublishedDate"] == "2026-09-19T00:00:00.000Z"
    assert ddgs_kwargs["timelimit"] == "m"
    (arxiv_params,) = fake.sent("GET")
    assert (arxiv_params["sortBy"], arxiv_params["sortOrder"]) == ("submittedDate", "descending")
    urls = [h["url"] for h in hits]
    assert "https://news.example/undated" not in urls, "an undated hit can't prove it is recent"
    assert {"https://news.example/a", "https://news.example/b", "https://web.example/1"} <= set(urls)
    snap = ledger.snapshot()
    assert snap["dropped_undated"] == 1
    assert next(h for h in hits if "arxiv" in h["url"])["published"] == "2026-10-01"


def test_no_recency_no_filters(providers):
    fake, ddgs_kwargs = providers
    rc = _rc()
    ledger = SpendLedger.for_run(rc, None)
    hits, _ = search.run_wave(QUERIES, cfg=rc.search, ledger=ledger, recency=None, mode="first")
    (exa_body,) = fake.sent("POST")
    assert "startPublishedDate" not in exa_body
    assert ddgs_kwargs["timelimit"] is None
    (arxiv_params,) = fake.sent("GET")
    assert arxiv_params["sortBy"] == "relevance"
    assert "https://news.example/undated" in [h["url"] for h in hits]
    assert ledger.snapshot()["dropped_undated"] == 0


def test_resolve_recency_rules():
    plain, weekly = _rc(), _rc(recency="w")
    assert search.resolve_recency(weekly, ResearchBrief(), today=TODAY) == (
        {"window": "w", "start_date": "2026-10-12"}, None), "an explicit run_config.recency wins"
    assert search.resolve_recency(
        plain, ResearchBrief(time_sensitive=True, recency_hint="m"), today=TODAY) == (
        {"window": "m", "start_date": "2026-09-19"}, None)
    assert search.resolve_recency(
        plain, ResearchBrief(time_sensitive=False, recency_hint="m"), today=TODAY) == (None, None), (
        "a hint on a brief that is not time-sensitive is not applied")
    assert search.resolve_recency(plain, ResearchBrief(time_sensitive=True), today=TODAY) == (None, None)
    assert search.resolve_recency(
        plain, ResearchBrief(time_sensitive=True, recency_hint="fortnight"), today=TODAY) == (
        None, "recency: ignoring the brief's hint 'fortnight' (expected d/w/m/y)")
    assert search.resolve_recency(_rc(recency="y"), ResearchBrief(), today=TODAY)[0]["start_date"] == "2025-10-19"
    assert search.resolve_recency(_rc(recency="d"), ResearchBrief(), today=TODAY)[0]["start_date"] == "2026-10-18"
