"""Hermetic tests for ``argus.tools.search_web`` and the two backend
implementations (``perplexity_search``, ``ddgs_search``) it dispatches to.

Handoff background (HANDOFF-RESEARCH-2026-07-08.md §3 + §6, action #4):
Argus's planner is bad at producing real URLs. The fix is to give it a
tool to find them. ``search_web`` is a thin unified wrapper that returns
``list[{url, title, snippet, source}]`` so the graph nodes can substitute
planner URLs with real ones.

We pin four behavioral contracts here so future refactors don't drift:

1. ``search_web`` returns a list of dicts with the four required keys
   (``url``, ``title``, ``snippet``, ``source``).
2. Empty input -> ``[]`` (no exception, no network call).
3. ``engine='perplexity'`` with no ``PERPLEXITY_API_KEY`` -> ``[]`` and a
   logged warning (we never crash the graph for a missing optional key).
4. ``engine='ddgs'`` returns the DDGS result list verbatim (mapped through
   the unified schema).

Everything is mocked: Perplexity HTTP via a fake ``urlopen`` (monkeypatched
onto ``argus.tools.urllib.request.urlopen``) and DDGS via a fake class
injected into ``argus.tools.DDGS``. No network. No real library calls.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from argus import tools


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeDDGS:
    """Drop-in fake for ``duckduckgo_search.DDGS`` that yields a canned list."""

    def __init__(self, results: list[dict[str, str]] | None = None) -> None:
        self.results = results or [
            {
                "href": "https://example.com/a",
                "title": "Example A",
                "body": "Snippet A",
            },
            {
                "href": "https://example.com/b",
                "title": "Example B",
                "body": "Snippet B",
            },
        ]
        self.last_text_kwargs: dict[str, Any] = {}  # captured for inspection

    def text(self, query: str, max_results: int | None = None,
             **_kwargs: Any) -> list[dict[str, str]]:
        # Capture all kwargs so tests can assert on timelimit/etc.
        self.last_text_kwargs = {"max_results": max_results, **_kwargs}
        return list(self.results)[: max_results] if max_results else list(self.results)


class _CapturingDDGS(_FakeDDGS):
    """Records every text() call (query + kwargs) so tests can assert."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[dict[str, Any]] = []

    def text(self, query: str, max_results: int | None = None,
             **_kwargs: Any) -> list[dict[str, str]]:
        self.calls.append({"query": query, "max_results": max_results, **_kwargs})
        return super().text(query, max_results=max_results, **_kwargs)


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# 1. Schema contract
# ---------------------------------------------------------------------------


def test_search_web_returns_dicts_with_required_keys(monkeypatch):
    """search_web('ddgs') returns list of {url,title,snippet,source} dicts."""
    monkeypatch.setattr(tools, "DDGS", lambda: _FakeDDGS())

    out = tools.search_web("anything", engine="ddgs")

    assert isinstance(out, list)
    assert out, "expected at least one result from the fake DDGS"
    for row in out:
        assert set(row.keys()) == {"url", "title", "snippet", "source"}
        assert row["source"] == "ddgs"
        assert row["url"].startswith("http")
        assert row["title"]
        assert row["snippet"]


# ---------------------------------------------------------------------------
# 2. Empty input -> []
# ---------------------------------------------------------------------------


def test_search_web_empty_query_returns_empty_list(monkeypatch):
    """Empty / whitespace input must short-circuit to [] without calling backends."""

    called = {"ddgs": 0, "perplexity": 0}

    def _fake_ddgs_factory() -> _FakeDDGS:
        called["ddgs"] += 1
        return _FakeDDGS()

    def _fake_urlopen(*a: Any, **kw: Any) -> Any:
        called["perplexity"] += 1
        return _FakeResponse({"results": []})

    monkeypatch.setattr(tools, "DDGS", _fake_ddgs_factory)
    monkeypatch.setattr(tools.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")

    assert tools.search_web("", engine="ddgs") == []
    assert tools.search_web("   ", engine="ddgs") == []
    assert tools.search_web("", engine="perplexity") == []

    assert called["ddgs"] == 0, "DDGS must not be invoked for empty queries"
    assert called["perplexity"] == 0, "Perplexity must not be invoked for empty queries"


# ---------------------------------------------------------------------------
# 3. perplexity without API key -> [] + warning (NOT exception)
# ---------------------------------------------------------------------------


def test_perplexity_without_api_key_returns_empty_and_warns(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    """No PERPLEXITY_API_KEY in env -> [] + a logged warning, never a raise."""
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

    # If we accidentally hit the network, the test fails loudly.
    def _explode(*a: Any, **kw: Any) -> Any:
        raise AssertionError("urlopen must NOT be called when API key is missing")

    monkeypatch.setattr(tools.urllib.request, "urlopen", _explode)

    with caplog.at_level(logging.WARNING, logger="argus.tools"):
        out = tools.search_web("anything", engine="perplexity")

    assert out == []
    assert any("PERPLEXITY_API_KEY" in rec.message for rec in caplog.records), (
        "expected a warning mentioning PERPLEXITY_API_KEY, got: "
        + repr([r.message for r in caplog.records])
    )


def test_perplexity_with_api_key_parses_response(monkeypatch) -> None:
    """With a key, perplexity HTTP path returns the unified schema."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")

    api_payload = {
        "results": [
            {
                "title": "Perplexity Hit 1",
                "url": "https://pplx.example/1",
                "snippet": "first snippet",
            },
            {
                "title": "Perplexity Hit 2",
                "url": "https://pplx.example/2",
                "snippet": "second snippet",
            },
        ]
    }

    captured: dict[str, Any] = {}

    def _fake_urlopen(req: Any, timeout: float = 0) -> Any:
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = req.data.decode("utf-8") if req.data else ""
        captured["timeout"] = timeout
        return _FakeResponse(api_payload)

    monkeypatch.setattr(tools.urllib.request, "urlopen", _fake_urlopen)

    out = tools.search_web("hello", engine="perplexity", max_results=5)

    assert captured["url"].startswith("https://api.perplexity.ai/"), captured
    assert "test-key" in captured["headers"].get("Authorization", "")
    assert json.loads(captured["body"])["query"] == "hello"
    assert captured["timeout"] >= 1

    assert len(out) == 2
    assert {r["source"] for r in out} == {"perplexity"}
    for row in out:
        assert set(row.keys()) == {"url", "title", "snippet", "source"}
    assert out[0]["url"] == "https://pplx.example/1"


def test_perplexity_http_failure_returns_empty(monkeypatch) -> None:
    """Network errors must degrade to [] (logged warning), not crash the graph."""
    import urllib.error

    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")

    def _fake_urlopen(*a: Any, **kw: Any) -> Any:
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(tools.urllib.request, "urlopen", _fake_urlopen)

    out = tools.search_web("anything", engine="perplexity")
    assert out == []


# ---------------------------------------------------------------------------
# 4. ddgs returns the (mapped) mocked result
# ---------------------------------------------------------------------------


def test_ddgs_returns_mapped_results(monkeypatch) -> None:
    """ddgs_search uses DDGS().text(...) and maps {href,title,body} -> schema."""
    monkeypatch.setattr(tools, "DDGS", lambda: _FakeDDGS())

    out = tools.ddgs_search("hi", max_results=10)

    assert len(out) == 2
    assert out[0] == {
        "url": "https://example.com/a",
        "title": "Example A",
        "snippet": "Snippet A",
        "source": "ddgs",
    }
    assert out[1]["url"] == "https://example.com/b"


def test_ddgs_failure_returns_empty(monkeypatch) -> None:
    """DDGS exceptions must degrade to [] (logged warning), not crash the graph."""

    class _BoomDDGS:
        def text(self, *_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("duckduckgo blocked us")

    monkeypatch.setattr(tools, "DDGS", lambda: _BoomDDGS())

    out = tools.ddgs_search("hi")
    assert out == []


# ---------------------------------------------------------------------------
# Misc: dispatch + default engine
# ---------------------------------------------------------------------------


def test_unknown_engine_returns_empty_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown engine -> [] with a warning (fail-soft, never raise)."""
    with caplog.at_level(logging.WARNING, logger="argus.tools"):
        out = tools.search_web("x", engine="bogus-engine")

    assert out == []
    assert any("bogus-engine" in rec.message for rec in caplog.records)


def test_default_engine_is_ddgs(monkeypatch) -> None:
    """search_web's default engine must be 'ddgs' (free, no key needed)."""
    monkeypatch.setattr(tools, "DDGS", lambda: _FakeDDGS())

    # Don't pass engine; should still hit DDGS path.
    out = tools.search_web("test")
    assert out and out[0]["source"] == "ddgs"


def test_max_results_respected(monkeypatch) -> None:
    """max_results is forwarded to DDGS and clamped."""
    captured: dict[str, Any] = {}

    class _SpyDDGS:
        def text(self, query: str, max_results: int | None = None,
                 **_kw: Any) -> list[dict[str, str]]:
            captured["max_results"] = max_results
            return [{"href": "https://x", "title": "t", "body": "b"}]

    monkeypatch.setattr(tools, "DDGS", lambda: _SpyDDGS())

    tools.search_web("hi", engine="ddgs", max_results=3)
    assert captured["max_results"] == 3


# ---------------------------------------------------------------------------
# 5. ddgs_search timelimit (lifted from NVIDIA AI-Q Blueprint
#    sources/duckduckgo_news_search/src/register.py)
# ---------------------------------------------------------------------------
# AI-Q's news-search wrapper forwards ``timelimit='d' | 'w' | 'm' | 'y' | None``
# to duckduckgo_search so news queries can scope to recent results. Argus
# currently has ddgs_search but doesn't expose the timelimit parameter.
# Adding it is a 3-line change with real value: news queries become time-scoped.


@pytest.mark.parametrize("timelimit,expected", [
    ("d", "d"),
    ("w", "w"),
    ("m", "m"),
    ("y", "y"),
    (None, None),
    ("", None),  # empty string means "no timelimit" (per duckduckgo_search)
])
def test_ddgs_search_forwards_timelimit(monkeypatch, timelimit, expected):
    """ddgs_search must forward the ``timelimit`` kwarg to DDGS().text()."""
    captured_kwargs: dict[str, Any] = {}

    class _SpyDDGS:
        def text(self, query: str, max_results: int | None = None,
                 **kwargs: Any) -> list[dict[str, str]]:
            captured_kwargs.update(kwargs)
            return [{"href": "https://x", "title": "t", "body": "b"}]

    monkeypatch.setattr(tools, "DDGS", lambda: _SpyDDGS())

    tools.ddgs_search("ai breakthrough", max_results=5, timelimit=timelimit)

    assert "timelimit" in captured_kwargs, (
        "ddgs_search must forward the timelimit kwarg to DDGS().text() — "
        "lifted from AI-Q's duckduckgo_news_search."
    )
    assert captured_kwargs["timelimit"] == expected, (
        f"timelimit not passed through correctly: "
        f"input={timelimit!r} got={captured_kwargs['timelimit']!r}"
    )


def test_ddgs_search_timelimit_rejects_invalid_value():
    """Invalid timelimit values (anything other than d/w/m/y/None) must
    raise ValueError so callers don't silently send garbage to DDGS."""
    for bad in ["z", "week", "7", "hour", 5, True]:
        with pytest.raises(ValueError):
            tools.ddgs_search("anything", timelimit=bad)


def test_search_web_forwards_timelimit(monkeypatch):
    """The unified ``search_web(engine='ddgs', timelimit=...)`` API must also
    accept and forward timelimit, so callers don't need to know about
    engine-specific kwargs."""
    captured_kwargs: dict[str, Any] = {}

    class _SpyDDGS:
        def text(self, query: str, max_results: int | None = None,
                 **kwargs: Any) -> list[dict[str, str]]:
            captured_kwargs.update(kwargs)
            return [{"href": "https://x", "title": "t", "body": "b"}]

    monkeypatch.setattr(tools, "DDGS", lambda: _SpyDDGS())

    tools.search_web("news", engine="ddgs", timelimit="w")
    assert captured_kwargs.get("timelimit") == "w", (
        "search_web(engine='ddgs', timelimit=...) must forward the kwarg."
    )
