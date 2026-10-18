"""Hermetic tests for ``argus.tools.ddgs_search``.

DDGS is faked by a class injected into ``argus.tools.DDGS``: no network,
no real library calls. Pinned contracts:

1. Hits are mapped to the unified ``{url, title, snippet, source}`` schema
   (``source == "ddgs"``).
2. A DDGS failure degrades to ``[]`` (logged warning), never a raise.
3. ``max_results`` and ``timelimit`` are forwarded to ``DDGS().text(...)``;
   an invalid ``timelimit`` raises ``ValueError``.
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# 1. ddgs returns the (mapped) mocked result
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
# 2. max_results is forwarded
# ---------------------------------------------------------------------------


def test_max_results_respected(monkeypatch) -> None:
    """max_results is forwarded to DDGS and clamped."""
    captured: dict[str, Any] = {}

    class _SpyDDGS:
        def text(self, query: str, max_results: int | None = None,
                 **_kw: Any) -> list[dict[str, str]]:
            captured["max_results"] = max_results
            return [{"href": "https://x", "title": "t", "body": "b"}]

    monkeypatch.setattr(tools, "DDGS", lambda: _SpyDDGS())

    tools.ddgs_search("hi", max_results=3)
    assert captured["max_results"] == 3


# ---------------------------------------------------------------------------
# 3. ddgs_search timelimit (lifted from NVIDIA AI-Q Blueprint
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
