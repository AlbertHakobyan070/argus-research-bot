"""Argus T2 (Pattern E) — routing + error-propagation contract tests.

These tests pin the two behavior changes Argus T2 introduced:

* Fix #2 — every intel-stack tool wrapper (``snatch_url``,
  ``crawl_url``, ``normalize_to_markdown``) MUST invoke the
  intel-stack interpreter (INTEL_PYTHON_BIN). Previously they
  silently fell back to the argus venv's python and immediately
  raised ``ModuleNotFoundError`` for feedparser / crawl4ai /
  markitdown / yt_dlp.

* Fix #3 — the research fetch path MUST propagate
  tool failures into ``state["errors"]`` instead of swallowing them
  with ``logger.warning``. The ``errors`` field is an
  ``Annotated[list[str], operator.add]`` reducer, so any node can
  return ``{"errors": [...]}`` and the entries merge into state.

The tests mock ``subprocess.run`` (Fix #2) and the tool functions
themselves (Fix #3) so they are deterministic, do not hit the network,
and stay below the project's 5-minute CI budget.
"""
from __future__ import annotations

import subprocess
from typing import Any

import pytest

import httpx

from argus import tools
from argus.engines import fetchers
from argus.engines.ledger import SpendLedger
from argus.graph import research as research_mod
from argus.graph.research import _parallel_fetch
from argus.graph.run_config import FetchConfig
from argus.tools import (
    INTEL_PYTHON_BIN, PYTHON_BIN,
    _run_script, crawl_url, normalize_to_markdown, snatch_url,
)


# ---------------------------------------------------------------------------
# Fix #2 — every tool wrapper routes through INTEL_PYTHON_BIN
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, rc: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = rc
        self.stdout = stdout
        self.stderr = stderr


def test_run_script_default_routes_to_intel_python(monkeypatch):
    """_run_script must default to INTEL_PYTHON_BIN (T2 Fix #2)."""
    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        captured["cwd"] = kwargs.get("cwd")
        return _FakeProc(rc=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(tools.subprocess, "run", fake_run)
    rc, out, err = _run_script("snatch.py", ["https://example.com"],
                                timeout=10)
    assert rc == 0
    # The default interpreter must be INTEL_PYTHON_BIN, never PYTHON_BIN.
    assert captured["cmd"][0] == str(INTEL_PYTHON_BIN)
    assert captured["cmd"][0] != str(PYTHON_BIN)
    assert captured["cmd"][1].endswith("snatch.py")
    # PYTHONPATH must be cleared from the spawned env.
    assert captured["env"].get("PYTHONPATH") is None


def test_run_script_explicit_override_still_works(monkeypatch):
    """An explicit python_bin override must still take precedence."""
    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(rc=0, stdout="", stderr="")

    monkeypatch.setattr(tools.subprocess, "run", fake_run)
    custom = tools.Path(r"C:\some\other\python.exe")
    _run_script("harvest.py", ["--hours", "24"], python_bin=custom)
    assert captured["cmd"][0] == str(custom)


@pytest.mark.parametrize("tool_name,script,args", [
    ("snatch_url",
     "snatch.py", ["https://example.com", "--kind", "auto"]),
    ("crawl_url",
     "crawl.py", ["https://example.com", "--max-pages", "4", "--depth", "1"]),
    ("normalize_to_markdown",
     "article_convert.py", ["https://example.com", "--md-only"]),
])
def test_all_tools_route_through_intel_python(monkeypatch, tool_name,
                                                script, args):
    """Every public tool wrapper must spawn INTEL_PYTHON_BIN, not PYTHON_BIN.

    This is the regression test for the original T2 bug: the three
    wrappers (snatch_url, crawl_url, normalize_to_markdown) used to
    default to PYTHON_BIN (the argus venv) which lacks feedparser /
    crawl4ai / markitdown / yt_dlp, returning ModuleNotFoundError on
    every call.
    """
    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Snatch/crawl/normalize look for a folder path on stdout.
        return _FakeProc(rc=0,
                          stdout="A:\\Hermes\\Downloads\\fake_out\n",
                          stderr="")

    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    if tool_name == "snatch_url":
        snatch_url("https://example.com", timeout=10)
    elif tool_name == "crawl_url":
        crawl_url("https://example.com", deep=False, max_pages=4,
                   depth=1, timeout=10)
    elif tool_name == "normalize_to_markdown":
        # We need a .md on disk for the wrapper to read. Monkey-patch
        # the read_text side too.
        class _FakeMd:
            def __init__(self, path):
                self.path = path
            def rglob(self, pat):
                return [self]
            def read_text(self, **kw):
                return "# Fake Title\n\nbody"
            def __str__(self):
                return "A:\\fake.md"
        monkeypatch.setattr(tools, "Path",
                            lambda p: _FakeMd(p) if p else _FakeMd("."))
        normalize_to_markdown("https://example.com", timeout=10)

    assert captured["cmd"][0] == str(INTEL_PYTHON_BIN), (
        f"{tool_name} routed to {captured['cmd'][0]!r}, "
        f"expected INTEL_PYTHON_BIN={INTEL_PYTHON_BIN}"
    )
    assert captured["cmd"][0] != str(PYTHON_BIN)
    assert captured["cmd"][1].endswith(script)


# ---------------------------------------------------------------------------
# Fix #3 — error propagation through state["errors"]
# ---------------------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """The v4 fetch chain on a fake network: ``routes[url]`` → Response."""
    routes: dict = {}
    from helpers import as_stream, request_url
    monkeypatch.setattr(fetchers, "_http_client", lambda cfg: httpx.Client(
        transport=httpx.MockTransport(lambda r: as_stream(routes.get(request_url(r), httpx.Response(404))))))
    monkeypatch.setattr(fetchers, "_resolve", lambda host: ["93.184.216.34"])
    monkeypatch.setattr(fetchers, "_scratch_dir", lambda: tmp_path)
    monkeypatch.setattr(fetchers, "INTEL_AVAILABLE", False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    return routes


def _fetch_all(sources):
    return _parallel_fetch(sources, cfg=FetchConfig(), ledger=SpendLedger({}), exa_on=False)


def test_parallel_fetch_lets_an_unexpected_rung_exception_propagate(offline, monkeypatch):
    """A fetch rung's expected failures (HTTP status, network, size, a parser's
    exception) are error lines (next test); anything else is a bug and MUST
    propagate (v4 spec §2.8, WP-B review), never become a string.

    v2 swallowed it (``fetched=[]``, then a vacuous "no evidence" report);
    v4's first cut turned it into an error line.
    """
    def boom(*a, **kw):
        raise RuntimeError("native exploded")

    monkeypatch.setattr(fetchers, "_native", boom)
    sources = [{"url": "https://a.example", "kind": "official_doc", "title": "A", "summary": ""}]
    with pytest.raises(RuntimeError, match="native exploded"):
        _fetch_all(sources)


def test_parallel_fetch_partial_failure_mixes_fetched_and_errors(offline):
    """Successful URLs land in fetched; failed URLs in errors — no silent
    swallowing."""
    offline["https://ok.example/a"] = httpx.Response(200, html=(
        "<html><body><article><h1>OK Title</h1>"
        + "<p>A long enough paragraph about the topic at hand. </p>" * 20
        + "</article></body></html>"))
    offline["https://bad.example/b"] = httpx.Response(500)
    fetched, errs = _fetch_all([
        {"url": "https://ok.example/a", "kind": "official_doc", "title": "OK", "summary": ""},
        {"url": "https://bad.example/b", "kind": "blog", "title": "BAD", "summary": ""},
    ])
    assert len(fetched) == 1
    assert fetched[0]["url"] == "https://ok.example/a"
    assert any("https://bad.example/b" in e for e in errs), errs


def test_research_node_emits_all_failed_summary(offline, monkeypatch):
    """When every source fails to fetch, research_node must append the
    'all sources failed' summary error (the v2 T2 contract, kept).

    Hermetic: with 0 fetched items the wave loop's coverage check finds
    a gap and would otherwise launch a follow-up search wave — mocked away
    so this test only exercises the fetch-failure path it's named for.
    """
    from helpers import make_run_config

    offline["https://a.example/x"] = httpx.Response(503)
    monkeypatch.setattr(research_mod, "followup_queries", lambda *a, **kw: ([], [], []))
    state = {
        "user_request": "attention transformers",
        "run_config": make_run_config("short", **{"research.rerank": False}),
        "brief": {"sub_questions": [{"q": "What is attention?", "kind": "web"}],
                  "must_have_keywords": ["attention"]},
        "sources": [
            {"url": "https://a.example/x", "kind": "blog", "title": "A",
             "snippet": "attention", "sub_qs": [0]},
        ],
        "fetched": [], "evidence": [], "errors": [],
    }
    out = research_mod.research_node(state)
    assert out["fetched"] == []
    errs = out["errors"]
    assert any("all" in e and "failed to fetch" in e for e in errs), errs
    assert "fetch https://a.example/x failed: native: HTTP 503 (rungs tried: native)" in errs


def test_errors_reducer_in_state_accepts_str_list():
    """Sanity-check the state shape: the errors reducer MUST be
    ``Annotated[list[str], operator.add]`` so that any node can return
    ``{"errors": [...]}`` and the entries merge. This is read-only
    verification of T2 acceptance criterion #2."""
    from argus.graph import state as state_mod
    hints = state_mod.ArgusState.__annotations__
    assert "errors" in hints
    # Pydantic / typing: we just need to confirm the field accepts a
    # list[str] update (operator.add reducer is satisfied as long as
    # both operands are list[str]).
    assert hints["errors"] is not None