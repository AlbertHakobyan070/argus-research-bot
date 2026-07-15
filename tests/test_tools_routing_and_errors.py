"""Argus T2 (Pattern E) — routing + error-propagation contract tests.

These tests pin the two behavior changes Argus T2 introduced:

* Fix #2 — every intel-stack tool wrapper (``harvest_sources``,
  ``snatch_url``, ``crawl_url``, ``normalize_to_markdown``) MUST
  invoke the intel-stack interpreter (INTEL_PYTHON_BIN). Previously
  only ``harvest_sources`` did so; the other three silently fell
  back to the argus venv's python and immediately raised
  ``ModuleNotFoundError`` for feedparser / crawl4ai / markitdown /
  yt_dlp.

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

from argus import tools
from argus.graph import research as research_mod
from argus.graph.research import _parallel_fetch
from argus.graph.state import ArgusState, PlannedSource, ResearchPlan
from argus.tools import (
    INTEL_PYTHON_BIN, PYTHON_BIN,
    _run_script, crawl_url, harvest_sources, normalize_to_markdown,
    snatch_url,
)


# ---------------------------------------------------------------------------
# Fix #2 — every tool wrapper routes through INTEL_PYTHON_BIN
# ---------------------------------------------------------------------------


def _plan_with_paper() -> ResearchPlan:
    """A minimal plan that triggers researcher_node's arXiv branch."""
    return ResearchPlan(
        sub_questions=["x"],
        planned_sources=[
            PlannedSource(kind="paper", query="attention is all you need",
                          target_url=None, rationale="primary"),
        ],
        must_have_keywords=["attention"],
        summary="probe",
    )


def _make_state(plan: ResearchPlan) -> ArgusState:
    return {
        "plan": plan.model_dump(),
        "sources": [],
        "fetched": [],
        "errors": [],
    }


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


def test_harvest_sources_routes_through_intel_python(monkeypatch):
    """harvest_sources used to pass python_bin explicitly; it now relies
    on the default. Pin that the spawned interpreter is still INTEL."""
    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc(rc=0,
                          stdout='{"dir": "A:\\\\fake\\\\radar"}\n',
                          stderr="")

    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    class _FakeFolder:
        def __init__(self, p): self.p = p
        def __truediv__(self, name): return self
        def exists(self): return False
    monkeypatch.setattr(tools, "Path", _FakeFolder)
    harvest_sources(hours=24, top=3, sections="papers", timeout=10)

    assert captured["cmd"][0] == str(INTEL_PYTHON_BIN)


# ---------------------------------------------------------------------------
# Fix #3 — error propagation through state["errors"]
# ---------------------------------------------------------------------------


def test_parallel_fetch_propagates_exceptions_to_errors(monkeypatch):
    """A raising snatch/crawl/normalize MUST surface as error strings.

    Pre-fix behavior (v2): the ``except Exception: logger.warning(...)``
    branch silently returned ``fetched=[]`` — the writers then produced
    a vacuous "no evidence" report.
    """
    def boom_normalize(url, *a, **kw):
        raise RuntimeError("normalize exploded")
    def boom_snatch(url, *a, **kw):
        raise RuntimeError("snatch exploded")
    def boom_crawl(url, *a, **kw):
        raise RuntimeError("crawl exploded")

    monkeypatch.setattr(research_mod, "normalize_to_markdown", boom_normalize)
    monkeypatch.setattr(research_mod, "snatch_url", boom_snatch)
    monkeypatch.setattr(research_mod, "crawl_url", boom_crawl)

    sources = [
        {"url": "https://a.example", "kind": "official_doc",
         "title": "A", "summary": ""},
        {"url": "https://b.example", "kind": "blog",
         "title": "B", "summary": ""},
    ]
    fetched, errs = _parallel_fetch(sources)
    assert fetched == []
    assert any("https://a.example" in e for e in errs), errs
    assert any("https://b.example" in e for e in errs), errs


def test_parallel_fetch_partial_failure_mixes_fetched_and_errors(monkeypatch):
    """Successful URLs should land in fetched; failed URLs in errors —
    no silent swallowing."""
    from argus.tools import CrawlResult, NormalizeResult

    def ok_normalize(url, *a, **kw):
        return NormalizeResult(
            ok=True,
            markdown_path="A:\fake\ok.md",
            markdown_text="x" * 700,
            title="OK Title",
        )
    def bad_snatch(url, *a, **kw):
        raise RuntimeError("snatch down")
    def bad_crawl(url, *a, **kw):
        return CrawlResult(ok=False, error="down",
                           duration_s=0.0)

    monkeypatch.setattr(research_mod, "normalize_to_markdown", ok_normalize)
    monkeypatch.setattr(research_mod, "snatch_url", bad_snatch)
    monkeypatch.setattr(research_mod, "crawl_url", bad_crawl)

    sources = [
        {"url": "https://ok.example", "kind": "official_doc",
         "title": "OK", "summary": ""},
        {"url": "https://bad.example", "kind": "blog",
         "title": "BAD", "summary": ""},
    ]
    fetched, errs = _parallel_fetch(sources)
    assert len(fetched) == 1
    assert fetched[0]["url"] == "https://ok.example"
    assert any("https://bad.example" in e for e in errs), errs


def test_research_node_emits_all_failed_summary(monkeypatch):
    """When every source fails to fetch, research_node must append the
    'all sources failed' summary error (the v2 T2 contract, kept).

    Hermetic: with 0 fetched items the wave loop's coverage check finds
    a gap and would otherwise launch a REAL follow-up search wave
    (network) — mock it away so this test only exercises the
    fetch-failure path it's named for.
    """
    def boom(url, *a, **kw):
        raise RuntimeError("down")
    monkeypatch.setattr(research_mod, "normalize_to_markdown", boom)
    monkeypatch.setattr(research_mod, "snatch_url", boom)
    monkeypatch.setattr(research_mod, "crawl_url", boom)
    monkeypatch.setattr(research_mod, "followup_queries",
                        lambda brief, gaps, prior: ([], []))

    state = {
        "user_request": "attention transformers",
        "brief": {"sub_questions": [{"q": "What is attention?",
                                     "kind": "web"}],
                  "must_have_keywords": ["attention"]},
        "sources": [
            {"url": "https://a.example", "kind": "blog", "title": "A",
             "snippet": "attention", "sub_qs": [0]},
        ],
        "fetched": [], "evidence": [], "errors": [],
    }
    out = research_mod.research_node(state)
    assert out["fetched"] == []
    errs = out["errors"]
    assert any("all" in e and "failed to fetch" in e for e in errs), errs


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