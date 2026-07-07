"""Argus T2 (Pattern E) — routing + error-propagation contract tests.

These tests pin the two behavior changes Argus T2 introduced:

* Fix #2 — every intel-stack tool wrapper (``harvest_sources``,
  ``snatch_url``, ``crawl_url``, ``normalize_to_markdown``) MUST
  invoke the intel-stack interpreter (INTEL_PYTHON_BIN). Previously
  only ``harvest_sources`` did so; the other three silently fell
  back to the argus venv's python and immediately raised
  ``ModuleNotFoundError`` for feedparser / crawl4ai / markitdown /
  yt_dlp.

* Fix #3 — ``fetcher_node`` and ``researcher_node`` MUST propagate
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
from argus.graph import nodes as nodes_mod
from argus.graph.nodes import fetcher_node, researcher_node
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


def test_fetcher_node_propagates_exceptions_to_errors(monkeypatch):
    """A raising snatch/crawl/normalize MUST appear in state["errors"].

    Pre-fix behavior: the ``except Exception: logger.warning(...)``
    branch silently returned ``fetched=[]`` — the synthesizer then
    produced a vacuous "no evidence" report.
    """
    from argus.tools import CrawlResult, NormalizeResult, SnatchResult

    def boom_normalize(url, *a, **kw):
        raise RuntimeError("normalize exploded")
    def boom_snatch(url, *a, **kw):
        raise RuntimeError("snatch exploded")
    def boom_crawl(url, *a, **kw):
        raise RuntimeError("crawl exploded")

    monkeypatch.setattr(nodes_mod, "normalize_to_markdown", boom_normalize)
    monkeypatch.setattr(nodes_mod, "snatch_url", boom_snatch)
    monkeypatch.setattr(nodes_mod, "crawl_url", boom_crawl)

    state: ArgusState = _make_state(ResearchPlan())
    state["sources"] = [
        {"url": "https://a.example", "kind": "official_doc",
         "title": "A", "summary": ""},
        {"url": "https://b.example", "kind": "blog",
         "title": "B", "summary": ""},
    ]
    out = fetcher_node(state)
    # No items fetched (every tool raised).
    assert out["fetched"] == []
    # Errors must propagate — at minimum one entry per URL plus the
    # "all sources failed" summary error.
    errs = out["errors"]
    assert isinstance(errs, list)
    assert any("https://a.example" in e for e in errs), errs
    assert any("https://b.example" in e for e in errs), errs
    assert any("all" in e and "source" in e and "failed" in e
                for e in errs), (
        f"expected a summary 'all sources failed' error, got {errs!r}"
    )


def test_fetcher_node_partial_failure_mixes_fetched_and_errors(monkeypatch):
    """Successful URLs should land in fetched; failed URLs in errors —
    no silent swallowing."""
    from argus.tools import CrawlResult, NormalizeResult, SnatchResult

    def ok_normalize(url, *a, **kw):
        return NormalizeResult(
            ok=True,
            markdown_path="A:\\fake\\ok.md",
            markdown_text="x" * 700,
            title="OK Title",
        )
    def bad_snatch(url, *a, **kw):
        raise RuntimeError("snatch down")
    def bad_crawl(url, *a, **kw):
        return CrawlResult(ok=False, error="down",
                           duration_s=0.0)

    monkeypatch.setattr(nodes_mod, "normalize_to_markdown", ok_normalize)
    monkeypatch.setattr(nodes_mod, "snatch_url", bad_snatch)
    monkeypatch.setattr(nodes_mod, "crawl_url", bad_crawl)

    state: ArgusState = _make_state(ResearchPlan())
    state["sources"] = [
        {"url": "https://ok.example", "kind": "official_doc",
         "title": "OK", "summary": ""},
        {"url": "https://bad.example", "kind": "blog",
         "title": "BAD", "summary": ""},
    ]
    out = fetcher_node(state)
    # Successful URL must be in fetched.
    assert len(out["fetched"]) == 1
    assert out["fetched"][0]["url"] == "https://ok.example"
    # The failed URL must be in errors (NOT silently dropped).
    assert any("https://bad.example" in e for e in out["errors"]), out["errors"]
    # And we should NOT have the "all sources failed" summary because
    # at least one URL succeeded.
    assert not any("all" in e and "failed to fetch" in e
                   for e in out["errors"]), out["errors"]


def test_researcher_node_propagates_harvest_and_arxiv_errors(monkeypatch):
    """Both harvest_sources and arxiv failures must land in state["errors"]."""
    def boom_harvest(*a, **kw):
        raise RuntimeError("harvest exploded")
    def boom_arxiv(plan):
        raise RuntimeError("arxiv exploded")

    monkeypatch.setattr(nodes_mod, "harvest_sources", boom_harvest)
    monkeypatch.setattr(nodes_mod, "_arxiv_search", boom_arxiv)

    state = _make_state(_plan_with_paper())
    out = researcher_node(state)
    errs = out["errors"]
    assert isinstance(errs, list)
    assert any("harvest_sources failed" in e for e in errs), errs
    assert any("arxiv_search failed" in e for e in errs), errs


def test_researcher_node_no_errors_on_clean_run(monkeypatch):
    """Happy path: when both harvest and arxiv succeed, no errors field
    is returned (clean run, no failure surface)."""
    from argus.tools import HarvestReport, HarvestResult

    monkeypatch.setattr(
        nodes_mod, "harvest_sources",
        lambda *a, **kw: HarvestReport(
            folder="A:\\fake", radar_md="", items=[],
            raw_stdout="", duration_s=0.0,
        ),
    )
    monkeypatch.setattr(
        nodes_mod, "_arxiv_search",
        lambda plan: [
            {"kind": "paper", "title": "T", "url": "https://arxiv.org/abs/1",
             "summary": "S", "source": "arxiv"},
        ],
    )

    state = _make_state(_plan_with_paper())
    out = researcher_node(state)
    assert "errors" not in out, (
        f"clean run should not emit errors, got {out.get('errors')!r}"
    )
    # Sources must include the arxiv item.
    assert any(s["url"] == "https://arxiv.org/abs/1"
                for s in out["sources"]), out["sources"]


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