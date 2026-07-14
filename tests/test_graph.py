"""Argus graph tests — drive the full v3 LangGraph in-memory.

LIVE tests (excluded from the hermetic CI subset): intake/brief/digest/
compose/panel hit the REAL FreeLLMAPI proxy. The search wave and the
page fetches are stubbed so the run is network-deterministic — the LLM
reads controlled source content and must still produce a grounded,
cited report through the v3 engine.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from argus.config import get_settings
from argus.graph import build_graph, quick_answer_graph


def _make_state_in(thread_id: str, user_id: int, text: str) -> dict:
    return {
        "thread_id": thread_id,
        "user_id": user_id,
        "user_request": text,
        "messages": [], "plan": None,
        "sources": [], "fetched": [], "findings": [],
        "draft_md": "", "revision_notes": [], "revision_rounds": 0,
        "model_calls": [], "hitl": {"pending": False},
    }


def test_quick_graph_runs():
    g = quick_answer_graph()
    cfg = {"configurable": {"thread_id": "test:quick"}}
    out = g.invoke(_make_state_in("test:quick", 1, "what is 2+2?"),
                   config=cfg)
    assert "quick_answer" in out
    assert isinstance(out["quick_answer"], str)
    assert out["quick_answer"]


_DOC = """# LLM agent benchmark survey (stub source)

AgentBench evaluates LLM agents across 8 environments including web
browsing, database querying, and operating-system tasks. GPT-4 scored
4.01 overall while the best open model reached 2.89.

SWE-bench measures repository-level code fixing; agents resolve under
20 percent of issues without scaffolding. WebArena provides realistic
self-hosted websites and reports a 14.4 percent success rate for the
best 2024 agent versus 78 percent for humans.

Benchmark contamination is a known weakness: static test sets leak into
training data, inflating scores over time.
"""


def test_deep_graph_pauses_for_plan_approval(monkeypatch, tmp_path):
    """v3 live drive: first interrupt fires AFTER scout (grounded plan
    gate) with plan + real sources in state; after Approve, research
    digests the (stubbed-fetch) sources with the live LLM, compose
    writes a sectioned report, panel reviews it, and a report lands on
    disk."""
    monkeypatch.setenv("ARGUS_REPORTS_ROOT", str(tmp_path))
    import importlib
    cfg_mod = importlib.import_module("argus.config")
    cfg_mod._cached = None
    s = get_settings()
    assert s.reports_root == tmp_path

    from argus.graph import research as research_mod
    from argus.graph import scout as scout_mod

    # Controlled source: the wave "finds" one page; the fetch writes a
    # real markdown document the live digest LLM must read.
    doc_path = tmp_path / "stub_source.md"
    doc_path.write_text(_DOC, encoding="utf-8")

    def fake_wave(queries, **kw):
        hits = [{
            "url": "https://example.org/llm-agent-benchmarks",
            "title": "LLM agent benchmark survey",
            "snippet": "AgentBench, SWE-bench, WebArena results",
            "kind": "blog", "provider": "ddgs",
            "sub_qs": sorted({i for q in queries
                              for i in (q.get("sub_qs") or [])}),
            "text": "", "published": "", "source": "ddgs",
            "summary": "benchmark survey",
        }]
        return hits, []

    monkeypatch.setattr(scout_mod, "run_query_wave", fake_wave)
    monkeypatch.setattr(research_mod, "run_query_wave", fake_wave)

    def fake_fetch(src):
        from argus.graph.state import FetchedItem
        return FetchedItem(
            url=src.get("url", ""), title=src.get("title", ""),
            markdown_path=str(doc_path), section=src.get("kind", ""),
            excerpt=_DOC[:600], sub_qs=src.get("sub_qs") or [],
            provider="stub",
        ).model_dump(), []

    monkeypatch.setattr(research_mod, "fetch_one_source", fake_fetch)

    g = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "test:deep"}}
    g.invoke(_make_state_in("test:deep", 1, "LLM agent benchmarks"),
             config=cfg)
    # Grounded plan gate: paused AFTER scout, before research.
    snap = g.get_state(cfg)
    assert snap.next == ("research",), (
        f"expected the grounded plan gate (next=research), got {snap.next!r}")
    cur = snap.values
    assert cur.get("brief"), "brief should be drafted"
    assert cur.get("plan"), "v2-compatible plan must be present for the bot"
    assert cur.get("sources"), (
        "live-search results must be available at the plan gate")

    # Approve -> resume through research/outline/compose/panel.
    g.invoke(Command(resume=True), config=cfg)
    snap = g.get_state(cfg)
    for _ in range(6):
        if not snap.next:
            break
        g.invoke(Command(resume=True), config=cfg)
        snap = g.get_state(cfg)
    cur = snap.values

    # Evidence actually flowed: digest notes exist and the report is real.
    assert cur.get("evidence"), "digest must produce evidence notes"
    assert cur.get("draft_md"), "compose must produce a draft"
    assert cur.get("findings") is not None
    paths = cur.get("report_paths") or {}
    assert paths.get("md") and Path(paths["md"]).exists(), paths
    md = Path(paths["md"]).read_text(encoding="utf-8")
    assert "example.org/llm-agent-benchmarks" in md, (
        "the report must cite the fetched source")
