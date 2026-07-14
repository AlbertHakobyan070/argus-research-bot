"""Grounded plan gate — the plan-approval HITL pauses AFTER real search.

Albert's 2026-07-10 report: the plan preview showed fabricated URLs
(``thelema.org/officialdocs``, ``github.com/otto-xyz/crowley``…) because
the planner LLM invents links from its training data and the preview
rendered them before any real search had run.

v3 contract:

1. The deep graph pauses AFTER ``scout`` (``interrupt_after``), so by
   the time the user reviews the plan, ``state["sources"]`` holds REAL
   results from the live discovery wave (exa/ddgs/arxiv/github) — and
   the preview can show them. ``snap.next == ("research",)`` identifies
   the gate.
2. The extend loop (``extend_prep`` → ``research``) must NOT re-trigger
   the plan gate: the gate is on the scout node, which extend
   deliberately bypasses.

Hermetic: every node is stubbed at the graph-module level; no LLM, no
network.
"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from argus.graph import graph as graph_mod


_PLAN = {"summary": "s", "sub_questions": ["q"],
         "planned_sources": [{"kind": "search_result", "query": "q",
                              "target_url": None, "rationale": ""}],
         "must_have_keywords": []}

_BRIEF = {"sub_questions": [{"q": "q?", "kind": "web"}],
          "must_have_keywords": ["q"], "summary": "s",
          "success_criteria": []}

_REAL_SOURCE = {"kind": "repo", "title": "Real repo",
                "url": "https://github.com/real/repo", "summary": "",
                "source": "github-search"}


def _stub_nodes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every node build_graph binds, at the graph-module namespace."""
    monkeypatch.setattr(graph_mod, "intake_node", lambda s: {"mode": "deep"})
    monkeypatch.setattr(graph_mod, "brief_node",
                        lambda s: {"brief": dict(_BRIEF),
                                   "plan": dict(_PLAN)})
    monkeypatch.setattr(graph_mod, "scout_node",
                        lambda s: {"sources": [dict(_REAL_SOURCE)]})
    monkeypatch.setattr(graph_mod, "research_node",
                        lambda s: {"fetched": [{"url": _REAL_SOURCE["url"],
                                                "title": "Real repo",
                                                "excerpt": "x"}],
                                   "evidence": [], "coverage": {}})
    monkeypatch.setattr(graph_mod, "outline_node",
                        lambda s: {"outline": {"sections": []}})
    monkeypatch.setattr(graph_mod, "compose_node",
                        lambda s: {"draft_md": "d", "findings": [],
                                   "sections": []})
    monkeypatch.setattr(graph_mod, "panel_node",
                        lambda s: {"panel_verdict": {"verdict": "pass"},
                                   "review_verdict": {"verdict": "pass"}})
    monkeypatch.setattr(graph_mod, "route_after_panel",
                        lambda s: "report_builder")
    monkeypatch.setattr(graph_mod, "report_builder_node",
                        lambda s: {"report_paths": {"md": "r.md",
                                                    "folder": "f"}})
    # deliver_node / route_after_deliver: keep the REAL routing semantics
    # but stub extend_prep's search work.
    monkeypatch.setattr(
        graph_mod, "extend_prep_node",
        lambda s: {"sources": (s.get("sources") or []) + [
            {"kind": "web", "title": "More", "url": "https://more.example/x",
             "summary": "", "source": "web-search"}],
            "extend_requested": False,
            "extend_rounds": int(s.get("extend_rounds") or 0) + 1})


def _state(thread: str) -> dict:
    return {"thread_id": thread, "user_id": 1, "user_request": "topic",
            "messages": [], "plan": None, "sources": [], "fetched": [],
            "findings": [], "draft_md": "", "revision_notes": [],
            "revision_rounds": 0, "model_calls": [],
            "hitl": {"pending": False}}


def test_plan_gate_pauses_after_scout_with_real_sources(monkeypatch):
    _stub_nodes(monkeypatch)
    g = graph_mod.build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t:gate"}}
    g.invoke(_state("t:gate"), config=cfg)

    snap = g.get_state(cfg)
    assert snap.next == ("research",), (
        f"plan gate must pause AFTER scout (next=research), got "
        f"{snap.next!r} — pausing before scout means the preview can "
        f"only show LLM-invented URLs")
    assert snap.values.get("plan"), "plan must be drafted at the gate"
    srcs = snap.values.get("sources") or []
    assert srcs and srcs[0]["url"] == _REAL_SOURCE["url"], (
        "REAL search results must be in state at the plan gate so the "
        "preview shows live-found sources, not brief guesses")


def test_extend_loop_does_not_rehit_plan_gate(monkeypatch):
    _stub_nodes(monkeypatch)
    g = graph_mod.build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t:extend"}}
    g.invoke(_state("t:extend"), config=cfg)          # → plan gate
    g.invoke(Command(resume=True), config=cfg)        # → report preview gate

    snap = g.get_state(cfg)
    assert snap.next == ("deliver",), (
        f"expected the report-preview pause next, got {snap.next!r}")

    # User taps Extend: set the flag, resume.
    g.update_state(cfg, {"extend_requested": True})
    g.invoke(Command(resume=True), config=cfg)

    snap = g.get_state(cfg)
    assert snap.next == ("deliver",), (
        f"extend must loop extend_prep→research→…→report_builder and pause "
        f"at the NEXT report preview — got {snap.next!r}. If this is "
        f"('research',) the plan gate re-fired inside the extend loop.")
    assert int(snap.values.get("extend_rounds") or 0) == 1
    # And the extend pass actually widened the source pool.
    assert len(snap.values.get("sources") or []) == 2
