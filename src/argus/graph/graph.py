"""Argus LangGraph: build_graph() + helpers.

Architecture (deep path):

  intake ──► planner ──► planner_reflect ──► researcher (LIVE search)
  ──HITL plan_approval (real found sources shown)──► fetcher ──►
  normalizer ──► credibility ──► filter ──► synthesizer ──► reviewer
  ──► (pass|revise) ▲                                │
                     └──────────── revision loop ────┘
                                                          ▼
                                                    report_builder
                                                          │
                                                  HITL report_preview
                                                          ▼
                                                       deliver

Quick path (for /ask): intake → quick_answer → deliver.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from .nodes import (
    credibility_node, deliver_node, extend_prep_node, fetcher_node,
    filter_node, intake_node, normalizer_node, planner_node,
    planner_reflect_node, quick_answer_node, report_builder_node,
    researcher_node, reviewer_node, revise_prep_node, route_after_deliver,
    route_after_review, synthesizer_node,
)
from .state import ArgusState

logger = logging.getLogger("argus.graph")


def build_graph(*, checkpointer=None) -> CompiledStateGraph:
    """Construct the deep-research LangGraph.

    Pass ``checkpointer=None`` for an in-memory saver (tests). For prod,
    pass a ``SqliteSaver`` rooted at the argus checkpoint DB.
    """
    g = StateGraph(ArgusState)

    g.add_node("intake", intake_node)
    g.add_node("planner", planner_node)
    g.add_node("planner_reflect", planner_reflect_node)
    g.add_node("researcher", researcher_node)
    g.add_node("fetcher", fetcher_node)
    g.add_node("normalizer", normalizer_node)
    g.add_node("credibility", credibility_node)
    g.add_node("filter", filter_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("report_builder", report_builder_node)
    g.add_node("deliver", deliver_node)
    g.add_node("extend_prep", extend_prep_node)
    g.add_node("revise_prep", revise_prep_node)

    g.add_edge(START, "intake")

    # intake → planner  (deep path always goes through planner)
    g.add_edge("intake", "planner")

    # HITL plan_approval: planner pauses for human approval of the plan.
    # We use LangGraph's interrupt mechanism via dynamic_break on the
    # `hitl.pending` flag — the actual wait + resume is driven by the
    # Telegram bot via Command(resume=...).
    g.add_edge("planner", "planner_reflect")
    g.add_edge("planner_reflect", "researcher")
    g.add_edge("researcher", "fetcher")
    g.add_edge("fetcher", "normalizer")
    g.add_edge("normalizer", "credibility")
    g.add_edge("credibility", "filter")
    g.add_edge("filter", "synthesizer")
    g.add_edge("synthesizer", "reviewer")

    g.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"synthesizer": "synthesizer", "report_builder": "report_builder"},
    )

    g.add_edge("report_builder", "deliver")
    # Phase 2 HITL "extend": after the report-preview gate, either finish or
    # loop back to gather more (extend_prep runs the researcher itself, then
    # rejoins at fetcher — it must NOT route through the interrupt-gated
    # `researcher` node or it would re-trigger the plan-approval pause).
    g.add_conditional_edges(
        "deliver",
        route_after_deliver,
        {"extend": "extend_prep", "revise": "revise_prep", "end": END},
    )
    g.add_edge("extend_prep", "fetcher")
    # revise re-synthesizes from the SAME evidence with the user's notes;
    # rejoin at synthesizer (not fetcher — no new sources to gather).
    g.add_edge("revise_prep", "synthesizer")

    if checkpointer is None:
        checkpointer = MemorySaver()
    return g.compile(
        checkpointer=checkpointer,
        # Grounded plan gate (v2): pause AFTER researcher, so the plan
        # preview can show REAL sources found by live search instead of
        # the planner LLM's invented URLs. The extend loop rejoins at
        # fetcher and thus never re-triggers this gate.
        interrupt_after=["researcher"],
        # report_builder runs, then we pause before deliver (preview).
        interrupt_before=["deliver"],
    )


def quick_answer_graph(*, checkpointer=None) -> CompiledStateGraph:
    """Construct the lightweight quick-answer graph (for /ask)."""
    g = StateGraph(ArgusState)
    g.add_node("intake", intake_node)
    g.add_node("quick_answer", quick_answer_node)
    g.add_node("deliver", deliver_node)
    g.add_edge(START, "intake")
    g.add_edge("intake", "quick_answer")
    g.add_edge("quick_answer", "deliver")
    g.add_edge("deliver", END)
    if checkpointer is None:
        checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


def sqlite_saver_cm(path: str):
    """Return the context manager for the sync SqliteSaver.

    Usage::

        with sqlite_saver_cm(path) as saver:
            graph = build_graph(checkpointer=saver)
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    return SqliteSaver.from_conn_string(path)


def async_sqlite_saver_cm(path: str):
    """Async counterpart of :func:`sqlite_saver_cm`."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    return AsyncSqliteSaver.from_conn_string(path)


# ---------------------------------------------------------------------------
# Driver helpers — used by the Telegram bot and by tests.
# ---------------------------------------------------------------------------

def run_research_sync(graph, *, thread_id: str, user_id: int,
                      user_request: str, recursion_limit: int = 40,
                      config_overrides: dict | None = None) -> dict:
    """Drive the deep graph to its first HITL pause and return state.

    The bot layer should call this, inspect `state["hitl"]`, then resume
    via `graph.invoke(Command(resume=...), config)` once the user clicks
    Approve/Edit/Cancel.
    """
    cfg = {"configurable": {"thread_id": thread_id}}
    if config_overrides:
        cfg.update(config_overrides)
    state_in: ArgusState = {
        "thread_id": thread_id,
        "user_id": user_id,
        "user_request": user_request,
        "messages": [],
        "plan": None,
        "sources": [],
        "fetched": [],
        "findings": [],
        "draft_md": "",
        "revision_notes": [],
        "revision_rounds": 0,
        "model_calls": [],
        "hitl": {"pending": False},
    }
    # Stream until first interrupt.
    last_state: dict = {}
    for ev in graph.stream(state_in, config=cfg,
                           recursion_limit=recursion_limit):
        last_state = ev
    return last_state