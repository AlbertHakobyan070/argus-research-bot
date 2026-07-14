"""Argus LangGraph: build_graph() + helpers.

v3 research engine — Deep Research

intake
  │
  ▼
brief (sub-questions + success criteria; no URLs)
  │
  ▼
scout (LIVE multi-query discovery: Exa/DDGS/arXiv/GitHub)
  │
  ▼
[HITL: Plan Approval — real live sources shown]
  │
  ▼
research (waves: triage → parallel fetch → per-source LLM digest
          → coverage check → targeted follow-up queries)
  │
  ▼
outline
  │
  ▼
compose (parallel section writers from EvidenceNotes; markdown-native)
  │
  ▼
panel (tripartite judges: grounding / coverage / precision)
  ├── pass ──────────────────────────────► report_builder
  └── revise ─► compose (section-targeted revision loop)

report_builder
  │
  ▼
[HITL: Report Preview]
  │
  ▼
deliver
  ├── end
  ├── extend ─► extend_prep ─► research (never re-hits the plan gate)
  └── revise ─► revise_prep ─► compose (same evidence + user notes)


Quick Path

intake → quick_answer → deliver
"""
from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from .brief import brief_node
from .compose import compose_node, outline_node
from .nodes import (
    deliver_node, extend_prep_node, intake_node, quick_answer_node,
    report_builder_node, revise_prep_node, route_after_deliver,
)
from .panel import panel_node, route_after_panel
from .research import research_node
from .scout import scout_node
from .state import ArgusState

logger = logging.getLogger("argus.graph")


def build_graph(*, checkpointer=None) -> CompiledStateGraph:
    """Construct the deep-research LangGraph (v3 engine).

    Pass ``checkpointer=None`` for an in-memory saver (tests). For prod,
    pass the shared ``AsyncSqliteSaver`` (see ``async_sqlite_saver_cm``),
    opened once at bot startup and reused across all runs.
    """
    g = StateGraph(ArgusState)

    g.add_node("intake", intake_node)
    g.add_node("brief", brief_node)
    g.add_node("scout", scout_node)
    g.add_node("research", research_node)
    g.add_node("outline", outline_node)
    g.add_node("compose", compose_node)
    g.add_node("panel", panel_node)
    g.add_node("report_builder", report_builder_node)
    g.add_node("deliver", deliver_node)
    g.add_node("extend_prep", extend_prep_node)
    g.add_node("revise_prep", revise_prep_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "brief")
    g.add_edge("brief", "scout")
    # PLAN GATE: interrupt_after=["scout"] pauses here so the plan
    # preview shows REAL sources from the live discovery wave. The bot
    # resumes with Command(resume=True) after Approve.
    g.add_edge("scout", "research")
    g.add_edge("research", "outline")
    g.add_edge("outline", "compose")
    g.add_edge("compose", "panel")
    g.add_conditional_edges(
        "panel",
        route_after_panel,
        {"compose": "compose", "report_builder": "report_builder"},
    )
    g.add_edge("report_builder", "deliver")
    # After the report-preview gate: finish, extend (more sources →
    # rejoin at research; never re-triggers the plan gate), or revise
    # (same evidence + user notes → rejoin at compose).
    g.add_conditional_edges(
        "deliver",
        route_after_deliver,
        {"extend": "extend_prep", "revise": "revise_prep", "end": END},
    )
    g.add_edge("extend_prep", "research")
    g.add_edge("revise_prep", "compose")

    if checkpointer is None:
        checkpointer = MemorySaver()
    return g.compile(
        checkpointer=checkpointer,
        # Grounded plan gate: pause AFTER scout (real live sources in the
        # preview). snap.next == ("research",) identifies this gate.
        interrupt_after=["scout"],
        # report_builder runs, then we pause before deliver (preview).
        # snap.next == ("deliver",) identifies the report gate.
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


def async_sqlite_saver_cm(path: str):
    """Context manager for the production AsyncSqliteSaver checkpointer.

    Opened once at bot startup (PTB ``post_init``); the compiled graph is
    shared across all runs, isolated by per-run ``thread_id``.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    return AsyncSqliteSaver.from_conn_string(path)
