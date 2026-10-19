"""Tests for the HITL 'extend research' loop (v3).

Covers deliver pass-through, routing, and extend_prep plus the graph
wiring. Hermetic: the follow-up query generation and the search wave are
mocked so no network/LLM is touched.
"""
from __future__ import annotations

from argus.graph import research as research_mod
from argus.graph import search_providers as sp_mod
from argus.graph.nodes import (
    MAX_EXTEND_ROUNDS, deliver_node, extend_prep_node, route_after_deliver,
)
from argus.graph.state import ResearchBrief, SubQuestion


# --- routing -----------------------------------------------------------------

def test_route_after_deliver_extends_when_requested_under_cap():
    assert route_after_deliver(
        {"extend_requested": True, "extend_rounds": 0}) == "extend"
    assert route_after_deliver(
        {"extend_requested": True, "extend_rounds": MAX_EXTEND_ROUNDS - 1}) == "extend"


def test_route_after_deliver_ends_at_cap_or_when_not_requested():
    assert route_after_deliver(
        {"extend_requested": True, "extend_rounds": MAX_EXTEND_ROUNDS}) == "end"
    assert route_after_deliver({"extend_requested": False}) == "end"
    assert route_after_deliver({}) == "end"


# --- deliver pass-through -----------------------------------------------------

def test_deliver_node_passthrough_when_extending():
    out = deliver_node({"extend_requested": True, "extend_rounds": 0,
                        "report_paths": {"folder": "X"}})
    # Must NOT finalise (no hitl-cleared, no "Delivered").
    assert "hitl" not in out
    joined = " ".join(m.get("content", "") for m in out.get("messages", []))
    assert "Delivered" not in joined


def test_deliver_node_delivers_when_not_extending():
    out = deliver_node({"report_paths": {"folder": "X"}})
    assert out["hitl"] == {"pending": False}
    joined = " ".join(m.get("content", "") for m in out.get("messages", []))
    assert "Delivered" in joined


def test_deliver_node_delivers_at_cap_even_if_requested():
    out = deliver_node({"extend_requested": True,
                        "extend_rounds": MAX_EXTEND_ROUNDS,
                        "report_paths": {"folder": "X"}})
    assert out["hitl"] == {"pending": False}  # cap reached → finalise


# --- extend_prep --------------------------------------------------------------

def test_extend_prep_only_resets_counters_and_flags(monkeypatch):
    """v4 (orchestrator ruling, option B): extend_prep keeps only its
    counters and flags, like revise_prep; research_node runs the Extend
    wave (tests/test_gap_mode.py::test_plain_continue_extend_pass_runs_a_fresh_search_wave)."""
    def _boom(*a, **kw):
        raise AssertionError("extend_prep must not search: research runs the Extend wave")

    monkeypatch.setattr(research_mod, "followup_queries", _boom)
    monkeypatch.setattr(sp_mod, "run_query_wave", _boom)
    out = extend_prep_node({
        "brief": ResearchBrief(sub_questions=[SubQuestion(q="How does RAG ground answers?")],
                               must_have_keywords=["rag"], summary="s").model_dump(),
        "extend_requested": True,
        "extend_rounds": 0,
        "revision_rounds": 3,
        "sources": [{"url": "https://old.example", "kind": "web"}],
    })
    assert out["extend_rounds"] == 1
    assert out["extend_requested"] is False
    assert out["revision_rounds"] == 0
    assert out["panel_verdict"] is None
    assert not {"sources", "queries", "model_calls", "errors"} & set(out), "no wave, nothing merged here"


def test_extend_prep_append_only_leaves_the_flag_for_research():
    srcs = [{"url": "file:///V/t.txt", "local_path": "V/t.txt", "source": "appended-asset"}]
    out = extend_prep_node({"append_only": True, "extend_rounds": 1, "sources": srcs})
    assert out["extend_rounds"] == 2
    assert "append_only" not in out, "research_node consumes and clears it"
    assert out["sources"] == srcs


# --- graph wiring -------------------------------------------------------------

def test_graph_builds_with_extend_node():
    from argus.graph.graph import build_graph
    g = build_graph()  # compiles the conditional deliver->extend_prep/END edge
    assert g is not None
