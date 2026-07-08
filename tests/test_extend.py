"""Tests for the Phase 2 HITL 'extend research' loop.

Covers the three new pieces of logic (deliver pass-through, routing, and
extend_prep) plus the graph wiring. Hermetic: run_researcher_subgraph is
mocked so no network/LLM is touched.
"""
from __future__ import annotations

from argus.graph import nodes as nodes_mod
from argus.graph.nodes import (
    MAX_EXTEND_ROUNDS, deliver_node, extend_prep_node, route_after_deliver,
)
from argus.graph.state import ResearchPlan


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

def test_extend_prep_broadens_keywords_bumps_round_and_merges(monkeypatch):
    captured = {}

    def fake_research(state, **kw):
        captured["keywords"] = (state.get("plan") or {}).get("must_have_keywords")
        return {"sources": [{"url": "https://new.example", "kind": "web"}],
                "messages": [], "errors": []}

    monkeypatch.setattr(nodes_mod, "run_researcher_subgraph", fake_research)

    plan = ResearchPlan(
        sub_questions=["How does retrieval augmentation improve grounding?"],
        must_have_keywords=["rag"],
        planned_sources=[], summary="s",
    )
    out = extend_prep_node({
        "plan": plan.model_dump(),
        "extend_requested": True,
        "extend_rounds": 0,
        "revision_rounds": 3,
        "sources": [{"url": "https://old.example", "kind": "web"}],
    })

    # Round bumped, flag cleared, review budget reset.
    assert out["extend_rounds"] == 1
    assert out["extend_requested"] is False
    assert out["revision_rounds"] == 0
    # Sources come from the (mocked) widened research pass.
    assert any(s["url"] == "https://new.example" for s in out["sources"])
    # Keywords were broadened with salient sub-question words (len > 4).
    kw = out["plan"]["must_have_keywords"]
    assert "rag" in kw
    assert "retrieval" in kw and "augmentation" in kw and "grounding" in kw
    # The widened keywords were actually passed to the research pass.
    assert "retrieval" in (captured["keywords"] or [])


def test_extend_prep_propagates_research_errors(monkeypatch):
    monkeypatch.setattr(
        nodes_mod, "run_researcher_subgraph",
        lambda state, **kw: {"sources": [], "messages": [],
                             "errors": ["web_sub failed: boom"]})
    out = extend_prep_node({
        "plan": ResearchPlan(must_have_keywords=["x"]).model_dump(),
        "extend_rounds": 1,
    })
    assert out["extend_rounds"] == 2
    assert any("web_sub failed" in e for e in out.get("errors", []))


# --- graph wiring -------------------------------------------------------------

def test_graph_builds_with_extend_node():
    from argus.graph.graph import build_graph
    g = build_graph()  # compiles the conditional deliver->extend_prep/END edge
    assert g is not None
