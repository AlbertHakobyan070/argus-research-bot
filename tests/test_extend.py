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

def test_extend_prep_widens_queries_bumps_round_and_merges(monkeypatch):
    captured = {}

    def fake_followups(brief, gaps, prior):
        captured["gaps"] = list(gaps)
        return ([{"query": "retrieval augmentation grounding",
                  "provider": "ddgs", "sub_qs": [0]}], [])

    def fake_wave(queries, **kw):
        captured["queries"] = queries
        return ([{"url": "https://new.example", "kind": "web",
                  "title": "n", "snippet": "", "provider": "ddgs",
                  "sub_qs": [0], "text": "", "published": "",
                  "source": "ddgs", "summary": ""}], [])

    monkeypatch.setattr(research_mod, "followup_queries", fake_followups)
    monkeypatch.setattr(sp_mod, "run_query_wave", fake_wave)

    brief = ResearchBrief(
        sub_questions=[SubQuestion(
            q="How does retrieval augmentation improve grounding?")],
        must_have_keywords=["rag"], summary="s")
    out = extend_prep_node({
        "brief": brief.model_dump(),
        "extend_requested": True,
        "extend_rounds": 0,
        "revision_rounds": 3,
        "sources": [{"url": "https://old.example", "kind": "web"}],
    })

    # Round bumped, flag cleared, review budget + stale panel reset.
    assert out["extend_rounds"] == 1
    assert out["extend_requested"] is False
    assert out["revision_rounds"] == 0
    assert out["panel_verdict"] is None
    # Every sub-question is treated as a gap for the widening wave.
    assert captured["gaps"] == [0]
    # New hits merge INTO the existing pool (old sources kept).
    urls = [s["url"] for s in out["sources"]]
    assert "https://old.example" in urls
    assert "https://new.example" in urls
    # The widened queries are recorded on state for later waves.
    assert any(q["query"] == "retrieval augmentation grounding"
               for q in out["queries"])


def test_extend_prep_propagates_wave_errors(monkeypatch):
    monkeypatch.setattr(research_mod, "followup_queries",
                        lambda brief, gaps, prior: ([{"query": "x",
                                                      "provider": "ddgs",
                                                      "sub_qs": [0]}], []))
    monkeypatch.setattr(sp_mod, "run_query_wave",
                        lambda queries, **kw: ([], ["ddgs: boom"]))
    out = extend_prep_node({
        "brief": ResearchBrief(
            sub_questions=[SubQuestion(q="x?")],
            must_have_keywords=["x"]).model_dump(),
        "extend_rounds": 1,
    })
    assert out["extend_rounds"] == 2
    assert any("boom" in e for e in out.get("errors", []))


# --- graph wiring -------------------------------------------------------------

def test_graph_builds_with_extend_node():
    from argus.graph.graph import build_graph
    g = build_graph()  # compiles the conditional deliver->extend_prep/END edge
    assert g is not None
