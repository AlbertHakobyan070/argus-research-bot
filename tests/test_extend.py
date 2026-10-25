"""Tests for the HITL 'extend research' loop (v3).

Covers deliver pass-through, routing, and extend_prep plus the graph
wiring. Hermetic: the follow-up query generation and the search wave are
mocked so no network/LLM is touched.
"""
from __future__ import annotations

from argus.graph import research as research_mod
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
    monkeypatch.setattr(research_mod, "run_wave", _boom)
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


def test_a_plain_continue_runs_the_extend_pass_through_the_real_graph(monkeypatch, tmp_path):
    """The real extend_prep and research_node on a MemorySaver graph (WP-B review):
    first pass → report gate → the plain /continue fork (bot/history.py) → the
    Extend pass. Only the search / fetch / LLM seams and the non-WP-B nodes are stubbed."""
    import json

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from argus import llm
    from argus.graph import graph as graph_mod
    from argus.graph.state import FetchedItem
    from helpers import make_run_config

    doc = "Y works by pausing the graph and resuming it with a value. " * 20
    brief = {"sub_questions": [{"q": "What is X?", "kind": "web"}, {"q": "How does Y work?", "kind": "web"}],
             "must_have_keywords": ["x", "y"], "summary": "s", "success_criteria": []}
    roles, waves, fetches = [], [], []

    def complete(role, messages, *, engines, validate=None, item="", **kw):
        roles.append(role)
        out = json.dumps({"research.digest": {"relevance": 4, "stance": "supports", "source_type": "blog",
                                              "claims": [{"text": "Y pauses", "quote": "pausing the graph",
                                                          "confidence": "high"}]},
                          "research.extend_querygen": {"queries": [{"sub_q": 0, "q": "x new angle", "provider": "web"},
                                                                   {"sub_q": 1, "q": "y new angle", "provider": "web"}]},
                          }.get(role, {"queries": []}))
        rec = llm.CallRecord(role=role, tier="cheap", profile="fake", provider="fake", requested_model="m",
                             served_model="m", routed_via="", item=item)
        return llm.LLMResult(text=out, parsed=validate(out) if validate else llm.NOT_VALIDATED,
                             record=rec, records=[rec])

    def wave(queries, **kw):
        waves.append(kw["mode"])
        return [{"url": f"https://ext{i}.example/p", "title": "e", "snippet": "x y", "kind": "blog",
                 "provider": "ddgs", "sub_qs": q["sub_qs"], "text": "", "published": ""}
                for i, q in enumerate(queries)], []

    def fetch(src, **kw):
        fetches.append(src["url"])
        md = tmp_path / f"d{len(fetches)}.md"
        md.write_text(doc, encoding="utf-8")
        return FetchedItem(url=src["url"], title="t", markdown_path=str(md), excerpt=doc[:600],
                           sub_qs=list(src.get("sub_qs") or []), provider="ddgs", fetcher="native").model_dump(), []

    monkeypatch.setattr(research_mod.llm, "complete", complete)
    monkeypatch.setattr(research_mod, "run_wave", wave)
    monkeypatch.setattr(research_mod, "fetch_one_source", fetch)
    stubs = {
        "intake_node": lambda s: {"mode": "deep"},
        "brief_node": lambda s: {"brief": dict(brief), "plan": {"summary": "s", "sub_questions": ["a", "b"],
                                                                "planned_sources": [], "must_have_keywords": []}},
        "scout_node": lambda s: {"sources": [{"url": f"https://s{i}.example/p", "title": "s", "snippet": "x y",
                                              "kind": "blog", "provider": "ddgs", "sub_qs": [i % 2]}
                                             for i in range(4)],
                                 "search_spend": {"exa_search_first": 2}},
        "outline_node": lambda s: {"outline": {"sections": []}},
        "compose_node": lambda s: {"draft_md": "d", "findings": [], "sections": []},
        "panel_node": lambda s: {"panel_verdict": {"verdict": "pass"}, "review_verdict": {"verdict": "pass"}},
        "route_after_panel": lambda s: "report_builder",
        "report_builder_node": lambda s: {"report_paths": {"md": "r.md", "folder": "f"}},
    }
    for name, fn in stubs.items():
        monkeypatch.setattr(graph_mod, name, fn)

    g = graph_mod.build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t:extend"}}
    g.invoke({"thread_id": "t:extend", "user_id": 1, "user_request": "x and y", "length": "medium",
              "run_config": make_run_config("medium", **{"research.rerank": False}),
              "messages": [], "plan": None, "sources": [], "fetched": [], "findings": [], "draft_md": "",
              "revision_notes": [], "revision_rounds": 0, "model_calls": [], "hitl": {"pending": False}},
             config=cfg)
    g.invoke(Command(resume=True), config=cfg)                      # plan gate → the report gate
    first = g.get_state(cfg)
    assert first.next == ("deliver",) and len(first.values["fetched"]) == 4 and waves == []
    first_fetches = list(fetches)

    g.invoke(Command(resume=True), config=cfg)                      # report gate → END
    g.update_state(cfg, {"extend_requested": True, "extend_rounds": 0, "revision_rounds": 0,
                         "append_only": False}, as_node="deliver")   # the plain /continue fork
    roles.clear()
    g.invoke(None, config=cfg)

    snap = g.get_state(cfg)
    v = snap.values
    assert snap.next == ("deliver",), "back at the report gate"
    assert "research.extend_querygen" in roles and waves == ["extend"]
    assert fetches[len(first_fetches):] == ["https://ext0.example/p", "https://ext1.example/p"]
    spend = v["search_spend"]
    assert (spend["fetch_first"], spend["fetch_extend"], spend["extend_allowances"]) == (4, 2, 1)
    assert spend["exa_search_first"] == 2, "the scout's spend carried through both passes"
    assert (v["extend_rounds"], v["research_rounds"]) == (1, 2)
