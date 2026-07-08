"""T-handler for handoff action #7 (HANDOFF-RESEARCH-2026-07-08.md §6).

Adds PlannerReflectNode: after the planner drafts a plan, the reflect
node runs five quality checks and re-plans ONCE if the plan is weak
(bounded: max 2 total plan attempts). This prevents T6-class
orphan-topic bugs at planning time instead of relying on the
researcher's fallback path.

Quality checks:
  (a) planned_sources count  >= 3 ideal; < 2 triggers re-plan
  (b) source_type diversity   >= 2 distinct kinds
  (c) sub_questions count     >= 3
  (d) no sub_question is too similar to user_request verbatim
  (e) at least 1 planned_source mentions a word from user_request

The tests below drive the reflect node directly with crafted state
dictionaries and (for the re-plan paths) monkey-patch the planner_node
to return a stub plan so the test is hermetic — no LLM, no network.
"""
from __future__ import annotations

import pytest

from argus.graph.nodes import (
    planner_reflect_node,
    _reflect_plan_quality,
)


# ---------------------------------------------------------------------------
# Helpers — minimal state dicts for each scenario.
# ---------------------------------------------------------------------------

USER_REQUEST = "How does retrieval augmented generation compare to long-context transformers?"


def _good_plan_dict() -> dict:
    return {
        "sub_questions": [
            "What is RAG and how does it retrieve external evidence?",
            "How do long-context transformers handle extended inputs?",
            "How is retrieval-augmented generation benchmarked?",
            "What are the trade-offs between RAG and long-context LLMs?",
        ],
        "planned_sources": [
            {"kind": "paper",
             "query": "retrieval augmented generation survey arxiv",
             "target_url": None,
             "rationale": "primary arxiv survey"},
            {"kind": "paper",
             "query": "long-context transformers evaluation",
             "target_url": None,
             "rationale": "primary paper"},
            {"kind": "blog",
             "query": "RAG vs long context production notes",
             "target_url": "https://example.com/rag-notes",
             "rationale": "engineer notes"},
            {"kind": "official_doc",
             "query": "LangChain RAG documentation",
             "target_url": "https://python.langchain.com/docs/rag",
             "rationale": "official docs"},
        ],
        "must_have_keywords": ["retrieval", "augmented", "transformer"],
        "summary": "Compare RAG vs long-context LLMs across benchmarks.",
    }


def _state_with_plan(plan: dict | None, user_request: str = USER_REQUEST,
                     plan_attempts: int = 1) -> dict:
    return {
        "user_request": user_request,
        "plan": plan,
        "plan_attempts": plan_attempts,
        "messages": [],
        "model_calls": [],
    }


def _stub_planner_node(state, plan_dict: dict) -> dict:
    """A drop-in replacement for planner_node that returns a fixed plan.

    Used to verify reflect node re-invokes the planner and merges its
    output into state without doing any real LLM call.
    """
    return {
        "plan": plan_dict,
        "model_calls": [{"tier": "stub", "purpose": "test",
                         "prompt_tokens": 0, "completion_tokens": 0}],
        "messages": [{"role": "assistant",
                      "content": "stub planner returned a plan"}],
    }


# ---------------------------------------------------------------------------
# (a) Empty plan → re-plan
# ---------------------------------------------------------------------------

def test_empty_plan_triggers_replan(monkeypatch):
    """Plan dict with no sub_questions and no planned_sources is weak.

    The reflect node should call planner_node exactly once, end up with
    plan_attempts=2, and the new plan should be present in state.
    """
    state = _state_with_plan({
        "sub_questions": [],
        "planned_sources": [],
        "must_have_keywords": [],
        "summary": "",
    })
    monkeypatch.setattr(
        "argus.graph.nodes.planner_node",
        lambda s: _stub_planner_node(s, _good_plan_dict()),
    )
    out = planner_reflect_node(state)
    assert out["plan_attempts"] == 2, (
        "planner_reflect_node must increment attempts on re-plan")
    assert out["plan"] is not None
    assert len(out["plan"].get("sub_questions", [])) >= 3
    # model_calls is additive, so the stubbed planner call lands in
    # the merged model_calls list.
    assert any(
        mc.get("purpose") == "test" for mc in out["model_calls"]
    ), "reflect node must invoke planner_node and merge its model_calls"


# ---------------------------------------------------------------------------
# (b) Single-source plan → re-plan (count < 2)
# ---------------------------------------------------------------------------

def test_single_source_plan_triggers_replan(monkeypatch):
    """A plan with one planned_source fails check (a) — re-plan."""
    weak_plan = {
        "sub_questions": ["q1", "q2", "q3"],
        "planned_sources": [{
            "kind": "paper", "query": "x",
            "target_url": None, "rationale": "only one",
        }],
        "must_have_keywords": ["retrieval"],
        "summary": "thin plan",
    }
    state = _state_with_plan(weak_plan)
    monkeypatch.setattr(
        "argus.graph.nodes.planner_node",
        lambda s: _stub_planner_node(s, _good_plan_dict()),
    )
    out = planner_reflect_node(state)
    assert out["plan_attempts"] == 2
    assert len(out["plan"]["planned_sources"]) >= 3


# ---------------------------------------------------------------------------
# (c) All-arxiv plan → re-plan (diversity fail: only 1 distinct kind)
# ---------------------------------------------------------------------------

def test_all_arxiv_plan_triggers_replan_diversity(monkeypatch):
    """All paper-kind sources → fails (b) source_type diversity < 2."""
    weak_plan = {
        "sub_questions": ["q1", "q2", "q3"],
        "planned_sources": [
            {"kind": "paper", "query": f"q{i}",
             "target_url": None, "rationale": "arxiv"} for i in range(5)
        ],
        "must_have_keywords": ["retrieval"],
        "summary": "arxiv-only",
    }
    state = _state_with_plan(weak_plan)
    monkeypatch.setattr(
        "argus.graph.nodes.planner_node",
        lambda s: _stub_planner_node(s, _good_plan_dict()),
    )
    out = planner_reflect_node(state)
    assert out["plan_attempts"] == 2, (
        "all-paper plan must fail diversity check (b) and trigger re-plan")


# ---------------------------------------------------------------------------
# (d) Duplicate sub_question (verbatim user_request) → re-plan
# ---------------------------------------------------------------------------

def test_duplicate_sub_question_triggers_replan(monkeypatch):
    """A sub_question that is the user_request verbatim signals a lazy
    plan; check (d) forces a re-plan."""
    weak_plan = {
        "sub_questions": [
            USER_REQUEST,  # verbatim
            "What are benchmark trade-offs?",
            "How is performance measured?",
        ],
        "planned_sources": [
            {"kind": "paper", "query": "x", "target_url": None,
             "rationale": "r"},
            {"kind": "blog", "query": "y", "target_url": "https://a",
             "rationale": "r"},
            {"kind": "official_doc", "query": "z",
             "target_url": "https://b", "rationale": "r"},
        ],
        "must_have_keywords": ["retrieval"],
        "summary": "lazy",
    }
    state = _state_with_plan(weak_plan)
    monkeypatch.setattr(
        "argus.graph.nodes.planner_node",
        lambda s: _stub_planner_node(s, _good_plan_dict()),
    )
    out = planner_reflect_node(state)
    assert out["plan_attempts"] == 2, (
        "verbatim sub_question must trigger re-plan")


# ---------------------------------------------------------------------------
# (e) Good plan → passes through (no re-plan, no LLM call)
# ---------------------------------------------------------------------------

def test_good_plan_passes_through(monkeypatch):
    """A high-quality plan satisfies all five checks; reflect node
    must NOT call planner_node and must NOT increment plan_attempts.
    """
    state = _state_with_plan(_good_plan_dict())

    called = {"n": 0}

    def _spy_planner(s):
        called["n"] += 1
        return {"plan": _good_plan_dict(), "model_calls": [],
                "messages": []}

    monkeypatch.setattr("argus.graph.nodes.planner_node", _spy_planner)
    out = planner_reflect_node(state)
    assert called["n"] == 0, "planner_node must not be called on good plan"
    assert out["plan_attempts"] == 1, (
        "plan_attempts must NOT increment when plan passes reflect")
    assert out["plan"] is state["plan"], "good plan must be left intact"


# ---------------------------------------------------------------------------
# Bonus: bounded re-plan — second weak plan proceeds with warning
# ---------------------------------------------------------------------------

def test_bounded_replan_after_two_attempts(monkeypatch):
    """If both plan attempts are weak, the reflect node must NOT loop
    a third time — instead it returns the (still-weak) plan with a
    warning message attached.
    """
    weak_plan = {
        "sub_questions": [],
        "planned_sources": [],
        "must_have_keywords": [],
        "summary": "still empty",
    }
    state = _state_with_plan(weak_plan, plan_attempts=2)  # already max
    out = planner_reflect_node(state)
    assert out["plan_attempts"] == 2, (
        "plan_attempts must not increment past MAX_ATTEMPTS=2")
    # A warning should be in the messages tail.
    msgs = out.get("messages") or []
    assert any(
        "weak" in (m.get("content") or "").lower() or
        "warn" in (m.get("content") or "").lower()
        for m in msgs
    ), "expected a warning message when reflect gives up"


# ---------------------------------------------------------------------------
# Direct unit test for the helper — keeps the signal logic readable.
# ---------------------------------------------------------------------------

def test_reflect_helper_returns_failure_codes():
    """_reflect_plan_quality returns a list of (check_name, ok, detail).

    Spot-check that the helper flags each weak dimension correctly on a
    known-bad plan, and reports ok=True across the board on a known-good
    plan.
    """
    weak = {
        "sub_questions": [USER_REQUEST],     # fails (c) AND (d)
        "planned_sources": [],               # fails (a)
        "must_have_keywords": [],
        "summary": "",
    }
    fails = _reflect_plan_quality(weak, USER_REQUEST)
    names_ok = {n: ok for (n, ok, _d) in fails}
    assert names_ok["sources_count"] is False
    assert names_ok["source_diversity"] is False
    assert names_ok["sub_questions_count"] is False
    assert names_ok["sub_question_duplicate"] is False
    assert names_ok["keyword_coverage"] is False

    strong = _good_plan_dict()
    passes = _reflect_plan_quality(strong, USER_REQUEST)
    assert all(ok for (_n, ok, _d) in passes), (
        f"good plan unexpectedly failed: {passes}")