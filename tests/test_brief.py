"""v3 brief node tests — scoping quality + the no-hallucinated-URLs
guarantee (successor of test_planner_no_hallucinated_urls.py and
test_planner_reflect.py).

Run with:
    PYTHONPATH='' ./venv/Scripts/python.exe -m pytest tests/test_brief.py -q
"""
from __future__ import annotations

import pytest

from argus.graph import brief as brief_mod
from argus.graph.brief import (
    BRIEF_SYSTEM, brief_node, brief_to_plan, check_brief_quality,
)
from argus.graph.state import ResearchBrief, SubQuestion


# ---------------------------------------------------------------------------
# URL integrity — structural, not prompt-begged
# ---------------------------------------------------------------------------

def test_brief_prompt_bans_urls():
    assert "Do NOT include URLs" in BRIEF_SYSTEM


def test_brief_model_has_no_url_field():
    """The v3 brief cannot carry URLs at all — hallucinated links are
    structurally impossible (v2 needed prompt rules + a scrubbing
    researcher; v3 removed the field)."""
    fields = set(ResearchBrief.model_fields) | set(SubQuestion.model_fields)
    assert not any("url" in f.lower() for f in fields), fields


def test_brief_to_plan_never_emits_target_urls():
    brief = ResearchBrief(
        sub_questions=[SubQuestion(q="How does attention work?",
                                   kind="paper")],
        must_have_keywords=["attention"], summary="s")
    plan = brief_to_plan(brief)
    for ps in plan["planned_sources"]:
        assert ps["target_url"] is None


def test_brief_to_plan_is_format_plan_compatible():
    """The Telegram plan-gate renderer reads these exact keys."""
    brief = ResearchBrief(
        sub_questions=[SubQuestion(q="a?"), SubQuestion(q="b?")],
        must_have_keywords=["kw"], summary="sum")
    plan = brief_to_plan(brief, queries=[
        {"query": "kw a", "provider": "ddgs", "sub_qs": [0]}])
    assert plan["sub_questions"] == ["a?", "b?"]
    assert plan["summary"] == "sum"
    assert plan["must_have_keywords"] == ["kw"]
    assert plan["planned_sources"][0]["query"] == "kw a"
    assert plan["planned_sources"][0]["kind"] == "search_result"


# ---------------------------------------------------------------------------
# Quality checks (successor of the planner_reflect checks)
# ---------------------------------------------------------------------------

def _good_brief() -> ResearchBrief:
    return ResearchBrief(
        sub_questions=[
            SubQuestion(q="What architectures dominate deep research "
                          "agents today?", kind="paper"),
            SubQuestion(q="Which open-source frameworks implement "
                          "supervisor patterns?", kind="repo"),
            SubQuestion(q="What are the unsolved reliability problems?",
                        kind="web"),
        ],
        must_have_keywords=["research", "agents", "supervisor"],
        summary="Survey the deep-research agent landscape.",
    )


def test_good_brief_passes_quality_checks():
    assert check_brief_quality(_good_brief(),
                               "deep research agents survey") == []


def test_too_few_sub_questions_fails():
    b = _good_brief()
    b.sub_questions = b.sub_questions[:1]
    failures = check_brief_quality(b, "deep research agents survey")
    assert any("sub_questions" in f for f in failures)


def test_verbatim_echo_sub_question_fails():
    b = _good_brief()
    b.sub_questions[0] = SubQuestion(q="deep research agents survey")
    failures = check_brief_quality(b, "deep research agents survey")
    assert any("verbatim" in f for f in failures)


def test_empty_keywords_fails():
    b = _good_brief()
    b.must_have_keywords = []
    failures = check_brief_quality(b, "deep research agents survey")
    assert any("keywords" in f for f in failures)


def test_unrelated_keywords_fail():
    b = _good_brief()
    b.must_have_keywords = ["cooking", "recipes"]
    failures = check_brief_quality(b, "deep research agents survey")
    assert any("no word" in f for f in failures)


# ---------------------------------------------------------------------------
# Node behaviour with a mocked LLM
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, content: str):
        self.content = content
        self.response_metadata = {"model_name": "fake"}
        self.usage_metadata = {"input_tokens": 1, "output_tokens": 1}


def _mock_llm(monkeypatch, responses: list[str]):
    calls = {"n": 0}

    def fake_chat_for_tier(tier, **kw):
        return object()

    def fake_invoke(chat, messages, **kw):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return _FakeResp(responses[i])

    monkeypatch.setattr(brief_mod.llm, "chat_for_tier", fake_chat_for_tier)
    monkeypatch.setattr(brief_mod.llm, "invoke_with_retry", fake_invoke)
    monkeypatch.setattr(brief_mod.llm, "resolve_tier", lambda t: "fake")
    return calls


GOOD_JSON = """{
  "sub_questions": [
    {"q": "What architectures dominate deep research agents?",
     "kind": "paper"},
    {"q": "Which frameworks implement supervisor patterns?",
     "kind": "repo"},
    {"q": "What reliability problems remain unsolved?", "kind": "web"}
  ],
  "must_have_keywords": ["research", "agents"],
  "summary": "Survey the landscape.",
  "success_criteria": ["names concrete architectures"]
}"""


def test_brief_node_happy_path(monkeypatch):
    calls = _mock_llm(monkeypatch, [GOOD_JSON])
    out = brief_node({"user_request": "deep research agents survey"})
    assert calls["n"] == 1
    assert len(out["brief"]["sub_questions"]) == 3
    assert out["plan"]["sub_questions"], "v2-compatible plan required"
    assert out["hitl"]["kind"] == "plan_approval"
    assert out["errors"] == []


def test_brief_node_redrafts_once_on_weak_brief(monkeypatch):
    weak = """{"sub_questions": [{"q": "deep research agents survey"}],
               "must_have_keywords": [], "summary": "s"}"""
    calls = _mock_llm(monkeypatch, [weak, GOOD_JSON])
    out = brief_node({"user_request": "deep research agents survey"})
    assert calls["n"] == 2, "weak brief must trigger exactly one re-draft"
    assert len(out["brief"]["sub_questions"]) == 3


def test_brief_node_fallback_summary_is_neutral_not_raw_llm_dump(monkeypatch):
    """The v2 bug class: raw LLM output must never reach the Telegram
    preview via the summary field."""
    garbage = "```json{ totally broken JSON >>>"
    calls = _mock_llm(monkeypatch, [garbage])
    out = brief_node({"user_request": "some tricky topic"})
    assert calls["n"] >= 1
    summary = out["plan"]["summary"]
    assert "```" not in summary
    assert "{" not in summary
    assert "⚠️" in summary
    assert any("brief" in e for e in out["errors"])


def test_brief_node_consumes_plan_feedback(monkeypatch):
    _mock_llm(monkeypatch, [GOOD_JSON])
    out = brief_node({"user_request": "topic",
                      "plan_feedback": "focus on streaming",
                      "brief": {"sub_questions": [], "summary": "old"}})
    assert out["plan_feedback"] == ""
