"""Brief node tests — scoping quality, the no-hallucinated-URLs guarantee
(successor of test_planner_no_hallucinated_urls.py and
test_planner_reflect.py), and the v4 fields (spec §3 WP-I): archetype,
time sensitivity, recency hint, academic flag and the entity phrases the
grounding check reads; ``brief.draft`` through ``llm.complete``.

Run with:
    PYTHONPATH='' ./venv/Scripts/python.exe -m pytest tests/test_brief.py -q
"""
from __future__ import annotations

import json

import pytest

from argus import llm
from argus.graph.brief import (
    BRIEF_SYSTEM, brief_node, brief_to_plan, check_brief_quality,
)
from argus.graph.grounding import check_grounding
from argus.graph.state import ResearchBrief, SubQuestion

from helpers import fake_llm, make_run_config


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
# Node behaviour: the real llm.complete on a fake driver
# ---------------------------------------------------------------------------

GOOD = {
    "sub_questions": [
        {"q": "What architectures dominate deep research agents?", "kind": "paper"},
        {"q": "Which frameworks implement supervisor patterns?", "kind": "repo"},
        {"q": "What reliability problems remain unsolved?", "kind": "web"}],
    "must_have_keywords": ["research", "agents"],
    "summary": "Survey the landscape.",
    "success_criteria": ["names concrete architectures"],
}
GOOD_JSON = json.dumps(GOOD)


def _state(request: str, **extra) -> dict:
    return {"user_request": request, "run_config": make_run_config("short"), **extra}


def _replies(*texts):
    """``reply`` for fake_llm: the given texts in turn (the last one repeats)."""
    it = iter(texts)
    last = [texts[-1]]

    def reply(system, human):
        last[0] = next(it, last[0])
        return last[0]
    return reply


def test_brief_node_happy_path(monkeypatch):
    seen = fake_llm(monkeypatch, _replies(GOOD_JSON))
    out = brief_node(_state("deep research agents survey"))
    assert len(seen) == 1 and seen[0][0] == BRIEF_SYSTEM
    assert len(out["brief"]["sub_questions"]) == 3
    assert out["plan"]["sub_questions"], "v2-compatible plan required"
    assert out["hitl"]["kind"] == "plan_approval"
    assert out["errors"] == []
    assert [(c["role"], c["ok"]) for c in out["model_calls"]] == [("brief.draft", True)]


def test_brief_node_redrafts_once_on_weak_brief(monkeypatch):
    weak = json.dumps({"sub_questions": [{"q": "deep research agents survey"}],
                       "must_have_keywords": [], "summary": "s"})
    seen = fake_llm(monkeypatch, _replies(weak, GOOD_JSON))
    out = brief_node(_state("deep research agents survey"))
    assert len(seen) == 2, "weak brief must trigger exactly one re-draft"
    assert "Your previous brief was rejected" in seen[1][1]
    assert len(out["brief"]["sub_questions"]) == 3
    assert [c["role"] for c in out["model_calls"]] == ["brief.draft", "brief.draft"]


def test_an_unparseable_reply_follows_the_facade_retry_and_both_records_are_kept(monkeypatch):
    """v3 dropped the successful call's record when parsing failed afterwards;
    the validate hook makes the failed parse an attempt: nudged, retried, kept."""
    seen = fake_llm(monkeypatch, _replies("I think the brief should cover agents.", GOOD_JSON))
    out = brief_node(_state("deep research agents survey"))
    assert seen[1][2] == "Reply with only the JSON object."
    calls = out["model_calls"]
    assert [(c["role"], c["ok"], c["retry_reason"]) for c in calls] == [
        ("brief.draft", False, ""), ("brief.draft", True, "unparseable")]
    assert calls[0]["error"].startswith("unparseable: ValueError: no useful JSON object found")
    assert len(out["brief"]["sub_questions"]) == 3


def test_brief_draft_raises_when_every_attempt_is_unparseable(monkeypatch):
    """``brief.draft`` must raise (spec §2.1): no generic fallback brief, and
    raw LLM output never reaches the plan preview."""
    fake_llm(monkeypatch, _replies("```json{ totally broken JSON >>>"))
    with pytest.raises(llm.EngineError) as exc:
        brief_node(_state("some tricky topic"))
    assert not isinstance(exc.value, (llm.EngineSkipped, llm.EngineDegraded))
    assert [a.ok for a in exc.value.attempts] == [False, False]


def test_a_sub_questions_value_that_is_not_a_list_is_unparseable(monkeypatch):
    bad = json.dumps({**GOOD, "sub_questions": "background; current state"})
    seen = fake_llm(monkeypatch, _replies(bad, GOOD_JSON))
    out = brief_node(_state("deep research agents survey"))
    assert len(seen) == 2 and len(out["brief"]["sub_questions"]) == 3


def test_brief_node_consumes_plan_feedback(monkeypatch):
    seen = fake_llm(monkeypatch, _replies(GOOD_JSON))
    out = brief_node(_state("topic", plan_feedback="focus on streaming",
                            brief={"sub_questions": [], "summary": "old"}))
    assert out["plan_feedback"] == ""
    assert "focus on streaming" in seen[0][1] and '"summary": "old"' in seen[0][1]


# ---------------------------------------------------------------------------
# v4 fields: archetype, time sensitivity, recency, academic flag, entities
# ---------------------------------------------------------------------------

def test_the_prompt_asks_for_every_v4_field():
    for key in ("archetype", "time_sensitive", "recency_hint", "is_academic_topic", "entities"):
        assert f'"{key}"' in BRIEF_SYSTEM
    for name in ("explainer", "definition", "comparison", "forecast", "sota_review", "howto",
                 "news_update", "decision"):
        assert f'"{name}"' in BRIEF_SYSTEM


def test_v4_fields_are_parsed(monkeypatch):
    reply = {**GOOD, "archetype": "forecast", "time_sensitive": True, "recency_hint": "M",
             "is_academic_topic": False, "entities": ["Gold"]}
    fake_llm(monkeypatch, _replies(json.dumps(reply)))
    b = brief_node(_state("gold price outlook for the next month"))["brief"]
    assert (b["archetype"], b["time_sensitive"], b["recency_hint"], b["is_academic_topic"],
            b["entities"]) == ("forecast", True, "m", False, ["gold"])


def test_missing_v4_fields_get_the_deterministic_defaults(monkeypatch):
    fake_llm(monkeypatch, _replies(GOOD_JSON))
    b = brief_node(_state("deep research agents survey"))["brief"]
    assert (b["archetype"], b["time_sensitive"], b["recency_hint"], b["is_academic_topic"],
            b["entities"]) == ("explainer", False, None, False, [])


@pytest.mark.parametrize("field,value,expected", [
    ("archetype", "listicle", "explainer"),        # not an archetype → the default
    ("archetype", " Comparison ", "comparison"),
    ("time_sensitive", "true", True),
    ("time_sensitive", "yes please", False),
    ("is_academic_topic", "TRUE", True),
    ("recency_hint", "null", None),
    ("recency_hint", "", None),
    ("recency_hint", "quarter", "quarter"),        # kept: resolve_recency ignores it with a note
    ("entities", "Nebulon AI", []),                # not a list
])
def test_malformed_v4_fields(monkeypatch, field, value, expected):
    fake_llm(monkeypatch, _replies(json.dumps({**GOOD, field: value})))
    assert brief_node(_state("Nebulon AI agents survey"))["brief"][field] == expected


Q013 = "Quantum Lattice Transformer by Nebulon AI benchmark results"


def test_entities_are_the_normalised_names_the_request_contains(monkeypatch):
    """The grounding check reads full entity phrases (WP-B): lower-cased and
    whitespace-collapsed, only phrases the request names — an invented
    abbreviation never becomes something the documents must contain."""
    reply = {**GOOD, "entities": ["Quantum  Lattice Transformer", "Nebulon AI", "QLT",
                                  "nebulon ai", 42]}
    fake_llm(monkeypatch, _replies(json.dumps(reply)))
    out = brief_node(_state(Q013))
    assert out["brief"]["entities"] == ["quantum lattice transformer", "nebulon ai"]
    # … and the grounding check is no longer vacuous: a page naming only one entity is not enough
    fetched = [{"url": f"https://n{i}.example/p", "title": "Nebulon AI ships a chatbot", "excerpt": ""}
               for i in range(3)]
    evidence = [{"source_id": i + 1, "source_url": f["url"], "relevance": 4, "authority": 1}
                for i, f in enumerate(fetched)]
    g = check_grounding({"brief": out["brief"], "fetched": fetched, "evidence": evidence})
    assert (g.grounded, g.entity_tokens, g.mention_count) == (
        False, ["quantum lattice transformer", "nebulon ai"], 0)


def test_a_generic_request_has_no_entities(monkeypatch):
    reply = {**GOOD, "entities": ["Doomscrolling Research Institute"]}     # not in the request
    fake_llm(monkeypatch, _replies(json.dumps(reply)))
    out = brief_node(_state("psychological and social factors behind doomscrolling"))
    assert out["brief"]["entities"] == []


def test_entities_may_come_from_the_plan_feedback(monkeypatch):
    reply = {**GOOD, "entities": ["Nebulon AI", "QLT-2"]}
    fake_llm(monkeypatch, _replies(json.dumps(reply)))
    out = brief_node(_state(Q013, plan_feedback="focus on the QLT-2 release",
                            brief={"sub_questions": [], "summary": "old"}))
    assert out["brief"]["entities"] == ["nebulon ai", "qlt-2"]
