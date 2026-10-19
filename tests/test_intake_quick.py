"""v4 WP-C — intake and the quick answer through ``llm.complete`` (spec §3 WP-C):
``intake.classify`` degrades to ``default_classification`` (deep, the raw
request); ``quick.answer`` raises."""
from __future__ import annotations

import pytest

from argus import llm
from argus.engines.llm_openai import DriverError
from argus.graph.nodes import intake_node, quick_answer_node
from helpers import fake_llm, make_run_config


def _state(**over) -> dict:
    return {"user_request": "  what is X?  ", "run_config": make_run_config("short"), **over}


def test_intake_classifies_and_refines(monkeypatch):
    seen = fake_llm(monkeypatch, lambda system, human: '{"mode": "quick", "refined_query": "What is X?"}')
    out = intake_node(_state())
    assert (out["mode"], out["user_request"]) == ("quick", "What is X?")
    assert [c["role"] for c in out["model_calls"]] == ["intake.classify"] and "errors" not in out
    assert seen[0][1] == "  what is X?  "


@pytest.mark.parametrize("reply", [
    lambda system, human: (_ for _ in ()).throw(DriverError("provider fake: boom")),
    lambda system, human: '{"mode": "maybe", "refined_query": "x"}',
])
def test_intake_degrades_to_deep_on_the_raw_request(monkeypatch, reply):
    fake_llm(monkeypatch, reply)
    out = intake_node(_state(revise_rounds=1))
    assert (out["mode"], out["user_request"]) == ("deep", "  what is X?  ")
    assert len(out["errors"]) == 1
    assert out["errors"][0].startswith("[r2] degrade: intake.classify → default_classification (role "
                                       "intake.classify: every engine failed")
    assert out["model_calls"] and all(c["role"] == "intake.classify" for c in out["model_calls"])


def test_the_quick_answer(monkeypatch):
    fake_llm(monkeypatch, lambda system, human: "X is a graph runtime.")
    out = quick_answer_node(_state())
    assert out["quick_answer"] == "X is a graph runtime."
    assert [c["role"] for c in out["model_calls"]] == ["quick.answer"]


def test_a_failed_quick_answer_raises(monkeypatch):
    fake_llm(monkeypatch, lambda system, human: (_ for _ in ()).throw(DriverError("provider fake: boom")))
    with pytest.raises(llm.EngineError, match="quick.answer"):
        quick_answer_node(_state())
