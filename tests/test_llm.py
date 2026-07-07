"""Argus LLM adapter tests (live against FreeLLMAPI).

These hit the real proxy because that is the contract we ship against.
We never hardcode model names — every test resolves from /v1/models.
"""
from __future__ import annotations

import pytest

from argus.llm import (
    fetch_live_models, resolve_tier, pick_strong_and_judge,
    chat_for_tier, _family,
)


def test_models_endpoint_alive():
    ids = fetch_live_models(force=True)
    assert isinstance(ids, list)
    assert len(ids) >= 50  # we observed 96+ in production
    assert "auto" in ids


def test_resolve_each_tier():
    for tier in ("cheap", "strong", "judge"):
        m = resolve_tier(tier, force=True)
        assert m and m != "", f"tier {tier} resolved empty"
    # cheap must be the smallest family name (heuristic: small id)
    cheap = resolve_tier("cheap", force=True)
    assert cheap in fetch_live_models()


def test_strong_and_judge_different_families():
    strong, judge = pick_strong_and_judge(force=True)
    assert strong and judge
    if strong != "auto" and judge != "auto":
        assert _family(strong) != _family(judge), (
            f"judge {judge} shares family with strong {strong}")


def test_family_buckets():
    assert _family("qwen/qwen3-coder:free").startswith("qwen")
    assert _family("openai/gpt-oss-120b:free").startswith("openai")
    assert _family("gemini-3.5-flash") == "gemini-3.5-flash"
    assert _family("auto") == "auto"


def test_chat_for_tier_smoke():
    chat = chat_for_tier("cheap", temperature=0, max_tokens=10)
    from langchain_core.messages import HumanMessage
    r = chat.invoke([HumanMessage(content="reply with the single word PONG")])
    assert r.content
    assert isinstance(r.content, str)


def test_record_from_response_extracts_model():
    class _R:
        response_metadata = {"model_name": "auto/test", "_routed_via":
                             {"platform": "groq"}}
        usage_metadata = {"input_tokens": 5, "output_tokens": 7}
    from argus.llm import record_from_response
    rec = record_from_response("cheap", "auto", _R())
    assert rec.served_model == "auto/test"
    assert rec.served_provider == "groq"
    assert rec.prompt_tokens == 5
    assert rec.completion_tokens == 7