"""Live contract tests against the FreeLLMAPI proxy (excluded from CI).

They hit the real proxy because that is the contract we ship against, and
never hardcode model names: the shipped argus_engines.json is resolved
against the live /models list."""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage

from argus import llm
from argus.engines import registry


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch):
    monkeypatch.setattr(registry, "_state", registry._State())


def test_models_endpoint_alive():
    asyncio.run(registry.load_config([str(registry._ROOT / "argus_engines.json")]))
    ids = registry.live_models("freellmapi")
    assert ids and "auto" in ids


def test_free_proxy_resolves_and_complete_records_provenance():
    asyncio.run(registry.load_config([str(registry._ROOT / "argus_engines.json")]))
    snap = asyncio.run(registry.active_snapshot())
    live = registry.live_models("freellmapi")
    for tier in ("cheap", "strong", "judge"):
        assert snap.roles[f"legacy.{tier}"].primary.model in live, tier

    out = llm.complete("legacy.cheap", [HumanMessage(content="reply with the single word PONG")],
                       engines=snap, temperature=0, max_tokens=32)
    assert out.text and isinstance(out.text, str)
    rec = out.record
    assert rec.ok and rec.served_model and rec.provider == "freellmapi"
    assert rec.routed_via, "the proxy's x-routed-via / _routed_via must be captured"
