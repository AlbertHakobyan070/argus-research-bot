"""Retry policy of the LLM facade (v4 spec §2.3). Hermetic (fake driver).

Connection errors / 5xx / 429 → up to 3 tries with backoff, honouring
Retry-After up to ``retry_after_cap_s``; a timeout is retried at most
``timeout_retry_max`` times; anything else fails the engine at once (next
fallback). The SDK itself never retries (``max_retries=0``, test_llm_facade)."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import openai
import pytest
from langchain_core.messages import HumanMessage

from argus import llm
from argus.engines import registry
from argus.engines.llm_openai import DriverResponse
from argus.engines.schema import EngineSpec, EnginesSnapshot, ProviderSpec, ResolvedBinding

_REQ = httpx.Request("POST", "http://p/v1/chat/completions")
MSGS = [HumanMessage(content="hi")]


def _ok(text="ok"):
    return DriverResponse(text=text, served_model="m", routed_via="", finish_reason="stop",
                          usage={}, headers={}, raw=None)


def _rate_limited(retry_after=None):
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return openai.RateLimitError("429", response=httpx.Response(429, headers=headers, request=_REQ),
                                 body=None)


class _Driver:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, engine, provider, messages, **kw):
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, BaseException):
            raise o
        return o


def _snap(fallback: bool = False) -> EnginesSnapshot:
    b = ResolvedBinding(role="brief.draft", tier="strong", primary=EngineSpec(provider="p", model="m"),
                        fallbacks=[EngineSpec(provider="f", model="m")] if fallback else [],
                        on_exhausted="raise", retry_on_truncation=True)
    return EnginesSnapshot(profile="t", resolved_at="2026-10-19T00:00:00+00:00",
                           roles={"brief.draft": b})


@pytest.fixture
def env(monkeypatch):
    """Providers looked up by the facade, a recording sleep, a fake driver."""
    state = SimpleNamespace(sleeps=[], providers={
        "p": ProviderSpec(name="p", base_url="http://p/v1"),
        "f": ProviderSpec(name="f", base_url="http://f/v1")})
    monkeypatch.setattr(registry, "provider", lambda name: state.providers[name])
    monkeypatch.setattr(llm, "_sleep", state.sleeps.append)

    def drive(*outcomes):
        state.driver = _Driver(outcomes)
        monkeypatch.setattr(llm, "_driver", state.driver)
        return state.driver
    state.drive = drive
    return state


def test_transient_errors_are_retried_then_succeed(env):
    d = env.drive(openai.APIConnectionError(request=_REQ), _rate_limited(), _ok())
    out = llm.complete("brief.draft", MSGS, engines=_snap())
    assert d.calls == 3 and out.text == "ok"
    assert [r.retry_reason for r in out.records] == ["", "transient", "transient"]
    assert env.sleeps == [1.0, 2.0]


def test_transient_errors_give_up_after_three_tries(env):
    d = env.drive(*[openai.APIConnectionError(request=_REQ)] * 5)
    with pytest.raises(llm.EngineError):
        llm.complete("brief.draft", MSGS, engines=_snap())
    assert d.calls == 3, "exactly three tries on one engine, no more"


def test_non_transient_error_fails_fast_to_the_next_fallback(env):
    bad = openai.BadRequestError("400", response=httpx.Response(400, request=_REQ), body=None)
    d = env.drive(bad, _ok("from fallback"))
    out = llm.complete("brief.draft", MSGS, engines=_snap(fallback=True))
    assert d.calls == 2 and out.text == "from fallback" and env.sleeps == []
    assert [(r.fallback_rank, r.ok) for r in out.records] == [(0, False), (1, True)]


def test_a_timeout_is_retried_once_and_no_more(env):
    d = env.drive(*[openai.APITimeoutError(request=_REQ)] * 4)
    with pytest.raises(llm.EngineError) as ei:
        llm.complete("brief.draft", MSGS, engines=_snap())
    assert d.calls == 2
    assert [r.retry_reason for r in ei.value.attempts] == ["", "timeout"]
    assert env.sleeps == [], "a timeout already waited; no backoff on top"

    env.providers["p"] = ProviderSpec(name="p", base_url="http://p/v1", timeout_retry_max=0)
    d = env.drive(*[openai.APITimeoutError(request=_REQ)] * 2)
    with pytest.raises(llm.EngineError):
        llm.complete("brief.draft", MSGS, engines=_snap())
    assert d.calls == 1


@pytest.mark.parametrize("header, cap, slept", [("2", 30.0, 2.0), ("9999", 30.0, 30.0), ("40", 5.0, 5.0)])
def test_retry_after_is_honoured_up_to_the_cap(env, header, cap, slept):
    env.providers["p"] = ProviderSpec(name="p", base_url="http://p/v1", retry_after_cap_s=cap)
    env.drive(_rate_limited(header), _ok())
    out = llm.complete("brief.draft", MSGS, engines=_snap())
    assert env.sleeps == [slept]
    assert out.records[1].retry_reason == "retry_after"


def test_rpm_limit_waits_for_the_sliding_window(env, monkeypatch):
    clock = SimpleNamespace(t=1000.0)
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock.t)

    def sleep(s):
        env.sleeps.append(s)
        clock.t += s
    monkeypatch.setattr(llm, "_sleep", sleep)
    env.providers["p"] = ProviderSpec(name="p", base_url="http://p/v1", rpm_limit=2)
    monkeypatch.setattr(llm, "_limiter", llm._RateLimiter())
    env.drive(_ok(), _ok(), _ok())
    for _ in range(3):
        llm.complete("brief.draft", MSGS, engines=_snap())
    assert env.sleeps == [60.0], "the third request in one minute waits for the window"
