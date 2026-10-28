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
        self.kw = kw
        o = self.outcomes.pop(0)
        if isinstance(o, BaseException):
            raise o
        return o


_POLICY_ROLE = {"raise": "brief.draft", "skip_item": "research.digest", "degrade": "research.rerank"}


def _snap(fallback: bool = False, on_exhausted: str = "raise") -> EnginesSnapshot:
    role = _POLICY_ROLE[on_exhausted]
    b = ResolvedBinding(role=role, tier="strong", primary=EngineSpec(provider="p", model="m"),
                        fallbacks=[EngineSpec(provider="f", model="m")] if fallback else [],
                        on_exhausted=on_exhausted, retry_on_truncation=True)
    return EnginesSnapshot(profile="t", resolved_at="2026-10-19T00:00:00+00:00", roles={role: b})


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


def test_a_429_keeps_waiting_while_quota_wait_lasts(env):
    env.providers["p"] = ProviderSpec(name="p", base_url="http://p/v1", quota_wait_s=70)
    d = env.drive(*[_rate_limited()] * 5, _ok())
    out = llm.complete("brief.draft", MSGS, engines=_snap())
    assert d.calls == 6 and out.text == "ok"
    assert env.sleeps == [1.0, 2.0, 30.0, 30.0, 10.0], "quick tries, then 30 s steps up to quota_wait_s"
    assert [r.retry_reason for r in out.records][-3:] == ["quota_wait"] * 3


def test_quota_wait_is_off_by_default_and_never_applies_to_5xx(env):
    d = env.drive(*[_rate_limited()] * 5)
    with pytest.raises(llm.EngineError):
        llm.complete("brief.draft", MSGS, engines=_snap())
    assert d.calls == 3

    llm._quota_out.clear()
    env.providers["p"] = ProviderSpec(name="p", base_url="http://p/v1", quota_wait_s=120)
    err = openai.InternalServerError("500", response=httpx.Response(500, request=_REQ), body=None)
    d = env.drive(*[err] * 5)
    with pytest.raises(llm.EngineError):
        llm.complete("brief.draft", MSGS, engines=_snap())
    assert d.calls == 3, "a provider error is not an empty quota"


@pytest.mark.parametrize("policy", ["raise", "skip_item", "degrade"])
def test_every_engine_out_of_quota_raises_quota_exhausted_whatever_the_policy(env, policy):
    env.drive(*[_rate_limited()] * 6)
    with pytest.raises(llm.QuotaExhausted) as ei:
        llm.complete(_POLICY_ROLE[policy], MSGS, engines=_snap(fallback=True, on_exhausted=policy))
    assert not isinstance(ei.value, (llm.EngineSkipped, llm.EngineDegraded)), "no node may skip past it"
    assert "out of quota" in str(ei.value) and len(ei.value.attempts) == 6


def test_a_fallback_that_fails_otherwise_keeps_the_binding_policy(env):
    bad = openai.BadRequestError("400", response=httpx.Response(400, request=_REQ), body=None)
    env.drive(*[_rate_limited()] * 3, bad)
    with pytest.raises(llm.EngineSkipped):
        llm.complete("research.digest", MSGS, engines=_snap(fallback=True, on_exhausted="skip_item"))


def test_an_out_of_quota_provider_fails_fast_until_the_latch_expires(env, monkeypatch):
    clock = SimpleNamespace(t=1000.0)
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock.t)
    d = env.drive(*[_rate_limited()] * 3, _ok())
    with pytest.raises(llm.QuotaExhausted):
        llm.complete("brief.draft", MSGS, engines=_snap())
    with pytest.raises(llm.QuotaExhausted) as ei:       # a parallel item: no call, no wait
        llm.complete("brief.draft", MSGS, engines=_snap())
    assert d.calls == 3 and len(env.sleeps) == 2 and "out of quota" in ei.value.attempts[0].error
    clock.t += llm._QUOTA_LATCH_S
    assert llm.complete("brief.draft", MSGS, engines=_snap()).text == "ok"
    assert d.calls == 4


@pytest.mark.parametrize("asked, sent", [(200, 2048), (9000, 9000), (None, None)])
def test_min_max_tokens_floors_the_requested_budget(env, asked, sent):
    env.providers["p"] = ProviderSpec(name="p", base_url="http://p/v1", min_max_tokens=2048)
    d = env.drive(_ok())
    llm.complete("brief.draft", MSGS, engines=_snap(), max_tokens=asked)
    assert d.kw["max_tokens"] == sent


def test_every_attempt_records_when_it_was_sent(env):
    from datetime import datetime
    env.drive(_rate_limited(), _ok())
    out = llm.complete("brief.draft", MSGS, engines=_snap())
    stamps = [datetime.fromisoformat(r.started_at) for r in out.records]
    assert len(stamps) == 2 and stamps[0] <= stamps[1] and stamps[0].tzinfo is not None


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


# Untagged reasoning on a markdown role, as shipped in run 8a0b122d's "Competitive
# Landscape" section (meta/muse-glimmer-30b, finish_reason=length) — verbatim excerpt.
_LEAK_ECHO = """Report topic: Analyze viability and monetization strategies for worldwide knowledge archive platform featuring interactive maps, timeline sliders, cultural data, future university integration. Provide recommendations for business model, revenue streams, growth strategy.
Section: The Competitive Landscape: Existing Platforms and Models
Target ~3200 chars.

We need use ONLY evidence notes provided. Cite with [n] immediately after each factual claim, where n is source id. Multiple ids allowed.

Every number, date, name must come from claim/quote in evidence.

We need to use evidence notes: [1] Memory Mapper, [4] Peripleo, [10] HistoAtlas, [17] Historica, [20] OldMapsOnline.

We must cite each factual claim.

Let's extract facts:

Memory Mapper: open-source web application for mapping cultural heritage developed at UCL [1]."""
# The other shape (run 20260715 DSpark lecture): reasoning opens the reply.
_LEAK_OPENER = ("Okay, let's tackle this section on DSpark's speculative decoding. The user wants a focus on core "
                "principles and mechanisms, specifically answering the sub-question about the core principles.\n\n"
                "Speculative decoding drafts tokens with a small model and verifies them with the large one [1].")
_CLEAN_SECTION = ("A worldwide knowledge archive with interactive maps, timeline sliders, and cultural data is technically "
                  "feasible: multiple production systems demonstrate the core stack. Annalium delivers a free, no-login "
                  "world-history map with multi-track timeline using PostGIS for spatial-temporal queries [26].")


def _section_snap() -> EnginesSnapshot:
    b = ResolvedBinding(role="compose.section", tier="strong", primary=EngineSpec(provider="p", model="m"),
                        fallbacks=[EngineSpec(provider="f", model="m")], on_exhausted="skip_item",
                        retry_on_truncation=True)
    return EnginesSnapshot(profile="t", resolved_at="2026-10-19T00:00:00+00:00", roles={"compose.section": b})


@pytest.mark.parametrize("leak", [_LEAK_ECHO, _LEAK_OPENER, "<think>\n" + _LEAK_OPENER],
                         ids=["prompt-echo", "opener", "unclosed-think"])
def test_an_untagged_reasoning_trace_on_a_markdown_role_is_rejected_and_retried(env, leak):
    d = env.drive(_ok(leak), _ok(_CLEAN_SECTION))
    out = llm.complete("compose.section", MSGS, engines=_section_snap())
    assert d.calls == 2 and out.text == _CLEAN_SECTION
    first, second = out.records
    assert not first.ok and first.error.startswith("unparseable: ") and "reasoning" in first.error
    assert second.ok and second.retry_reason == "unparseable"


def test_a_reasoning_trace_from_every_engine_exhausts_the_role(env):
    env.drive(*[_ok(_LEAK_ECHO)] * 4)
    with pytest.raises(llm.EngineSkipped):
        llm.complete("compose.section", MSGS, engines=_section_snap())


def test_report_prose_is_not_mistaken_for_reasoning(env):
    env.drive(_ok(_CLEAN_SECTION))
    assert llm.complete("compose.section", MSGS, engines=_section_snap()).text == _CLEAN_SECTION
