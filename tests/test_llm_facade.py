"""LLM facade (v4 WP-A, spec §2.3): ``complete()`` semantics on a fake
driver, the openai-SDK driver's request shape on a mock transport, and the
Phase 1–2 legacy wrappers. Hermetic: no network."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from argus import llm
from argus.engines import llm_openai, registry
from argus.engines.llm_openai import DriverResponse, OpenAICompatDriver
from argus.engines.registry import EngineConfigError
from argus.engines.roles import ALL_SKIPPED_RAISES, DEGRADABLE_ROLES, ROLE_TIERS, SKIPPABLE_ROLES
from argus.engines.schema import EngineSpec, EnginesSnapshot, ProviderSpec, ResolvedBinding
from argus.graph.jsonx import parse_json_obj

from helpers import fake_engines_snapshot

PROVIDERS = {
    "p1": ProviderSpec(name="p1", base_url="http://p1/v1"),
    "p2": ProviderSpec(name="p2", base_url="http://p2/v1"),
    "pj": ProviderSpec(name="pj", base_url="http://pj/v1", supports_json_mode=True),
    "fake": ProviderSpec(name="fake", base_url="http://fake/v1"),
}
MSGS = [SystemMessage(content="sys"), HumanMessage(content="hi")]
_REQ = httpx.Request("POST", "http://p1/v1/chat/completions")


def _status(code: int, headers: dict | None = None) -> openai.APIStatusError:
    cls = {400: openai.BadRequestError, 429: openai.RateLimitError,
           503: openai.InternalServerError}[code]
    return cls(f"status {code}", response=httpx.Response(code, headers=headers or {}, request=_REQ),
               body=None)


def _resp(text: str, *, finish: str = "stop", served: str = "served-m",
          routed: str = "groq/served-m") -> DriverResponse:
    return DriverResponse(text=text, served_model=served, routed_via=routed, finish_reason=finish,
                          usage={"prompt_tokens": 3, "completion_tokens": 5, "reasoning_tokens": 1},
                          headers={"x-ratelimit-remaining-requests": "9"}, raw=None)


def _snap(role: str, primary: str = "p1", fallbacks: tuple[str, ...] = ("p2",), *,
          policy: str | None = None, rot: bool = True) -> EnginesSnapshot:
    tier = ROLE_TIERS.get(role) or role.removeprefix("legacy.")
    policy = policy or ("degrade" if role in DEGRADABLE_ROLES else
                        "skip_item" if role in SKIPPABLE_ROLES else "raise")
    b = ResolvedBinding(role=role, tier=tier, primary=EngineSpec(provider=primary, model=f"{primary}-m"),
                        fallbacks=[EngineSpec(provider=f, model=f"{f}-m") for f in fallbacks],
                        on_exhausted=policy, retry_on_truncation=rot)
    return EnginesSnapshot(profile="t", resolved_at="2026-10-19T00:00:00+00:00", roles={role: b})


class FakeDriver:
    """Scripted per provider: each call pops the next outcome (a
    DriverResponse, or an exception to raise)."""

    def __init__(self, **script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[SimpleNamespace] = []

    def complete(self, engine, provider, messages, *, temperature, max_tokens, json_mode):
        self.calls.append(SimpleNamespace(provider=provider.name, model=engine.model,
                                          messages=list(messages), temperature=temperature,
                                          max_tokens=max_tokens, json_mode=json_mode))
        outcome = self.script[provider.name].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.fixture
def sleeps(monkeypatch):
    seen: list[float] = []
    monkeypatch.setattr(llm, "_sleep", seen.append)
    return seen


@pytest.fixture(autouse=True)
def _providers(monkeypatch):
    monkeypatch.setattr(registry, "provider", PROVIDERS.__getitem__)
    monkeypatch.setattr(registry, "_state", registry._State())


def _drive(monkeypatch, **script) -> FakeDriver:
    d = FakeDriver(**script)
    monkeypatch.setattr(llm, "_driver", d)
    return d


# ---------------------------------------------------------------------------
# Review Focus 2 — untagged reasoning instead of JSON on a JSON role
# ---------------------------------------------------------------------------

def _queries(text: str) -> dict:
    return parse_json_obj(text, require_any=("queries",))


REASONING = "We need to scan the report… The user wants queries, so let us think step by step."


def test_untagged_reasoning_on_json_role_retries_then_falls_back(monkeypatch, sleeps):
    d = _drive(monkeypatch, p1=[_resp(REASONING), _resp(REASONING)],
               p2=[_resp('{"queries": [{"q": "x", "sub_q": 0}]}')])
    out = llm.complete("scout.querygen", MSGS, engines=_snap("scout.querygen"), validate=_queries)

    assert out.text == '{"queries": [{"q": "x", "sub_q": 0}]}', "the text comes from the fallback"
    assert out.parsed == {"queries": [{"q": "x", "sub_q": 0}]}, "parsed is the validator's value"
    first, second, last = out.records
    assert (first.ok, first.fallback_rank, first.retry_reason) == (False, 0, "")
    assert (second.ok, second.attempt, second.retry_reason) == (False, 2, "unparseable")
    assert (last.ok, last.fallback_rank) == (True, 1)
    assert out.record is last
    nudge = d.calls[1].messages[-1]
    assert isinstance(nudge, HumanMessage) and nudge.content == "Reply with only the JSON object."
    assert d.calls[2].messages == MSGS, "the fallback gets the original messages"
    assert sleeps == [], "an unparseable reply is not a transient error"


def test_untagged_reasoning_everywhere_degrades_never_anything_else(monkeypatch):
    _drive(monkeypatch, p1=[_resp(REASONING)] * 2, p2=[_resp(REASONING)] * 2)
    with pytest.raises(llm.EngineDegraded) as ei:
        llm.complete("scout.querygen", MSGS, engines=_snap("scout.querygen"), validate=_queries)
    e = ei.value
    assert type(e) is llm.EngineDegraded
    assert (e.role, e.degradation) == ("scout.querygen", "subquestion_text_queries")
    assert len(e.attempts) == 4 and not any(r.ok for r in e.attempts)
    assert all(r.error.startswith("unparseable") for r in e.attempts)


# ---------------------------------------------------------------------------
# fallbacks, policy classes, ALL_SKIPPED_RAISES
# ---------------------------------------------------------------------------

def test_fallback_walk_records_every_attempt(monkeypatch, sleeps):
    _drive(monkeypatch, p1=[_status(503)] * 3, p2=[_resp("ok")])
    out = llm.complete("brief.draft", MSGS, engines=_snap("brief.draft"))
    assert [(r.fallback_rank, r.attempt, r.ok, r.retry_reason) for r in out.records] == [
        (0, 1, False, ""), (0, 2, False, "transient"), (0, 3, False, "transient"), (1, 1, True, "")]
    assert out.records[0].error.startswith("InternalServerError")
    assert sleeps == [1.0, 2.0], "exponential backoff between transient tries"
    rec = out.record
    assert (rec.role, rec.tier, rec.profile, rec.provider, rec.requested_model) == (
        "brief.draft", "strong", "t", "p2", "p2-m")
    assert (rec.served_model, rec.routed_via, rec.served_provider) == ("served-m", "groq/served-m", "groq")
    assert (rec.prompt_tokens, rec.completion_tokens, rec.reasoning_tokens) == (3, 5, 1)
    assert rec.extra == {"headers": {"x-ratelimit-remaining-requests": "9"}}


@pytest.mark.parametrize("role, exc_type, degradation", [
    ("brief.draft", llm.EngineError, None),
    ("compose.section", llm.EngineSkipped, None),
    ("outline.plan", llm.EngineDegraded, "one_section_per_subquestion_roundrobin_notes"),
])
def test_policy_classes_map_to_exception_classes(monkeypatch, role, exc_type, degradation):
    _drive(monkeypatch, p1=[_status(400)], p2=[_status(400)])
    with pytest.raises(llm.EngineError) as ei:
        llm.complete(role, MSGS, engines=_snap(role))
    assert type(ei.value) is exc_type
    assert ei.value.role == role and len(ei.value.attempts) == 2, "a 400 is not retried"
    assert getattr(ei.value, "degradation", None) == degradation


def test_all_skipped_raises_rule_at_a_fake_per_item_call_site(monkeypatch):
    """The §2.3 rule as a per-item node applies it. The facade's own raise
    path hands the node one EngineSkipped per item (role, item and attempts
    recorded); the node raises EngineError, carrying those attempts, only
    when every item of an ALL_SKIPPED_RAISES role was skipped."""
    skipped: list[llm.EngineError] = []

    def section_node(titles, snap):
        bodies, errors = [], []
        for t in titles:
            try:
                bodies.append(llm.complete("compose.section", MSGS, engines=snap, item=t).text)
            except llm.EngineSkipped as e:
                skipped.append(e)
                errors.append(f"skip: compose.section {t} ({e.attempts[-1].error})")
        if "compose.section" in ALL_SKIPPED_RAISES and not bodies:
            raise llm.EngineError("compose: every section was skipped", role="compose.section",
                                  attempts=[a for e in skipped for a in e.attempts])
        return bodies, errors

    _drive(monkeypatch, p1=[_status(400)] * 2, p2=[_status(400)] * 2)
    with pytest.raises(llm.EngineError) as ei:
        section_node(["A", "B"], _snap("compose.section"))
    assert [type(e) for e in skipped] == [llm.EngineSkipped, llm.EngineSkipped], (
        "the facade raised EngineSkipped for each item, not a bare EngineError")
    assert all(e.role == "compose.section" and
               str(e).startswith("role compose.section: every engine failed after 2 attempts")
               for e in skipped)
    assert str(ei.value) == "compose: every section was skipped"
    assert [(a.item, a.provider, a.ok) for a in ei.value.attempts] == [
        ("A", "p1", False), ("A", "p2", False), ("B", "p1", False), ("B", "p2", False)]

    skipped.clear()
    _drive(monkeypatch, p1=[_status(400), _resp("body B")], p2=[_status(400)])
    bodies, errors = section_node(["A", "B"], _snap("compose.section"))
    assert bodies == ["body B"] and errors and errors[0].startswith("skip: compose.section A (")
    assert [type(e) for e in skipped] == [llm.EngineSkipped]


# ---------------------------------------------------------------------------
# strip_reasoning, truncation, validate/parsed, json_mode, roles
# ---------------------------------------------------------------------------

def test_strip_reasoning_removes_tagged_blocks_only():
    text = "<think>plan\nsteps</think>{\"a\": 1}<THINKING>x</THINKING> <reasoning>r</reasoning><analysis>z</analysis>"
    assert llm.strip_reasoning(text) == '{"a": 1}'
    assert llm.strip_reasoning("We need to scan the report") == "We need to scan the report"
    assert llm.strip_reasoning("<think>never closed") == "<think>never closed"
    assert llm.strip_reasoning("") == ""


def test_complete_strips_reasoning_before_validating(monkeypatch):
    _drive(monkeypatch, p1=[_resp('<think>hmm</think>{"queries": [{"q": "y"}]}')])
    out = llm.complete("scout.querygen", MSGS, engines=_snap("scout.querygen"), validate=_queries)
    assert out.text == '{"queries": [{"q": "y"}]}' and len(out.records) == 1


def test_truncation_retry_doubles_the_budget_once_with_a_cap(monkeypatch):
    d = _drive(monkeypatch, p1=[_resp('{"mode": "de', finish="length"), _resp('{"mode": "deep"}')])
    out = llm.complete("intake.classify", MSGS, engines=_snap("intake.classify"), max_tokens=3000)
    assert [c.max_tokens for c in d.calls] == [3000, 4096]
    assert out.records[1].retry_reason == "truncated" and out.text == '{"mode": "deep"}'

    d = _drive(monkeypatch, p1=[_resp('{"mode": "de', finish="length"), _resp('{"mode": "deep"}')])
    llm.complete("intake.classify", MSGS, engines=_snap("intake.classify"))      # no budget given
    assert [c.max_tokens for c in d.calls] == [None, 4096], "the retry uses the cap"

    d = _drive(monkeypatch, p1=[_resp("cut", finish="length")])        # not a JSON role
    llm.complete("legacy.cheap", MSGS, engines=_snap("legacy.cheap"), max_tokens=100)
    assert len(d.calls) == 1

    d = _drive(monkeypatch, p1=[_resp("cut", finish="length")])        # retry_on_truncation off
    llm.complete("intake.classify", MSGS, engines=_snap("intake.classify", rot=False), max_tokens=100)
    assert len(d.calls) == 1


def test_parsed_is_the_validator_value_else_not_validated(monkeypatch):
    _drive(monkeypatch, p1=[_resp("anything"), _resp("{}")])
    assert llm.complete("legacy.cheap", MSGS, engines=_snap("legacy.cheap")).parsed is llm.NOT_VALIDATED
    out = llm.complete("legacy.cheap", MSGS, engines=_snap("legacy.cheap"), validate=lambda t: {})
    assert out.parsed == {} and out.parsed is not llm.NOT_VALIDATED


def test_json_mode_default_follows_role_and_provider(monkeypatch):
    d = _drive(monkeypatch, pj=[_resp("{}")] * 3, p1=[_resp("{}")])
    llm.complete("intake.classify", MSGS, engines=_snap("intake.classify", "pj"))
    llm.complete("legacy.cheap", MSGS, engines=_snap("legacy.cheap", "pj"))
    llm.complete("intake.classify", MSGS, engines=_snap("intake.classify", "pj"), json_mode=False)
    llm.complete("intake.classify", MSGS, engines=_snap("intake.classify", "p1"))
    assert [c.json_mode for c in d.calls] == [True, False, False, False]


def test_temperature_and_budget_fall_back_to_the_engine_spec(monkeypatch):
    snap = _snap("legacy.cheap")
    snap.roles["legacy.cheap"].primary = EngineSpec(provider="p1", model="p1-m", temperature=0.7,
                                                    max_tokens=321)
    d = _drive(monkeypatch, p1=[_resp("a"), _resp("b")])
    llm.complete("legacy.cheap", MSGS, engines=snap)
    llm.complete("legacy.cheap", MSGS, engines=snap, temperature=0.1, max_tokens=9)
    assert [(c.temperature, c.max_tokens) for c in d.calls] == [(0.7, 321), (0.1, 9)]


def test_unknown_or_unbound_role_raises_value_error(monkeypatch):
    _drive(monkeypatch)
    with pytest.raises(ValueError, match="unknown role"):
        llm.complete("compose.tldr", MSGS, engines=_snap("brief.draft"))
    with pytest.raises(ValueError, match="not bound"):
        llm.complete("quick.answer", MSGS, engines=_snap("brief.draft"))


def test_a_config_error_at_call_time_fails_that_engine_visibly(monkeypatch):
    def provider(name):
        if name == "p1":
            raise EngineConfigError("provider p1: P1_KEY is not set")
        return PROVIDERS[name]
    monkeypatch.setattr(registry, "provider", provider)
    _drive(monkeypatch, p2=[_resp("ok")])
    out = llm.complete("brief.draft", MSGS, engines=_snap("brief.draft"))
    assert out.records[0].ok is False and "P1_KEY" in out.records[0].error
    assert out.record.provider == "p2"


# ---------------------------------------------------------------------------
# the openai-SDK driver on a mock transport
# ---------------------------------------------------------------------------

BODY = {"id": "x", "object": "chat.completion", "created": 1, "model": "openai/gpt-oss-120b",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "PONG"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12,
                  "completion_tokens_details": {"reasoning_tokens": 3}}}


def _transport(monkeypatch, *, headers=None, body=None):
    """Patch the driver's OpenAI constructor (looked up in llm_openai) so the
    real SDK sends to a mock transport; returns (constructor kwargs, requests)."""
    made, sent = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append({"json": json.loads(request.content), "headers": dict(request.headers),
                     "url": str(request.url)})
        return httpx.Response(200, headers=headers or {}, json=body or BODY)

    def fake_openai(**kw):
        made.append(kw)
        return openai.OpenAI(**kw, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(llm_openai, "OpenAI", fake_openai)
    return made, sent


def test_driver_client_has_no_sdk_retries_and_never_a_none_key(monkeypatch):
    made, _ = _transport(monkeypatch)
    monkeypatch.setenv("T_DRV_KEY", "sk-drv")
    keyless = ProviderSpec(name="local", base_url="http://local/v1", timeout_s=42.0)
    keyed = ProviderSpec(name="keyed", base_url="http://default/v1", base_url_env="T_DRV_URL",
                         api_key_env="T_DRV_KEY", headers={"X-Title": "Argus"})
    monkeypatch.setenv("T_DRV_URL", "http://from-env/v1")
    d = OpenAICompatDriver()
    eng = EngineSpec(provider="local", model="m")
    d.complete(eng, keyless, MSGS, temperature=None, max_tokens=None, json_mode=False)
    d.complete(eng, keyless, MSGS, temperature=None, max_tokens=None, json_mode=False)
    d.complete(EngineSpec(provider="keyed", model="m"), keyed, MSGS, temperature=None,
               max_tokens=None, json_mode=False)
    assert len(made) == 2, "one client per (provider, model)"
    assert all(kw["max_retries"] == 0 for kw in made)
    assert made[0]["api_key"] == "none" and made[0]["timeout"] == 42.0
    assert made[1]["api_key"] == "sk-drv" and made[1]["base_url"] == "http://from-env/v1"
    assert made[1]["default_headers"] == {"X-Title": "Argus"}
    assert all(kw["api_key"] is not None for kw in made)


def test_driver_refuses_a_missing_key_instead_of_sending_none(monkeypatch):
    _transport(monkeypatch)
    monkeypatch.delenv("T_ABSENT_KEY", raising=False)
    p = ProviderSpec(name="x", base_url="http://x/v1", api_key_env="T_ABSENT_KEY")
    with pytest.raises(EngineConfigError, match="T_ABSENT_KEY is not set"):
        OpenAICompatDriver().complete(EngineSpec(provider="x", model="m"), p, MSGS,
                                      temperature=None, max_tokens=None, json_mode=False)


def test_driver_request_shape_token_param_temperature_json_and_merged_extra_body(monkeypatch):
    _, sent = _transport(monkeypatch)
    d = OpenAICompatDriver()
    prov = ProviderSpec(name="o", base_url="http://o/v1", token_param="max_completion_tokens",
                        extra_body={"a": 1, "b": 1})
    eng = EngineSpec(provider="o", model="m1", extra_body={"b": 2, "thinking": {"type": "enabled"}})
    d.complete(eng, prov, MSGS, temperature=0.2, max_tokens=16, json_mode=True)
    j = sent[-1]["json"]
    assert j["max_completion_tokens"] == 16 and "max_tokens" not in j
    assert j["temperature"] == 0.2
    assert j["response_format"] == {"type": "json_object"}
    assert (j["a"], j["b"], j["thinking"]) == (1, 2, {"type": "enabled"}), "engine merged over provider"
    assert j["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

    plain = ProviderSpec(name="q", base_url="http://q/v1")             # default token_param
    d.complete(EngineSpec(provider="q", model="m"), plain, MSGS, temperature=0.2, max_tokens=8,
               json_mode=False)
    j = sent[-1]["json"]
    assert j["max_tokens"] == 8 and "max_completion_tokens" not in j
    assert "response_format" not in j and "a" not in j

    no_temp = EngineSpec(provider="q", model="m2", supports_temperature=False)   # per-spec override
    d.complete(no_temp, plain, MSGS, temperature=0.2, max_tokens=None, json_mode=False)
    assert "temperature" not in sent[-1]["json"] and "max_tokens" not in sent[-1]["json"]

    prov_off = ProviderSpec(name="a", base_url="http://a/v1", supports_temperature=False)
    d.complete(EngineSpec(provider="a", model="m"), prov_off, MSGS, temperature=0.2, max_tokens=None,
               json_mode=False)
    assert "temperature" not in sent[-1]["json"]
    d.complete(EngineSpec(provider="a", model="m3", supports_temperature=True), prov_off, MSGS,
               temperature=0.2, max_tokens=None, json_mode=False)
    assert sent[-1]["json"]["temperature"] == 0.2, "the engine spec overrides the provider default"


@pytest.mark.parametrize("headers, extra, routed", [
    ({"x-routed-via": "groq/openai/gpt-oss-120b"}, {}, "groq/openai/gpt-oss-120b"),
    ({}, {"_routed_via": {"platform": "groq", "model": "openai/gpt-oss-120b"}}, "groq/openai/gpt-oss-120b"),
    ({}, {"_routed_via": "cerebras/llama"}, "cerebras/llama"),
    ({}, {"x_groq": {"id": "req_1"}}, "groq"),
    ({}, {}, ""),
])
def test_driver_routed_via_from_headers_and_body(monkeypatch, headers, extra, routed):
    _transport(monkeypatch, headers={**headers, "x-ratelimit-limit-requests": "120"},
               body={**BODY, **extra})
    r = OpenAICompatDriver().complete(EngineSpec(provider="p", model="m"),
                                      ProviderSpec(name="p", base_url="http://p/v1"), MSGS,
                                      temperature=None, max_tokens=None, json_mode=False)
    assert r.routed_via == routed
    assert (r.text, r.served_model, r.finish_reason) == ("PONG", "openai/gpt-oss-120b", "stop")
    assert r.usage == {"prompt_tokens": 5, "completion_tokens": 7, "reasoning_tokens": 3}
    assert r.headers["x-ratelimit-limit-requests"] == "120"


# ---------------------------------------------------------------------------
# unusable response bodies are engine failures, never empty successes
# (the real driver and SDK on a mock transport, through the facade)
# ---------------------------------------------------------------------------

def _hosts(monkeypatch, **answers) -> list[str]:
    """Each request to http://<host>/v1 pops that host's next httpx.Response;
    returns the hosts asked, in order."""
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.host)
        return answers[request.url.host].pop(0)
    monkeypatch.setattr(llm_openai, "OpenAI", lambda **kw: openai.OpenAI(
        **kw, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    monkeypatch.setattr(llm, "_driver", OpenAICompatDriver())
    return sent


def _ok() -> httpx.Response:
    return httpx.Response(200, json=BODY)


def _error_body(message: str, code) -> httpx.Response:
    return httpx.Response(200, json={"error": {"message": message, "code": code}})


def test_a_429_error_body_is_transient_retried_then_falls_back(monkeypatch, sleeps):
    sent = _hosts(monkeypatch, p1=[_error_body("All models exhausted", 429)] * 3, p2=[_ok()])
    out = llm.complete("brief.draft", MSGS, engines=_snap("brief.draft"))
    assert out.text == "PONG" and out.record.provider == "p2"
    assert sent == ["p1", "p1", "p1", "p2"] and sleeps == [1.0, 2.0]
    failed = out.records[:3]
    assert [r.retry_reason for r in failed] == ["", "transient", "transient"]
    assert all(not r.ok and r.error.startswith("DriverError") and "All models exhausted" in r.error
               for r in failed)


@pytest.mark.parametrize("answer, reason", [
    (_error_body("model not found", 404), "model not found"),
    (_error_body("slow down", "rate_limit_exceeded"), "slow down"),     # a symbolic code is no status
    (httpx.Response(200, json={**BODY, "choices": []}), "no choices"),
    (httpx.Response(200, json={k: v for k, v in BODY.items() if k != "choices"}), "no choices"),
    (httpx.Response(200, text="<html>bad gateway</html>", headers={"content-type": "text/html"}),
     "not a chat completion"),
    (httpx.Response(200, content=b"{not json", headers={"content-type": "application/json"}), "not JSON"),
])
def test_an_unusable_body_fails_the_engine_and_goes_to_the_fallback(monkeypatch, sleeps, answer, reason):
    sent = _hosts(monkeypatch, p1=[answer], p2=[_ok()])
    out = llm.complete("brief.draft", MSGS, engines=_snap("brief.draft"))
    assert sent == ["p1", "p2"] and sleeps == [], "not transient: straight to the fallback"
    first = out.records[0]
    assert first.ok is False and first.error.startswith("DriverError") and reason in first.error
    assert out.text == "PONG" and out.record.ok


def test_error_bodies_everywhere_raise_by_policy_never_an_empty_success(monkeypatch, sleeps):
    registry._state.snapshot = fake_engines_snapshot()
    _hosts(monkeypatch, fake=[_error_body("All models exhausted", 429)] * 6)
    with pytest.raises(llm.EngineError) as ei:
        llm.invoke_with_retry(llm.chat_for_tier("cheap"), MSGS)
    assert type(ei.value) is llm.EngineError and len(ei.value.attempts) == 3
    assert str(ei.value).startswith("role legacy.cheap: every engine failed after 3 attempts")
    with pytest.raises(llm.EngineDegraded) as ei:
        llm.complete("scout.querygen", MSGS, engines=registry._state.snapshot, validate=json.loads)
    assert all(not a.ok and "All models exhausted" in a.error for a in ei.value.attempts)


# Shapes the SDK lets through unvalidated (re-review round 3): each must be a
# failed attempt naming the shape, never an AttributeError or a TypeError.
def _bad(mutate) -> httpx.Response:
    body = json.loads(json.dumps(BODY))
    mutate(body)
    return httpx.Response(200, json=body)


TYPE_SHAPES = {
    "choices-str": (lambda b: b.update(choices="x"), "unexpected choices type str"),
    "choices-dict": (lambda b: b.update(choices={"a": 1}), "unexpected choices type dict"),
    "choices-number": (lambda b: b.update(choices=5), "unexpected choices type int"),
    "choice-str": (lambda b: b.update(choices=["x"]), "unexpected choice type str"),
    "message-missing": (lambda b: b["choices"][0].pop("message"), "the response has no message"),
    "message-str": (lambda b: b["choices"][0].update(message="hi"), "unexpected message type str"),
    "content-list": (lambda b: b["choices"][0]["message"].update(content=["a", "b"]), "unexpected content type list"),
    "content-dict": (lambda b: b["choices"][0]["message"].update(content={"x": 1}), "unexpected content type dict"),
    "content-number": (lambda b: b["choices"][0]["message"].update(content=5), "unexpected content type int"),
    "model-number": (lambda b: b.update(model=5), "unexpected model type int"),
    "finish-number": (lambda b: b["choices"][0].update(finish_reason=7), "unexpected finish_reason type int"),
    "usage-str": (lambda b: b.update(usage="x"), "unexpected usage type str"),
    "tokens-str": (lambda b: b["usage"].update(prompt_tokens="five"), "unexpected prompt_tokens type str"),  # "5" the SDK coerces
}


@pytest.mark.parametrize("shape", list(TYPE_SHAPES))
@pytest.mark.parametrize("role, exc_type", [("brief.draft", llm.EngineError),
                                            ("compose.section", llm.EngineSkipped),
                                            ("outline.plan", llm.EngineDegraded)])
def test_a_mistyped_answer_fails_the_engine_and_raises_only_by_policy(monkeypatch, sleeps, shape, role, exc_type):
    mutate, named = TYPE_SHAPES[shape]
    sent = _hosts(monkeypatch, p1=[_bad(mutate)], p2=[_bad(mutate)])
    with pytest.raises(llm.EngineError) as ei:             # an AttributeError / TypeError would escape this
        llm.complete(role, MSGS, engines=_snap(role))
    assert type(ei.value) is exc_type
    assert sent == ["p1", "p2"] and sleeps == [], "not transient: each engine once, then on_exhausted"
    assert all(not a.ok and a.error.startswith("DriverError") and named in a.error for a in ei.value.attempts)


def test_a_mistyped_answer_falls_back_to_an_engine_that_answers_properly(monkeypatch, sleeps):
    seen: list[type] = []
    real = llm.strip_reasoning
    monkeypatch.setattr(llm, "strip_reasoning", lambda text: seen.append(type(text)) or real(text))
    sent = _hosts(monkeypatch, p1=[_bad(TYPE_SHAPES["content-list"][0])], p2=[_ok()])
    out = llm.complete("brief.draft", MSGS, engines=_snap("brief.draft"))
    assert out.text == "PONG" and out.record.provider == "p2" and sent == ["p1", "p2"]
    assert "unexpected content type list" in out.records[0].error
    assert seen == [str], "strip_reasoning saw the fallback's text only"


def test_strip_reasoning_only_ever_sees_a_str_whatever_the_driver_returns(monkeypatch):
    """The facade refuses non-str text from any driver (Driver is a protocol)."""
    seen: list[type] = []
    real = llm.strip_reasoning
    monkeypatch.setattr(llm, "strip_reasoning", lambda text: seen.append(type(text)) or real(text))
    _drive(monkeypatch, p1=[_resp(["not", "text"])], p2=[_resp("ok")])
    out = llm.complete("brief.draft", MSGS, engines=_snap("brief.draft"))
    assert out.text == "ok" and seen == [str]
    assert out.records[0].error == "DriverError: provider p1: unexpected content type list"


def test_null_content_cut_off_at_the_limit_still_gets_the_truncation_retry(monkeypatch):
    def answer(content, finish):
        return httpx.Response(200, json={**BODY, "choices": [
            {"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content}}]})
    answers, budgets = [answer(None, "length"), answer('{"mode": "deep"}', "stop")], []

    def handler(request: httpx.Request) -> httpx.Response:
        budgets.append(json.loads(request.content)["max_tokens"])
        return answers.pop(0)
    monkeypatch.setattr(llm_openai, "OpenAI", lambda **kw: openai.OpenAI(
        **kw, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    monkeypatch.setattr(llm, "_driver", OpenAICompatDriver())
    out = llm.complete("intake.classify", MSGS, engines=_snap("intake.classify"), max_tokens=100)
    assert out.text == '{"mode": "deep"}' and budgets == [100, 200]
    assert out.records[0].error == "truncated (finish_reason=length)"


def test_a_key_in_a_provider_error_is_never_recorded_or_logged(monkeypatch, sleeps, caplog):
    caplog.set_level(logging.WARNING, logger="argus.llm")
    monkeypatch.setenv("T_PK_KEY", "sk-leaky-123")
    keyed = ProviderSpec(name="pk", base_url="http://pk/v1", api_key_env="T_PK_KEY")
    monkeypatch.setattr(registry, "provider", {**PROVIDERS, "pk": keyed}.__getitem__)
    _hosts(monkeypatch, pk=[_error_body("invalid key sk-leaky-123", 401),
                            httpx.Response(401, json={"error": {"message": "bad key sk-leaky-123"}})],
           p2=[_ok(), _ok()])
    for _ in range(2):                    # the 200 error body, then a real 401 status
        out = llm.complete("brief.draft", MSGS, engines=_snap("brief.draft", "pk"))
        assert "sk-leaky-123" not in out.records[0].error and "***" in out.records[0].error
    assert caplog.records and not any("sk-leaky-123" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# the Phase 1–2 legacy wrappers
# ---------------------------------------------------------------------------

def test_legacy_wrappers_route_through_complete_on_the_cached_snapshot(monkeypatch, sleeps):
    with pytest.raises(EngineConfigError):
        llm.resolve_tier("cheap")                         # nothing resolved yet
    registry._state.snapshot = fake_engines_snapshot()
    d = _drive(monkeypatch, fake=[_status(503), _resp("hello")])

    chat = llm.chat_for_tier("strong", temperature=0.3, max_tokens=50)
    resp = llm.invoke_with_retry(chat, MSGS)
    assert resp.content == "hello"
    assert (d.calls[-1].model, d.calls[-1].temperature, d.calls[-1].max_tokens) == ("fake-model", 0.3, 50)

    rec = llm.record_from_response("strong", llm.resolve_tier("strong"), resp)
    dumped = rec.model_dump()
    assert dumped["role"] == "legacy.strong" and dumped["routed_via"] == "groq/served-m"
    assert dumped["attempt"] == 2 and dumped["retry_reason"] == "transient"
    prior = dumped["extra"]["prior_attempts"]
    assert len(prior) == 1 and prior[0]["ok"] is False, "failed attempts ride along"

    assert llm.pick_strong_and_judge() == ("fake-model", "fake-model")
    with pytest.raises(ValueError, match="model_override"):
        llm.invoke_with_retry(llm.chat_for_tier("judge", model_override="other-model"), MSGS)
    _drive(monkeypatch, fake=[_resp("judged")])
    ok = llm.invoke_with_retry(llm.chat_for_tier("judge", model_override="fake-model"), MSGS)
    assert ok.content == "judged"


DIGEST_OK = '{"relevance": 3, "stance": "supports", "claims": []}'


def test_exhausted_digests_count_as_calls_in_the_report_metadata(monkeypatch, tmp_path, sleeps):
    """Review risk 9: two digests exhaust their engine (429 three times
    each) and one succeeds after a 429, through the real call site and the
    facade. Since WP-B the call site is research.digest on llm.complete: it
    appends every attempt record and writes the skip line quoting the
    facade's message, so the summary report_builder writes counts three
    logical calls, two of them exhausted, and every request exactly once."""
    from argus import config as config_mod
    from argus.graph import nodes as nodes_mod
    from argus.graph import research
    from argus.graph.run_config import RunConfig
    from argus.graph.state import ResearchBrief, SubQuestion

    _drive(monkeypatch, fake=[_status(429)] * 7 + [_resp(DIGEST_OK)])
    rc = RunConfig.from_preset(length="medium", engines=fake_engines_snapshot(), origin="telegram")
    doc = tmp_path / "doc.md"
    doc.write_text("Retrieval augmented generation. " * 40, encoding="utf-8")
    brief = ResearchBrief(sub_questions=[SubQuestion(q="what is RAG?")], summary="s")
    model_calls, errors = [], []
    for i in range(3):
        item = {"url": f"https://{i}.example", "title": "t", "markdown_path": str(doc), "sub_qs": [0]}
        _note, calls, errs = research.digest_one(item, i + 1, brief, "q", rc=rc, rnd=1)
        model_calls += calls
        errors += errs
    assert len(model_calls) == 8 and len(errors) == 2, "3 + 3 attempts exhausted, 1 failed + 1 ok"
    assert all(e.startswith("[r1] skip: research.digest https://") and
               "role research.digest: every engine failed after 3 attempts" in e for e in errors)
    assert sleeps == [1.0, 2.0, 1.0, 2.0, 1.0]

    monkeypatch.setattr(config_mod, "get_settings", lambda: SimpleNamespace(reports_root=tmp_path))
    monkeypatch.setattr(nodes_mod, "markdown_to_pdf", lambda *a, **k: None)
    out = nodes_mod.report_builder_node({"user_request": "risk nine", "length": "short",
                                         "draft_md": "# risk nine\n\nBody.\n",
                                         "model_calls": model_calls, "errors": errors})
    meta = json.loads((Path(out["report_paths"]["folder"]) / "metadata.json").read_text(encoding="utf-8"))
    row = meta["engines"]["by_role"]["research.digest"]
    assert (row["calls"], row["ok"], row["exhausted"], row["skipped"]) == (3, 1, 2, 2)
    assert (row["attempts"], row["failed_attempts"], row["exhausted_attempts"]) == (8, 7, 0), (
        "every request is one attempt record; a real role's exhausted attempts are never re-counted from the text")
    assert row["retries"] == {"transient": 5}
    assert (meta["engines"]["calls"], meta["engines"]["exhausted"], meta["engines"]["attempts"]) == (3, 2, 8)
