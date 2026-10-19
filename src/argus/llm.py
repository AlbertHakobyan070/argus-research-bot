"""LLM facade (v4 spec §2.3): ``complete()``, ``CallRecord``, ``strip_reasoning()``.

Every call names a role; its binding (primary, fallbacks, failure policy)
comes from an ``EnginesSnapshot``. On each engine: connection errors, 5xx
and 429 get up to three tries with backoff, honouring ``Retry-After`` up to
the provider's ``retry_after_cap_s``; a timeout is retried at most
``timeout_retry_max`` times; any other error fails that engine (the SDK
itself never retries). Then tagged reasoning blocks are stripped; a JSON
role cut off at the token limit gets one retry with double the budget; a
reply the ``validate`` hook rejects gets one retry with "Reply with only the
JSON object.". Then the next fallback, then the binding's ``on_exhausted``:
``raise`` → ``EngineError``, ``skip_item`` → ``EngineSkipped``, ``degrade`` →
``EngineDegraded``. Every attempt is a ``CallRecord``.

The wrappers at the bottom keep the Phase 1 nodes unchanged: they route
through ``complete`` with role ``legacy.<tier>`` on the registry's cached
snapshot. WP-H deletes them.
"""
from __future__ import annotations

import dataclasses
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import openai
from langchain_core.messages import BaseMessage, HumanMessage

from .engines import registry
from .engines.llm_openai import OpenAICompatDriver
from .engines.registry import EngineConfigError  # noqa: F401  (re-export: llm.EngineConfigError)
from .engines.roles import DEGRADABLE_ROLES, JSON_ROLES, LEGACY_ROLES, ROLES
from .engines.schema import EngineSpec, EnginesSnapshot, ProviderSpec, ResolvedBinding

logger = logging.getLogger("argus.llm")


@dataclass
class CallRecord:
    """One attempt's provenance; ``model_dump()`` is the flat dict appended
    to ``state["model_calls"]``."""
    role: str
    tier: str
    profile: str
    provider: str
    requested_model: str
    served_model: str
    routed_via: str                 # header x-routed-via, else body _routed_via / x_groq, else ""
    served_provider: str = ""       # v3-compatible: the provider segment of routed_via
    attempt: int = 1
    fallback_rank: int = 0
    ok: bool = True
    error: str = ""
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str = ""
    json_mode: bool = False
    retry_reason: str = ""          # "transient" | "timeout" | "retry_after" | "truncated" | "unparseable" | "diversity_rerun"
    item: str = ""
    cached: bool = False
    extra: dict = field(default_factory=dict)

    def model_dump(self) -> dict:
        return dataclasses.asdict(self)


NOT_VALIDATED = object()            # LLMResult.parsed when no validate() was given


@dataclass
class LLMResult:
    text: str
    parsed: Any
    record: CallRecord              # the attempt that succeeded
    records: list[CallRecord]       # every attempt, failed fallbacks included


class EngineError(RuntimeError):
    """Every engine of a role failed (``on_exhausted == "raise"``)."""

    def __init__(self, message: str, *, role: str, attempts: list[CallRecord]):
        super().__init__(message)
        self.role = role
        self.attempts = attempts


class EngineSkipped(EngineError):
    """``on_exhausted == "skip_item"``: caught only by per-item call sites."""


class EngineDegraded(EngineError):
    """``on_exhausted == "degrade"``: caught only by the owning node, which
    applies exactly ``degradation``."""

    def __init__(self, message: str, *, role: str, attempts: list[CallRecord], degradation: str):
        super().__init__(message, role=role, attempts=attempts)
        self.degradation = degradation


_REASONING_BLOCK = re.compile(r"<(think|thinking|reasoning|analysis)>.*?</\1>", re.IGNORECASE | re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove tagged ``<think>``/``<thinking>``/``<reasoning>``/``<analysis>``
    blocks, nothing else (untagged reasoning is the validate hook's job)."""
    return _REASONING_BLOCK.sub("", text or "").strip()


_TRANSIENT_TRIES = 3
_TRUNCATION_CAP = 4096
_NUDGE = "Reply with only the JSON object."
_driver = OpenAICompatDriver()
_sleep = time.sleep


class _RateLimiter:
    """``ProviderSpec.rpm_limit``: at most that many requests per provider
    in any 60 s window; thread-safe (nodes call from pools)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sent: dict[str, deque] = {}

    def wait(self, provider: ProviderSpec) -> None:
        if not provider.rpm_limit:
            return
        while True:
            with self._lock:
                sent = self._sent.setdefault(provider.name, deque())
                now = time.monotonic()
                while sent and now - sent[0] >= 60.0:
                    sent.popleft()
                if len(sent) < provider.rpm_limit:
                    sent.append(now)
                    return
                delay = 60.0 - (now - sent[0])
            _sleep(delay)


_limiter = _RateLimiter()


def complete(role: str, messages: list[BaseMessage], *, engines: EnginesSnapshot,
             temperature: float | None = None, max_tokens: int | None = None,
             json_mode: bool | None = None, expect_json: bool | None = None,
             validate: Callable[[str], Any] | None = None, item: str = "") -> LLMResult:
    if role not in ROLES and role not in LEGACY_ROLES:
        raise ValueError(f"unknown role {role!r}")
    binding = engines.roles.get(role)
    if binding is None:
        raise ValueError(f"role {role} is not bound in the engines snapshot (profile {engines.profile})")
    if expect_json is None:
        expect_json = role in JSON_ROLES
    records: list[CallRecord] = []
    for rank, engine in enumerate([binding.primary, *binding.fallbacks]):
        result = _on_engine(role, binding, engine, rank, messages, engines.profile, records,
                            temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                            expect_json=expect_json, validate=validate, item=item)
        if result is not None:
            return result
    msg = f"role {role}: every engine failed (last: {records[-1].error})"
    logger.warning("llm %s", msg)
    if binding.on_exhausted == "skip_item":
        raise EngineSkipped(msg, role=role, attempts=records)
    if binding.on_exhausted == "degrade":
        raise EngineDegraded(msg, role=role, attempts=records, degradation=DEGRADABLE_ROLES[role])
    raise EngineError(msg, role=role, attempts=records)


def _on_engine(role: str, binding: ResolvedBinding, engine: EngineSpec, rank: int,
               messages: list[BaseMessage], profile: str, records: list[CallRecord], *,
               temperature: float | None, max_tokens: int | None, json_mode: bool | None,
               expect_json: bool, validate: Callable[[str], Any] | None, item: str) -> LLMResult | None:
    """Every attempt on one engine: its LLMResult, or None when it failed."""
    def record(**kw) -> CallRecord:
        rec = CallRecord(**{"role": role, "tier": binding.tier, "profile": profile,
                            "provider": engine.provider, "requested_model": engine.model or "",
                            "served_model": "", "routed_via": "", "fallback_rank": rank,
                            "item": item, **kw})
        records.append(rec)
        return rec

    try:
        provider = registry.provider(engine.provider)
    except EngineConfigError as e:
        record(ok=False, error=str(e))
        return None
    if json_mode is None:
        json_mode = (engine.json_mode if engine.json_mode is not None
                     else role in JSON_ROLES and provider.supports_json_mode)
    temperature = temperature if temperature is not None else engine.temperature
    budget = max_tokens if max_tokens is not None else engine.max_tokens
    msgs = list(messages)
    attempt = transient = timeouts = 0
    reason = ""
    truncated = nudged = False
    while True:
        attempt += 1
        _limiter.wait(provider)
        t0 = time.monotonic()
        try:
            resp = _driver.complete(engine, provider, msgs, temperature=temperature,
                                    max_tokens=budget, json_mode=json_mode)
        except (openai.APIError, EngineConfigError) as e:
            record(attempt=attempt, ok=False, error=f"{type(e).__name__}: {e}"[:300],
                   latency_ms=_ms(t0), json_mode=json_mode, retry_reason=reason)
            logger.warning("llm %s %s/%s attempt %d failed: %s: %s", role, engine.provider,
                           engine.model, attempt, type(e).__name__, e)
            if isinstance(e, openai.APITimeoutError):
                if timeouts < provider.timeout_retry_max:
                    timeouts += 1
                    reason = "timeout"
                    continue
                return None
            if _is_transient(e) and transient < _TRANSIENT_TRIES - 1:
                transient += 1
                wait = _retry_after(e, provider)
                reason = "transient" if wait is None else "retry_after"
                _sleep(min(8.0, 2.0 ** (transient - 1)) if wait is None else wait)
                continue
            return None
        rec = record(attempt=attempt, latency_ms=_ms(t0), json_mode=json_mode, retry_reason=reason,
                     served_model=resp.served_model, routed_via=resp.routed_via,
                     served_provider=resp.routed_via.split("/", 1)[0] if resp.routed_via else "",
                     prompt_tokens=int(resp.usage.get("prompt_tokens") or 0),
                     completion_tokens=int(resp.usage.get("completion_tokens") or 0),
                     reasoning_tokens=int(resp.usage.get("reasoning_tokens") or 0),
                     finish_reason=resp.finish_reason,
                     extra={"headers": dict(resp.headers)} if resp.headers else {})
        text = strip_reasoning(resp.text)
        if (expect_json and resp.finish_reason == "length" and binding.retry_on_truncation
                and not truncated and budget and budget < _TRUNCATION_CAP):
            rec.ok, rec.error = False, "truncated (finish_reason=length)"
            truncated, budget, reason = True, min(budget * 2, _TRUNCATION_CAP), "truncated"
            continue
        if validate is None:
            return LLMResult(text=text, parsed=NOT_VALIDATED, record=rec, records=list(records))
        try:
            parsed = validate(text)
        except Exception as e:   # the hook rejecting a reply is an outcome: recorded, retried, then on_exhausted
            rec.ok, rec.error = False, f"unparseable: {type(e).__name__}: {e}"[:300]
            if nudged:
                return None
            nudged, reason = True, "unparseable"
            msgs = [*messages, HumanMessage(content=_NUDGE)]
            continue
        return LLMResult(text=text, parsed=parsed, record=rec, records=list(records))


def _is_transient(e: Exception) -> bool:
    if isinstance(e, openai.APIConnectionError):
        return True
    return isinstance(e, openai.APIStatusError) and (e.status_code == 429 or e.status_code >= 500)


def _retry_after(e: Exception, provider: ProviderSpec) -> float | None:
    response = getattr(e, "response", None)
    value = response.headers.get("retry-after") if response is not None else None
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError):
            return None
    return max(0.0, min(seconds, provider.retry_after_cap_s))


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


# ---------------------------------------------------------------------------
# Phase 1–2 legacy wrappers (WP-H deletes them): the v3 call sites stay
# unchanged and route through complete(role="legacy.<tier>").
# ---------------------------------------------------------------------------

@dataclass
class _LegacyChat:
    tier: str
    temperature: float | None
    max_tokens: int | None
    model_override: str | None


@dataclass
class _LegacyResponse:
    content: str
    result: LLMResult


def chat_for_tier(tier: str, *, temperature: float | None = 0.2, max_tokens: int | None = None,
                  model_override: str | None = None) -> _LegacyChat:
    return _LegacyChat(tier, temperature, max_tokens, model_override)


def invoke_with_retry(chat: _LegacyChat, messages: list[BaseMessage]) -> _LegacyResponse:
    engines = registry.current_snapshot_cached()
    role = f"legacy.{chat.tier}"
    bound = engines.roles[role].primary.model
    if chat.model_override is not None and chat.model_override != bound:
        raise ValueError(f"{role}: model_override {chat.model_override!r} differs from the bound "
                         f"model {bound!r}")
    result = complete(role, messages, engines=engines, temperature=chat.temperature,
                      max_tokens=chat.max_tokens)
    return _LegacyResponse(content=result.text, result=result)


def record_from_response(tier: str, requested: str, response: _LegacyResponse) -> CallRecord:
    """The successful attempt's record; earlier failed attempts ride along
    in ``extra["prior_attempts"]`` (the call sites append one dict)."""
    result = response.result
    prior = [a.model_dump() for a in result.records[:-1]]
    if not prior:
        return result.record
    return dataclasses.replace(result.record, extra={**result.record.extra, "prior_attempts": prior})


def resolve_tier(tier: str) -> str:
    return registry.current_snapshot_cached().roles[f"legacy.{tier}"].primary.model


def pick_strong_and_judge() -> tuple[str, str]:
    return resolve_tier("strong"), resolve_tier("judge")
