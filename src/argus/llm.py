"""LLM facade (v4 spec §2.3): ``complete()``, ``CallRecord``, ``strip_reasoning()``.

Every call names a role; its binding (primary, fallbacks, failure policy)
comes from an ``EnginesSnapshot``. On each engine: connection errors, 5xx
and 429 (also as the code of an error object in a 200 body) get up to three
tries with backoff, honouring ``Retry-After`` up to the provider's
``retry_after_cap_s``; a 429 then keeps waiting (≤ 30 s a step) while the
provider's ``quota_wait_s`` lasts; a timeout is retried at most ``timeout_retry_max``
times; any other error fails that engine (the SDK itself never retries).
Then tagged reasoning blocks are stripped; a JSON role cut off at the token
limit gets one retry with double the budget (4096 when it had none); a
reply the ``validate`` hook rejects gets one retry with "Reply with only the
JSON object." (a JSON role) or "Reply with only the complete markdown text."
(a markdown role). Then the next fallback, then the binding's ``on_exhausted``:
``raise`` → ``EngineError``, ``skip_item`` → ``EngineSkipped``, ``degrade`` →
``EngineDegraded``, each with the message "role <id>: every engine failed
after <n> attempts (last: …)" — unless every engine ended on a 429: then
``QuotaExhausted`` whatever the policy (no node skips past it; the run
pauses), and each such provider is latched out for ``_QUOTA_LATCH_S``, so
parallel calls to it fail at once instead of each waiting. Every attempt is
a ``CallRecord``; recorded and logged errors never contain the provider's key.
"""
from __future__ import annotations

import dataclasses
import logging
import os
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
from .engines.llm_openai import DriverError, OpenAICompatDriver
from .engines.registry import EngineConfigError  # noqa: F401  (re-export: llm.EngineConfigError)
from .engines.roles import DEGRADABLE_ROLES, JSON_ROLES, ROLES
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
    retry_reason: str = ""          # "transient" | "timeout" | "retry_after" | "quota_wait" | "truncated" | "unparseable" | "diversity_rerun"
    item: str = ""
    started_at: str = ""            # UTC ISO time the attempt was sent
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


class QuotaExhausted(EngineError):
    """Every engine of a role ended on a 429 (an exhausted quota): caught
    only by the run driver, which pauses the run at the node."""


_REASONING_BLOCK = re.compile(r"<(think|thinking|reasoning|analysis)>.*?</\1>", re.IGNORECASE | re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove tagged ``<think>``/``<thinking>``/``<reasoning>``/``<analysis>``
    blocks, nothing else (untagged reasoning is the validate hook's job)."""
    return _REASONING_BLOCK.sub("", text or "").strip()


_TRANSIENT_TRIES = 3
_QUOTA_STEP_S = 30.0                # one wait while a provider's quota_wait_s lasts
_QUOTA_LATCH_S = 60.0               # a provider found out of quota fails at once this long
_quota_out: dict[str, float] = {}   # provider name -> time.monotonic() its latch ends
_TRUNCATION_CAP = 4096
_NUDGE = "Reply with only the JSON object."
_MD_NUDGE = "Reply with only the complete markdown text."      # a markdown role's validate retry
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
    """Whatever a provider answers, this raises only ``EngineError`` (or its
    subclasses ``EngineSkipped`` / ``EngineDegraded``, by policy), never a
    provider's own exception; ``ValueError`` only for an unknown or unbound
    role. (A ``validate`` hook's exception is a failed attempt, not raised.)"""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    binding = engines.roles.get(role)
    if binding is None:
        raise ValueError(f"role {role} is not bound in the engines snapshot (profile {engines.profile})")
    if expect_json is None:
        expect_json = role in JSON_ROLES
    records: list[CallRecord] = []
    out_of_quota: list[int] = []
    chain = [binding.primary, *binding.fallbacks]
    for rank, engine in enumerate(chain):
        result = _on_engine(role, binding, engine, rank, messages, engines.profile, records, out_of_quota,
                            temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                            expect_json=expect_json, validate=validate, item=item)
        if result is not None:
            return result
    n = len(records)
    attempts = f"{n} attempt{'s' if n != 1 else ''} (last: {records[-1].error})"
    if len(out_of_quota) == len(chain):
        msg = f"role {role}: every engine is out of quota after {attempts}"
        logger.warning("llm %s", msg)
        raise QuotaExhausted(msg, role=role, attempts=records)
    msg = f"role {role}: every engine failed after {attempts}"
    logger.warning("llm %s", msg)
    if binding.on_exhausted == "skip_item":
        raise EngineSkipped(msg, role=role, attempts=records)
    if binding.on_exhausted == "degrade":
        raise EngineDegraded(msg, role=role, attempts=records, degradation=DEGRADABLE_ROLES[role])
    raise EngineError(msg, role=role, attempts=records)


def _on_engine(role: str, binding: ResolvedBinding, engine: EngineSpec, rank: int,
               messages: list[BaseMessage], profile: str, records: list[CallRecord], out_of_quota: list[int], *,
               temperature: float | None, max_tokens: int | None, json_mode: bool | None,
               expect_json: bool, validate: Callable[[str], Any] | None, item: str) -> LLMResult | None:
    """Every attempt on one engine: its LLMResult, or None when it failed
    (``rank`` appended to ``out_of_quota`` when it ended on a 429)."""
    started = [_now()]

    def record(**kw) -> CallRecord:
        rec = CallRecord(**{"role": role, "tier": binding.tier, "profile": profile,
                            "provider": engine.provider, "requested_model": engine.model or "",
                            "served_model": "", "routed_via": "", "fallback_rank": rank,
                            "item": item, "started_at": started[0], **kw})
        records.append(rec)
        return rec

    try:
        provider = registry.provider(engine.provider)
    except EngineConfigError as e:
        record(ok=False, error=str(e))
        return None
    latched = _quota_out.get(provider.name, 0.0) - time.monotonic()
    if latched > 0:
        record(ok=False, error=f"provider {provider.name} is out of quota (latched for {latched:.0f} s more)")
        out_of_quota.append(rank)
        return None
    if json_mode is None:
        json_mode = (engine.json_mode if engine.json_mode is not None
                     else role in JSON_ROLES and provider.supports_json_mode)
    temperature = temperature if temperature is not None else engine.temperature
    budget = max_tokens if max_tokens is not None else engine.max_tokens
    if budget is not None and provider.min_max_tokens:
        budget = max(budget, provider.min_max_tokens)
    msgs = list(messages)
    attempt = transient = timeouts = 0
    quota_waited = 0.0
    reason = ""
    truncated = nudged = False
    while True:
        attempt += 1
        _limiter.wait(provider)
        started[0] = _now()
        t0 = time.monotonic()
        try:
            resp = _driver.complete(engine, provider, msgs, temperature=temperature,
                                    max_tokens=budget, json_mode=json_mode)
            if not isinstance(resp.text, str):     # any driver: strip_reasoning and validate only see text
                raise DriverError(f"provider {provider.name}: unexpected content type {type(resp.text).__name__}")
        except (openai.APIError, DriverError, EngineConfigError) as e:
            error = _redact(f"{type(e).__name__}: {e}", provider)
            record(attempt=attempt, ok=False, error=error[:300],
                   latency_ms=_ms(t0), json_mode=json_mode, retry_reason=reason)
            logger.warning("llm %s %s/%s attempt %d failed: %s", role, engine.provider,
                           engine.model, attempt, error)
            if isinstance(e, openai.APITimeoutError):
                if timeouts < provider.timeout_retry_max:
                    timeouts += 1
                    reason = "timeout"
                    continue
                return None
            if not _is_transient(e):
                return None
            wait = _retry_after(e, provider)
            if transient < _TRANSIENT_TRIES - 1:
                transient += 1
                reason = "transient" if wait is None else "retry_after"
                _sleep(min(8.0, 2.0 ** (transient - 1)) if wait is None else wait)
                continue
            if _code(e) == 429 and quota_waited < provider.quota_wait_s:   # an exhausted quota refills by waiting
                step = min(_QUOTA_STEP_S if wait is None else wait, provider.quota_wait_s - quota_waited)
                quota_waited += step
                reason = "quota_wait"
                _sleep(step)
                continue
            if _code(e) == 429:
                _quota_out[provider.name] = time.monotonic() + _QUOTA_LATCH_S
                out_of_quota.append(rank)
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
                and not truncated and (budget is None or budget < _TRUNCATION_CAP)):
            rec.ok, rec.error = False, "truncated (finish_reason=length)"
            truncated, reason = True, "truncated"
            budget = _TRUNCATION_CAP if budget is None else min(budget * 2, _TRUNCATION_CAP)
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
            msgs = [*messages, HumanMessage(content=_NUDGE if expect_json else _MD_NUDGE)]
            continue
        return LLMResult(text=text, parsed=parsed, record=rec, records=list(records))


def _code(e: Exception) -> int | None:
    code = e.code if isinstance(e, DriverError) else getattr(e, "status_code", None)
    return code if isinstance(code, int) else None


def _is_transient(e: Exception) -> bool:
    if isinstance(e, openai.APIConnectionError):
        return True
    code = _code(e)
    return code is not None and (code == 429 or code >= 500)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _redact(text: str, provider: ProviderSpec) -> str:
    key = os.environ.get(provider.api_key_env) if provider.api_key_env else None
    return text.replace(key, "***") if key else text


def _retry_after(e: Exception, provider: ProviderSpec) -> float | None:
    if isinstance(e, DriverError):
        headers = e.headers
    else:
        response = getattr(e, "response", None)
        headers = response.headers if response is not None else {}
    value = headers.get("retry-after")
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
