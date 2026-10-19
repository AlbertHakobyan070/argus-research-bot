"""OpenAI-compatible driver (v4 spec §2.3): the openai SDK, called directly.

One client per (provider, model), built with ``max_retries=0`` (the facade
owns every retry) and never with ``api_key=None`` (a keyless provider gets
the placeholder "none"). A request uses the provider's token parameter,
sends ``temperature`` only when the model accepts it and ``response_format``
only in JSON mode, and merges the engine's ``extra_body`` over the
provider's. From the raw response it keeps the served model, ``routed_via``
(header ``x-routed-via``, else body ``_routed_via`` / ``x_groq``), the finish
reason, token usage including reasoning tokens, and the rate-limit headers.

``Driver`` is the interface a native driver (``ProviderSpec.kind``) would
implement later."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from openai import OpenAI

from . import registry
from .registry import EngineConfigError
from .schema import EngineSpec, ProviderSpec

_ROLE_OF = {"system": "system", "human": "user", "ai": "assistant"}
_KEPT_HEADERS = frozenset({"retry-after", "x-fallback-attempts"})


@dataclass
class DriverResponse:
    text: str
    served_model: str
    routed_via: str
    finish_reason: str
    usage: dict            # prompt_tokens, completion_tokens, reasoning_tokens
    headers: dict          # retry-after, x-fallback-attempts, x-ratelimit-*
    raw: Any


class Driver(Protocol):
    def complete(self, engine: EngineSpec, provider: ProviderSpec, messages: list[BaseMessage], *,
                 temperature: float | None, max_tokens: int | None,
                 json_mode: bool) -> DriverResponse: ...


def to_openai_messages(messages: list[BaseMessage]) -> list[dict]:
    out = []
    for m in messages:
        role = _ROLE_OF.get(m.type)
        if role is None:
            raise ValueError(f"unsupported message type {m.type!r}")
        out.append({"role": role, "content": m.content})
    return out


def _routed_via(headers, extra: dict) -> str:
    if headers.get("x-routed-via"):
        return headers["x-routed-via"]
    body = extra.get("_routed_via")
    if isinstance(body, str) and body:
        return body
    if isinstance(body, dict):
        parts = [str(body[k]) for k in ("platform", "model") if body.get(k)]
        if parts:
            return "/".join(parts)
    return "groq" if extra.get("x_groq") else ""


class OpenAICompatDriver:
    def __init__(self) -> None:
        self._clients: dict[tuple[str, str], OpenAI] = {}
        self._lock = threading.Lock()

    def _client(self, engine: EngineSpec, provider: ProviderSpec) -> OpenAI:
        key = (provider.name, engine.model or "")
        with self._lock:
            client = self._clients.get(key)
            if client is None:
                client = OpenAI(base_url=registry.base_url(provider), api_key=registry.api_key(provider),
                                timeout=provider.timeout_s, max_retries=0,
                                default_headers=provider.headers)
                self._clients[key] = client
        return client

    def complete(self, engine: EngineSpec, provider: ProviderSpec, messages: list[BaseMessage], *,
                 temperature: float | None, max_tokens: int | None, json_mode: bool) -> DriverResponse:
        if not engine.model:
            raise EngineConfigError(f"provider {provider.name}: no model resolved for this engine")
        kwargs: dict[str, Any] = {"model": engine.model, "messages": to_openai_messages(messages)}
        if max_tokens is not None:
            kwargs[provider.token_param] = max_tokens
        accepts_temperature = (provider.supports_temperature if engine.supports_temperature is None
                               else engine.supports_temperature)
        if temperature is not None and accepts_temperature:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        extra = {**provider.extra_body, **engine.extra_body}
        if extra:
            kwargs["extra_body"] = extra
        raw = self._client(engine, provider).chat.completions.with_raw_response.create(**kwargs)
        completion = raw.parse()
        choice = completion.choices[0] if completion.choices else None
        usage = completion.usage
        details = usage.completion_tokens_details if usage else None
        return DriverResponse(
            text=(choice.message.content or "") if choice else "",
            served_model=completion.model or "",
            routed_via=_routed_via(raw.headers, completion.model_extra or {}),
            finish_reason=(choice.finish_reason or "") if choice else "",
            usage={"prompt_tokens": (usage.prompt_tokens or 0) if usage else 0,
                   "completion_tokens": (usage.completion_tokens or 0) if usage else 0,
                   "reasoning_tokens": (details.reasoning_tokens or 0) if details else 0},
            headers={k: v for k, v in raw.headers.items()
                     if k in _KEPT_HEADERS or k.startswith("x-ratelimit-")},
            raw=completion)
