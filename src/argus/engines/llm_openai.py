"""OpenAI-compatible driver (v4 spec §2.3): the openai SDK, called directly.

One client per (provider, model), built with ``max_retries=0`` (the facade
owns every retry) and never with ``api_key=None`` (a keyless provider gets
the placeholder "none"). A request uses the provider's token parameter,
sends ``temperature`` only when the model accepts it and ``response_format``
only in JSON mode, and merges the engine's ``extra_body`` over the
provider's. From the raw response it keeps the served model, ``routed_via``
(header ``x-routed-via``, else body ``_routed_via`` / ``x_groq``), the finish
reason, token usage including reasoning tokens, and the rate-limit headers.
A response that is not a usable completion — an error object in the body
(often with HTTP 200), no choices, or a body that is not a chat completion —
raises ``DriverError``; the facade treats it as a failed attempt.

``Driver`` is the interface a native driver (``ProviderSpec.kind``) would
implement later."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import BaseMessage
from openai import OpenAI
from openai.types.chat import ChatCompletion

from . import registry
from .registry import EngineConfigError
from .schema import EngineSpec, ProviderSpec

_ROLE_OF = {"system": "system", "human": "user", "ai": "assistant"}
_KEPT_HEADERS = frozenset({"retry-after", "x-fallback-attempts"})


class DriverError(RuntimeError):
    """The provider answered, but not with a usable chat completion.
    ``code`` is the error object's numeric code (429 or 5xx → transient),
    else None; ``headers`` are the response headers (``Retry-After``)."""

    def __init__(self, message: str, *, code: int | None = None, headers: Any = None):
        super().__init__(message)
        self.code = code
        self.headers = headers if headers is not None else {}


def _body_error(provider: ProviderSpec, error: Any, headers: Any) -> DriverError:
    message, code = (error.get("message") or error, error.get("code")) if isinstance(error, dict) else (error, None)
    try:
        code = int(code) if code is not None else None
    except (TypeError, ValueError):     # a symbolic code ("rate_limit_exceeded") is not a status
        code = None
    return DriverError(f"provider {provider.name}: error in the response body: {message}", code=code,
                       headers=headers)


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
        try:
            completion = raw.parse()
        except ValueError as e:           # a JSON content type whose body is not JSON
            raise DriverError(f"provider {provider.name}: the response body is not JSON ({e})",
                              headers=raw.headers) from e
        if not isinstance(completion, ChatCompletion):    # the SDK hands back a non-JSON body as text
            raise DriverError(f"provider {provider.name}: the response is not a chat completion: "
                              f"{str(completion)[:200]!r}", headers=raw.headers)
        error = (completion.model_extra or {}).get("error")
        if error:
            raise _body_error(provider, error, raw.headers)
        choice = completion.choices[0] if completion.choices else None
        if choice is None or choice.message is None:
            raise DriverError(f"provider {provider.name}: the response has no choices", headers=raw.headers)
        usage = completion.usage
        details = usage.completion_tokens_details if usage else None
        return DriverResponse(
            text=choice.message.content or "",
            served_model=completion.model or "",
            routed_via=_routed_via(raw.headers, completion.model_extra or {}),
            finish_reason=choice.finish_reason or "",
            usage={"prompt_tokens": (usage.prompt_tokens or 0) if usage else 0,
                   "completion_tokens": (usage.completion_tokens or 0) if usage else 0,
                   "reasoning_tokens": (details.reasoning_tokens or 0) if details else 0},
            headers={k: v for k, v in raw.headers.items()
                     if k in _KEPT_HEADERS or k.startswith("x-ratelimit-")},
            raw=completion)
