"""FreeLLMAPI model-tier routing.

FreeLLMAPI exposes an OpenAI-compatible /v1/models endpoint behind a single
Bearer token. We:

  1. Resolve the live model ID list at startup (never assume a fixed list).
  2. Map three tiers (cheap / strong / judge) onto concrete model IDs,
     preferring specific names but falling back to ``auto`` if a preferred
     model disappears.
  3. Expose helpers that build a ``ChatOpenAI`` pointed at the proxy for any
     given tier.

Every call logs the actual model+provider that served it so we can prove
fallback happened.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Literal

import httpx
from langchain_openai import ChatOpenAI

from .config import get_settings

logger = logging.getLogger("argus.llm")

Tier = Literal["cheap", "strong", "judge"]

# Preferred model IDs per tier. We try these in order against the live
# /v1/models list and pick the first one that's present. If none match
# we fall back to "auto" (the proxy's own router).
PREFERRED: dict[Tier, list[str]] = {
    "cheap": [
        "llama-3.1-8b-instant",
        "gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite-preview",
        "Meta-Llama-3.3-70B-Instruct",
        "groq/compound-mini",
    ],
    "strong": [
        "qwen/qwen3-coder:free",
        "deepseek-ai/deepseek-v4-flash",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "gemini-3.5-flash",
        "DeepSeek-V3.2",
        "meta-llama/llama-3.1-70b-instruct",
        "mistral-large-latest",
    ],
    # The judge should be from a DIFFERENT family than the strong model
    # so we don't double-count the same model's blind spots.
    "judge": [
        "openai/gpt-oss-120b:free",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "Llama-4-Maverick-17B-128E-Instruct",
        "cogito-2.1:671b",
    ],
}

# Cache resolved choices per process. Invalidated when /v1/models changes.
_resolved_cache: dict[Tier, str] = {}
_model_list_cache: list[str] = []
_cache_signature: tuple[str, int] | None = None


@dataclass
class CallRecord:
    """One LLM call's provenance for telemetry + per-call transparency."""

    tier: Tier
    requested_model: str
    served_model: str
    served_provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extra: dict = field(default_factory=dict)

    def model_dump(self) -> dict:
        """Pydantic-style dict for symmetry with our other state models."""
        return {
            "tier": self.tier,
            "requested_model": self.requested_model,
            "served_model": self.served_model,
            "served_provider": self.served_provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "extra": dict(self.extra),
        }


def _signature() -> tuple[str, int]:
    s = get_settings()
    return (s.freellmapi_api_key, s.freellmapi_base_url.__hash__())


def fetch_live_models(force: bool = False) -> list[str]:
    """Return the live FreeLLMAPI /v1/models IDs (cached)."""
    global _model_list_cache, _cache_signature
    sig = _signature()
    if not force and _model_list_cache and _cache_signature == sig:
        return _model_list_cache
    s = get_settings()
    url = f"{s.freellmapi_base_url}/models"
    headers = {"Authorization": f"Bearer {s.freellmapi_api_key}"}
    with httpx.Client(timeout=15.0) as c:
        r = c.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
    ids = sorted({m["id"] for m in data.get("data", []) if "id" in m})
    _model_list_cache = ids
    _cache_signature = sig
    logger.info("FreeLLMAPI models resolved: %d live IDs", len(ids))
    return ids


def resolve_tier(tier: Tier, force: bool = False) -> str:
    """Pick a concrete model ID for the given tier, or ``auto`` as fallback."""
    if not force and tier in _resolved_cache:
        return _resolved_cache[tier]
    ids = fetch_live_models(force=force)
    for candidate in PREFERRED[tier]:
        if candidate in ids:
            _resolved_cache[tier] = candidate
            logger.info("Tier %s -> %s", tier, candidate)
            return candidate
    # Tier's preferred list all gone -> use the proxy's own router.
    _resolved_cache[tier] = "auto"
    logger.warning(
        "Tier %s: no preferred model present, falling back to proxy 'auto'",
        tier,
    )
    return "auto"


def pick_strong_and_judge(force: bool = False) -> tuple[str, str]:
    """Pick strong + judge model IDs such that they are from different families.

    Family is approximated by the leading path segment (``openai/``,
    ``gemini``, ``qwen``, ``deepseek``, ``meta-llama``, etc.). If the chosen
    strong model and the chosen judge share a family, walk the judge
    preference list to find a different family.
    """
    ids = fetch_live_models(force=force)
    strong = _first_present(PREFERRED["strong"], ids) or "auto"
    judge = _first_present(PREFERRED["judge"], ids) or "auto"
    if judge != "auto" and strong != "auto":
        if _family(judge) == _family(strong):
            for cand in PREFERRED["judge"]:
                if cand in ids and _family(cand) != _family(strong):
                    judge = cand
                    break
    _resolved_cache["strong"] = strong
    _resolved_cache["judge"] = judge
    logger.info("strong=%s judge=%s (families %s/%s)",
                strong, judge, _family(strong), _family(judge))
    return strong, judge


def _first_present(candidates: list[str], ids: list[str]) -> str | None:
    for c in candidates:
        if c in ids:
            return c
    return None


def _family(model_id: str) -> str:
    """Coarse family bucket so judge != strong."""
    mid = model_id.lower()
    if mid.startswith("auto") or "/" not in mid:
        return mid.split(":")[0]
    return mid.split("/")[0].lstrip("@")


def chat_for_tier(tier: Tier, *, temperature: float = 0.2,
                  max_tokens: int | None = None,
                  model_override: str | None = None) -> ChatOpenAI:
    """Build a ChatOpenAI pointed at FreeLLMAPI for a given tier."""
    s = get_settings()
    model = model_override or resolve_tier(tier)
    # Disable langchain tracing / callbacks that ping langsmith.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    kwargs: dict = dict(
        model=model,
        base_url=s.freellmapi_base_url,
        api_key=s.freellmapi_api_key,
        temperature=temperature,
        timeout=s.request_timeout_seconds,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def record_from_response(tier: Tier, requested: str,
                         response) -> CallRecord:
    """Extract provenance from a langchain AIMessage-or-ChatResult response."""
    served = getattr(response, "response_metadata", {}).get("model_name") \
        or requested
    provider = ""
    meta = getattr(response, "response_metadata", {}) or {}
    # FreeLLMAPI embeds _routed_via under raw response; langchain may surface
    # part of it. Try a few keys.
    for k in ("_routed_via", "routed_via", "x_groq", "provider"):
        v = meta.get(k)
        if isinstance(v, dict) and "platform" in v:
            provider = v["platform"]
            break
        if isinstance(v, str):
            provider = v
            break
    usage = getattr(response, "usage_metadata", None) or {}
    return CallRecord(
        tier=tier,
        requested_model=requested,
        served_model=served,
        served_provider=provider,
        prompt_tokens=int(usage.get("input_tokens", 0) or 0),
        completion_tokens=int(usage.get("output_tokens", 0) or 0),
    )