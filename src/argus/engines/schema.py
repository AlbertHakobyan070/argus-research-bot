"""Engines configuration schema (v4 spec §2.2) — models and validators, no behaviour.

WP-A/WP-D may add methods and validators here, never fields.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, get_args

from pydantic import BaseModel, model_validator

from .roles import DEGRADABLE_ROLES, LEGACY_ROLES, MUST_RAISE_ROLES, ROLES, SKIPPABLE_ROLES, Tier


def _check_policy(role: str, on_exhausted: str, *, known: tuple[str, ...] = ROLES) -> None:
    """Unknown role → ValueError; on_exhausted must fit the role's §2.1 class."""
    if role not in known:
        raise ValueError(f"unknown role {role!r}")
    if on_exhausted not in ("raise", "skip_item", "degrade"):
        raise ValueError(f"role {role}: unknown on_exhausted {on_exhausted!r}")
    if role in MUST_RAISE_ROLES and on_exhausted != "raise":
        raise ValueError(f"role {role}: on_exhausted must be 'raise' (got {on_exhausted!r})")
    if on_exhausted == "skip_item" and role not in SKIPPABLE_ROLES:
        raise ValueError(f"role {role}: on_exhausted 'skip_item' is only allowed for skippable roles")
    if on_exhausted == "degrade" and role not in DEGRADABLE_ROLES:
        raise ValueError(f"role {role}: on_exhausted 'degrade' is only allowed for degradable roles")


class ProviderSpec(BaseModel):
    name: str
    kind: Literal["openai_compatible"] = "openai_compatible"      # future: "anthropic_native" etc. → new driver class, optional dep
    base_url: str                                                    # may be overridden by base_url_env
    base_url_env: str | None = None
    api_key_env: str | None = None                                   # env var NAME; None for keyless local servers
    headers: dict[str, str] = {}                                     # e.g. OpenRouter HTTP-Referer / X-Title
    extra_body: dict = {}                                            # provider-level request fields; EngineSpec.extra_body is merged over it
    supports_json_mode: bool = False                                 # send response_format={"type":"json_object"} when a role asks
    supports_temperature: bool = True                                # provider default; EngineSpec may override per model
    token_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"   # openai → "max_completion_tokens"
    models_endpoint: bool = True                                     # GET {base_url}/models works with the same auth
    known_models: list[str] = []                                     # static list when models_endpoint is False (only VERIFIED ids)
    model_preferences: dict[str, list[str]] = {}                     # tier -> ordered candidate ids (FreeLLMAPI only); "auto" allowed last
    rpm_limit: int | None = None                                     # client-side token bucket; None = unlimited
    timeout_s: float = 60.0                                          # paid providers set ≥ 300 in argus_engines.json (R7)
    timeout_retry_max: int = 1                                       # a timed-out request is retried at most once
    retry_after_cap_s: float = 30.0                                  # Retry-After is honoured up to this bound
    notes: str = ""                                                  # caveats shown in /engines and the console

    @model_validator(mode="after")
    def _preference_keys_are_tiers(self) -> ProviderSpec:
        bad = sorted(set(self.model_preferences) - set(get_args(Tier)))
        if bad:
            raise ValueError(f"provider {self.name}: model_preferences keys must be tiers "
                             f"{get_args(Tier)}, got {bad}")
        return self


class EngineSpec(BaseModel):
    provider: str
    model: str | None = None                                         # concrete id, or None → see model_from_preferences (not both: validator)
    model_from_preferences: str | None = None                        # tier name resolved against provider.model_preferences + live list
    temperature: float | None = None
    max_tokens: int | None = None
    json_mode: bool | None = None                                    # None → role default (JSON_ROLES and provider.supports_json_mode)
    extra_body: dict = {}                                            # per-model/per-role request fields (thinking/reasoning controls), merged over the provider's
    supports_temperature: bool | None = None                         # None → provider value

    @model_validator(mode="after")
    def _one_model_source(self) -> EngineSpec:
        if self.model is not None and self.model_from_preferences is not None:
            raise ValueError(f"provider {self.provider}: set model or model_from_preferences, not both "
                             f"(got {self.model!r} and {self.model_from_preferences!r})")
        return self


class RoleBinding(BaseModel):
    primary: EngineSpec
    fallbacks: list[EngineSpec] = []                                 # explicit, ordered; empty = none
    on_exhausted: Literal["raise", "skip_item", "degrade"] = "raise" # skip_item only for SKIPPABLE_ROLES; degrade only for DEGRADABLE_ROLES;
                                                                     # MUST_RAISE_ROLES must be "raise" (validators)
    retry_on_truncation: bool = True                                 # finish_reason=="length" on a JSON role → one retry with 2× max_tokens (cap 4096), recorded


def _check_roles(roles: dict[str, RoleBinding]) -> None:
    for role, binding in roles.items():
        _check_policy(role, binding.on_exhausted)


class Profile(BaseModel):
    name: str
    description: str = ""
    tiers: dict[Tier, EngineSpec]                                    # keys can only be tiers; the validator requires all three
    roles: dict[str, RoleBinding] = {}                               # overrides by role id

    @model_validator(mode="after")
    def _all_tiers(self) -> Profile:
        missing = [t for t in get_args(Tier) if t not in self.tiers]
        if missing:
            raise ValueError(f"profile {self.name}: tiers must hold exactly {get_args(Tier)} (missing {missing})")
        return self

    @model_validator(mode="after")
    def _roles_valid(self) -> Profile:
        _check_roles(self.roles)
        return self


class ProfileOverrides(BaseModel):                                   # persisted per profile (library key engines.overrides)
    tiers: dict[Tier, EngineSpec] = {}
    roles: dict[str, RoleBinding] = {}

    @model_validator(mode="after")
    def _roles_valid(self) -> ProfileOverrides:
        _check_roles(self.roles)
        return self


class TranscriberConfig(BaseModel):                                  # data lives in argus_engines.json (WP-F); persisted under library key engines.transcriber
    youtube: list[Literal["captions", "asr"]] = ["captions", "asr"]  # ordered; the documented captions→ASR fallback, recorded in TranscriptResult.backend
    asr_engine: Literal["faster_whisper", "openai_audio"] = "faster_whisper"
    faster_whisper: dict = {}                                        # {"model","device","compute_type"} from the JSON block
    openai_audio: dict = {}                                          # {"provider","model","chunk_seconds","language"} from the JSON block


def _engine_specs(label: str, tiers: dict, roles: dict[str, RoleBinding]) -> Iterator[tuple[str, EngineSpec]]:
    for tier, spec in tiers.items():
        yield f"{label} tier {tier}", spec
    for role, binding in roles.items():
        for spec in (binding.primary, *binding.fallbacks):
            yield f"{label} role {role}", spec


class ExaPrices(BaseModel):
    price_per_search_usd: float
    price_per_page_usd: float
    prices_as_of: str


class PricingConfig(BaseModel):                                      # dated data, lives in argus_engines.json
    exa: ExaPrices | None = None                                     # None → estimates show "n/a"


class EnginesConfig(BaseModel):
    version: int = 1
    providers: dict[str, ProviderSpec]
    profiles: dict[str, Profile]
    active_profile: str
    overrides: dict[str, ProfileOverrides] = {}                      # merged from library settings `engines.overrides` at load
    transcriber: TranscriberConfig = TranscriberConfig()
    pricing: PricingConfig = PricingConfig()

    @model_validator(mode="after")
    def _references_resolve(self) -> EnginesConfig:
        if self.active_profile not in self.profiles:
            raise ValueError(f"active_profile {self.active_profile!r} is not a defined profile")
        unknown = sorted(set(self.overrides) - set(self.profiles))
        if unknown:
            raise ValueError(f"overrides for unknown profile(s) {unknown}")
        specs = [s for name, p in self.profiles.items() for s in _engine_specs(f"profile {name}", p.tiers, p.roles)]
        specs += [s for name, o in self.overrides.items() for s in _engine_specs(f"overrides {name}", o.tiers, o.roles)]
        for where, spec in specs:
            if spec.provider not in self.providers:
                raise ValueError(f"{where}: unknown provider {spec.provider!r}")
        return self


class ResolvedBinding(BaseModel):                                    # concrete, after model resolution
    role: str
    tier: str
    primary: EngineSpec
    fallbacks: list[EngineSpec]
    on_exhausted: str
    retry_on_truncation: bool

    @model_validator(mode="after")
    def _role_and_policy(self) -> ResolvedBinding:
        _check_policy(self.role, self.on_exhausted, known=ROLES + LEGACY_ROLES)
        return self


class EnginesSnapshot(BaseModel):                                    # stored in RunConfig.engines (checkpointed) → resumes reproduce
    profile: str
    resolved_at: str                                                 # ISO
    roles: dict[str, ResolvedBinding]                                # includes LEGACY_ROLES during Phases 1–2
    warnings: list[str] = []                                         # dropped/unresolved fallbacks etc. — shown in /engines, console, report engines line

    @model_validator(mode="after")
    def _keys_are_binding_roles(self) -> EnginesSnapshot:
        for key, binding in self.roles.items():
            if key != binding.role:
                raise ValueError(f"snapshot key {key!r} holds the binding for role {binding.role!r}")
        return self
