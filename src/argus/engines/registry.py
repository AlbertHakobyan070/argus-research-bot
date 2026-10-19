"""Engines registry (v4 spec §2.2/§2.3): load the engines config, validate
it, resolve the active profile into an ``EnginesSnapshot``, and persist the
``/engines`` changes in the library's settings.

Async-safe: the library is aiosqlite, and a live ``/models`` listing runs in
a worker thread (``asyncio.to_thread``), never on the event-loop thread.
Startup only loads and checks (``validate_schema``, ``static_problems``):
no network. A run start or a profile switch resolves (``validate_active``).

Layers, lowest first: the defaults file (``ARGUS_ENGINES_FILE``), the
optional local file (``ARGUS_ENGINES_LOCAL_FILE``), then the library keys
``engines.active_profile``, ``engines.overrides``, ``engines.transcriber``.
Dicts merge key by key, except that an engine spec naming its ``provider``
replaces the old spec whole; lists and scalars replace.

Binding rules:
- A role's binding is the override's, else the profile's, else the tier
  spec with ``on_exhausted="raise"``. The ``legacy.*`` roles (Phases 1–2)
  always take the tier spec.
- A binding whose primary repeats the profile's own spec for the role's tier
  only declares the role's failure policy: its engine follows the tier,
  overrides and ``/engines tier`` included. Any other primary is pinned. So
  change a shipped profile's tier engine with an ``overrides`` block or
  ``/engines tier``: editing the profile's own ``tiers`` pins the roles that
  repeated the old spec.
- ``model`` → that id (a ``known_models`` provider must list it);
  ``model_from_preferences`` → the first preferred id in the live
  ``/models`` list (``known_models`` without an endpoint); neither → the
  user must choose a model.
- An unresolved or keyless primary raises ``EngineConfigError`` naming role
  and provider; an unresolved or keyless fallback is dropped with a line in
  ``snapshot.warnings``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from ..config import _PROJECT_ROOT  # importing argus.config also loads .env
from .roles import LEGACY_ROLES, ROLE_TIERS, ROLES
from .schema import (EngineSpec, EnginesConfig, EnginesSnapshot, ProfileOverrides, ProviderSpec,
                     ResolvedBinding, RoleBinding, TranscriberConfig, _engine_specs)

logger = logging.getLogger("argus.engines")


class EngineConfigError(RuntimeError):
    """An engines configuration that cannot run; the message names the role
    and/or the provider. Re-exported as ``argus.llm.EngineConfigError``."""


_ROOT = _PROJECT_ROOT
MODELS_TTL_S = 600.0
_MODELS_TIMEOUT_S = 15.0
SETTINGS_KEYS = {"active_profile": "engines.active_profile",
                 "overrides": "engines.overrides",
                 "transcriber": "engines.transcriber"}
# Env knobs no longer read (spec §2.4, §7): one startup WARNING each if set.
# The WP that stops reading a knob adds it here.
REMOVED_ENV_KNOBS = {
    "ARGUS_TIMEOUT": "ProviderSpec.timeout_s in argus_engines.json",
    "ARGUS_LLM_RETRIES": "the LLM facade's retry policy",
    "PERPLEXITY_API_KEY": "nothing (Perplexity search was removed)",
    "ARGUS_RESEARCH_WAVES": "run_config research.waves (the depth preset)",
    "ARGUS_FETCH_PER_WAVE": "run_config research.fetch_per_wave (the depth preset)",
    "ARGUS_TOTAL_FETCH_CAP": "run_config research.total_fetch_cap (the depth preset)",
    "ARGUS_EXA_MAX_CALLS": "run_config search.exa.max_searches (a per-run budget)",
    "ARGUS_FETCH_CONCURRENCY": "run_config fetch.concurrency",
    "ARGUS_DIGEST_CONCURRENCY": "run_config research.digest_concurrency",
    "ARGUS_MAX_REVISIONS": "run_config panel.max_compose_revisions",
    "ARGUS_COMPOSE_CONCURRENCY": "run_config compose.concurrency",
}


@dataclass
class _State:
    cfg: EnginesConfig | None = None          # merged and validated
    snapshot: EnginesSnapshot | None = None   # the active profile, resolved
    base: dict | None = None                  # defaults file merged with the local file
    layer: dict = field(default_factory=dict)  # the library layer, by SETTINGS_KEYS name
    library: Any = None


_state = _State()
_models_cache: dict[tuple[str, str], tuple[float, list[str]]] = {}
_models_lock = threading.Lock()


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def default_paths() -> list[Path]:
    """The defaults file, then the optional local file (relative to the repo)."""
    return [_env_path("ARGUS_ENGINES_FILE", "argus_engines.json"),
            _env_path("ARGUS_ENGINES_LOCAL_FILE", "argus_engines.local.json")]


def _env_path(name: str, default: str) -> Path:
    p = Path(os.environ.get(name) or default).expanduser()
    return p if p.is_absolute() else _ROOT / p


async def load_config(paths: Sequence[str | Path] | None = None, library: Any = None) -> EnginesConfig:
    """Read and merge the layers, validate the schema, and make the result
    the registry's config (dropping the cached snapshot). The first path
    must exist, later ones are optional. No network."""
    if library is not None:
        _state.library = library
    files = [Path(p) for p in (paths or default_paths())]
    _state.cfg = _state.snapshot = None
    _state.base = await asyncio.to_thread(_read_files, files)
    _state.layer = await _read_layer(_state.library) if _state.library is not None else {}
    _state.cfg = _build(_state.base, _state.layer)
    return _state.cfg


def _read_files(files: list[Path]) -> dict:
    merged: dict = {}
    for i, path in enumerate(files):
        try:                                       # the stat too: is_file() raises PermissionError
            if not path.is_file():
                if i == 0:
                    raise EngineConfigError(f"engines file not found: {path}")
                continue
            text = path.read_text(encoding="utf-8-sig")                   # a BOM (Notepad) is fine
        except UnicodeDecodeError as e:
            raise EngineConfigError(f"{path}: is not UTF-8 ({e}); save it as UTF-8") from e
        except OSError as e:
            raise EngineConfigError(f"{path}: cannot be read ({type(e).__name__}: {e})") from e
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise EngineConfigError(f"{path}: invalid JSON ({e})") from e
        if not isinstance(data, dict):
            raise EngineConfigError(f"{path}: the top level must be a JSON object")
        merged = _deep_merge(merged, data)
    return merged


async def _read_layer(library: Any) -> dict:
    layer = {}
    for name, key in SETTINGS_KEYS.items():
        raw = await library.get_setting(key)
        if raw is None:
            continue
        try:
            layer[name] = json.loads(raw)
        except json.JSONDecodeError as e:
            raise EngineConfigError(f"library setting {key}: invalid JSON ({e})") from e
    return layer


def _deep_merge(base: Any, over: Any) -> Any:
    """Dicts merge key by key; an engine spec naming its provider, a list or
    a scalar replaces."""
    if isinstance(base, dict) and isinstance(over, dict) and "provider" not in over:
        out = dict(base)
        for k, v in over.items():
            out[k] = _deep_merge(base[k], v) if k in base else v
        return out
    return over


def _build(base: dict, layer: dict) -> EnginesConfig:
    try:
        return EnginesConfig.model_validate(_deep_merge(base, layer))
    except ValidationError as e:
        raise EngineConfigError(f"engines config invalid: {_errors(e)}") from e


def _errors(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, err['loc'])) or 'config'}: {err['msg']}"
                     for err in e.errors(include_url=False))


# ---------------------------------------------------------------------------
# static validation (startup): no I/O
# ---------------------------------------------------------------------------

def validate_schema(cfg: EnginesConfig) -> None:
    """Schema and references, without I/O: the pydantic validators, names
    that match their keys, and every ``model_from_preferences`` pointing at
    a preference list its provider has."""
    try:
        EnginesConfig.model_validate(cfg.model_dump())
    except ValidationError as e:
        raise EngineConfigError(f"engines config invalid: {_errors(e)}") from e
    problems = [f"{kind} {key!r}: its name is {obj.name!r}"
                for kind, table in (("provider", cfg.providers), ("profile", cfg.profiles))
                for key, obj in table.items() if obj.name != key]
    for where, spec in _all_specs(cfg):
        pref = spec.model_from_preferences
        if pref is not None and not cfg.providers[spec.provider].model_preferences.get(pref):
            problems.append(f"{where}: provider {spec.provider} has no model_preferences[{pref!r}]")
    if problems:
        raise EngineConfigError("; ".join(problems))


def _all_specs(cfg: EnginesConfig) -> Iterator[tuple[str, EngineSpec]]:
    for name, p in cfg.profiles.items():
        yield from _engine_specs(f"profile {name}", p.tiers, p.roles)
    for name, o in cfg.overrides.items():
        yield from _engine_specs(f"overrides {name}", o.tiers, o.roles)


def static_problems(cfg: EnginesConfig) -> list[str]:
    """What is wrong with the active profile's primaries without I/O: a
    missing ``api_key_env``, a fixed id outside a provider's ``known_models``."""
    keyless: dict[str, str] = {}
    problems: list[str] = []
    for role, (_tier, b) in bindings(cfg, cfg.active_profile).items():
        prov = cfg.providers[b.primary.provider]
        if prov.api_key_env and not os.environ.get(prov.api_key_env):
            keyless.setdefault(prov.name, prov.api_key_env)
        model = b.primary.model
        if not prov.models_endpoint and model is not None and model not in prov.known_models:
            problems.append(f"role {role}: model {model} is not in provider {prov.name}'s known_models")
    return [f"provider {name}: {env} is not set (profile {cfg.active_profile})"
            for name, env in keyless.items()] + problems


# ---------------------------------------------------------------------------
# bindings and resolution
# ---------------------------------------------------------------------------

def _tier(role: str) -> str:
    if role in ROLE_TIERS:
        return ROLE_TIERS[role]
    if role in LEGACY_ROLES:
        return role.removeprefix("legacy.")
    raise ValueError(f"unknown role {role!r}")


def _profile(cfg: EnginesConfig, profile: str):
    if profile not in cfg.profiles:
        raise EngineConfigError(f"unknown profile {profile!r} (known: {', '.join(cfg.profiles)})")
    return cfg.profiles[profile]


def configured_binding(cfg: EnginesConfig, profile: str, role: str) -> RoleBinding:
    """A role's binding as configured — the override's, else the profile's,
    else the profile's tier spec — before tier-following."""
    p = _profile(cfg, profile)
    ov = cfg.overrides.get(profile)
    b = (ov.roles.get(role) if ov else None) or p.roles.get(role)
    return b if b is not None else RoleBinding(primary=p.tiers[_tier(role)])


def bindings(cfg: EnginesConfig, profile: str) -> dict[str, tuple[str, RoleBinding]]:
    """Every role (``ROLES`` + ``LEGACY_ROLES``) → (tier, effective binding)."""
    p = _profile(cfg, profile)
    ov = cfg.overrides.get(profile)
    tiers = {**p.tiers, **(ov.tiers if ov else {})}
    out = {}
    for role in ROLES + LEGACY_ROLES:
        tier = _tier(role)
        b = configured_binding(cfg, profile, role)
        if b.primary == p.tiers[tier]:          # a policy-only declaration follows the tier
            b = b.model_copy(update={"primary": tiers[tier]})
        out[role] = (tier, b)
    return out


async def resolve_snapshot(cfg: EnginesConfig, profile: str) -> EnginesSnapshot:
    """Resolve every role of one profile to concrete models."""
    listings: dict[str, list[str] | EngineConfigError] = {}

    async def resolve(spec: EngineSpec) -> EngineSpec:
        prov = cfg.providers[spec.provider]
        if prov.api_key_env and not os.environ.get(prov.api_key_env):
            raise EngineConfigError(f"provider {prov.name}: {prov.api_key_env} is not set")
        if spec.model is not None:
            if not prov.models_endpoint and spec.model not in prov.known_models:
                raise EngineConfigError(f"model {spec.model} is not in provider {prov.name}'s known_models")
            return spec
        if spec.model_from_preferences is None:
            raise EngineConfigError(f"choose a model for provider {prov.name} "
                                    f"(/engines models {prov.name}, or the console)")
        if prov.name not in listings:
            try:
                listings[prov.name] = await asyncio.to_thread(_models_for, prov)
            except EngineConfigError as e:      # one failed listing serves every role of this call
                listings[prov.name] = e
        live = listings[prov.name]
        if isinstance(live, EngineConfigError):
            raise live
        wanted = prov.model_preferences.get(spec.model_from_preferences) or []
        model = next((m for m in wanted if m in live), None)
        if model is None:
            raise EngineConfigError(f"provider {prov.name}: none of its {spec.model_from_preferences} "
                                    f"preferences is live ({', '.join(wanted)})")
        return spec.model_copy(update={"model": model, "model_from_preferences": None})

    roles: dict[str, ResolvedBinding] = {}
    warnings: list[str] = []
    for role, (tier, b) in bindings(cfg, profile).items():
        try:
            primary = await resolve(b.primary)
        except EngineConfigError as e:
            raise EngineConfigError(f"role {role}: {e}") from e
        fallbacks = []
        for i, fb in enumerate(b.fallbacks, 1):
            try:
                fallbacks.append(await resolve(fb))
            except EngineConfigError as e:
                warnings.append(f"role {role}: fallback {i} ({fb.provider}) dropped — {e}")
        roles[role] = ResolvedBinding(role=role, tier=tier, primary=primary, fallbacks=fallbacks,
                                      on_exhausted=b.on_exhausted,
                                      retry_on_truncation=b.retry_on_truncation)
    return EnginesSnapshot(profile=profile, resolved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           roles=roles, warnings=warnings)


async def validate_active(cfg: EnginesConfig) -> EnginesSnapshot:
    """Full validation of the active profile (run start, profile switch)."""
    validate_schema(cfg)
    return await resolve_snapshot(cfg, cfg.active_profile)


async def active_snapshot() -> EnginesSnapshot:
    """The active profile's snapshot: resolved once (loading the config if
    nothing is loaded), then cached until a ``set_*`` call replaces it.
    Every drive entry awaits it before driving (spec §2.3)."""
    if _state.snapshot is not None:
        return _state.snapshot
    cfg = _state.cfg if _state.cfg is not None else await load_config()
    snap = await validate_active(cfg)
    if _state.cfg is cfg:  # ponytail: no lock — a set_* that landed meanwhile wins (one user)
        _state.snapshot = snap
    return snap


def current_snapshot_cached() -> EnginesSnapshot:
    """Sync, for the Phase 1–2 legacy wrappers inside a running graph."""
    if _state.snapshot is None:
        raise EngineConfigError("no engines snapshot resolved: every drive entry must "
                                "`await registry.active_snapshot()` first")
    return _state.snapshot


def current_config() -> EnginesConfig:
    if _state.cfg is None:
        raise EngineConfigError("engines config not loaded (registry.load_config)")
    return _state.cfg


def provider(name: str) -> ProviderSpec:
    cfg = current_config()
    if name not in cfg.providers:
        raise EngineConfigError(f"provider {name!r} is not configured")
    return cfg.providers[name]


# ---------------------------------------------------------------------------
# changes from /engines (and the console): validated, then persisted
# ---------------------------------------------------------------------------

def library_overrides(profile: str) -> ProfileOverrides:
    """One profile's overrides as stored in the library layer."""
    raw = (_state.layer.get("overrides") or {}).get(profile)
    return ProfileOverrides.model_validate(raw) if raw else ProfileOverrides()


async def set_active_profile(name: str) -> EnginesSnapshot:
    """Switch profiles: full validation first; nothing persists or changes
    when it fails. Works on the loaded layers even when the merged config
    failed to build, so a bad library value can be switched away from."""
    if _state.cfg is not None:
        _profile(_state.cfg, name)
    change = {"active_profile": name}
    cfg = _rebuild(change)
    snap = await validate_active(cfg)
    await _persist("active_profile", name)
    _commit(cfg, change, snap)
    return snap


async def set_overrides(profile: str, overrides: ProfileOverrides) -> EnginesSnapshot | None:
    """Replace one profile's library-layer overrides. The active profile is
    fully re-validated first; another profile gets the schema check (its
    full check runs when it is activated). Returns the active snapshot."""
    if _state.cfg is not None:
        _profile(_state.cfg, profile)
    stored = {**(_state.layer.get("overrides") or {}), profile: overrides.model_dump()}
    change = {"overrides": stored}
    cfg = _rebuild(change)
    validate_schema(cfg)
    snap = await validate_active(cfg) if profile == cfg.active_profile else _state.snapshot
    await _persist("overrides", stored)
    _commit(cfg, change, snap)
    return snap


async def set_transcriber(transcriber: TranscriberConfig) -> None:
    change = {"transcriber": transcriber.model_dump()}
    cfg = _rebuild(change)
    await _persist("transcriber", change["transcriber"])
    _commit(cfg, change, _state.snapshot)


def _rebuild(change: dict) -> EnginesConfig:
    if _state.base is None:
        raise EngineConfigError("engines config not loaded (registry.load_config)")
    return _build(_state.base, {**_state.layer, **change})


async def _persist(name: str, value: Any) -> None:
    if _state.library is None:
        raise EngineConfigError("engines settings cannot be persisted: no library")
    await _state.library.set_setting(SETTINGS_KEYS[name], json.dumps(value))


def _commit(cfg: EnginesConfig, change: dict, snap: EnginesSnapshot | None) -> None:
    _state.cfg, _state.layer, _state.snapshot = cfg, {**_state.layer, **change}, snap


# ---------------------------------------------------------------------------
# providers: URL, key, live model list
# ---------------------------------------------------------------------------

def base_url(prov: ProviderSpec) -> str:
    return (os.environ.get(prov.base_url_env) if prov.base_url_env else None) or prov.base_url


def api_key(prov: ProviderSpec) -> str:
    """The key to send. A keyless provider gets the placeholder "none",
    never None: with None the openai SDK falls back to OPENAI_API_KEY and
    would send it to this base_url (spec §2.3)."""
    if prov.api_key_env is None:
        return "none"
    key = os.environ.get(prov.api_key_env)
    if not key:
        raise EngineConfigError(f"provider {prov.name}: {prov.api_key_env} is not set")
    return key


def live_models(provider_name: str) -> list[str]:
    """The ids a provider serves: its live ``/models`` list (cached
    ``MODELS_TTL_S``), or ``known_models``. Sync: async callers use
    ``asyncio.to_thread``."""
    return _models_for(provider(provider_name))


def _models_for(prov: ProviderSpec) -> list[str]:
    if not prov.models_endpoint:
        return list(prov.known_models)
    url = base_url(prov).rstrip("/") + "/models"
    key = (prov.name, url)
    with _models_lock:
        hit = _models_cache.get(key)
    if hit and time.monotonic() - hit[0] < MODELS_TTL_S:
        return list(hit[1])
    headers = dict(prov.headers)
    secret = api_key(prov) if prov.api_key_env else None
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    try:
        data = _get_json(url, headers)
        ids = sorted({m["id"] for m in data["data"] if isinstance(m, dict) and "id" in m})
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
        detail = f"{type(e).__name__}: {e}"
        leaked = bool(secret) and secret in detail
        raise EngineConfigError(f"provider {prov.name}: cannot list models at {url} "
                                f"({detail.replace(secret, '***') if leaked else detail})"
                                ) from (None if leaked else e)   # never chain an error that holds the key
    with _models_lock:
        _models_cache[key] = (time.monotonic(), ids)
    return ids


def _get_json(url: str, headers: dict) -> Any:
    with httpx.Client(timeout=_MODELS_TIMEOUT_S) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

# Path prefixes are stripped, then the first matching prefix names the family
# (the table of the original WP-A text); else the name's first token.
_FAMILY_PREFIXES = (
    ("claude", "anthropic"),
    ("gpt-oss", "openai"), ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"), ("o4", "openai"),
    ("gemini", "google"), ("gemma", "google"),
    ("llama", "meta"), ("meta", "meta"),
    ("qwen", "qwen"), ("deepseek", "deepseek"),
    ("mistral", "mistral"), ("ministral", "mistral"), ("codestral", "mistral"),
    ("magistral", "mistral"), ("devstral", "mistral"),
    ("nemotron", "nvidia"), ("glm", "zhipu"), ("z-ai", "zhipu"),
    ("kimi", "moonshot"), ("moonshot", "moonshot"), ("command", "cohere"),
    ("cogito", "deepcogito"), ("minimax", "minimax"), ("granite", "ibm"), ("phi", "microsoft"),
)


def family(model_id: str) -> str:
    """Coarse model family, for judge/writer diversity."""
    name = (model_id or "").strip().lower().rsplit("/", 1)[-1]
    for prefix, fam in _FAMILY_PREFIXES:
        if name.startswith(prefix):
            return fam
    return re.split(r"[^a-z0-9]+", name, maxsplit=1)[0] or name


_WRITER_ROLES = frozenset({"compose.section", "compose.flat", "compose.analyst", "legacy.strong"})
_JUDGE_ROLES = frozenset({"panel.grounding", "legacy.judge"})
_ITEM_LINE = re.compile(r"^(?:\[r\d+\]\s*)?(skip|degrade):\s*(?:\[r\d+\]\s*)?(\S+)")
# the facade's exhaustion message (llm.complete), wherever a call site quotes it
_EXHAUSTED = re.compile(r"\brole ([a-z0-9_.]+): every engine failed after (\d+) attempts?")


def _role_of(call: dict) -> str:
    return call.get("role") or (f"legacy.{call['tier']}" if call.get("tier") else "unknown")


def _empty_row() -> dict:
    return {"calls": 0, "ok": 0, "exhausted": 0, "attempts": 0, "failed_attempts": 0,
            "exhausted_attempts": 0, "served_models": {}, "fallbacks": 0, "skipped": 0, "degraded": 0,
            "retries": {}, "avg_latency_ms": 0.0}


def summarize_calls(model_calls: list[dict], errors: list[str] | None = None, *,
                    exa_usd: float | None = None) -> dict:
    """The provenance summary for ``metadata.json["engines"]``.

    Per role, ``calls`` counts logical calls: the successful ones (an ok
    record) plus the exhausted ones (``exhausted``: each "role <id>: every
    engine failed after <n> attempts" quoted in ``errors``). The attempt
    records are counted apart: ``attempts`` / ``failed_attempts`` over the
    records in ``model_calls`` (with any ``extra["prior_attempts"]`` the
    legacy wrappers attach), and ``exhausted_attempts`` = the <n> of the
    exhausted ``legacy.*`` calls, whose records the Phase 1–2 call sites do
    not keep. A real role's call site appends every attempt record
    (``EngineError.attempts``) to ``model_calls``, so its exhausted attempts
    are already counted there and never again from the text (0 here).
    ``skipped`` / ``degraded`` come from the ``skip:`` / ``degrade:`` lines.
    v3 records (no role) count as ``legacy.<tier>``."""
    records = [a for c in model_calls or []
               for a in [*((c.get("extra") or {}).get("prior_attempts") or []), c]]
    groups: dict[str, list[dict]] = {}
    for c in records:
        groups.setdefault(_role_of(c), []).append(c)
    by_role: dict[str, dict] = {}
    for role, cs in groups.items():
        oks = [c for c in cs if c.get("ok", True)]
        by_role[role] = {**_empty_row(), "ok": len(oks), "attempts": len(cs), "failed_attempts": len(cs) - len(oks),
                         "served_models": dict(Counter(c["served_model"] for c in oks if c.get("served_model"))),
                         "fallbacks": sum(1 for c in oks if c.get("fallback_rank")),
                         "retries": dict(Counter(c["retry_reason"] for c in cs if c.get("retry_reason"))),
                         "avg_latency_ms": round(sum(int(c.get("latency_ms") or 0) for c in cs) / len(cs), 1)}
    degradations = []
    for line in errors or []:
        for m in _EXHAUSTED.finditer(line):
            row = by_role.setdefault(m.group(1), _empty_row())
            row["exhausted"] += 1
            if m.group(1).startswith("legacy."):
                row["exhausted_attempts"] += int(m.group(2))
        m = _ITEM_LINE.match(line)
        if m is None:
            continue
        kind, role = m.groups()
        by_role.setdefault(role, _empty_row())["skipped" if kind == "skip" else "degraded"] += 1
        if kind == "degrade":
            degradations.append(line)
    for row in by_role.values():
        row["calls"] = row["ok"] + row["exhausted"]
    served = [c for c in records if c.get("ok", True) and c.get("served_model")]
    rerouted = sum(1 for c in served if c["served_model"] != c.get("requested_model"))

    def families(roles: frozenset) -> list[str]:
        return sorted({family(c["served_model"]) for c in served if _role_of(c) in roles})
    judge_f, writer_f = families(_JUDGE_ROLES), families(_WRITER_ROLES)
    shared = sorted(set(judge_f) & set(writer_f))
    out = {"calls": sum(r["calls"] for r in by_role.values()),
           "exhausted": sum(r["exhausted"] for r in by_role.values()),
           "attempts": len(records), "by_role": by_role,
           "reroute_rate": round(rerouted / len(served), 3) if served else None,
           "judge_families": judge_f, "writer_families": writer_f,
           "judge_writer_same_family": bool(shared) if judge_f and writer_f else None,
           "warnings": [f"judge and writer share a model family ({', '.join(shared)})"] if shared else [],
           "degradations": degradations}
    if exa_usd is not None:
        out["exa_usd"] = exa_usd
    return out
