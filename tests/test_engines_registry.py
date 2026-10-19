"""Engines registry (v4 WP-A, spec §2.2/§2.3): loading and merging the
config, static validation without I/O, model resolution, the cached active
snapshot, persistence of /engines changes, family() and summarize_calls().

Hermetic: the live ``/models`` listing is replaced by a fake ``_get_json``,
and every test starts from an empty registry state."""
from __future__ import annotations

import copy
import json

import httpx
import pytest

from argus.engines import llm_openai, registry
from argus.engines.registry import EngineConfigError
from argus.engines.roles import DEGRADABLE_ROLES, LEGACY_ROLES, ROLES, SKIPPABLE_ROLES
from argus.engines.schema import EngineSpec, EnginesConfig, ProfileOverrides
from argus.library import Library

# A small, fully valid config: one preference-based provider with a
# /models endpoint, one fixed-list provider, one keyless local server.
BASE: dict = {
    "version": 1,
    "active_profile": "main",
    "providers": {
        "proxy": {"name": "proxy", "base_url": "http://proxy/v1", "api_key_env": "T_PROXY_KEY",
                  "model_preferences": {"cheap": ["c1", "c2", "auto"], "strong": ["s1", "auto"],
                                        "judge": ["j1", "auto"]}},
        "fixed": {"name": "fixed", "base_url": "http://fixed/v1", "api_key_env": "T_FIXED_KEY",
                  "models_endpoint": False, "known_models": ["k1", "k2"]},
        "local": {"name": "local", "base_url": "http://127.0.0.1:9/v1"},
    },
    "profiles": {
        "main": {
            "name": "main",
            "tiers": {"cheap": {"provider": "proxy", "model_from_preferences": "cheap"},
                      "strong": {"provider": "proxy", "model_from_preferences": "strong"},
                      "judge": {"provider": "proxy", "model_from_preferences": "judge"}},
            "roles": {
                # policy-only declaration: its primary repeats the tier spec
                "research.digest": {"primary": {"provider": "proxy", "model_from_preferences": "cheap"},
                                    "on_exhausted": "skip_item"},
                "panel.grounding": {"primary": {"provider": "proxy", "model_from_preferences": "judge"},
                                    "fallbacks": [{"provider": "local", "model": None},
                                                  {"provider": "fixed", "model": "k2"}],
                                    "on_exhausted": "skip_item"},
                # pinned: a primary that differs from the tier spec
                "research.rerank": {"primary": {"provider": "fixed", "model": "k1"},
                                    "on_exhausted": "degrade"},
            },
        },
        "other": {
            "name": "other",
            "tiers": {"cheap": {"provider": "fixed", "model": "k1"},
                      "strong": {"provider": "fixed", "model": "k2"},
                      "judge": {"provider": "fixed", "model": "k2"}},
        },
        "broken": {
            "name": "broken",
            "tiers": {"cheap": {"provider": "local", "model": None},
                      "strong": {"provider": "local", "model": None},
                      "judge": {"provider": "local", "model": None}},
        },
    },
}
LIVE = {"http://proxy/v1/models": ["c2", "s1", "j1", "auto"]}


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch):
    monkeypatch.setattr(registry, "_state", registry._State())
    monkeypatch.setattr(registry, "_models_cache", {})
    monkeypatch.setenv("T_PROXY_KEY", "sk-proxy-secret")
    monkeypatch.setenv("T_FIXED_KEY", "sk-fixed-secret")


@pytest.fixture
def fetches(monkeypatch):
    """Fake /models: answers from LIVE and records every URL asked."""
    seen: list[str] = []

    def fake_get_json(url, headers):
        seen.append(url)
        return {"data": [{"id": i} for i in LIVE[url]]}
    monkeypatch.setattr(registry, "_get_json", fake_get_json)
    return seen


@pytest.fixture
def no_network(monkeypatch):
    """Every network path raises: /models, the openai client, raw httpx."""
    def boom(*a, **k):
        raise AssertionError("network I/O attempted")
    monkeypatch.setattr(registry, "_get_json", boom)
    monkeypatch.setattr(llm_openai, "OpenAI", boom)
    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr(httpx.AsyncClient, "send", boom)


def _cfg(data: dict | None = None) -> EnginesConfig:
    return EnginesConfig.model_validate(copy.deepcopy(data or BASE))


def _write(tmp_path, name, data) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


@pytest.fixture
async def lib(tmp_path):
    lib = Library(tmp_path / "lib.sqlite")
    await lib.open()
    yield lib
    await lib.close()


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

async def test_resolution_fixed_id_and_preferences_against_live_list(fetches):
    snap = await registry.resolve_snapshot(_cfg(), "main")
    assert set(snap.roles) == set(ROLES) | set(LEGACY_ROLES)
    cheap = snap.roles["intake.classify"].primary
    assert (cheap.provider, cheap.model, cheap.model_from_preferences) == ("proxy", "c2", None), (
        "c1 is not live, so the first live preference (c2) is chosen")
    assert snap.roles["research.rerank"].primary.model == "k1", "a fixed id resolves to itself"
    legacy = snap.roles["legacy.strong"]
    assert (legacy.tier, legacy.primary.model, legacy.on_exhausted) == ("strong", "s1", "raise")
    assert snap.roles["research.digest"].on_exhausted == "skip_item"
    assert snap.roles["brief.draft"].on_exhausted == "raise", "unbound role: tier spec + raise"
    assert fetches == ["http://proxy/v1/models"], "one listing per provider; fixed ids need none"


async def test_unresolved_primary_names_role_and_provider(fetches, monkeypatch):
    monkeypatch.setitem(LIVE, "http://proxy/v1/models", ["c1", "s1"])     # no judge preference live
    with pytest.raises(EngineConfigError) as ei:
        await registry.resolve_snapshot(_cfg(), "main")
    msg = str(ei.value)
    assert msg.startswith("role panel.grounding:") and "proxy" in msg and "judge" in msg

    with pytest.raises(EngineConfigError) as ei:
        await registry.resolve_snapshot(_cfg(), "broken")
    assert str(ei.value) == ("role intake.classify: choose a model for provider local "
                             "(/engines models local, or the console)")


async def test_unresolved_fallback_is_dropped_with_a_warning(fetches):
    snap = await registry.resolve_snapshot(_cfg(), "main")
    fbs = snap.roles["panel.grounding"].fallbacks
    assert [(f.provider, f.model) for f in fbs] == [("fixed", "k2")]
    assert len(snap.warnings) == 1
    assert "panel.grounding" in snap.warnings[0] and "local" in snap.warnings[0]


async def test_validate_active_key_check_is_for_primaries_only(fetches, monkeypatch):
    monkeypatch.delenv("T_FIXED_KEY")
    with pytest.raises(EngineConfigError) as ei:          # research.rerank's primary is on fixed
        await registry.validate_active(_cfg())
    assert str(ei.value) == "role research.rerank: provider fixed: T_FIXED_KEY is not set"

    data = copy.deepcopy(BASE)
    del data["profiles"]["main"]["roles"]["research.rerank"]   # fixed is now only a fallback
    snap = await registry.validate_active(_cfg(data))
    assert snap.roles["panel.grounding"].fallbacks == []
    assert any("T_FIXED_KEY" in w for w in snap.warnings), snap.warnings


async def test_policy_only_binding_follows_a_tier_override_pinned_does_not(fetches):
    data = copy.deepcopy(BASE)
    data["overrides"] = {"main": {"tiers": {"cheap": {"provider": "fixed", "model": "k2"}}}}
    snap = await registry.resolve_snapshot(_cfg(data), "main")
    assert snap.roles["research.digest"].primary.model == "k2", "policy-only binding follows the tier"
    assert snap.roles["research.digest"].on_exhausted == "skip_item", "and keeps its policy"
    assert snap.roles["legacy.cheap"].primary.model == "k2"
    assert snap.roles["research.rerank"].primary.model == "k1", "a pinned primary does not move"


# ---------------------------------------------------------------------------
# static validation (startup): no I/O
# ---------------------------------------------------------------------------

def test_validate_schema_does_no_io_and_checks_preference_references(no_network):
    registry.validate_schema(_cfg())                      # valid → returns, nothing fetched
    data = copy.deepcopy(BASE)
    data["profiles"]["other"]["tiers"]["judge"] = {"provider": "fixed", "model_from_preferences": "judge"}
    with pytest.raises(EngineConfigError) as ei:
        registry.validate_schema(_cfg(data))
    assert "fixed" in str(ei.value) and "model_preferences" in str(ei.value)


def test_static_problems_names_missing_key_env_and_does_no_io(no_network, monkeypatch):
    assert registry.static_problems(_cfg()) == []
    monkeypatch.delenv("T_PROXY_KEY")
    problems = registry.static_problems(_cfg())
    assert problems == ["provider proxy: T_PROXY_KEY is not set (profile main)"]

    data = copy.deepcopy(BASE)
    data["profiles"]["main"]["roles"]["research.rerank"]["primary"]["model"] = "k9"
    monkeypatch.setenv("T_PROXY_KEY", "x")
    assert registry.static_problems(_cfg(data)) == [
        "role research.rerank: model k9 is not in provider fixed's known_models"]


# ---------------------------------------------------------------------------
# loading, merge order, overrides scoping
# ---------------------------------------------------------------------------

async def test_load_config_merge_order_and_per_profile_overrides(tmp_path, lib, fetches):
    local = {"active_profile": "other",
             "providers": {"proxy": {"timeout_s": 120}},
             "profiles": {"main": {"tiers": {"judge": {"provider": "fixed", "model": "k2"}}}}}
    await lib.set_setting("engines.active_profile", json.dumps("main"))
    await lib.set_setting("engines.overrides", json.dumps(
        {"main": {"tiers": {"cheap": {"provider": "fixed", "model": "k1"}}}}))
    cfg = await registry.load_config(
        [_write(tmp_path, "d.json", BASE), _write(tmp_path, "l.json", local)], library=lib)

    assert cfg.active_profile == "main", "library beats the local file, which beats the defaults"
    assert cfg.providers["proxy"].timeout_s == 120
    assert cfg.providers["proxy"].api_key_env == "T_PROXY_KEY", "a partial dict merges field-wise"
    assert cfg.profiles["main"].tiers["judge"] == EngineSpec(provider="fixed", model="k2"), (
        "an engine spec that names its provider replaces the old one whole")

    main = await registry.resolve_snapshot(cfg, "main")
    other = await registry.resolve_snapshot(cfg, "other")
    assert main.roles["legacy.cheap"].primary.model == "k1", "main's override applies to main"
    assert other.roles["legacy.cheap"].primary.model == "k1"
    assert other.roles["legacy.strong"].primary.model == "k2"
    assert main.roles["legacy.strong"].primary.model == "s1", "overrides are scoped per profile"


async def test_load_config_reports_bad_files(tmp_path):
    with pytest.raises(EngineConfigError, match="not found"):
        await registry.load_config([str(tmp_path / "missing.json")])
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(EngineConfigError, match="invalid JSON"):
        await registry.load_config([str(bad)])
    data = copy.deepcopy(BASE)
    data["active_profile"] = "nope"
    with pytest.raises(EngineConfigError, match="active_profile"):
        await registry.load_config([_write(tmp_path, "x.json", data)])


# ---------------------------------------------------------------------------
# active snapshot cache and persistence of /engines changes
# ---------------------------------------------------------------------------

def test_current_snapshot_cached_raises_when_nothing_resolved():
    with pytest.raises(EngineConfigError, match="active_snapshot"):
        registry.current_snapshot_cached()


async def test_active_snapshot_caches_and_set_calls_validate_before_persisting(tmp_path, lib, fetches):
    await registry.load_config([_write(tmp_path, "d.json", BASE)], library=lib)
    snap = await registry.active_snapshot()
    assert await registry.active_snapshot() is snap
    assert registry.current_snapshot_cached() is snap
    assert fetches == ["http://proxy/v1/models"], "cached: resolved once"

    new = await registry.set_overrides("main", ProfileOverrides(
        tiers={"judge": EngineSpec(provider="fixed", model="k2")}))
    assert registry.current_snapshot_cached() is new
    assert new.roles["legacy.judge"].primary.model == "k2"
    stored = json.loads(await lib.get_setting("engines.overrides"))
    assert stored["main"]["tiers"]["judge"]["model"] == "k2"

    await registry.set_active_profile("other")
    assert registry.current_snapshot_cached().profile == "other"
    assert json.loads(await lib.get_setting("engines.active_profile")) == "other"

    for refused in ("nope", "broken"):
        with pytest.raises(EngineConfigError):
            await registry.set_active_profile(refused)
        assert json.loads(await lib.get_setting("engines.active_profile")) == "other", (
            "a refused switch persists nothing")
        assert registry.current_snapshot_cached().profile == "other", "and changes nothing"


async def test_set_transcriber_persists_and_keeps_the_snapshot(tmp_path, lib, fetches):
    from argus.engines.schema import TranscriberConfig
    await registry.load_config([_write(tmp_path, "d.json", BASE)], library=lib)
    snap = await registry.active_snapshot()
    await registry.set_transcriber(TranscriberConfig(asr_engine="openai_audio"))
    assert registry.current_config().transcriber.asr_engine == "openai_audio"
    assert json.loads(await lib.get_setting("engines.transcriber"))["asr_engine"] == "openai_audio"
    assert registry.current_snapshot_cached() is snap


async def test_live_models_ttl_cache_known_models_and_no_key_in_errors(tmp_path, fetches, monkeypatch):
    await registry.load_config([_write(tmp_path, "d.json", BASE)])
    assert registry.live_models("proxy") == sorted(LIVE["http://proxy/v1/models"])
    registry.live_models("proxy")
    assert fetches == ["http://proxy/v1/models"], "TTL cache"
    assert registry.live_models("fixed") == ["k1", "k2"], "known_models, no network"

    def refused(url, headers):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(registry, "_get_json", refused)
    monkeypatch.setattr(registry, "_models_cache", {})
    with pytest.raises(EngineConfigError) as ei:
        registry.live_models("proxy")
    assert "cannot list models" in str(ei.value) and "sk-proxy-secret" not in str(ei.value)


# ---------------------------------------------------------------------------
# the shipped argus_engines.json
# ---------------------------------------------------------------------------

async def _shipped() -> EnginesConfig:
    return await registry.load_config([str(registry._ROOT / "argus_engines.json")])


async def test_shipped_engines_file_validates_without_io(no_network):
    cfg = await _shipped()
    registry.validate_schema(cfg)
    assert cfg.active_profile == "free-proxy"
    fp = cfg.providers["freellmapi"]
    assert (fp.base_url_env, fp.api_key_env, fp.token_param, fp.rpm_limit, fp.models_endpoint,
            fp.supports_json_mode) == ("FREELLMAPI_BASE_URL", "FREELLMAPI_API_KEY",
                                       "max_completion_tokens", 100, True, False)
    for tier, prefs in fp.model_preferences.items():
        assert prefs and prefs[-1] == "auto" and prefs.count("auto") == 1, tier
    assert all(spec == EngineSpec(provider="freellmapi", model_from_preferences=tier)
               for tier, spec in cfg.profiles["free-proxy"].tiers.items())
    assert cfg.providers["openai"].token_param == "max_completion_tokens"
    an = cfg.providers["anthropic"]
    assert (an.models_endpoint, an.supports_temperature, an.supports_json_mode, an.timeout_s) == (
        False, False, False, 300)
    assert an.known_models == ["claude-fable-5-1", "claude-opus-5-5", "claude-sonnet-5-5",
                               "claude-haiku-5-5"]
    assert cfg.pricing.exa is not None and cfg.pricing.exa.prices_as_of
    assert set(cfg.providers) == {"freellmapi", "groq", "openrouter", "gemini", "openai",
                                  "anthropic", "ollama", "lmstudio"}
    assert set(cfg.profiles) == {"free-proxy", "proxy+direct-judge", "max-quality", "local-ollama",
                                 "local-lmstudio", "hybrid-local"}


async def test_shipped_profiles_declare_the_v3_failure_policy(no_network):
    cfg = await _shipped()
    expected = {r: ("degrade" if r in DEGRADABLE_ROLES else "skip_item" if r in SKIPPABLE_ROLES
                    else "raise") for r in ROLES}
    for name in cfg.profiles:
        bindings = registry.bindings(cfg, name)
        got = {r: bindings[r][1].on_exhausted for r in ROLES}
        assert got == expected, name


async def test_activating_local_ollama_raises_choose_a_model(lib, no_network):
    await registry.load_config([str(registry._ROOT / "argus_engines.json")], library=lib)
    with pytest.raises(EngineConfigError) as ei:
        await registry.set_active_profile("local-ollama")
    assert "choose a model for provider ollama" in str(ei.value)
    assert await lib.get_setting("engines.active_profile") is None, "refused: nothing persisted"


# ---------------------------------------------------------------------------
# family() and summarize_calls()
# ---------------------------------------------------------------------------

def test_family_prefix_table():
    """Path prefixes are stripped, then the prefix table applies (spec §2.3,
    the table of the original WP-A text); else the first token."""
    f = registry.family
    assert f("qwen/qwen3-coder:free") == "qwen"
    assert f("openai/gpt-oss-120b:free") == f("gpt-4.1") == f("o3-mini") == "openai"
    assert f("@cf/meta/llama-3.3-70b") == f("Meta-Llama-3.3-70B-Instruct") == "meta"
    assert f("meta-llama/llama-3.1-70b-instruct") == f("llama-3.1-8b-instant") == "meta"
    assert f("gemini-3.5-flash") == f("google/gemma-3-27b") == "google"
    assert f("claude-opus-5-5") == f("claude-haiku-5-5") == "anthropic"
    assert f("DeepSeek-V3.2") == f("deepseek-ai/deepseek-v4-flash") == "deepseek"
    assert f("mistral-large-latest") == f("codestral-2508") == "mistral"
    assert f("nvidia/nemotron-3-super-120b-a12b:free") == "nvidia"
    assert f("z-ai/glm-4.6") == "zhipu" and f("moonshotai/kimi-k2") == "moonshot"
    assert f("cogito-2.1:671b") == "deepcogito" and f("command-r-plus") == "cohere"
    assert f("groq/compound-mini") == "compound", "no table entry: first token"
    assert f("auto") == "auto"


def _rec(role, **kw):
    base = dict(role=role, tier=role.split(".")[-1], profile="p", provider="proxy",
                requested_model="req", served_model="openai/gpt-oss-120b", routed_via="groq/x",
                served_provider="groq", attempt=1, fallback_rank=0, ok=True, error="",
                latency_ms=100, prompt_tokens=1, completion_tokens=1, reasoning_tokens=0,
                finish_reason="stop", json_mode=False, retry_reason="", item="", cached=False,
                extra={})
    base.update(kw)
    return base


def test_summarize_calls():
    failed = _rec("legacy.strong", ok=False, error="RateLimitError", served_model="", latency_ms=5)
    calls = [
        _rec("legacy.strong", attempt=2, retry_reason="transient", extra={"prior_attempts": [failed]}),
        _rec("legacy.judge", fallback_rank=1),
        _rec("legacy.cheap", served_model="req"),                   # served what was asked
        {"tier": "cheap", "requested_model": "cheap", "served_model": "llama-x",   # a v3 record
         "served_provider": "", "prompt_tokens": 3, "completion_tokens": 4, "extra": {}},
    ]
    errors = ["skip: research.digest https://a.example (429)",
              "degrade: scout.querygen → subquestion_text_queries (unparseable)",
              "compose: section 'x' failed (boom)"]
    s = registry.summarize_calls(calls, errors, exa_usd=0.12)

    strong = s["by_role"]["legacy.strong"]
    assert (strong["calls"], strong["ok"], strong["errors"]) == (2, 1, 1)
    assert strong["retries"] == {"transient": 1}
    assert strong["served_models"] == {"openai/gpt-oss-120b": 1}
    assert strong["avg_latency_ms"] == 52.5
    assert s["by_role"]["legacy.judge"]["fallbacks"] == 1
    assert s["by_role"]["legacy.cheap"]["served_models"] == {"req": 1, "llama-x": 1}, (
        "a v3 record (no role) counts as legacy.<tier>")
    assert s["by_role"]["research.digest"]["skipped"] == 1
    assert s["by_role"]["scout.querygen"]["degraded"] == 1
    assert s["degradations"] == ["degrade: scout.querygen → subquestion_text_queries (unparseable)"]
    assert s["reroute_rate"] == round(3 / 4, 3)
    assert s["judge_families"] == ["openai"] and s["writer_families"] == ["openai"]
    assert s["judge_writer_same_family"] is True
    assert s["warnings"] and "openai" in s["warnings"][0]
    assert s["exa_usd"] == 0.12
    assert s["calls"] == 5

    empty = registry.summarize_calls([])
    assert empty["reroute_rate"] is None and empty["judge_writer_same_family"] is None
    assert "exa_usd" not in empty
