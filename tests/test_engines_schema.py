"""v4 WP-0 — engines schema validators (spec §2.1-2.2)."""
import pytest

from argus.engines.roles import (
    DEGRADABLE_ROLES, JSON_ROLES, LEGACY_ROLES, MUST_RAISE_ROLES, ROLE_TIERS,
    ROLES, SKIPPABLE_ROLES)
from argus.engines.schema import (
    EnginesConfig, EnginesSnapshot, ProviderSpec, ResolvedBinding)
from conftest import fake_engines_snapshot

_TIER = {"provider": "freellmapi", "model_from_preferences": "cheap"}


def _config(**changes) -> dict:
    """Minimal valid EnginesConfig input; each test breaks one rule."""
    cfg = {
        "providers": {"freellmapi": {"name": "freellmapi",
                                     "base_url": "http://127.0.0.1:3001/v1"}},
        "profiles": {"free-proxy": {"name": "free-proxy",
                                    "tiers": {"cheap": _TIER, "strong": _TIER,
                                              "judge": _TIER}}},
        "active_profile": "free-proxy",
    }
    cfg.update(changes)
    return cfg


def _with_roles(roles: dict) -> dict:
    cfg = _config()
    cfg["profiles"]["free-proxy"]["roles"] = roles
    return cfg


def _binding(on_exhausted: str, **extra) -> dict:
    return {"primary": _TIER, "on_exhausted": on_exhausted, **extra}


def test_role_tables_are_consistent():
    assert set(ROLE_TIERS) == set(ROLES)
    assert JSON_ROLES <= set(ROLES)
    # The three failure-policy classes partition ROLES, so every role has
    # exactly one shipped on_exhausted policy.
    classes = [SKIPPABLE_ROLES, set(DEGRADABLE_ROLES), MUST_RAISE_ROLES]
    assert sum(len(c) for c in classes) == len(ROLES)
    assert set().union(*classes) == set(ROLES)
    assert not set(LEGACY_ROLES) & set(ROLES)


def test_minimal_config_and_allowed_policies_validate():
    cfg = EnginesConfig.model_validate(_with_roles({
        "research.digest": _binding("skip_item"),
        "intake.classify": _binding("degrade"),
        "brief.draft": _binding("raise"),
    }))
    assert cfg.active_profile == "free-proxy"
    assert cfg.transcriber.youtube == ["captions", "asr"]


@pytest.mark.parametrize("where", ["tier", "role primary", "role fallback",
                                   "override tier"])
def test_unknown_provider_rejected(where):
    bad = {"provider": "nope", "model": "m"}
    if where == "tier":
        cfg = _config()
        cfg["profiles"]["free-proxy"]["tiers"]["judge"] = bad
    elif where == "role primary":
        cfg = _with_roles({"panel.grounding": {"primary": bad,
                                               "on_exhausted": "skip_item"}})
    elif where == "role fallback":
        cfg = _with_roles({"panel.grounding": _binding("skip_item",
                                                       fallbacks=[bad])})
    else:
        cfg = _config(overrides={"free-proxy": {"tiers": {"cheap": bad}}})
    with pytest.raises(ValueError, match="unknown provider 'nope'"):
        EnginesConfig.model_validate(cfg)


def test_skip_item_on_non_skippable_role_rejected():
    with pytest.raises(ValueError, match="skip_item"):
        EnginesConfig.model_validate(_with_roles(
            {"intake.classify": _binding("skip_item")}))


def test_degrade_on_non_degradable_role_rejected():
    with pytest.raises(ValueError, match="degrade"):
        EnginesConfig.model_validate(_with_roles(
            {"research.digest": _binding("degrade")}))


def test_must_raise_role_with_skip_item_rejected():
    with pytest.raises(ValueError, match="must be 'raise'"):
        EnginesConfig.model_validate(_with_roles(
            {"brief.draft": _binding("skip_item")}))


def test_active_profile_missing_rejected():
    with pytest.raises(ValueError, match="active_profile 'gone'"):
        EnginesConfig.model_validate(_config(active_profile="gone"))


def test_overrides_for_unknown_profile_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        EnginesConfig.model_validate(_config(overrides={"gone": {}}))


@pytest.mark.parametrize("in_overrides", [False, True])
def test_unknown_role_rejected(in_overrides):
    roles = {"compose.editor": _binding("raise")}   # optional/later, not a role
    cfg = (_config(overrides={"free-proxy": {"roles": roles}}) if in_overrides
           else _with_roles(roles))
    with pytest.raises(ValueError, match="unknown role 'compose.editor'"):
        EnginesConfig.model_validate(cfg)


def test_model_preferences_keys_must_be_tiers():
    ProviderSpec(name="p", base_url="http://x",
                 model_preferences={"cheap": ["a", "auto"]})
    with pytest.raises(ValueError, match="model_preferences"):
        ProviderSpec(name="p", base_url="http://x",
                     model_preferences={"fast": ["a"]})


def test_resolved_binding_checks_role_and_policy():
    spec = {"provider": "fake", "model": "m"}
    ResolvedBinding(role="legacy.cheap", tier="cheap", primary=spec,
                    fallbacks=[], on_exhausted="raise", retry_on_truncation=True)
    for role, policy, match in [("nope.role", "raise", "unknown role"),
                                ("legacy.judge", "skip_item", "skip_item"),
                                ("panel.grounding", "ignore", "on_exhausted")]:
        with pytest.raises(ValueError, match=match):
            ResolvedBinding(role=role, tier="judge", primary=spec, fallbacks=[],
                            on_exhausted=policy, retry_on_truncation=True)


def test_snapshot_role_keys_must_match_bindings():
    snap = fake_engines_snapshot().model_dump()
    snap["roles"]["quick.answer"] = snap["roles"]["brief.draft"]
    with pytest.raises(ValueError, match="quick.answer"):
        EnginesSnapshot.model_validate(snap)


def test_fake_snapshot_binds_every_role_to_fake_provider():
    snap = fake_engines_snapshot()
    assert set(snap.roles) == set(ROLES) | set(LEGACY_ROLES)
    assert {b.primary.provider for b in snap.roles.values()} == {"fake"}
    assert snap.roles["legacy.strong"].tier == "strong"
    assert snap.roles["panel.grounding"].on_exhausted == "skip_item"
    assert snap.roles["compose.tldr"].on_exhausted == "degrade"
    assert snap.roles["compose.flat"].on_exhausted == "raise"
