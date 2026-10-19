"""Shared test helpers (v4 WP-0).

Plain functions, not pytest fixtures — import them where needed::

    from helpers import fake_engines_snapshot, make_run_config
    state = {"run_config": make_run_config("medium", **{"research.waves": 1})}
"""
from argus.engines.roles import (
    DEGRADABLE_ROLES, LEGACY_ROLES, ROLE_TIERS, ROLES, SKIPPABLE_ROLES)
from argus.engines.schema import EngineSpec, EnginesSnapshot, ResolvedBinding
from argus.graph.run_config import RunConfig


def fake_engines_snapshot() -> EnginesSnapshot:
    """Every role in ROLES plus the legacy.* ids bound to provider "fake",
    with the shipped failure policy of spec §2.2 (degrade / skip_item / raise)."""
    roles = {}
    for role in ROLES + LEGACY_ROLES:
        policy = ("degrade" if role in DEGRADABLE_ROLES
                  else "skip_item" if role in SKIPPABLE_ROLES else "raise")
        roles[role] = ResolvedBinding(
            role=role, tier=ROLE_TIERS.get(role) or role.removeprefix("legacy."),
            primary=EngineSpec(provider="fake", model="fake-model"),
            fallbacks=[], on_exhausted=policy, retry_on_truncation=True)
    return EnginesSnapshot(profile="fake", resolved_at="2026-10-19T00:00:00+00:00",
                           roles=roles)


def request_url(request) -> str:
    """The URL an httpx request is FOR. The fetch chain sends every hop to the
    address its SSRF guard checked, with the original ``Host`` header (spec
    §2.8), so ``request.url`` carries that address, not the name."""
    return f"{request.url.scheme}://{request.headers['host']}{request.url.raw_path.decode('ascii')}"


def as_stream(response):
    """``response`` as a real transport hands it over: unread. httpx pre-reads
    (and decodes) a Response built from bytes, while the fetch chain reads the
    raw stream itself — so build an encoded body with ``content=iter([...])``."""
    import httpx
    if not response.is_stream_consumed:
        return response
    assert "content-encoding" not in response.headers, "build an encoded body with content=iter([...])"
    return httpx.Response(response.status_code, headers=response.headers, content=iter([response.content]))


def make_run_config(length="short", depth=None, **overrides) -> dict:
    """A ``state["run_config"]`` dict from the presets; overrides are dotted
    keys (``**{"research.waves": 1}``)."""
    return RunConfig.from_preset(length=length, depth=depth,
                                 engines=fake_engines_snapshot(),
                                 overrides=overrides, origin="telegram").model_dump()
