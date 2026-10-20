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


def fake_llm(monkeypatch, reply):
    """The real ``llm.complete`` on a fake driver (no network): every attempt
    answers ``reply(system, human)`` — the text of its first two messages — and
    the returned list records each attempt's message texts (a validate retry
    adds the role's nudge as a third: the JSON object, or the markdown text)."""
    from argus import llm
    from argus.engines import registry
    from argus.engines.llm_openai import DriverResponse
    from argus.engines.schema import ProviderSpec
    seen: list[list[str]] = []

    class _Driver:
        def complete(self, engine, provider, messages, **kw):
            seen.append([m.content for m in messages])
            return DriverResponse(text=reply(messages[0].content, messages[1].content),
                                  served_model="fake-model", routed_via="", finish_reason="stop",
                                  usage={}, headers={}, raw=None)

    monkeypatch.setattr(registry, "provider", lambda name: ProviderSpec(name=name, base_url="http://fake/v1"))
    monkeypatch.setattr(llm, "_driver", _Driver())
    return seen


async def settle_runs(timeout: float = 10.0) -> None:
    """Await every drive the run driver still holds (``bot/runs.py``), so a test
    ends with no task pending; a drive still running after ``timeout`` is
    cancelled and the test fails."""
    import asyncio
    from argus.bot import common
    tasks = [h.task for h in list(common._inflight.values()) if getattr(h, "task", None) is not None]
    if not tasks:
        return
    _, pending = await asyncio.wait(tasks, timeout=timeout)
    for task in pending:
        task.cancel()
    assert not pending, f"drives still running after {timeout} s: {[t.get_name() for t in pending]}"


def make_run_config(length="short", depth=None, **overrides) -> dict:
    """A ``state["run_config"]`` dict from the presets; overrides are dotted
    keys (``**{"research.waves": 1}``)."""
    return RunConfig.from_preset(length=length, depth=depth,
                                 engines=fake_engines_snapshot(),
                                 overrides=overrides, origin="telegram").model_dump()
