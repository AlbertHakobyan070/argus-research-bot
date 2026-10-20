"""/engines and the bot side of the engines registry (v4 WP-A):

- the handlers on fake Updates (show, use, set, tier — which also fixes the
  legacy.* bindings — fallback, reset, models);
- wiring: engines_cmd is registered through HANDLER_MODULES, and a raising
  static-validation path at startup is logged and DM'd, never fatal, with no
  network I/O on the startup path;
- an EngineConfigError from a drive entry reaches the chat as its text;
- every drive entry awaits registry.active_snapshot() before driving, so a
  parked run resumed after a restart never meets an empty cache."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from telegram.ext import ApplicationHandlerStop

from argus import llm
from argus.bot import app as bot_app
from argus.bot import common as bot_common
from argus.bot import engines_cmd
from argus.bot import history as bot_history
from argus.bot import research as bot_research
from argus.bot import runs
from argus.config import Settings
from argus.engines import llm_openai, registry
from argus.engines.registry import EngineConfigError
from argus.library import Library

from helpers import fake_engines_snapshot, make_run_config, settle_runs

CFG: dict = {
    "active_profile": "main",
    "providers": {
        "proxy": {"name": "proxy", "base_url": "http://proxy/v1", "api_key_env": "T_PROXY_KEY",
                  "model_preferences": {"cheap": ["c1", "auto"], "strong": ["s1", "auto"],
                                        "judge": ["j1", "auto"]}},
        "fixed": {"name": "fixed", "base_url": "http://fixed/v1", "api_key_env": "T_FIXED_KEY",
                  "models_endpoint": False, "known_models": ["k1", "k2"]},
        "gem": {"name": "gem", "base_url": "http://gem/v1", "api_key_env": "T_GEM_KEY"},
        "local": {"name": "local", "base_url": "http://127.0.0.1:9/v1"},
    },
    "profiles": {
        "main": {
            "name": "main",
            "tiers": {"cheap": {"provider": "proxy", "model_from_preferences": "cheap"},
                      "strong": {"provider": "proxy", "model_from_preferences": "strong"},
                      "judge": {"provider": "gem", "model": None}},
            "roles": {"panel.grounding": {"primary": {"provider": "gem", "model": None},
                                          "fallbacks": [{"provider": "fixed", "model": "k2"}],
                                          "on_exhausted": "skip_item"}},
        },
        "other": {"name": "other",
                  "tiers": {t: {"provider": "fixed", "model": "k1"} for t in ("cheap", "strong", "judge")}},
        "broken": {"name": "broken",
                   "tiers": {t: {"provider": "local", "model": None} for t in ("cheap", "strong", "judge")}},
    },
}
LIVE = {"http://proxy/v1/models": ["c1", "s1", "auto"], "http://gem/v1/models": ["g1", "g2"]}


def _settings(tmp_path: Path, **kw) -> Settings:
    base = dict(freellmapi_base_url="http://127.0.0.1:9/v1", freellmapi_api_key="",
                telegram_bot_token="123:test", telegram_allowed_user_id=42,
                reports_root=tmp_path / "reports", checkpoint_db=tmp_path / "cp.sqlite",
                library_db=tmp_path / "lib.sqlite", vault_root=tmp_path / "vault",
                media_root=tmp_path / "vault/media", transcripts_root=tmp_path / "vault/transcripts",
                history_root=tmp_path / "vault/history", ffmpeg_path=None)
    base.update(kw)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(registry, "_state", registry._State())
    monkeypatch.setattr(registry, "_models_cache", {})
    for k in ("T_PROXY_KEY", "T_FIXED_KEY", "T_GEM_KEY"):
        monkeypatch.setenv(k, "sk-test")
    bot_common._inflight.clear()
    bot_common._pending_reply.clear()
    yield
    bot_common._inflight.clear()
    bot_common._pending_reply.clear()


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network I/O attempted")
    monkeypatch.setattr(registry, "_get_json", boom)
    monkeypatch.setattr(llm_openai, "OpenAI", boom)
    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setattr(httpx.AsyncClient, "send", boom)


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(registry, "_get_json",
                        lambda url, headers: {"data": [{"id": i} for i in LIVE[url]]})


@pytest.fixture
async def lib(tmp_path):
    lib = Library(tmp_path / "lib.sqlite")
    await lib.open()
    yield lib
    await lib.close()


@pytest.fixture
async def loaded(tmp_path, lib, live, monkeypatch):
    p = tmp_path / "engines.json"
    p.write_text(json.dumps(CFG), encoding="utf-8")
    await registry.load_config([str(p)], library=lib)
    monkeypatch.setattr(engines_cmd, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(engines_cmd, "_allowed", lambda _s, _uid: True)
    return lib


def _update(chat_id: int = 1) -> MagicMock:
    u = MagicMock()
    u.effective_user.id = chat_id
    u.effective_chat.id = chat_id
    u.message.reply_text = AsyncMock()
    return u


def _ctx(args, lib=None, **bot_data) -> MagicMock:
    application = MagicMock()
    application.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    application.bot.edit_message_text = AsyncMock()
    application.bot.send_chat_action = AsyncMock()
    application.bot_data = {"library": lib, **bot_data}
    ctx = MagicMock()
    ctx.application = application
    ctx.bot = application.bot
    ctx.args = list(args)
    return ctx


async def _engines(lib, *args) -> str:
    u = _update()
    await engines_cmd.engines_cmd(u, _ctx(args, lib))
    u.message.reply_text.assert_awaited_once()
    return u.message.reply_text.await_args.args[0]


# ---------------------------------------------------------------------------
# /engines handlers
# ---------------------------------------------------------------------------

async def test_engines_tier_fixes_the_legacy_bindings(loaded):
    text = await _engines(loaded)
    assert "main" in text and "❌" in text and "choose a model for provider gem" in text

    text = await _engines(loaded, "tier", "judge", "gem/g2")
    assert "✅" in text, text
    snap = registry.current_snapshot_cached()
    assert snap.roles["legacy.judge"].primary.model == "g2", "/engines tier fixes legacy.judge"
    pg = snap.roles["panel.grounding"]
    assert pg.primary.model == "g2" and [f.model for f in pg.fallbacks] == ["k2"], (
        "a policy-only binding follows the tier and keeps its fallbacks")
    stored = json.loads(await loaded.get_setting("engines.overrides"))
    assert stored["main"]["tiers"]["judge"]["model"] == "g2"
    shown = await _engines(loaded)
    assert "✅" in shown and "legacy.judge: gem/g2" in shown


async def test_engines_use_switches_and_refuses_an_invalid_profile(loaded):
    assert "✅" in await _engines(loaded, "use", "other")
    assert registry.current_snapshot_cached().profile == "other"
    assert json.loads(await loaded.get_setting("engines.active_profile")) == "other"

    text = await _engines(loaded, "use", "broken")
    assert "choose a model for provider local" in text
    assert registry.current_snapshot_cached().profile == "other", "the switch is refused"
    assert json.loads(await loaded.get_setting("engines.active_profile")) == "other"
    assert "unknown profile" in await _engines(loaded, "use", "nope")


async def test_engines_set_fallback_reset_and_models(loaded):
    await _engines(loaded, "use", "other")
    assert "✅" in await _engines(loaded, "set", "research.rerank", "fixed/k2")
    b = registry.current_snapshot_cached().roles["research.rerank"]
    assert b.primary.model == "k2"
    stored = json.loads(await loaded.get_setting("engines.overrides"))
    assert stored["other"]["roles"]["research.rerank"]["primary"]["model"] == "k2"

    assert "✅" in await _engines(loaded, "fallback", "research.rerank", "fixed/k1")
    b = registry.current_snapshot_cached().roles["research.rerank"]
    assert (b.primary.model, [f.model for f in b.fallbacks]) == ("k2", ["k1"])

    assert "✅" in await _engines(loaded, "reset")
    assert registry.current_snapshot_cached().roles["research.rerank"].primary.model == "k1"
    assert json.loads(await loaded.get_setting("engines.overrides")).get("other") in (None, {"tiers": {}, "roles": {}})

    models = await _engines(loaded, "models", "proxy")
    assert all(m in models for m in ("c1", "s1", "auto"))
    assert "k1" in await _engines(loaded, "models", "fixed")
    assert "provider/model" in await _engines(loaded, "set", "research.rerank", "nosuch")
    assert "unknown role" in await _engines(loaded, "set", "no.such", "fixed/k1")
    assert "/engines use <profile>" in await _engines(loaded, "bogus")


async def test_engines_show_last_run_tally_judge_diversity_and_exa(loaded, tmp_path):
    await _engines(loaded, "tier", "judge", "gem/g2")
    rep = tmp_path / "report"
    rep.mkdir()
    calls = [dict(role=r, tier=r.split(".")[1], profile="main", provider="proxy",
                  requested_model="s1", served_model="openai/gpt-oss-120b", routed_via="groq/x",
                  ok=True) for r in ("legacy.strong", "legacy.judge")]
    (rep / "metadata.json").write_text(json.dumps({"model_calls": calls}), encoding="utf-8")
    await loaded.create_run(run_id="ab12cd34", thread_id="tg:1:ab12cd34", chat_id=1, topic="t",
                            config_json="{}", depth="quick", origin="telegram")
    await loaded.set_run_status("ab12cd34", "done", report_dir=str(rep))
    await loaded.update_run("ab12cd34", exa_usd=0.12)

    text = await _engines(loaded)
    assert "Last run ab12cd34" in text and "openai/gpt-oss-120b ×2" in text
    assert "judge diversity ⚠" in text and "/engines tier judge" in text
    assert "Exa this month: $0.12" in text


async def test_engines_is_refused_for_strangers(loaded, monkeypatch):
    monkeypatch.setattr(engines_cmd, "_allowed", lambda _s, _uid: False)
    u = _update()
    await engines_cmd.engines_cmd(u, _ctx(["use", "other"], loaded))
    u.message.reply_text.assert_awaited_once_with("Unauthorized.")
    assert registry.current_config().active_profile == "main"


# ---------------------------------------------------------------------------
# wiring and startup
# ---------------------------------------------------------------------------

def test_engines_cmd_is_registered_via_handler_modules():
    assert engines_cmd in bot_app.HANDLER_MODULES
    fake = MagicMock()
    engines_cmd.register(fake)
    commands = [c.args[0].commands for c in fake.add_handler.call_args_list]
    assert frozenset({"engines"}) in commands
    fake.add_error_handler.assert_called_once_with(engines_cmd.on_error)


async def _post_init(monkeypatch, tmp_path, engines_text: str, local: bytes | None = None) -> MagicMock:
    f = tmp_path / "engines.json"
    f.write_text(engines_text, encoding="utf-8")
    local_file = tmp_path / "engines.local.json"
    if local is not None:
        local_file.write_bytes(local)
    monkeypatch.setenv("ARGUS_ENGINES_FILE", str(f))
    monkeypatch.setenv("ARGUS_ENGINES_LOCAL_FILE", str(local_file))
    monkeypatch.setattr(bot_app, "get_settings", lambda: _settings(tmp_path))
    app = MagicMock()
    app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    app.bot_data = {}
    try:
        await bot_app._post_init(app)
    finally:   # close the saver + library even if startup raised: open aiosqlite threads hang pytest's exit
        await bot_app._post_shutdown(app)
    return app


async def test_post_init_survives_a_broken_engines_file(monkeypatch, tmp_path, no_network, caplog):
    caplog.set_level(logging.WARNING, logger="argus.bot")
    app = await _post_init(monkeypatch, tmp_path, "{not json")
    assert "graph" in app.bot_data, "startup completed"
    app.bot.send_message.assert_awaited_once()
    kw = app.bot.send_message.await_args.kwargs
    assert kw["chat_id"] == 42 and "invalid JSON" in kw["text"]
    assert any("engines:" in r.getMessage() and "invalid JSON" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("encoding", ["cp1252", "utf-16"])
async def test_post_init_survives_a_local_engines_file_that_is_not_utf8(
        monkeypatch, tmp_path, no_network, caplog, encoding):
    caplog.set_level(logging.WARNING, logger="argus.bot")
    local = json.dumps({"profiles": {"main": {"description": "proxy — local"}}}, ensure_ascii=False)
    app = await _post_init(monkeypatch, tmp_path, json.dumps(CFG), local=local.encode(encoding))
    assert "graph" in app.bot_data, "startup completed"
    app.bot.send_message.assert_awaited_once()
    text = app.bot.send_message.await_args.kwargs["text"]
    assert "engines.local.json" in text and "save it as UTF-8" in text
    assert any("engines:" in r.getMessage() and "engines.local.json" in r.getMessage()
               for r in caplog.records)


async def test_post_init_survives_a_local_engines_file_it_cannot_stat(monkeypatch, tmp_path, no_network):
    """Re-review residual: a PermissionError from the stat is a startup
    problem like the others, with no encoding advice."""
    real_is_file = Path.is_file

    def is_file(self):
        if self.name == "engines.local.json":
            raise PermissionError(13, "Access is denied", str(self))
        return real_is_file(self)
    monkeypatch.setattr(Path, "is_file", is_file)
    app = await _post_init(monkeypatch, tmp_path, json.dumps(CFG))
    assert "graph" in app.bot_data, "startup completed"
    app.bot.send_message.assert_awaited_once()
    text = app.bot.send_message.await_args.kwargs["text"]
    assert "engines.local.json" in text and "cannot be read (PermissionError:" in text
    assert "UTF-8" not in text


async def test_post_init_dms_static_problems_once_and_warns_on_removed_knobs(
        monkeypatch, tmp_path, no_network, caplog):
    caplog.set_level(logging.WARNING, logger="argus.bot")
    monkeypatch.delenv("T_PROXY_KEY")
    monkeypatch.setenv("ARGUS_TIMEOUT", "90")
    monkeypatch.setenv("ARGUS_LLM_RETRIES", "5")
    app = await _post_init(monkeypatch, tmp_path, json.dumps(CFG))
    assert "graph" in app.bot_data
    app.bot.send_message.assert_awaited_once()
    assert "T_PROXY_KEY is not set" in app.bot.send_message.await_args.kwargs["text"]
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "ARGUS_TIMEOUT" in logged and "ARGUS_LLM_RETRIES" in logged


async def test_post_init_sends_nothing_when_the_config_is_clean(monkeypatch, tmp_path, no_network):
    app = await _post_init(monkeypatch, tmp_path, json.dumps(CFG))
    app.bot.send_message.assert_not_awaited()
    assert registry.current_config().active_profile == "main", "loaded at startup, not resolved"
    with pytest.raises(EngineConfigError):
        registry.current_snapshot_cached()


async def test_engine_config_error_reaches_the_chat_and_stops_other_handlers():
    ctx = MagicMock()
    ctx.bot.send_message = AsyncMock()
    ctx.error = EngineConfigError("role intake.classify: provider freellmapi: FREELLMAPI_API_KEY is not set")
    update = MagicMock()
    update.effective_chat.id = 7
    with pytest.raises(ApplicationHandlerStop):
        await engines_cmd.on_error(update, ctx)
    kw = ctx.bot.send_message.await_args.kwargs
    assert kw["chat_id"] == 7 and kw["text"] == f"⚠️ {ctx.error}"

    ctx.bot.send_message.reset_mock()
    ctx.error = RuntimeError("unrelated")
    assert await engines_cmd.on_error(update, ctx) is None
    ctx.bot.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# drive entries: the snapshot is resolved before anything is driven
# ---------------------------------------------------------------------------

class _Graph:
    """Records the order of events; a node calls a legacy wrapper first,
    as every Phase 1 node does."""

    def __init__(self, order: list[str]):
        self.order = order
        self.aupdate_state = AsyncMock()
        self.driven = False
        self.nodes: dict = {}
        self.checkpointer = SimpleNamespace(aget_tuple=AsyncMock(return_value=None))

    def astream(self, inp, config=None, **_kw):
        self.driven = True

        async def gen():
            self.order.append("node:" + llm.resolve_tier("cheap"))
            yield "updates", {"scout": {"sources": []}}
        return gen()

    async def aget_state(self, cfg):
        # a v4 checkpoint: ensure_in_checkpoint finds its run_config and writes nothing
        return SimpleNamespace(values={"plan": {"summary": "s", "sub_questions": ["q"],
                                                "planned_sources": []}, "report_paths": {},
                                       "length": "short", "run_config": make_run_config("short")},
                               next=(), created_at="2026-10-19T00:00:00+00:00")


_HANDLERS = ("research_cmd", "ask_cmd", "continue_cmd")
_AT_GATE = {"approve": runs.PLAN, "plan_edit": runs.PLAN, "report_send": runs.REPORT,
            "report_extend": runs.REPORT, "report_revise": runs.REPORT}


async def _call_entry(name: str, graph, lib):
    """Run one drive entry with minimal fakes: a command handler (which
    relays a refusal to the chat; returns None) or a driver action on a run
    resting at its gate (returns its Outcome). Returns (outcome, ctx)."""
    if name in _HANDLERS:
        ctx = _ctx(["ab12"] if name == "continue_cmd" else ["topic"], lib, graph=graph,
                   quick_graph=graph)
        await runs.init(ctx.application)
        mod = bot_history if name == "continue_cmd" else bot_research
        await getattr(mod, name)(_update(5), ctx)
        return None, ctx
    ctx = _ctx([], lib, graph=graph)
    await runs.init(ctx.application)
    h = runs.RunHandle(run_id="ab12cd34", chat_id=5, thread_id="tg:5:ab12cd34", topic="t")
    h.gate_kind, h.choices = _AT_GATE[name], runs.Choices(length="short")
    bot_common._inflight[h.run_id] = h
    action = {"approve": lambda: runs.approve_plan(h.run_id),
              "plan_edit": lambda: runs.edit_plan(h.run_id, "feedback"),
              "report_send": lambda: runs.report_action(h.run_id, "send"),
              "report_extend": lambda: runs.report_action(h.run_id, "extend"),
              "report_revise": lambda: runs.report_action(h.run_id, "revise", feedback="feedback")}[name]
    return await action(), ctx


@pytest.fixture
async def entry_env(monkeypatch, tmp_path):
    for mod in (bot_research, bot_history):
        monkeypatch.setattr(mod, "get_settings", lambda: _settings(tmp_path))
        monkeypatch.setattr(mod, "_allowed", lambda _s, _uid: True)
    lib = MagicMock()
    lib.create_run = AsyncMock()
    lib.resolve_run = AsyncMock(return_value={"run_id": "ab12cd34", "thread_id": "tg:5:ab12cd34",
                                              "topic": "t", "length": "short"})
    lib.set_run_status = AsyncMock()
    lib.pending_sources = AsyncMock(return_value=[])
    bot_common._inflight.clear()
    yield lib
    await settle_runs()
    bot_common._inflight.clear()


@pytest.mark.parametrize("name", [*_HANDLERS, *_AT_GATE])
async def test_every_drive_entry_awaits_the_snapshot_before_driving(monkeypatch, entry_env, name):
    async def refuse():
        raise EngineConfigError("role intake.classify: provider freellmapi: cannot list models")
    monkeypatch.setattr(registry, "active_snapshot", refuse)
    graph = _Graph([])
    out, ctx = await _call_entry(name, graph, entry_env)
    assert not graph.driven and not graph.aupdate_state.await_count
    entry_env.create_run.assert_not_awaited()
    if out is None:                                  # a handler: the refusal reached the chat (pointer 4)
        (sent,) = ctx.application.bot.send_message.await_args_list
        assert "⚠️" in sent.kwargs["text"] and "cannot list models" in sent.kwargs["text"]
    else:                                            # the handler relays it; the run is as it was
        assert out.code == "refused" and "cannot list models" in out.text
        ctx.application.bot.send_message.assert_not_awaited()
        assert not bot_common._inflight["ab12cd34"].live


@pytest.mark.parametrize("name", ["research_cmd", "plan_edit", "approve"])
async def test_a_resumed_run_after_restart_resolves_before_the_wrappers_run(
        monkeypatch, entry_env, name):
    order: list[str] = []

    async def fill():
        order.append("active_snapshot")
        registry._state.snapshot = fake_engines_snapshot()
        return registry._state.snapshot
    monkeypatch.setattr(registry, "active_snapshot", fill)   # restart: the cache starts empty
    graph = _Graph(order)
    await _call_entry(name, graph, None)
    await settle_runs()
    assert order == ["active_snapshot", "node:fake-model"]
