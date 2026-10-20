"""The run driver (spec §2.5, §3 WP-E1): drain-based cancel, one live drive
per run, one executing run at a time, the shutdown drain, the progress card,
the plan knobs and the preview quality line.

The real Argus graph (gates, deliver, extend_prep, revise_prep and the
routers real; the LLM and network nodes stubbed) on a MemorySaver, a real
Library on tmp_path, the shipped engines config (no network: the active
snapshot is the test one); Telegram is a mock bot.
"""
from __future__ import annotations

import asyncio
import dataclasses
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from telegram import InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import BadRequest, RetryAfter

from argus.bot import app as bot_app
from argus.bot import common, history, progress, runs
from argus.bot import media as bot_media
from argus.bot import research as bot_research
from argus.bot.progress import RunEventBus
from argus.config import Settings
from argus.engines import registry, search
from argus.engines.registry import EngineConfigError
from argus.graph import graph as graph_mod
from argus.graph.nodes import MAX_REVISE_ROUNDS
from argus.graph.run_config import RunConfig, require
from argus.graph.state import ArgusState, ResearchBrief
from argus.library import Library
from helpers import fake_engines_snapshot, make_run_config, settle_runs

CHAT = 5
PLAN = {"summary": "s", "sub_questions": ["q"], "planned_sources": []}
BRIEF = {"sub_questions": [{"q": "q?", "kind": "web"}], "must_have_keywords": ["q"], "summary": "s",
         "archetype": "comparison", "time_sensitive": True, "recency_hint": "m"}
WINDOW = {"window": "m", "start_date": "2026-09-20"}
BANNERS = ("> ⚠ Not fact-checked — 2 of 9 findings were not judged (429)\n\n"
           "> ⚠ Degraded run — 14 of 28 sources could not be digested (429)\n\n")


class Research:
    """A research node whose own pool work blocks on ``release``; with
    ``reads_runtime`` it checks ``runtime.drain_requested`` between items and
    returns early with the ``cancelled:`` line, as graph/research.py does
    (``announces_drain``: also its own ``stage draining``, as research.py does)."""

    def __init__(self, *, block: bool = True, reads_runtime: bool = False, items: int = 3,
                 fail: Exception | None = None, announces_drain: bool = False) -> None:
        self.block, self.reads_runtime, self.items, self.fail = block, reads_runtime, items, fail
        self.announces_drain = announces_drain
        self.started, self.release = threading.Event(), threading.Event()
        self.calls = 0

    def __call__(self, state, runtime: Runtime) -> dict:
        self.calls += 1
        runtime.stream_writer({"kind": "stage", "node": "research",
                               "data": {"status": "start", "summary": "first pass"}})
        self.started.set()
        if self.fail is not None:
            raise self.fail
        out = {"fetched": [{"url": "https://w.example/x", "title": "W", "excerpt": "x"}], "evidence": [],
               "search_spend": {**(state.get("search_spend") or {}), "exa_usd": 0.042}}
        for done in range(self.items):
            if self.reads_runtime and runtime.drain_requested:
                if self.announces_drain:
                    runtime.stream_writer({"kind": "stage", "node": "research",
                                           "data": {"status": "draining", "summary": runtime.drain_reason}})
                return {**out, "errors": [f"cancelled: research stopped after {done}/{self.items} fetches in "
                                          f"wave 1 ({runtime.drain_reason})"]}
            if self.block:
                assert self.release.wait(10), "the test never released the node"
        return out


@pytest.fixture
async def env(monkeypatch, tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    (report / "report.md").write_text("# topic\n\n" + BANNERS + "## Findings\n\nX happened [1].\n",
                                      encoding="utf-8")
    state = SimpleNamespace(research=Research(block=False), gates=[])

    def research(s, runtime: Runtime):
        return state.research(s, runtime)

    monkeypatch.setattr(graph_mod, "intake_node", lambda s: {"mode": "deep"})
    monkeypatch.setattr(graph_mod, "brief_node", lambda s: {"plan": dict(PLAN), "brief": dict(BRIEF),
                                                            "plan_feedback": ""})
    monkeypatch.setattr(graph_mod, "scout_node", lambda s: {
        "sources": [{"kind": "web", "title": "W", "url": "https://w.example/x"}],
        "search_spend": {"exa_usd": 0.007, "recency": dict(WINDOW)}})
    monkeypatch.setattr(graph_mod, "research_node", research)
    monkeypatch.setattr(graph_mod, "outline_node", lambda s: {"outline": {"sections": []}})
    monkeypatch.setattr(graph_mod, "compose_node", lambda s: {"draft_md": "d", "findings": [], "sections": []})
    monkeypatch.setattr(graph_mod, "analyst_node", lambda s: {"analysis": None})
    monkeypatch.setattr(graph_mod, "panel_node", lambda s: {
        "panel_verdict": {"verdict": "pass", "judged": 7, "total": 9}, "review_verdict": {"verdict": "pass"}})
    monkeypatch.setattr(graph_mod, "route_after_panel", lambda s: "report_builder")
    monkeypatch.setattr(graph_mod, "report_builder_node", lambda s: {
        "report_paths": {"md": str(report / "report.md"), "pdf": None, "folder": str(report)},
        "findings": [{"id": "f0", "status": "cut"}, {"id": "f1", "status": "kept"}],
        "errors": ["[r1] skip: research.digest https://w.example/y (429)",
                   "[r1] degrade: research.rerank → keyword_triage (429)"]})
    graph = graph_mod.build_graph(checkpointer=MemorySaver())

    lib = Library(tmp_path / "lib.sqlite")
    await lib.open()
    ids = iter(range(100, 10_000))
    bot = MagicMock()
    bot.send_message = AsyncMock(side_effect=lambda **kw: SimpleNamespace(message_id=next(ids)))
    bot.edit_message_text = AsyncMock()
    bot.send_document = AsyncMock()
    bot.send_chat_action = AsyncMock()
    app = SimpleNamespace(bot=bot, bot_data={"graph": graph, "library": lib})

    settings = Settings(freellmapi_base_url="http://127.0.0.1:9/v1", freellmapi_api_key="k",
                        telegram_bot_token="123:test", telegram_allowed_user_id=CHAT,
                        reports_root=tmp_path, checkpoint_db=tmp_path / "cp.sqlite",
                        library_db=tmp_path / "lib.sqlite", vault_root=tmp_path, media_root=tmp_path,
                        transcripts_root=tmp_path, history_root=tmp_path, ffmpeg_path=None)
    monkeypatch.setattr(runs, "get_settings", lambda: settings)
    monkeypatch.setattr(runs, "bus", RunEventBus())
    monkeypatch.setattr(runs, "exa_enabled", lambda: True)
    monkeypatch.setattr(registry, "_state", registry._State())
    await registry.load_config()                                  # the shipped file; no network
    active = fake_engines_snapshot().model_copy(update={"profile": "free-proxy"})
    monkeypatch.setattr(registry, "active_snapshot", AsyncMock(return_value=active))
    registry._state.snapshot = active                            # what the real active_snapshot() caches
    for mod in (bot_app, bot_research, history):                 # the handlers' settings and ACL
        monkeypatch.setattr(mod, "get_settings", lambda: settings)
        monkeypatch.setattr(mod, "_allowed", lambda _s, _u: True)
    monkeypatch.setattr(runs, "_app", None)                      # restored after the test
    monkeypatch.setattr(runs, "_shutting_down", False, raising=False)
    monkeypatch.setattr(runs, "_launch_closed", False, raising=False)
    common._inflight.clear()
    common._pending_reply.clear()
    await runs.init(app)
    env = SimpleNamespace(app=app, bot=bot, lib=lib, graph=graph, settings=settings, state=state,
                          active=active, monkeypatch=monkeypatch, tmp_path=tmp_path)
    yield env
    for gate in state.gates:
        gate.release.set()
    await settle_runs()
    common._inflight.clear()
    common._pending_reply.clear()
    await lib.close()


def _blocking(env, **kw) -> Research:
    env.state.research = Research(**kw)
    env.state.gates.append(env.state.research)
    return env.state.research


def _sent(env) -> list[str]:
    return [str(c.kwargs.get("text", "")) for c in env.bot.send_message.await_args_list]


def _events(run_id: str, kind: str | None = None) -> list:
    return [e for e in runs.bus.since(run_id, 0, limit=500) if kind is None or e.kind == kind]


async def _at_plan_gate(env, topic: str = "topic") -> str:
    out = await runs.start_run(chat_id=CHAT, user_id=CHAT, topic=topic, length="short")
    assert out.code == "started", out
    await common._inflight[out.run_id].task
    assert (await env.lib.get_run(out.run_id))["status"] == "awaiting_plan"
    return out.run_id


async def _executing(env, research: Research, topic: str = "topic") -> str:
    """A run approved past its plan gate, its research node now busy."""
    run_id = await _at_plan_gate(env, topic)
    assert (await runs.approve_plan(run_id)).code == "started"
    assert await asyncio.to_thread(research.started.wait, 5)
    return run_id


async def _snap(env, run_id: str):
    return await env.graph.aget_state(common._inflight[run_id].cfg if run_id in common._inflight
                                      else {"configurable": {"thread_id": f"tg:{CHAT}:{run_id}"}})


async def _at_report_gate(env, topic: str = "topic") -> str:
    run_id = await _at_plan_gate(env, topic)
    assert (await runs.approve_plan(run_id)).code == "started"
    await common._inflight[run_id].task
    assert (await env.lib.get_run(run_id))["status"] == "awaiting_report"
    return run_id


_TAPPED = 900          # the message id of a card the user taps (one minted before a restart)


def _ctx(env, args=()) -> SimpleNamespace:
    return SimpleNamespace(application=env.app, args=list(args))


def _update(text: str = "") -> MagicMock:
    u = MagicMock()
    u.effective_user.id = u.effective_chat.id = CHAT
    u.message.text = text
    u.message.reply_text = AsyncMock()
    return u


async def _tap(env, data: str, message_id: int = _TAPPED) -> MagicMock:
    """A button tap on message ``message_id`` through ``app.on_callback``."""
    q = MagicMock()
    q.data, q.answer, q.edit_message_text = data, AsyncMock(), AsyncMock()
    q.message.message_id = message_id
    update = _update()
    update.callback_query = q
    await bot_app.on_callback(update, _ctx(env))
    return update


async def _reply(env, text: str) -> None:
    """A chat text through ``app.on_text`` (the pending HITL reply, else the media branch)."""
    await bot_app.on_text(_update(text), _ctx(env))


def _edits(env, message_id: int) -> list[dict]:
    return [c.kwargs for c in env.bot.edit_message_text.await_args_list if c.kwargs.get("message_id") == message_id]


def _edited(env) -> list[str]:
    return [str(c.kwargs.get("text", "")) for c in env.bot.edit_message_text.await_args_list]


async def _source_statuses(env, run_id: str) -> dict[str, str]:
    async with env.lib._conn().execute("SELECT ref, status FROM run_sources WHERE run_id = ?", (run_id,)) as cur:
        return {ref: status for ref, status in await cur.fetchall()}


class FetchAll:
    """A research node that fetches every source it is given; records each
    pass's ``append_only`` flag and source URLs."""

    def __init__(self) -> None:
        self.seen: list[tuple] = []

    def __call__(self, state, runtime: Runtime) -> dict:
        sources = list(state.get("sources") or [])
        self.seen.append((state.get("append_only"), [s.get("url") for s in sources]))
        return {"fetched": [{"url": s.get("url"), "title": s.get("title") or "", "excerpt": "x"} for s in sources],
                "evidence": [], "append_only": False}


# ---------------------------------------------------------------------------
# Review Focus 5 — cancel while a node is busy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reads_runtime", [False, True], ids=["node_ignores_runtime", "research_pattern"])
async def test_cancel_while_node_busy(env, reads_runtime):
    research = _blocking(env, reads_runtime=reads_runtime)
    run_id = await _executing(env, research)
    h = common._inflight[run_id]
    assert h.stage == "research"

    out = await runs.cancel_run(run_id)
    assert out.code == "draining"
    assert h.control.drain_requested and h.control.drain_reason == "user cancel"
    assert not h.task.done()                                   # the node is still busy
    await h.card.refresh()
    assert "cancelling — finishing the current step" in h.card.text

    research.release.set()
    await asyncio.wait([h.task], timeout=10)
    assert h.task.done() and not h.task.cancelled() and h.task.exception() is None
    assert [e.data["status"] for e in _events(run_id, "stage")][-1] == "cancelled"   # GraphDrained caught

    assert (await env.lib.get_run(run_id))["status"] == "cancelled"
    snap = await _snap(env, run_id)
    assert snap.values["cancel_note"] == "cancelled after node research (user cancel)"
    assert tuple(snap.next) == ("outline",)                    # the node after the one that finished
    cancelled = [e for e in snap.values.get("errors") or [] if e.startswith("cancelled:")]
    assert cancelled == (["cancelled: research stopped after 1/3 fetches in wave 1 (user cancel)"]
                         if reads_runtime else [])
    assert run_id not in common._inflight
    assert any("Cancelled after research" in t and f"/continue {run_id}" in t
               for t in [c.kwargs.get("text", "") for c in env.bot.edit_message_text.await_args_list])

    # /continue resumes from the checkpoint and completes the run
    out = await runs.continue_run(await env.lib.get_run(run_id), chat_id=CHAT)
    assert out.code == "started" and "outline" in out.text
    assert (await env.lib.get_run(run_id))["status"] == "running"
    await common._inflight[run_id].task
    assert (await env.lib.get_run(run_id))["status"] == "awaiting_report"
    assert tuple((await _snap(env, run_id)).next) == ("deliver",)
    assert research.calls == 1                                 # research was not run again


async def test_cancel_falls_back_to_task_cancel_after_the_drain_timeout(env, monkeypatch, caplog):
    waited: list[float] = []

    async def clock_past_the_deadline(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(runs, "_sleep", clock_past_the_deadline)
    research = _blocking(env)                                  # never reads runtime: the drain cannot finish
    run_id = await _executing(env, research)
    h = common._inflight[run_id]
    with caplog.at_level("WARNING", logger="argus.bot"):
        assert (await runs.cancel_run(run_id)).code == "draining"
        await asyncio.wait([h.task], timeout=10)
    assert h.task.cancelled()
    assert waited == [300.0]                                   # ARGUS_CANCEL_DRAIN_TIMEOUT_S default
    assert f"run {run_id}: the drain did not finish within 300 s" in caplog.text
    assert (await env.lib.get_run(run_id))["status"] == "cancelled"
    # R-c: the forced cancel writes its own cancel_note; research never wrote, so the
    # run rests at ("research",) — the plan gate — and the card says what /continue does
    snap = await _snap(env, run_id)
    assert snap.values["cancel_note"] == "force-cancelled during node research (user cancel); the node was abandoned"
    assert tuple(snap.next) == ("research",)
    assert any(f"⏹ Force-cancelled during research — /continue {run_id} re-attaches the plan; "
               "✅ Approve reruns research." in t for t in _sent(env) + _edited(env))
    research.release.set()                                     # the abandoned thread may now return


async def test_a_drive_that_ends_in_time_never_hits_the_deadline(env, monkeypatch):
    sleeping = asyncio.Event()

    async def slow_clock(seconds: float) -> None:
        sleeping.set()
        await asyncio.sleep(3600)

    monkeypatch.setattr(runs, "_sleep", slow_clock)
    research = _blocking(env)
    run_id = await _executing(env, research)
    h = common._inflight[run_id]
    await runs.cancel_run(run_id)
    await sleeping.wait()
    research.release.set()
    await h.task
    assert not h.task.cancelled() and h.deadline.cancelled()  # the drive's end cancelled its deadline


async def test_a_cancel_during_the_plan_drive_wins_over_the_plan_gate(env, monkeypatch):
    """The plan gate (interrupt_after scout) stops the drive before the drain
    boundary would: the cancel still wins — no plan card, status cancelled,
    and /continue re-attaches the plan from the checkpoint."""
    started, release = threading.Event(), threading.Event()

    def scout(s):
        started.set()
        assert release.wait(10), "the test never released the scout"
        return {"sources": [], "search_spend": {"exa_usd": 0.007, "recency": None}}

    monkeypatch.setattr(graph_mod, "scout_node", scout)
    graph = env.app.bot_data["graph"] = graph_mod.build_graph(checkpointer=MemorySaver())
    out = await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="topic", length="short")
    h = common._inflight[out.run_id]
    assert await asyncio.to_thread(started.wait, 5)
    assert (await runs.cancel_run(out.run_id)).code == "draining"
    release.set()
    await h.task
    assert (await env.lib.get_run(out.run_id))["status"] == "cancelled"
    assert not any(t.startswith("📋") for t in _sent(env)), "no plan card after the cancel"
    assert out.run_id not in common._inflight
    snap = await graph.aget_state(h.cfg)
    assert snap.values["cancel_note"] == "cancelled after node scout (user cancel)"
    assert tuple(snap.next) == ("research",)
    assert (await runs.continue_run(await env.lib.get_run(out.run_id), chat_id=CHAT)).code == "attached"


# ---------------------------------------------------------------------------
# One executing run; one live drive per run (Review Focus 8)
# ---------------------------------------------------------------------------

async def test_second_executing_run_is_busy(env):
    waiting = await _at_plan_gate(env, "a run waiting at its plan gate")
    research = _blocking(env)
    running = await _executing(env, research, "the first executing run, with a long topic past the cut")

    for out in (await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="another", length="short"),
                await runs.approve_plan(waiting)):
        assert out.code == "busy"
        assert f"busy — run `{running}`" in out.text and "is at `research`" in out.text
        assert "the first executing run" in out.text and "try again when it reaches a gate" in out.text
    assert (await env.lib.get_run(waiting))["status"] == "awaiting_plan"        # unchanged
    research.release.set()
    await common._inflight[running].task
    assert (await runs.approve_plan(waiting)).code == "started"                  # runs at a gate do not count


async def test_two_approvals_produce_one_drive(env):
    research = _blocking(env)
    run_id = await _at_plan_gate(env)
    chat, console = await asyncio.gather(runs.approve_plan(run_id, actor="telegram"),
                                         runs.approve_plan(run_id, actor="webapp", length="medium"))
    assert sorted([chat.code, console.code]) == ["already_running", "started"]
    assert await asyncio.to_thread(research.started.wait, 5)
    snap = await runs.snapshot(run_id)
    assert snap["in_flight"] is True and not snap["awaiting"]
    assert snap["next"] == ["research"]                        # snap.next alone is no gate signal
    assert (await runs.report_action(run_id, "extend")).code == "already_running"
    research.release.set()
    await common._inflight[run_id].task
    assert research.calls == 1
    snap = await runs.snapshot(run_id)
    assert (snap["in_flight"], snap["awaiting"]) == (False, "report")
    gates = [e.data for e in _events(run_id, "gate")]
    assert gates == [{"kind": "plan", "actor": "telegram"}]   # the second actor changed nothing


async def test_status_and_cancel_commands_read_the_driver(env, monkeypatch):
    """/status: stage, elapsed, the last event, the console URL and its health;
    /cancel drains the live drive (spec §3 WP-E1)."""
    from argus.bot import history
    monkeypatch.setattr(history, "get_settings", lambda: env.settings)
    monkeypatch.setattr(history, "_allowed", lambda _s, _u: True)
    research = _blocking(env)
    run_id = await _executing(env, research)
    env.app.bot_data.update(webapp_public_url="https://x.trycloudflare.com", webapp_ok=True)
    ctx = SimpleNamespace(application=env.app, args=[])
    update = MagicMock()
    update.effective_user.id = update.effective_chat.id = CHAT
    update.message.reply_text = AsyncMock()

    await history.status_cmd(update, ctx)
    text = update.message.reply_text.await_args.args[0]
    assert f"- `{run_id}` · stage `research` · 0:" in text
    assert "    last: research start" in text      # the driver's task start, or the node's own after it
    assert text.splitlines()[-1] == "Console: https://x.trycloudflare.com (ok)"

    await history.cancel_cmd(update, ctx)
    assert common._inflight[run_id].control.drain_requested
    assert _sent(env)[-1].startswith(f"⏹ Cancelling run `{run_id}`")
    research.release.set()
    await common._inflight[run_id].task
    env.app.bot_data["webapp_public_url"] = None
    await history.status_cmd(update, ctx)
    assert update.message.reply_text.await_args.args[0].splitlines()[-1] == "Console: not running"


async def test_shutdown_drains_then_cancels(env, caplog):
    env.settings = dataclasses.replace(env.settings, shutdown_drain_s=0.2)
    env.monkeypatch.setattr(runs, "get_settings", lambda: env.settings)
    research = _blocking(env)                                  # ignores the drain
    run_id = await _executing(env, research)
    h = common._inflight[run_id]
    with caplog.at_level("WARNING", logger="argus.bot"):
        await asyncio.wait_for(runs.drain_all(env.app), timeout=5)
    assert h.control.drain_reason == "shutdown"
    assert h.task.cancelled()
    assert f"run {run_id}: still running 0.2 s after the shutdown drain" in caplog.text
    assert (await env.lib.get_run(run_id))["status"] == "cancelled"
    snap = await _snap(env, run_id)                            # R-c: the shutdown bound's forced cancel too
    assert snap.values["cancel_note"] == "force-cancelled during node research (shutdown); the node was abandoned"
    assert tuple(snap.next) == ("research",)
    research.release.set()


async def test_shutdown_waits_for_a_drive_that_drains_in_time(env):
    research = _blocking(env, reads_runtime=True)
    run_id = await _executing(env, research)
    h = common._inflight[run_id]
    drain = asyncio.ensure_future(runs.drain_all(env.app))
    await asyncio.sleep(0.05)
    assert not drain.done() and h.control.drain_reason == "shutdown"
    research.release.set()
    await asyncio.wait_for(drain, timeout=5)
    assert h.task.done() and not h.task.cancelled()
    assert (await _snap(env, run_id)).values["cancel_note"] == "cancelled after node research (shutdown)"


async def test_drives_are_plain_tasks_not_application_tasks(env):
    env.app.create_task = MagicMock(side_effect=AssertionError("PTB awaits these in stop()"))
    run_id = await _at_plan_gate(env)
    assert isinstance(common._inflight[run_id].task, asyncio.Task)


# ---------------------------------------------------------------------------
# The driver reports its own failures; spend on every drive end (pointer 8)
# ---------------------------------------------------------------------------

async def test_a_drive_failure_is_reported_to_the_chat(env):
    env.state.research = Research(block=False, fail=RuntimeError("digest pool exploded"))
    run_id = await _at_plan_gate(env)
    assert (await runs.approve_plan(run_id)).code == "started"
    await common._inflight[run_id].task
    assert (await env.lib.get_run(run_id))["status"] == "error"
    assert "⚠️ Research failed: digest pool exploded" in _sent(env)
    assert _events(run_id, "stage")[-1].data == {"status": "error",
                                                 "summary": "RuntimeError: digest pool exploded"}
    assert run_id not in common._inflight


@pytest.mark.parametrize("end", ["report_gate", "cancelled", "error"])
async def test_exa_usd_is_written_on_every_drive_end(env, end):
    if end == "error":
        env.state.research = Research(block=False, fail=RuntimeError("boom"))
        expected = 0.007                                        # the scout's spend; research never returned
    else:
        expected = 0.042
    research = _blocking(env) if end == "cancelled" else env.state.research
    run_id = await _at_plan_gate(env)
    assert (await env.lib.get_run(run_id))["exa_usd"] == 0.007   # also at the plan gate
    await runs.approve_plan(run_id)
    if end == "cancelled":
        assert await asyncio.to_thread(research.started.wait, 5)
        await runs.cancel_run(run_id)
        research.release.set()
    await common._inflight[run_id].task
    assert (await env.lib.get_run(run_id))["exa_usd"] == expected


async def test_a_failed_spend_write_is_visible_but_the_preview_still_ships(env):
    run_id = await _at_plan_gate(env)
    env.lib.update_run = AsyncMock(side_effect=RuntimeError("database is locked"))
    await runs.approve_plan(run_id)
    await common._inflight[run_id].task
    sent = _sent(env)
    assert any("Report preview" in t for t in sent)
    assert sent[-1] == f"⚠️ Could not record run {run_id}'s Exa spend: database is locked"
    assert (await env.lib.get_run(run_id))["status"] == "awaiting_report"   # not "resume failed"


async def test_the_run_driver_writes_runs_exa_usd_after_the_preview(env, caplog):
    """Moved from test_cut_not_flag.py (Task 9's contract, now the driver's):
    the spend write follows the preview; a run without a library row is a
    warning, not a failure."""
    order: list[str] = []
    run_id = await _at_plan_gate(env)
    send = env.bot.send_message.side_effect

    def recording_send(**kw):
        if "Report preview" in str(kw.get("text")):
            order.append("preview")
        return send(**kw)

    async def update_run(rid, **cols):
        order.append(f"update_run {rid} {cols}")
        return False                                             # no library row

    env.bot.send_message.side_effect = recording_send
    env.lib.update_run = update_run
    with caplog.at_level("WARNING", logger="argus.bot"):
        await runs.approve_plan(run_id)
        await common._inflight[run_id].task
    # R-b (Task 10 review): the same drive-end write carries the brief's archetype
    assert order == ["preview", f"update_run {run_id} {{'exa_usd': 0.042, 'archetype': 'comparison'}}"]
    assert f"run {run_id} has no library row; its exa_usd is in the checkpoint only" in caplog.text


# ---------------------------------------------------------------------------
# Resume entries: refusals reach the chat and change nothing (pointers 2, 4, 5, 6)
# ---------------------------------------------------------------------------

async def test_a_refused_approve_names_the_error_and_leaves_the_run_at_its_gate(env, monkeypatch):
    run_id = await _at_plan_gate(env)
    monkeypatch.setattr(registry, "active_snapshot", AsyncMock(side_effect=EngineConfigError(
        "role intake.classify: provider freellmapi: cannot list models")))
    before = await _snap(env, run_id)
    out = await runs.approve_plan(run_id)
    assert out.code == "refused"
    assert out.text == (f"⚠️ Could not approve run `{run_id}`: role intake.classify: provider freellmapi: "
                        "cannot list models")
    after = await _snap(env, run_id)
    assert (after.config, after.values) == (before.config, before.values)
    assert (await env.lib.get_run(run_id))["status"] == "awaiting_plan"
    assert not common._inflight[run_id].live


async def _v2_run_at_its_report_gate(env, run_id: str, status: str) -> dict:
    """A v2-topology thread (fetcher → report_builder → deliver) resting at its
    report gate, with a library row; returns its config."""
    old = StateGraph(ArgusState)
    for name in ("fetcher", "report_builder", "deliver"):
        old.add_node(name, lambda s: {})
    old.add_edge(START, "fetcher")
    old.add_edge("fetcher", "report_builder")
    old.add_edge("report_builder", "deliver")
    old.add_edge("deliver", END)
    old = old.compile(checkpointer=env.graph.checkpointer, interrupt_before=["deliver"])
    cfg = {"configurable": {"thread_id": f"tg:{CHAT}:{run_id}"}}
    folder = env.tmp_path / f"v2-{run_id}"                    # Send mirrors run.md into the report folder
    await old.ainvoke({"user_request": "t", "length": "short", "run_config": make_run_config("short"),
                       "report_paths": {"md": str(folder / "r.md"), "folder": str(folder)}}, cfg)
    await env.lib.create_run(run_id=run_id, thread_id=cfg["configurable"]["thread_id"], chat_id=CHAT,
                             topic="v2 run", status=status, config_json="{}", depth="quick", origin="telegram")
    return cfg


async def test_extend_and_revise_refuse_an_older_topology_checkpoint(env):
    """v2 checkpoints name nodes this graph lacks (fetcher, synthesizer, …); a
    resume from their report gate runs no node and stops at the same gate.
    Untracked after a restart, the run attaches on action (R-a) under the same
    rules: Extend and Revise are refused, the preview re-attaches and Send works."""
    cfg = await _v2_run_at_its_report_gate(env, "0ld0ld00", "awaiting_report")
    for action, feedback in (("extend", None), ("revise", "more detail")):
        out = await runs.report_action("0ld0ld00", action, feedback=feedback)
        assert out.code == "refused"
        assert "older Argus graph" in out.text and "fetcher" in out.text
    assert (await env.graph.aget_state(cfg)).values.get("extend_requested") is None
    out = await runs.continue_run(await env.lib.get_run("0ld0ld00"), chat_id=CHAT)
    assert out.code == "attached"                               # the preview is re-sent
    assert (await runs.report_action("0ld0ld00", "send")).code == "delivered"
    assert tuple((await env.graph.aget_state(cfg)).next) == ()  # Send ends it, as for a v4 run


async def test_continue_refuses_a_sent_older_topology_run_before_advancing_it(env):
    """A v2 run an older bot sent (``done``, still at its report gate): /continue
    would advance it and fork an Extend, which an older graph cannot run — the
    refusal comes first and the checkpoint stays where it was."""
    cfg = await _v2_run_at_its_report_gate(env, "0ld0ld01", "done")
    out = await runs.continue_run(await env.lib.get_run("0ld0ld01"), chat_id=CHAT)
    assert out.code == "refused" and "older Argus graph" in out.text
    assert tuple((await env.graph.aget_state(cfg)).next) == ("deliver",)
    assert "0ld0ld01" not in common._inflight


async def test_extend_at_its_round_cap_is_refused(env):
    run_id = await _at_plan_gate(env)
    await runs.approve_plan(run_id)
    await common._inflight[run_id].task
    await env.graph.aupdate_state(common._inflight[run_id].cfg, {"extend_rounds": 2})
    out = await runs.report_action(run_id, "extend")
    assert out.code == "refused" and "2 Extend rounds" in out.text


async def test_send_delivers_the_files_and_finishes_the_run(env):
    run_id = await _at_plan_gate(env)
    await runs.approve_plan(run_id)
    await common._inflight[run_id].task
    out = await runs.report_action(run_id, "send")
    assert out.code == "delivered"
    assert env.bot.send_document.await_count == 1               # report.md (no pdf)
    assert "✅ Delivered." in _sent(env)
    row = await env.lib.get_run(run_id)
    assert row["status"] == "done" and row["report_dir"]
    assert run_id not in common._inflight


# ---------------------------------------------------------------------------
# Plan knobs: length, depth, engines profile, recency (pointers 7, 9; ruling 1)
# ---------------------------------------------------------------------------

async def test_approve_passes_the_depth_knob(env):
    run_id = await _at_plan_gate(env)
    assert (await runs.set_plan_choice(run_id, "length", "long")).code == "updated"
    payload = await runs.plan_payload(run_id)
    assert (payload["length"], payload["depth"], payload["depth_auto"]) == ("long", "deep", True)
    assert (await runs.set_plan_choice(run_id, "depth", "exhaustive")).code == "updated"
    assert (await runs.set_plan_choice(run_id, "length", "medium")).code == "updated"   # depth stays chosen
    await runs.approve_plan(run_id)
    rc = require((await _snap(env, run_id)).values)
    assert (rc.length, rc.depth) == ("medium", "exhaustive")
    assert await env.lib.get_run_config(run_id) == rc.model_dump()
    await common._inflight[run_id].task


async def test_a_bad_knob_value_is_refused(env):
    run_id = await _at_plan_gate(env)
    for knob, value in (("length", "huge"), ("depth", "bottomless"), ("colour", "red")):
        assert (await runs.set_plan_choice(run_id, knob, value)).code == "refused"


async def test_the_recency_knob_any_turns_the_filter_off_for_a_time_sensitive_brief(env):
    rc = RunConfig.model_validate(make_run_config("short", recency="any"))
    assert search.resolve_recency(rc, ResearchBrief.model_validate(BRIEF)) == (None, None)
    run_id = await _at_plan_gate(env)
    payload = await runs.plan_payload(run_id)
    assert (payload["recency"], payload["recency_window"]["window"]) == (None, "m")   # auto: the brief's
    for expected in ("d", "w", "m", "y", "any"):
        await runs.set_plan_choice(run_id, "recency", "next")
        assert (await runs.plan_payload(run_id))["recency"] == expected
    assert (await runs.plan_payload(run_id))["recency_window"] is None
    await runs.approve_plan(run_id)
    values = (await _snap(env, run_id)).values
    rc = require(values)
    assert rc.recency == "any" and rc.overrides_applied == {"recency": "any"}
    assert values["search_spend"]["recency"] is None           # research reuses it: no filter, nothing dropped
    assert values["search_spend"]["exa_usd"] == 0.007          # the rest of the scout's spend kept
    await common._inflight[run_id].task


async def test_the_profile_knob_cycles_cached_validation_results(env, monkeypatch):
    calls: list[str] = []

    async def resolve(cfg, name):
        calls.append(name)
        if name == "max-quality":
            raise EngineConfigError("role compose.section: provider anthropic: ANTHROPIC_API_KEY is not set")
        return env.active.model_copy(update={"profile": name})

    monkeypatch.setattr(registry, "resolve_snapshot", resolve)
    names = list(registry.current_config().profiles)
    run_id = await _at_plan_gate(env)
    assert (await runs.plan_payload(run_id))["profile"] == "free-proxy"
    seen = []
    for _ in range(len(names) * 2):                             # twice round the cycle
        await runs.set_plan_choice(run_id, "profile", "next")
        p = await runs.plan_payload(run_id)
        seen.append((p["profile"], p["profile_problem"]))
    assert [name for name, _ in seen[:len(names)]] == names[1:] + names[:1]
    assert dict(seen)["max-quality"].endswith("ANTHROPIC_API_KEY is not set")
    assert sorted(calls) == sorted(n for n in names if n != "free-proxy")      # each validated once
    while (await runs.plan_payload(run_id))["profile"] != "max-quality":
        await runs.set_plan_choice(run_id, "profile", "next")
    out = await runs.approve_plan(run_id)
    assert out.code == "refused" and "ANTHROPIC_API_KEY is not set" in out.text
    while (await runs.plan_payload(run_id))["profile"] != "hybrid-local":
        await runs.set_plan_choice(run_id, "profile", "next")
    assert (await runs.approve_plan(run_id)).code == "started"
    assert require((await _snap(env, run_id)).values).engines.profile == "hybrid-local"
    await common._inflight[run_id].task


@pytest.mark.parametrize("exa_on,shown", [(True, "Exa: ≈ $"), (False, "Exa: off")])
async def test_the_plan_card_shows_the_exa_estimate_only_for_an_enabled_exa(env, monkeypatch, exa_on, shown):
    monkeypatch.setattr(runs, "exa_enabled", lambda: exa_on)
    run_id = await _at_plan_gate(env)
    payload = await runs.plan_payload(run_id)
    assert payload["exa_enabled"] is exa_on
    assert (payload["exa_estimate_usd"] is not None) is exa_on
    card = next(t for t in _sent(env) if t.startswith("📋"))
    assert shown in card and "Archetype: comparison" in card
    assert ("≈ $" in card) is exa_on


async def test_exa_left_out_of_the_providers_shows_no_estimate(env):
    """Exa disabled in the run's own search config (a console override) is off too."""
    out = await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="topic", length="short",
                               overrides={"search.providers_enabled": ["ddgs", "arxiv"]})
    await common._inflight[out.run_id].task
    payload = await runs.plan_payload(out.run_id)
    assert (payload["exa_enabled"], payload["exa_estimate_usd"]) == (False, None)
    assert "Exa: off" in next(t for t in _sent(env) if t.startswith("📋"))


async def test_a_console_approve_syncs_the_telegram_plan_card(env):
    run_id = await _at_plan_gate(env)
    card_id = common._inflight[run_id].gate_msg[0]
    assert (await runs.approve_plan(run_id, actor="webapp")).code == "started"
    edit = next(c.kwargs for c in env.bot.edit_message_text.await_args_list
                if c.kwargs.get("message_id") == card_id)
    assert edit["text"].endswith("✅ approved from console") and edit["text"].startswith("📋")
    assert "reply_markup" not in edit or edit["reply_markup"] is None
    await common._inflight[run_id].task


# ---------------------------------------------------------------------------
# The report preview's quality line repeats the banners (pointer 10)
# ---------------------------------------------------------------------------

async def test_the_preview_quality_line_repeats_the_banners_and_notes(env):
    run_id = await _at_plan_gate(env)
    await runs.approve_plan(run_id)
    await common._inflight[run_id].task
    preview = next(t for t in _sent(env) if "Report preview" in t)
    head = preview.split("<pre>")[0]
    assert "🧪 Quality: judged 7/9 · 1 cut" in head
    assert "⚠ Not fact-checked — 2 of 9 findings were not judged (429)" in head
    assert "⚠ Degraded run — 14 of 28 sources could not be digested (429)" in head
    assert "[r1] skip: research.digest https://w.example/y (429)" in head
    assert "[r1] degrade: research.rerank → keyword_triage (429)" in head


# ---------------------------------------------------------------------------
# Untracked gate runs attach on action (§2.5 ruling R-a, Task 10 review)
# ---------------------------------------------------------------------------

_ACTIONS = {"approve": "plan:approve", "edit_reply": "plan:edit", "knob": "len:long", "cancel": "plan:cancel",
            "send": "report:send", "extend": "report:extend", "revise": "report:revise"}


@pytest.mark.parametrize("action", list(_ACTIONS))
async def test_an_untracked_run_at_its_gate_attaches_on_action(env, action):
    """A restart empties the registry: an action on a run resting at its gate
    attaches it, then acts. The tapped card becomes the gate message, so the
    driver's edits land on it (re-rendered from the payload: the tap carries
    no text), and snapshot()["awaiting"] agrees with the outcome."""
    report = action in ("send", "extend", "revise")
    run_id = await (_at_report_gate(env) if report else _at_plan_gate(env))
    common._inflight.clear()                                   # the restart
    assert (await runs.snapshot(run_id))["awaiting"] == ("report" if report else "plan")
    tap = await _tap(env, f"{_ACTIONS[action]}:{run_id}")
    tap.callback_query.edit_message_text.assert_not_awaited()  # no "not in flight" notice over the card
    edits = _edits(env, _TAPPED)
    if action in ("edit_reply", "revise"):
        assert common._pending_reply[CHAT] == ("plan_edit" if action == "edit_reply" else "revise_feedback",
                                               run_id)
        assert ("Reply with your changes" if action == "edit_reply" else "Reply with what to change") in \
            edits[-1]["text"]
        common._inflight.clear()                               # the registry lost again, the reply still armed
        await _reply(env, "focus on security")
    await settle_runs()
    snap = await _snap(env, run_id)
    row = await env.lib.get_run(run_id)
    edits = _edits(env, _TAPPED)
    plan_card = (ParseMode.MARKDOWN, "📋")
    match action:
        case "approve":
            assert (row["status"], tuple(snap.next)) == ("awaiting_report", ("deliver",))
            assert edits[-1]["text"].endswith("_✅ Approved — continuing._")
        case "edit_reply":
            assert row["status"] == "awaiting_plan" and tuple(snap.next) == ("research",)
            assert len([t for t in _sent(env) if t.startswith("📋")]) == 2      # the redrafted plan card
        case "knob":
            assert (await runs.plan_payload(run_id))["length"] == "long"
            assert "mode: Long" in edits[-1]["text"]
            assert isinstance(edits[-1]["reply_markup"], InlineKeyboardMarkup)
        case "cancel":
            assert row["status"] == "cancelled" and run_id not in common._inflight
            assert edits[-1]["text"].endswith("❌ Cancelled.")
        case "send":
            assert row["status"] == "done" and tuple(snap.next) == ()
            assert edits[-1]["text"].startswith("📝") and edits[-1]["text"].endswith("📤 Sent.")
            assert edits[-1]["parse_mode"] == ParseMode.HTML
            env.bot.send_document.assert_awaited()
        case "extend":
            assert (snap.values["extend_rounds"], tuple(snap.next)) == (1, ("deliver",))
            assert edits[-1]["text"].endswith("🔎 Extending research — gathering more…")
        case "revise":
            assert (snap.values["revise_rounds"], tuple(snap.next)) == (1, ("deliver",))
    if action in ("approve", "edit_reply", "knob", "cancel"):
        assert (edits[0]["parse_mode"], edits[0]["text"][:1]) == plan_card


async def test_a_run_not_resting_at_a_gate_keeps_the_notice(env):
    """The notice (over the tapped card) stays only for a run with no library
    row or one not resting at a gate — a sent run is finished, never awaiting."""
    run_id = await _at_report_gate(env)
    assert (await runs.report_action(run_id, "send")).code == "delivered"
    assert (await runs.snapshot(run_id))["awaiting"] is None
    for data in (f"report:extend:{run_id}", "plan:approve:deadbeef"):
        tap = await _tap(env, data)
        notice = tap.callback_query.edit_message_text.await_args.args[0]
        assert "is not in flight" in notice and "/continue" in notice
    assert run_id not in common._inflight


async def test_cancel_command_resolves_an_untracked_run(env):
    """/cancel <id> finds an untracked run through the library, as /continue
    does; one resting at its gate gets the gate cancel."""
    run_id = await _at_plan_gate(env)
    common._inflight.clear()
    await history.cancel_cmd(_update(), _ctx(env, [run_id[:4]]))
    assert (await env.lib.get_run(run_id))["status"] == "cancelled"
    assert _sent(env)[-1] == f"❌ Cancelled run `{run_id}`."
    assert run_id not in common._inflight


# ---------------------------------------------------------------------------
# A deadline per drive; a cancel during gate presentation is not lost (review A)
# ---------------------------------------------------------------------------

async def test_a_cancel_during_the_plan_presentation_ends_the_run_cancelled(env, monkeypatch):
    """The drain arrives after the stream returned, while the plan card is
    presented: no boundary is left to drain at, so the gate cancel follows the
    presentation — the user was told "Cancelling…", and the run ends cancelled."""
    present, outcomes = runs._present_plan, []

    async def cancel_then_present(h, values):
        outcomes.append(await runs.cancel_run(h.run_id))
        await present(h, values)

    monkeypatch.setattr(runs, "_present_plan", cancel_then_present)
    out = await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="topic", length="short")
    h = common._inflight[out.run_id]
    await h.task
    assert outcomes[0].code == "draining"
    assert (await env.lib.get_run(out.run_id))["status"] == "cancelled"
    assert out.run_id not in common._inflight
    assert _edits(env, h.gate_msg[0])[-1]["text"].endswith("❌ Cancelled.")
    assert h.deadline is not None and h.deadline.cancelled()   # the drive's end stopped its deadline


async def test_an_old_cancel_deadline_never_cancels_a_later_drive(env, monkeypatch):
    """A deadline is bound to the drive it was armed for: when the first
    cancel's window elapses during a later drive, that drive goes on."""
    wake = asyncio.Event()

    async def held_clock(seconds: float) -> None:
        await wake.wait()

    monkeypatch.setattr(runs, "_sleep", held_clock)
    present = runs._present_plan

    async def cancel_then_present(h, values):
        await runs.cancel_run(h.run_id)
        await present(h, values)

    monkeypatch.setattr(runs, "_present_plan", cancel_then_present)
    out = await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="topic", length="short")
    await common._inflight[out.run_id].task
    monkeypatch.setattr(runs, "_present_plan", present)
    research = _blocking(env)
    assert (await runs.approve_plan(out.run_id)).code == "started"
    h = common._inflight[out.run_id]
    assert await asyncio.to_thread(research.started.wait, 5)
    wake.set()                                                 # the first cancel's 300 s elapse now
    for _ in range(5):
        await asyncio.sleep(0)
    research.release.set()
    await asyncio.wait([h.task], timeout=5)
    assert h.task.done() and not h.task.cancelled()
    assert (await env.lib.get_run(out.run_id))["status"] == "awaiting_report"


# ---------------------------------------------------------------------------
# R-b: runs.archetype; R-d / test gaps: busy everywhere, the lock
# ---------------------------------------------------------------------------

async def test_the_drive_end_writes_the_archetype(env):
    run_id = await _at_plan_gate(env)
    assert (await env.lib.get_run(run_id))["archetype"] == "comparison"


async def test_busy_covers_ask_continue_edit_extend_and_revise(env):
    env.app.bot_data["quick_graph"] = graph_mod.quick_answer_graph(checkpointer=MemorySaver())
    at_report = await _at_report_gate(env, "a run at its report gate")
    at_plan = await _at_plan_gate(env, "a run at its plan gate")
    first = _blocking(env)
    stopped = await _executing(env, first, "a run cancelled mid-pipeline")
    await runs.cancel_run(stopped)
    first.release.set()
    await common._inflight[stopped].task
    research = _blocking(env)
    running = await _executing(env, research, "the executing run")
    outs = {
        "ask": await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="what is x?", quick=True),
        "continue_resume": await runs.continue_run(await env.lib.get_run(stopped), chat_id=CHAT),
        "edit": await runs.edit_plan(at_plan, "focus on security"),
        "extend": await runs.report_action(at_report, "extend"),
        "revise": await runs.report_action(at_report, "revise", feedback="tighten it"),
    }
    assert {name: out.code for name, out in outs.items()} == dict.fromkeys(outs, "busy")
    assert all(f"busy — run `{running}`" in out.text for out in outs.values())
    research.release.set()


async def test_a_cancel_racing_an_approves_preparation_drains_the_new_drive(env, monkeypatch):
    """The approve holds the run's lock while it prepares: the cancel waits,
    then drains the drive the approve launched (a cancel that saw only
    ``starting`` would drain the old control and be lost)."""
    run_id = await _at_plan_gate(env)
    research = _blocking(env)
    gate, resumable = asyncio.Event(), runs._resumable

    async def slow_resumable(h, engines=None):
        await gate.wait()
        return await resumable(h, engines)

    monkeypatch.setattr(runs, "_resumable", slow_resumable)
    approve = asyncio.ensure_future(runs.approve_plan(run_id))
    await asyncio.sleep(0.05)
    cancel = asyncio.ensure_future(runs.cancel_run(run_id))
    await asyncio.sleep(0.05)
    assert not cancel.done()                                   # it waits for the run's lock
    gate.set()
    assert (await approve).code == "started"
    assert (await cancel).code == "draining"
    h = common._inflight[run_id]
    assert h.control.drain_requested                           # the new drive's control
    research.release.set()
    await h.task
    assert (await env.lib.get_run(run_id))["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Send ends the run; pinned appends are always ingested (§2.4 + the ruling)
# ---------------------------------------------------------------------------

async def test_send_ends_the_run(env):
    run_id = await _at_report_gate(env)
    assert (await runs.report_action(run_id, "send")).code == "delivered"
    assert tuple((await _snap(env, run_id)).next) == ()
    view = await runs.snapshot(run_id)
    assert (view["in_flight"], view["awaiting"]) == (False, None)


async def test_append_then_continue_on_a_sent_run_ingests_and_marks_the_appends(env):
    run_id = await _at_report_gate(env)
    assert (await runs.report_action(run_id, "send")).code == "delivered"
    env.state.research = research = FetchAll()
    ref = "https://a.example/p"
    await history.append_cmd(_update(), _ctx(env, [run_id, ref]))
    await history.continue_cmd(_update(), _ctx(env, [run_id]))
    await settle_runs()
    ((append_only, urls),) = research.seen
    assert append_only is True and ref in urls                 # the append pass: exactly the appends
    assert await _source_statuses(env, run_id) == {ref: "ingested"}
    assert tuple((await _snap(env, run_id)).next) == ("deliver",)


async def test_continue_advances_a_run_an_older_bot_sent_then_forks_it(env, monkeypatch):
    run_id = await _at_report_gate(env)
    await env.lib.set_run_status(run_id, "done")              # an older Send: done, still at its report gate
    common._inflight.clear()
    writes, update_state = [], env.graph.aupdate_state

    async def recording(cfg, values, as_node=None, **kw):
        writes.append((dict(values), as_node))
        return await update_state(cfg, values, as_node=as_node, **kw)

    monkeypatch.setattr(env.graph, "aupdate_state", recording)
    out = await runs.continue_run(await env.lib.get_run(run_id), chat_id=CHAT)
    assert out.code == "started" and "Continuing run" in out.text
    await settle_runs()
    assert writes[0] == ({}, "deliver")                        # advanced past its report gate first
    assert writes[1][1] == "deliver" and writes[1][0]["extend_requested"] is True    # then the fork
    snap = await _snap(env, run_id)
    assert (tuple(snap.next), snap.values["extend_rounds"]) == (("deliver",), 1)


async def test_a_report_gate_extend_merges_the_pending_appends_and_marks_them(env):
    run_id = await _at_report_gate(env)
    env.state.research = research = FetchAll()
    ref = "https://a.example/p"
    await history.append_cmd(_update(), _ctx(env, [run_id, ref]))
    assert (await runs.report_action(run_id, "extend")).code == "started"
    await settle_runs()
    ((append_only, urls),) = research.seen
    assert not append_only and ref in urls and "https://w.example/x" in urls     # a normal Extend pass
    assert await _source_statuses(env, run_id) == {ref: "ingested"}


async def test_a_reattach_names_the_pending_appends(env):
    run_id = await _at_plan_gate(env)
    common._inflight.clear()
    await env.lib.add_run_source(run_id, "https://a.example/p")
    await history.continue_cmd(_update(), _ctx(env, [run_id]))
    assert _sent(env)[-1] == (f"📎 Run `{run_id}` has 1 pending appended source(s): 🔎 Extend at the report "
                              f"preview ingests them, or /continue {run_id} once the report is sent.")


async def test_a_resume_that_leaves_pending_appends_unused_says_so(env):
    """Approve and Send ingest no appends: each says how many are pending and
    which path ingests them (the re-attach is the test above)."""
    run_id = await _at_plan_gate(env)
    await env.lib.add_run_source(run_id, "https://a.example/p")
    out = await runs.approve_plan(run_id)
    assert (out.code, out.text) == ("started", f"📎 Run `{run_id}` has 1 pending appended source(s): 🔎 Extend "
                                               f"at the report preview ingests them, or /continue {run_id} once "
                                               "the report is sent.")
    await common._inflight[run_id].task
    out = await runs.report_action(run_id, "send")
    assert (out.code, out.text) == ("delivered", f"📎 Run `{run_id}` has 1 pending appended source(s): "
                                                 f"/continue {run_id} ingests them.")


async def test_a_failed_report_callback_reaches_the_chat(env):
    run_id = await _at_plan_gate(env)

    async def broken() -> None:
        raise RuntimeError("database is locked")

    common._inflight[run_id].on_report = broken
    await runs.approve_plan(run_id)
    await common._inflight[run_id].task
    assert f"⚠️ Could not record run {run_id}'s appended sources: database is locked" in _sent(env)


# ---------------------------------------------------------------------------
# start_run(engines_profile=…) (§2.7 POST /api/runs)
# ---------------------------------------------------------------------------

async def test_start_run_takes_an_engines_profile(env, monkeypatch):
    async def resolve(cfg, name):
        if name == "max-quality":
            raise EngineConfigError("role compose.section: provider anthropic: ANTHROPIC_API_KEY is not set")
        return env.active.model_copy(update={"profile": name})

    monkeypatch.setattr(registry, "resolve_snapshot", resolve)
    out = await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="topic", length="short",
                               engines_profile="hybrid-local")
    assert out.code == "started"
    await common._inflight[out.run_id].task
    assert require((await _snap(env, out.run_id)).values).engines.profile == "hybrid-local"   # the plan drive's
    assert (await runs.plan_payload(out.run_id))["profile"] == "hybrid-local"                # the card's choice
    assert "Engines: hybrid-local" in next(t for t in _sent(env) if t.startswith("📋"))
    assert (await runs.approve_plan(out.run_id)).code == "started"
    await common._inflight[out.run_id].task
    assert require((await _snap(env, out.run_id)).values).engines.profile == "hybrid-local"
    for profile, why in (("max-quality", "role compose.section: provider anthropic: ANTHROPIC_API_KEY is not set"),
                         ("no-such", "unknown engines profile 'no-such'")):
        refused = await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="t", engines_profile=profile)
        assert (refused.code, refused.text) == ("refused", f"⚠️ Could not start the research: {why}")


# ---------------------------------------------------------------------------
# /status: a run at its gate, the console failure (§2.7 webapp_ok / webapp_error)
# ---------------------------------------------------------------------------

async def test_status_shows_a_run_at_its_gate_and_the_console_failure(env):
    run_id = await _at_plan_gate(env)
    env.app.bot_data.update(webapp_public_url=None, webapp_ok=False,
                            webapp_error="cloudflared: executable not found")
    update = _update()
    await history.status_cmd(update, _ctx(env))
    text = update.message.reply_text.await_args.args[0]
    assert f"- `{run_id}` · waiting at its plan gate" in text
    assert text.splitlines()[-1] == "Console: down — cloudflared: executable not found"


# ---------------------------------------------------------------------------
# Each stage transition is published once; a delivered run's buffer expires
# ---------------------------------------------------------------------------

async def test_each_stage_transition_is_published_once(env):
    run_id = await _at_report_gate(env)
    stages = [(e.node, e.data["status"]) for e in _events(run_id, "stage")]
    nodes = ("intake", "brief", "scout", "research", "outline", "compose", "analyst", "panel", "report_builder")
    assert stages == [(node, status) for node in nodes for status in ("start", "end")]


async def test_a_drain_is_published_once(env):
    research = _blocking(env, reads_runtime=True, announces_drain=True)
    run_id = await _executing(env, research)
    h = common._inflight[run_id]
    await runs.cancel_run(run_id)
    research.release.set()
    await h.task
    assert [e.data["status"] for e in _events(run_id, "stage") if e.node == "research"] == [
        "start", "draining", "end", "cancelled"]


async def test_a_delivered_runs_bus_buffer_expires(env, monkeypatch):
    clock = SimpleNamespace(t=1_000.0)
    monkeypatch.setattr(runs, "bus", RunEventBus(clock=lambda: clock.t))
    run_id = await _at_report_gate(env)
    assert (await runs.report_action(run_id, "send")).code == "delivered"
    assert runs.bus.since(run_id, 0)
    clock.t += 3_601
    assert runs.bus.since(run_id, 0) == []


# ---------------------------------------------------------------------------
# Telegram edit errors on the gate message (the card's are in test_progress_bus)
# ---------------------------------------------------------------------------

async def test_a_gate_edit_that_changes_nothing_is_a_success(env):
    """A knob tap that changes nothing: Telegram answers "message is not
    modified" — a success, with no plain-text retry (that flipped the card to
    raw Markdown)."""
    run_id = await _at_plan_gate(env)
    card = common._inflight[run_id].gate_msg[0]
    env.bot.edit_message_text.side_effect = BadRequest(
        "Message is not modified: specified new message content and reply markup are exactly the same as a "
        "current content and reply markup of the message")
    assert (await runs.set_plan_choice(run_id, "length", "short")).code == "updated"
    (edit,) = _edits(env, card)
    assert edit["parse_mode"] == ParseMode.MARKDOWN


async def test_a_gate_edit_honours_retry_after(env, monkeypatch):
    waited: list[float] = []

    async def clock(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(progress, "_sleep", clock, raising=False)
    run_id = await _at_plan_gate(env)
    card = common._inflight[run_id].gate_msg[0]
    env.bot.edit_message_text.side_effect = [RetryAfter(7), None]
    assert (await runs.set_plan_choice(run_id, "length", "long")).code == "updated"
    assert waited == [7]
    assert [e["parse_mode"] for e in _edits(env, card)] == [ParseMode.MARKDOWN, ParseMode.MARKDOWN]


async def test_a_gate_message_that_no_longer_exists_is_not_edited_again(env, caplog):
    run_id = await _at_plan_gate(env)
    card = common._inflight[run_id].gate_msg[0]
    env.bot.edit_message_text.side_effect = BadRequest("Message to edit not found")
    with caplog.at_level("WARNING", logger="argus.bot"):
        for length in ("long", "medium"):
            assert (await runs.set_plan_choice(run_id, "length", length)).code == "updated"
    assert len(_edits(env, card)) == 1                         # no plain-text try, no second tap's edit
    assert caplog.text.count("no longer exists") == 1


# ---------------------------------------------------------------------------
# Claims: `starting` never sticks; console knobs resolve first; shutdown
# ---------------------------------------------------------------------------

async def test_a_raising_gate_edit_never_leaves_the_run_claimed(env):
    run_id = await _at_plan_gate(env)
    env.bot.edit_message_text.side_effect = RuntimeError("socket closed")
    out = await runs.approve_plan(run_id)
    assert (out.code, out.text) == ("refused", f"⚠️ Could not approve run `{run_id}`: socket closed")
    assert not common._inflight[run_id].live
    env.bot.edit_message_text.side_effect = None
    assert (await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="another", length="short")).code == "started"


async def test_a_refused_console_approve_leaves_the_choices_unchanged(env, monkeypatch):
    async def resolve(cfg, name):
        raise EngineConfigError("role compose.section: provider anthropic: ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(registry, "resolve_snapshot", resolve)
    run_id = await _at_plan_gate(env)
    knobs = ("length", "depth", "depth_auto", "profile")
    before = {k: (await runs.plan_payload(run_id))[k] for k in knobs}
    out = await runs.approve_plan(run_id, actor="webapp", length="long", depth="deep",
                                  engines_profile="max-quality")
    assert out.code == "refused" and "ANTHROPIC_API_KEY is not set" in out.text
    assert {k: (await runs.plan_payload(run_id))[k] for k in knobs} == before


_SHUTTING_DOWN = "⏹ Argus is shutting down — try again after the restart"


async def test_shutdown_refuses_new_drives(env):
    waiting = await _at_plan_gate(env)
    await runs.drain_all(env.app)                              # nothing live: it returns at once
    for out in (await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="another", length="short"),
                await runs.approve_plan(waiting)):
        assert (out.code, out.text) == ("refused", _SHUTTING_DOWN)
    assert (await env.lib.get_run(waiting))["status"] == "awaiting_plan"


async def _prepared_across_the_shutdown_start(env, monkeypatch, action) -> runs.Outcome:
    """``action`` claims its run, then prepares while ``drain_all`` starts: the
    shutdown waits for the claim made before it began."""
    gate, resumable = asyncio.Event(), runs._resumable

    async def slow_resumable(h, engines=None):
        await gate.wait()
        return await resumable(h, engines)

    monkeypatch.setattr(runs, "_resumable", slow_resumable)
    acting = asyncio.ensure_future(action())
    await asyncio.sleep(0.05)
    drain = asyncio.ensure_future(runs.drain_all(env.app))
    await asyncio.sleep(0.05)
    assert not drain.done()                                    # it waits for the claim still preparing
    gate.set()
    out = await acting
    await asyncio.wait_for(drain, timeout=5)
    monkeypatch.setattr(runs, "_resumable", resumable)
    return out


def _restart(monkeypatch) -> None:
    monkeypatch.setattr(runs, "_shutting_down", False)
    monkeypatch.setattr(runs, "_launch_closed", False)
    common._inflight.clear()


async def test_an_extend_preparing_at_the_shutdown_start_launches_then_drains(env, monkeypatch):
    """C4: the Extend claimed before drain_all began launches, and its drive is
    drained; its flag never outlives it — after a restart, Send ends the run."""
    run_id = await _at_report_gate(env)
    out = await _prepared_across_the_shutdown_start(
        env, monkeypatch, lambda: runs.report_action(run_id, "extend", actor="webapp"))
    assert out.code == "started"
    assert (await env.lib.get_run(run_id))["status"] == "cancelled"
    snap = await _snap(env, run_id)
    assert tuple(snap.next) == ("deliver",) and snap.values["cancel_note"].endswith("(shutdown)")
    _restart(monkeypatch)
    assert (await runs.report_action(run_id, "send")).code == "delivered"
    assert tuple((await _snap(env, run_id)).next) == ()


async def test_an_approve_preparing_at_the_shutdown_start_ends_drained(env, monkeypatch):
    """C4: the Approve claimed before drain_all began launches and is drained:
    it ends cancelled with its drive's card, never ``running`` without a drive."""
    run_id = await _at_plan_gate(env)
    h, plan_drive = common._inflight[run_id], common._inflight[run_id].task
    out = await _prepared_across_the_shutdown_start(env, monkeypatch, lambda: runs.approve_plan(run_id))
    assert out.code == "started"
    assert h.task is not plan_drive and h.task.done() and not h.task.cancelled()
    assert h.control.drain_reason == "shutdown"
    assert (await env.lib.get_run(run_id))["status"] == "cancelled"
    assert tuple((await _snap(env, run_id)).next) == ("research",)
    assert h.card.final == (f"⏹ Cancelled before it started — /continue {run_id} re-attaches the plan card — "
                            "repeat the action there.")


async def test_a_start_preparing_at_the_shutdown_start_never_launches(env, monkeypatch):
    """A fresh thread drained before its first node keeps no input and cannot be
    resumed, so a start claimed before drain_all began is refused before its
    launch; its library row is marked cancelled."""
    gate, active = asyncio.Event(), runs.registry.active_snapshot

    async def slow_active():
        await gate.wait()
        return await active()

    monkeypatch.setattr(registry, "active_snapshot", slow_active)
    starting = asyncio.ensure_future(runs.start_run(chat_id=CHAT, user_id=CHAT, topic="topic", length="short"))
    await asyncio.sleep(0.05)
    drain = asyncio.ensure_future(runs.drain_all(env.app))
    await asyncio.sleep(0.05)
    assert not drain.done()                                    # it waits for the start still preparing
    gate.set()
    out = await starting
    await asyncio.wait_for(drain, timeout=5)
    assert (out.code, out.text) == ("refused", _SHUTTING_DOWN)
    assert (await env.lib.get_run(out.run_id))["status"] == "cancelled"
    assert out.run_id not in common._inflight
    assert not any(t.startswith("🔄") or t.startswith("⚠️") for t in _sent(env))   # no card, no failure


async def test_an_action_claimed_after_the_shutdown_start_is_refused_before_any_write(env):
    at_plan = await _at_plan_gate(env, "a run at its plan gate")
    at_report = await _at_report_gate(env, "a run at its report gate")

    async def state() -> dict:
        return {rid: ((await env.lib.get_run(rid))["status"], (await _snap(env, rid)).values,
                      _edits(env, common._inflight[rid].gate_msg[0])) for rid in (at_plan, at_report)}

    before = await state()
    await runs.drain_all(env.app)
    for out in (await runs.approve_plan(at_plan, actor="webapp"),
                await runs.report_action(at_report, "extend", actor="webapp"),
                await runs.report_action(at_report, "revise", feedback="tighten it", actor="webapp")):
        assert (out.code, out.text) == ("refused", _SHUTTING_DOWN)
    assert await state() == before                             # status, checkpoint and card unchanged


# ---------------------------------------------------------------------------
# The Edit / Revise replies (review B)
# ---------------------------------------------------------------------------

async def test_edit_and_revise_check_busy_and_the_cap_before_arming(env):
    at_plan = await _at_plan_gate(env, "a run at its plan gate")
    research = _blocking(env)
    running = await _executing(env, research, "the executing run")
    out = await runs.await_reply(at_plan, "plan_edit")
    assert out.code == "busy" and CHAT not in common._pending_reply
    research.release.set()
    await common._inflight[running].task                      # its report gate
    await env.graph.aupdate_state(common._inflight[running].cfg, {"revise_rounds": MAX_REVISE_ROUNDS})
    out = await runs.await_reply(running, "revise_feedback")
    assert (out.code, out.text) == ("refused", f"⚠️ Could not revise run `{running}`: it already used its "
                                               f"{MAX_REVISE_ROUNDS} Revise rounds")
    assert CHAT not in common._pending_reply


async def test_a_reply_refused_after_it_was_consumed_says_so(env):
    at_plan = await _at_plan_gate(env)
    assert (await runs.await_reply(at_plan, "plan_edit")).code == "updated"
    research = _blocking(env)
    running = await _executing(env, research, "another run")
    await _reply(env, "focus on security")
    assert CHAT not in common._pending_reply                   # consumed
    last = _sent(env)[-1]
    assert last.startswith(f"⏳ busy — run `{running}`") and last.endswith("tap Edit/Revise and send it again)")
    research.release.set()


async def test_a_launch_clears_the_runs_pending_reply(env, monkeypatch):
    run_id = await _at_plan_gate(env)
    assert (await runs.await_reply(run_id, "plan_edit")).code == "updated"
    assert (await runs.approve_plan(run_id)).code == "started"
    assert CHAT not in common._pending_reply
    media_branch = AsyncMock()
    monkeypatch.setattr(bot_media, "handle_text", media_branch)
    await _reply(env, "https://youtu.be/abc123DEF45")
    media_branch.assert_awaited_once()
    await common._inflight[run_id].task


# ---------------------------------------------------------------------------
# Plan-card renders never validate live; the knobs through on_callback
# ---------------------------------------------------------------------------

async def test_plan_card_renders_never_validate_live(env, monkeypatch):
    """Renders read the cached active snapshot and each other profile's cached
    result (a failure included); only Approve validates the chosen one live."""
    resolved: list[str] = []

    async def resolve(cfg, name):
        resolved.append(name)
        raise EngineConfigError(f"provider {name}: down")

    monkeypatch.setattr(registry, "resolve_snapshot", resolve)
    run_id = await _at_plan_gate(env)
    live = AsyncMock(side_effect=EngineConfigError("role intake.classify: provider freellmapi: cannot list models"))
    monkeypatch.setattr(registry, "active_snapshot", live)
    names = list(registry.current_config().profiles)
    for _ in range(2 * len(names)):                            # twice round the cycle
        await runs.set_plan_choice(run_id, "profile", "next")
    for knob, value in (("length", "long"), ("depth", "deep"), ("recency", "next")):
        await runs.set_plan_choice(run_id, knob, value)
    live.assert_not_awaited()                                  # the active profile: the cached snapshot
    assert sorted(resolved) == sorted(n for n in names if n != "free-proxy")     # each other one once
    while (await runs.plan_payload(run_id))["profile"] != "hybrid-local":
        await runs.set_plan_choice(run_id, "profile", "next")
    out = await runs.approve_plan(run_id)
    assert out.code == "refused" and "provider hybrid-local: down" in out.text
    assert resolved.count("hybrid-local") == 2                 # Approve validated it live


async def test_the_plan_knobs_through_on_callback(env, monkeypatch):
    monkeypatch.setattr(registry, "resolve_snapshot",
                        AsyncMock(side_effect=lambda cfg, name: env.active.model_copy(update={"profile": name})))
    run_id = await _at_plan_gate(env)
    card = common._inflight[run_id].gate_msg[0]
    names = list(registry.current_config().profiles)
    await _tap(env, f"dep:deep:{run_id}", card)
    p = await runs.plan_payload(run_id)
    assert (p["depth"], p["depth_auto"]) == ("deep", False)
    await _tap(env, f"dep:auto:{run_id}", card)
    p = await runs.plan_payload(run_id)
    assert (p["depth"], p["depth_auto"]) == (RunConfig.default_depth("short"), True)
    await _tap(env, f"prof:next:{run_id}", card)
    assert (await runs.plan_payload(run_id))["profile"] == names[1]
    await _tap(env, f"rec:next:{run_id}", card)
    assert (await runs.plan_payload(run_id))["recency"] == "d"
    rows = [[(b.text, b.callback_data) for b in row] for row in _edits(env, card)[-1]["reply_markup"].inline_keyboard]
    assert rows[2][-1] == ("Auto ✅", f"dep:auto:{run_id}")


# ---------------------------------------------------------------------------
# Fix round 2: the guarded cancel note, quick runs, the attach race,
# gate-matched cards, the profile on re-attach
# ---------------------------------------------------------------------------

async def test_a_failing_cancel_note_write_still_ends_the_forced_cancel(env, monkeypatch, caplog):
    """C5: the forced cancel's note write fails — it is logged and said in the
    chat; the run still ends cancelled, its card stopped, its task cancelled."""
    async def past(seconds: float) -> None:
        return None

    monkeypatch.setattr(runs, "_sleep", past)
    research = _blocking(env)                                  # never reads runtime: the deadline fires
    run_id = await _executing(env, research)
    h = common._inflight[run_id]
    update_state = env.graph.aupdate_state

    async def failing(cfg, values, *a, **kw):
        if "cancel_note" in values:
            raise RuntimeError("database is locked")
        return await update_state(cfg, values, *a, **kw)

    monkeypatch.setattr(env.graph, "aupdate_state", failing)
    with caplog.at_level("ERROR", logger="argus.bot"):
        await runs.cancel_run(run_id)
        await asyncio.wait([h.task], timeout=10)
    assert h.task.cancelled()
    assert (await env.lib.get_run(run_id))["status"] == "cancelled"
    assert h.card.final == f"⏹ Force-cancelled during research — /continue {run_id} resumes it."
    assert f"⚠️ Run {run_id}: could not record the cancel note: database is locked" in _sent(env)
    assert any(r.exc_info and r.getMessage() == f"run {run_id}: could not record the cancel note"
               for r in caplog.records)
    research.release.set()


@pytest.mark.parametrize("forced", [False, True], ids=["drained", "force_cancelled"])
async def test_a_cancelled_ask_is_never_told_to_continue(env, monkeypatch, forced):
    """continue_run refuses quick runs: a cancelled /ask's card and chat lines
    name no /continue, whether it drained or its drain timed out."""
    started, release = threading.Event(), threading.Event()

    def quick_answer(s):
        started.set()
        assert release.wait(10), "the test never released the node"
        return {"quick_answer": "42"}

    async def past(seconds: float) -> None:
        return None

    monkeypatch.setattr(graph_mod, "quick_answer_node", quick_answer)
    env.app.bot_data["quick_graph"] = graph_mod.quick_answer_graph(checkpointer=MemorySaver())
    if forced:
        monkeypatch.setattr(runs, "_sleep", past)
    out = await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="what is x?", quick=True)
    h = common._inflight[out.run_id]
    assert await asyncio.to_thread(started.wait, 5)
    await history.cancel_cmd(_update(), _ctx(env, [out.run_id]))
    if not forced:
        release.set()
    await asyncio.wait([h.task], timeout=10)
    release.set()
    assert (await env.lib.get_run(out.run_id))["status"] == "cancelled"
    assert h.card.final == ("⏹ Force-cancelled during quick_answer." if forced
                            else "⏹ Cancelled after quick_answer.")
    assert not [t for t in _sent(env) + _edited(env) if "/continue" in t]


async def test_an_actor_waiting_on_a_failed_attach_attaches_again(env, monkeypatch):
    """R-a race: the console's approve attaches the untracked run and fails; the
    tap that waited on that handle looks the run up again and attaches itself.
    It gets the same refusal — never the not-in-flight notice over its card,
    since the run still rests at its plan gate."""
    run_id = await _at_plan_gate(env)
    common._inflight.clear()                                   # the restart

    async def down():
        await asyncio.sleep(0.01)                              # a live validation suspends, then fails
        raise EngineConfigError("proxy down")

    monkeypatch.setattr(registry, "active_snapshot", down)
    console = asyncio.ensure_future(runs.approve_plan(run_id, actor="webapp"))
    await asyncio.sleep(0)
    tap = await _tap(env, f"plan:approve:{run_id}")
    refusal = f"⚠️ Could not approve run `{run_id}`: proxy down"
    first = await console
    assert (first.code, first.text) == ("refused", refusal)
    tap.callback_query.edit_message_text.assert_not_awaited()
    assert _sent(env)[-1] == refusal                           # the tap's own outcome
    assert (await runs.snapshot(run_id))["awaiting"] == "plan"


async def test_a_tapped_card_of_the_other_gate_is_never_adopted(env):
    """R-a: a tapped card becomes the gate message only for its own gate. An old
    plan card tapped while the run rests at its report gate gets the plan
    buttons' refusal and is never adopted, so a later console Cancel leaves it."""
    run_id = await _at_plan_gate(env)
    plan_card = common._inflight[run_id].gate_msg[0]
    assert (await runs.approve_plan(run_id)).code == "started"
    await common._inflight[run_id].task                        # at its report gate
    common._inflight.clear()                                   # the restart
    edits = _edits(env, plan_card)
    for data in (f"plan:approve:{run_id}", f"plan:cancel:{run_id}"):
        sent = len(_sent(env))
        tap = await _tap(env, data, plan_card)
        tap.callback_query.edit_message_text.assert_not_awaited()
        assert _sent(env)[sent:] == [f"Run `{run_id}` is not at its plan gate."]
    assert common._inflight[run_id].gate_msg is None           # attached at its report gate, no card adopted
    assert (await env.lib.get_run(run_id))["status"] == "awaiting_report"
    assert (await runs.cancel_run(run_id, actor="webapp")).code == "cancelled"
    assert _edits(env, plan_card) == edits                     # the old plan card was never touched


@pytest.mark.parametrize("entry", ["tap", "continue"])
async def test_a_reattached_run_keeps_its_engines_profile(env, monkeypatch, entry):
    monkeypatch.setattr(registry, "resolve_snapshot",
                        AsyncMock(side_effect=lambda cfg, name: env.active.model_copy(update={"profile": name})))
    out = await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="topic", length="short",
                               engines_profile="hybrid-local")
    await common._inflight[out.run_id].task
    common._inflight.clear()                                   # the restart
    if entry == "tap":
        await _tap(env, f"len:short:{out.run_id}")             # attached by a knob tap
    else:
        assert (await runs.continue_run(await env.lib.get_run(out.run_id), chat_id=CHAT)).code == "attached"
    assert (await runs.plan_payload(out.run_id))["profile"] == "hybrid-local"
    assert (await runs.approve_plan(out.run_id)).code == "started"
    await common._inflight[out.run_id].task
    assert require((await _snap(env, out.run_id)).values).engines.profile == "hybrid-local"


# ---------------------------------------------------------------------------
# Fix round 3: the cancel note keeps next (I-1), drained-run hints (M-1), a
# missing profile (M-2), quick not-found (9a), /continue's retry (9b),
# idempotent appends
# ---------------------------------------------------------------------------

_FIRST_NODE = {"approve": "research", "re-plan": "brief", "extend": "deliver", "revise": "deliver",
               "resume": "outline", "fork": "extend_prep"}


class _Hold:
    """Holds one graph node while ``armed`` (its thread ignores any drain)."""

    def __init__(self) -> None:
        self.armed, self.started, self.release = False, threading.Event(), threading.Event()

    def __call__(self, fn):
        def node(s):
            if self.armed:
                self.started.set()
                assert self.release.wait(10), "the test never released the node"
            return fn(s)
        return node


async def _at_launch(env, kind: str):
    """The run where a ``kind`` drive launches from, and the action that launches it."""
    if kind in ("approve", "re-plan"):
        run_id = await _at_plan_gate(env)
        return run_id, ((lambda: runs.approve_plan(run_id)) if kind == "approve"
                        else (lambda: runs.edit_plan(run_id, "focus on security")))
    if kind in ("extend", "revise"):
        run_id = await _at_report_gate(env)
        return run_id, ((lambda: runs.report_action(run_id, "extend")) if kind == "extend"
                        else (lambda: runs.report_action(run_id, "revise", feedback="tighten it")))
    if kind == "fork":                                         # a sent run: /continue forks an Extend
        run_id = await _at_report_gate(env)
        assert (await runs.report_action(run_id, "send")).code == "delivered"
    else:                                                      # resume: cancelled after research, at ("outline",)
        research = _blocking(env, reads_runtime=True)
        run_id = await _executing(env, research)
        await runs.cancel_run(run_id)
        research.release.set()
        await common._inflight[run_id].task

    async def resume() -> runs.Outcome:
        return await runs.continue_run(await env.lib.get_run(run_id), chat_id=CHAT)
    return run_id, resume


@pytest.mark.parametrize("ending", ["drained_at_launch", "force_cancelled_in_its_first_node"])
@pytest.mark.parametrize("kind", list(_FIRST_NODE))
async def test_the_cancel_note_never_moves_next(env, monkeypatch, kind, ending):
    """I-1: the note is written as the last node the drive finished, else as the
    node the launching action wrote as — LangGraph's own guess re-runs the
    edges of the last node that ran, which moved a re-plan or a fork drained
    (or force-cancelled) before its first node finished back to its old gate."""
    first, hold = _FIRST_NODE[kind], _Hold()
    if first != "research":                                    # research is held by _blocking
        monkeypatch.setattr(graph_mod, f"{first}_node", hold(getattr(graph_mod, f"{first}_node")))
        env.graph = env.app.bot_data["graph"] = graph_mod.build_graph(checkpointer=env.graph.checkpointer)
    run_id, launch = await _at_launch(env, kind)
    writes, update_state = [], env.graph.aupdate_state

    async def recording(cfg, values, *a, **kw):
        if "cancel_note" not in values:
            return await update_state(cfg, values, *a, **kw)
        before = tuple((await env.graph.aget_state(cfg)).next)
        out = await update_state(cfg, values, *a, **kw)
        writes.append((before, tuple((await env.graph.aget_state(cfg)).next)))
        return out

    monkeypatch.setattr(env.graph, "aupdate_state", recording)
    if ending == "drained_at_launch":
        assert (await _prepared_across_the_shutdown_start(env, monkeypatch, launch)).code == "started"
    else:
        async def past(seconds: float) -> None:
            return None

        monkeypatch.setattr(runs, "_sleep", past)
        if first == "research":
            hold = _blocking(env)
        hold.armed = True
        assert (await launch()).code == "started"
        h = common._inflight[run_id]
        assert await asyncio.to_thread(hold.started.wait, 5)
        assert (await runs.cancel_run(run_id)).code == "draining"
        await asyncio.wait([h.task], timeout=10)
        hold.release.set()
        assert h.task.cancelled()
    assert len(writes) == 1 and writes[0][0] == writes[0][1], writes


async def test_a_drained_replan_resumes_its_redraft(env, monkeypatch):
    """I-1: a re-plan drained at its launch rests at ("brief",), so /continue
    redrafts the plan with the user's edit."""
    feedback, brief = [], graph_mod.brief_node

    def recording_brief(s):
        feedback.append(s.get("plan_feedback"))
        return brief(s)

    monkeypatch.setattr(graph_mod, "brief_node", recording_brief)
    env.graph = env.app.bot_data["graph"] = graph_mod.build_graph(checkpointer=env.graph.checkpointer)
    run_id = await _at_plan_gate(env)
    out = await _prepared_across_the_shutdown_start(env, monkeypatch,
                                                    lambda: runs.edit_plan(run_id, "focus on security"))
    assert out.code == "started"
    assert tuple((await _snap(env, run_id)).next) == ("brief",)
    _restart(monkeypatch)
    out = await runs.continue_run(await env.lib.get_run(run_id), chat_id=CHAT)
    assert out.code == "started" and "from `brief`" in out.text
    await common._inflight[run_id].task
    assert feedback[-1] == "focus on security"                 # the redraft saw the edit
    assert (await env.lib.get_run(run_id))["status"] == "awaiting_plan"


async def test_a_drained_fork_resumes_its_own_append_pass(env, monkeypatch):
    """I-1: a /continue fork drained at its launch rests at ("extend_prep",);
    its resumed drive runs its own append-only pass with the appends (once)
    and marks them, leaving no append_only behind."""
    run_id = await _at_report_gate(env)
    assert (await runs.report_action(run_id, "send")).code == "delivered"
    env.state.research = research = FetchAll()
    ref = "https://a.example/p"
    await history.append_cmd(_update(), _ctx(env, [run_id, ref]))
    await _prepared_across_the_shutdown_start(env, monkeypatch,
                                              lambda: history.continue_cmd(_update(), _ctx(env, [run_id])))
    snap = await _snap(env, run_id)
    assert tuple(snap.next) == ("extend_prep",) and snap.values["append_only"] is True
    assert await _source_statuses(env, run_id) == {ref: "pending"}
    _restart(monkeypatch)
    await history.continue_cmd(_update(), _ctx(env, [run_id]))
    await settle_runs()
    ((append_only, urls),) = research.seen
    assert append_only is True and urls.count(ref) == 1
    assert await _source_statuses(env, run_id) == {ref: "ingested"}
    snap = await _snap(env, run_id)
    assert tuple(snap.next) == ("deliver",) and not snap.values.get("append_only")


async def test_a_drained_runs_card_words_its_hint_from_next(env, monkeypatch):
    """M-1: away from a gate the drive resumes from ``next``; at a gate its card
    is re-attached, where the action is repeated; a drive drained before any
    node ran says it never started."""
    research = _blocking(env, reads_runtime=True)
    run_id = await _executing(env, research)
    h = common._inflight[run_id]
    await runs.cancel_run(run_id)
    research.release.set()
    await h.task
    assert h.card.final == f"⏹ Cancelled after research — /continue {run_id} resumes it from outline."
    other = await _at_report_gate(env, "another run")
    h = common._inflight[other]
    await _prepared_across_the_shutdown_start(env, monkeypatch, lambda: runs.report_action(other, "extend"))
    assert h.card.final == (f"⏹ Cancelled before it started — /continue {other} re-attaches the report card — "
                            "repeat the action there.")
    assert (await _snap(env, other)).values["cancel_note"] == "cancelled before it started (shutdown)"


async def test_a_reattached_runs_profile_gone_from_the_engines_file(env, monkeypatch):
    """M-2: after a restart whose engines file dropped the run's profile, the
    card shows the problem, the profile knob cycles from the first profile,
    and Approve refuses until a profile that exists is chosen."""
    async def resolve(cfg, name):
        registry.bindings(cfg, name)                           # an unknown profile raises, as the real one
        return env.active.model_copy(update={"profile": name})

    monkeypatch.setattr(registry, "resolve_snapshot", resolve)
    out = await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="topic", length="short",
                               engines_profile="hybrid-local")
    run_id = out.run_id
    await common._inflight[run_id].task
    common._inflight.clear()                                   # the restart, with an edited engines file
    cfg = registry.current_config()
    monkeypatch.setattr(registry._state, "cfg", cfg.model_copy(update={
        "profiles": {k: v for k, v in cfg.profiles.items() if k != "hybrid-local"}}))
    names = list(registry.current_config().profiles)
    problem = f"unknown profile 'hybrid-local' (known: {', '.join(names)})"
    await _tap(env, f"len:short:{run_id}")                     # attached by a knob tap
    p = await runs.plan_payload(run_id)
    assert (p["profile"], p["profile_problem"]) == ("hybrid-local", problem)
    out = await runs.approve_plan(run_id)
    assert (out.code, out.text) == ("refused", f"⚠️ Could not approve run `{run_id}`: {problem}")
    sent = len(_sent(env))
    await _tap(env, f"prof:next:{run_id}")
    assert _sent(env)[sent:] == []                             # no "is not in list" refusal
    assert (await runs.plan_payload(run_id))["profile"] == names[0]
    assert (await runs.approve_plan(run_id)).code == "started"


async def test_cancel_on_a_finished_ask_names_no_continue(env, monkeypatch):
    """9a: /cancel <id> on a finished /ask: nothing to cancel or continue."""
    monkeypatch.setattr(graph_mod, "quick_answer_node", lambda s: {"quick_answer": "42"})
    env.app.bot_data["quick_graph"] = graph_mod.quick_answer_graph(checkpointer=MemorySaver())
    out = await runs.start_run(chat_id=CHAT, user_id=CHAT, topic="what is x?", quick=True)
    await common._inflight[out.run_id].task
    await history.cancel_cmd(_update(), _ctx(env, [out.run_id]))
    assert _sent(env)[-1] == f"(run {out.run_id} was a quick /ask — nothing to cancel or continue)"


async def test_continue_behind_a_failed_attach_gets_its_own_outcome(env, monkeypatch):
    """9b: /continue waiting on a handle the console's failed attach forgets
    registers the run again, once, and gets its own outcome."""
    run_id = await _at_plan_gate(env)
    common._inflight.clear()                                   # the restart
    row, fail = await env.lib.get_run(run_id), asyncio.Event()

    async def down():
        await fail.wait()                                      # a live validation suspends, then fails
        raise EngineConfigError("proxy down")

    monkeypatch.setattr(registry, "active_snapshot", down)
    console = asyncio.ensure_future(runs.approve_plan(run_id, actor="webapp"))
    for _ in range(100):
        if run_id in common._inflight:
            break
        await asyncio.sleep(0.01)
    assert run_id in common._inflight                          # the console's handle, attaching
    cont = asyncio.ensure_future(runs.continue_run(row, chat_id=CHAT))
    await asyncio.sleep(0.05)
    assert not cont.done()                                     # it waits for that handle's lock
    fail.set()
    assert (await console).code == "refused"
    out = await cont
    assert (out.code, out.text) == ("refused", f"⚠️ Could not continue run `{run_id}`: proxy down")


async def test_a_drained_extends_appends_are_merged_once_and_marked_by_the_next(env, monkeypatch):
    """Idempotent appends: an Extend drained at its launch leaves its appends
    pending (merged into sources); the next Extend merges none twice, and its
    report gate marks them."""
    run_id = await _at_report_gate(env)
    env.state.research = research = FetchAll()
    ref = "https://a.example/p"
    await history.append_cmd(_update(), _ctx(env, [run_id, ref]))
    out = await _prepared_across_the_shutdown_start(env, monkeypatch, lambda: runs.report_action(run_id, "extend"))
    assert out.code == "started"
    assert await _source_statuses(env, run_id) == {ref: "pending"}
    _restart(monkeypatch)
    assert (await runs.report_action(run_id, "extend")).code == "started"
    await settle_runs()
    assert [s["url"] for s in (await _snap(env, run_id)).values["sources"]].count(ref) == 1
    ((append_only, urls),) = research.seen
    assert not append_only and urls.count(ref) == 1
    assert await _source_statuses(env, run_id) == {ref: "ingested"}
