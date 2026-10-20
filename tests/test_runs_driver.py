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

from argus.bot import common, runs
from argus.bot.progress import RunEventBus
from argus.config import Settings
from argus.engines import registry, search
from argus.engines.registry import EngineConfigError
from argus.graph import graph as graph_mod
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
    returns early with the ``cancelled:`` line, as graph/research.py does."""

    def __init__(self, *, block: bool = True, reads_runtime: bool = False, items: int = 3,
                 fail: Exception | None = None) -> None:
        self.block, self.reads_runtime, self.items, self.fail = block, reads_runtime, items, fail
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
    assert order == ["preview", f"update_run {run_id} {{'exa_usd': 0.042}}"]
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


async def test_extend_and_revise_refuse_an_older_topology_checkpoint(env):
    """v2 checkpoints name nodes this graph lacks (fetcher, synthesizer, …); a
    resume from their report gate runs no node and stops at the same gate."""
    old = StateGraph(ArgusState)
    for name in ("fetcher", "report_builder", "deliver"):
        old.add_node(name, lambda s: {})
    old.add_edge(START, "fetcher")
    old.add_edge("fetcher", "report_builder")
    old.add_edge("report_builder", "deliver")
    old.add_edge("deliver", END)
    saver = env.graph.checkpointer
    old = old.compile(checkpointer=saver, interrupt_before=["deliver"])
    cfg = {"configurable": {"thread_id": f"tg:{CHAT}:0ld0ld00"}}
    await old.ainvoke({"user_request": "t", "length": "short", "run_config": make_run_config("short"),
                       "report_paths": {"md": "r.md", "folder": "f"}}, cfg)
    await env.lib.create_run(run_id="0ld0ld00", thread_id=cfg["configurable"]["thread_id"], chat_id=CHAT,
                             topic="v2 run", status="done", config_json="{}", depth="quick", origin="telegram")
    out = await runs.continue_run(await env.lib.get_run("0ld0ld00"), chat_id=CHAT)
    assert out.code == "attached"                               # the preview is re-sent; Send still works
    for action, feedback in (("extend", None), ("revise", "more detail")):
        out = await runs.report_action("0ld0ld00", action, feedback=feedback)
        assert out.code == "refused"
        assert "older Argus graph" in out.text and "fetcher" in out.text
    assert (await env.graph.aget_state(cfg)).values.get("extend_requested") is None


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
