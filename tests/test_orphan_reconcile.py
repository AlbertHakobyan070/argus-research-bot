"""Startup reconciliation of runs orphaned by a crash/restart (2026-07-12).

A bot restart wipes the in-memory _inflight registry, so any run left in
a non-terminal status has no drive behind it — the user is left staring
at a frozen chat (its buttons attach a run resting at a gate on the next
tap). _reconcile_orphaned_runs inspects each orphan's
checkpoint and marks it recoverable — awaiting_* at a gate, cancelled
mid-drive (Task 10 ruling) — with a /continue nudge, or errors it when it
has no checkpoint, and DMs the user either way. A quick /ask (which
/continue refuses) is cancelled and told to ask again.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from argus.bot.app import _reconcile_orphaned_runs
from argus.library import Library


@pytest.fixture
async def lib(tmp_path):
    l = Library(tmp_path / "lib.sqlite")
    await l.open()
    yield l
    await l.close()


def _app(lib, graph):
    app = MagicMock()
    app.bot = MagicMock()
    app.bot.send_message = AsyncMock()
    app.bot_data = {"library": lib, "graph": graph}
    return app


def _graph_with_next(next_map: dict[str, tuple], *, no_checkpoint: tuple[str, ...] = ()):
    """Fake graph whose aget_state returns snap.next based on thread_id (a
    thread in ``no_checkpoint`` has none: empty values, as LangGraph reports)."""
    g = MagicMock()

    async def aget_state(cfg):
        tid = cfg["configurable"]["thread_id"]
        if tid in no_checkpoint:
            return SimpleNamespace(values={}, next=())
        return SimpleNamespace(values={"plan": {}},
                               next=next_map.get(tid, ()))
    g.aget_state = aget_state
    return g


async def _seed(lib, run_id, chat_id, status, topic="t", mode="deep"):
    return await lib.create_run(
        run_id=run_id, thread_id=f"tg:{chat_id}:{run_id}", chat_id=chat_id,
        topic=topic, mode=mode, status=status, config_json="{}", depth="quick", origin="telegram")


async def test_orphan_at_plan_gate_becomes_awaiting_plan_and_notifies(lib):
    await _seed(lib, "aaaa1111", 5, "running", "langchain vs langgraph")
    graph = _graph_with_next({"tg:5:aaaa1111": ("fetcher",)})
    app = _app(lib, graph)

    await _reconcile_orphaned_runs(app)

    run = await lib.get_run("aaaa1111")
    assert run["status"] == "awaiting_plan", "plan-gate orphan is recoverable"
    app.bot.send_message.assert_awaited_once()
    kw = app.bot.send_message.await_args.kwargs
    assert kw["chat_id"] == 5
    assert "/continue aaaa1111" in kw["text"]


async def test_orphan_at_report_gate_becomes_awaiting_report(lib):
    await _seed(lib, "bbbb2222", 6, "running")
    graph = _graph_with_next({"tg:6:bbbb2222": ("deliver",)})
    app = _app(lib, graph)

    await _reconcile_orphaned_runs(app)

    assert (await lib.get_run("bbbb2222"))["status"] == "awaiting_report"
    assert "/continue bbbb2222" in app.bot.send_message.await_args.kwargs["text"]


async def test_orphan_mid_drive_is_cancelled_and_resumable(lib):
    """Task 10 ruling: a run interrupted mid-drive (its checkpoint's next is
    not a gate) is cancelled — /continue resumes it from its last saved step."""
    await _seed(lib, "cccc3333", 7, "running", "langchain vs langgraph")
    graph = _graph_with_next({"tg:7:cccc3333": ("compose",)})
    app = _app(lib, graph)

    await _reconcile_orphaned_runs(app)

    assert (await lib.get_run("cccc3333"))["status"] == "cancelled"
    assert app.bot.send_message.await_args.kwargs["text"] == (
        "♻️ Run `cccc3333` (langchain vs langgraph) was interrupted by a bot restart mid-step. "
        "/continue cccc3333 resumes it from its last saved step.")


@pytest.mark.parametrize("nxt", [(), ("deliver",)], ids=["mid_drive", "at_its_deliver_step"])
async def test_a_quick_orphan_is_cancelled_and_told_to_ask_again(lib, nxt):
    """continue_run refuses quick runs: a quick /ask orphan is cancelled with
    its own DM — also when its thread, read through the deep graph, rests at
    ("deliver",) (the quick graph's own deliver step)."""
    await _seed(lib, "qqqq1111", 7, "running", "what is x?", mode="quick")
    graph = _graph_with_next({"tg:7:qqqq1111": nxt})
    app = _app(lib, graph)

    await _reconcile_orphaned_runs(app)

    assert (await lib.get_run("qqqq1111"))["status"] == "cancelled"
    assert app.bot.send_message.await_args.kwargs["text"] == (
        "♻️ /ask `qqqq1111` (what is x?) was interrupted by a bot restart — ask again with /ask.")


async def test_orphan_without_a_checkpoint_is_errored(lib):
    # No checkpoint to resume from — the run cannot be resumed at all.
    await _seed(lib, "ffff6666", 7, "planning")
    graph = _graph_with_next({}, no_checkpoint=("tg:7:ffff6666",))
    app = _app(lib, graph)

    await _reconcile_orphaned_runs(app)

    assert (await lib.get_run("ffff6666"))["status"] == "error"
    assert "research it again" in app.bot.send_message.await_args.kwargs["text"].lower()


async def test_terminal_runs_are_left_alone(lib):
    await _seed(lib, "dddd4444", 8, "done")
    graph = _graph_with_next({})
    app = _app(lib, graph)
    await _reconcile_orphaned_runs(app)
    assert (await lib.get_run("dddd4444"))["status"] == "done"
    app.bot.send_message.assert_not_awaited()


async def test_notify_failure_does_not_crash_reconcile(lib):
    await _seed(lib, "eeee5555", 9, "awaiting_plan")
    graph = _graph_with_next({"tg:9:eeee5555": ("fetcher",)})
    app = _app(lib, graph)
    app.bot.send_message = AsyncMock(side_effect=RuntimeError("blocked by user"))
    # Must still update the registry status even if the DM fails.
    await _reconcile_orphaned_runs(app)
    assert (await lib.get_run("eeee5555"))["status"] == "awaiting_plan"


@pytest.mark.parametrize("values,status", [
    ({"plan": {}}, "awaiting_plan"),                                   # its first pass: the plan gate
    ({"plan": {}, "extend_rounds": 1}, "cancelled"),                   # an Extend interrupted mid-research
    ({"plan": {}, "research_rounds": 3, "coverage_gaps": [1]}, "cancelled"),   # a gap wave
], ids=["first_pass", "extend", "gap_wave"])
async def test_an_orphan_at_research_is_the_plan_gate_only_before_its_first_pass(lib, values, status):
    """Live check 12: ("research",) is the plan gate only while research's next
    pass would be the first (runs._gate) — the reconcile agrees with the driver."""
    await _seed(lib, "eeee5555", 8, "running", "an extend")
    graph = MagicMock()
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(values=values, next=("research",)))
    app = _app(lib, graph)

    await _reconcile_orphaned_runs(app)

    assert (await lib.get_run("eeee5555"))["status"] == status
    text = app.bot.send_message.await_args.kwargs["text"]
    assert ("at the plan step" in text) is (status == "awaiting_plan"), text
    assert "/continue eeee5555" in text
