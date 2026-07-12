"""Startup reconciliation of runs orphaned by a crash/restart (2026-07-12).

A bot restart wipes the in-memory _inflight registry, so any run left in
a non-terminal status has dead inline buttons — the user is left staring
at a frozen chat. _reconcile_orphaned_runs inspects each orphan's
checkpoint and either marks it recoverable (awaiting_*) with a /continue
nudge, or errors it, and DMs the user either way.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from argus import bot as bot_mod
from argus.bot import _reconcile_orphaned_runs
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


def _graph_with_next(next_map: dict[str, tuple]):
    """Fake graph whose aget_state returns snap.next based on thread_id."""
    g = MagicMock()

    async def aget_state(cfg):
        tid = cfg["configurable"]["thread_id"]
        return SimpleNamespace(values={"plan": {}},
                               next=next_map.get(tid, ()))
    g.aget_state = aget_state
    return g


async def _seed(lib, run_id, chat_id, status, topic="t"):
    return await lib.create_run(
        run_id=run_id, thread_id=f"tg:{chat_id}:{run_id}", chat_id=chat_id,
        topic=topic, status=status)


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


async def test_orphan_mid_pipeline_is_errored(lib):
    # Frozen at a non-gate node (e.g. synthesizer) — not resumable in place.
    await _seed(lib, "cccc3333", 7, "running")
    graph = _graph_with_next({"tg:7:cccc3333": ("synthesizer",)})
    app = _app(lib, graph)

    await _reconcile_orphaned_runs(app)

    assert (await lib.get_run("cccc3333"))["status"] == "error"
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
