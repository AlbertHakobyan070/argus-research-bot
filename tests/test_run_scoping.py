"""v2 run-scoping tests — per-run thread ids + messages reducer.

Phase 1 contracts:

1. Every /research invocation mints its OWN run id and checkpoint
   thread (``tg:<chat>:<run8>``). The old scheme reused ``tg:<chat>``
   for every run in a chat, so a new run overwrote / entangled the
   previous run's checkpoint history and nothing was resumable.

2. ``ArgusState.messages`` accumulates across nodes. The previous
   annotation metadata was the STRING "appended-only chat history",
   which LangGraph ignores — the channel was silently last-write-wins.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from argus import bot as bot_mod
from argus.bot import _inflight, research_cmd


@pytest.fixture(autouse=True)
def _clear_inflight():
    _inflight.clear()
    yield
    _inflight.clear()


@pytest.fixture(autouse=True)
def _bypass_acl(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bot_mod, "_allowed", lambda _s, _uid: True)


class _FakeGraph:
    """Async-graph double: streams one planner event, then reports a
    plan in the state snapshot (so research_cmd reaches the plan gate)."""

    def __init__(self):
        self.seen_thread_ids: list[str] = []

    def astream(self, _state, config=None, stream_mode=None):
        self.seen_thread_ids.append(config["configurable"]["thread_id"])

        async def _gen():
            yield {"planner": {"plan": {"summary": "s"}}}

        return _gen()

    async def aget_state(self, _cfg):
        return SimpleNamespace(values={"plan": {
            "summary": "s", "sub_questions": ["q"], "planned_sources": []}})


def _make_update(chat_id: int) -> MagicMock:
    u = MagicMock()
    u.effective_user.id = chat_id
    u.effective_chat.id = chat_id
    u.message.reply_text = AsyncMock()
    return u


def _make_ctx(graph) -> MagicMock:
    application = MagicMock()
    application.bot = MagicMock()
    application.bot.send_message = AsyncMock(
        return_value=SimpleNamespace(message_id=1))
    application.bot.edit_message_text = AsyncMock()
    application.bot.send_chat_action = AsyncMock()
    application.bot_data = {"graph": graph}   # no library in unit tests
    ctx = MagicMock()
    ctx.application = application
    ctx.args = ["some", "topic"]
    return ctx


async def test_each_research_gets_its_own_thread_id():
    graph = _FakeGraph()
    ctx = _make_ctx(graph)

    await research_cmd(_make_update(777), ctx)
    await research_cmd(_make_update(777), ctx)

    assert len(graph.seen_thread_ids) == 2
    t1, t2 = graph.seen_thread_ids
    assert t1 != t2, (
        "two /research runs in one chat must NOT share a checkpoint "
        f"thread (got {t1!r} twice)")
    for t in (t1, t2):
        parts = t.split(":")
        assert parts[0] == "tg" and parts[1] == "777" and len(parts[2]) == 8, (
            f"thread id must look like tg:<chat>:<run8>, got {t!r}")

    # Both runs are individually tracked in flight, keyed by run id.
    assert len(_inflight) == 2
    for run_id, info in _inflight.items():
        assert info["thread_id"].endswith(run_id)
        assert info["awaiting"] == "plan_approval"


async def test_plan_keyboard_of_each_run_is_scoped_to_it():
    graph = _FakeGraph()
    ctx = _make_ctx(graph)
    await research_cmd(_make_update(778), ctx)

    (run_id,) = list(_inflight)
    # The plan message was sent through bot.send_message with the keyboard.
    sends = ctx.application.bot.send_message.await_args_list
    markups = [c.kwargs.get("reply_markup") for c in sends
               if c.kwargs.get("reply_markup") is not None]
    assert markups, "plan message must carry the approval keyboard"
    datas = [b.callback_data
             for row in markups[-1].inline_keyboard for b in row]
    assert all(d.endswith(f":{run_id}") for d in datas), datas


def test_messages_channel_accumulates_across_nodes():
    """A two-node graph over ArgusState must APPEND to messages, not
    overwrite — regression for the string-annotation reducer bug."""
    from langgraph.graph import END, START, StateGraph
    from argus.graph.state import ArgusState

    def node_a(_state):
        return {"messages": [{"role": "assistant", "content": "a"}]}

    def node_b(_state):
        return {"messages": [{"role": "assistant", "content": "b"}]}

    g = StateGraph(ArgusState)
    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    out = g.compile().invoke({"messages": []})

    contents = [m["content"] for m in out["messages"]]
    assert contents == ["a", "b"], (
        f"messages must accumulate across nodes (append reducer); "
        f"got {contents!r} — last-write-wins means the reducer regressed")
