"""Tests for the plan-approval HITL gate (v2 run-scoped callbacks, v4 run driver).

Locks down the contract that:

1. Tapping a *Length* button during the plan-approval gate MUST only
   update the chosen length and re-render the plan message (WITH the
   keyboard — see 3). It MUST NOT resume the LangGraph. Resuming right
   after picking the length forces the user past the plan-review
   window, and any subsequent Approve-tap fires a second
   ``Command(resume=True)`` on an already-running graph (the "approve
   button unpressable / research plan skipped" failure mode reported
   by Albert on 2026-07-10).

2. Tapping *Approve* IS the only path that resumes the graph — and it
   resumes at most ONCE per plan gate, even if double-tapped.

3. Telegram's ``editMessageText`` REMOVES the inline keyboard when
   ``reply_markup`` is omitted. So every edit that must keep the gate
   interactive (Length taps, Edit mode) MUST re-send the plan keyboard,
   or the Approve button vanishes and the run dead-locks at the gate.

4. Callbacks are run-scoped (``plan:approve:<run8>``): a button whose
   run is not in flight must be answered with a clear notice, never
   applied to some other run in the same chat.

These tests are unit tests — no Telegram token, no FreeLLMAPI. They
exercise ``on_callback`` and the run driver (``bot/runs.py``) on a
recording graph double at its plan gate.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.types import Command
from telegram import InlineKeyboardMarkup

from argus.bot import app as bot_app
from argus.bot import common as bot_common
from argus.bot import runs
from argus.bot.app import on_callback
from argus.config import Settings
from argus.engines import registry
from argus.graph.run_config import require

from helpers import fake_engines_snapshot, make_run_config, settle_runs

RUN_ID = "ab12cd34"


@pytest.fixture(autouse=True)
def _dummy_settings(monkeypatch: pytest.MonkeyPatch):
    """Handlers call get_settings() on entry; CI has no .env, so give
    the bot module a synthetic Settings instead of requiring secrets."""
    dummy = Settings(
        freellmapi_base_url="http://127.0.0.1:3001/v1",
        freellmapi_api_key="freellmapi-test",
        telegram_bot_token="123:test",
        telegram_allowed_user_id=1,
        reports_root=Path("reports"),
        checkpoint_db=Path("cp.sqlite"),
        library_db=Path("lib.sqlite"),
        vault_root=Path("vault"),
        media_root=Path("vault/media"),
        transcripts_root=Path("vault/transcripts"),
        history_root=Path("vault/history"),
        ffmpeg_path=None,
    )
    monkeypatch.setattr(bot_app, "get_settings", lambda: dummy)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def _clear_inflight():
    """Each test starts with a clean in-flight registry so a leaked
    entry from a prior test doesn't pollute the assertion."""
    bot_common._inflight.clear()
    bot_common._pending_reply.clear()
    yield
    await settle_runs()
    bot_common._inflight.clear()
    bot_common._pending_reply.clear()


@pytest.fixture(autouse=True)
def _bypass_acl(monkeypatch: pytest.MonkeyPatch):
    """Force ``_allowed(...)`` to return True for any user id, so the
    dispatcher actually runs instead of rejecting the mock user."""
    monkeypatch.setattr(bot_app, "_allowed", lambda _settings, _uid: True)


@pytest.fixture(autouse=True)
async def _engines_resolved(monkeypatch: pytest.MonkeyPatch):
    """Approve first resolves the engines snapshot (v4 WP-A/WP-D); the plan
    card names the profiles of the shipped engines file (no network)."""
    monkeypatch.setattr(registry, "_state", registry._State())
    await registry.load_config()
    monkeypatch.setattr(registry, "active_snapshot",
                        AsyncMock(return_value=fake_engines_snapshot()))


class _Graph:
    """A plan-gate checkpoint; each ``astream`` call is one drive, recorded.
    A resume (``Command``) moves the run on to its report gate, as the real
    graph does."""

    def __init__(self, length: str):
        self.values = {"length": length, "run_config": make_run_config(length), "user_request": "topic",
                       "plan": {"summary": "Test plan summary", "sub_questions": ["q1"], "planned_sources": []},
                       "sources": [{"title": "W", "url": "https://w.example/x"}]}
        self.next = ("research",)
        self.drives: list = []
        self.nodes: dict = {}
        self.checkpointer = SimpleNamespace(aget_tuple=AsyncMock(return_value=None))

    async def aget_state(self, _cfg):
        return SimpleNamespace(values=self.values, next=self.next, created_at="2026-10-19T00:00:00+00:00")

    async def aupdate_state(self, _cfg, values, as_node=None):
        self.values = {**self.values, **values}

    def astream(self, resume, _cfg, **_kw):
        self.drives.append(resume)
        if isinstance(resume, Command):
            self.next = ("deliver",)

        async def _gen():
            return
            yield

        return _gen()


def _make_ctx(graph) -> MagicMock:
    """Stand-in for ``ContextTypes.DEFAULT_TYPE`` with awaitable bot
    methods and a REAL bot_data dict (so registry lookups behave)."""
    application = MagicMock()
    application.bot = MagicMock()
    ids = iter(range(100, 10_000))
    application.bot.send_message = AsyncMock(side_effect=lambda **kw: SimpleNamespace(message_id=next(ids)))
    application.bot.edit_message_text = AsyncMock()
    application.bot.send_chat_action = AsyncMock()
    application.bot.send_document = AsyncMock()
    application.bot_data = {"graph": graph}          # no library in unit tests
    ctx = MagicMock()
    ctx.application = application
    return ctx


async def _at_plan_gate(*, chat_id: int, length: str = "medium"):
    """A run resting at its plan gate, attached through /continue's path;
    returns (graph, ctx, the plan card's message id)."""
    graph = _Graph(length)
    ctx = _make_ctx(graph)
    await runs.init(ctx.application)
    out = await runs.continue_run({"run_id": RUN_ID, "thread_id": f"tg:{chat_id}:{RUN_ID}", "topic": "topic"},
                                  chat_id=chat_id)
    assert out.code == "attached", out
    return graph, ctx, bot_common._inflight[RUN_ID].gate_msg[0]


def _make_callback_update(*, chat_id: int, callback_data: str,
                          message_text: str = "Plan text") -> MagicMock:
    q = MagicMock()
    q.data = callback_data
    q.answer = AsyncMock()
    q.edit_message_text = AsyncMock()
    q.message = MagicMock()
    q.message.chat.id = chat_id
    q.message.text = message_text

    update = MagicMock()
    update.effective_user.id = chat_id
    update.effective_chat.id = chat_id
    update.callback_query = q
    return update


def _card_edits(ctx, card_id: int) -> list[dict]:
    return [c.kwargs for c in ctx.application.bot.edit_message_text.await_args_list
            if c.kwargs.get("message_id") == card_id]


# ---------------------------------------------------------------------------
# Length tap contract
# ---------------------------------------------------------------------------


async def test_len_tap_does_NOT_resume_graph():
    graph, ctx, _ = await _at_plan_gate(chat_id=999010)
    await on_callback(_make_callback_update(chat_id=999010, callback_data=f"len:long:{RUN_ID}"), ctx)

    assert (await runs.plan_payload(RUN_ID))["length"] == "long", "a Length tap updates the chosen length"
    assert graph.drives == [], (
        "Tapping a Length button must NOT resume the graph — "
        "the Approve button is the single resume gate.")
    assert require(graph.values).length == "medium", "the checkpoint changes only on Approve"


async def test_len_tap_re_renders_plan_with_updated_label_AND_keyboard():
    """Telegram removes the inline keyboard on editMessageText unless
    reply_markup is re-sent. A Length tap must therefore re-send the
    plan keyboard (with the ✅ moved to the tapped mode) or the user
    loses the Approve button — the dead-lock Albert hit live."""
    _, ctx, card_id = await _at_plan_gate(chat_id=10011)
    await on_callback(_make_callback_update(chat_id=10011, callback_data=f"len:lecture:{RUN_ID}"), ctx)

    (edit,) = _card_edits(ctx, card_id)
    assert "Lecture" in edit["text"], f"Edited plan message must mention the chosen length; got {edit['text']!r}"
    markup = edit.get("reply_markup")
    assert isinstance(markup, InlineKeyboardMarkup), (
        "Length tap must re-send the plan keyboard — editMessageText "
        "without reply_markup REMOVES the buttons (Telegram Bot API).")
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert any("Approve" in l for l in labels), "Re-sent keyboard must still contain Approve."
    assert any(l.startswith("Lecture") and "✅" in l for l in labels), (
        f"The ✅ marker must move onto the tapped length; labels: {labels}")
    # And the re-sent buttons stay run-scoped.
    datas = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert all(d.endswith(f":{RUN_ID}") for d in datas), datas


async def test_repeated_len_taps_do_not_stack_status_suffixes():
    """Each re-render builds the card from the plan, not from the
    previously-edited message — otherwise every tap would append another
    status line forever."""
    _, ctx, card_id = await _at_plan_gate(chat_id=10015)
    for mode in ("long", "tldr", "lecture"):
        await on_callback(_make_callback_update(chat_id=10015, callback_data=f"len:{mode}:{RUN_ID}"), ctx)
    last = _card_edits(ctx, card_id)[-1]["text"]
    assert last.count("Research plan") == 1 and last.count("Test plan summary") == 1
    assert last == bot_common._inflight[RUN_ID].gate_msg[1], "the stored card text is the rendered card"


# ---------------------------------------------------------------------------
# Approve contract
# ---------------------------------------------------------------------------


async def test_approve_tap_DOES_resume_graph_with_chosen_length():
    graph, ctx, _ = await _at_plan_gate(chat_id=10012)
    await on_callback(_make_callback_update(chat_id=10012, callback_data=f"len:long:{RUN_ID}"), ctx)
    await on_callback(_make_callback_update(chat_id=10012, callback_data=f"plan:approve:{RUN_ID}"), ctx)
    await settle_runs()

    assert len(graph.drives) == 1 and isinstance(graph.drives[0], Command), (
        f"Approve tap must resume exactly once, got {graph.drives!r}.")
    rc = require(graph.values)
    assert (rc.length, graph.values["length"]) == ("long", "long"), (
        "Approve must carry the previously-chosen length into the checkpoint.")


async def test_len_then_approve_runs_resume_only_once():
    graph, ctx, _ = await _at_plan_gate(chat_id=10013)
    await on_callback(_make_callback_update(chat_id=10013, callback_data=f"len:short:{RUN_ID}"), ctx)
    assert graph.drives == []
    await on_callback(_make_callback_update(chat_id=10013, callback_data=f"plan:approve:{RUN_ID}"), ctx)
    await settle_runs()
    assert len(graph.drives) == 1


async def test_double_approve_resumes_only_once():
    """Fat-thumb guard: two Approve taps must not start two astream
    sessions on the same thread (the race behind the original bug)."""
    graph, ctx, _ = await _at_plan_gate(chat_id=10016)
    for _ in range(2):
        await on_callback(_make_callback_update(chat_id=10016, callback_data=f"plan:approve:{RUN_ID}"), ctx)
    await settle_runs()
    assert len(graph.drives) == 1, f"Double-tapped Approve must resume exactly once, got {graph.drives!r}."


async def test_edit_tap_does_not_resume_and_keeps_keyboard():
    graph, ctx, card_id = await _at_plan_gate(chat_id=10014)
    await on_callback(_make_callback_update(chat_id=10014, callback_data=f"plan:edit:{RUN_ID}"), ctx)

    assert graph.drives == [], "Edit tap must not resume the graph."
    assert bot_common._pending_reply[10014] == ("plan_edit", RUN_ID), (
        "Edit tap must arm the chat's next reply as the plan feedback.")
    # Edit mode must keep the gate interactive too.
    (edit,) = _card_edits(ctx, card_id)
    assert isinstance(edit.get("reply_markup"), InlineKeyboardMarkup), (
        "Edit tap must re-send the keyboard — the user may still Approve.")


# ---------------------------------------------------------------------------
# Run scoping
# ---------------------------------------------------------------------------


async def test_callback_for_unknown_run_is_answered_not_applied():
    graph, ctx, _ = await _at_plan_gate(chat_id=10017)          # RUN_ID in flight
    update = _make_callback_update(chat_id=10017, callback_data="plan:approve:deadbeef")
    await on_callback(update, ctx)
    assert graph.drives == [], (
        "A button for a run that is not in flight must never resume "
        "another run in the same chat.")
    update.callback_query.edit_message_text.assert_awaited()
    args = update.callback_query.edit_message_text.await_args.args
    text = update.callback_query.edit_message_text.await_args.kwargs.get("text") or (args[0] if args else "")
    assert "deadbeef" in text, "notice should name the unknown run id"


async def test_stale_pre_v2_button_gets_notice():
    """Buttons minted before the v2 upgrade carry 2-part data. They must
    be answered with a clear notice, not crash or mis-route."""
    graph, ctx, _ = await _at_plan_gate(chat_id=10018)
    update = _make_callback_update(chat_id=10018, callback_data="plan:approve")
    await on_callback(update, ctx)
    assert graph.drives == []
    update.callback_query.edit_message_text.assert_awaited()
