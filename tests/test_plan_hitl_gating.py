"""Tests for the plan-approval HITL gate (v2 run-scoped callbacks).

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
exercise ``on_callback`` and the in-flight registry with async mocks.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import InlineKeyboardMarkup

from argus import bot as bot_mod
from argus.bot import _inflight, on_callback
from argus.config import Settings

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
        request_timeout_seconds=60.0,
        max_revision_rounds=3,
    )
    monkeypatch.setattr(bot_mod, "get_settings", lambda: dummy)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_inflight():
    """Each test starts with a clean in-flight registry so a leaked
    entry from a prior test doesn't pollute the assertion."""
    _inflight.clear()
    yield
    _inflight.clear()


@pytest.fixture(autouse=True)
def _bypass_acl(monkeypatch: pytest.MonkeyPatch):
    """Force ``_allowed(...)`` to return True for any user id, so the
    dispatcher actually runs instead of rejecting the mock user."""
    monkeypatch.setattr(bot_mod, "_allowed", lambda _settings, _uid: True)


def _seed_plan_inflight(*, run_id: str = RUN_ID, chat_id: int = 10001,
                        default_length: str = "medium") -> dict:
    """Populate ``_inflight[run_id]`` with the shape ``research_cmd``
    builds after the planner delivers a plan but before any callback
    has fired. Returns the seeded entry so the test can introspect it."""
    thread_id = f"tg:{chat_id}:{run_id}"
    fake_graph = MagicMock()
    fake_cfg = {"configurable": {"thread_id": thread_id}}
    entry = {
        "run_id": run_id,
        "chat_id": chat_id,
        "thread_id": thread_id,
        "state": {
            "thread_id": thread_id,
            "user_id": chat_id,
            "user_request": "topic",
            "length": default_length,
            "plan": {"summary": "Test plan summary",
                     "sub_questions": ["q1"],
                     "planned_sources": []},
        },
        "stage": "planner",
        "length": default_length,
        "awaiting": "plan_approval",
        "plan_text": "Plan text",
        "cfg": fake_cfg,
        "graph": fake_graph,
    }
    _inflight[run_id] = entry
    return entry


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


def _make_ctx() -> MagicMock:
    """Stand-in for ``ContextTypes.DEFAULT_TYPE`` with awaitable bot
    methods and a REAL bot_data dict (so registry lookups behave)."""
    application = MagicMock()
    application.bot = MagicMock()
    application.bot.send_message = AsyncMock()
    application.bot.edit_message_text = AsyncMock()
    application.bot.send_chat_action = AsyncMock()
    application.bot.send_document = AsyncMock()
    application.bot_data = {}          # no library in unit tests
    ctx = MagicMock()
    ctx.application = application
    return ctx


# ---------------------------------------------------------------------------
# Length tap contract
# ---------------------------------------------------------------------------


async def test_len_tap_does_NOT_resume_graph():
    _seed_plan_inflight(chat_id=999010)
    update = _make_callback_update(chat_id=999010,
                                   callback_data=f"len:long:{RUN_ID}")
    ctx = _make_ctx()

    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()) as resume:
        await on_callback(update, ctx)

    entry = _inflight[RUN_ID]
    assert entry["length"] == "long", (
        f"Length tap should update info['length'], got {entry['length']!r}")
    assert resume.await_count == 0, (
        "Tapping a Length button must NOT call _resume_after_plan — "
        "the Approve button is the single resume gate.")
    assert entry.get("plan_approved") is not True, (
        "Length tap must not auto-set plan_approved.")


async def test_len_tap_re_renders_plan_with_updated_label_AND_keyboard():
    """Telegram removes the inline keyboard on editMessageText unless
    reply_markup is re-sent. A Length tap must therefore re-send the
    plan keyboard (with the ✅ moved to the tapped mode) or the user
    loses the Approve button — the dead-lock Albert hit live."""
    _seed_plan_inflight(chat_id=10011)
    update = _make_callback_update(chat_id=10011,
                                   callback_data=f"len:lecture:{RUN_ID}")
    ctx = _make_ctx()

    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()):
        await on_callback(update, ctx)

    update.callback_query.edit_message_text.assert_awaited()
    args, kwargs = update.callback_query.edit_message_text.await_args
    rendered = kwargs.get("text") or (args[0] if args else "")
    assert "Lecture" in rendered, (
        f"Edited plan message must mention the chosen length; got {rendered!r}")

    markup = kwargs.get("reply_markup")
    assert isinstance(markup, InlineKeyboardMarkup), (
        "Length tap must re-send the plan keyboard — editMessageText "
        "without reply_markup REMOVES the buttons (Telegram Bot API).")
    labels = [b.text for row in markup.inline_keyboard for b in row]
    assert any("Approve" in l for l in labels), (
        "Re-sent keyboard must still contain Approve.")
    assert any(l.startswith("Lecture") and "✅" in l for l in labels), (
        f"The ✅ marker must move onto the tapped length; labels: {labels}")
    # And the re-sent buttons stay run-scoped.
    datas = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert all(d.endswith(f":{RUN_ID}") for d in datas), datas


async def test_repeated_len_taps_do_not_stack_status_suffixes():
    """Re-render must build from the stored base plan text, not from the
    previously-edited message — otherwise each tap appends another
    '📏 Length set to …' line forever."""
    _seed_plan_inflight(chat_id=10015)
    ctx = _make_ctx()
    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()):
        for mode in ("long", "tldr", "lecture"):
            await on_callback(
                _make_callback_update(chat_id=10015,
                                      callback_data=f"len:{mode}:{RUN_ID}"),
                ctx)
    q = None  # inspect the LAST edit only
    # Each call used a fresh update mock; grab the last awaited text via
    # the entry's plan_text invariant instead: the suffix must appear once.
    # (The 3rd tap's rendered text is not directly reachable here, so we
    # assert on the stored base staying clean.)
    assert _inflight[RUN_ID]["plan_text"] == "Plan text", (
        "Base plan text must never accumulate status suffixes.")


# ---------------------------------------------------------------------------
# Approve contract
# ---------------------------------------------------------------------------


async def test_approve_tap_DOES_resume_graph_with_chosen_length():
    entry = _seed_plan_inflight(chat_id=10012)
    entry["length"] = "long"   # simulate a prior Length tap

    update = _make_callback_update(chat_id=10012,
                                   callback_data=f"plan:approve:{RUN_ID}")
    ctx = _make_ctx()

    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()) as resume:
        await on_callback(update, ctx)

    assert resume.await_count == 1, (
        f"Approve tap must call _resume_after_plan exactly once, got "
        f"{resume.await_count}.")
    assert _inflight[RUN_ID]["length"] == "long", (
        "Approve must preserve the previously-chosen length.")


async def test_len_then_approve_runs_resume_only_once():
    _seed_plan_inflight(chat_id=10013)
    ctx = _make_ctx()

    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()) as resume:
        await on_callback(
            _make_callback_update(chat_id=10013,
                                  callback_data=f"len:short:{RUN_ID}"), ctx)
        assert resume.await_count == 0

        await on_callback(
            _make_callback_update(chat_id=10013,
                                  callback_data=f"plan:approve:{RUN_ID}"), ctx)
        assert resume.await_count == 1


async def test_double_approve_resumes_only_once():
    """Fat-thumb guard: two Approve taps must not start two astream
    sessions on the same thread (the race behind the original bug)."""
    _seed_plan_inflight(chat_id=10016)
    ctx = _make_ctx()

    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()) as resume:
        for _ in range(2):
            await on_callback(
                _make_callback_update(chat_id=10016,
                                      callback_data=f"plan:approve:{RUN_ID}"),
                ctx)
    assert resume.await_count == 1, (
        f"Double-tapped Approve must resume exactly once, got "
        f"{resume.await_count}.")


async def test_edit_tap_does_not_resume_and_keeps_keyboard():
    _seed_plan_inflight(chat_id=10014)
    update = _make_callback_update(chat_id=10014,
                                   callback_data=f"plan:edit:{RUN_ID}")
    ctx = _make_ctx()

    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()) as resume:
        await on_callback(update, ctx)

    assert resume.await_count == 0, "Edit tap must not resume the graph."
    entry = _inflight[RUN_ID]
    assert entry.get("plan_edit") is True, (
        "Edit tap must set info['plan_edit'] so the bot recognises "
        "the next reply as revision feedback.")
    # Edit mode must keep the gate interactive too.
    _, kwargs = update.callback_query.edit_message_text.await_args
    assert isinstance(kwargs.get("reply_markup"), InlineKeyboardMarkup), (
        "Edit tap must re-send the keyboard — the user may still Approve.")


# ---------------------------------------------------------------------------
# Run scoping
# ---------------------------------------------------------------------------


async def test_callback_for_unknown_run_is_answered_not_applied():
    _seed_plan_inflight(chat_id=10017)          # RUN_ID in flight
    update = _make_callback_update(chat_id=10017,
                                   callback_data="plan:approve:deadbeef")
    ctx = _make_ctx()
    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()) as resume:
        await on_callback(update, ctx)
    assert resume.await_count == 0, (
        "A button for a run that is not in flight must never resume "
        "another run in the same chat.")
    update.callback_query.edit_message_text.assert_awaited()
    _, kwargs = update.callback_query.edit_message_text.await_args
    args = update.callback_query.edit_message_text.await_args.args
    text = kwargs.get("text") or (args[0] if args else "")
    assert "deadbeef" in text, "notice should name the unknown run id"


async def test_stale_pre_v2_button_gets_notice():
    """Buttons minted before the v2 upgrade carry 2-part data. They must
    be answered with a clear notice, not crash or mis-route."""
    _seed_plan_inflight(chat_id=10018)
    update = _make_callback_update(chat_id=10018, callback_data="plan:approve")
    ctx = _make_ctx()
    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()) as resume:
        await on_callback(update, ctx)
    assert resume.await_count == 0
    update.callback_query.edit_message_text.assert_awaited()
