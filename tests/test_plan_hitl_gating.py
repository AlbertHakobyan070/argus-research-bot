"""Tests for the plan-approval HITL gate.

Locks down the contract that:

1. Tapping a *Length* button during the plan-approval gate MUST only
   update the chosen length (and re-render the plan message so the user
   sees which length they picked). It MUST NOT resume the LangGraph.
   Resuming right after picking the length is a state-machine bug:
   the user is forced to commit to the plan before they've had a
   chance to read it, and any subsequent Approve-tap fires a second
   ``Command(resume=True)`` on an already-running graph (which then
   races with the first astream, causing the "approve button
   unpressable / research plan skipped" failure mode reported by
   Albert on 2026-07-10).

2. Tapping the *Approve* button (after picking a length, or with the
   default length) IS the only path that resumes the graph.

3. The plan message keyboard MUST survive a ``_safe_edit_cb`` edit
   (Telegram's editMessageText contract — ``reply_markup=None``
   preserves the existing inline keyboard). If the keyboard were
   silently dropped, the user would have no way to Approve and the
   pipeline would dead-lock at the plan gate.

These tests are unit tests — they do NOT require Telegram tokens or
the FreeLLMAPI proxy. They exercise ``on_callback`` and the in-flight
registry with async mocks. They are non-e2e, marked ``-m "not e2e"``
friendly so the day-to-day pytest run catches any regression.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from argus import bot as bot_mod
from argus.bot import _inflight, on_callback


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
    """Force ``_allowed(...)`` to return True for any user id.

    The production code reads the allowed user id from
    ``telegram_allowed_user_id`` in the bot's settings, and rejects
    any callback whose ``effective_user.id`` does not match. Without
    this patch, every callback in this test file would be rejected
    at the guard clause and the on_callback dispatcher would never
    even run — masking the bugs we want to expose.
    """
    monkeypatch.setattr(bot_mod, "_allowed", lambda _settings, _uid: True)


def _seed_plan_inflight(*, chat_id: int = 10001,
                        default_length: str = "medium") -> dict:
    """Populate ``_inflight[thread_id]`` with the shape that
    ``research_cmd`` builds after the planner delivers a plan but
    before any callback has fired. Returns the seeded entry so the
    test can introspect it."""
    # NB: thread_id must mirror ``research_cmd`` line ``thread_id = f"tg:{chat_id}"``.
    # We use simple integer chat_ids (no underscores) so the
    # ``tg:{n}`` string never has a separator confusion.
    thread_id = f"tg:{chat_id}"
    fake_graph = MagicMock()
    fake_cfg = {"configurable": {"thread_id": thread_id}}
    entry = {
        "state": {
            "thread_id": thread_id,
            "user_id": chat_id,
            "user_request": "topic",
            "length": default_length,
            "plan": {
                "summary": "Test plan summary",
                "sub_questions": ["q1", "q2", "q3"],
                "planned_sources": [
                    {"kind": "paper", "query": "x",
                     "target_url": None, "rationale": "p"},
                    {"kind": "blog", "query": "y",
                     "target_url": None, "rationale": "b"},
                    {"kind": "official_doc", "query": "z",
                     "target_url": None, "rationale": "o"},
                ],
            },
        },
        "stage": "planner",
        "length": default_length,
        "awaiting": "plan_approval",
        "cfg": fake_cfg,
        "graph": fake_graph,
        "saver_cm": None,
        "saver": None,
    }
    _inflight[thread_id] = entry
    return entry


def _make_callback_update(*, chat_id: int, callback_data: str,
                          message_text: str = "Plan text") -> MagicMock:
    """Synthesise an ``Update`` whose ``callback_query.data`` carries the
    given callback (e.g. ``"len:long"`` or ``"plan:approve"``)."""
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
    """Stand-in for ``telegram.ext.ContextTypes.DEFAULT_TYPE``.

    All bot methods used by on_callback / _resume_after_plan /
    _safe_edit_cb / _safe_send / _stream_progress MUST be AsyncMock so
    an ``await ctx.application.bot.X(...)`` actually returns. A bare
    MagicMock is not awaitable and would abort the SUT early,
    silently masking the bug we want to expose.
    """
    application = MagicMock()
    application.bot = MagicMock()
    application.bot.send_message = AsyncMock()
    application.bot.edit_message_text = AsyncMock()
    application.bot.send_chat_action = AsyncMock()
    application.bot.send_document = AsyncMock()
    application.bot.answer_callback_query = AsyncMock()
    ctx = MagicMock()
    ctx.application = application
    return ctx


# ---------------------------------------------------------------------------
# RED tests — these currently fail because tapping Length resumes the graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_len_tap_does_NOT_resume_graph():
    """Contract: tapping a Length button changes the chosen length and
    re-renders the plan message, but does NOT call ``_resume_after_plan``.
    Resuming belongs to the Approve button. Regression target:
    the bug reported on 2026-07-10 where research plan was effectively
    skipped because every Length tap also fired the resume."""
    _seed_plan_inflight(chat_id=999010)
    update = _make_callback_update(chat_id=999010, callback_data="len:long")
    ctx = _make_ctx()

    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()) as resume:
        await on_callback(update, ctx)

    # The Length choice MUST be recorded on the inflight entry.
    entry = _inflight["tg:999010"]
    assert entry["length"] == "long", (
        f"Length tap should update info['length'], got {entry['length']!r}"
    )
    # But it MUST NOT trigger the resume path.
    assert resume.await_count == 0, (
        f"Tapping a Length button must NOT call _resume_after_plan "
        f"({resume.await_count} call(s) observed). The Approve button is the "
        f"single resume gate."
    )
    # And the Approve flag is only set by Approve, not by Length.
    assert entry.get("plan_approved") is not True, (
        f"Length tap must not auto-set plan_approved (got "
        f"{entry.get('plan_approved')!r}). Auto-approval hides the review "
        f"window the user asked for."
    )


@pytest.mark.asyncio
async def test_len_tap_re_renders_plan_with_updated_label():
    """Contract: after picking a Length, the plan message is re-edited
    so the user can see the new choice (and so the Approve/Edit/Cancel
    buttons remain tappable — Telegram preserves an inline keyboard
    across edits when ``reply_markup=None`` is passed)."""
    _seed_plan_inflight(chat_id=10011)
    update = _make_callback_update(chat_id=10011, callback_data="len:lecture")
    ctx = _make_ctx()

    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()):
        await on_callback(update, ctx)

    # The plan message MUST be re-edited to acknowledge the choice
    # (otherwise the user has no visual confirmation their tap registered).
    update.callback_query.edit_message_text.assert_awaited()
    args, kwargs = update.callback_query.edit_message_text.await_args
    # The text is the first positional arg (the wrapper signature passes text positionally).
    rendered = kwargs.get("text") or (args[0] if args else "")
    assert "Lecture" in rendered, (
        f"After tapping a Length button, the edited plan message must "
        f"mention the chosen length so the user can confirm the tap registered. "
        f"Got: {rendered!r}"
    )


@pytest.mark.asyncio
async def test_approve_tap_DOES_resume_graph_with_chosen_length():
    """Contract: tapping Approve (with or without a preceding Length
    tap) is the ONE place that resumes the graph. The chosen length is
    whatever was last set on the inflight entry (``info['length']``)."""
    entry = _seed_plan_inflight(chat_id=10012)
    # Simulate the user having first tapped Length:long.
    entry["length"] = "long"

    update = _make_callback_update(chat_id=10012, callback_data="plan:approve")
    ctx = _make_ctx()

    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()) as resume:
        await on_callback(update, ctx)

    assert resume.await_count == 1, (
        f"Approve tap must call _resume_after_plan exactly once, got "
        f"{resume.await_count}."
    )
    # The inflight entry must still carry the chosen length so the
    # resume payload pushes the right tier into the graph state.
    entry = _inflight["tg:10012"]
    assert entry["length"] == "long", (
        f"Approve must preserve the previously-chosen length, got "
        f"{entry['length']!r}."
    )


@pytest.mark.asyncio
async def test_len_then_approve_runs_resume_only_once():
    """Contract: a Length tap followed by an Approve tap must resume
    the graph exactly ONCE — not twice. The previous (buggy) behavior
    resumed once on Length and once again on Approve, racing the
    astream and making the Approve button appear 'unpressable' to
    the user."""
    _seed_plan_inflight(chat_id=10013)
    ctx = _make_ctx()

    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()) as resume:
        # Tap Length first.
        await on_callback(
            _make_callback_update(chat_id=10013, callback_data="len:short"),
            ctx,
        )
        assert resume.await_count == 0, (
            "Length tap must not resume; Approve must be the only resume."
        )

        # Then tap Approve.
        await on_callback(
            _make_callback_update(chat_id=10013, callback_data="plan:approve"),
            ctx,
        )
        assert resume.await_count == 1, (
            f"Approve after Length must resume exactly once, got "
            f"{resume.await_count}. Double-resume is the bug that "
            f"makes the user feel Approve is 'broken'."
        )


@pytest.mark.asyncio
async def test_edit_tap_does_not_resume_either():
    """Contract: Edit does not resume either; it just flips the
    inflight entry into a 'plan_edit' state so the next free-form
    message is treated as revision feedback."""
    _seed_plan_inflight(chat_id=10014)
    update = _make_callback_update(chat_id=10014, callback_data="plan:edit")
    ctx = _make_ctx()

    with patch.object(bot_mod, "_resume_after_plan", new=AsyncMock()) as resume:
        await on_callback(update, ctx)

    assert resume.await_count == 0, (
        "Edit tap must not resume the graph."
    )
    entry = _inflight["tg:10014"]
    assert entry.get("plan_edit") is True, (
        "Edit tap must set info['plan_edit'] so the bot recognises "
        "the next reply as revision feedback."
    )


# ---------------------------------------------------------------------------
# Regression guard: Telegram edit_message_text contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_safe_edit_cb_preserves_keyboard_when_no_markup_passed():
    """Contract: ``_safe_edit_cb`` forwards ``reply_markup=None`` to
    ``edit_message_text`` and Telegram preserves the existing inline
    keyboard when ``reply_markup`` is omitted (Telegram Bot API spec:
    "omitted — leave the existing one"). Verifies our wrapper doesn't
    accidentally pass ``reply_markup=InlineKeyboardMarkup([])`` or any
    other value that would cause Telegram to clear the buttons.

    If this test fails after a future change, the plan gate will start
    to 'lose' its Approve button after every edit — which is exactly
    the symptom the user is reporting.
    """
    q = MagicMock()
    q.edit_message_text = AsyncMock()

    # First scenario: no reply_markup passed (the Length-tap path
    # passes none because it just wants to re-render the text).
    from argus.bot import _safe_edit_cb
    await _safe_edit_cb(q, "new text")
    a, k = q.edit_message_text.await_args
    # reply_markup should be missing/None at the Telegram layer.
    # The wrapper signature accepts ``reply_markup=None`` as default.
    passed_markup = q.edit_message_text.await_args.kwargs.get(
        "reply_markup", None,
    )
    # The default value of the wrapper is None; we re-check the call.
    assert passed_markup is None, (
        f"_safe_edit_cb(..., reply_markup=None) must forward "
        f"reply_markup=None to edit_message_text so Telegram leaves "
        f"the existing inline keyboard in place. Got {passed_markup!r}."
    )
