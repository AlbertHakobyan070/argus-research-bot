"""Bot-side media command tests — /fetch, /find, /quality, URL paste,
and the m:* pool callbacks. Hermetic (mocks; no network, no Telegram)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import InlineKeyboardMarkup

from argus import bot as bot_mod
from argus.bot import (
    _media_keyboard, _pool_get, _pool_put, _video_pool,
    fetch_cmd, find_cmd, on_callback, on_text, quality_cmd,
)
from argus.config import Settings


@pytest.fixture(autouse=True)
def _clean_pools():
    _video_pool.clear()
    yield
    _video_pool.clear()


@pytest.fixture(autouse=True)
def _bypass_acl(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bot_mod, "_allowed", lambda _s, _uid: True)


@pytest.fixture(autouse=True)
def _dummy_settings(monkeypatch: pytest.MonkeyPatch):
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


def _make_update(chat_id: int = 42, text: str = "") -> MagicMock:
    u = MagicMock()
    u.effective_user.id = chat_id
    u.effective_chat.id = chat_id
    u.message.text = text
    u.message.reply_text = AsyncMock()
    return u


def _make_ctx(args=None, library=None) -> MagicMock:
    application = MagicMock()
    application.bot = MagicMock()
    application.bot.send_message = AsyncMock(
        return_value=SimpleNamespace(message_id=7))
    application.bot.edit_message_text = AsyncMock()
    application.bot.send_chat_action = AsyncMock()
    application.bot.send_document = AsyncMock()
    application.bot.send_video = AsyncMock()
    application.bot_data = {}
    if library is not None:
        application.bot_data["library"] = library

    created: list = []

    def _capture(coro):
        created.append(coro)
        coro.close()   # never actually run it — hermetic
        return MagicMock()

    application.create_task = _capture
    application.created_tasks = created
    ctx = MagicMock()
    ctx.application = application
    ctx.args = args or []
    return ctx


# ---------------------------------------------------------------------------
# /fetch
# ---------------------------------------------------------------------------


async def test_fetch_queues_one_task_per_supported_url():
    ctx = _make_ctx(args=["https://youtu.be/abc123DEF45",
                          "https://x.com/u/status/9",
                          "https://example.com/nope"])
    update = _make_update()
    await fetch_cmd(update, ctx)

    assert len(ctx.application.created_tasks) == 2, (
        "one background download per supported URL")
    # The unsupported link is called out, and a queue notice goes out.
    texts = " ".join(str(c) for c in update.message.reply_text.await_args_list)
    assert "example.com/nope" in texts
    assert "Queued 2" in texts


async def test_fetch_without_args_prints_usage():
    ctx = _make_ctx(args=[])
    update = _make_update()
    await fetch_cmd(update, ctx)
    assert not ctx.application.created_tasks
    texts = str(update.message.reply_text.await_args_list)
    assert "Usage" in texts


# ---------------------------------------------------------------------------
# /quality
# ---------------------------------------------------------------------------


class _FakeLib:
    def __init__(self):
        self.kv: dict[str, str] = {}

    async def get_setting(self, key, default=None):
        return self.kv.get(key, default)

    async def set_setting(self, key, value):
        self.kv[key] = value


async def test_quality_shows_current_and_sets_new():
    lib = _FakeLib()
    update = _make_update()

    await quality_cmd(update, _make_ctx(args=[], library=lib))
    shown = str(update.message.reply_text.await_args_list[-1])
    assert "auto" in shown and "ffmpeg" in shown

    await quality_cmd(update, _make_ctx(args=["720"], library=lib))
    assert lib.kv["media_quality"] == "720"

    await quality_cmd(update, _make_ctx(args=["potato"], library=lib))
    assert lib.kv["media_quality"] == "720", "invalid value must not persist"
    err = str(update.message.reply_text.await_args_list[-1])
    assert "auto|min|max" in err


# ---------------------------------------------------------------------------
# /find (reddit path, mocked search)
# ---------------------------------------------------------------------------


async def test_find_reddit_pools_results_with_media_keyboard(monkeypatch):
    results = [
        {"title": "Post A", "url": "https://www.reddit.com/r/v/comments/a/x/",
         "channel": "r/v", "duration": 30, "platform": "reddit"},
        {"title": "Post B", "url": "https://www.reddit.com/r/v/comments/b/y/",
         "channel": "r/v", "duration": 12, "platform": "reddit"},
    ]

    async def fake_search(query, *, limit=8, **kw):
        return list(results)

    monkeypatch.setattr(bot_mod, "reddit_search", fake_search)
    update = _make_update(chat_id=77)
    ctx = _make_ctx(args=["reddit", "funny", "cats"])
    await find_cmd(update, ctx)

    pool = _pool_get("tg:77")
    assert pool and pool[0]["platform"] == "reddit"
    sent = ctx.application.bot.send_message.await_args_list[-1]
    markup = sent.kwargs.get("reply_markup")
    assert isinstance(markup, InlineKeyboardMarkup)
    datas = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert "m:dl:1" in datas and "m:tr:2" in datas and "m:both:1" in datas


# ---------------------------------------------------------------------------
# URL paste → action keyboard
# ---------------------------------------------------------------------------


async def test_on_text_with_media_url_offers_actions():
    update = _make_update(chat_id=88,
                          text="look at this https://youtu.be/abc123DEF45 !")
    ctx = _make_ctx()
    await on_text(update, ctx)

    pool = _pool_get("tg:88")
    assert pool and pool[0]["platform"] == "youtube"
    call = update.message.reply_text.await_args
    assert isinstance(call.kwargs.get("reply_markup"), InlineKeyboardMarkup)


async def test_on_text_plain_chat_is_ignored():
    update = _make_update(chat_id=88, text="just chatting, no links")
    ctx = _make_ctx()
    await on_text(update, ctx)
    assert _pool_get("tg:88") is None
    update.message.reply_text.assert_not_awaited()


# ---------------------------------------------------------------------------
# m:* callbacks
# ---------------------------------------------------------------------------


def _make_cb_update(chat_id: int, data: str) -> MagicMock:
    q = MagicMock()
    q.data = data
    q.answer = AsyncMock()
    q.edit_message_text = AsyncMock()
    q.message = MagicMock()
    q.message.chat.id = chat_id
    q.message.text = "listing"
    update = MagicMock()
    update.effective_user.id = chat_id
    update.effective_chat.id = chat_id
    update.callback_query = q
    return update


async def test_media_download_button_spawns_task():
    _pool_put("tg:99", [{"title": "T", "url": "https://youtu.be/abc123DEF45",
                         "platform": "youtube"}])
    ctx = _make_ctx()
    await on_callback(_make_cb_update(99, "m:dl:1"), ctx)
    assert len(ctx.application.created_tasks) == 1
    note = str(ctx.application.bot.send_message.await_args_list[-1])
    assert "Queued download" in note


async def test_media_both_button_spawns_two_tasks():
    _pool_put("tg:99", [{"title": "T", "url": "https://youtu.be/abc123DEF45",
                         "platform": "youtube"}])
    ctx = _make_ctx()
    await on_callback(_make_cb_update(99, "m:both:1"), ctx)
    assert len(ctx.application.created_tasks) == 2


async def test_media_button_on_expired_pool_says_so():
    ctx = _make_ctx()
    await on_callback(_make_cb_update(99, "m:dl:1"), ctx)
    assert not ctx.application.created_tasks
    note = str(ctx.application.bot.send_message.await_args_list[-1])
    assert "expired" in note
