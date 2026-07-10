"""/delete browser tests — selection state machine, safety guards, and
actual deletion against tmp dirs + an in-memory registry."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import InlineKeyboardMarkup

from argus import bot as bot_mod
from argus.bot import (
    _delete_sessions, _fmt_bytes, _inflight, _perform_delete,
    _render_delete_page, _safe_to_delete, on_callback,
)
from argus.config import Settings
from argus.library import Library


@pytest.fixture(autouse=True)
def _clean():
    _delete_sessions.clear()
    _inflight.clear()
    yield
    _delete_sessions.clear()
    _inflight.clear()


@pytest.fixture(autouse=True)
def _bypass_acl(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bot_mod, "_allowed", lambda _s, _uid: True)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        freellmapi_base_url="http://127.0.0.1:3001/v1",
        freellmapi_api_key="freellmapi-test",
        telegram_bot_token="123:test", telegram_allowed_user_id=1,
        reports_root=tmp_path / "vault" / "history",
        checkpoint_db=tmp_path / "cp.sqlite",
        library_db=tmp_path / "lib.sqlite",
        vault_root=tmp_path / "vault",
        media_root=tmp_path / "vault" / "media",
        transcripts_root=tmp_path / "vault" / "transcripts",
        history_root=tmp_path / "vault" / "history",
        ffmpeg_path=None,
        request_timeout_seconds=60.0, max_revision_rounds=3,
    )


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    monkeypatch.setattr(bot_mod, "get_settings", lambda: s)
    return s


# ---------------------------------------------------------------------------
# rendering / state machine
# ---------------------------------------------------------------------------


def _sess(n_items=7, selected=None):
    return {"category": "media", "page": 0,
            "selected": set(selected or ()),
            "items": [{"kind": "asset", "asset_id": i, "path": f"p{i}",
                       "label": f"item {i}", "bytes": 1000 * (i + 1)}
                      for i in range(n_items)]}


def test_render_paginates_five_per_page():
    sess = _sess(7)
    text, kb = _render_delete_page(sess)
    assert "page 1/2" in text
    datas = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert "del:tgl:0" in datas and "del:tgl:4" in datas
    assert "del:tgl:5" not in datas, "page 1 shows only the first five"
    assert "del:pg:next" in datas and "del:pg:prev" not in datas

    sess["page"] = 1
    text, kb = _render_delete_page(sess)
    assert "page 2/2" in text
    datas = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert "del:tgl:5" in datas and "del:pg:prev" in datas


def test_render_marks_selection_and_total():
    sess = _sess(3, selected={0, 2})
    text, kb = _render_delete_page(sess)
    assert "Selected: 2" in text
    assert _fmt_bytes(1000 + 3000) in text
    labels = [b.text for row in kb.inline_keyboard for b in row]
    assert any(l.startswith("☑ 1") for l in labels)
    assert any(l.startswith("☐ 2") for l in labels)


def test_fmt_bytes_scales():
    assert _fmt_bytes(500) == "500 B"
    assert _fmt_bytes(2048) == "2 KB"
    assert "MB" in _fmt_bytes(5 * 1024 * 1024)
    assert "GB" in _fmt_bytes(3 * 1024 ** 3)


# ---------------------------------------------------------------------------
# safety guard
# ---------------------------------------------------------------------------


def test_safe_to_delete_only_inside_argus_roots(tmp_path, settings):
    inside = settings.media_root / "youtube" / "v.mp4"
    outside = tmp_path / "elsewhere" / "v.mp4"
    root_itself = settings.media_root
    assert _safe_to_delete(inside, settings) is True
    assert _safe_to_delete(outside, settings) is False
    assert _safe_to_delete(root_itself, settings) is False, (
        "the root folder itself must never be deletable")
    assert _safe_to_delete(Path(r"C:\Windows\system32"), settings) is False


# ---------------------------------------------------------------------------
# performing deletion
# ---------------------------------------------------------------------------


def _ctx(lib=None, saver=None):
    application = MagicMock()
    application.bot = MagicMock()
    application.bot.send_message = AsyncMock()
    application.bot_data = {}
    if lib is not None:
        application.bot_data["library"] = lib
    if saver is not None:
        application.bot_data["saver"] = saver
    ctx = MagicMock()
    ctx.application = application
    return ctx


async def test_perform_delete_removes_assets_and_registry_rows(tmp_path,
                                                               settings):
    lib = Library(tmp_path / "lib.sqlite")
    await lib.open()
    f1 = settings.media_root / "youtube" / "a.mp4"
    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_bytes(b"x" * 10)
    f1.with_suffix(".info.json").write_text("{}", encoding="utf-8")
    a1 = await lib.add_asset(kind="media", path=str(f1), size_bytes=10)

    sess = {"category": "media", "page": 0, "selected": {0},
            "items": [{"kind": "asset", "asset_id": a1, "path": str(f1),
                       "label": "a", "bytes": 10}]}
    summary = await _perform_delete(_ctx(lib=lib), 1, sess)

    assert "Deleted 1" in summary
    assert not f1.exists()
    assert not f1.with_suffix(".info.json").exists(), "sidecar removed too"
    assert await lib.get_assets([a1]) == []
    await lib.close()


async def test_perform_delete_run_removes_dir_checkpoint_and_row(tmp_path,
                                                                 settings):
    lib = Library(tmp_path / "lib.sqlite")
    await lib.open()
    rd = settings.history_root / "20260101_topic_short"
    rd.mkdir(parents=True)
    (rd / "report.md").write_text("r", encoding="utf-8")
    await lib.create_run(run_id="dead1234", thread_id="tg:1:dead1234",
                         chat_id=1, topic="t", status="done")
    saver = MagicMock()
    saver.adelete_thread = AsyncMock()

    sess = {"category": "runs", "page": 0, "selected": {0},
            "items": [{"kind": "run", "run_id": "dead1234",
                       "thread_id": "tg:1:dead1234", "report_dir": str(rd),
                       "label": "run", "bytes": 1}]}
    summary = await _perform_delete(_ctx(lib=lib, saver=saver), 1, sess)

    assert "Deleted 1" in summary
    assert not rd.exists()
    saver.adelete_thread.assert_awaited_once_with("tg:1:dead1234")
    assert await lib.get_run("dead1234") is None
    await lib.close()


async def test_perform_delete_refuses_inflight_run(tmp_path, settings):
    _inflight["live1234"] = {"run_id": "live1234", "chat_id": 1}
    sess = {"category": "runs", "page": 0, "selected": {0},
            "items": [{"kind": "run", "run_id": "live1234",
                       "thread_id": "tg:1:live1234", "report_dir": None,
                       "label": "live run", "bytes": 0}]}
    summary = await _perform_delete(_ctx(), 1, sess)
    assert "Deleted 0" in summary
    assert "in flight" in summary


async def test_perform_delete_keeps_files_outside_roots(tmp_path, settings):
    stray = tmp_path / "elsewhere" / "keep.mp4"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"x")
    lib = Library(tmp_path / "lib.sqlite")
    await lib.open()
    aid = await lib.add_asset(kind="media", path=str(stray), size_bytes=1)
    sess = {"category": "media", "page": 0, "selected": {0},
            "items": [{"kind": "asset", "asset_id": aid, "path": str(stray),
                       "label": "stray", "bytes": 1}]}
    summary = await _perform_delete(_ctx(lib=lib), 1, sess)
    assert stray.exists(), "files outside Argus roots must never be unlinked"
    assert "outside Argus roots" in summary
    # The registry row IS removed (it pointed outside; deregistering is safe).
    assert await lib.get_assets([aid]) == []
    await lib.close()


# ---------------------------------------------------------------------------
# callback flow
# ---------------------------------------------------------------------------


def _cb_update(chat_id, data):
    q = MagicMock()
    q.data = data
    q.answer = AsyncMock()
    q.edit_message_text = AsyncMock()
    q.message = MagicMock()
    q.message.chat.id = chat_id
    update = MagicMock()
    update.effective_user.id = chat_id
    update.effective_chat.id = chat_id
    update.callback_query = q
    return update


async def test_callback_flow_toggle_then_confirm(settings):
    _delete_sessions[9] = _sess(3)
    ctx = _ctx()

    await on_callback(_cb_update(9, "del:tgl:1"), ctx)
    assert _delete_sessions[9]["selected"] == {1}

    await on_callback(_cb_update(9, "del:tgl:1"), ctx)
    assert _delete_sessions[9]["selected"] == set()

    await on_callback(_cb_update(9, "del:tgl:2"), ctx)
    upd = _cb_update(9, "del:ok:_")
    await on_callback(upd, ctx)
    args, kwargs = upd.callback_query.edit_message_text.await_args
    text = kwargs.get("text") or (args[0] if args else "")
    assert "cannot be undone" in text
    kb = kwargs.get("reply_markup")
    datas = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert "del:yes:_" in datas and "del:no:_" in datas


async def test_callback_expired_session_notice(settings):
    ctx = _ctx()
    upd = _cb_update(9, "del:tgl:0")
    await on_callback(upd, ctx)
    args, kwargs = upd.callback_query.edit_message_text.await_args
    text = kwargs.get("text") or (args[0] if args else "")
    assert "expired" in text
