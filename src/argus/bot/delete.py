"""/delete: the vault space browser and its del:* buttons."""
from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, ContextTypes

from ..config import Settings, get_settings
from . import common
from .common import _allowed, _bot_data_get, _fmt_bytes, _get_library

logger = logging.getLogger("argus.bot")


# ---------------------------------------------------------------------------
# /delete — vault space browser (v2 Phase 5)
# ---------------------------------------------------------------------------


_DELETE_PAGE_SIZE = 5
# Per-chat delete sessions: {category, page, selected:set[int], items:[...]}
# Items are snapshotted at listing time so toggle indices stay stable.
_delete_sessions: dict[int, dict[str, Any]] = {}

_DEL_CATEGORY_LABELS = {"runs": "🗂 Runs", "media": "🎬 Media",
                        "transcript": "📝 Transcripts"}


def _dir_size(path: str | Path) -> int:
    """Recursive directory size (sync — call via to_thread)."""
    total = 0
    try:
        for p in Path(path).rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    except Exception:
        logger.exception("dir size failed for %s", path)
    return total


def _safe_to_delete(path: Path, s: Settings) -> bool:
    """Only ever delete inside Argus-owned roots. A corrupt registry row
    must not be able to point the delete browser at arbitrary disk."""
    roots = (s.media_root, s.transcripts_root, s.history_root,
             s.reports_root)
    try:
        rp = path.resolve()
    except Exception:
        return False
    for root in roots:
        try:
            rr = Path(root).resolve()
        except Exception:
            continue
        if rp != rr and rr in rp.parents:
            return True
    return False


def _render_delete_page(sess: dict) -> tuple[str, InlineKeyboardMarkup]:
    items = sess["items"]
    page = sess["page"]
    selected: set[int] = sess["selected"]
    pages = max(1, (len(items) + _DELETE_PAGE_SIZE - 1) // _DELETE_PAGE_SIZE)
    page = max(0, min(page, pages - 1))
    sess["page"] = page
    start = page * _DELETE_PAGE_SIZE
    chunk = items[start:start + _DELETE_PAGE_SIZE]

    label = _DEL_CATEGORY_LABELS.get(sess["category"], sess["category"])
    lines = [f"🗑 Delete — {label} · page {page + 1}/{pages}",
             f"Selected: {len(selected)} · "
             f"{_fmt_bytes(sum(items[i]['bytes'] for i in selected))}", ""]
    toggle_row: list[InlineKeyboardButton] = []
    for offset, item in enumerate(chunk):
        i = start + offset
        mark = "☑" if i in selected else "☐"
        lines.append(f"{mark} {i + 1}. {item['label'][:58]} · "
                     f"{_fmt_bytes(item['bytes'])}")
        toggle_row.append(InlineKeyboardButton(
            f"{mark} {i + 1}", callback_data=f"del:tgl:{i}"))
    rows = [toggle_row] if toggle_row else []
    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("⬅ Prev", callback_data="del:pg:prev"))
    if page < pages - 1:
        nav.append(InlineKeyboardButton("Next ➡", callback_data="del:pg:next"))
    if nav:
        rows.append(nav)
    rows.append([
        InlineKeyboardButton("🗑 Delete selected", callback_data="del:ok:_"),
        InlineKeyboardButton("✖ Close", callback_data="del:cancel:_"),
    ])
    return "\n".join(lines), InlineKeyboardMarkup(rows)


async def _delete_load_items(ctx, chat_id: int, category: str) -> list[dict]:
    """Snapshot deletable items for one category. Each item:
    {label, bytes, kind, plus kind-specific fields}."""
    lib = _get_library(ctx)
    if lib is None:
        return []
    items: list[dict] = []
    if category == "runs":
        for r in await lib.list_runs(chat_id=chat_id, limit=50):
            size = 0
            if r.get("report_dir") and Path(r["report_dir"]).exists():
                size = await asyncio.to_thread(_dir_size, r["report_dir"])
            items.append({
                "kind": "run", "run_id": r["run_id"],
                "thread_id": r["thread_id"],
                "report_dir": r.get("report_dir"),
                "label": (f"{r['run_id']} · {r['status']} · "
                          f"{(r['topic'] or '')[:36]}"),
                "bytes": size,
            })
    else:
        for a in await lib.list_assets(kind=category, limit=50):
            items.append({
                "kind": "asset", "asset_id": a["asset_id"],
                "path": a["path"],
                "label": (f"[{a.get('platform') or '?'}] "
                          f"{(a.get('title') or Path(a['path']).name)[:40]}"),
                "bytes": a.get("bytes") or 0,
            })
    return items


async def delete_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/delete — browse and delete old runs / media / transcripts."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    if _get_library(ctx) is None:
        await update.message.reply_text("⚠️ Library unavailable.")
        return
    _delete_sessions.pop(update.effective_chat.id, None)
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton(lbl, callback_data=f"del:cat:{cat}")
        for cat, lbl in _DEL_CATEGORY_LABELS.items()
    ]])
    await update.message.reply_text(
        "🗑 What do you want to clean up?", reply_markup=kb)


async def _perform_delete(ctx, chat_id: int, sess: dict) -> str:
    """Delete every selected item. Returns a human summary."""
    s = get_settings()
    lib = _get_library(ctx)
    saver = _bot_data_get(ctx, "saver")
    freed = 0
    deleted = 0
    problems: list[str] = []
    for i in sorted(sess["selected"]):
        item = sess["items"][i]
        try:
            if item["kind"] == "run":
                if item["run_id"] in common._inflight:
                    problems.append(f"{item['run_id']}: in flight — skipped")
                    continue
                rd = item.get("report_dir")
                if rd and Path(rd).exists():
                    if _safe_to_delete(Path(rd), s):
                        await asyncio.to_thread(shutil.rmtree, rd,
                                                ignore_errors=True)
                    else:
                        problems.append(f"{item['run_id']}: report dir "
                                        f"outside Argus roots — files kept")
                if saver is not None:
                    try:
                        await saver.adelete_thread(item["thread_id"])
                    except Exception:
                        logger.exception("checkpoint delete failed")
                if lib is not None:
                    await lib.delete_run(item["run_id"])
            else:
                p = Path(item["path"])
                if p.exists():
                    if _safe_to_delete(p, s):
                        p.unlink()
                        sidecar = p.with_suffix(".info.json")
                        if sidecar.exists():
                            sidecar.unlink()
                    else:
                        problems.append(f"{p.name}: outside Argus roots — "
                                        "file kept")
                if lib is not None:
                    await lib.delete_assets([item["asset_id"]])
            freed += item["bytes"]
            deleted += 1
        except Exception as e:
            logger.exception("delete failed for %r", item.get("label"))
            problems.append(f"{item['label'][:40]}: {e}")
    summary = f"🗑 Deleted {deleted} item(s) · freed {_fmt_bytes(freed)}."
    if problems:
        summary += "\n⚠️ " + "\n⚠️ ".join(problems[:6])
    return summary


async def _on_delete_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                              action: str, arg: str) -> None:
    q = update.callback_query
    chat_id = q.message.chat.id if q.message else update.effective_chat.id
    sess = _delete_sessions.get(chat_id)

    if action == "cat":
        items = await _delete_load_items(ctx, chat_id, arg)
        if not items:
            await q.edit_message_text(
                f"Nothing recorded under {_DEL_CATEGORY_LABELS.get(arg, arg)}.")
            return
        sess = {"category": arg, "page": 0, "selected": set(), "items": items}
        _delete_sessions[chat_id] = sess
        text, kb = _render_delete_page(sess)
        await q.edit_message_text(text, reply_markup=kb)
        return

    if sess is None:
        await q.edit_message_text("(delete session expired — /delete again)")
        return

    if action == "tgl":
        try:
            i = int(arg)
        except ValueError:
            return
        if 0 <= i < len(sess["items"]):
            if i in sess["selected"]:
                sess["selected"].discard(i)
            else:
                sess["selected"].add(i)
        text, kb = _render_delete_page(sess)
        await q.edit_message_text(text, reply_markup=kb)
    elif action == "pg":
        sess["page"] += 1 if arg == "next" else -1
        text, kb = _render_delete_page(sess)
        await q.edit_message_text(text, reply_markup=kb)
    elif action == "ok":
        if not sess["selected"]:
            text, kb = _render_delete_page(sess)
            await q.edit_message_text(
                "(nothing selected)\n\n" + text, reply_markup=kb)
            return
        total = sum(sess["items"][i]["bytes"] for i in sess["selected"])
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Yes, delete", callback_data="del:yes:_"),
            InlineKeyboardButton("↩ Back", callback_data="del:no:_"),
        ]])
        await q.edit_message_text(
            f"⚠️ Delete {len(sess['selected'])} item(s), "
            f"{_fmt_bytes(total)} total?\nFiles are removed from disk and "
            "the registry — this cannot be undone.", reply_markup=kb)
    elif action == "yes":
        summary = await _perform_delete(ctx, chat_id, sess)
        _delete_sessions.pop(chat_id, None)
        await q.edit_message_text(summary)
    elif action == "no":
        text, kb = _render_delete_page(sess)
        await q.edit_message_text(text, reply_markup=kb)
    elif action == "cancel":
        _delete_sessions.pop(chat_id, None)
        await q.edit_message_text("🗑 Closed — nothing deleted.")


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                          ns: str, action: str, run_id: str) -> None:
    """``del:*`` buttons, routed here by ``app.on_callback``."""
    await _on_delete_callback(update, ctx, action, run_id)


def register(app: Application) -> None:
    app.add_handler(CommandHandler("delete", delete_cmd))
