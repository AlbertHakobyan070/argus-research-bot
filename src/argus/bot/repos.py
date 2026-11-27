"""Investigate in the chat (spec 2026-11-27): the 🔎 item button's result —
what a saved video mentions, checked — with a tick per entry (``rp:*``),
and ``/repos``, the bookmarks (``bk:*``).

Callback data stays three ``:`` parts; the third packs ``.``-separated
numbers: ``rp:t:<asset>.<rev>.<i>`` ticks entry i, ``rp:ok:<asset>.<rev>``
attaches the ticked ones, ``rp:bm:`` also bookmarks them;
``bk:pg:<page>``, ``bk:o|gh|rm|rmy:<bookmark>.<page>``. ``rev`` is the
investigation's number: a tap on an older list says so instead of acting.
Everything a tap needs is in the library, so the buttons work after a
restart.
"""
from __future__ import annotations

import html
import logging
import time
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions, Update, WebAppInfo
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes

from .. import media_service as ms
from ..config import get_settings
from .common import _allowed, _get_library, _tg_len
from .progress import edit_text

logger = logging.getLogger("argus.bot")

PAGE = 10
_NO_PREVIEW = LinkPreviewOptions(is_disabled=True)
_LIMIT = 3900                       # Telegram's 4096, with room to spare
_WHERE = {"title": "in the title", "description": "in the description", "comments": "in the uploader's comment",
          "tags": "in the tags", "chapters": "in the chapters", "transcript": "said"}


def _console(app) -> str | None:
    return (app.bot_data.get("webapp_public_url") if isinstance(app.bot_data, dict) else None) or None


def _e(s) -> str:
    return html.escape(str(s or ""), quote=False)


def _link(url: str, text: str) -> str:
    return f'<a href="{html.escape(url, quote=True)}">{_e(text)}</a>'


def stars(n) -> str:
    """45485 → 45.5k; None → ""."""
    if n is None:
        return ""
    return f"{n / 1000:.1f}k".replace(".0k", "k") if n >= 1000 else str(n)


def clock(s) -> str:
    s = int(s)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _entry(i: int, f: dict, *, ticks: bool, brief: bool) -> list[str]:
    facts = [x for x in (f"★{stars(f['stars'])}" if f.get("stars") is not None else "",
                         "archived" if f.get("archived") else "", "fork" if f.get("fork") else "",
                         f"was {f['renamed_from']}" if f.get("renamed_from") else "",
                         "" if f.get("sure") else "looked up") if x]
    mark = ("☑ " if f.get("on") else "☐ ") if ticks else ""
    lines = [f"{i + 1}. {mark}{_link(f['url'], f['name'])}" + (f" · {_e(' · '.join(facts))}" if facts else "")]
    if f.get("about") and not brief:
        lines.append(f"    {_e(f['about'][:110])}")
    at = f" ({clock(f['at_s'])})" if f.get("at_s") is not None else ""
    quote = _e((f.get("quote") or "")[:100 if brief else 140])
    lines.append(f"    <i>{_WHERE.get(f.get('from'), 'in the material')}: “{quote}”{at}</i>")
    return lines


def result_view(a: dict, console: str | None) -> tuple[str, InlineKeyboardMarkup | None]:
    """The investigation of item ``a`` as (HTML text, buttons): numbered
    entries with ☑/☐ while it waits, what was skipped and why."""
    inv = a["meta"].get("investigate") or {}
    found, rev, aid = inv.get("found") or [], int(inv.get("rev") or 0), a["asset_id"]
    pending = inv.get("state") == "pending" and bool(found)
    title = _e((a.get("title") or Path(a["path"]).name)[:70])
    skipped, read = inv.get("skipped") or {}, inv.get("read") or {}
    tail = []
    if skipped.get("not_found"):
        tail.append("Not public on GitHub (gone or private): " + _e(", ".join(skipped["not_found"][:6])))
    if skipped.get("unresolved"):
        tail.append("No match found for: " + _e(", ".join(skipped["unresolved"][:6])))
    if skipped.get("unchecked"):
        tail.append(f"Not checked ({len(skipped['unchecked'])}): {_e(skipped.get('stopped') or 'the per-run limit')}")
    if skipped.get("errors"):
        tail.append("Lookups that failed: " + _e("; ".join(skipped["errors"][:3])[:300]))
    if skipped.get("more"):
        tail.append(f"{skipped['more']} more were left out (the list keeps the first {len(found)}).")
    if inv.get("note"):
        tail.append(_e(inv["note"]))
    if read.get("cut"):
        tail.append("The transcript is long: its first part was read.")
    elif not read.get("transcript") and not inv.get("note"):
        tail.append("No transcript was read.")
    attached = a["meta"].get("repos") or []
    if attached and not pending:
        tail.append("📎 Attached: " + _e(", ".join(r["name"] for r in attached[:8])))
    if pending:
        tail.append("Tick what fits, then 📎 Attach.")
    repos = [(i, f) for i, f in enumerate(found) if f["kind"] == "repo"]
    tools = [(i, f) for i, f in enumerate(found) if f["kind"] != "repo"]

    def build(brief: bool, upto: int) -> str:
        lines = [f"🔎 <b>{'Found in' if found else 'Nothing checked out in'}</b> {title}"]
        shown = 0
        for label, group in (("Repos", repos), ("Tools &amp; sites", tools)):
            if group and repos and tools:
                lines.append(f"\n<b>{label}</b>")
            for i, f in group:
                if shown < upto:
                    lines += _entry(i, f, ticks=pending, brief=brief)
                shown += 1
        if shown > upto:
            lines.append(f"… and {shown - upto} more: open it in the console to see them.")
        return "\n".join(lines + ([""] + tail if tail else []))

    text = build(False, len(found))
    for brief, upto in ((True, len(found)), (True, 8), (True, 4)):
        if _tg_len(text) <= _LIMIT:
            break
        text = build(brief, upto)
    rows: list[list[InlineKeyboardButton]] = []
    if pending:
        ticks = [InlineKeyboardButton(f"{'☑' if f.get('on') else '☐'} {i + 1} {f['name'].split('/')[-1][:16]}",
                                      callback_data=f"rp:t:{aid}.{rev}.{i}") for i, f in enumerate(found)]
        rows += [ticks[j:j + 2] for j in range(0, len(ticks), 2)]
        rows.append([InlineKeyboardButton("📎 Attach", callback_data=f"rp:ok:{aid}.{rev}", style="success"),
                     InlineKeyboardButton("⭐ Attach + bookmark", callback_data=f"rp:bm:{aid}.{rev}")])
    if console:
        rows.append([InlineKeyboardButton("✏️ Review" if pending else "📱 Open",
                                          web_app=WebAppInfo(ms.item_url(console, aid)))])
    return text, InlineKeyboardMarkup(rows) if rows else None


async def _show(app, chat_id: int, message_id: int | None, text: str, markup) -> None:
    """Edit the message into ``text`` (HTML), or send it when the message is gone."""
    if message_id is not None and await edit_text(app.bot, chat_id=chat_id, message_id=message_id, text=text,
                                                  reply_markup=markup, parse_mode=ParseMode.HTML,
                                                  link_preview_options=_NO_PREVIEW):
        return
    await app.bot.send_message(chat_id=chat_id, text=text, reply_markup=markup, parse_mode=ParseMode.HTML,
                               link_preview_options=_NO_PREVIEW)


async def start(app, chat_id: int, a: dict) -> None:
    """🔎 on an item: one message that follows the job, then shows its result."""
    lib = app.bot_data["library"]
    title = (a.get("title") or Path(a["path"]).name)[:80]
    head = f"🔎 Investigating {title}…"
    msg = await app.bot.send_message(chat_id=chat_id, text=head, link_preview_options=_NO_PREVIEW)
    last = {"t": 0.0}

    async def stage(text: str) -> None:
        if time.monotonic() - last["t"] < 3.0:             # one edit per 3 s per message
            return
        last["t"] = time.monotonic()
        try:
            await edit_text(app.bot, chat_id=chat_id, message_id=msg.message_id, text=f"{head}\n{text}",
                            link_preview_options=_NO_PREVIEW)
        except TelegramError as e:                         # only the progress line is lost; the job goes on
            logger.warning("investigate progress for asset %s not shown: %s", a["asset_id"], e)

    async def then(job: ms.Job) -> None:
        if job.state != "done":
            await _show(app, chat_id, msg.message_id, f"🔎 Investigate failed for {_e(title)}: {_e(job.error)}", None)
            return
        text, markup = result_view(await lib.get_asset(a["asset_id"]), _console(app))
        await _show(app, chat_id, msg.message_id, text, markup)

    try:
        ms.investigate_job(lib, get_settings(), a["asset_id"], title, then=then, stage=stage)
    except ms.MediaError as e:
        await edit_text(app.bot, chat_id=chat_id, message_id=msg.message_id, text=f"⚠️ {e}")


# ---------------------------------------------------------------------------
# /repos — the bookmarks
# ---------------------------------------------------------------------------

def _open_button(console: str) -> InlineKeyboardButton:
    return InlineKeyboardButton("📱 Open", web_app=WebAppInfo(f"{console.rstrip('/')}/#/media/repos"))


def _bm_line(n: int, b: dict) -> str:
    facts = " · ".join(x for x in (f"★{stars(b['stars'])}" if b.get("stars") is not None else "",
                                   "tool" if b["kind"] == "tool" else "") if x)
    note = f" — {_e(b['note'][:90])}" if b.get("note") else ""
    return f"{n}. {_link(b['url'], b['name'])}" + (f" · {facts}" if facts else "") + note


async def list_view(lib, page: int, console: str | None) -> tuple[str, InlineKeyboardMarkup | None]:
    """A page of bookmarks, newest first, with a number button for each."""
    rows, total = await lib.list_bookmarks(limit=PAGE, offset=page * PAGE)
    if total == 0:
        return ("⭐ No bookmarks yet. Tap 🔎 Investigate on a saved video, then ⭐ Attach + bookmark.",
                InlineKeyboardMarkup([[_open_button(console)]]) if console else None)
    if not rows and page > 0:
        return await list_view(lib, 0, console)
    pages = (total + PAGE - 1) // PAGE
    text = "\n".join([f"⭐ <b>Bookmarks</b> · {total} saved" + (f" · page {page + 1} of {pages}" if pages > 1 else ""),
                      *(_bm_line(page * PAGE + i + 1, b) for i, b in enumerate(rows)),
                      "", "Tap a number to open one."])
    nums = [InlineKeyboardButton(str(page * PAGE + i + 1), callback_data=f"bk:o:{b['bookmark_id']}.{page}")
            for i, b in enumerate(rows)]
    kb = [nums[j:j + 5] for j in range(0, len(nums), 5)]
    nav = ([InlineKeyboardButton("◀ Newer", callback_data=f"bk:pg:{page - 1}")] if page > 0 else []) + \
          ([InlineKeyboardButton("Older ▶", callback_data=f"bk:pg:{page + 1}")] if page + 1 < pages else [])
    if nav:
        kb.append(nav)
    if console:
        kb.append([_open_button(console)])
    return text, InlineKeyboardMarkup(kb)


async def card_view(lib, b: dict, page: int, console: str | None, *, confirm: bool = False
                    ) -> tuple[str, InlineKeyboardMarkup]:
    """One bookmark: its note, the project's own words, the videos that carry it."""
    items = await lib.bookmark_items(b["kind"], b["key"])
    facts = " · ".join(x for x in (f"★{stars(b['stars'])}" if b.get("stars") is not None else "",
                                   "repo" if b["kind"] == "repo" else "tool or site") if x)
    source = "GitHub" if b["kind"] == "repo" else "the site"
    lines = [f"⭐ <b>{_link(b['url'], b['name'])}</b> · {_e(facts)}",
             f"<b>Note:</b> {_e(b['note']) if b.get('note') else '— (write one in the console)'}",
             f"<b>From {source}:</b> {_e(b['about']) if b.get('about') else '— tap 📥 to fill it'}"]
    if items:
        lines.append("<b>In:</b> " + "; ".join(f"{_e((i.get('title') or '')[:50])} (asset:{i['asset_id']})"
                                                for i in items[:5])
                     + (f"; and {len(items) - 5} more" if len(items) > 5 else ""))
    bid = b["bookmark_id"]
    if confirm:
        lines.append("\nRemove this bookmark? The videos keep it attached.")
        kb = [[InlineKeyboardButton("🗑 Yes, remove", callback_data=f"bk:rmy:{bid}.{page}", style="danger"),
               InlineKeyboardButton("Keep", callback_data=f"bk:o:{bid}.{page}")]]
    else:
        kb = [[InlineKeyboardButton(f"📥 From {source}", callback_data=f"bk:gh:{bid}.{page}"),
               InlineKeyboardButton("🗑 Remove", callback_data=f"bk:rm:{bid}.{page}")],
              [InlineKeyboardButton("↩ All bookmarks", callback_data=f"bk:pg:{page}")]]
        if console:
            kb[1].append(InlineKeyboardButton("✏️ Note", web_app=WebAppInfo(f"{console.rstrip('/')}/#/media/repos")))
    return "\n".join(lines), InlineKeyboardMarkup(kb)


async def repos_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/repos — the saved repos, tools and sites, with a card for each."""
    if not _allowed(get_settings(), update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    lib = _get_library(ctx)
    if lib is None:
        await update.message.reply_text("⚠️ Library unavailable.")
        return
    text, kb = await list_view(lib, 0, _console(ctx.application))
    await update.message.reply_text(text, reply_markup=kb, parse_mode=ParseMode.HTML, link_preview_options=_NO_PREVIEW)


def _nums(arg: str, n: int) -> list[int] | None:
    parts = arg.split(".")
    return [int(p) for p in parts] if len(parts) == n and all(p.isdigit() for p in parts) else None


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE, ns: str, action: str, arg: str) -> None:
    """``rp:*`` (an investigation's ticks and Attach) and ``bk:*`` (bookmarks)."""
    q = update.callback_query
    app = ctx.application
    chat_id = q.message.chat.id if q.message else update.effective_chat.id
    message_id = q.message.message_id if q.message else None
    lib = _get_library(ctx)
    if lib is None:
        await app.bot.send_message(chat_id=chat_id, text="⚠️ Library unavailable.")
        return
    try:
        if ns == "rp":
            await _on_review(app, lib, chat_id, message_id, action, arg)
        else:
            await _on_bookmark(app, lib, chat_id, message_id, action, arg)
    except (ms.MediaError, ValueError) as e:                # expected refusals: said in the chat
        await app.bot.send_message(chat_id=chat_id, text=f"⚠️ {e}")


async def _on_review(app, lib, chat_id: int, message_id: int | None, action: str, arg: str) -> None:
    if action == "t" and (n := _nums(arg, 3)):
        a = await ms.toggle_found(lib, n[0], n[1], n[2])
        await _show(app, chat_id, message_id, *result_view(a, _console(app)))
    elif action in ("ok", "bm") and (n := _nums(arg, 2)):
        a, chosen = await ms.attach_chosen(lib, n[0], n[1], bookmark=action == "bm")
        await _show(app, chat_id, message_id, *result_view(a, _console(app)))
        await app.bot.send_message(chat_id=chat_id, text=(
            f"📎 Attached {len(chosen)} to {(a.get('title') or '')[:60]}"
            + (" and bookmarked them — /repos lists your bookmarks." if action == "bm" else ".")))
    else:
        await app.bot.send_message(chat_id=chat_id, text="(stale button — tap 🔎 Investigate again)")


async def _on_bookmark(app, lib, chat_id: int, message_id: int | None, action: str, arg: str) -> None:
    console = _console(app)
    if action == "pg" and arg.isdigit():
        await _show(app, chat_id, message_id, *await list_view(lib, int(arg), console))
        return
    n = _nums(arg, 2)
    if n is None or action not in ("o", "gh", "rm", "rmy"):
        await app.bot.send_message(chat_id=chat_id, text="(stale button — send /repos again)")
        return
    b, page = await lib.get_bookmark(n[0]), n[1]
    if b is None:
        await _show(app, chat_id, message_id, *await list_view(lib, page, console))
        await app.bot.send_message(chat_id=chat_id, text="(that bookmark was removed)")
        return
    if action == "rmy":
        await lib.delete_bookmark(b["bookmark_id"])
        await _show(app, chat_id, message_id, *await list_view(lib, page, console))
        await app.bot.send_message(chat_id=chat_id, text=f"🗑 Removed the bookmark {b['name']}.")
        return
    if action == "gh":
        b = await ms.bookmark_about(lib, b["bookmark_id"])
    await _show(app, chat_id, message_id, *await card_view(lib, b, page, console, confirm=action == "rm"))


def register(app: Application) -> None:
    app.add_handler(CommandHandler("repos", repos_cmd))
