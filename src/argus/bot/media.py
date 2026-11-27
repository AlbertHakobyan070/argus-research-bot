"""Media: /video, /find, /fetch, /quality, /widgets, /transcript and
/transcripts, pasted media links, the per-chat result pool and its m:*
buttons, the video card (vc:* buttons and widget queries, see cards.py),
the qs:* quality buttons and the t:send transcript-library buttons."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from telegram import (
    InputFile,
    InlineKeyboardButton, InlineKeyboardMarkup, Update, WebAppInfo,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from .. import media_service as ms
from ..config import get_settings
from ..media import (
    detect_platform, download_media, extract_urls, probe_url, quality_format,
    reddit_search, resolve_ffmpeg,
)
from ..engines.registry import EngineConfigError
from ..engines.transcribers import asr_label
from ..transcribe import engines_config, transcribe_url
from . import cards, tgx
from .common import (
    _TG_UPLOAD_CAP_BYTES, _allowed, _fmt_bytes, _fmt_duration, _get_library,
    _html_escape_for_tg, _md_escape, _parse_indices, _safe_send,
)
from .progress import edit_text

logger = logging.getLogger("argus.bot")


# Per-chat last /video result pool. Keyed by thread_id; entries hold
# (results list, timestamp). Capped to a TTL so we never leak if the
# user walks away mid-session. ``/transcript <indices>`` reads from here.
_VIDEO_POOL_TTL_S = 1800   # 30 minutes — long enough for a follow-up,
                            # short enough that a stale pool doesn't
                            # silently outlive its relevance.
_video_pool: dict[str, dict[str, Any]] = {}


def _pool_get(thread_id: str) -> list[dict[str, Any]] | None:
    """Return the cached /video result list for ``thread_id``, or None
    if it's missing/expired. Touches nothing on miss."""
    entry = _video_pool.get(thread_id)
    if not entry:
        return None
    if (time.time() - entry["ts"]) > _VIDEO_POOL_TTL_S:
        _video_pool.pop(thread_id, None)
        return None
    return entry["results"]


def _pool_put(thread_id: str, results: list[dict[str, Any]]) -> None:
    _video_pool[thread_id] = {"results": list(results), "ts": time.time()}


# ---------------------------------------------------------------------------
# /video — YouTube search (Phase 2)
# ---------------------------------------------------------------------------


async def video_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/video [shorts] <query> — search YouTube and list ranked videos.

    ``/video shorts <query>`` biases toward Shorts. Metadata-only (title,
    channel, duration, link); no download, so it works without yt-dlp
    impersonation.
    """
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    args = list(ctx.args or [])
    shorts = False
    if args and args[0].lower().lstrip("-/") == "shorts":
        shorts = True
        args = args[1:]
    if not args:
        await update.message.reply_text("Usage: /video [shorts] <query>")
        return
    query = " ".join(args)
    chat_id = update.effective_chat.id
    await ctx.application.bot.send_chat_action(chat_id, ChatAction.TYPING)

    from ..tools import youtube_search
    # youtube_search shells out to yt-dlp (blocking); run off the event loop.
    try:
        results = await asyncio.to_thread(
            youtube_search, query, max_results=6, shorts=shorts)
    except Exception as e:
        logger.exception("video_cmd search failed")
        await update.message.reply_text(f"⚠️ YouTube search failed: {e}")
        return
    if not results:
        await update.message.reply_text(
            f"🎬 No videos found for “{query}”. Try different terms.")
        return

    header = f"🎬 *YouTube {'Shorts ' if shorts else ''}results* for _{_md_escape(query)}_:\n"
    lines = [header]
    for i, v in enumerate(results, 1):
        dur = _fmt_duration(v.get("duration"))
        title = _md_escape((v.get("title") or "")[:100])
        chan = _md_escape(v.get("channel") or "")
        meta = " · ".join(x for x in (chan, dur) if x)
        lines.append(f"{i}. [{title}]({v['url']})" + (f"\n   _{meta}_" if meta else ""))
    # CTA — tell the user how to pick videos by index. Plain text is
    # simplest on mobile; the /transcript command also accepts "all".
    lines.append("")
    lines.append(
        f"_↪ Reply with `/transcript <indices>` to fetch captions — "
        f"e.g._ `/transcript 2,4` _or_ `/transcript all`. _Pool expires "
        f"in 30 min; results replace any previous pool for this chat._"
    )
    text = "\n".join(lines)
    try:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        # Legacy-markdown parse choked on an exotic title — send plain text
        # (URLs still clickable in Telegram) rather than crash.
        plain = "\n".join(
            [f"YouTube{' Shorts' if shorts else ''} results for {query}:"]
            + [f"{i}. {v.get('title','')} — {v['url']}"
               for i, v in enumerate(results, 1)]
            + ["",
               "Reply: /transcript <indices>  (e.g. /transcript 2,4 or /transcript all)"])
        await update.message.reply_text(plain)
    # Stash the result list for /transcript follow-ups (keyed by thread_id).
    for v in results:
        v.setdefault("platform", "youtube")
    thread_id = f"tg:{update.effective_chat.id}"
    _pool_put(thread_id, results)
    logger.info("/video pooled %d result(s) for %s (query=%r)",
                len(results), thread_id, query)


# ---------------------------------------------------------------------------
# Media engine commands (Phase 2): /fetch, /find, /quality + URL paste
# ---------------------------------------------------------------------------


def _media_keyboard(n: int) -> InlineKeyboardMarkup:
    """One row per result: download / transcript / both, index-scoped.

    Media buttons act on the per-chat result pool (30-min TTL), not on a
    run — so the callback carries the 1-based pool index.
    """
    rows = []
    for i in range(1, n + 1):
        rows.append([
            InlineKeyboardButton(f"⬇ {i}", callback_data=f"m:dl:{i}"),
            InlineKeyboardButton(f"🎵 {i}", callback_data=f"m:mp3:{i}"),
            InlineKeyboardButton(f"📝 {i}", callback_data=f"m:tr:{i}"),
            InlineKeyboardButton(f"⬇+📝 {i}", callback_data=f"m:both:{i}"),
        ])
    return InlineKeyboardMarkup(rows)


# ---------------------------------------------------------------------------
# Library items in the chat (spec 2026-10-26): item buttons (mx:*), the tag
# suggestion with Accept, file imports, /media
# ---------------------------------------------------------------------------

# Bots may download files only up to 20 MB through the cloud Bot API.
_TG_DOWNLOAD_CAP_BYTES = 20 * 1024 * 1024


def _public(app) -> str | None:
    return (app.bot_data.get("webapp_public_url") if isinstance(app.bot_data, dict) else None) or None


def _item_keyboard(app, a: dict) -> InlineKeyboardMarkup:
    """MP3 (videos) · Transcribe (video, audio) · Tags · Open in the console."""
    aid, kind = a["asset_id"], ms.media_type(a)
    first = []
    if kind == "video":
        first.append(InlineKeyboardButton("🎵 MP3", callback_data=f"mx:mp3:{aid}"))
    if kind in ("video", "audio"):
        first.append(InlineKeyboardButton("📝 Transcribe", callback_data=f"mx:tr:{aid}"))
    second = [InlineKeyboardButton("🏷 Tags", callback_data=f"mx:tag:{aid}")]
    url = _public(app)
    if url:
        second.append(InlineKeyboardButton("📱 Open", web_app=WebAppInfo(ms.item_url(url, aid))))
    return InlineKeyboardMarkup([row for row in (first, second) if row])


async def _post_suggestion(app, chat_id: int, asset_id: int, job: ms.Job) -> None:
    """The tag job's result in the chat: the suggestion with ✅ Accept and
    ✏️ Edit (the console), or why there is none."""
    if job.state != "done":
        await app.bot.send_message(chat_id=chat_id, text=f"🏷 No tag suggestion: {job.error}")
        return
    a = await app.bot_data["library"].get_asset(asset_id)
    sug = (a or {}).get("meta", {}).get("suggest") or {}
    if not sug.get("tags"):
        return
    lines = [f"🏷 Suggested tags for {(a.get('title') or '')[:80]}:", " ".join(f"#{t}" for t in sug["tags"])]
    if sug.get("group"):
        lines.append(f"Group: {sug['group']}")
    if sug.get("summary"):
        lines += ["", sug["summary"]]
    repos = [link["url"] for link in sug.get("links") or [] if link.get("kind") == "repo"][:5]
    if repos:
        lines += ["", "Repos:", *repos]
    rows = [[InlineKeyboardButton("✅ Accept", callback_data=f"mx:ok:{asset_id}", style="success")]]
    url = _public(app)
    if url:
        rows[0].append(InlineKeyboardButton("✏️ Edit", web_app=WebAppInfo(ms.item_url(url, asset_id))))
    await app.bot.send_message(chat_id=chat_id, text="\n".join(lines), reply_markup=InlineKeyboardMarkup(rows),
                               disable_web_page_preview=True)


async def _autotag(app, chat_id: int, asset_id: int | None, title: str) -> None:
    async def then(job: ms.Job) -> None:
        await _post_suggestion(app, chat_id, asset_id, job)
    await ms.maybe_autotag(app.bot_data["library"], asset_id, title, then=then)


async def _deliver(app, chat_id: int, job: ms.Job, what: str) -> None:
    """A finished chat job's file in the chat (a link when too big), or why it failed."""
    if job.state != "done":
        await app.bot.send_message(chat_id=chat_id, text=f"⚠️ {what} failed: {job.error}")
        return
    a = await app.bot_data["library"].get_asset(job.asset_id)
    s = get_settings()
    try:
        await ms.send_to_chat(app.bot, chat_id, a, bot_token=s.telegram_bot_token, public_url=_public(app),
                              hours=(await ms.load_settings(app.bot_data["library"]))["link_hours"])
    except ms.MediaError as e:
        await app.bot.send_message(chat_id=chat_id, text=f"📁 {what} saved to the vault, but: {e}\n{a['path']}")
    except Exception as e:
        logger.exception("%s delivery failed", what)
        await app.bot.send_message(chat_id=chat_id, text=(f"📁 {what} saved to the vault, but Telegram delivery "
                                                          f"failed ({type(e).__name__}: {str(e)[:200]}).\n{a['path']}"))


def _start_item_job(app, chat_id: int, kind: str, a: dict) -> None:
    """mp3 / transcribe for one library item, its result sent to the chat."""
    lib, s, aid = app.bot_data["library"], get_settings(), a["asset_id"]
    title = a.get("title") or Path(a["path"]).name
    if kind == "mp3":
        ms.start_job("convert", f"MP3 · {title}", lambda job: ms.convert(lib, aid, job),
                     then=lambda job: _deliver(app, chat_id, job, "MP3"))
        return

    async def then(job: ms.Job) -> None:
        await _deliver(app, chat_id, job, "Transcript")
        if job.state == "done" and not a["meta"].get("tags"):
            await _autotag(app, chat_id, aid, title)        # a transcript makes better tags
    ms.start_job("transcribe", f"Transcript · {title}", lambda job: ms.transcribe(lib, s, aid, job), then=then)


async def handle_item_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE, action: str, raw_id: str) -> None:
    """``mx:<action>:<asset_id>``: mp3 · tr (transcribe) · tag (suggest) · ok (accept the suggestion)."""
    q = update.callback_query
    app = ctx.application
    chat_id = q.message.chat.id if q.message else update.effective_chat.id
    lib = _get_library(ctx)
    a = await lib.get_asset(int(raw_id)) if lib is not None and raw_id.isdigit() else None
    if a is None:
        await app.bot.send_message(chat_id=chat_id, text="(that library item no longer exists)")
        return
    title = a.get("title") or Path(a["path"]).name
    if action in ("mp3", "tr"):
        if not Path(a["path"]).is_file():
            await app.bot.send_message(chat_id=chat_id, text=f"⚠️ The file of {title} is missing on disk.")
            return
        _start_item_job(app, chat_id, action, a)
        await app.bot.send_message(chat_id=chat_id, text=(
            f"🎵 Converting to mp3: {title[:80]}…" if action == "mp3" else
            f"📝 Transcribing: {title[:80]}… (speech-to-text can take a few minutes)"))
    elif action == "tag":
        ms.tag_job(lib, a["asset_id"], title, then=lambda job: _post_suggestion(app, chat_id, a["asset_id"], job))
        await app.bot.send_message(chat_id=chat_id, text=f"🏷 Asking for tags: {title[:80]}…")
    elif action == "ok":
        sug = a["meta"].get("suggest") or {}
        if not sug.get("tags"):
            await app.bot.send_message(chat_id=chat_id, text="(no suggestion to accept — tap 🏷 Tags first)")
            return
        row = await ms.accept(lib, a["asset_id"], tags=sug["tags"], group=sug.get("group"))
        done = " ".join(f"#{t}" for t in row["meta"]["tags"]) + (f" · {row['meta']['group']}" if row["meta"].get("group") else "")
        try:
            await q.edit_message_reply_markup(reply_markup=None)
        except Exception:
            logger.info("could not clear the suggestion's buttons (an old message?)")
        await app.bot.send_message(chat_id=chat_id, text=f"✅ Tagged {title[:60]}: {done}")


_KIND_NAMES = {"video": ("video", ".mp4"), "audio": ("audio", ".mp3"), "voice": ("voice", ".ogg"),
               "video_note": ("video_note", ".mp4")}


async def handle_file(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """A video, audio, voice or video note, or a document (video, audio or a
    .txt/.md/.srt/.vtt transcript) sent to the bot: saved to the library
    with its item buttons. Over the Bot API's 20 MB download cap the bot
    says so and points to the console upload."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        return
    msg = update.message
    app = ctx.application
    lib = _get_library(ctx)
    kind = next((k for k in ("video", "audio", "voice", "video_note", "document") if getattr(msg, k, None)), None)
    if kind is None or lib is None:
        return
    att = getattr(msg, kind)
    stem, ext = _KIND_NAMES.get(kind, ("file", ""))
    name = getattr(att, "file_name", None) or f"{stem}_{msg.message_id}{ext}"
    try:
        ms.import_kind(name)
    except ms.MediaError as e:
        await msg.reply_text(f"⚠️ Not saved: {e}")
        return
    size = att.file_size or 0
    if size > _TG_DOWNLOAD_CAP_BYTES:
        url = _public(app)
        kb = (InlineKeyboardMarkup([[InlineKeyboardButton("⬆️ Upload in Argus",
                                                          web_app=WebAppInfo(f"{url.rstrip('/')}/#/media/get"))]])
              if url else None)
        await msg.reply_text(
            f"📦 {name} is {size / 1048576:.0f} MB. Telegram lets bots download files only up to 20 MB, so "
            "upload it from the Argus console instead (Media → Get → From this device)"
            + ("." if url else " — open the console from the Argus menu button."), reply_markup=kb)
        return
    note = await msg.reply_text(f"⬇️ Saving {name}…")
    part = Path(s.media_root) / ".uploads" / f"tg_{msg.chat_id}_{msg.message_id}.part"
    part.parent.mkdir(parents=True, exist_ok=True)
    try:
        tg_file = await att.get_file()
        await tg_file.download_to_drive(custom_path=part)
        caption = (msg.caption or "").strip()
        asset_id = await ms.import_file(lib, s, part, filename=name, platform="telegram",
                                        title=caption.splitlines()[0][:200] if caption else None)
    except Exception as e:
        part.unlink(missing_ok=True)
        logger.exception("chat import failed for %s", name)
        await note.edit_text(f"⚠️ Could not save {name}: {type(e).__name__}: {str(e)[:300]}")
        return
    a = await lib.get_asset(asset_id)
    extra = (f"\n↪ feed it into research: /append <run-id> asset:{asset_id}" if a["kind"] == "transcript" else "")
    await note.edit_text(f"✅ Saved to the library as asset:{asset_id} — {a.get('title')}"
                         f" ({a.get('bytes', 0) / 1048576:.1f} MB){extra}", reply_markup=_item_keyboard(app, a))
    await _autotag(app, msg.chat_id, asset_id, a.get("title") or name)


async def rename_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/rename <asset:N|N> <new title> — retitle a library item (and its parts)."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    lib = _get_library(ctx)
    if lib is None:
        await update.message.reply_text("⚠️ Library unavailable.")
        return
    args = list(ctx.args or [])
    ref = args[0].removeprefix("asset:") if args else ""
    if not ref.isdigit() or len(args) < 2:
        await update.message.reply_text("Usage: /rename <asset:N> <new title>\n"
                                        "Ids show on saved items, in /transcripts and on the console's item page.")
        return
    try:
        row = await ms.rename(lib, int(ref), " ".join(args[1:]))
    except (ValueError, ms.MediaError) as e:
        await update.message.reply_text(f"⚠️ {e}")
        return
    await update.message.reply_text(f"✏️ asset:{row['asset_id']} is now “{row['title']}”.",
                                    reply_markup=_item_keyboard(ctx.application, row))


async def media_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/media — what the library holds, with buttons into the console."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    lib = _get_library(ctx)
    if lib is None:
        await update.message.reply_text("⚠️ Library unavailable.")
        return
    _, total = await lib.search_assets(limit=1)
    _, untagged = await lib.search_assets(review="untagged", limit=1)
    facets = await lib.asset_facets()
    top = " ".join(f"#{t}" for t, _ in facets["tags"][:12])
    text = (f"🗃 Media library: {total} item(s) · {facets['pending']} tag suggestion(s) to review · {untagged} untagged"
            + (f"\nTop tags: {top}" if top else "")
            + "\n\nSend me a video, audio or transcript file to save it; paste a link to download it."
            + "\nRename one: /rename asset:<N> <new title>")
    url = _public(ctx.application)
    kb = None
    if url:
        base = url.rstrip("/")
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("📚 Library", web_app=WebAppInfo(f"{base}/#/media")),
                                    InlineKeyboardButton("🏷 Review", web_app=WebAppInfo(f"{base}/#/media/review"))],
                                   [InlineKeyboardButton("⬆️ Get / upload", web_app=WebAppInfo(f"{base}/#/media/get"))]])
    else:
        text += "\n(The console is not reachable right now — see /status.)"
    await update.message.reply_text(text, reply_markup=kb)


async def _get_media_quality(app) -> str:
    lib = app.bot_data.get("library") if isinstance(app.bot_data, dict) else None
    if lib is None:
        return "auto"
    try:
        return (await lib.get_setting("media_quality", "auto")) or "auto"
    except Exception:
        logger.exception("could not read media_quality; using auto")
        return "auto"


def _lib(app):
    return app.bot_data.get("library") if isinstance(app.bot_data, dict) else None


async def _send_card(app, card: cards.Card) -> bool:
    """Post a video card; a card Telegram refuses outright is said in the
    chat (the download, if any, still reports its outcome)."""
    try:
        await cards.send(app.bot, card, lib=_lib(app))
        return True
    except TelegramError as e:
        logger.exception("the video card for %s could not be sent", card.url)
        await app.bot.send_message(chat_id=card.chat_id, text=(
            f"⚠️ Could not show the video card ({type(e).__name__}: {e.message}).\n{card.url}"))
        return False


async def open_card(app, chat_id: int, url: str, *, fmt: str = "v", title: str = "") -> cards.Card | None:
    """The video card for one link: posted at once ("Reading the link…"),
    then filled by a probe with the formats on offer."""
    lib = _lib(app)
    card = cards.Card(chat_id=chat_id, url=url, platform=detect_platform(url) or "", fmt=fmt,
                      title=title if title != url else "",
                      mp3=(await ms.load_settings(lib))["mp3_quality"] if lib is not None else "320")
    if not await _send_card(app, card):
        return None
    app.create_task(_probe_card(app, card))
    return card


async def _probe_card(app, card: cards.Card) -> None:
    """Background task: what the link offers (title, heights, sizes) onto
    the card, which then asks for a format and quality."""
    try:
        info = await probe_url(card.url)
    except Exception as e:                       # RuntimeError carries the downloader's explanation
        if not isinstance(e, RuntimeError):
            logger.exception("probing %s crashed", card.url)
        if card.phase == "read":                 # not cancelled meanwhile
            card.phase, card.failed_at, card.error = "fail", "probe", f"Could not read the link: {e}"
            await cards.settle(app.bot, card)
        return
    if card.phase != "read":
        return
    cards.fill(card, info, await _get_media_quality(app))
    card.phase = "pick"
    await cards.settle(app.bot, card)


def _on_download_done(app, card: cards.Card):
    """A download cancelled before it ever ran still ends its card."""
    def done(task) -> None:
        if task.cancelled() and card.phase not in ("done", "fail", "x"):
            card.phase = "x"
            asyncio.ensure_future(cards.settle(app.bot, card))
    return done


async def card_action(app, chat_id: int | None, message_id: int | None, action: str, arg: str, *,
                      answer=None) -> cards.Card | None:
    """One tap on a video card, from its buttons or its widget: the state
    changes at once, ``answer(card)`` (a widget's answer, due within 15 s)
    goes next, then the card is redrawn. None: no such card (expired)."""
    c = cards.get(chat_id, message_id)
    if c is None:
        if answer is not None:
            await answer(None)
        return None
    bot = app.bot
    if action == "plain":
        if answer is not None:
            await answer(c)
        if c.widget:
            await cards.to_native(bot, c)
        return c
    changed = False
    if action in ("f", "q", "a"):
        changed = cards.apply(c, action, arg)
    elif action == "ok" and c.phase == "pick":
        c.phase, c.stage, c.eta = "queue", "starting", ""     # nothing left from an earlier attempt
        c.pct = c.done_bytes = c.total_bytes = None
        c.task = app.create_task(_fetch_one_media(app, chat_id, c.url, card=c))
        c.task.add_done_callback(_on_download_done(app, c))
        changed = True
    elif action == "x":
        if c.phase in ("read", "pick"):
            c.phase, changed = "x", True
        elif c.phase in ("queue", "dl") and c.task is not None:
            c.task.cancel()                      # the download draws its cancelled state
    elif action == "retry" and c.phase == "fail":
        c.error = ""
        if c.failed_at == "probe":
            c.phase = "read"
            app.create_task(_probe_card(app, c))
        else:
            c.phase = "pick"
        changed = True
    if answer is not None:
        await answer(c)
    if changed:
        await cards.settle(bot, c)
    return c


async def handle_card_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE, action: str, arg: str) -> None:
    """``vc:<action>:<arg>``: a tap on a native video card."""
    q = update.callback_query
    chat_id = q.message.chat.id if q.message else update.effective_chat.id
    message_id = q.message.message_id if q.message else None
    if await card_action(ctx.application, chat_id, message_id, action, arg) is not None:
        return
    rows = [[b for b in row if not (b.callback_data or "").startswith("vc:")]   # the item buttons still work
            for row in (q.message.reply_markup.inline_keyboard
                        if q.message and q.message.reply_markup else ())]
    rows = [r for r in rows if r]
    try:
        await q.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(rows) if rows else None)
    except TelegramError as e:
        logger.info("could not clear an expired video card's buttons (%s)", e)
    await ctx.application.bot.send_message(chat_id=chat_id, text=(
        "(this video card expired — tap ⬇ on the list again or paste the link)"))


async def handle_widget_query(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """A tap inside a video-card widget (Update.widget_query): answered with
    the card's new state, which the widget draws at once."""
    wq = cards.widget_query(update)
    bot = ctx.application.bot

    async def answer(c: cards.Card | None) -> None:
        data = (cards.widget_state(c) if c is not None else
                cards.off_state("This card expired — paste the link again."))
        try:
            await tgx.answer_widget_query(bot, wq["id"], data)
        except TelegramError:
            logger.exception("could not answer widget query %s", wq["id"])

    if not _allowed(get_settings(), (wq.get("from") or {}).get("id")):
        try:
            await tgx.answer_widget_query(bot, wq["id"], cards.off_state("Unauthorized."))
        except TelegramError:
            logger.exception("could not answer widget query %s", wq["id"])
        return
    action, _, arg = (wq.get("data") or "").partition(":")
    await card_action(ctx.application, (wq.get("chat") or {}).get("id"), wq.get("message_id"),
                      action, arg or "_", answer=answer)


async def _fetch_one_media(app, chat_id: int, url: str, *, card: cards.Card | None = None) -> None:
    """Background task: download one URL into the vault on its video card
    (live progress, Cancel), register it, deliver the file when small
    enough. Without a card (/fetch) it opens one at the /quality setting."""
    bot = app.bot
    s = get_settings()
    lib = _lib(app)
    if card is None:
        card = cards.Card(chat_id=chat_id, url=url, platform=detect_platform(url) or "",
                          quality=await _get_media_quality(app), phase="queue", stage="starting")
        await _send_card(app, card)
        card.task = asyncio.current_task()
    audio = card.mp3 if card.fmt == "a" else None
    quality = card.quality

    def on_status(st: dict) -> None:
        if st.get("stage") and st["stage"] != card.stage:         # a new stream: its own numbers
            card.stage, card.pct, card.eta, card.done_bytes, card.total_bytes = st["stage"], None, "", None, None
        card.phase = "queue" if card.stage == "queued" else "dl"
        for k in ("pct", "eta", "done_bytes", "total_bytes"):
            if st.get(k) is not None:
                setattr(card, k, st[k])
        cards.kick(bot, card)

    try:
        r = await download_media(url, dest_root=s.media_root, quality=quality, audio=audio,
                                 any_url=audio is not None, on_status=on_status)
    except asyncio.CancelledError:
        card.phase = "x"
        await cards.settle(bot, card)
        raise
    except Exception as e:
        logger.exception("download_media crashed for %s", url)
        card.phase, card.failed_at, card.error = "fail", "download", f"Download crashed: {e}"
        await cards.settle(bot, card)
        return
    card.task = None                             # the file is here: too late to cancel

    if not r.ok:
        card.phase, card.failed_at = "fail", "download"
        card.error = f"Download failed ({r.platform or '?'}): {(r.error or 'unknown error')[:800]}"
        await cards.settle(bot, card)
        return

    asset = None
    if lib is not None:
        meta: dict[str, Any] = ({"via": "telegram", "format": "mp3", "mp3_quality": audio} if audio
                                else {"quality": quality})
        meta["info_json"] = r.info_json_path
        if r.thumb_path:
            meta["thumb"] = r.thumb_path
        try:
            asset = await lib.get_asset(await lib.add_asset(
                kind="media", platform=r.platform, source_url=url,
                media_id=r.media_id, title=r.title or url, path=r.path,
                size_bytes=r.size_bytes, duration_s=r.duration_s, meta=meta))
        except Exception:
            logger.exception("asset registration failed for %s", r.path)

    kind = (f"MP3 {cards._alabel(audio)}" if audio
            else f"{cards._qlabel(quality)} {Path(r.path).suffix.lstrip('.').upper()}")
    card.title = card.title or r.title or url
    card.summary = " · ".join(x for x in (kind, _fmt_bytes(r.size_bytes), _fmt_duration(r.duration_s),
                                          f"asset:{asset['asset_id']}" if asset else "") if x)
    card.done_markup = _item_keyboard(app, asset) if asset is not None else None
    card.phase = "done"
    await cards.settle(bot, card)
    if asset is not None:
        await _autotag(app, chat_id, asset["asset_id"], r.title or url)

    if audio and asset is not None:
        try:
            await ms.send_to_chat(bot, chat_id, asset, bot_token=s.telegram_bot_token, public_url=_public(app),
                                  hours=(await ms.load_settings(lib))["link_hours"])
        except ms.MediaError as e:
            await bot.send_message(chat_id=chat_id, text=f"📁 MP3 saved to the vault, but: {e}\n{r.path}")
        except Exception as e:
            logger.exception("MP3 delivery failed")
            await bot.send_message(chat_id=chat_id, text=(
                f"📁 MP3 saved to the vault, but Telegram delivery failed "
                f"({type(e).__name__}: {str(e)[:200]}).\n{r.path}"))
        return

    if r.size_bytes > _TG_UPLOAD_CAP_BYTES and asset is not None and _public(app):
        # Too big for a bot upload: a private link from this PC instead of a dead end.
        try:
            await ms.send_to_chat(bot, chat_id, asset, bot_token=s.telegram_bot_token, public_url=_public(app),
                                  hours=(await ms.load_settings(lib))["link_hours"])
            return
        except Exception:
            logger.exception("download link for %s failed; saying where the file is", r.path)

    if r.size_bytes <= _TG_UPLOAD_CAP_BYTES:
        try:
            with Path(r.path).open("rb") as f:
                if audio:                        # an MP3 the library could not take
                    await bot.send_audio(chat_id=chat_id, audio=f, caption=(r.title or Path(r.path).name)[:1024])
                else:
                    await bot.send_video(
                        chat_id=chat_id, video=f,
                        caption=(r.title or Path(r.path).name)[:1024],
                        supports_streaming=True)
        except Exception as e_vid:
            logger.warning("send_video failed (%s); retrying as document",
                           e_vid)
            try:
                with Path(r.path).open("rb") as f:
                    await bot.send_document(
                        chat_id=chat_id, document=f,
                        filename=Path(r.path).name,
                        caption=(r.title or "")[:1024])
            except Exception as e_doc:
                # Both sends failed. Say so — the old code only logged,
                # so the user saw "✅ Saved" and silently got no file.
                logger.exception("document send failed too")
                await bot.send_message(
                    chat_id=chat_id,
                    text=("📁 Saved to the vault, but Telegram delivery "
                          f"failed ({type(e_doc).__name__}: "
                          f"{str(e_doc)[:200]}).\nThe file is intact at "
                          f"{r.path}"))
    else:
        # Name the cause: a high quality is what puts most long videos
        # over the cap, and "over the cap" alone doesn't say so.
        why = ("\n\nA lower quality keeps most clips under the cap: pick 720p on the card "
               "(or /quality 720 for /fetch)."
               if not audio and (quality == "max" or (quality.isdigit() and int(quality) > 720)) else "")
        await bot.send_message(
            chat_id=chat_id,
            text=(f"📁 File is {_fmt_bytes(r.size_bytes)} — over Telegram's 50 MB bot "
                  f"upload cap, so it stays in the vault at {r.path}{why}"))


async def _transcript_for_item(app, chat_id: int, item: dict) -> None:
    """Background task: transcript for one pool item, persisted into the
    DS vault, by the configured transcriber (YouTube: its captions/asr
    order; X/Reddit/Instagram: the ASR engine, after a download)."""
    bot = app.bot
    s = get_settings()
    url = item.get("url") or ""
    platform = item.get("platform") or detect_platform(url) or "?"
    lib = app.bot_data.get("library") if isinstance(app.bot_data, dict) else None

    try:
        t = (await engines_config(lib)).transcriber
    except EngineConfigError:
        note = "🎙 Transcribing…"   # transcribe_url reports the configuration problem itself, next
    else:
        steps = t.youtube if platform == "youtube" else ["asr"]
        note = "🎙 Transcribing via " + " → ".join(
            asr_label(t) if step == "asr" else "captions" for step in steps)
        if "asr" in steps:
            note += (" (speech-to-text downloads the media first — this can take a few minutes"
                     + ("; the first ever run also downloads the model" if t.asr_engine == "faster_whisper" else "")
                     + ")")
        note += "…"
    await bot.send_message(chat_id=chat_id, text=note)

    quality = await _get_media_quality(app)
    try:
        r = await transcribe_url(
            url, transcripts_root=s.transcripts_root,
            media_root=s.media_root, quality=quality, library=lib)
    except Exception as e:
        logger.exception("transcribe_url crashed")
        await bot.send_message(chat_id=chat_id,
                               text=f"⚠️ Transcript crashed: {e}")
        return
    if not r.ok:
        await bot.send_message(
            chat_id=chat_id,
            text=f"⚠️ Transcript unavailable: {(r.error or '?')[:500]}")
        return
    title = r.title or item.get("title") or "(untitled)"
    p = Path(r.path)
    try:
        with p.open("rb") as f:
            await bot.send_document(
                chat_id=chat_id,
                document=InputFile(f, filename=p.name),
                caption=(f"{title} — {r.backend}"
                         + (f" ({r.language})" if r.language else ""))[:1024])
    except Exception:
        logger.exception("transcript document send failed")
        await bot.send_message(
            chat_id=chat_id, text=f"📝 Transcript saved:\n{r.path}")


async def fetch_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/fetch <url> [url…] — download media links into the vault."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    urls = extract_urls(" ".join(ctx.args or []))
    if not urls:
        await update.message.reply_text(
            "Usage: /fetch <url> [url…]\n"
            "Supported: YouTube (incl. shorts), X/Twitter, Reddit, "
            "Instagram reels/posts.")
        return
    chat_id = update.effective_chat.id
    good = [u for u in urls if detect_platform(u)]
    bad = [u for u in urls if not detect_platform(u)]
    if bad:
        await update.message.reply_text(
            "⚠️ Skipping unsupported link(s):\n" + "\n".join(bad[:5]))
    if not good:
        return
    for u in good:
        ctx.application.create_task(
            _fetch_one_media(ctx.application, chat_id, u))
    await update.message.reply_text(
        f"⏬ Queued {len(good)} download(s) — progress follows.")


async def find_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/find [yt|shorts|reddit] <query> — search, then act via buttons."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    args = list(ctx.args or [])
    where = "yt"
    if args and args[0].lower().lstrip("-/") in ("yt", "youtube", "shorts",
                                                 "reddit"):
        where = args.pop(0).lower().lstrip("-/")
    if not args:
        await update.message.reply_text(
            "Usage: /find [yt|shorts|reddit] <query>")
        return
    query = " ".join(args)
    chat_id = update.effective_chat.id
    await ctx.application.bot.send_chat_action(chat_id, ChatAction.TYPING)

    if where == "reddit":
        results = await reddit_search(query, limit=6)
    else:
        from ..tools import youtube_search
        try:
            results = await asyncio.to_thread(
                youtube_search, query, max_results=6,
                shorts=(where == "shorts"))
        except Exception as e:
            logger.exception("find: youtube search failed")
            await update.message.reply_text(f"⚠️ Search failed: {e}")
            return
        for r in results:
            r.setdefault("platform", "youtube")
    if not results:
        await update.message.reply_text(
            f"🔍 No {where} videos found for “{query}”.")
        return

    lines = [f"🔍 *{where} results* for _{_md_escape(query)}_:", ""]
    for i, v in enumerate(results, 1):
        dur = _fmt_duration(v.get("duration"))
        title = _md_escape((v.get("title") or "")[:90])
        chan = _md_escape(v.get("channel") or "")
        meta = " · ".join(x for x in (chan, dur) if x)
        lines.append(f"{i}. [{title}]({v['url']})"
                     + (f"\n   _{meta}_" if meta else ""))
    lines.append("")
    lines.append("_⬇ download to vault · 📝 transcript · ⬇+📝 both. "
                 "Pool lives 30 min; /transcript <indices> also works._")
    await _safe_send(ctx.application.bot, chat_id, "\n".join(lines),
                     reply_markup=_media_keyboard(len(results)),
                     parse_mode=ParseMode.MARKDOWN)
    _pool_put(f"tg:{chat_id}", results)


async def quality_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/quality [auto|min|max|<height>] — global download quality."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    lib = _get_library(ctx)
    ffmpeg = resolve_ffmpeg()
    if not ctx.args:
        cur = "auto"
        if lib is not None:
            cur = (await lib.get_setting("media_quality", "auto")) or "auto"
        await update.message.reply_text(_quality_text(cur, ffmpeg), reply_markup=_quality_keyboard(cur))
        return
    v = ctx.args[0].lower().strip()
    try:
        quality_format(v, ffmpeg_available=bool(ffmpeg))
    except ValueError as e:
        await update.message.reply_text(f"⚠️ {e}")
        return
    if lib is None:
        await update.message.reply_text(
            "⚠️ Library unavailable — setting not persisted.")
        return
    await lib.set_setting("media_quality", v)
    await update.message.reply_text(f"🎚 Download quality set to: {v}")


_QUALITY_ROWS = (("auto", "max", "min"), ("2160", "1440", "1080"), ("720", "480", "360"))


def _quality_text(cur: str, ffmpeg) -> str:
    ff_note = "available" if ffmpeg else "MISSING (single-file formats only — no merged hi-res)"
    return (f"🎚 Download quality: {cur}\n"
            f"ffmpeg: {ff_note}\n\n"
            "The default on new video cards and for /fetch. Tap one, or\n"
            "/quality auto | min | max | <pixel height>, e.g. /quality 720")


def _quality_keyboard(cur: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(
        cards._qlabel(v) + (" ✓" if v == cur else ""), callback_data=f"qs:set:{v}",
        style="primary" if v == cur else None) for v in row] for row in _QUALITY_ROWS])


async def handle_quality_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE, action: str, arg: str) -> None:
    """``qs:set:<quality>``: a /quality button."""
    q = update.callback_query
    lib = _get_library(ctx)
    ffmpeg = resolve_ffmpeg()
    try:
        quality_format(arg, ffmpeg_available=bool(ffmpeg))
    except ValueError as e:
        await q.edit_message_text(f"⚠️ {e}")
        return
    if action != "set" or lib is None:
        await q.edit_message_text("⚠️ Library unavailable — setting not persisted.")
        return
    await lib.set_setting("media_quality", arg)
    await edit_text(ctx.application.bot, chat_id=q.message.chat.id, message_id=q.message.message_id,
                    text=_quality_text(arg, ffmpeg) + f"\n\n✅ Set to {arg}.", reply_markup=_quality_keyboard(arg))


async def widgets_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/widgets [on|off] — send video cards as Telegram HTML widgets."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    lib = _get_library(ctx)
    if lib is None:
        await update.message.reply_text("⚠️ Library unavailable.")
        return
    arg = ctx.args[0].lower().strip() if ctx.args else ""
    if arg not in ("", "on", "off"):
        await update.message.reply_text("Usage: /widgets [on|off]")
        return
    if arg:
        await lib.set_setting(cards.WIDGETS_KEY, arg)
        if arg == "on":                          # a new refusal or failure is told again
            for key in ("widget", "widget-error"):
                tgx._noticed.discard((update.effective_chat.id, key))
    cur = (await lib.get_setting(cards.WIDGETS_KEY, "off")) or "off"
    await update.message.reply_text(
        f"🧩 HTML widgets: {cur}\n\n" + (
            "Video cards go out as Telegram widgets (Bot API 10.4). If Telegram refuses one, "
            "Argus says so once, switches widgets off and sends the regular card. /widgets off goes back."
            if cur == "on" else
            "Video cards use regular buttons. /widgets on sends them as Telegram widgets where "
            "Telegram allows it; a refusal switches them back off with one notice."))


async def transcripts_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/transcripts — list the latest vault transcripts with Send buttons."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    lib = _get_library(ctx)
    if lib is None:
        await update.message.reply_text("⚠️ Library unavailable.")
        return
    rows = await lib.list_assets(kind="transcript", limit=10)
    if not rows:
        await update.message.reply_text(
            "No transcripts in the vault yet — use /find + 📝, paste a "
            "link, or /transcript after /video.")
        return
    lines = ["📚 Latest transcripts:"]
    buttons: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for i, a in enumerate(rows, 1):
        kb = (a.get("bytes") or 0) / 1024
        title = (a.get("title") or Path(a["path"]).name)[:60]
        backend = (a.get("meta") or {}).get("backend", "")
        lines.append(f"{i}. [{a.get('platform','?')}] {title}"
                     f" · {kb:.0f} KB" + (f" · {backend}" if backend else "")
                     + f" · asset:{a['asset_id']}")
        row.append(InlineKeyboardButton(
            f"📤 {i}", callback_data=f"t:send:{a['asset_id']}"))
        if len(row) == 5:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    lines.append("")
    lines.append("↪ feed one into research: /append <run-id> asset:<N>")
    await update.message.reply_text(
        "\n".join(lines), reply_markup=InlineKeyboardMarkup(buttons))


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                      chat_id: int, text: str) -> None:
    """The media branch of ``app.on_text``: pasted media links get an
    action keyboard; any other text is ignored."""
    # (2) pasted media links
    urls = [u for u in extract_urls(text) if detect_platform(u)]
    if not urls:
        return
    results = [{"title": u, "url": u, "platform": detect_platform(u)}
               for u in urls]
    _pool_put(f"tg:{chat_id}", results)
    lines = [f"🔗 Detected {len(urls)} media link(s):"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r['platform']}] {r['url']}")
    await update.message.reply_text(
        "\n".join(lines), reply_markup=_media_keyboard(len(results)),
        disable_web_page_preview=True)


# ---------------------------------------------------------------------------
# /transcript — fetch captions for videos from the last /video pool.
# ---------------------------------------------------------------------------


async def transcript_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/transcript <indices|all> — fetch YouTube auto-captions for the
    picked videos from this chat's result pool (filled by /video, /find
    or a pasted link; 30-minute TTL).

    Index syntax: ``2``, ``2,4``, ``2 4``, ``1-3``, ``all``, ``*``.
    Each fetched transcript is delivered as:
      - a short Telegram message with the title + a preview snippet, then
      - a ``.txt`` attachment carrying the FULL plain text (with
        ``format=srt``: the raw VTT, timestamps kept, named ``.srt``).
    """
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    chat_id = update.effective_chat.id
    thread_id = f"tg:{chat_id}"
    pool = _pool_get(thread_id)
    if not pool:
        await update.message.reply_text(
            "No recent /video results for this chat. "
            "Run `/video <query>` first, then pick indices."
        )
        return
    raw = " ".join(ctx.args or []).strip()
    if not raw:
        await update.message.reply_text(
            "Usage: /transcript <indices> [format=txt|srt]\n"
            "  /transcript 2,4            (plain text, default)\n"
            "  /transcript 2 format=srt   (timestamps preserved)"
        )
        return

    # Strip `format=txt|srt` tokens from `raw` so _parse_indices sees
    # only index tokens. Default format is 'txt' (legacy behaviour).
    fmt = "txt"
    tokens = raw.split()
    kept: list[str] = []
    for tok in tokens:
        if tok.lower().startswith("format="):
            value = tok.split("=", 1)[1].lower()
            if value not in ("txt", "srt"):
                await update.message.reply_text(
                    f"Unknown format {value!r}; use 'txt' or 'srt'."
                )
                return
            fmt = value
        else:
            kept.append(tok)
    raw_indices = " ".join(kept).strip()
    indices = _parse_indices(raw_indices) if raw_indices else []
    n_total = len(pool)
    # Special-case ``all`` / empty-list-means-all: caller passed "all" /
    # ``*`` sentinel.
    if not indices:
        if raw_indices.lower() in ("all", "*"):
            indices = list(range(1, n_total + 1))
        else:
            await update.message.reply_text(
                f"No valid indices in _{_md_escape(raw)}_. "
                f"Pool has {n_total} video(s); try "
                f"`/transcript 1` or `/transcript all`."
            )
            return
    # Clamp / dedupe / preserve order
    seen: set[int] = set()
    picked: list[tuple[int, dict[str, Any]]] = []
    for i in indices:
        if i in seen:
            continue
        seen.add(i)
        if 1 <= i <= n_total:
            picked.append((i, pool[i - 1]))

    invalid = [i for i in indices if i < 1 or i > n_total]
    if not picked:
        await update.message.reply_text(
            f"None of {indices} are in the pool "
            f"(valid range: 1–{n_total})."
        )
        return
    if invalid:
        await update.message.reply_text(
            f"⚠️ Ignoring out-of-range indices: {invalid} (valid range 1–{n_total})."
        )

    await ctx.application.bot.send_chat_action(chat_id, ChatAction.TYPING)

    from ..tools import youtube_video_transcript

    delivered = 0
    for idx, v in picked:
        url = v.get("url") or ""
        title = v.get("title") or "(untitled)"
        channel = v.get("channel") or ""
        dur = _fmt_duration(v.get("duration"))
        meta = " · ".join(x for x in (channel, dur) if x)
        # Sequential (one fetch per video) — yt-dlp is rate-limited and
        # parallel fetches can hammer YouTube. For 6 videos this is fine
        # (~60 s total); batch parallelism is future work.
        # v2 Phase 3: deliverables persist into the DS-vault transcripts
        # folder (per-platform subdir) instead of the temp cache.
        try:
            r = await asyncio.to_thread(
                youtube_video_transcript, url, timeout=90, format=fmt,
                out_dir=s.transcripts_root / "youtube")
        except ValueError as ve:
            await update.message.reply_text(
                f"⚠️ {idx}. {_md_escape(title)}\n"
                f"   `{_md_escape(url)}`\n"
                f"   bad format: {_md_escape(str(ve))}"
            )
            continue
        except Exception as e:
            logger.exception("youtube_video_transcript crashed")
            await update.message.reply_text(
                f"⚠️ {idx}. {_md_escape(title)} — fetch crashed: {e}"
            )
            continue
        if not r.ok:
            await update.message.reply_text(
                f"⚠️ {idx}. {_md_escape(title)}\n"
                f"   `{_md_escape(url)}`\n"
                f"   transcript unavailable: {_md_escape(r.error or 'unknown error')}"
            )
            continue
        # Prefer the richer metadata from the tool result; fall back to the
        # /video pool values if the info-json didn't include them.
        t_title = r.title or title
        t_channel = r.channel or channel
        t_dur = _fmt_duration(r.duration) or dur
        header = (
            f"📝 *Transcript {idx}/{n_total}* — "
            f"_{_md_escape(t_title)}_\n"
            f"   `{_md_escape(url)}`\n"
            f"   _{_md_escape(' · '.join(x for x in (t_channel, t_dur) if x))}_"
            f"   ·  lang: `{_md_escape(r.language or '?')}`"
            f"   ·  format: `{fmt}`"
            + (" _(timestamps preserved)_" if fmt == "srt" else "")
        )
        text = r.transcript_text or ""
        # Telegram message limit ~4096 — keep the preview tight.
        # 3500 chars leaves headroom for the header.
        PREVIEW = 3500 - len(header)
        if PREVIEW < 800:
            PREVIEW = 800
        if len(text) > PREVIEW:
            snippet = text[:PREVIEW].rstrip() + "\n… _(see attached .txt)_"
        else:
            snippet = text
        await _safe_send(
            ctx.application.bot, chat_id,
            header + "\n\n<pre>" + _html_escape_for_tg(snippet) + "</pre>",
            parse_mode=ParseMode.HTML,
        )
        # Attach the full text as a .txt file. Prefer the in-memory bytes
        # (always present, no disk race with the tempdir cleanup). The
        # legacy transcript_path is kept as a fallback if bytes are
        # somehow empty.
        fname = r.suggested_filename or "transcript.txt"
        import io
        payload: io.BytesIO | None = None
        if r.transcript_bytes:
            payload = io.BytesIO(r.transcript_bytes)
        elif r.transcript_path and Path(r.transcript_path).exists():
            payload = Path(r.transcript_path).open("rb")
        if payload is not None:
            cap = f"{t_title} — captions"
            await ctx.application.bot.send_document(
                chat_id=chat_id,
                document=InputFile(payload, filename=fname),
                caption=cap[:1024],
            )
        # Register the vault copy in the library.
        lib = _get_library(ctx)
        if lib is not None and r.transcript_path:
            try:
                tp = Path(r.transcript_path)
                await lib.add_asset(
                    kind="transcript", platform="youtube", source_url=url,
                    title=t_title, path=r.transcript_path,
                    size_bytes=tp.stat().st_size if tp.exists() else 0,
                    duration_s=float(r.duration) if r.duration else None,
                    meta={"backend": "captions", "language": r.language,
                          "format": fmt})
            except Exception:
                logger.exception("transcript asset registration failed")
        delivered += 1

    await update.message.reply_text(
        f"✅ Delivered {delivered}/{len(picked)} transcript(s)."
    )


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                          ns: str, action: str, run_id: str) -> None:
    """Media buttons, routed here by ``app.on_callback``: ``m:*`` (the
    per-chat result pool; ``run_id`` is the 1-based pool index) and
    ``t:send`` (``run_id`` is the transcript asset id)."""
    q = update.callback_query
    if ns == "m":
        chat_id = q.message.chat.id if q.message else update.effective_chat.id
        pool = _pool_get(f"tg:{chat_id}")
        try:
            idx = int(run_id)
        except ValueError:
            idx = 0
        if not pool or idx < 1 or idx > len(pool):
            await ctx.application.bot.send_message(
                chat_id=chat_id,
                text="(result pool expired — run /find or paste the "
                     "link again)")
            return
        item = pool[idx - 1]
        if action in ("dl", "mp3", "both"):
            # The video card asks for the format and quality, then shows the download.
            await open_card(ctx.application, chat_id, item["url"], fmt="a" if action == "mp3" else "v",
                            title=item.get("title") or "")
        if action in ("tr", "both"):
            ctx.application.create_task(
                _transcript_for_item(ctx.application, chat_id, item))
            await ctx.application.bot.send_message(
                chat_id=chat_id,
                text=f"⏬ Queued transcript for #{idx}: "
                     f"{(item.get('title') or item['url'])[:80]}")
        return

    if ns == "t" and action == "send":
        chat_id = q.message.chat.id if q.message else update.effective_chat.id
        lib = _get_library(ctx)
        if lib is None:
            await ctx.application.bot.send_message(
                chat_id=chat_id, text="⚠️ Library unavailable.")
            return
        try:
            rows = await lib.get_assets([int(run_id)])
        except Exception:
            logger.exception("t:send lookup failed")
            rows = []
        if not rows or not Path(rows[0]["path"]).exists():
            await ctx.application.bot.send_message(
                chat_id=chat_id,
                text="(transcript missing on disk — was it deleted?)")
            return
        a = rows[0]
        p = Path(a["path"])
        with p.open("rb") as f:
            await ctx.application.bot.send_document(
                chat_id=chat_id, document=InputFile(f, filename=p.name),
                caption=(a.get("title") or p.name)[:1024])
        return


def register(app: Application) -> None:
    app.add_handler(CommandHandler("video", video_cmd))
    app.add_handler(CommandHandler("find", find_cmd))
    app.add_handler(CommandHandler("fetch", fetch_cmd))
    app.add_handler(CommandHandler("quality", quality_cmd))
    app.add_handler(CommandHandler("transcript", transcript_cmd))
    app.add_handler(CommandHandler("transcripts", transcripts_cmd))
    app.add_handler(CommandHandler("media", media_cmd))
    app.add_handler(CommandHandler("rename", rename_cmd))
    app.add_handler(CommandHandler("widgets", widgets_cmd))
    app.add_handler(cards.WidgetQueryHandler(handle_widget_query))
    app.add_handler(MessageHandler(filters.VIDEO | filters.AUDIO | filters.VOICE | filters.VIDEO_NOTE
                                   | filters.Document.ALL, handle_file))
