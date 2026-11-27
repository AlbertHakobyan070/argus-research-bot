"""/read — read a page or a site on the spot and answer from it; and the
buttons a pasted article or repository link gets.

- ``/read <url> [<url>…] [question]``: up to ``anchors.ANCHOR_MAX`` links, read
  through the free fetch chain (``FREE_FETCH_CHAIN``: native → scrapling → jina →
  browser → intel_crawl, each only when this install has it) — never Exa, no
  budget. A GitHub repository or a site's front page also reads a few of its
  parts (``anchors.read_anchors``), unless the question keeps it on the page
  ("only the README", "just this page").
- The text read goes to the ``read.answer`` role: the answer to the question
  (a summary when there is none), from that text only, cited ``[n]``. It comes as
  a rich message (a Markdown message when Telegram refuses it, said once), and
  the text read follows as a ``.md`` file. No model answering is said, and the
  file still comes.
- A link that could not be read, or came back thin (``THIN_CHARS``), offers
  🌐 *Try with the browser* when the browser rung is installed: the ``browser``
  rung alone then reads it. Every answer offers 🔎 *Research this*, which
  starts ``/research`` on the same request.
- A pasted link that is not media gets 📖 Read and 🔎 Research buttons.
- One read per chat at a time; a read never waits for, or blocks, a research
  run (it runs beside the run driver, like the media tags).
"""
from __future__ import annotations

import asyncio
import logging
import re
import secrets
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes

from .. import llm
from ..config import get_settings
from ..engines import fetchers, registry
from ..graph import anchors
from ..graph.run_config import FREE_FETCH_CHAIN, FetchConfig
from ..graph.state import DEFAULT_LENGTH
from ..media import detect_platform, extract_urls
from . import rich, runs, tgx
from .common import _allowed, _md_escape, _safe_send, _send_md_doc

logger = logging.getLogger("argus.bot")

ROLE = "read.answer"
TEXT_BUDGET = 48_000           # page text the model reads, in characters (the links first, then their parts)
THIN_CHARS = 1_500             # less text read than this offers the browser
POOL_TTL_S = 1800              # a button's request is kept this long
NS = "rd"                      # callback namespace: rd:read|res|brw:<key>
USAGE = ("Usage: /read <link> [question]\n"
         "Reads the page (a repository or a site's front page: a few of its parts too) "
         "and answers from it — or summarises it when there is no question. No Exa.")

SYSTEM = """You read web pages for the user and answer from them.
- Use ONLY the page text below. When it does not answer the question, say so plainly, then say what the pages do cover.
- Cite the page each statement comes from as [n], the page's number below.
- Keep numbers, names, versions and dates exactly as the pages give them.
- No question: summarise. What it is, the key points, notable facts or numbers, and anything a reader should check (a date, a version, a claim).
- Markdown, at most about 350 words. No preamble and no closing offer."""

_pool: dict[str, dict] = {}    # key → {"chat_id", "request", "ts"}
_reading: set[int] = set()     # chats with a read in progress


def _pool_put(chat_id: int, request: str) -> str:
    """A random key: a button sent before a restart must not match a new request."""
    now = time.time()
    for key in [k for k, v in _pool.items() if now - v["ts"] > POOL_TTL_S]:
        del _pool[key]
    key = secrets.token_hex(4)
    while key in _pool:
        key = secrets.token_hex(4)
    _pool[key] = {"chat_id": chat_id, "request": request, "ts": now}
    return key


def _pool_get(chat_id: int, key: str) -> str | None:
    entry = _pool.get(key)
    if entry is None or entry["chat_id"] != chat_id or time.time() - entry["ts"] > POOL_TTL_S:
        return None
    return entry["request"]


def question_of(request: str) -> str:
    """The request without its links: what the user asks about them ("" for none)."""
    return " ".join(anchors._LINK.sub(" ", request or "").split())


def _keyboard(key: str, *, browser: bool) -> InlineKeyboardMarkup:
    row = [InlineKeyboardButton("🔎 Research this", callback_data=f"{NS}:res:{key}")]
    if browser:
        row.insert(0, InlineKeyboardButton("🌐 Try with the browser", callback_data=f"{NS}:brw:{key}"))
    return InlineKeyboardMarkup([row])


def _material(sources: list[dict]) -> str:
    """The numbered page texts the model reads, within ``TEXT_BUDGET``."""
    parts, left = [], TEXT_BUDGET
    for i, s in enumerate(sources, 1):
        if left <= 0:
            break
        text = (s.get("text") or "").strip()[:left]
        left -= len(text)
        parts.append(f"[{i}] {s.get('title') or s['url']}\nURL: {s['url']}\n\n{text}")
    return "\n\n-----\n\n".join(parts)


def _check(text: str) -> str:
    if not text.strip():
        raise ValueError("empty answer")
    return text


def _slug(text: str) -> str:
    return re.sub(r"[^\w.-]+", "-", text).strip("-")[:60] or "page"


def _text_file(sources: list[dict], key: str) -> Path:
    """Every page read, numbered as the answer cites them, in one .md file."""
    folder = Path(tempfile.gettempdir()) / "argus_read"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_slug(sources[0].get('title') or sources[0]['url'])}-{key}.md"
    path.write_text("\n\n".join(f"# [{i}] {s.get('title') or s['url']}\n\n> {s['url']}\n\n{(s.get('text') or '').strip()}"
                                for i, s in enumerate(sources, 1)), encoding="utf-8")
    return path


def _via(found: list[dict]) -> str:
    """"read via scrapling, jina" — the rungs that read the links and their parts."""
    rungs = [p.get("fetcher") for a in found for p in [a, *(a.get("pages") or [])] if p.get("ok") and p.get("fetcher")]
    return ", ".join(dict.fromkeys(rungs))


async def _send_answer(bot, chat_id: int, md: str, keyboard: InlineKeyboardMarkup) -> None:
    """A rich message; too big or refused (said once), the Markdown message."""
    body = rich.answer_html(md)
    if body is not None:
        try:
            await tgx.send_rich(bot, chat_id, body, reply_markup=keyboard)
            return
        except TelegramError as e:
            logger.warning("read: the rich answer could not be sent: %s: %s", type(e).__name__, e)
            if tgx.refused(e):
                await tgx.notice_once(bot, chat_id, "rich-answer", "ℹ️ Telegram refused the rich answer message, "
                                      "so the answer comes as text.")
    await _safe_send(bot, chat_id, md, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)


async def read_request(bot, chat_id: int, request: str, *, browser: bool = False) -> None:
    """Read the links in ``request`` and answer from them (see the module doc);
    ``browser``: the browser rung alone."""
    if chat_id in _reading:
        await _safe_send(bot, chat_id, "📖 Still reading the last link — one at a time.", parse_mode=None)
        return
    if browser and not fetchers.BROWSER_AVAILABLE:
        await _safe_send(bot, chat_id, f"⚠️ The browser fetcher is not installed — {fetchers.INSTALL['web']}, "
                         f"then {fetchers.INSTALL['chromium']}, and restart the bot.", parse_mode=None)
        return
    links = anchors.extract_links(request)
    if not links:
        await _safe_send(bot, chat_id, USAGE, parse_mode=None)
        return
    _reading.add(chat_id)
    try:
        await bot.send_message(chat_id=chat_id, disable_web_page_preview=True,
                               text=f"📖 Reading {', '.join(links)}" + (" in the browser…" if browser else "…"))
        cfg = FetchConfig(chain=["browser"] if browser else list(FREE_FETCH_CHAIN))
        found, sources, errors = await asyncio.to_thread(anchors.read_anchors, request,
                                                         rc=SimpleNamespace(fetch=cfg), ledger=None)
        for line in errors:
            logger.info("read: %s", line)
        key = _pool_put(chat_id, request)
        chars = sum(len(s.get("text") or "") for s in sources)
        offer_browser = (fetchers.BROWSER_AVAILABLE and not browser and chars < THIN_CHARS
                         and "browser" not in _via(found))
        if not sources:
            why = "; ".join(f"{a['url']}: {a['error']}" for a in found if not a.get("ok")) or "nothing to read"
            await _safe_send(bot, chat_id, f"⚠️ Could not read it — {why}", parse_mode=None,
                             reply_markup=_keyboard(key, browser=True) if offer_browser else None)
            return
        question = question_of(request)
        engines = await registry.active_snapshot()
        prompt = (f"QUESTION: {question}" if question else "No question: summarise these pages.") + \
            "\n\nPAGES:\n\n" + _material(sources)
        title = found[0].get("title") or links[0]
        # Markdown as the /ask answer's (rich.answer_html); a refused rich message falls back to it as is
        head = f"📖 **{_md_escape(title)}**\n_read via {_via(found) or 'the fetch chain'} · {len(sources)} page(s)_\n\n"
        cited = "\n\n**Pages**\n" + "\n".join(f"[{i}] {_md_escape(s.get('title') or s['url'])} — {s['url']}"
                                              for i, s in enumerate(sources, 1))
        try:
            res = await asyncio.to_thread(llm.complete, ROLE, [SystemMessage(content=SYSTEM),
                                                               HumanMessage(content=prompt)],
                                          engines=engines, temperature=0.2, max_tokens=1200, validate=_check)
            await _send_answer(bot, chat_id, head + res.text.strip() + cited,
                               _keyboard(key, browser=offer_browser))
        except llm.EngineError as e:
            logger.warning("read: no model answered: %s", e)
            await _safe_send(bot, chat_id, f"⚠️ The page was read, but no model answered ({e}). "
                             "The text read follows.", parse_mode=None, reply_markup=_keyboard(key, browser=offer_browser))
        await _send_md_doc(bot, chat_id, _text_file(sources, key), caption=f"Text read from {title}"[:1024])
    finally:
        _reading.discard(chat_id)


async def read_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not _allowed(get_settings(), update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    request = " ".join(ctx.args or [])
    if not anchors.extract_links(request):
        await update.message.reply_text(USAGE)
        return
    # a task: updates are handled one at a time, and a read takes seconds to a minute
    ctx.application.create_task(read_request(ctx.application.bot, update.effective_chat.id, request), update=update)


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str) -> None:
    """The reading branch of ``app.on_text``: links that are not media (the
    media branch has those) get 📖 Read and 🔎 Research buttons, each for that
    link and the rest of the message as its question."""
    links = [u for u in extract_urls(text) if not detect_platform(u)][:anchors.ANCHOR_MAX]
    if not links:
        return
    question = question_of(text)
    rows, lines = [], [f"🔗 {len(links)} link(s) — read it here, or research it:"]
    for i, url in enumerate(links, 1):
        key = _pool_put(chat_id, f"{url} {question}".strip())
        lines.append(f"{i}. {url}")
        rows.append([InlineKeyboardButton(f"📖 Read {i}" if len(links) > 1 else "📖 Read",
                                          callback_data=f"{NS}:read:{key}"),
                     InlineKeyboardButton(f"🔎 Research {i}" if len(links) > 1 else "🔎 Research",
                                          callback_data=f"{NS}:res:{key}")])
    await update.message.reply_text("\n".join(lines), reply_markup=InlineKeyboardMarkup(rows),
                                    disable_web_page_preview=True)


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE, action: str, key: str) -> None:
    """``rd:read|brw|res:<key>`` — read, read with the browser, or research the pooled request."""
    chat_id = update.effective_chat.id
    request = _pool_get(chat_id, key)
    bot = ctx.application.bot
    if request is None:
        await _safe_send(bot, chat_id, "This button has expired — send /read <link> again.", parse_mode=None)
        return
    if action in ("read", "brw"):
        ctx.application.create_task(read_request(bot, chat_id, request, browser=action == "brw"), update=update)
    elif action == "res":
        out = await runs.start_run(chat_id=chat_id, user_id=update.effective_user.id, topic=request,
                                   length=DEFAULT_LENGTH)
        if out.text:
            await _safe_send(bot, chat_id, out.text)
    else:
        raise ValueError(f"unknown read action {action!r}")


def register(app: Application) -> None:
    app.add_handler(CommandHandler("read", read_cmd))


__all__ = ["read_cmd", "read_request", "handle_text", "handle_callback", "register", "question_of", "ROLE"]
