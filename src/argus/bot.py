"""Argus Telegram bot layer.

python-telegram-bot v22 async, long-polling. Commands:
  /research <topic>              — full deep loop with both HITL gates
  /research /length <m> <topic>  — pre-set output length mode (T7)
  /ask <question>                — quick grounded answer
  /status                        — show in-flight run for this chat
  /cancel                        — drop in-flight run
  /help                          — usage

The bot is single-user (TELEGRAM_ALLOWED_USER_ID) for safety. It manages
the SqliteSaver lifecycle inside the running event loop and exposes
resume() so inline-keyboard callbacks can continue a paused LangGraph
run via Command(resume=...).

T7 additions
------------
- 5-button length selector at plan approval (TLDR / Short / Medium /
  Long / Lecture). The chosen mode is pushed back into the graph
  checkpoint via ``graph.aupdate_state`` before ``Command(resume=)``,
  so the synthesizer + report_builder see the user's HITL choice.
- /length flag in /research for users who want to skip the keyboard.
- Validation summary on title page + per-section confidence in the
  markdown + redesigned PDF.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

import re
import shutil
import time
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from telegram import (
    InputFile,
    InlineKeyboardButton, InlineKeyboardMarkup, Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes,
)

from .config import Settings, get_settings
from .cache_cleanup import cleanup_argus_ytt_cache
from .graph import build_graph, quick_answer_graph
from .graph.state import DEFAULT_LENGTH, VALID_LENGTHS, Length

logger = logging.getLogger("argus.bot")

HELP_TEXT = """\
*Argus — multi-agent Telegram research bot* (brain: FreeLLMAPI proxy)

Commands:
  /research <topic>                 — full deep loop with plan-approval + report-preview HITL gates
  /research /length <m> <topic>     — pre-pin output length mode (see below)
  /ask <question>                   — quick grounded single-shot answer
  /video [shorts] <query>           — search YouTube (videos or shorts)
  /transcript <indices>             — fetch captions for picked videos
                                       from the last /video results
                                       (e.g. 2,4 or `all` or 1-3)
  /status                           — show the latest report paths for this chat
  /cancel                           — drop in-flight runs for this chat
  /help                             — this message

*Length modes* (selectable at plan approval):
  • `tldr`     — single short paragraph
  • `short`    — current report (~300-700 chars)
  • `medium`   — 2-3 page MD (~3-6k chars, sub-headings)
  • `long`     — 5-8 page MD (~10-15k chars, 10-15 findings)
  • `lecture`  — 8-10 page MD, lecture format with Part I-IV + References + Appendix

Without `/length`, the default is `short`.

Inline-keyboard buttons appear after the plan is drafted and after the
report is built. Click *Approve* / *Send* to advance, *Cancel* to drop.

Per-chat memory is checkpointed in SQLite; reports are persisted to
`A:\\Hermes\\Downloads\\reports\\`.
"""

# ---------------------------------------------------------------------------
# Length selector (T7.1) — HITL keyboard button + /length CLI flag.
# ---------------------------------------------------------------------------

# Human-readable labels for the 5 modes, ordered shortest → longest.
_LENGTH_LABELS: dict[Length, str] = {
    "tldr":    "TL;DR",
    "short":   "Short",
    "medium":  "Medium",
    "long":    "Long",
    "lecture": "Lecture",
}


def _parse_length(args: list[str]) -> tuple[str, list[str]]:
    """Recognise ``/length <mode>`` at the head of an args list.

    Returns ``(mode, remaining_args)``. If no flag is present (or the
    flag is unrecognised) ``mode`` is ``""`` and the original ``args``
    list is returned untouched as the remaining-args. Recognised flag
    spellings: ``/length`` (Telegram-flavoured), ``--length``,
    ``-l``.
    """
    if not args:
        return "", []
    head = args[0].lower()
    if head not in ("/length", "--length", "-l"):
        return "", list(args)
    if len(args) < 2:
        return "", []  # malformed: /length with no mode at all
    mode = args[1].lower().strip()
    if mode not in _LENGTH_LABELS:
        # unrecognised mode — keep original args so caller can complain
        return "", list(args)
    return mode, list(args[2:])


# In-memory bookkeeping: which thread_id currently has an in-flight run.
# Persisted checkpoint state is the source of truth; this is just for
# progress messages and "is there a paused run?" checks.
_inflight: dict[str, dict[str, Any]] = {}

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


def _parse_indices(text: str) -> list[int]:
    """Parse a free-form user reply into 1-based list indices.

    Accepts: ``"2,4"``, ``"2 4"``, ``"2, 4"``, ``"all"``, ``"1-3"``.
    - Whitespace, trailing commas, and a leading ``/transcript`` are
      stripped upstream before this function is called.
    - Returns indices in the order they appear; ``all`` returns every
      index currently in the pool. Bad tokens are silently skipped.

    Returns an empty list for empty / unparseable input — callers
    decide whether to error."""
    s = (text or "").strip().lower()
    if not s:
        return []
    # Drop leading command if the user did ``/transcript 2,4`` mistakenly
    if s.startswith("/transcript"):
        s = s[len("/transcript"):].strip()
    if s in ("all", "*"):
        return []  # "" means caller should expand to "every index"
    out: list[int] = []
    for tok in re.split(r"[,\s]+", s):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            # Range like "1-3". Same number twice = single.
            try:
                a, b = tok.split("-", 1)
                a_i, b_i = int(a), int(b)
            except Exception:
                continue
            if a_i <= 0 or b_i <= 0:
                continue
            if a_i > b_i:
                a_i, b_i = b_i, a_i
            out.extend(range(a_i, b_i + 1))
            continue
        try:
            n = int(tok)
        except Exception:
            continue
        if n > 0:
            out.append(n)
    return out


def _allowed(settings: Settings, user_id: int | None) -> bool:
    if not settings.telegram_allowed_user_id:
        return False
    return user_id == settings.telegram_allowed_user_id


async def _send_md_doc(bot, chat_id: int, path: Path, caption: str = ""):
    if not path.exists():
        return
    with path.open("rb") as f:
        await bot.send_document(chat_id=chat_id, document=f,
                                filename=path.name, caption=caption[:1024])


async def _safe_send(bot, chat_id: int, text: str, *, reply_markup=None,
                     parse_mode=ParseMode.MARKDOWN):
    """send_message that never dies on a Telegram markdown parse error.

    Legacy-markdown mode chokes on raw dynamic content (file paths with ``_``,
    topics with ``*``/``[``, etc.) with "Can't parse entities". We try the
    requested parse_mode first, then retry as plain text so the message always
    goes out. Returns the sent Message (or None if even the plain send failed).
    """
    try:
        return await bot.send_message(chat_id=chat_id, text=text,
                                      reply_markup=reply_markup,
                                      parse_mode=parse_mode)
    except Exception as e:
        logger.warning("send (%s) failed (%s); retrying plain text",
                       parse_mode, e)
        try:
            return await bot.send_message(chat_id=chat_id, text=text,
                                          reply_markup=reply_markup)
        except Exception:
            logger.exception("plain-text send also failed")
            return None


async def _safe_edit_cb(q, text: str, *, reply_markup=None,
                        parse_mode=ParseMode.MARKDOWN):
    """callback_query.edit_message_text with the same markdown->plain fallback."""
    try:
        return await q.edit_message_text(text, reply_markup=reply_markup,
                                         parse_mode=parse_mode)
    except Exception as e:
        logger.warning("edit (%s) failed (%s); retrying plain text",
                       parse_mode, e)
        try:
            return await q.edit_message_text(text, reply_markup=reply_markup)
        except Exception:
            logger.exception("plain-text edit also failed")
            return None


async def _stream_progress(bot, chat_id: int, text: str, last_msg: dict):
    """Edit the previous progress message in place, or send a new one.

    Bulletproof against markdown parse errors: progress strings often embed
    report folder paths whose ``_`` underscores break legacy-markdown
    ("Can't parse entities" -> the old "resume failed" crash). We try
    markdown then plain for both the edit and the send.
    """
    mid = last_msg.get("message_id")
    if mid:
        for pm in (ParseMode.MARKDOWN, None):
            try:
                await bot.edit_message_text(chat_id=chat_id, message_id=mid,
                                            text=text, parse_mode=pm)
                return last_msg
            except Exception:
                continue  # try plain, then fall through to a fresh send
    for pm in (ParseMode.MARKDOWN, None):
        try:
            msg = await bot.send_message(chat_id=chat_id, text=text,
                                         parse_mode=pm)
            last_msg["message_id"] = msg.message_id
            return last_msg
        except Exception:
            continue
    logger.warning("progress update failed entirely for: %r", text[:80])
    return last_msg


# ---------------------------------------------------------------------------
# /research
# ---------------------------------------------------------------------------


def _html_escape_for_tg(s: str) -> str:
    """Escape ``&``, ``<``, ``>`` for Telegram's HTML parse mode.

    Used by the report-preview path (which was previously MARKDOWN mode and
    crashed on excerpts containing ``_``, ``*``, ``[``, or ``&`` with
    ``BadRequest: Can't parse entities``). Telegram's HTML mode is far more
    permissive: ``<i>``, ``<b>``, ``<code>``, ``<pre>`` are formatting; any
    other ``<`` would be a tag, so we escape everything.

    The output is safe to embed inside a Telegram HTML message. Newlines are
    preserved (Telegram renders ``\n`` as a real line break).
    """
    if not s:
        return ""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))
def _md_escape(s: str) -> str:
    """Escape characters that Telegram's legacy Markdown parser treats as
    formatting, so LLM-produced plan/report text doesn't crash the parser
    when the message is later edited (Bug C from t_b0665cfc).
    Only applies to DYNAMIC content; static formatting is left intact."""
    if not s:
        return ""
    # Order matters: escape backslash first to avoid double-escaping.
    out = s.replace("\\", "\\\\")
    for ch in ("_", "*", "`", "[", "]"):
        out = out.replace(ch, "\\" + ch)
    return out


def _format_plan(plan: dict, length: str = DEFAULT_LENGTH) -> str:
    summary = plan.get("summary", "")
    sub_qs = plan.get("sub_questions") or []
    sources = plan.get("planned_sources") or []
    mode_label = _LENGTH_LABELS.get(length, length)
    lines = [f"📋 *Research plan*  ·  mode: {mode_label}", ""]
    if summary:
        lines += [f"_{_md_escape(summary)}_", ""]
    if sub_qs:
        lines.append("*Sub-questions:*")
        for q in sub_qs:
            lines.append(f"- {_md_escape(q)}")
        lines.append("")
    if sources:
        lines.append(
            "*Planned sources "
            f"({len(sources)}, queries only — URLs are verified at fetch time):*"
        )
        for s in sources[:14]:
            kind = s.get("kind", "search")
            q = s.get("query")
            url = s.get("target_url")
            if q:
                intent = f"_search intent:_ {_md_escape(q)}"
            elif url:
                intent = f"_candidate (verified at fetch):_ {_md_escape(url)}"
            else:
                intent = "_live search_"
            lines.append(f"- `{_md_escape(kind)}` — {intent}")
    return "\n".join(lines)


def _plan_keyboard(default_length: str = DEFAULT_LENGTH) -> InlineKeyboardMarkup:
    """T7.1 — five-button length selector at plan approval.

    Button order is tldr→lecture (cheapest→deepest). The button matching
    ``default_length`` gets a check-mark in its label so the user can
    see what they'll get if they tap nothing besides Approve.
    """
    def label(mode: str) -> str:
        return (_LENGTH_LABELS[mode]
                + (" ✅" if mode == default_length else ""))

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(label("tldr"), callback_data="len:tldr"),
            InlineKeyboardButton(label("short"), callback_data="len:short"),
            InlineKeyboardButton(label("medium"), callback_data="len:medium"),
        ],
        [
            InlineKeyboardButton(label("long"), callback_data="len:long"),
            InlineKeyboardButton(label("lecture"), callback_data="len:lecture"),
        ],
        [
            InlineKeyboardButton("✅ Approve", callback_data="plan:approve"),
            InlineKeyboardButton("✏️ Edit", callback_data="plan:edit"),
            InlineKeyboardButton("❌ Cancel", callback_data="plan:cancel"),
        ],
    ])


def _report_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📤 Send", callback_data="report:send"),
            InlineKeyboardButton("🔎 Extend", callback_data="report:extend"),
        ],
        [
            InlineKeyboardButton("🔁 Revise", callback_data="report:revise"),
            InlineKeyboardButton("❌ Cancel", callback_data="report:cancel"),
        ],
    ])


async def research_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    if not ctx.args:
        await update.message.reply_text(
            "Usage: /research [/length <m>] <topic>\n"
            "Lengths: " + " / ".join(VALID_LENGTHS)
        )
        return
    # T7.1: allow ``/research /length <m> <topic>`` so the user can
    # pre-pin a length without waiting for the keyboard. Empty /length
    # is silently ignored (caller falls back to DEFAULT_LENGTH).
    length, topic_args = _parse_length(list(ctx.args))
    if not length:
        length = DEFAULT_LENGTH
        topic_args = list(ctx.args)
    topic = " ".join(topic_args).strip()
    if not topic:
        await update.message.reply_text(
            "Usage: /research [/length <m>] <topic>"
        )
        return
    chat_id = update.effective_chat.id
    thread_id = f"tg:{chat_id}"

    progress = {"message_id": None}
    progress = await _stream_progress(ctx.application.bot, chat_id,
        "🧠 *Starting research…*", progress)
    await ctx.application.bot.send_chat_action(chat_id, ChatAction.TYPING)

    # Build a per-run saver. We must keep it alive across the HITL pause
    # because the in-memory graph holds a reference to it as checkpointer.
    # The saver is closed only when the run actually terminates
    # (Send in _resume_after_report, Cancel in on_callback, /cancel_cmd,
    # planner-abort in the `else` branch below, or any exception).
    saver_cm = AsyncSqliteSaver.from_conn_string(str(s.checkpoint_db))
    saver = await saver_cm.__aenter__()
    keep_saver_alive = False
    try:
        graph = build_graph(checkpointer=saver)
        cfg = {"configurable": {"thread_id": thread_id}}
        state_in: dict[str, Any] = {
            "thread_id": thread_id,
            "user_id": update.effective_user.id,
            "user_request": topic,
            "length": length,           # T7 — Length selector carry
            "messages": [], "plan": None,
            "sources": [], "fetched": [], "findings": [],
            "draft_md": "", "revision_notes": [], "revision_rounds": 0,
            "model_calls": [], "hitl": {"pending": False},
        }
        _inflight[thread_id] = {
            "state": state_in, "stage": "intake", "length": length,
        }

        # Stream until first interrupt.
        async for ev in graph.astream(state_in, config=cfg,
                                      stream_mode="updates"):
            for node_name, delta in ev.items():
                _inflight[thread_id]["stage"] = node_name
                if node_name == "intake":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🎯 Mode detected: _{delta.get('mode','?')}_", progress)
                elif node_name == "planner":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        "📋 Plan drafted — awaiting your approval.", progress)
                elif node_name == "researcher":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        "🔍 Researching primary sources…", progress)
                elif node_name == "fetcher":
                    n = len(delta.get("fetched") or [])
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"📥 Fetched {n} items.", progress)
                elif node_name == "synthesizer":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🧠 Synthesizing ({_LENGTH_LABELS.get(length,'?')})…",
                        progress)
                elif node_name == "reviewer":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🔬 Reviewer: {delta.get('review_verdict',{}).get('verdict','?')}",
                        progress)

        # Snapshot current state from the checkpointer.
        snap = await graph.aget_state(cfg)
        cur = snap.values if snap else {}

        # First HITL = plan approval.
        plan = cur.get("plan") or {}
        if plan:
            text = _format_plan(plan, length=length)
            # _safe_send falls back to plain text if the plan body (topic /
            # sub-questions / URLs) contains markdown-special chars — so the
            # plan proposal ALWAYS renders instead of silently failing to send.
            await _safe_send(
                ctx.application.bot, chat_id, text,
                reply_markup=_plan_keyboard(default_length=length),
                parse_mode=ParseMode.MARKDOWN,
            )
            # Hand the saver to the in-flight registry so _resume_after_plan
            # can use the same live checkpointer. Do NOT close here — that's
            # what produced the "resume failed: no active connection" error.
            keep_saver_alive = True
            _inflight[thread_id]["awaiting"] = "plan_approval"
            _inflight[thread_id]["cfg"] = cfg
            _inflight[thread_id]["graph"] = graph
            _inflight[thread_id]["saver_cm"] = saver_cm
            _inflight[thread_id]["saver"] = saver
        else:
            await ctx.application.bot.send_message(
                chat_id=chat_id,
                text="(planner produced no plan; aborting.)",
            )
            _inflight.pop(thread_id, None)
    except Exception:
        # On any error, drop the in-flight entry and let `finally` close the saver.
        _inflight.pop(thread_id, None)
        raise
    finally:
        if not keep_saver_alive:
            await saver_cm.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# /ask  (quick path)
# ---------------------------------------------------------------------------

async def ask_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    if not ctx.args:
        await update.message.reply_text("Usage: /ask <question>")
        return
    question = " ".join(ctx.args)
    chat_id = update.effective_chat.id
    thread_id = f"tg:{chat_id}"
    await ctx.application.bot.send_chat_action(chat_id, ChatAction.TYPING)

    saver_cm = AsyncSqliteSaver.from_conn_string(str(s.checkpoint_db))
    saver = await saver_cm.__aenter__()
    try:
        graph = quick_answer_graph(checkpointer=saver)
        cfg = {"configurable": {"thread_id": thread_id}}
        state_in: dict[str, Any] = {
            "thread_id": thread_id,
            "user_id": update.effective_user.id,
            "user_request": question,
            "length": DEFAULT_LENGTH,
            "messages": [], "plan": None,
            "sources": [], "fetched": [], "findings": [],
            "draft_md": "", "revision_notes": [], "revision_rounds": 0,
            "model_calls": [], "hitl": {"pending": False},
        }
        answer_text = ""
        async for ev in graph.astream(state_in, config=cfg,
                                      stream_mode="updates"):
            for node, delta in ev.items():
                if node == "intake":
                    pass
                elif node == "quick_answer":
                    answer_text = delta.get("quick_answer") or answer_text
                elif node == "deliver":
                    pass
        # LLM answers can contain unbalanced markdown; fall back to plain.
        try:
            await update.message.reply_text(
                answer_text or "(no answer)", parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(answer_text or "(no answer)")
    finally:
        await saver_cm.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# /video — YouTube search (Phase 2)
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: Any) -> str:
    """Seconds -> H:MM:SS or M:SS. Empty string for missing/invalid."""
    try:
        s = int(seconds)
    except (TypeError, ValueError):
        return ""
    if s <= 0:
        return ""
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


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

    from .tools import youtube_search
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
    thread_id = f"tg:{update.effective_chat.id}"
    _pool_put(thread_id, results)
    logger.info("/video pooled %d result(s) for %s (query=%r)",
                len(results), thread_id, query)


# ---------------------------------------------------------------------------
# /transcript — fetch captions for videos from the last /video pool.
# ---------------------------------------------------------------------------


async def transcript_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/transcript <indices|all> — fetch YouTube auto-captions for the
    picked videos from the last /video result pool.

    Index syntax: ``2``, ``2,4``, ``2 4``, ``1-3``, ``all``, ``*``.
    Each fetched transcript is delivered as:
      - a short Telegram message with the title + a preview snippet, then
      - a ``.txt`` file attachment carrying the FULL plain text.
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
            "Usage: /transcript <indices>  (e.g. /transcript 2,4 or /transcript all)"
        )
        return

    indices = _parse_indices(raw)
    n_total = len(pool)
    # Special-case ``all`` / empty-list-means-all: caller passed "" sentinel.
    if not indices:
        if raw.strip().lower() in ("all", "*"):
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

    from .tools import youtube_video_transcript

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
        try:
            r = await asyncio.to_thread(
                youtube_video_transcript, url, timeout=90)
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
        delivered += 1

    await update.message.reply_text(
        f"✅ Delivered {delivered}/{len(picked)} transcript(s)."
    )


# ---------------------------------------------------------------------------
# /status, /cancel, /help
# ---------------------------------------------------------------------------

async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    chat_id = update.effective_chat.id
    thread_id = f"tg:{chat_id}"
    info = _inflight.get(thread_id)
    if not info:
        await update.message.reply_text("No in-flight run for this chat.")
        return
    snap_lines = [
        f"Stage: `{info.get('stage','?')}`",
        f"Awaiting: `{info.get('awaiting','-')}`",
        f"Length: `{info.get('length','?')}`",
    ]
    try:
        await update.message.reply_text("\n".join(snap_lines),
                                        parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text("\n".join(snap_lines))


async def cancel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    chat_id = update.effective_chat.id
    thread_id = f"tg:{chat_id}"
    info = _inflight.pop(thread_id, None)
    # Close any held-over saver so we don't leak the connection.
    if info and info.get("saver_cm"):
        try:
            await info["saver_cm"].__aexit__(None, None, None)
        except Exception:
            pass
    await update.message.reply_text("In-flight run dropped from memory.")


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# Inline-keyboard callbacks
# ---------------------------------------------------------------------------

async def on_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.callback_query.answer("Unauthorized.", show_alert=True)
        return
    q = update.callback_query
    await q.answer()
    chat_id = q.message.chat.id if q.message else update.effective_chat.id
    thread_id = f"tg:{chat_id}"
    info = _inflight.get(thread_id)
    if not info:
        await q.edit_message_text("(no in-flight run)")
        return
    data = q.data or ""
    # T7.1 — length selector clicks act as "approve with this length".
    if data.startswith("len:"):
        chosen = data.split(":", 1)[1]
        if chosen not in _LENGTH_LABELS:
            await q.edit_message_text("Unknown length mode.")
            return
        info["length"] = chosen
        info["plan_approved"] = True
        await _safe_edit_cb(
            q,
            (q.message.text or "")
            + f"\n\n_📏 Length set to *{_LENGTH_LABELS.get(chosen, chosen)}* "
              "— continuing._",
        )
        await _resume_after_plan(ctx, thread_id, info)
        return
    if data == "plan:approve":
        info["plan_approved"] = True
        await _safe_edit_cb(
            q, (q.message.text or "") + "\n\n_✅ Approved — continuing._")
        await _resume_after_plan(ctx, thread_id, info)
    elif data == "plan:edit":
        info["plan_approved"] = "edit"
        info["plan_edit"] = True
        await _safe_edit_cb(
            q,
            (q.message.text or "") + "\n\n_✏️ Edit mode — reply with your changes._")
    elif data == "plan:cancel":
        await _drop_inflight_with_saver(thread_id)
        await q.edit_message_text("❌ Cancelled.")
    elif data == "report:send":
        await _resume_after_report(ctx, thread_id, info, q)
    elif data == "report:extend":
        # Phase 2 HITL: deepen the research. Set the graph flag, then reuse
        # the plan-resume streamer — it drives deliver → extend_prep →
        # fetcher → … → report_builder and shows the next preview.
        graph = info["graph"]
        cfg = info["cfg"]
        try:
            await graph.aupdate_state(cfg, {"extend_requested": True})
        except Exception:
            logger.exception("could not set extend_requested; aborting extend")
            await _safe_edit_cb(
                q, (q.message.text or "") + "\n\n_⚠️ Extend failed to start._")
            return
        await _safe_edit_cb(
            q,
            (q.message.text or "") + "\n\n_🔎 Extending research — gathering more…_")
        await _resume_after_plan(ctx, thread_id, info)
    elif data == "report:revise":
        info["report_revise"] = True
        await _safe_edit_cb(
            q, (q.message.text or "") + "\n\n_🔁 Revision requested._")
        await _resume_after_report(ctx, thread_id, info, q,
                                    revision_requested=True)
    elif data == "report:cancel":
        await _drop_inflight_with_saver(thread_id)
        await q.edit_message_text("❌ Cancelled (report dropped).")


async def _resume_after_plan(ctx: ContextTypes.DEFAULT_TYPE,
                              thread_id: str, info: dict):
    """Resume the graph from the 'researcher' interrupt."""
    graph = info["graph"]
    cfg = info["cfg"]
    chat_id = int(thread_id.split(":", 1)[1])
    progress = {"message_id": None}

    # T7.1 — push the chosen length back into the graph state BEFORE
    # resuming, so the synthesizer + report_builder see the user's
    # HITL choice rather than the default. ``aupdate_state`` is the
    # canonical way to fork a thread checkpoint from outside.
    chosen_length = (info.get("length")
                     or info.get("state", {}).get("length")
                     or DEFAULT_LENGTH)
    try:
        await graph.aupdate_state(cfg, {"length": chosen_length})
    except AttributeError:
        # Compiled graph without async API — fall back to sync.
        try:
            graph.update_state(cfg, {"length": chosen_length})
        except Exception:
            logger.exception(
                "could not push length=%s into state; defaulting",
                chosen_length,
            )
    except Exception:
        logger.exception(
            "aupdate_state failed; defaulting length=%s", chosen_length,
        )

    progress = await _stream_progress(ctx.application.bot, chat_id,
        "🔍 Researching primary sources…", progress)

    try:
        async for ev in graph.astream(Command(resume=True), config=cfg,
                                      stream_mode="updates"):
            for node_name, delta in ev.items():
                info["stage"] = node_name
                if node_name == "researcher":
                    n = len(delta.get("sources") or [])
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🔍 {n} candidate sources.", progress)
                elif node_name == "fetcher":
                    n = len(delta.get("fetched") or [])
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"📥 {n} fetched + normalized.", progress)
                elif node_name == "synthesizer":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🧠 Synthesizing ({_LENGTH_LABELS.get(chosen_length,'?')})…",
                        progress)
                elif node_name == "reviewer":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🔬 Reviewer verdict: "
                        f"{delta.get('review_verdict',{}).get('verdict','?')}",
                        progress)
                elif node_name == "report_builder":
                    paths = delta.get("report_paths") or {}
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"📝 Report ready: {paths.get('folder','?')}", progress)
                    info["report_paths"] = paths
                    info["length"] = chosen_length

        snap = await graph.aget_state(cfg)
        cur = snap.values if snap else {}
        paths = cur.get("report_paths") or info.get("report_paths") or {}
        info["awaiting"] = "report_preview"
        info["paths"] = paths
        info["length"] = chosen_length
        if paths.get("md"):
            md = Path(paths["md"])
            text = (
                f"📝 *Report preview*  ·  {_LENGTH_LABELS.get(chosen_length,'?')}\n\n"
                f"Folder: `{_md_escape(paths.get('folder','?'))}`\n"
                f"Markdown: `{_md_escape(md.name)}`  "
                + (f"· PDF: `report.pdf`" if paths.get("pdf") else "· PDF: _failed_")
                + f"\n\n_(first 1200 chars below)_"
            )
            excerpt = ""
            try:
                excerpt = md.read_text(encoding="utf-8", errors="replace")[:1200]
            except Exception:
                pass

            # Build two candidate bodies: an HTML-escaped preview for the
            # primary send (ParseMode.HTML tolerates arbitrary content),
            # and a plain-text fallback (no parse_mode) for when the
            # excerpt is too large or Telegram chokes on the HTML.
            # The old code used ParseMode.MARKDOWN and crashed with
            # ``Can't parse entities`` whenever the excerpt contained
            # ``_``, ``*``, ``[``, or ``&`` (Bug observed 2026-07-08).
            html_body = _html_escape_for_tg(text) + (
                ("\n\n<pre>" + _html_escape_for_tg(excerpt) + "</pre>")
                if excerpt else ""
            )
            plain_body = text + ("\n\n```\n" + excerpt + "\n```" if excerpt else "")

            try:
                await ctx.application.bot.send_message(
                    chat_id=chat_id,
                    text=html_body,
                    reply_markup=_report_keyboard(),
                    parse_mode=ParseMode.HTML,
                )
            except Exception as send_err:
                # Telegram HTML parse still failed (oversize, exotic chars,
                # network blip). Degrade to a plain-text preview rather than
                # crashing the resume — the user can still see the path and
                # open the file from disk / folder link.
                logger.warning(
                    "report preview HTML send failed (%s); falling back to plain text",
                    send_err,
                )
                try:
                    await ctx.application.bot.send_message(
                        chat_id=chat_id,
                        text=plain_body,
                        reply_markup=_report_keyboard(),
                    )
                except Exception as plain_err:
                    # Last resort: surface the error rather than crash silently.
                    logger.exception("report preview plain-text send also failed")
                    await ctx.application.bot.send_message(
                        chat_id=chat_id,
                        text=f"⚠️ report preview unavailable: {plain_err}",
                    )
        else:
            await ctx.application.bot.send_message(
                chat_id=chat_id, text="(no report_paths found)")
    except Exception as e:
        logger.exception("resume failed")
        await ctx.application.bot.send_message(chat_id=chat_id,
            text=f"⚠️ resume failed: {e}")


async def _resume_after_report(ctx: ContextTypes.DEFAULT_TYPE,
                                thread_id: str, info: dict, q,
                                revision_requested: bool = False):
    graph = info["graph"]
    cfg = info["cfg"]
    chat_id = int(thread_id.split(":", 1)[1])
    paths = info.get("paths") or {}
    if revision_requested:
        # Force another revision round by tweaking revision_notes (simplest
        # way without rebuilding the graph: skip and ask user to retype).
        await ctx.application.bot.send_message(
            chat_id=chat_id,
            text=("Send your revision feedback as a reply — "
                  "it will be appended to the next revision round."),
        )
        # For simplicity we stop here: the user re-runs /research.
        await _drop_inflight_with_saver(thread_id)
        return
    # Send the report files.
    try:
        md_path = Path(paths["md"]) if paths.get("md") else None
        pdf_path = Path(paths["pdf"]) if paths.get("pdf") else None
        folder = paths.get("folder", "")
        cap = f"Argus report — folder: `{folder}`"
        if md_path and md_path.exists():
            await _send_md_doc(ctx.application.bot, chat_id, md_path, caption=cap)
        if pdf_path and pdf_path.exists():
            await _send_md_doc(ctx.application.bot, chat_id, pdf_path,
                                caption=cap)
        await ctx.application.bot.send_message(
            chat_id=chat_id, text="✅ Delivered.")
    except Exception as e:
        await ctx.application.bot.send_message(chat_id=chat_id,
            text=f"⚠️ send failed: {e}")
    finally:
        await _drop_inflight_with_saver(thread_id)


async def _drop_inflight_with_saver(thread_id: str) -> None:
    """Pop the in-flight entry and close its SqliteSaver if held.

    Helper for all terminal paths (Cancel button, normal end of a run,
    /cancel command) so the per-run saver connection is always released.
    Without this, every research run would leak one aiosqlite connection.
    """
    info = _inflight.pop(thread_id, None)
    if info and info.get("saver_cm"):
        try:
            await info["saver_cm"].__aexit__(None, None, None)
        except Exception:
            logger.exception("saver close on drop failed")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def _on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler.

    Without one, python-telegram-bot logs 'No error handlers are registered'
    for every polling hiccup (e.g. a 409 Conflict when a second bot instance
    is running). We log concisely; transient polling conflicts (409) are
    demoted to a warning since they resolve once the duplicate poller exits.
    """
    err = getattr(ctx, "error", None)
    from telegram.error import Conflict
    if isinstance(err, Conflict):
        logger.warning("Telegram 409 Conflict — another bot instance is "
                       "polling this token. Stop the duplicate (see run.sh).")
        return
    logger.error("Unhandled bot error: %r", err, exc_info=err)


def build_application() -> Application:
    s = get_settings()
    if not s.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing from .env")
    app = Application.builder().token(s.telegram_bot_token).build()
    app.add_handler(CommandHandler("research", research_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("video", video_cmd))
    app.add_handler(CommandHandler("transcript", transcript_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_error_handler(_on_error)
    return app


def main():
    logging.basicConfig(
        level=os.environ.get("ARGUS_LOG", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = build_application()
    logger.info("Argus bot starting; long-polling…")
    # Trim persistent /transcript cache. Best-effort, exception-tolerant.
    _removed = cleanup_argus_ytt_cache()
    if _removed:
        logger.info(
            "argus_ytt cache cleanup: removed %d stale deliverable(s)", _removed,
        )
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
