"""Argus Telegram bot layer.

python-telegram-bot v22 async, long-polling. Commands:
  /research <topic>              — full deep loop with both HITL gates
  /research /length <m> <topic>  — pre-set output length mode (T7)
  /ask <question>                — quick grounded answer
  /status                        — show in-flight run for this chat
  /cancel                        — drop in-flight run
  /help                          — usage

The bot is single-user (TELEGRAM_ALLOWED_USER_ID) for safety.

v2 architecture (Phase 1)
-------------------------
- ONE long-lived AsyncSqliteSaver + ONE compiled graph per flavour
  (deep/quick), opened in PTB ``post_init`` and stored on
  ``application.bot_data``. Runs are isolated by per-run thread ids
  ``tg:<chat>:<run8>`` — multiple concurrent runs per chat are legal,
  and any run can be re-attached after a bot restart.
- Every run is registered in the Library (SQLite) so /status lists
  history and later phases can /append + /continue + /delete.
- Callback data is run-scoped (``plan:approve:<run8>``) so buttons on
  old messages can never act on the wrong run.

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

from langgraph.types import Command
from telegram import (
    InputFile,
    InlineKeyboardButton, InlineKeyboardMarkup, Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, ContextTypes,
    MessageHandler, filters,
)

from .config import Settings, get_settings
from .cache_cleanup import cleanup_argus_ytt_cache
from .graph import async_sqlite_saver_cm, build_graph, quick_answer_graph
from .graph.state import DEFAULT_LENGTH, VALID_LENGTHS, Length
from .library import Library, mirror_run_md, new_run_id
from .media import (
    detect_platform, download_media, extract_urls, quality_format,
    reddit_search, resolve_ffmpeg,
)
from .transcribe import transcribe_url

logger = logging.getLogger("argus.bot")

# Compact quick-reference. The full, explained guide lives in
# ``docs/help.md`` and is sent by ``/help full``.
HELP_TEXT = """\
*Argus* — research + media bot. `/help full` for the detailed guide.

*Research*
  /research <topic> — deep report (plan → approve → preview)
  /ask <question> — quick one-shot answer

*Media → vault*
  /find [yt|shorts|reddit] <query> — search, then ⬇/📝 buttons
  /fetch <url…> — download links (or just paste them)
  /quality [auto|min|max|<h>] — download quality
  /transcript <indices> — captions for /find picks
  /transcripts — browse saved transcripts

*History*
  /runs — list runs · /status — what's in flight
  /append <id> <url|asset:N> — queue sources for a run
  /continue <id> — resume or extend a run
  /delete — free up space · /cancel [id] — stop a run

_Tip: paste any YouTube / X / Reddit / Instagram link for a download menu._
"""

# Loaded lazily from docs/help.md (repo) for `/help full`.
_HELP_MD_PATH = Path(__file__).resolve().parents[2] / "docs" / "help.md"

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


# In-memory bookkeeping of in-flight runs, keyed by run_id (8-hex).
# Persisted checkpoint state + the Library registry are the sources of
# truth; this holds live objects (graph cfg, progress message ids,
# chosen length, resuming flag) for runs the current process started.
_inflight: dict[str, dict[str, Any]] = {}


def _chat_entries(chat_id: int) -> dict[str, dict[str, Any]]:
    """All in-flight entries belonging to one chat, keyed by run_id."""
    return {rid: i for rid, i in _inflight.items()
            if i.get("chat_id") == chat_id}


def _bot_data_get(ctx: ContextTypes.DEFAULT_TYPE, key: str) -> Any:
    """Fetch a shared object placed on ``application.bot_data`` by
    ``_post_init``. Returns None when the app isn't fully wired (unit
    tests with bare mocks; a production hit means post_init didn't run)."""
    bd = getattr(ctx.application, "bot_data", None)
    if isinstance(bd, dict):
        return bd.get(key)
    return None


def _get_library(ctx: ContextTypes.DEFAULT_TYPE) -> Library | None:
    return _bot_data_get(ctx, "library")


async def _set_run_status(ctx: ContextTypes.DEFAULT_TYPE, run_id: str,
                          status: str, *, report_dir: str | None = None) -> None:
    """Record a run's status transition in the registry.

    Failures are logged loudly but never crash a Telegram handler —
    losing a status row is recoverable; dying mid-delivery is not.
    """
    lib = _get_library(ctx)
    if lib is None:
        logger.warning("library not configured; run %s -> %s not recorded",
                       run_id, status)
        return
    try:
        await lib.set_run_status(run_id, status, report_dir=report_dir)
    except Exception:
        logger.exception("registry status update failed (%s -> %s)",
                         run_id, status)

# Per-chat last /video result pool. Keyed by thread_id; entries hold
# (results list, timestamp). Capped to a TTL so we never leak if the
# user walks away mid-session. ``/transcript <indices>`` reads from here.
_VIDEO_POOL_TTL_S = 1800   # 30 minutes — long enough for a follow-up,
                            # short enough that a stale pool doesn't
                            # silently outlive its relevance.
_video_pool: dict[str, dict[str, Any]] = {}

# v2 Phase 6a — the next free-text message from a chat is interpreted as
# a reply to a HITL prompt: {chat_id: (kind, run_id)} where kind is
# "plan_edit" or "revise_feedback". Consumed once by on_text.
_pending_reply: dict[int, tuple[str, str]] = {}


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


def _format_plan(plan: dict, length: str = DEFAULT_LENGTH,
                 sources: list[dict] | None = None) -> str:
    """Render the plan-approval preview.

    v2 grounded gate: ``sources`` are the REAL results the researcher
    found via live search (ddgs/arxiv/github) before this gate. Planner
    ``target_url`` values are NEVER rendered — they are LLM guesses
    (knowledge-cutoff fantasy links) and the researcher ignores them.
    """
    summary = plan.get("summary", "")
    sub_qs = plan.get("sub_questions") or []
    intents = plan.get("planned_sources") or []
    mode_label = _LENGTH_LABELS.get(length, length)
    lines = [f"📋 *Research plan*  ·  mode: {mode_label}", ""]
    if summary:
        lines += [f"_{_md_escape(summary)}_", ""]
    if sub_qs:
        lines.append("*Sub-questions:*")
        for q in sub_qs:
            lines.append(f"- {_md_escape(q)}")
        lines.append("")
    if intents:
        lines.append(f"*Search intents ({len(intents)}):*")
        for s in intents[:14]:
            kind = s.get("kind", "search")
            q = s.get("query")
            intent = f"_{_md_escape(q)}_" if q else "_live search_"
            lines.append(f"- `{_md_escape(kind)}` — {intent}")
        lines.append("")
    if sources is not None:
        if sources:
            shown = sources[:12]
            lines.append(f"*Found sources ({len(sources)}, live search):*")
            for s in shown:
                title = (s.get("title") or "").strip()
                url = s.get("url") or ""
                if not title:
                    title = url.split("//", 1)[-1][:60] or "(untitled)"
                lines.append(f"- [{_md_escape(title[:70])}]({url})")
            if len(sources) > len(shown):
                lines.append(f"  _…and {len(sources) - len(shown)} more_")
        else:
            lines.append(
                "⚠️ *Live search found no sources.* Approving will fetch "
                "nothing — Edit the plan or Cancel.")
    return "\n".join(lines)


def _plan_keyboard(default_length: str = DEFAULT_LENGTH,
                   run_id: str = "") -> InlineKeyboardMarkup:
    """T7.1 — five-button length selector at plan approval.

    Button order is tldr→lecture (cheapest→deepest). The button matching
    ``default_length`` gets a check-mark in its label so the user can
    see what they'll get if they tap nothing besides Approve.

    v2: callback data is run-scoped (``len:long:<run8>``) so buttons on
    an old plan message can never act on a newer run in the same chat.
    """
    sfx = f":{run_id}" if run_id else ""

    def label(mode: str) -> str:
        return (_LENGTH_LABELS[mode]
                + (" ✅" if mode == default_length else ""))

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(label("tldr"), callback_data=f"len:tldr{sfx}"),
            InlineKeyboardButton(label("short"), callback_data=f"len:short{sfx}"),
            InlineKeyboardButton(label("medium"), callback_data=f"len:medium{sfx}"),
        ],
        [
            InlineKeyboardButton(label("long"), callback_data=f"len:long{sfx}"),
            InlineKeyboardButton(label("lecture"), callback_data=f"len:lecture{sfx}"),
        ],
        [
            InlineKeyboardButton("✅ Approve", callback_data=f"plan:approve{sfx}"),
            InlineKeyboardButton("✏️ Edit", callback_data=f"plan:edit{sfx}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"plan:cancel{sfx}"),
        ],
    ])


def _report_keyboard(run_id: str = "") -> InlineKeyboardMarkup:
    sfx = f":{run_id}" if run_id else ""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("📤 Send", callback_data=f"report:send{sfx}"),
            InlineKeyboardButton("🔎 Extend", callback_data=f"report:extend{sfx}"),
        ],
        [
            InlineKeyboardButton("🔁 Revise", callback_data=f"report:revise{sfx}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"report:cancel{sfx}"),
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
    graph = _bot_data_get(ctx, "graph")
    if graph is None:
        await update.message.reply_text(
            "⚠️ Bot not fully initialized (no graph) — restart the bot.")
        return

    # v2: every run gets its own id + checkpoint thread. Multiple runs
    # per chat can coexist, and a run can be re-attached after a restart.
    run_id = new_run_id()
    thread_id = f"tg:{chat_id}:{run_id}"
    lib = _get_library(ctx)
    if lib is not None:
        try:
            await lib.create_run(run_id=run_id, thread_id=thread_id,
                                 chat_id=chat_id, topic=topic, length=length)
        except Exception:
            logger.exception("create_run failed; continuing without registry row")

    progress = {"message_id": None}
    progress = await _stream_progress(ctx.application.bot, chat_id,
        f"🧠 *Starting research…*  run `{run_id}`", progress)
    await ctx.application.bot.send_chat_action(chat_id, ChatAction.TYPING)

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
    info: dict[str, Any] = {
        "run_id": run_id, "chat_id": chat_id, "thread_id": thread_id,
        "state": state_in, "stage": "intake", "length": length,
        "cfg": cfg, "graph": graph,
    }
    _inflight[run_id] = info
    try:
        # Stream until first interrupt (the plan-approval gate).
        async for ev in graph.astream(state_in, config=cfg,
                                      stream_mode="updates"):
            for node_name, delta in ev.items():
                info["stage"] = node_name
                if node_name == "intake":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🎯 Mode detected: _{delta.get('mode','?')}_", progress)
                elif node_name == "planner":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        "📋 Plan drafted — searching live sources…", progress)
                elif node_name == "researcher":
                    n = len(delta.get("sources") or [])
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🔍 Live search found {n} candidate source(s) — "
                        "awaiting your approval.", progress)
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

        # First HITL = plan approval — grounded: the researcher already
        # ran, so the preview lists REAL sources from live search.
        plan = cur.get("plan") or {}
        if plan:
            text = _format_plan(plan, length=length,
                                sources=cur.get("sources") or [])
            # Remember the base plan text so length taps can re-render it
            # without stacking status suffixes on top of each other.
            info["plan_text"] = text
            # _safe_send falls back to plain text if the plan body (topic /
            # sub-questions / URLs) contains markdown-special chars — so the
            # plan proposal ALWAYS renders instead of silently failing to send.
            await _safe_send(
                ctx.application.bot, chat_id, text,
                reply_markup=_plan_keyboard(default_length=length,
                                            run_id=run_id),
                parse_mode=ParseMode.MARKDOWN,
            )
            info["awaiting"] = "plan_approval"
            await _set_run_status(ctx, run_id, "awaiting_plan")
        else:
            await ctx.application.bot.send_message(
                chat_id=chat_id,
                text="(planner produced no plan; aborting.)",
            )
            _inflight.pop(run_id, None)
            await _set_run_status(ctx, run_id, "error")
    except Exception:
        _inflight.pop(run_id, None)
        await _set_run_status(ctx, run_id, "error")
        raise


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
    graph = _bot_data_get(ctx, "quick_graph")
    if graph is None:
        await update.message.reply_text(
            "⚠️ Bot not fully initialized (no graph) — restart the bot.")
        return
    await ctx.application.bot.send_chat_action(chat_id, ChatAction.TYPING)

    run_id = new_run_id()
    thread_id = f"tg:{chat_id}:{run_id}"
    lib = _get_library(ctx)
    if lib is not None:
        try:
            await lib.create_run(run_id=run_id, thread_id=thread_id,
                                 chat_id=chat_id, topic=question,
                                 mode="quick", status="running")
        except Exception:
            logger.exception("create_run failed; continuing without registry row")

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
    try:
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
        await _set_run_status(ctx, run_id, "done")
    except Exception:
        await _set_run_status(ctx, run_id, "error")
        raise


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
    for v in results:
        v.setdefault("platform", "youtube")
    thread_id = f"tg:{update.effective_chat.id}"
    _pool_put(thread_id, results)
    logger.info("/video pooled %d result(s) for %s (query=%r)",
                len(results), thread_id, query)


# ---------------------------------------------------------------------------
# Media engine commands (Phase 2): /fetch, /find, /quality + URL paste
# ---------------------------------------------------------------------------

# Telegram bots can upload at most 50 MB; leave headroom.
_TG_UPLOAD_CAP_BYTES = 49 * 1024 * 1024
_MEDIA_PROGRESS_INTERVAL_S = 3.0


def _media_keyboard(n: int) -> InlineKeyboardMarkup:
    """One row per result: download / transcript / both, index-scoped.

    Media buttons act on the per-chat result pool (30-min TTL), not on a
    run — so the callback carries the 1-based pool index.
    """
    rows = []
    for i in range(1, n + 1):
        rows.append([
            InlineKeyboardButton(f"⬇ {i}", callback_data=f"m:dl:{i}"),
            InlineKeyboardButton(f"📝 {i}", callback_data=f"m:tr:{i}"),
            InlineKeyboardButton(f"⬇+📝 {i}", callback_data=f"m:both:{i}"),
        ])
    return InlineKeyboardMarkup(rows)


async def _get_media_quality(app) -> str:
    lib = app.bot_data.get("library") if isinstance(app.bot_data, dict) else None
    if lib is None:
        return "auto"
    try:
        return (await lib.get_setting("media_quality", "auto")) or "auto"
    except Exception:
        logger.exception("could not read media_quality; using auto")
        return "auto"


async def _fetch_one_media(app, chat_id: int, url: str) -> None:
    """Background task: download one URL into the vault, register it,
    stream throttled progress edits, deliver the file when small enough."""
    bot = app.bot
    s = get_settings()
    quality = await _get_media_quality(app)

    prog = {"message_id": None, "last": 0.0}
    await _stream_progress(bot, chat_id, f"⬇️ Downloading…\n{url}", prog)

    def on_progress(pct: float, eta: str) -> None:
        now = time.monotonic()
        if pct < 100.0 and (now - prog["last"]) < _MEDIA_PROGRESS_INTERVAL_S:
            return
        prog["last"] = now
        asyncio.ensure_future(_stream_progress(
            bot, chat_id, f"⬇️ {pct:.0f}%  ·  ETA {eta}\n{url}", prog))

    try:
        r = await download_media(url, dest_root=s.media_root,
                                 quality=quality, on_progress=on_progress)
    except Exception as e:
        logger.exception("download_media crashed for %s", url)
        await _stream_progress(bot, chat_id, f"⚠️ Download crashed: {e}", prog)
        return

    if not r.ok:
        await _stream_progress(
            bot, chat_id, f"⚠️ Download failed ({r.platform or '?'}):\n"
                          f"{(r.error or 'unknown error')[:800]}", prog)
        return

    lib = app.bot_data.get("library") if isinstance(app.bot_data, dict) else None
    if lib is not None:
        try:
            await lib.add_asset(
                kind="media", platform=r.platform, source_url=url,
                media_id=r.media_id, title=r.title or url, path=r.path,
                size_bytes=r.size_bytes, duration_s=r.duration_s,
                meta={"quality": quality,
                      "info_json": r.info_json_path})
        except Exception:
            logger.exception("asset registration failed for %s", r.path)

    size_mb = r.size_bytes / (1024 * 1024)
    dur = _fmt_duration(r.duration_s)
    summary = (f"✅ Saved ({r.platform}, {size_mb:.1f} MB"
               + (f", {dur}" if dur else "") + f")\n{r.path}")
    await _stream_progress(bot, chat_id, summary, prog)

    if r.size_bytes <= _TG_UPLOAD_CAP_BYTES:
        try:
            with Path(r.path).open("rb") as f:
                await bot.send_video(
                    chat_id=chat_id, video=f,
                    caption=(r.title or Path(r.path).name)[:1024],
                    supports_streaming=True)
        except Exception:
            logger.warning("send_video failed; retrying as document")
            try:
                with Path(r.path).open("rb") as f:
                    await bot.send_document(
                        chat_id=chat_id, document=f,
                        filename=Path(r.path).name,
                        caption=(r.title or "")[:1024])
            except Exception:
                logger.exception("document send failed too; path was "
                                 "already delivered in the summary")
    else:
        await bot.send_message(
            chat_id=chat_id,
            text=(f"📁 File is {size_mb:.0f} MB (over Telegram's bot upload "
                  "cap) — it stays in the vault at the path above."))


async def _transcript_for_item(app, chat_id: int, item: dict) -> None:
    """Background task: transcript for one pool item, persisted into the
    DS vault. YouTube goes via captions; X/Reddit/Instagram go via local
    whisper ASR (download → speech-to-text)."""
    bot = app.bot
    s = get_settings()
    url = item.get("url") or ""
    platform = item.get("platform") or detect_platform(url) or "?"
    lib = app.bot_data.get("library") if isinstance(app.bot_data, dict) else None

    note = ("🎙 Transcribing captions…" if platform == "youtube" else
            "🎙 Transcribing via whisper (download + speech-to-text — this "
            "can take a few minutes; the first ever run also downloads "
            "the model)…")
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
        from .tools import youtube_search
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
        ff_note = ("available" if ffmpeg
                   else "MISSING (single-file formats only — no merged hi-res)")
        await update.message.reply_text(
            f"🎚 Download quality: {cur}\n"
            f"ffmpeg: {ff_note}\n\n"
            "Set with /quality auto | min | max | <pixel height>\n"
            "e.g. /quality 720")
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


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Free-text router. Order: (1) a pending HITL reply (plan-edit /
    revise feedback), (2) pasted media links → action keyboard, else
    ignore."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        return  # silently ignore strangers
    text = (update.message.text or "") if update.message else ""
    chat_id = update.effective_chat.id

    # (1) pending HITL reply
    pending = _pending_reply.get(chat_id)
    if pending:
        kind, run_id = pending
        info = _inflight.get(run_id)
        if info is None:
            _pending_reply.pop(chat_id, None)
            await update.message.reply_text(
                "(that run is no longer active — reply ignored)")
            return
        if not text.strip():
            await update.message.reply_text("Please send some text.")
            return
        _pending_reply.pop(chat_id, None)
        if kind == "plan_edit":
            await _replan_after_edit(ctx, info, text.strip())
        elif kind == "revise_feedback":
            await _revise_after_feedback(ctx, info, text.strip())
        return

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


async def _replan_after_edit(ctx: ContextTypes.DEFAULT_TYPE, info: dict,
                             feedback: str) -> None:
    """plan-edit: push the user's feedback into the checkpoint as if
    intake just ran, then re-drive planner→planner_reflect→researcher to
    the grounded plan gate and re-render the plan message."""
    if info.get("resuming"):
        return
    info["resuming"] = True
    graph = info["graph"]
    cfg = info["cfg"]
    run_id = info["run_id"]
    chat_id = info["chat_id"]
    progress = {"message_id": None}
    await _stream_progress(ctx.application.bot, chat_id,
                           "✏️ Redrafting the plan with your changes…",
                           progress)
    try:
        # Fork positioned after intake → next node is planner. planner
        # reads plan_feedback + the previous plan and revises.
        await graph.aupdate_state(cfg, {"plan_feedback": feedback},
                                  as_node="intake")
        async for ev in graph.astream(None, config=cfg,
                                      stream_mode="updates"):
            for node_name, _delta in ev.items():
                info["stage"] = node_name
                if node_name == "researcher":
                    n = len(_delta.get("sources") or [])
                    progress = await _stream_progress(
                        ctx.application.bot, chat_id,
                        f"🔍 Re-searched: {n} source(s) — review the new plan.",
                        progress)
        snap = await graph.aget_state(cfg)
        cur = snap.values if snap else {}
        plan = cur.get("plan") or {}
        length = info.get("length", DEFAULT_LENGTH)
        text = _format_plan(plan, length=length,
                            sources=cur.get("sources") or [])
        info["plan_text"] = text
        info["awaiting"] = "plan_approval"
        await _safe_send(ctx.application.bot, chat_id, text,
                         reply_markup=_plan_keyboard(default_length=length,
                                                     run_id=run_id),
                         parse_mode=ParseMode.MARKDOWN)
        await _set_run_status(ctx, run_id, "awaiting_plan")
    except Exception as e:
        logger.exception("re-plan failed")
        await ctx.application.bot.send_message(
            chat_id=chat_id, text=f"⚠️ Re-plan failed: {e}")
    finally:
        info["resuming"] = False


async def _revise_after_feedback(ctx: ContextTypes.DEFAULT_TYPE, info: dict,
                                 feedback: str) -> None:
    """report-revise: append the user's feedback to revision_notes, set
    revision_requested, then resume from the report-preview interrupt —
    deliver passes through → revise_prep → synthesizer → … → new preview."""
    if info.get("resuming"):
        return
    graph = info["graph"]
    cfg = info["cfg"]
    chat_id = info["chat_id"]
    try:
        snap = await graph.aget_state(cfg)
        notes = list((snap.values if snap else {}).get("revision_notes") or [])
        notes.append(f"USER REVISION REQUEST: {feedback}")
        # revision_notes is a plain (last-value) channel — send the FULL list.
        await graph.aupdate_state(
            cfg, {"revision_requested": True, "revision_notes": notes})
    except Exception as e:
        logger.exception("could not queue revision")
        await ctx.application.bot.send_message(
            chat_id=chat_id, text=f"⚠️ Could not start the revision: {e}")
        return
    await _resume_after_plan(ctx, info)


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


# ---------------------------------------------------------------------------
# /status, /cancel, /help
# ---------------------------------------------------------------------------

_STATUS_GLYPHS = {
    "planning": "🧠", "awaiting_plan": "📋", "running": "🔄",
    "awaiting_report": "📝", "done": "✅", "cancelled": "❌", "error": "⚠️",
}


async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Registry-backed status: live in-flight runs + recent run history."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    chat_id = update.effective_chat.id
    lines: list[str] = []
    entries = _chat_entries(chat_id)
    if entries:
        lines.append("*In flight:*")
        for rid, i in entries.items():
            lines.append(
                f"- `{rid}` · stage `{i.get('stage','?')}` · awaiting "
                f"`{i.get('awaiting','-')}` · length `{i.get('length','?')}`")
    lib = _get_library(ctx)
    runs: list[dict[str, Any]] = []
    if lib is not None:
        try:
            runs = await lib.list_runs(chat_id=chat_id, limit=5)
        except Exception:
            logger.exception("status: list_runs failed")
    if runs:
        lines.append("*Recent runs:*")
        for r in runs:
            glyph = _STATUS_GLYPHS.get(r["status"], "·")
            topic = (r["topic"] or "")[:48]
            lines.append(
                f"- {glyph} `{r['run_id']}` · {r['status']} · "
                f"{r['length']} · {topic}")
            if r.get("report_dir"):
                lines.append(f"    `{r['report_dir']}`")
    if not lines:
        await update.message.reply_text("No runs recorded for this chat yet.")
        return
    text = "\n".join(lines)
    try:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(text)


async def cancel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/cancel [run-id|all] — cancel in-flight run(s) for this chat.

    Without an argument: cancels when exactly one run is in flight,
    otherwise lists the candidates (never guesses which one you meant).
    """
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    chat_id = update.effective_chat.id
    target = (ctx.args[0].lower().strip() if ctx.args else "")
    entries = _chat_entries(chat_id)
    if not entries:
        await update.message.reply_text("No in-flight runs for this chat.")
        return
    if target == "all":
        picked = list(entries)
    elif target:
        picked = [rid for rid in entries if rid.startswith(target)]
        if not picked:
            await update.message.reply_text(
                f"No in-flight run matches '{target}'. "
                "In flight: " + ", ".join(entries))
            return
    elif len(entries) == 1:
        picked = list(entries)
    else:
        await update.message.reply_text(
            "Multiple runs in flight: " + ", ".join(entries)
            + "\nUse /cancel <run-id> or /cancel all.")
        return
    for rid in picked:
        _inflight.pop(rid, None)
        await _set_run_status(ctx, rid, "cancelled")
    await update.message.reply_text(
        f"❌ Cancelled {len(picked)} run(s): " + ", ".join(picked))


# ---------------------------------------------------------------------------
# /delete — vault space browser (v2 Phase 5)
# ---------------------------------------------------------------------------

_DELETE_PAGE_SIZE = 5
# Per-chat delete sessions: {category, page, selected:set[int], items:[...]}
# Items are snapshotted at listing time so toggle indices stay stable.
_delete_sessions: dict[int, dict[str, Any]] = {}

_DEL_CATEGORY_LABELS = {"runs": "🗂 Runs", "media": "🎬 Media",
                        "transcript": "📝 Transcripts"}


def _fmt_bytes(n: int | float | None) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


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
                if item["run_id"] in _inflight:
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


# ---------------------------------------------------------------------------
# Research history: /runs, /append, /continue (v2 Phase 4)
# ---------------------------------------------------------------------------


async def runs_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/runs — the run registry for this chat (newest first)."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    lib = _get_library(ctx)
    if lib is None:
        await update.message.reply_text("⚠️ Library unavailable.")
        return
    runs = await lib.list_runs(chat_id=update.effective_chat.id, limit=10)
    if not runs:
        await update.message.reply_text("No runs recorded yet — /research!")
        return
    lines = ["🗂 Runs (newest first):"]
    for r in runs:
        glyph = _STATUS_GLYPHS.get(r["status"], "·")
        lines.append(f"{glyph} {r['run_id']} · {r['status']} · {r['length']}"
                     f" · {(r['topic'] or '')[:44]}")
    lines.append("")
    lines.append("↪ /continue <id> resumes or extends a run; "
                 "/append <id> <url|asset:N> queues sources for it.")
    await update.message.reply_text("\n".join(lines))


async def _sources_from_refs(lib: Library,
                             pending: list[dict]) -> list[dict]:
    """Map pending run_sources rows onto researcher-style source dicts.

    URLs become search_result sources the fetcher downloads; asset refs
    (``asset:<id>``, e.g. vault transcripts) become local_path sources
    the fetcher ingests from disk (file:/// citations).
    """
    out: list[dict] = []
    for row in pending:
        ref = row.get("ref") or ""
        if ref.startswith("asset:"):
            try:
                rows = await lib.get_assets([int(ref.split(":", 1)[1])])
            except Exception:
                logger.exception("asset ref lookup failed for %s", ref)
                rows = []
            if not rows:
                logger.warning("appended %s no longer exists; skipping", ref)
                continue
            a = rows[0]
            out.append({
                "kind": "local",
                "title": a.get("title") or Path(a["path"]).name,
                "url": "file:///" + str(a["path"]).replace("\\", "/"),
                "local_path": a["path"],
                "summary": "", "source": "appended-asset",
            })
        else:
            out.append({
                "kind": "search_result", "title": ref, "url": ref,
                "summary": "", "source": "appended-url",
            })
    return out


async def append_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/append <run-id> <url…|asset:N…> — queue sources for a run.

    They are ingested on the next /continue <run-id> (append-only pass:
    no fresh searches, exactly your sources)."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    lib = _get_library(ctx)
    if lib is None:
        await update.message.reply_text("⚠️ Library unavailable.")
        return
    args = list(ctx.args or [])
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: /append <run-id> <url…|asset:N…>\n"
            "Find run ids with /runs, asset ids with /transcripts.")
        return
    run = await lib.resolve_run(update.effective_chat.id, args[0])
    if run is None:
        await update.message.reply_text(
            f"No unique run matches '{args[0]}' — see /runs.")
        return
    rest = " ".join(args[1:])
    refs = [t for t in rest.split() if t.startswith("asset:")]
    refs += extract_urls(rest)
    if not refs:
        await update.message.reply_text(
            "Nothing to append — give me URLs or asset:N refs.")
        return
    for ref in refs:
        await lib.add_run_source(run["run_id"], ref,
                                 kind="asset" if ref.startswith("asset:")
                                 else "url")
    pending = await lib.pending_sources(run["run_id"])
    await update.message.reply_text(
        f"➕ Appended {len(refs)} source(s) to run {run['run_id']} "
        f"({len(pending)} pending total).\n"
        f"Run /continue {run['run_id']} to ingest them.")


async def continue_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/continue <run-id> — re-attach a paused run, or extend a finished
    one (with appended sources if any, else a fresh search pass)."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    lib = _get_library(ctx)
    graph = _bot_data_get(ctx, "graph")
    if lib is None or graph is None:
        await update.message.reply_text("⚠️ Bot not fully initialized.")
        return
    if not ctx.args:
        await update.message.reply_text("Usage: /continue <run-id> — see /runs.")
        return
    chat_id = update.effective_chat.id
    run = await lib.resolve_run(chat_id, ctx.args[0])
    if run is None:
        await update.message.reply_text(
            f"No unique run matches '{ctx.args[0]}' — see /runs.")
        return
    run_id = run["run_id"]
    if run_id in _inflight and _inflight[run_id].get("resuming"):
        await update.message.reply_text(
            f"Run {run_id} is already resuming — hang on.")
        return
    cfg = {"configurable": {"thread_id": run["thread_id"]}}
    try:
        snap = await graph.aget_state(cfg)
    except Exception:
        logger.exception("aget_state failed for %s", run["thread_id"])
        snap = None
    if snap is None or not (snap.values or {}):
        await update.message.reply_text(
            f"No checkpoint found for run {run_id} — it may predate the "
            "v2 upgrade or its checkpoint DB was deleted.")
        return
    cur = snap.values
    info: dict[str, Any] = {
        "run_id": run_id, "chat_id": chat_id,
        "thread_id": run["thread_id"],
        "state": {"user_request": run.get("topic", "")},
        "stage": "continue", "length": cur.get("length") or run.get("length")
                                        or DEFAULT_LENGTH,
        "cfg": cfg, "graph": graph,
    }

    nxt = tuple(snap.next or ())
    if nxt == ("fetcher",):
        # Paused at the plan gate (possibly before a restart) — re-render.
        _inflight[run_id] = info
        info["awaiting"] = "plan_approval"
        plan = cur.get("plan") or {}
        text = _format_plan(plan, length=info["length"],
                            sources=cur.get("sources") or [])
        info["plan_text"] = text
        await _safe_send(ctx.application.bot, chat_id, text,
                         reply_markup=_plan_keyboard(
                             default_length=info["length"], run_id=run_id),
                         parse_mode=ParseMode.MARKDOWN)
        await _set_run_status(ctx, run_id, "awaiting_plan")
        return
    if nxt == ("deliver",):
        # Paused at the report preview — re-render preview + keyboard.
        _inflight[run_id] = info
        info["awaiting"] = "report_preview"
        info["paths"] = cur.get("report_paths") or {}
        await _send_report_preview(ctx, info)
        await _set_run_status(ctx, run_id, "awaiting_report")
        return
    if nxt:
        await update.message.reply_text(
            f"Run {run_id} is mid-pipeline (next: {', '.join(nxt)}) — "
            "wait for it to pause or finish.")
        return

    # Finished run → extend fork. Merge pending appended sources into the
    # FULL sources list (last-value channel: never send a delta).
    pending = await lib.pending_sources(run_id)
    appended = await _sources_from_refs(lib, pending)
    merged = list(cur.get("sources") or []) + appended
    append_only = bool(appended)
    try:
        await graph.aupdate_state(
            cfg,
            {"sources": merged, "extend_requested": True,
             "extend_rounds": 0, "revision_rounds": 0,
             "append_only": append_only},
            as_node="deliver")
    except Exception as e:
        logger.exception("extend fork failed for %s", run_id)
        await update.message.reply_text(f"⚠️ Could not fork the run: {e}")
        return
    _inflight[run_id] = info
    await _set_run_status(ctx, run_id, "running")
    mode_note = (f"with {len(appended)} appended source(s), no new searches"
                 if append_only else "with a fresh search pass")
    await update.message.reply_text(
        f"▶️ Continuing run {run_id} {mode_note}…")
    await _resume_after_plan(ctx, info, resume_input=None)
    if pending and info.get("awaiting") == "report_preview":
        try:
            await lib.mark_sources(run_id, [p["ref"] for p in pending],
                                   "ingested")
        except Exception:
            logger.exception("mark_sources failed for %s", run_id)


async def help_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    full = bool(ctx.args) and ctx.args[0].lower() in ("full", "guide", "all")
    if full and _HELP_MD_PATH.exists():
        # Send the detailed guide as a document (it's longer than a
        # single Telegram message and reads better as a file).
        try:
            with _HELP_MD_PATH.open("rb") as f:
                await ctx.application.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=InputFile(f, filename="argus-help.md"),
                    caption="📖 Argus — full command guide")
            return
        except Exception:
            logger.exception("could not send help.md; falling back to text")
    await _safe_send(ctx.application.bot, update.effective_chat.id,
                     HELP_TEXT, parse_mode=ParseMode.MARKDOWN)


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
    data = q.data or ""
    # v2: every callback is run-scoped — ``<ns>:<action>:<run8>``.
    parts = data.split(":")
    if len(parts) != 3:
        # Buttons minted before the v2 upgrade (or garbage) — the run they
        # belonged to cannot be identified; say so instead of guessing.
        await q.edit_message_text("(stale button — start a new /research)")
        return
    ns, action, run_id = parts

    # Media buttons (m:dl:<i> / m:tr:<i> / m:both:<i>) act on the
    # per-chat result pool, not on a run — handle before run lookup.
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
        queued: list[str] = []
        if action in ("dl", "both"):
            ctx.application.create_task(
                _fetch_one_media(ctx.application, chat_id, item["url"]))
            queued.append("download")
        if action in ("tr", "both"):
            ctx.application.create_task(
                _transcript_for_item(ctx.application, chat_id, item))
            queued.append("transcript")
        if queued:
            await ctx.application.bot.send_message(
                chat_id=chat_id,
                text=f"⏬ Queued {' + '.join(queued)} for #{idx}: "
                     f"{(item.get('title') or item['url'])[:80]}")
        return

    # Delete-browser buttons (del:cat/tgl/pg/ok/yes/no/cancel).
    if ns == "del":
        await _on_delete_callback(update, ctx, action, run_id)
        return

    # Transcript library buttons (t:send:<asset_id>).
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

    info = _inflight.get(run_id)
    if not info:
        await q.edit_message_text(
            f"(run {run_id} is not in flight — it may have finished, been "
            "cancelled, or been lost in a bot restart)")
        return

    # T7.1 + 2026-07-10 fix: length taps update the chosen length and
    # re-render the plan message, but MUST NOT resume the graph —
    # resuming belongs to the Approve button (single resume gate).
    if ns == "len":
        chosen = action
        if chosen not in _LENGTH_LABELS:
            await q.edit_message_text("Unknown length mode.")
            return
        info["length"] = chosen
        # Re-render from the stored base plan text (no suffix stacking)
        # AND re-send the keyboard: Telegram's editMessageText REMOVES
        # the inline keyboard when reply_markup is omitted, so the old
        # code made the Approve button vanish after every length tap.
        # Re-sending also moves the ✅ marker onto the tapped length.
        base = info.get("plan_text") or (q.message.text or "")
        await _safe_edit_cb(
            q,
            base + f"\n\n_📏 Length set to *{_LENGTH_LABELS[chosen]}* "
                   "— tap ✅ Approve to continue._",
            reply_markup=_plan_keyboard(default_length=chosen, run_id=run_id),
        )
        return

    if ns == "plan":
        if action == "approve":
            if info.get("plan_approved") is True or info.get("resuming"):
                # Double-tap guard — one resume per plan gate.
                return
            info["plan_approved"] = True
            await _safe_edit_cb(
                q, (info.get("plan_text") or q.message.text or "")
                + "\n\n_✅ Approved — continuing._")
            await _set_run_status(ctx, run_id, "running")
            await _resume_after_plan(ctx, info)
        elif action == "edit":
            info["plan_edit"] = True
            _pending_reply[info["chat_id"]] = ("plan_edit", run_id)
            base = info.get("plan_text") or (q.message.text or "")
            await _safe_edit_cb(
                q, base + "\n\n_✏️ Reply with your changes and I'll redraft "
                          "the plan (or tap ✅ Approve to keep this one)._",
                reply_markup=_plan_keyboard(
                    default_length=info.get("length", DEFAULT_LENGTH),
                    run_id=run_id),
            )
        elif action == "cancel":
            _inflight.pop(run_id, None)
            await _set_run_status(ctx, run_id, "cancelled")
            await q.edit_message_text("❌ Cancelled.")
        return

    if ns == "report":
        if action == "send":
            await _resume_after_report(ctx, info, q)
        elif action == "extend":
            if info.get("resuming"):
                return
            # Phase 2 HITL: deepen the research. Set the graph flag, then
            # reuse the plan-resume streamer — it drives deliver →
            # extend_prep → fetcher → … → report_builder and shows the
            # next preview.
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
            await _set_run_status(ctx, run_id, "running")
            await _resume_after_plan(ctx, info)
        elif action == "revise":
            if info.get("resuming"):
                return
            _pending_reply[info["chat_id"]] = ("revise_feedback", run_id)
            await _safe_edit_cb(
                q, (q.message.text or "")
                + "\n\n_🔁 Reply with what to change and I'll revise the "
                  "report._")
        elif action == "cancel":
            _inflight.pop(run_id, None)
            await _set_run_status(ctx, run_id, "cancelled")
            await q.edit_message_text("❌ Cancelled (report dropped).")
        return


async def _resume_after_plan(ctx: ContextTypes.DEFAULT_TYPE, info: dict,
                             resume_input: Any = ...):
    """Resume the graph from the plan-gate interrupt (or re-drive the
    deliver→extend loop). Re-entrancy-guarded: only one astream session
    per run may be live — a second resume on the same thread races the
    first and produced the 'Approve unpressable' bug class.

    ``resume_input`` defaults to ``Command(resume=True)`` (continuing
    from a static interrupt). /continue on a FINISHED run forks the
    checkpoint via ``aupdate_state(as_node="deliver")`` first and passes
    ``None`` — there is no pending interrupt to resume, the graph just
    proceeds from the fork.
    """
    if info.get("resuming"):
        logger.warning("resume already in flight for run %s; ignoring",
                       info.get("run_id"))
        return
    info["resuming"] = True
    graph = info["graph"]
    cfg = info["cfg"]
    run_id = info["run_id"]
    chat_id = info["chat_id"]
    progress = {"message_id": None}
    if resume_input is ...:
        resume_input = Command(resume=True)

    # T7.1 — push the chosen length back into the graph state BEFORE
    # resuming, so the synthesizer + report_builder see the user's
    # HITL choice rather than the default. ``aupdate_state`` is the
    # canonical way to fork a thread checkpoint from outside.
    chosen_length = (info.get("length")
                     or info.get("state", {}).get("length")
                     or DEFAULT_LENGTH)
    try:
        await graph.aupdate_state(cfg, {"length": chosen_length})
    except Exception:
        logger.exception(
            "aupdate_state failed; defaulting length=%s", chosen_length,
        )

    progress = await _stream_progress(ctx.application.bot, chat_id,
        "📥 Fetching approved sources…", progress)

    try:
        async for ev in graph.astream(resume_input, config=cfg,
                                      stream_mode="updates"):
            for node_name, delta in ev.items():
                info["stage"] = node_name
                if node_name == "extend_prep":
                    n = len(delta.get("sources") or [])
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🔎 Extend: source pool now {n}.", progress)
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
        await _set_run_status(ctx, run_id, "awaiting_report",
                              report_dir=paths.get("folder"))
        await _send_report_preview(ctx, info)
    except Exception as e:
        logger.exception("resume failed")
        await _set_run_status(ctx, run_id, "error")
        await ctx.application.bot.send_message(chat_id=chat_id,
            text=f"⚠️ resume failed: {e}")
    finally:
        info["resuming"] = False


async def _send_report_preview(ctx: ContextTypes.DEFAULT_TYPE,
                               info: dict) -> None:
    """Send the report-preview message + keyboard for a run whose
    report_builder has produced ``info["paths"]``. Shared by the normal
    resume flow and /continue's re-attach."""
    chat_id = info["chat_id"]
    run_id = info["run_id"]
    paths = info.get("paths") or {}
    chosen_length = info.get("length") or DEFAULT_LENGTH
    if not paths.get("md"):
        await ctx.application.bot.send_message(
            chat_id=chat_id, text="(no report_paths found)")
        return
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

    # HTML primary (tolerates arbitrary content), plain-text fallback —
    # legacy-markdown crashed on ``_ * [ &`` in excerpts (2026-07-08 bug).
    html_body = _html_escape_for_tg(text) + (
        ("\n\n<pre>" + _html_escape_for_tg(excerpt) + "</pre>")
        if excerpt else ""
    )
    plain_body = text + ("\n\n```\n" + excerpt + "\n```" if excerpt else "")

    try:
        await ctx.application.bot.send_message(
            chat_id=chat_id,
            text=html_body,
            reply_markup=_report_keyboard(run_id),
            parse_mode=ParseMode.HTML,
        )
    except Exception as send_err:
        logger.warning(
            "report preview HTML send failed (%s); falling back to plain text",
            send_err,
        )
        try:
            await ctx.application.bot.send_message(
                chat_id=chat_id,
                text=plain_body,
                reply_markup=_report_keyboard(run_id),
            )
        except Exception as plain_err:
            logger.exception("report preview plain-text send also failed")
            await ctx.application.bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ report preview unavailable: {plain_err}",
            )


async def _resume_after_report(ctx: ContextTypes.DEFAULT_TYPE,
                                info: dict, q):
    run_id = info["run_id"]
    chat_id = info["chat_id"]
    paths = info.get("paths") or {}
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
        _inflight.pop(run_id, None)
        await _finalize_run(ctx, info, paths)


async def _finalize_run(ctx: ContextTypes.DEFAULT_TYPE, info: dict,
                        paths: dict) -> None:
    """Terminal bookkeeping: registry status → done, register the report
    as a library asset, and write the human-readable run.md mirror into
    the run's report folder."""
    run_id = info["run_id"]
    await _set_run_status(ctx, run_id, "done",
                          report_dir=paths.get("folder"))
    lib = _get_library(ctx)
    if lib is None:
        return
    try:
        if paths.get("md"):
            md = Path(paths["md"])
            size = md.stat().st_size if md.exists() else 0
            topic = (info.get("state") or {}).get("user_request", "")
            await lib.add_asset(
                kind="report", path=str(md), title=topic, size_bytes=size,
                meta={"folder": paths.get("folder", ""),
                      "pdf": paths.get("pdf", "")},
            )
        run = await lib.get_run(run_id)
        if run and run.get("report_dir"):
            await asyncio.to_thread(mirror_run_md, run)
    except Exception:
        logger.exception("finalize (asset/mirror) failed for run %s", run_id)


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


async def _post_init(app: Application) -> None:
    """Open the shared checkpoint saver + library and compile both graphs.

    ONE AsyncSqliteSaver serves every run — isolation comes from per-run
    thread ids, not per-run connections. This replaces the old per-run
    saver keep-alive dance (and its connection leaks) entirely.
    """
    s = get_settings()
    saver_cm = async_sqlite_saver_cm(str(s.checkpoint_db))
    saver = await saver_cm.__aenter__()
    lib = Library(s.library_db)
    await lib.open()
    app.bot_data["saver_cm"] = saver_cm
    app.bot_data["saver"] = saver
    app.bot_data["graph"] = build_graph(checkpointer=saver)
    app.bot_data["quick_graph"] = quick_answer_graph(checkpointer=saver)
    app.bot_data["library"] = lib
    ffmpeg = resolve_ffmpeg()
    if ffmpeg:
        logger.info("ffmpeg: %s", ffmpeg)
    else:
        logger.warning("ffmpeg NOT found — media downloads fall back to "
                       "single-file formats (set ARGUS_FFMPEG or install "
                       "imageio-ffmpeg)")
    loop_name = type(asyncio.get_running_loop()).__name__
    if "Proactor" not in loop_name and os.name == "nt":
        # asyncio subprocesses (Phase 2 media engine) need the Proactor
        # loop on Windows. PTB does not override the policy; this fires
        # only if some future import does.
        logger.warning("event loop is %s, not Proactor — asyncio "
                       "subprocesses will not work on Windows", loop_name)
    logger.info("post_init: saver + library + graphs ready (loop=%s, "
                "library=%s)", loop_name, s.library_db)


async def _post_shutdown(app: Application) -> None:
    lib = app.bot_data.get("library")
    if lib is not None:
        try:
            await lib.close()
        except Exception:
            logger.exception("library close on shutdown failed")
    saver_cm = app.bot_data.get("saver_cm")
    if saver_cm is not None:
        try:
            await saver_cm.__aexit__(None, None, None)
        except Exception:
            logger.exception("saver close on shutdown failed")


def build_application() -> Application:
    s = get_settings()
    if not s.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN missing from .env")
    app = (
        Application.builder()
        .token(s.telegram_bot_token)
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )
    app.add_handler(CommandHandler("research", research_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("video", video_cmd))
    app.add_handler(CommandHandler("find", find_cmd))
    app.add_handler(CommandHandler("fetch", fetch_cmd))
    app.add_handler(CommandHandler("quality", quality_cmd))
    app.add_handler(CommandHandler("transcript", transcript_cmd))
    app.add_handler(CommandHandler("transcripts", transcripts_cmd))
    app.add_handler(CommandHandler("runs", runs_cmd))
    app.add_handler(CommandHandler("append", append_cmd))
    app.add_handler(CommandHandler("continue", continue_cmd))
    app.add_handler(CommandHandler("delete", delete_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
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
