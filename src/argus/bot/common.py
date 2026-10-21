"""Shared bot helpers and state: the ACL check, markdown-safe sends and
edits, escaping and formatting helpers, the run-registry accessors, the
in-flight run and pending-reply maps, the Telegram client limits and
timeouts, and the help text."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from ..config import Settings
from ..graph.state import Length
from ..library import Library

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

*Engines*
  /engines — models per role · /engines use <profile>

_Tip: paste any YouTube / X / Reddit / Instagram link for a download menu._
"""

# Loaded lazily from docs/help.md (repo) for `/help full`.
_HELP_MD_PATH = Path(__file__).resolve().parents[3] / "docs" / "help.md"

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


# The run driver's in-flight registry (``runs.py``): {run_id (8-hex):
# RunHandle} for every run this process drives or holds at a gate. The
# checkpoints and the Library are the sources of truth; a handle holds the
# live objects (the drive task, its RunControl, the per-run lock, the gate
# message, the plan-gate choices).
_inflight: dict[str, Any] = {}


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


# v2 Phase 6a — the next free-text message from a chat is interpreted as
# a reply to a HITL prompt: {chat_id: (kind, run_id)} where kind is
# "plan_edit" or "revise_feedback". Consumed once by on_text.
_pending_reply: dict[int, tuple[str, str]] = {}


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
    # Unset (0) allows nobody — the bot still starts, then refuses everyone.
    if not settings.telegram_allowed_user_id:
        return False
    return user_id == settings.telegram_allowed_user_id


async def _send_md_doc(bot, chat_id: int, path: Path, caption: str = ""):
    if not path.exists():
        return
    with path.open("rb") as f:
        await bot.send_document(chat_id=chat_id, document=f,
                                filename=path.name, caption=caption[:1024])


TG_LIMIT = 4096        # Telegram's message limit, in UTF-16 code units after entity parsing


def _tg_len(text: str) -> int:
    """``text``'s length as Telegram counts it (UTF-16 code units). Measured on
    the source, it is never below the parsed length: parsing only removes."""
    return len(text.encode("utf-16-le")) // 2


def _tg_cut(text: str, limit: int = TG_LIMIT) -> str:
    """``text`` cut to ``limit`` (``_tg_len``), with a visible "…(cut)" mark
    when the limit has room for it; any limit terminates (≤ 0 gives "")."""
    if _tg_len(text) <= limit:
        return text
    mark = "\n…(cut)"
    if limit < _tg_len(mark):
        mark = ""
    room = max(limit - _tg_len(mark), 0)
    keep = min(len(text), room)
    while keep > 0 and _tg_len(text[:keep]) > room:
        keep -= 1
    return text[:keep] + mark


async def _safe_send(bot, chat_id: int, text: str, *, reply_markup=None,
                     parse_mode=ParseMode.MARKDOWN, fallback: str | None = None):
    """send_message that never dies on a Telegram markdown parse error.

    Legacy-markdown mode chokes on raw dynamic content (file paths with ``_``,
    topics with ``*``/``[``, etc.) with "Can't parse entities". We try the
    requested parse_mode first, then retry as plain text (cut to Telegram's
    limit) so the message always goes out. A plain send that still fails is
    never silent: ``fallback`` (or a "could not be shown" line naming the
    error) is sent in its place, with the same ``reply_markup``, so a gate
    card keeps its buttons. Returns the sent Message (None only when the
    fallback failed too).
    """
    try:
        return await bot.send_message(chat_id=chat_id, text=text,
                                      reply_markup=reply_markup,
                                      parse_mode=parse_mode)
    except Exception as e:
        logger.warning("send (%s) failed (%s); retrying plain text",
                       parse_mode, e)
        try:
            return await bot.send_message(chat_id=chat_id, text=_tg_cut(text),
                                          reply_markup=reply_markup)
        except Exception as plain_err:
            logger.exception("plain-text send also failed; sending the fallback")
            note = fallback or f"⚠️ This message could not be shown ({plain_err})."
            try:
                return await bot.send_message(chat_id=chat_id, text=_tg_cut(note),
                                              reply_markup=reply_markup)
            except Exception:
                logger.exception("the fallback send failed too")
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
    # Telegram's legacy parser honours "\" only before _ * ` [ (outside an
    # entity; inside one nothing is escaped): a "\" or "]" escaped too would
    # show its backslash.
    out = s
    for ch in ("_", "*", "`", "["):
        out = out.replace(ch, "\\" + ch)
    return out


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


# Telegram bots can upload at most 50 MB; leave headroom.
_TG_UPLOAD_CAP_BYTES = 49 * 1024 * 1024


# Network timeouts for the Telegram HTTP client (see build_application).
# Uploads are the slow path: a 49 MB video on a 2 Mbit/s uplink needs
# ~200s of write time alone, then Telegram transcodes before replying.
_TG_CONNECT_TIMEOUT_S = float(os.environ.get("ARGUS_TG_CONNECT_TIMEOUT", "30"))
_TG_READ_TIMEOUT_S = float(os.environ.get("ARGUS_TG_READ_TIMEOUT", "180"))
_TG_WRITE_TIMEOUT_S = float(os.environ.get("ARGUS_TG_WRITE_TIMEOUT", "120"))
_TG_POOL_TIMEOUT_S = float(os.environ.get("ARGUS_TG_POOL_TIMEOUT", "60"))
_TG_MEDIA_WRITE_TIMEOUT_S = float(
    os.environ.get("ARGUS_TG_MEDIA_WRITE_TIMEOUT", "1800"))


_STATUS_GLYPHS = {
    "planning": "🧠", "awaiting_plan": "📋", "running": "🔄",
    "awaiting_report": "📝", "done": "✅", "cancelled": "❌", "error": "⚠️",
}


def _fmt_bytes(n: int | float | None) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"
