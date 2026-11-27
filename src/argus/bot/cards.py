"""The video card: one chat message that previews a media link, lets the
user switch format (video / mp3) and quality, and then becomes that
download's live progress card (bar, stage, percent, size, ETA), ending
done, failed or cancelled.

Two forms of the same state:

- native (always): an HTML text message under Telegram's own link preview
  of the video, with a ``vc:<action>:<arg>`` inline keyboard;
- widget (``/widgets on``, Bot API 10.4): one HTML widget
  (assets/widgets/media_card.html) drawn from ``widget_state()``, its taps
  arriving as ``widget_query`` updates. If Telegram refuses the widget, the
  switch goes off, the chat is told once, and the native card is sent.

The download itself runs in bot/media.py; this module holds the state, the
rendering and the message upkeep.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import BaseHandler

from ..media import MP3_QUALITIES
from . import tgx
from .common import _fmt_bytes, _fmt_duration, _html_escape_for_tg as _esc
from .progress import edit_text

logger = logging.getLogger("argus.bot")

WIDGETS_KEY = "chat.widgets"          # library setting: "on" | "off" (default)
_PROGRESS_INTERVAL_S = 3.0           # at most one progress edit per card this often
_MAX_CARDS = 200                     # ponytail: in-memory, lost on restart; a tap on an old card says it expired
_WIDGET_HTML = Path(__file__).resolve().parents[1] / "assets" / "widgets" / "media_card.html"

_STAGES = {"queued": "Waiting for a free slot", "starting": "Starting", "video": "Video stream",
           "audio": "Audio stream", "download": "Downloading", "merging": "Merging video and audio",
           "converting": "Converting to MP3", "finishing": "Finishing"}
_MP3_KBPS = {"320": 320, "v0": 245, "192": 192, "128": 128}   # v0 is VBR: its typical rate
_HEAD = {"read": "🎬", "pick": "🎬", "queue": "⏳", "dl": "⬇️", "done": "✅", "fail": "⚠️", "x": "✖️"}


@dataclass(eq=False)
class Card:
    chat_id: int
    url: str
    platform: str = ""
    title: str = ""
    meta: str = ""
    duration_s: float | None = None
    fmt: str = "v"                     # v (video) | a (mp3)
    quality: str = "auto"              # video: a short-side height, or auto
    mp3: str = "320"
    video_opts: list[str] = field(default_factory=list)
    sizes: dict[str, int] = field(default_factory=dict)
    audio_bytes: int | None = None
    phase: str = "read"                # read | pick | queue | dl | done | fail | x
    stage: str = ""
    pct: float | None = None
    eta: str = ""
    done_bytes: int | None = None
    total_bytes: int | None = None
    summary: str = ""
    error: str = ""
    failed_at: str = ""                # probe | download: what Try again repeats
    message_id: int | None = None
    widget: bool = False
    done_markup: InlineKeyboardMarkup | None = None
    task: object = None                # the running download (app.create_task)
    gone: bool = False                 # the message no longer exists
    shown_at: float = 0.0
    pending: asyncio.Future | None = None
    dirty: bool = False                # a progress change no redraw has drawn yet
    cap_phase: str = ""                # the phase a widget's caption shows
    rev: int = 0                       # widget_state's revision: the widget keeps the newest
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)   # one edit at a time


_cards: dict[tuple[int, int], Card] = {}


def get(chat_id: int | None, message_id: int | None) -> Card | None:
    return _cards.get((chat_id, message_id)) if chat_id is not None and message_id is not None else None


def _remember(c: Card) -> None:
    _cards[(c.chat_id, c.message_id)] = c
    for key in [k for k, v in _cards.items() if v.phase in ("done", "fail", "x")][:max(0, len(_cards) - _MAX_CARDS)]:
        del _cards[key]


# ---------------------------------------------------------------------------
# filling from a probe, and the user's choices
# ---------------------------------------------------------------------------

def default_quality(setting: str | None, opts: list[str]) -> str:
    """The /quality setting mapped onto the heights a link offers."""
    if not opts or opts == ["auto"]:
        return "auto"
    nums = [int(o) for o in opts]                      # highest first
    s = (setting or "auto").strip().lower()
    if s == "max":
        return opts[0]
    if s == "min":
        return opts[-1]
    cap = int(s) if s.isdigit() else 1080
    fit = [n for n in nums if n <= cap]
    return str(fit[0] if fit else nums[-1])


def fill(c: Card, info: dict, quality_setting: str | None) -> None:
    """A media.probe_url result into the card (title, meta, the heights on
    offer, size estimates, the default quality)."""
    c.title = (info.get("title") or c.title or c.url)[:200]
    c.duration_s = info.get("duration_s")
    c.meta = " · ".join(x for x in (info.get("uploader") or "", _fmt_duration(c.duration_s),
                                    info.get("platform") or c.platform) if x)
    heights = sorted({int(h) for h in info.get("heights") or []}, reverse=True)[:6]
    c.video_opts = [str(h) for h in heights] or ["auto"]
    c.sizes = {str(k): int(v) for k, v in (info.get("sizes") or {}).items() if v}
    c.audio_bytes = info.get("audio_bytes")
    c.quality = default_quality(quality_setting, c.video_opts)


def apply(c: Card, action: str, arg: str) -> bool:
    """A format or quality tap on a card at its pick step; True when it changed."""
    if c.phase != "pick":
        return False
    if action == "f" and arg in ("v", "a") and arg != c.fmt:
        c.fmt = arg
        return True
    if action == "q" and arg in c.video_opts and (c.fmt, c.quality) != ("v", arg):
        c.fmt, c.quality = "v", arg
        return True
    if action == "a" and arg in MP3_QUALITIES and (c.fmt, c.mp3) != ("a", arg):
        c.fmt, c.mp3 = "a", arg
        return True
    return False


_QUALITY_LABELS = {"auto": "Auto (≤1080p)", "max": "Max", "min": "Min"}


def _qlabel(q: str) -> str:
    return f"{q}p" if q.isdigit() else _QUALITY_LABELS.get(q, q)


def _alabel(a: str) -> str:
    return "V0" if a == "v0" else f"{a} kbps"


def size_estimate(c: Card) -> str:
    if c.fmt == "a":
        n = c.duration_s * _MP3_KBPS.get(c.mp3, 320) * 125 if c.duration_s else None
    else:
        n = c.sizes.get(c.quality)
    return f"≈ {_fmt_bytes(n)}" if n else ""


def choice(c: Card) -> str:
    return f"MP3 · {_alabel(c.mp3)}" if c.fmt == "a" else f"Video · {_qlabel(c.quality)}"


def _bar(pct: float | None, cells: int = 12) -> str:
    n = round(max(0.0, min(100.0, pct or 0.0)) / 100 * cells)
    return "█" * n + "░" * (cells - n)


def _progress_line(c: Card) -> str:
    got = (f"{_fmt_bytes(c.done_bytes)} / {_fmt_bytes(c.total_bytes)}" if c.total_bytes
           else _fmt_bytes(c.done_bytes) if c.done_bytes else "")
    eta = f"ETA {c.eta}" if c.eta and c.eta not in ("NA", "Unknown") else ""
    return " · ".join(x for x in (_STAGES.get(c.stage, c.stage), got, eta) if x)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_text(c: Card, *, short: bool = False) -> str:
    """The card as Telegram HTML: the message text (native) or the caption
    under a widget (``short``: under the 1024-character caption cap, and
    changing with the phase only, since the widget shows the rest)."""
    title = (c.title or c.url)[: 120 if short else 200]
    lines = [f"{_HEAD[c.phase]} <b>{_esc(title)}</b>"]
    if c.meta:
        lines.append(_esc(c.meta[:200]))
    if c.phase == "read":
        lines.append("Reading the link…")
    elif c.phase == "pick" and short:
        lines += ["", "Choose a format and quality."]
    elif c.phase == "pick":
        est = size_estimate(c)
        lines += ["", f"<b>{_esc(choice(c))}</b>" + (f" · {est}" if est else ""),
                  "Pick a format and quality, then tap Download."]
    elif c.phase in ("queue", "dl") and short:
        lines += ["", "Queued…" if c.phase == "queue" else "Downloading…"]
    elif c.phase in ("queue", "dl"):
        lines += ["", f"{_esc(choice(c))}",
                  f"<code>{_bar(c.pct)}</code> {c.pct or 0:.0f}%" if c.phase == "dl" else "",
                  _esc(_progress_line(c))]
    elif c.phase == "done":
        lines += ["", f"Saved · {_esc(c.summary)}"]
    elif c.phase == "fail":
        lines += ["", _esc(c.error[: 300 if short else 800])]
    elif c.phase == "x":
        lines += ["", "Cancelled."]
    return "\n".join(x for x in lines if x is not None).replace("\n\n\n", "\n\n").strip()


def _btn(text: str, action: str, arg: str = "_", style: str | None = None) -> InlineKeyboardButton:
    return InlineKeyboardButton(text, callback_data=f"vc:{action}:{arg}", style=style)


def keyboard(c: Card) -> InlineKeyboardMarkup | None:
    """The native card's buttons for its phase."""
    if c.phase == "pick":
        rows = [[_btn("🎬 Video" + (" ✓" if c.fmt == "v" else ""), "f", "v", "primary" if c.fmt == "v" else None),
                 _btn("🎵 MP3" + (" ✓" if c.fmt == "a" else ""), "f", "a", "primary" if c.fmt == "a" else None)]]
        if c.fmt == "v":
            chips = [_btn(_qlabel(q) + (" ✓" if q == c.quality else ""), "q", q,
                          "primary" if q == c.quality else None) for q in c.video_opts]
        else:
            chips = [_btn(_alabel(a) + (" ✓" if a == c.mp3 else ""), "a", a,
                          "primary" if a == c.mp3 else None) for a in MP3_QUALITIES]
        rows += [chips[i:i + 3] for i in range(0, len(chips), 3)]
        rows.append([_btn("⬇️ Download", "ok", style="success"), _btn("✖ Cancel", "x", style="danger")])
        return InlineKeyboardMarkup(rows)
    if c.phase in ("read", "queue", "dl"):
        return InlineKeyboardMarkup([[_btn("✖ Cancel", "x", style="danger")]])
    if c.phase == "fail":
        return InlineKeyboardMarkup([[_btn("🔁 Try again", "retry", style="primary")]])
    if c.phase == "done":
        return c.done_markup
    return None


def _widget_markup(c: Card) -> InlineKeyboardMarkup:
    """Under a widget: a way back to regular buttons (all an app without
    widgets can tap), and the item buttons once it is done."""
    rows = list(c.done_markup.inline_keyboard) if c.phase == "done" and c.done_markup else []
    return InlineKeyboardMarkup([*rows, [_btn("🔘 Regular buttons", "plain")]])


def _compact(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def widget_state(c: Card) -> str:
    """The widget's chat_data (or a query's answer): compact JSON, at most
    1024 UTF-8 bytes. ``n`` rises with every call, so the widget can drop
    a state that arrives after a newer one (an answer and a push race)."""
    c.rev += 1
    d: dict = {"v": 1, "n": c.rev, "ph": c.phase, "t": (c.title or c.url)[:90], "m": c.meta[:80], "f": c.fmt,
               "vq": c.video_opts, "aq": list(MP3_QUALITIES), "q": c.quality, "a": c.mp3,
               "sz": size_estimate(c)}
    if c.phase in ("queue", "dl"):
        if c.pct is not None:                  # unknown: the widget shows no percent
            d["p"] = int(c.pct)
        d.update(st=c.stage,
                 g=(f"{_fmt_bytes(c.done_bytes)} / {_fmt_bytes(c.total_bytes)}" if c.total_bytes
                    else _fmt_bytes(c.done_bytes) if c.done_bytes else ""),
                 e=c.eta if c.eta not in ("NA", "Unknown") else "")
    elif c.phase == "done":
        d["r"] = c.summary[:120]
    elif c.phase == "fail":
        d["err"] = c.error[:300]
    while True:
        s = _compact(d)
        if len(s.encode("utf-8")) <= 1024:
            return s
        if len(d["t"]) > 20:                   # a long non-Latin title costs 2-4 bytes a character
            d["t"] = d["t"][:-10]
        elif d.get("m"):
            d["m"] = ""
        else:
            d["err"] = d.get("err", "")[:100]
            d["t"] = d["t"][:10]


def off_state(text: str, c: Card | None = None) -> str:
    """chat_data for a widget that no longer acts (expired, or moved to a
    regular card): ``text`` and no buttons."""
    d: dict = {"v": 1, "ph": "off", "err": text}
    if c is not None:
        c.rev += 1
        d.update(n=c.rev, t=(c.title or c.url)[:90])
    return _compact(d)


def _preview(c: Card) -> LinkPreviewOptions:
    """Telegram's own preview of the link, large and above the card text."""
    return LinkPreviewOptions(url=c.url, prefer_large_media=True, show_above_text=True)


def widget_html() -> str:
    return _WIDGET_HTML.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# sending and upkeep
# ---------------------------------------------------------------------------

async def send(bot, c: Card, *, lib) -> None:
    """Post the card: as a widget when the switch is on and Telegram takes
    it, else as the native message. A refused widget turns the switch off
    and says so once; the native card follows either way."""
    if lib is not None and (await lib.get_setting(WIDGETS_KEY, "off")) == "on":
        try:
            c.message_id = await tgx.send_widget(bot, c.chat_id, widget_html(), chat_data=widget_state(c),
                                                 caption=render_text(c, short=True), reply_markup=_widget_markup(c))
            c.widget, c.cap_phase = True, c.phase
            tgx._noticed.discard((c.chat_id, "widget-error"))     # the next failure is said again
        except TelegramError as e:
            if tgx.refused(e):
                await lib.set_setting(WIDGETS_KEY, "off")
                await tgx.notice_once(bot, c.chat_id, "widget", (
                    f"ℹ️ Telegram refused the widget ({e.message}), so Argus switched widgets off and "
                    "shows regular buttons. /widgets on tries again."))
            else:
                logger.warning("widget send failed (%s); sending the regular card", e)
                await tgx.notice_once(bot, c.chat_id, "widget-error", (
                    f"ℹ️ The widget could not be sent ({type(e).__name__}: {e.message}); "
                    "this card uses regular buttons."))
    if c.message_id is None:
        msg = await bot.send_message(chat_id=c.chat_id, text=render_text(c), parse_mode=ParseMode.HTML,
                                     reply_markup=keyboard(c), link_preview_options=_preview(c))
        c.message_id = msg.message_id
    _remember(c)


async def show(bot, c: Card) -> bool:
    """Redraw the card in place: one edit at a time per card, each of the
    state at the moment it goes out. True when the card shows that state
    (or has no message to draw on); False when the edit failed. A widget's
    caption is edited on phase changes only (a caption edit marks the
    message edited, a widget data edit does not)."""
    async with c.lock:
        c.dirty = False
        if c.gone or c.message_id is None:
            return True
        try:
            if c.widget:
                ok = await tgx.edit_widget_data(bot, c.chat_id, c.message_id, widget_state(c))
                if ok and c.cap_phase != c.phase:
                    await tgx.edit_caption(bot, c.chat_id, c.message_id, render_text(c, short=True),
                                           reply_markup=_widget_markup(c))
                    c.cap_phase = c.phase
            else:
                ok = await edit_text(bot, chat_id=c.chat_id, message_id=c.message_id, text=render_text(c),
                                     reply_markup=keyboard(c), parse_mode=ParseMode.HTML,
                                     link_preview_options=_preview(c))
        except TelegramError as e:
            logger.warning("video card %s/%s: update failed (%s)", c.chat_id, c.message_id, e)
            return False
        if not ok:
            c.gone = True
            logger.warning("video card %s/%s: the message no longer exists", c.chat_id, c.message_id)
        return True


def kick(bot, c: Card) -> None:
    """A progress change: redrawn at most every _PROGRESS_INTERVAL_S; a
    change that arrives while a redraw is on its way is drawn after it.
    Callable from a synchronous progress callback."""
    c.dirty = True
    if c.pending is not None and not c.pending.done():
        return

    async def later() -> None:
        while c.dirty:
            await asyncio.sleep(max(0.0, c.shown_at + _PROGRESS_INTERVAL_S - time.monotonic()))
            c.shown_at = time.monotonic()
            await show(bot, c)
    c.pending = asyncio.ensure_future(later())


async def settle(bot, c: Card) -> None:
    """A state change (a tap, the probe, an outcome) drawn now. A sleeping
    progress redraw is dropped; one already on the wire finishes first.
    When the card cannot be edited (or, at the end, no longer exists), it
    is sent again as a new message and the card moves there: a choice to
    make or an outcome is never only logged."""
    if c.pending is not None and not c.pending.done() and not c.lock.locked():
        c.pending.cancel()
    c.shown_at = time.monotonic()
    if (c.message_id is not None and await show(bot, c)
            and not (c.gone and c.phase in ("done", "fail", "x"))):
        return
    async with c.lock:
        msg = await bot.send_message(
            chat_id=c.chat_id, text=render_text(c), parse_mode=ParseMode.HTML, reply_markup=keyboard(c),
            link_preview_options=(LinkPreviewOptions(is_disabled=True) if c.phase in ("done", "fail", "x")
                                  else _preview(c)))
        _cards.pop((c.chat_id, c.message_id), None)
        c.message_id, c.widget, c.gone = msg.message_id, False, False
        _remember(c)


async def to_native(bot, c: Card) -> None:
    """The 🔘 Regular buttons tap under a widget: the same card as a native
    message (an app without widgets shows only the caption and this button).
    The old widget is told it moved, so it shows no buttons that act."""
    old = c.message_id
    async with c.lock:
        msg = await bot.send_message(chat_id=c.chat_id, text=render_text(c), parse_mode=ParseMode.HTML,
                                     reply_markup=keyboard(c), link_preview_options=_preview(c))
        _cards.pop((c.chat_id, old), None)
        c.message_id, c.widget = msg.message_id, False
        _remember(c)
    try:
        await tgx.edit_widget_data(bot, c.chat_id, old, off_state("Moved to the regular card below.", c))
        await tgx.edit_caption(bot, c.chat_id, old, f"{_HEAD[c.phase]} <b>{_esc((c.title or c.url)[:120])}</b>\n"
                                                    "Moved to the regular card below.")
    except TelegramError as e:
        logger.info("could not mark the widget %s as moved (%s)", old, e)


# ---------------------------------------------------------------------------
# widget queries (Update.widget_query, Bot API 10.4; PTB 22.8 keeps it in api_kwargs)
# ---------------------------------------------------------------------------

def widget_query(update: object) -> dict | None:
    if isinstance(update, Update):
        q = update.api_kwargs.get("widget_query")
        if isinstance(q, dict) and q.get("id"):
            return q
    return None


class WidgetQueryHandler(BaseHandler):
    """Matches the updates that carry a ``widget_query`` (python-telegram-bot
    22.8 predates them, so they arrive as an Update with only api_kwargs)."""

    def check_update(self, update: object) -> dict | None:
        return widget_query(update)
