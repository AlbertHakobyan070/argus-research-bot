"""Research commands and buttons: /research and /ask, the plan-gate and
report-gate callbacks, the HITL replies — each calls ``runs.py`` and relays
its outcome — and the renderers the driver sends: the plan card, its
keyboard, the report preview with its quality line."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from ..config import get_settings
from ..graph.report_builder_helpers import render_degradation_notes
from ..graph.state import DEFAULT_LENGTH, VALID_LENGTHS
from . import common, runs
from .common import (
    _LENGTH_LABELS, TG_LIMIT, _allowed, _html_escape_for_tg, _md_escape, _parse_length, _safe_send, _tg_cut,
    _tg_len,
)

logger = logging.getLogger("argus.bot")

_RECENCY_LABELS = {None: "auto", "d": "past day", "w": "past week", "m": "past month", "y": "past year",
                   "any": "any date"}
_KNOBS = {"len": "length", "dep": "depth", "prof": "profile", "rec": "recency"}
_EXCERPT_CAP = 3000      # raw report chars in the preview; the full file follows on Send
_MAX_NOTES = 8           # degradation notes in the preview; the report has them all
_NOTE_ROOM = 200         # a gate edit appends a note (runs._REPLY_PROMPTS, "✅ approved from console", …)
_PLAN_BUDGET = TG_LIMIT - _NOTE_ROOM
_PREVIEW_BUDGET = TG_LIMIT - _NOTE_ROOM
_HEAD_CHARS = 300        # one header fact (a profile problem, a rebuild error): the header always fits
# The plan card's shortening steps, in order, until it fits: (limit, cap) caps a text; (limit, None)
# drops a list's items from its end ("…and N more").
_PLAN_STEPS = (("sq_chars", 200), ("intent_chars", 100), ("summary", 300),
               ("sq_chars", 140), ("intent_chars", 70), ("summary", 150),
               ("sources", None), ("intents", None), ("sq", None))


# ---------------------------------------------------------------------------
# commands, buttons, replies
# ---------------------------------------------------------------------------

async def _relay(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int, out: runs.Outcome) -> None:
    if out.text:
        await _safe_send(ctx.application.bot, chat_id, out.text)


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
    await _relay(ctx, chat_id, await runs.start_run(chat_id=chat_id, user_id=update.effective_user.id,
                                                    topic=topic, length=length))


async def ask_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    if not ctx.args:
        await update.message.reply_text("Usage: /ask <question>")
        return
    chat_id = update.effective_chat.id
    await _relay(ctx, chat_id, await runs.start_run(chat_id=chat_id, user_id=update.effective_user.id,
                                                    topic=" ".join(ctx.args), quick=True))


async def handle_pending_reply(update: Update,
                               ctx: ContextTypes.DEFAULT_TYPE,
                               chat_id: int, text: str) -> bool:
    """The HITL branch of ``app.on_text``: consume this chat's pending
    plan-edit / revise-feedback reply. Returns False when none is
    pending, so the text goes on to the media branch."""
    pending = common._pending_reply.get(chat_id)
    if not pending:
        return False
    if not text.strip():
        await update.message.reply_text("Please send some text.")
        return True
    common._pending_reply.pop(chat_id, None)
    kind, run_id = pending
    if kind == "plan_edit":
        out = await runs.edit_plan(run_id, text.strip())
    else:
        out = await runs.report_action(run_id, "revise", feedback=text.strip())
    if out.code != "started":                     # the reply is consumed: say it was not used
        out = runs.Outcome(out.code, f"{out.text}\n(your reply was not used — tap Edit/Revise and send it again)",
                           out.run_id)
    await _relay(ctx, chat_id, out)
    return True


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                          ns: str, action: str, run_id: str) -> None:
    """Run-scoped buttons (``len`` / ``dep`` / ``prof`` / ``rec`` knobs,
    ``plan:*``, ``report:*``), routed here by ``app.on_callback`` for every
    namespace not handled before the run lookup. The tapped message becomes
    the gate message of a run the action attaches (one untracked after a
    restart) when its button belongs to that gate: the knobs and ``plan:*``
    to the plan gate, ``report:*`` to the report gate."""
    q = update.callback_query
    mid = q.message.message_id if q.message is not None else None
    if ns in _KNOBS:
        out = await runs.set_plan_choice(run_id, _KNOBS[ns], action, message_id=mid)
    elif ns == "plan" and action == "approve":
        out = await runs.approve_plan(run_id, message_id=mid)
    elif ns == "plan" and action == "edit":
        out = await runs.await_reply(run_id, "plan_edit", message_id=mid)
    elif ns in ("plan", "report") and action == "cancel":
        out = await runs.cancel_run(run_id, message_id=mid, card=runs.PLAN if ns == "plan" else runs.REPORT)
    elif ns == "report" and action == "revise":
        out = await runs.await_reply(run_id, "revise_feedback", message_id=mid)
    elif ns == "report":
        out = await runs.report_action(run_id, action, message_id=mid)
    else:
        return
    if out.code == "not_found":
        await q.edit_message_text(out.text)
    elif out.code != "cancelled":                 # a cancelled gate's own message already says so
        await _relay(ctx, update.effective_chat.id, out)


# ---------------------------------------------------------------------------
# renderers (the driver sends these)
# ---------------------------------------------------------------------------

def _recency_text(p: dict) -> str:
    recency, window = p.get("recency"), p.get("recency_window")
    if recency == "any":
        return "any date (no filter)"
    shown = (f"{_RECENCY_LABELS.get(window['window'], window['window'])} (since {window['start_date']})"
             if window else "no filter")
    return f"auto — {shown}" if recency is None else shown


def _clip(text: str, n: int | None) -> str:
    text = str(text)
    return text if n is None or len(text) <= n else text[:n - 1].rstrip() + "…"


def _md_inside(text: str, end: str) -> str:
    """Model or scraped text placed INSIDE a legacy-Markdown entity: Telegram
    honours no escape there and ends the entity at its first ``end``, so that
    character is dropped (the 05:18:22 card: ``\\_`` inside ``_…_`` closed the
    italic early, and the next ``_`` opened one that never closed)."""
    return str(text).replace(end, "")


def _more(n: int) -> list[str]:
    return [f"  _…and {n} more_"] if n > 0 else []


def _format_plan(p: dict) -> str:
    """The plan card, from ``runs`` plan payload (keys optional).

    Grounded gate: ``sources`` are the REAL results the scout found via
    live search (exa/ddgs/arxiv/github) before this gate. The brief
    never contains URLs — hallucinated links are structurally
    impossible at this stage.

    The card always fits ``_PLAN_BUDGET`` (Telegram's limit less room for a
    gate edit's note) and its Markdown always parses: model text sits outside
    entities, escaped (``_md_escape``), or inside one with its end character
    dropped (``_md_inside``); a URL's parentheses are percent-encoded. A card
    too long is shortened step by step until it fits (``_PLAN_STEPS``): the
    sub-question texts, then the intent texts and the summary (twice, each
    time shorter), then the lists from their end — sources, intents,
    sub-questions last — each cut list ending "…and N more". The header, the
    knobs line and the buttons (``_plan_keyboard``) always survive.
    """
    plan = p.get("plan") or {}
    length = p.get("length") or DEFAULT_LENGTH
    head = f"📋 *Research plan*  ·  mode: {_LENGTH_LABELS.get(length, length)}"
    if p.get("depth"):
        head += f"  ·  depth: {p['depth']}" + (" (auto)" if p.get("depth_auto") else "")
    lines = [head]
    facts = []
    if p.get("archetype"):
        facts.append(f"Archetype: {_md_escape(_clip(p['archetype'], _HEAD_CHARS))}")
    if "recency" in p:
        facts.append(f"Recency: {_recency_text(p)}")
    if facts:
        lines.append(" · ".join(facts))
    facts = []
    if p.get("profile"):
        facts.append(f"Engines: {_md_escape(_clip(p['profile'], _HEAD_CHARS))}"
                     + (f" — ⚠ {_md_escape(_clip(p['profile_problem'], _HEAD_CHARS))}"
                        if p.get("profile_problem") else ""))
    if "exa_enabled" in p:
        est = p.get("exa_estimate_usd")
        facts.append("Exa: " + ((f"≈ ${est:.3f}" if est is not None else "n/a") if p["exa_enabled"] else "off"))
    if facts:
        lines.append(" · ".join(facts))
    if p.get("error"):
        lines.append(f"⚠️ {_md_escape(_clip(p['error'], _HEAD_CHARS))}")
    lines.append("")
    summary = plan.get("summary", "")
    sub_qs = plan.get("sub_questions") or []
    intents = plan.get("planned_sources") or []
    sources = p.get("sources")

    def render(lim: dict) -> str:
        out = list(lines)
        if summary:
            out += [_md_escape(_clip(summary, lim["summary"])), ""]
        if sub_qs:
            out.append("*Sub-questions:*")
            out += [f"- {_md_escape(_clip(q, lim['sq_chars']))}" for q in sub_qs[:lim["sq"]]]
            out += _more(len(sub_qs) - lim["sq"])
            out.append("")
        if intents:
            out.append(f"*Search intents ({len(intents)}):*")
            for s in intents[:lim["intents"]]:
                kind = _md_inside(_clip(s.get("kind", "search"), _HEAD_CHARS), "`")
                q = s.get("query")
                intent = _md_escape(_clip(q, lim["intent_chars"])) if q else "_live search_"
                out.append(f"- `{kind}` — {intent}")
            out += _more(len(intents) - lim["intents"])
            out.append("")
        if sources is not None:
            if sources:
                out.append(f"*Found sources ({len(sources)}, live search):*")
                for s in sources[:lim["sources"]]:
                    title = (s.get("title") or "").strip()
                    url = s.get("url") or ""
                    if not title:
                        title = url.split("//", 1)[-1][:60] or "(untitled)"
                    url = url.replace("(", "%28").replace(")", "%29")     # a URL ends at its first ")"
                    out.append(f"- [{_md_inside(title[:70], ']')}]({url})")
                out += _more(len(sources) - lim["sources"])
            else:
                out.append(
                    "⚠️ *Live search found no sources.* Approving will fetch "
                    "nothing — Edit the plan or Cancel.")
        return "\n".join(out)

    lim = {"summary": None, "sq": len(sub_qs), "sq_chars": None, "intents": min(len(intents), 14),
           "intent_chars": None, "sources": min(len(sources or []), 12)}
    text = render(lim)
    for key, value in _PLAN_STEPS:
        if _tg_len(text) <= _PLAN_BUDGET:
            break
        if value is not None:                     # a text cap
            lim[key] = value
            text = render(lim)
            continue
        while lim[key] > 0 and _tg_len(text) > _PLAN_BUDGET:   # a list, from its end
            lim[key] -= 1
            text = render(lim)
    return text


def _plan_keyboard(p: dict) -> InlineKeyboardMarkup:
    """The plan-gate keyboard: lengths (tldr→lecture), depths and Auto (the
    length's own), the engines profile and recency (each tap cycles),
    Approve / Edit / Cancel. The ✅ marks the current length and depth (Auto
    while the depth follows the length). Callback data is run-scoped
    (``len:long:<run8>``) so an old card never acts on a newer run."""
    rid = p.get("run_id") or ""
    sfx = f":{rid}" if rid else ""
    length = p.get("length") or DEFAULT_LENGTH
    auto = bool(p.get("depth_auto"))

    def button(label: str, data: str, on: bool = False) -> InlineKeyboardButton:
        return InlineKeyboardButton(label + (" ✅" if on else ""), callback_data=data + sfx)

    profile = (p.get("profile") or "engines") + (" ⚠" if p.get("profile_problem") else "")
    return InlineKeyboardMarkup([
        [button(_LENGTH_LABELS[m], f"len:{m}", m == length) for m in ("tldr", "short", "medium")],
        [button(_LENGTH_LABELS[m], f"len:{m}", m == length) for m in ("long", "lecture")],
        [*(button(d.capitalize(), f"dep:{d}", not auto and d == p.get("depth")) for d in runs.DEPTHS),
         button("Auto", "dep:auto", auto)],
        [button(f"⚙ {profile}", "prof:next"),
         button(f"🕒 {_RECENCY_LABELS.get(p.get('recency'), p.get('recency'))}", "rec:next")],
        [button("✅ Approve", "plan:approve"), button("✏️ Edit", "plan:edit"), button("❌ Cancel", "plan:cancel")],
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


def _quality_lines(values: dict, md_text: str) -> list[str]:
    """The preview's quality line, the report's "⚠" banners (its title block:
    the lines before the first ``## `` heading) and this round's degradation
    notes (as the report shows them)."""
    pv = values.get("panel_verdict") or {}
    cut = sum(1 for f in values.get("findings") or [] if f.get("status") == "cut")
    lines = [f"🧪 Quality: judged {int(pv.get('judged') or 0)}/{int(pv.get('total') or 0)} · {cut} cut"]
    for line in md_text.splitlines():
        if line.startswith("## "):
            break
        if line.startswith("> ⚠ "):
            lines.append(line[2:])
    rnd = 1 + int(values.get("revise_rounds") or 0) + int(values.get("extend_rounds") or 0)
    notes = render_degradation_notes(list(values.get("errors") or []), rnd).splitlines()[2:]
    if notes:
        lines += ["Degradation notes:", *notes[:_MAX_NOTES]]
        if len(notes) > _MAX_NOTES:
            lines.append(f"…and {len(notes) - _MAX_NOTES} more in the report")
    return lines


async def _send_report_preview(bot, chat_id: int, run_id: str, values: dict) -> tuple[Any, str, str | None]:
    """Send the report preview + keyboard for a run at its report gate;
    returns ``(message | None, text, parse mode)`` for later edits."""
    if not (values.get("report_paths") or {}).get("md"):
        text = "(no report_paths found)"
        return await _safe_send(bot, chat_id, text, parse_mode=None), text, None
    html_body, plain_body = _preview_bodies(values)
    # HTML primary (tolerates arbitrary content), plain-text fallback —
    # legacy-markdown crashed on ``_ * [ &`` in excerpts (2026-07-08 bug).
    try:
        msg = await bot.send_message(chat_id=chat_id, text=html_body,
                                     reply_markup=_report_keyboard(run_id), parse_mode=ParseMode.HTML)
        return msg, html_body, ParseMode.HTML
    except Exception as send_err:
        logger.warning(
            "report preview HTML send failed (%s); falling back to plain text",
            send_err,
        )
    try:
        msg = await bot.send_message(chat_id=chat_id, text=plain_body, reply_markup=_report_keyboard(run_id))
        return msg, plain_body, None
    except Exception as plain_err:
        logger.exception("report preview plain-text send also failed")
        text = f"⚠️ report preview unavailable: {plain_err}"
        # the gate's buttons travel with it: the run never rests at its gate without them
        return await _safe_send(bot, chat_id, text, parse_mode=None, reply_markup=_report_keyboard(run_id)), text, None


def _preview_bodies(values: dict) -> tuple[str, str]:
    """The report preview as ``(HTML body, plain body)`` (also the text of a
    preview message attached from a tap, re-rendered)."""
    paths = values.get("report_paths") or {}
    length = values.get("length") or DEFAULT_LENGTH
    if not paths.get("md"):
        text = "(no report_paths found)"
        return text, text
    md = Path(paths["md"])
    try:
        full_text = md.read_text(encoding="utf-8", errors="replace")
        tail_note = ""
    except OSError as e:
        full_text, tail_note = "", f"⚠️ {md.name} could not be read: {e}"
    # Most of Telegram's 4096-char budget goes to the excerpt; the header,
    # the <pre> wrapper and the escaping expansion need headroom.
    excerpt = full_text[:_EXCERPT_CAP]
    tail_note = tail_note or (f"_(first {len(excerpt)} of {len(full_text)} chars below)_"
                              if len(full_text) > _EXCERPT_CAP else "_(full report below)_")
    text = (
        f"📝 *Report preview*  ·  {_LENGTH_LABELS.get(length, '?')}\n\n"
        + "\n".join(_quality_lines(values, full_text)) + "\n\n"
        f"Folder: `{_md_escape(paths.get('folder', '?'))}`\n"
        f"Markdown: `{_md_escape(md.name)}`  "
        + ("· PDF: `report.pdf`" if paths.get("pdf") else "· PDF: _failed_")
        + f"\n\n{tail_note}"
    )

    # HTML-escaping expands text (& → &amp;), so shrink the excerpt until
    # the whole message fits Telegram's cap (less a gate note); a head too
    # long on its own (long degradation notes) is cut before it is escaped.
    def _fit(exc: str) -> str:
        return _html_escape_for_tg(text) + (
            ("\n\n<pre>" + _html_escape_for_tg(exc) + "</pre>")
            if exc else "")
    html_body = _fit(excerpt)
    while excerpt and _tg_len(html_body) > _PREVIEW_BUDGET:
        excerpt = excerpt[:len(excerpt) - 200]
        html_body = _fit(excerpt)
    if _tg_len(html_body) > _PREVIEW_BUDGET:
        # the head alone is too long (the excerpt is gone): keep its longest cut whose ESCAPED
        # (sent) length fits — escaping grows "&" fivefold, so the raw overflow is no measure.
        # A binary search: it always ends, and the cut keeps the head's start (its title line).
        lo, hi = 0, _tg_len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _tg_len(_html_escape_for_tg(_tg_cut(text, mid))) <= _PREVIEW_BUDGET:
                lo = mid
            else:
                hi = mid - 1
        text = _tg_cut(text, lo)
        html_body = _fit(excerpt)
    plain_body = _tg_cut(text + ("\n\n```\n" + excerpt + "\n```" if excerpt else ""), _PREVIEW_BUDGET)
    return html_body, plain_body


def register(app: Application) -> None:
    app.add_handler(CommandHandler("research", research_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
