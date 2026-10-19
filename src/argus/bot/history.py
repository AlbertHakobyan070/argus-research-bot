"""Run history: /runs, /status, /cancel, /append, /continue, and /help."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from telegram import InputFile, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from ..config import get_settings
from ..engines import registry
from ..graph.state import DEFAULT_LENGTH
from ..library import Library
from ..media import extract_urls
from . import common
from .common import (
    HELP_TEXT, _HELP_MD_PATH, _STATUS_GLYPHS, _allowed, _bot_data_get,
    _chat_entries, _get_library, _safe_send, _set_run_status,
)
from .research import (
    _format_plan, _plan_keyboard, _resume_after_plan, _send_report_preview,
)

logger = logging.getLogger("argus.bot")


# ---------------------------------------------------------------------------
# /status, /cancel, /help
# ---------------------------------------------------------------------------


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

    Bookkeeping only: it forgets the run and marks it cancelled. A graph
    that is mid-stream is NOT interrupted — ``_resume_after_plan`` keeps
    running, then sets the status back to awaiting_report and sends a
    preview whose buttons no longer resolve (probe-verified). Runs that
    are paused at a gate are genuinely stopped.
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
        common._inflight.pop(rid, None)
        await _set_run_status(ctx, rid, "cancelled")
    await update.message.reply_text(
        f"❌ Cancelled {len(picked)} run(s): " + ", ".join(picked))


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
    """Map pending run_sources rows onto scout-style source dicts.

    URLs become search_result sources the research node fetches; asset refs
    (``asset:<id>``, e.g. vault transcripts) become local_path sources
    the research node ingests from disk (file:/// citations).
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
    if run_id in common._inflight and common._inflight[run_id].get("resuming"):
        await update.message.reply_text(
            f"Run {run_id} is already resuming — hang on.")
        return
    await registry.active_snapshot()
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
    if nxt in (("research",), ("fetcher",)):
        # Paused at the plan gate (possibly before a restart) — re-render.
        # ("fetcher",) accepted for pre-v3 checkpoints.
        common._inflight[run_id] = info
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
        common._inflight[run_id] = info
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
    common._inflight[run_id] = info
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


def register(app: Application) -> None:
    app.add_handler(CommandHandler("runs", runs_cmd))
    app.add_handler(CommandHandler("append", append_cmd))
    app.add_handler(CommandHandler("continue", continue_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("cancel", cancel_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
