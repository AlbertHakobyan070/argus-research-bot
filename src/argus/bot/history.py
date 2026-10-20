"""Run history: /runs, /status, /cancel, /append, /continue, and /help."""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from telegram import InputFile, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from ..config import get_settings
from ..engines.urls import canonical_url
from ..graph.research import _source_key
from ..library import Library
from ..media import extract_urls
from . import runs
from .common import (
    HELP_TEXT, _HELP_MD_PATH, _STATUS_GLYPHS, _allowed, _bot_data_get,
    _fmt_duration, _get_library, _safe_send,
)
from .progress import describe

logger = logging.getLogger("argus.bot")


# ---------------------------------------------------------------------------
# /status, /cancel, /help
# ---------------------------------------------------------------------------


async def status_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Registry-backed status: the tracked runs (stage, elapsed, last event,
    or the gate they wait at), the console, recent run history."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    chat_id = update.effective_chat.id
    lines: list[str] = []
    entries = runs.tracked(chat_id)
    if entries:
        lines.append("*In flight:*")
        for rid, h in entries.items():
            if not h.live:
                lines.append(f"- `{rid}` · waiting at its {h.gate_kind} gate")
                continue
            elapsed = _fmt_duration(time.time() - h.started_at) or "0:00"
            lines.append(f"- `{rid}` · stage `{h.stage or 'start'}` · {elapsed}"
                         + (" · ⏹ cancelling" if h.control.drain_requested else ""))
            last = runs.bus.since(rid, runs.bus.last_seq(rid) - 1)
            if last:
                lines.append(f"    last: {describe(last[-1])}")
    lib = _get_library(ctx)
    recent: list[dict[str, Any]] = []
    if lib is not None:
        try:
            recent = await lib.list_runs(chat_id=chat_id, limit=5)
        except Exception:
            logger.exception("status: list_runs failed")
    if recent:
        lines.append("*Recent runs:*")
        for r in recent:
            glyph = _STATUS_GLYPHS.get(r["status"], "·")
            topic = (r["topic"] or "")[:48]
            lines.append(
                f"- {glyph} `{r['run_id']}` · {r['status']} · "
                f"{r['length']} · {topic}")
            if r.get("report_dir"):
                lines.append(f"    `{r['report_dir']}`")
    if not lines:
        lines.append("No runs recorded for this chat yet.")
    # WP-E2 sets webapp_public_url, webapp_ok and webapp_error (spec §2.7)
    url, ok, err = runs.console_url(), _bot_data_get(ctx, "webapp_ok"), _bot_data_get(ctx, "webapp_error")
    down = "down" + (f" — {err}" if err else "")
    if url:
        lines.append(f"Console: {url} ({'ok' if ok else down})")
    else:
        lines.append(f"Console: {down}" if ok is False else "Console: not running")
    text = "\n".join(lines)
    try:
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(text)


async def cancel_cmd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/cancel [run-id|all] — cancel in-flight run(s) for this chat.

    Without an argument: cancels when exactly one run is in flight,
    otherwise lists the candidates (never guesses which one you meant).
    A running drive drains — it finishes its current step, then stops
    with its checkpoint saved (/continue resumes it); a run waiting at a
    gate is cancelled at once. A run id that is not tracked (a restart
    empties the registry) is looked up in the library, as /continue does.
    """
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        await update.message.reply_text("Unauthorized.")
        return
    chat_id = update.effective_chat.id
    target = (ctx.args[0].lower().strip() if ctx.args else "")
    entries = runs.tracked(chat_id)
    if target and target != "all":
        picked = [rid for rid in entries if rid.startswith(target)]
        if not picked:
            lib = _get_library(ctx)
            run = await lib.resolve_run(chat_id, target) if lib is not None else None
            if run is None:
                await update.message.reply_text(
                    f"No run matches '{target}' — see /runs."
                    + (" In flight: " + ", ".join(entries) if entries else ""))
                return
            picked = [run["run_id"]]
    elif not entries:
        await update.message.reply_text("No in-flight runs for this chat.")
        return
    elif target == "all":
        picked = list(entries)
    elif len(entries) == 1:
        picked = list(entries)
    else:
        await update.message.reply_text(
            "Multiple runs in flight: " + ", ".join(entries)
            + "\nUse /cancel <run-id> or /cancel all.")
        return
    outcomes = [await runs.cancel_run(rid) for rid in picked]
    await _safe_send(ctx.application.bot, chat_id, "\n".join(o.text for o in outcomes))


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
    the research node ingests from disk (file:/// citations). Each carries
    its ``ref``, so ``_mark_appended`` can tell which ones were ingested.
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
                "summary": "", "source": "appended-asset", "ref": ref,
            })
        else:
            out.append({
                "kind": "search_result", "title": ref, "url": ref,
                "summary": "", "source": "appended-url", "ref": ref,
            })
    return out


async def _mark_appended(lib: Library, graph, cfg: dict, run_id: str,
                         pending: list[dict], appended: list[dict]) -> list[str]:
    """After the append pass: a pending ref is ``ingested`` only when research
    fetched its source; every other one is ``failed``. Returns one line per
    failed ref, naming it with research's error line for it (the fetch error
    and the rungs tried)."""
    snap = await graph.aget_state(cfg)
    values = (snap.values if snap else None) or {}
    fetched = {canonical_url(f.get("url") or "") for f in values.get("fetched") or []}
    errors = list(values.get("errors") or [])
    ingested, failed, lines = [], [], []
    by_ref = {s["ref"]: s for s in appended}
    for ref in (p["ref"] for p in pending):
        src = by_ref.get(ref)
        try:
            got = src is not None and _source_key(src) in fetched
        except ValueError:                       # research dropped it: a malformed URL
            got = False
        if got:
            ingested.append(ref)
            continue
        failed.append(ref)
        needle = (src or {}).get("local_path") or ref
        why = (next((e for e in reversed(errors) if needle in e), "not fetched") if src is not None
               else "the asset no longer exists")
        lines.append(f"• {ref}: {why}")
    for refs, status in ((ingested, "ingested"), (failed, "failed")):
        if refs:
            await lib.mark_sources(run_id, refs, status)
    return lines


async def pending_appends(lib: Library, run_id: str) -> tuple[list[dict], list[dict]]:
    """A run's pending /append rows and the sources they become (§2.4:
    pinned appends are always ingested — by the /continue fork or an Extend)."""
    pending = await lib.pending_sources(run_id)
    return pending, await _sources_from_refs(lib, pending)


def _append_key(src: dict) -> str:
    """A source as research dedupes it (``_source_key``: the canonical URL);
    a malformed URL, which research drops, as given."""
    try:
        return _source_key(src)
    except ValueError:
        return src.get("url") or ""


def merged_sources(sources: list[dict] | None, appended: list[dict]) -> list[dict]:
    """``sources`` plus each appended source it does not carry yet: an Extend
    or a fork drained or refused after its merge never duplicates them."""
    out = list(sources or [])
    keys = {_append_key(s) for s in out}
    for src in appended:
        if _append_key(src) not in keys:
            keys.add(_append_key(src))
            out.append(src)
    return out


def carried_appends(sources: list[dict] | None, pending: list[dict],
                    appended: list[dict]) -> tuple[list[dict], list[dict]]:
    """The pending /append rows, and their sources, that ``sources`` already
    carries — merged by an Extend or a fork whose pass was drained: resuming
    that pass ingests them."""
    keys = {_append_key(s) for s in sources or []}
    carried = [s for s in appended if _append_key(s) in keys]
    refs = {s["ref"] for s in carried}
    return [p for p in pending if p["ref"] in refs], carried


def appends_marker(lib: Library, graph, cfg: dict, run_id: str, pending: list[dict], appended: list[dict],
                   reply: Callable[[str], Awaitable[Any]]) -> Callable[[], Awaitable[None]]:
    """The callback the next report gate runs after a pass that ingested the
    appends: each one is marked ingested or failed, the failures named with
    ``reply``."""
    async def on_report() -> None:
        if not pending:
            return
        lines = await _mark_appended(lib, graph, cfg, run_id, pending, appended)
        if lines:
            await reply(f"⚠️ {len(lines)} appended source(s) could not be ingested "
                        "(marked failed — /append them again to retry):\n" + "\n".join(lines))
    return on_report


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
    """/continue <run-id> — re-attach a run waiting at a gate, resume a
    stopped (cancelled) one from its checkpoint, or extend a finished one
    (with appended sources if any, else a fresh search pass)."""
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
    cfg = {"configurable": {"thread_id": run["thread_id"]}}
    # A finished run's Extend fork ingests the pending appended sources;
    # after its report gate each one is marked ingested or failed.
    pending, appended = await pending_appends(lib, run_id)
    on_report = appends_marker(lib, graph, cfg, run_id, pending, appended, update.message.reply_text)
    out = await runs.continue_run(run, chat_id=chat_id, appended=appended, on_report=on_report)
    if out.text:
        await _safe_send(ctx.application.bot, chat_id, out.text)


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
