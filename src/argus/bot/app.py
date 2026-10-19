"""Application wiring: the handler modules, the free-text and callback
dispatchers, startup and shutdown (saver, library, graphs, orphaned-run
reconciliation), the hook points for later work packages, and main()."""
from __future__ import annotations

import asyncio
import logging
import os
from types import SimpleNamespace

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CallbackQueryHandler, ContextTypes, MessageHandler, filters,
)

from ..cache_cleanup import cleanup_argus_ytt_cache
from ..config import get_settings
from ..graph import async_sqlite_saver_cm, build_graph, quick_answer_graph
from ..library import Library
from ..media import resolve_ffmpeg
from . import delete, history, media, research
from .common import (
    _TG_CONNECT_TIMEOUT_S, _TG_MEDIA_WRITE_TIMEOUT_S, _TG_POOL_TIMEOUT_S,
    _TG_READ_TIMEOUT_S, _TG_WRITE_TIMEOUT_S, _allowed, _set_run_status,
)

logger = logging.getLogger("argus.bot")


# Hook points for later work packages. Each HANDLER_MODULES entry has
# register(app), called in list order by build_application. Each POST_*
# hook is ``async def hook(app: Application) -> None``, awaited in list
# order with exceptions propagating: POST_INIT_HOOKS at the end of
# _post_init; POST_STOP_HOOKS from post_stop, after Application.stop()
# has awaited the app.create_task tasks; POST_SHUTDOWN_HOOKS at the start
# of _post_shutdown, before the library and checkpoint saver close.
HANDLER_MODULES = [research, media, history, delete]
POST_INIT_HOOKS: list = []
POST_STOP_HOOKS: list = []
POST_SHUTDOWN_HOOKS: list = []


async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Free-text router. Order: (1) a pending HITL reply (plan-edit /
    revise feedback), (2) pasted media links → action keyboard, else
    ignore."""
    s = get_settings()
    if not _allowed(s, update.effective_user.id):
        return  # silently ignore strangers
    text = (update.message.text or "") if update.message else ""
    chat_id = update.effective_chat.id
    if await research.handle_pending_reply(update, ctx, chat_id, text):
        return
    await media.handle_text(update, ctx, chat_id, text)


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
        await media.handle_callback(update, ctx, ns, action, run_id)
        return

    # Delete-browser buttons (del:cat/tgl/pg/ok/yes/no/cancel).
    if ns == "del":
        await delete.handle_callback(update, ctx, ns, action, run_id)
        return

    # Transcript library buttons (t:send:<asset_id>).
    if ns == "t" and action == "send":
        await media.handle_callback(update, ctx, ns, action, run_id)
        return

    # Everything else is run-scoped: the run lookup and its not-in-flight
    # notice, then len:* / plan:* / report:* (other namespaces stop at the
    # lookup or fall through doing nothing, as before the split).
    await research.handle_callback(update, ctx, ns, action, run_id)


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
    # Surface the downloader version up front. When a platform changes its
    # API this is the first thing to check, and a missing yt-dlp should
    # not first show up as a failed download half an hour later.
    try:
        from yt_dlp.version import __version__ as ytdlp_version
        logger.info("yt-dlp: %s (if a platform breaks, update it: "
                    "uv pip install -U --pre yt-dlp)", ytdlp_version)
    except Exception:
        logger.error("yt-dlp is NOT importable — every media download will "
                     "fail. Install it into this venv: uv pip install yt-dlp")
    loop_name = type(asyncio.get_running_loop()).__name__
    if "Proactor" not in loop_name and os.name == "nt":
        # asyncio subprocesses (Phase 2 media engine) need the Proactor
        # loop on Windows. PTB does not override the policy; this fires
        # only if some future import does.
        logger.warning("event loop is %s, not Proactor — asyncio "
                       "subprocesses will not work on Windows", loop_name)
    logger.info("post_init: saver + library + graphs ready (loop=%s, "
                "library=%s)", loop_name, s.library_db)
    # Reconcile runs orphaned by the previous process (crash / restart):
    # their in-memory _inflight is gone, so their inline buttons are dead.
    # Recover what we can from checkpoints and tell each user.
    try:
        await _reconcile_orphaned_runs(app)
    except Exception:
        logger.exception("orphaned-run reconciliation failed")
    for hook in POST_INIT_HOOKS:
        await hook(app)


async def _reconcile_orphaned_runs(app: Application) -> None:
    """A restart wipes _inflight, so any run left non-terminal is orphaned
    (its plan/report buttons no longer resolve). For each, inspect the
    checkpoint: if it's paused at a gate, mark it awaiting_* and DM the
    user how to resume; otherwise mark it error. Bounded + best-effort."""
    lib: Library | None = app.bot_data.get("library")
    graph = app.bot_data.get("graph")
    if lib is None or graph is None:
        return
    orphans = await lib.list_runs_by_status(
        ("planning", "awaiting_plan", "running", "awaiting_report"), limit=25)
    if not orphans:
        return
    logger.info("reconciling %d orphaned run(s) from a prior process",
                len(orphans))
    for run in orphans:
        run_id = run["run_id"]
        cfg = {"configurable": {"thread_id": run["thread_id"]}}
        try:
            snap = await graph.aget_state(cfg)
        except Exception:
            logger.exception("reconcile: aget_state failed for %s", run_id)
            snap = None
        nxt = tuple(snap.next) if (snap and snap.next) else ()
        topic = (run.get("topic") or "")[:48]
        if nxt in (("research",), ("fetcher",)):
            await _set_run_status(app_ctx(app), run_id, "awaiting_plan")
            msg = (f"♻️ Run `{run_id}` ({topic}) was interrupted by a bot "
                   f"restart at the plan step.\nSend /continue {run_id} to "
                   "review the plan and resume.")
        elif nxt == ("deliver",):
            await _set_run_status(app_ctx(app), run_id, "awaiting_report")
            msg = (f"♻️ Run `{run_id}` ({topic}) was interrupted by a bot "
                   f"restart at the report preview.\nSend /continue {run_id} "
                   "to see the report.")
        else:
            await _set_run_status(app_ctx(app), run_id, "error")
            msg = (f"⚠️ Run `{run_id}` ({topic}) was interrupted by a bot "
                   "restart and can't be resumed from where it stopped.\n"
                   "Please /research it again.")
        try:
            await app.bot.send_message(chat_id=run["chat_id"], text=msg,
                                       parse_mode=ParseMode.MARKDOWN)
        except Exception:
            logger.warning("reconcile: could not notify chat %s about %s",
                           run.get("chat_id"), run_id)


def app_ctx(app: Application):
    """Minimal shim exposing ``.application`` so the registry helpers
    (_set_run_status / _get_library) work from post_init, where there's
    no real Context object yet."""
    return SimpleNamespace(application=app)


async def _post_stop(app: Application) -> None:
    for hook in POST_STOP_HOOKS:
        await hook(app)


async def _post_shutdown(app: Application) -> None:
    for hook in POST_SHUTDOWN_HOOKS:
        await hook(app)
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
    # Timeouts. PTB's defaults (connect/read/write 5s, pool 1s,
    # media_write 20s) are sized for text messages and silently break
    # file delivery: a 40 MB video cannot leave a home uplink in 20s, so
    # send_video raised TimedOut, the send_document fallback timed out
    # too, and the user got "✅ Saved <path>" with no file ever arriving.
    # media_write_timeout covers the upload body; read_timeout covers
    # Telegram transcoding the video server-side before it answers.
    app = (
        Application.builder()
        .token(s.telegram_bot_token)
        .connect_timeout(_TG_CONNECT_TIMEOUT_S)
        .read_timeout(_TG_READ_TIMEOUT_S)
        .write_timeout(_TG_WRITE_TIMEOUT_S)
        .pool_timeout(_TG_POOL_TIMEOUT_S)
        .media_write_timeout(_TG_MEDIA_WRITE_TIMEOUT_S)
        .post_init(_post_init)
        .post_stop(_post_stop)
        .post_shutdown(_post_shutdown)
        .build()
    )
    for mod in HANDLER_MODULES:
        mod.register(app)
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
    # Age out the legacy tempdir transcript cache (current callers write
    # to the vault instead). Best-effort, exception-tolerant.
    _removed = cleanup_argus_ytt_cache()
    if _removed:
        logger.info(
            "argus_ytt cache cleanup: removed %d stale deliverable(s)", _removed,
        )
    app.run_polling(allowed_updates=["message", "callback_query"])
