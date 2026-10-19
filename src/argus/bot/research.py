"""Research flow: /research and /ask, the plan and report keyboards and
formatters, the HITL replies and callbacks, and the resume and delivery
paths."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from langgraph.types import Command
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes

from ..config import get_settings
from ..engines import registry
from ..graph.run_config import RunConfig, ensure_in_checkpoint
from ..graph.state import DEFAULT_LENGTH, VALID_LENGTHS
from ..library import mirror_run_md, new_run_id
from . import common
from .common import (
    _LENGTH_LABELS, _allowed, _bot_data_get, _get_library,
    _html_escape_for_tg, _md_escape, _parse_length, _safe_edit_cb,
    _safe_send, _send_md_doc, _set_run_status, _stream_progress,
)

logger = logging.getLogger("argus.bot")


async def _ensure_run_config(info: dict) -> RunConfig:
    """First step of every resume entry, before anything changes (spec §2.3,
    §2.4): resolve the engines snapshot, then make sure the run's checkpoint
    carries its RunConfig (a pre-v4 checkpoint gets the legacy defaults,
    recorded in ``errors``). Raises — leaving the run as it was — when the
    engines cannot be resolved or the checkpoint cannot get a config."""
    engines = await registry.active_snapshot()
    return await ensure_in_checkpoint(info["graph"], info["cfg"], engines=engines)


async def _approve_run_config(ctx: ContextTypes.DEFAULT_TYPE, info: dict) -> None:
    """Plan approve, before anything else changes: the run's config rebuilt
    for the chosen length — ``rebuild``, so console overrides survive; depth
    follows the length (the plan gate has no depth control yet) — written
    with that length to the checkpoint, then to the library row."""
    engines = await registry.active_snapshot()
    rc = await ensure_in_checkpoint(info["graph"], info["cfg"], engines=engines)
    rc = rc.rebuild(length=info["length"], engines=engines)
    await info["graph"].aupdate_state(info["cfg"], {"length": rc.length,
                                                    "run_config": rc.model_dump()})
    lib = _get_library(ctx)
    if lib is not None and not await lib.set_run_config(info["run_id"], rc.model_dump()):
        logger.warning("run %s has no library row; its run_config is in the checkpoint only",
                       info["run_id"])


# ---------------------------------------------------------------------------
# /research
# ---------------------------------------------------------------------------


def _format_plan(plan: dict, length: str = DEFAULT_LENGTH,
                 sources: list[dict] | None = None) -> str:
    """Render the plan-approval preview.

    Grounded gate: ``sources`` are the REAL results the scout found via
    live search (exa/ddgs/arxiv/github) before this gate. The brief
    never contains URLs — hallucinated links are structurally
    impossible at this stage.
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
    engines = await registry.active_snapshot()
    rc = RunConfig.from_preset(length=length, engines=engines, origin="telegram")

    # v2: every run gets its own id + checkpoint thread. Multiple runs
    # per chat can coexist, and a run can be re-attached after a restart.
    run_id = new_run_id()
    thread_id = f"tg:{chat_id}:{run_id}"
    lib = _get_library(ctx)
    if lib is not None:
        try:
            await lib.create_run(run_id=run_id, thread_id=thread_id,
                                 chat_id=chat_id, topic=topic, length=length,
                                 config_json=rc.model_dump_json(), depth=rc.depth,
                                 origin=rc.origin)
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
        "run_config": rc.model_dump(),
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
    common._inflight[run_id] = info
    try:
        # Stream until first interrupt (the plan-approval gate).
        async for ev in graph.astream(state_in, config=cfg,
                                      stream_mode="updates"):
            for node_name, delta in ev.items():
                info["stage"] = node_name
                if node_name == "intake":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🎯 Mode detected: _{delta.get('mode','?')}_", progress)
                elif node_name == "brief":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        "📋 Brief drafted — scouting live sources…", progress)
                elif node_name == "scout":
                    n = len(delta.get("sources") or [])
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🔍 Scout found {n} candidate source(s) — "
                        "awaiting your approval.", progress)
                elif node_name == "research":
                    n = len(delta.get("fetched") or [])
                    ne = len(delta.get("evidence") or [])
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"📚 Read {n} sources → {ne} evidence notes.", progress)
                elif node_name == "compose":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🧠 Composing ({_LENGTH_LABELS.get(length,'?')})…",
                        progress)
                elif node_name == "panel":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🔬 Panel: {delta.get('review_verdict',{}).get('verdict','?')}",
                        progress)

        # Snapshot current state from the checkpointer.
        snap = await graph.aget_state(cfg)
        cur = snap.values if snap else {}

        # First HITL = plan approval — grounded: the scout already
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
                text="(brief produced no plan; aborting.)",
            )
            common._inflight.pop(run_id, None)
            await _set_run_status(ctx, run_id, "error")
    except Exception as e:
        common._inflight.pop(run_id, None)
        await _set_run_status(ctx, run_id, "error")
        await ctx.application.bot.send_message(chat_id=chat_id, text=f"⚠️ Research failed: {e}")
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
    engines = await registry.active_snapshot()
    rc = RunConfig.from_preset(length=DEFAULT_LENGTH, engines=engines, origin="telegram")
    await ctx.application.bot.send_chat_action(chat_id, ChatAction.TYPING)

    run_id = new_run_id()
    thread_id = f"tg:{chat_id}:{run_id}"
    lib = _get_library(ctx)
    if lib is not None:
        try:
            await lib.create_run(run_id=run_id, thread_id=thread_id,
                                 chat_id=chat_id, topic=question, length=rc.length,
                                 mode="quick", status="running",
                                 config_json=rc.model_dump_json(), depth=rc.depth,
                                 origin=rc.origin)
        except Exception:
            logger.exception("create_run failed; continuing without registry row")

    cfg = {"configurable": {"thread_id": thread_id}}
    state_in: dict[str, Any] = {
        "thread_id": thread_id,
        "user_id": update.effective_user.id,
        "user_request": question,
        "length": DEFAULT_LENGTH,
        "run_config": rc.model_dump(),
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


async def handle_pending_reply(update: Update,
                               ctx: ContextTypes.DEFAULT_TYPE,
                               chat_id: int, text: str) -> bool:
    """The HITL branch of ``app.on_text``: consume this chat's pending
    plan-edit / revise-feedback reply. Returns False when none is
    pending, so the text goes on to the media branch."""
    # (1) pending HITL reply
    pending = common._pending_reply.get(chat_id)
    if pending:
        kind, run_id = pending
        info = common._inflight.get(run_id)
        if info is None:
            common._pending_reply.pop(chat_id, None)
            await update.message.reply_text(
                "(that run is no longer active — reply ignored)")
            return True
        if not text.strip():
            await update.message.reply_text("Please send some text.")
            return True
        common._pending_reply.pop(chat_id, None)
        if kind == "plan_edit":
            await _replan_after_edit(ctx, info, text.strip())
        elif kind == "revise_feedback":
            await _revise_after_feedback(ctx, info, text.strip())
        return True
    return False


async def _replan_after_edit(ctx: ContextTypes.DEFAULT_TYPE, info: dict,
                             feedback: str) -> None:
    """plan-edit: push the user's feedback into the checkpoint as if
    intake just ran, then re-drive brief→scout to the grounded plan
    gate and re-render the plan message.

    NB: ``state["sources"]`` is not cleared here, so scout re-seeds the
    previous wave's hits first (see the note in ``scout_node``)."""
    await _ensure_run_config(info)
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
        # Fork positioned after intake → next node is brief. brief
        # reads plan_feedback + the previous brief and revises.
        await graph.aupdate_state(cfg, {"plan_feedback": feedback},
                                  as_node="intake")
        async for ev in graph.astream(None, config=cfg,
                                      stream_mode="updates"):
            for node_name, _delta in ev.items():
                info["stage"] = node_name
                if node_name == "scout":
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
    deliver passes through → revise_prep → compose → … → new preview."""
    await _ensure_run_config(info)
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


async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                          ns: str, action: str, run_id: str) -> None:
    """Run-scoped buttons (``len:*`` / ``plan:*`` / ``report:*``), routed
    here by ``app.on_callback`` for every namespace not handled before
    the run lookup."""
    q = update.callback_query
    info = common._inflight.get(run_id)
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
            # Before anything changes: a failure here (engines unresolvable,
            # no rebuildable config) leaves the run at its gate, keyboard on.
            await _approve_run_config(ctx, info)
            info["plan_approved"] = True
            await _safe_edit_cb(
                q, (info.get("plan_text") or q.message.text or "")
                + "\n\n_✅ Approved — continuing._")
            await _set_run_status(ctx, run_id, "running")
            await _resume_after_plan(ctx, info)
        elif action == "edit":
            info["plan_edit"] = True
            common._pending_reply[info["chat_id"]] = ("plan_edit", run_id)
            base = info.get("plan_text") or (q.message.text or "")
            await _safe_edit_cb(
                q, base + "\n\n_✏️ Reply with your changes and I'll redraft "
                          "the plan (or tap ✅ Approve to keep this one)._",
                reply_markup=_plan_keyboard(
                    default_length=info.get("length", DEFAULT_LENGTH),
                    run_id=run_id),
            )
        elif action == "cancel":
            common._inflight.pop(run_id, None)
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
            # extend_prep → research → … → report_builder and shows the
            # next preview.
            graph = info["graph"]
            cfg = info["cfg"]
            await _ensure_run_config(info)          # before the flag and the status change
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
            common._pending_reply[info["chat_id"]] = ("revise_feedback", run_id)
            await _safe_edit_cb(
                q, (q.message.text or "")
                + "\n\n_🔁 Reply with what to change and I'll revise the "
                  "report._")
        elif action == "cancel":
            common._inflight.pop(run_id, None)
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

    The chosen length is already in the checkpoint: Approve writes it
    together with the rebuilt run_config (``_approve_run_config``); the
    report-gate paths and /continue never change it.
    """
    await _ensure_run_config(info)
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
    chosen_length = (info.get("length")
                     or info.get("state", {}).get("length")
                     or DEFAULT_LENGTH)

    progress = await _stream_progress(ctx.application.bot, chat_id,
        "📚 Deep-researching approved sources…", progress)

    try:
        async for ev in graph.astream(resume_input, config=cfg,
                                      stream_mode="updates"):
            for node_name, delta in ev.items():
                info["stage"] = node_name
                if node_name == "extend_prep":
                    n = len(delta.get("sources") or [])
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🔎 Extend: source pool now {n}.", progress)
                elif node_name == "research":
                    n = len(delta.get("fetched") or [])
                    ne = len(delta.get("evidence") or [])
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"📚 Read {n} sources → {ne} evidence notes.", progress)
                elif node_name == "compose":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🧠 Composing ({_LENGTH_LABELS.get(chosen_length,'?')})…",
                        progress)
                elif node_name == "panel":
                    progress = await _stream_progress(ctx.application.bot, chat_id,
                        f"🔬 Panel verdict: "
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
        if (lib := _get_library(ctx)) is not None:      # v4: runs.exa_usd, the run's Exa spend so far (§2.4)
            await lib.update_run(run_id, exa_usd=(cur.get("search_spend") or {}).get("exa_usd"))
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
    # Use most of Telegram's 4096-char message budget for the excerpt
    # (was 1200 — the handoff §6 "preview truncates" tech debt). The
    # header + HTML <pre> wrapper + escaping expansion need headroom, so
    # cap at 3000 raw chars and let the full file follow on Approve.
    excerpt_cap = 3000
    excerpt = ""
    full_len = 0
    try:
        full_text = md.read_text(encoding="utf-8", errors="replace")
        full_len = len(full_text)
        excerpt = full_text[:excerpt_cap]
    except Exception:
        pass
    tail_note = (f"_(first {min(full_len, excerpt_cap)} of {full_len} "
                 f"chars below)_" if full_len > excerpt_cap
                 else "_(full report below)_")
    text = (
        f"📝 *Report preview*  ·  {_LENGTH_LABELS.get(chosen_length,'?')}\n\n"
        f"Folder: `{_md_escape(paths.get('folder','?'))}`\n"
        f"Markdown: `{_md_escape(md.name)}`  "
        + (f"· PDF: `report.pdf`" if paths.get("pdf") else "· PDF: _failed_")
        + f"\n\n{tail_note}"
    )

    # HTML primary (tolerates arbitrary content), plain-text fallback —
    # legacy-markdown crashed on ``_ * [ &`` in excerpts (2026-07-08 bug).
    # HTML-escaping expands text (& → &amp;), so shrink the excerpt until
    # the whole message fits Telegram's 4096-char cap.
    def _fit(exc: str) -> str:
        return _html_escape_for_tg(text) + (
            ("\n\n<pre>" + _html_escape_for_tg(exc) + "</pre>")
            if exc else "")
    html_body = _fit(excerpt)
    while excerpt and len(html_body) > 4000:
        excerpt = excerpt[:len(excerpt) - 200]
        html_body = _fit(excerpt)
    plain_body = (text + ("\n\n```\n" + excerpt + "\n```"
                          if excerpt else ""))[:4000]

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
    await _ensure_run_config(info)
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
        common._inflight.pop(run_id, None)
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


def register(app: Application) -> None:
    app.add_handler(CommandHandler("research", research_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
