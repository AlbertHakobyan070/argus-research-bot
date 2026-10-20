"""The run driver (spec §2.5): every research drive and every gate action.

- **One live drive per run**: the in-flight registry (``common._inflight``,
  ``{run_id: RunHandle}``) holds each tracked run's handle with its own
  lock; every action takes the lock and checks the registry first, so a
  second actor gets ``already_running``. A run is tracked while it drives
  or rests at a gate; ``awaiting`` is its gate only while no drive is live
  (``snap.next`` alone is no gate signal).
- **One executing run at a time** (free-tier limits, Exa budget): a start,
  approve, re-plan, Extend, Revise or resume while another run drives gets
  ``busy``; runs resting at a gate do not count.
- **Drives are plain** ``asyncio.create_task`` (never
  ``application.create_task``, whose tasks PTB awaits in ``stop()`` before
  ``post_stop``) with a fresh ``RunControl``; the driver reports its own
  failures (status ``error``, a chat message, a ``stage error`` event).
- **Cancel = drain**: ``request_drain``; LangGraph raises ``GraphDrained``
  at the next superstep boundary with the checkpoint saved. ``task.cancel()``
  only after ``ARGUS_CANCEL_DRAIN_TIMEOUT_S`` (it cannot stop a node's
  running thread, only the wait for it); shutdown drains every live drive
  for at most ``ARGUS_SHUTDOWN_DRAIN_S``, then cancels.

Handlers (``research.py``, ``history.py``) and the console call these
functions and relay ``Outcome.text``; the gate messages (plan card, report
preview) belong to the driver, which edits them when a gate is acted on
from either UI.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl
from langgraph.types import Command
from telegram.constants import ParseMode
from telegram.error import TelegramError

from ..config import get_settings
from ..engines import registry
from ..engines.registry import EngineConfigError
from ..engines.search import resolve_recency
from ..graph.nodes import MAX_EXTEND_ROUNDS, MAX_REVISE_ROUNDS
from ..graph.run_config import PRESETS, RunConfig, ensure_in_checkpoint, require
from ..graph.search_providers import exa_enabled
from ..graph.state import DEFAULT_LENGTH, VALID_LENGTHS, ResearchBrief
from ..library import mirror_run_md, new_run_id
from . import common
from .common import _safe_send, _send_md_doc, _set_run_status
from .progress import RunEventBus, TelegramCard

logger = logging.getLogger("argus.bot")

bus = RunEventBus()
_app: Any = None                  # the Application (bot, graphs, library), set by init()
_sleep = asyncio.sleep            # the cancel deadline's clock (tests patch it)

PLAN, REPORT = "plan", "report"
_GATES = {("research",): PLAN, ("fetcher",): PLAN, ("deliver",): REPORT}   # ("fetcher",): pre-v3 plan gate
DEPTHS = tuple(PRESETS)
RECENCY_CYCLE = (None, "d", "w", "m", "y", "any")      # None: the brief's time-sensitivity hint decides
_REPLY_PROMPTS = {
    "plan_edit": (PLAN, "_✏️ Reply with your changes and I'll redraft the plan "
                        "(or tap ✅ Approve to keep this one)._"),
    "revise_feedback": (REPORT, "🔁 Reply with what to change and I'll revise the report."),
}


@dataclass(frozen=True)
class Outcome:
    """``code``: started · attached · updated · draining · cancelled ·
    delivered · busy · already_running · refused · not_found. ``text`` is
    the chat line ("" when the driver's own messages already say it)."""
    code: str
    text: str
    run_id: str | None = None


@dataclass
class Choices:
    """The plan-gate knobs; None follows the default (depth: the length's;
    profile: the active one; recency: the brief's hint)."""
    length: str
    depth: str | None = None
    profile: str | None = None
    recency: str | None = None


@dataclass(eq=False)
class RunHandle:
    run_id: str
    chat_id: int
    thread_id: str
    topic: str
    quick: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task | None = None
    control: RunControl = field(default_factory=RunControl)
    stage: str = ""                 # the node running (or last seen)
    done_node: str = ""             # the last node whose update arrived: a drain stops after it
    started_at: float = 0.0         # this drive's start (time.time())
    card: TelegramCard | None = None
    deadline: asyncio.Task | None = None
    starting: bool = False          # an action holds the run while it prepares a drive
    gate_kind: str | None = None    # PLAN / REPORT while the run rests at a gate
    gate_msg: tuple[int, str, str | None] | None = None   # the gate message: (id, text, parse mode)
    choices: Choices | None = None
    profile_checks: dict[str, Any] = field(default_factory=dict)   # profile → snapshot | problem text
    on_report: Callable[[], Awaitable[None]] | None = None

    @classmethod
    def of(cls, row: dict, chat_id: int) -> RunHandle:
        return cls(run_id=row["run_id"], chat_id=chat_id, thread_id=row["thread_id"],
                   topic=row.get("topic") or "", quick=row.get("mode") == "quick")

    @property
    def cfg(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    @property
    def graph(self):
        return _app.bot_data["quick_graph" if self.quick else "graph"]

    @property
    def live(self) -> bool:
        return self.starting or (self.task is not None and not self.task.done())


async def init(app) -> None:
    """POST_INIT hook: the application the drives use."""
    global _app
    _app = app


def tracked(chat_id: int | None = None) -> dict[str, RunHandle]:
    """The tracked runs (all, or one chat's), keyed by run id."""
    return {rid: h for rid, h in common._inflight.items() if chat_id is None or h.chat_id == chat_id}


def console_url() -> str | None:
    """The console's public URL (the WP-E2 start hook sets it); None when the console is not running."""
    return (_app.bot_data.get("webapp_public_url") if _app is not None else None) or None


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _cut(text: str, n: int = 32) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def _library():
    return _app.bot_data.get("library")


async def _status(run_id: str, status: str, *, report_dir: str | None = None) -> None:
    await _set_run_status(SimpleNamespace(application=_app), run_id, status, report_dir=report_dir)


async def _say(chat_id: int, text: str, *, mode: str | None = None, markup=None):
    return await _safe_send(_app.bot, chat_id, text, reply_markup=markup, parse_mode=mode)


def _not_found(run_id: str) -> Outcome:
    return Outcome("not_found", f"(run {run_id} is not in flight — it may have finished, been cancelled, or "
                                f"been lost in a bot restart; /continue {run_id} re-attaches it)", run_id)


def _running(h: RunHandle) -> Outcome:
    return Outcome("already_running", f"Run `{h.run_id}` is already running (at `{h.stage or 'start'}`) — "
                                      "wait for it to reach a gate.", h.run_id)


def _refused(run_id: str, what: str, e: Exception) -> Outcome:
    """A refused action reaches the chat; the run is as it was. An error
    outside the expected kinds is also logged with its traceback."""
    if not isinstance(e, (EngineConfigError, ValueError)):
        logger.error("run %s: %s failed", run_id, what, exc_info=e)
    return Outcome("refused", f"⚠️ Could not {what} run `{run_id}`: {e}", run_id)


def _busy(exclude: str | None = None) -> Outcome | None:
    """One executing run at a time: any live drive of another run."""
    for h in common._inflight.values():
        if h.run_id != exclude and h.live:
            return Outcome("busy", f"⏳ busy — run `{h.run_id}` (`{_cut(h.topic)}`) is at `{h.stage or 'start'}`; "
                                   "try again when it reaches a gate.", exclude)
    return None


@contextlib.asynccontextmanager
async def _hold(run_id: str):
    """The run's handle with its lock held; None when the run is not tracked
    (also when it was forgotten while this actor waited for the lock)."""
    h = common._inflight.get(run_id)
    if h is None:
        yield None
        return
    async with h.lock:
        yield h if common._inflight.get(run_id) is h else None


def _forget(h: RunHandle) -> None:
    if common._inflight.get(h.run_id) is h:
        del common._inflight[h.run_id]
    pending = common._pending_reply.get(h.chat_id)
    if pending is not None and pending[1] == h.run_id:
        del common._pending_reply[h.chat_id]


async def _edit_gate(h: RunHandle, note: str = "", *, markup=None) -> None:
    """Edit the gate message: its text plus ``note``; ``markup`` None removes
    the keyboard. A Telegram failure is logged (the run is unaffected)."""
    if h.gate_msg is None:
        return
    mid, text, mode = h.gate_msg
    body = f"{text}\n\n{note}" if note else text
    for pm in dict.fromkeys((mode, None)):
        try:
            await _app.bot.edit_message_text(chat_id=h.chat_id, message_id=mid, text=body,
                                             reply_markup=markup, parse_mode=pm)
            return
        except TelegramError as e:
            logger.warning("run %s: editing the gate message (%s) failed: %s", h.run_id, pm or "plain", e)


async def _foreign_nodes(h: RunHandle) -> list[str]:
    """Nodes the run's checkpoint names that this graph lacks — an older
    (v2) topology, from which a resume runs no node and stops at the same gate."""
    tup = await h.graph.checkpointer.aget_tuple(h.cfg)
    if tup is None:
        return []
    names = {k.removeprefix("branch:to:") for k in tup.checkpoint["channel_versions"] if k.startswith("branch:to:")}
    return sorted(names - set(h.graph.nodes))


async def _resumable(h: RunHandle, engines=None) -> dict:
    """Resume entries, before anything changes: the engines snapshot resolves
    (``engines``, else the active one), the checkpoint carries its RunConfig
    (legacy injection) and this graph has its nodes. Returns the checkpoint's
    values; raises otherwise."""
    await ensure_in_checkpoint(h.graph, h.cfg, engines=engines or await registry.active_snapshot())
    foreign = await _foreign_nodes(h)
    if foreign:
        raise ValueError(f"it was made by an older Argus graph (its checkpoint names {', '.join(foreign)}, "
                         "which this graph lacks) and cannot be resumed")
    return (await h.graph.aget_state(h.cfg)).values or {}


# ---------------------------------------------------------------------------
# drives
# ---------------------------------------------------------------------------

def _launch(h: RunHandle, resume_input: Any, *, stage: str) -> None:
    h.control, h.deadline, h.gate_kind = RunControl(), None, None
    h.stage, h.done_node, h.started_at = stage, "", time.time()
    h.card = TelegramCard(_app.bot, h.chat_id, h.run_id, bus,
                          title=f"🔄 Run {h.run_id} — {_cut(h.topic, 60)}")
    h.task = asyncio.create_task(_drive(h, resume_input), name=f"run {h.run_id}")
    h.starting = False
    logger.info("run %s: task started (%s)", h.run_id, stage)


async def _drive(h: RunHandle, resume_input: Any) -> None:
    h.card.start()
    try:
        try:
            try:
                await _stream(h, resume_input)
            finally:
                await _stop_deadline(h)
        except GraphDrained as e:
            await _drained(h, e.reason)
        else:
            await _at_rest(h)
    except asyncio.CancelledError:
        await _abandoned(h)
        raise
    except Exception as e:
        await _failed(h, e)
    finally:
        await _record_spend(h)
        bus.finish(h.run_id)
        if h.gate_kind is None:
            _forget(h)


async def _stream(h: RunHandle, resume_input: Any) -> None:
    """Drive to the next gate, the end or a drain; the nodes' custom events
    go to the bus as they arrive, plus a ``stage start`` per task and a
    ``stage end`` per update (unless the node published its own end)."""
    ended: set[str] = set()
    async for mode, chunk in h.graph.astream(resume_input, h.cfg, stream_mode=["tasks", "updates", "custom"],
                                             durability="sync", control=h.control):
        if mode == "custom":
            kind, node, data = chunk.get("kind") or "custom", chunk.get("node") or h.stage, chunk.get("data") or {}
            bus.publish(h.run_id, kind, node, data)
            if kind == "stage" and data.get("status") == "end":
                ended.add(node)
        elif mode == "tasks" and "input" in chunk:            # a task starts (its result chunk is ignored)
            h.stage = chunk["name"]
            bus.publish(h.run_id, "stage", h.stage, {"status": "start", "summary": ""})
        elif mode == "updates":
            for node, delta in chunk.items():
                if node.startswith("__"):                       # __interrupt__ at a gate
                    continue
                h.done_node = node
                if node in ended:
                    ended.discard(node)
                else:
                    bus.publish(h.run_id, "stage", node, {"status": "end", "summary": _summary(node, delta or {})})


def _summary(node: str, delta: dict) -> str:
    match node:
        case "intake":
            return f"mode {delta.get('mode', '?')}"
        case "scout":
            return f"{len(delta.get('sources') or [])} candidate sources"
        case "research":
            return f"read {len(delta.get('fetched') or [])} sources"
        case "panel":
            return f"verdict {(delta.get('review_verdict') or {}).get('verdict', '?')}"
        case "extend_prep":
            return f"source pool {len(delta.get('sources') or [])}"
    return ""


async def _stop_deadline(h: RunHandle) -> None:
    if h.deadline is not None and not h.deadline.done():
        h.deadline.cancel()
        await asyncio.wait([h.deadline])


async def _at_rest(h: RunHandle) -> None:
    """The drive stopped without a drain: a gate, or the end."""
    snap = await h.graph.aget_state(h.cfg)
    values, nxt = snap.values or {}, tuple(snap.next or ())
    kind = _GATES.get(nxt)
    if kind is not None and h.control.drain_requested:
        # a gate interrupt fires before the drain boundary (interrupt_after scout):
        # the cancel still wins over presenting the gate
        await _drained(h, h.control.drain_reason)
    elif kind == PLAN:
        if not values.get("plan"):
            raise RuntimeError("the brief produced no plan")
        await h.card.stop("📋 Plan ready — review it below.")
        if h.choices is None:
            h.choices = _default_choices(values)
        await _present_plan(h, values)
    elif kind == REPORT:
        await h.card.stop("📝 Report ready — preview below.")
        await _present_report(h, values)
        if h.on_report is not None:
            callback, h.on_report = h.on_report, None
            try:
                await callback()
            except Exception:
                logger.exception("run %s: the report-gate callback failed", h.run_id)
    elif not nxt:
        if h.quick:
            await h.card.stop("✅ Answered below.")
            await _say(h.chat_id, values.get("quick_answer") or "(no answer)", mode=ParseMode.MARKDOWN)
            await _status(h.run_id, "done")
        else:
            await h.card.stop("✅ Finished.")
            await _status(h.run_id, "done", report_dir=(values.get("report_paths") or {}).get("folder"))
            await _say(h.chat_id, f"✅ Run {h.run_id} finished — /continue {h.run_id} extends it.")
    else:
        raise RuntimeError(f"the drive stopped before {', '.join(nxt)}, which is not a gate")


async def _drained(h: RunHandle, reason: str) -> None:
    node = h.done_node or h.stage
    note = f"cancelled after node {node} ({reason})"
    await h.graph.aupdate_state(h.cfg, {"cancel_note": note})
    logger.info("run %s: %s", h.run_id, note)
    await _status(h.run_id, "cancelled")
    bus.publish(h.run_id, "stage", node, {"status": "cancelled", "summary": note})
    await h.card.stop(f"⏹ Cancelled after {node} — /continue {h.run_id} resumes it.")


async def _abandoned(h: RunHandle) -> None:
    """``task.cancel()`` after the drain bound: the node's thread may still run;
    the checkpoint is the one before it, so /continue reruns that node."""
    note = f"cancelled during node {h.stage} ({h.control.drain_reason or 'cancel'}); the node was abandoned"
    await _status(h.run_id, "cancelled")
    bus.publish(h.run_id, "stage", h.stage, {"status": "cancelled", "summary": note})
    await h.card.stop(f"⏹ Cancelled during {h.stage} (abandoned) — /continue {h.run_id} reruns it.")


async def _failed(h: RunHandle, e: Exception) -> None:
    h.gate_kind = None
    logger.error("run %s: the drive failed at %s", h.run_id, h.stage, exc_info=e)
    bus.publish(h.run_id, "stage", h.stage, {"status": "error", "summary": f"{type(e).__name__}: {e}"})
    await h.card.stop(f"⚠️ Failed at {h.stage or 'start'}.")
    await _status(h.run_id, "error")
    await _say(h.chat_id, f"⚠️ Research failed: {e}")


async def _record_spend(h: RunHandle) -> None:
    """``runs.exa_usd`` at every drive end (report gate, cancelled, error):
    the month-to-date total counts failed runs' spend too. A failed write is
    visible but changes nothing else."""
    lib = _library()
    if lib is None:
        return
    try:
        values = (await h.graph.aget_state(h.cfg)).values or {}
        usd = (values.get("search_spend") or {}).get("exa_usd")
        if not await lib.update_run(h.run_id, exa_usd=usd):
            logger.warning("run %s has no library row; its exa_usd is in the checkpoint only", h.run_id)
    except Exception as e:
        logger.exception("run %s: recording the Exa spend failed", h.run_id)
        await _say(h.chat_id, f"⚠️ Could not record run {h.run_id}'s Exa spend: {e}")


# ---------------------------------------------------------------------------
# cancel and shutdown
# ---------------------------------------------------------------------------

def _drain(h: RunHandle, reason: str) -> None:
    h.control.request_drain(reason)
    logger.info("run %s: drain requested (%s)", h.run_id, reason)
    bus.publish(h.run_id, "stage", h.stage, {"status": "draining", "summary": reason})


async def _drain_deadline(h: RunHandle, seconds: float) -> None:
    await _sleep(seconds)
    if h.task is not None and not h.task.done():
        logger.warning("run %s: the drain did not finish within %g s; task.cancel() — it cannot stop the "
                       "node's running thread, only the wait for it", h.run_id, seconds)
        h.task.cancel()


async def cancel_run(run_id: str, *, actor: str = "telegram") -> Outcome:
    """A live drive drains (finishes its current step); a run at a gate is
    cancelled now."""
    async with _hold(run_id) as h:
        if h is None:
            return _not_found(run_id)
        if h.live:
            if h.control.drain_requested:
                return Outcome("draining", f"⏹ Run `{run_id}` is already cancelling — finishing `{h.stage}`.", run_id)
            _drain(h, "user cancel")
            h.deadline = asyncio.create_task(_drain_deadline(h, get_settings().cancel_drain_timeout_s),
                                             name=f"run {run_id} drain deadline")
            return Outcome("draining", f"⏹ Cancelling run `{run_id}` — finishing the current step (`{h.stage}`).",
                           run_id)
        await _status(run_id, "cancelled")
        await _edit_gate(h, "❌ Cancelled" + (" from console." if actor == "webapp" else "."))
        _forget(h)
        return Outcome("cancelled", f"❌ Cancelled run `{run_id}`.", run_id)


async def drain_all(app) -> None:
    """POST_STOP hook: drain every live drive, wait up to
    ``ARGUS_SHUTDOWN_DRAIN_S``, then cancel what still runs."""
    live = [h for h in list(common._inflight.values()) if h.task is not None and not h.task.done()]
    if not live:
        return
    bound = get_settings().shutdown_drain_s
    for h in live:
        if not h.control.drain_requested:
            _drain(h, "shutdown")
    _, pending = await asyncio.wait([h.task for h in live], timeout=bound)
    for h in live:
        if h.task in pending:
            logger.warning("run %s: still running %g s after the shutdown drain; cancelling it", h.run_id, bound)
            h.task.cancel()
    if pending:
        await asyncio.wait(pending)


# ---------------------------------------------------------------------------
# starting a run
# ---------------------------------------------------------------------------

async def start_run(*, chat_id: int, user_id: int, topic: str, length: str = DEFAULT_LENGTH,
                    depth: str | None = None, overrides: dict | None = None, quick: bool = False,
                    origin: str = "telegram") -> Outcome:
    """/research (``quick``: /ask): a new run and its first drive — to the
    plan gate, or (quick) to the answer."""
    if _app is None or ("quick_graph" if quick else "graph") not in _app.bot_data:
        return Outcome("refused", "⚠️ Bot not fully initialized (no graph) — restart the bot.")
    busy = _busy()
    if busy is not None:
        return busy
    run_id = new_run_id()
    h = RunHandle(run_id=run_id, chat_id=chat_id, thread_id=f"tg:{chat_id}:{run_id}", topic=topic, quick=quick)
    h.starting = True                                  # claimed before the first await: a second start is busy
    common._inflight[run_id] = h
    async with h.lock:
        try:
            rc = RunConfig.from_preset(length=length, depth=depth, engines=await registry.active_snapshot(),
                                       overrides=overrides, origin=origin)
            lib = _library()
            if lib is not None:
                await lib.create_run(run_id=run_id, thread_id=h.thread_id, chat_id=chat_id, topic=topic,
                                     length=rc.length, mode="quick" if quick else "deep",
                                     status="running" if quick else "planning", config_json=rc.model_dump_json(),
                                     depth=rc.depth, origin=rc.origin)
        except Exception as e:
            _forget(h)
            if not isinstance(e, (EngineConfigError, ValueError)):
                logger.error("run %s: start failed", run_id, exc_info=e)
            return Outcome("refused", f"⚠️ Could not start the research: {e}")
        state_in = {
            "thread_id": h.thread_id, "user_id": user_id, "user_request": topic, "length": rc.length,
            "run_config": rc.model_dump(), "messages": [], "plan": None, "sources": [], "fetched": [],
            "findings": [], "draft_md": "", "revision_notes": [], "revision_rounds": 0, "model_calls": [],
            "hitl": {"pending": False},
        }
        _launch(h, state_in, stage="intake")
    return Outcome("started", "", run_id)


# ---------------------------------------------------------------------------
# the plan gate
# ---------------------------------------------------------------------------

def _default_choices(values: dict) -> Choices:
    rc = require(values)
    return Choices(length=rc.length, depth=None if rc.depth == RunConfig.default_depth(rc.length) else rc.depth,
                   recency=rc.recency)


def _rebuilt(h: RunHandle, rc: RunConfig, engines) -> RunConfig:
    """``rc`` for the plan-gate choices, through ``rebuild`` (console
    overrides survive); the recency knob is the ``recency`` override."""
    c = h.choices
    overrides = {k: v for k, v in rc.overrides_applied.items() if k != "recency"}
    if c.recency is not None:
        overrides["recency"] = c.recency
    return rc.model_copy(update={"overrides_applied": overrides}).rebuild(length=c.length, depth=c.depth,
                                                                           engines=engines)


async def _chosen_engines(h: RunHandle):
    """The chosen profile's snapshot: the active one resolved now, any other
    the validation cached when the knob reached it."""
    if h.choices.profile is None:
        return await registry.active_snapshot()
    check = h.profile_checks[h.choices.profile]
    if isinstance(check, str):
        raise EngineConfigError(check)
    return check


async def _payload(h: RunHandle, values: dict) -> dict:
    """The plan card's data (also the console's): plan, sources, knobs, Exa estimate."""
    c, rc = h.choices, require(values)
    brief = ResearchBrief.model_validate(values.get("brief") or {})
    cfg = registry.current_config()
    profile = c.profile or cfg.active_profile
    if c.profile is None:
        try:
            await registry.active_snapshot()
            problem = None
        except EngineConfigError as e:
            problem = str(e)
    else:
        check = h.profile_checks[profile]
        problem = check if isinstance(check, str) else None
    error = estimate = None
    try:
        preview = _rebuilt(h, rc, rc.engines)
        if exa_enabled():
            estimate = preview.exa_cost_estimate_usd(cfg.pricing.exa)
    except ValueError as e:
        error = str(e)
    return {
        "run_id": h.run_id, "topic": h.topic, "plan": values.get("plan") or {},
        "sources": list(values.get("sources") or []),
        "archetype": brief.archetype if values.get("brief") else None,
        "length": c.length, "depth": c.depth or RunConfig.default_depth(c.length), "depth_auto": c.depth is None,
        "profile": profile, "profile_problem": problem, "profiles": list(cfg.profiles),
        "recency": c.recency,
        "recency_window": resolve_recency(rc.model_copy(update={"recency": c.recency}), brief)[0],
        "exa_enabled": exa_enabled(), "exa_estimate_usd": estimate, "error": error,
    }


async def plan_payload(run_id: str) -> dict | None:
    """The plan card's data for a tracked run resting at its plan gate, else None."""
    h = common._inflight.get(run_id)
    if h is None or h.live or h.gate_kind != PLAN:
        return None
    return await _payload(h, (await h.graph.aget_state(h.cfg)).values or {})


async def _present_plan(h: RunHandle, values: dict) -> None:
    from . import research                          # lazy: research imports this module
    payload = await _payload(h, values)
    text = research._format_plan(payload)
    msg = await _say(h.chat_id, text, mode=ParseMode.MARKDOWN, markup=research._plan_keyboard(payload))
    h.gate_msg = (msg.message_id, text, ParseMode.MARKDOWN) if msg is not None else None
    await _status(h.run_id, "awaiting_plan")
    h.gate_kind = PLAN


async def _rerender_plan(h: RunHandle) -> None:
    from . import research                          # lazy: research imports this module
    payload = await _payload(h, (await h.graph.aget_state(h.cfg)).values or {})
    if h.gate_msg is not None:
        h.gate_msg = (h.gate_msg[0], research._format_plan(payload), h.gate_msg[2])
        await _edit_gate(h, markup=research._plan_keyboard(payload))


async def _choose(h: RunHandle, knob: str, value: str) -> None:
    c = h.choices
    if knob == "length":
        if value not in VALID_LENGTHS:
            raise ValueError(f"unknown length {value!r}")
        c.length = value
    elif knob == "depth":
        if value != "auto" and value not in DEPTHS:
            raise ValueError(f"unknown depth {value!r}")
        c.depth = None if value == "auto" else value
    elif knob == "recency":
        if value == "next":
            value = RECENCY_CYCLE[(RECENCY_CYCLE.index(c.recency) + 1) % len(RECENCY_CYCLE)]
        elif value == "auto":
            value = None
        elif value not in RECENCY_CYCLE:
            raise ValueError(f"unknown recency {value!r}")
        c.recency = value
    elif knob == "profile":
        cfg = registry.current_config()
        names = list(cfg.profiles)
        if value == "next":
            value = names[(names.index(c.profile or cfg.active_profile) + 1) % len(names)]
        elif value not in names:
            raise ValueError(f"unknown engines profile {value!r}")
        if value != cfg.active_profile and value not in h.profile_checks:
            try:                                    # validated once per run; the cycle shows the cached result
                h.profile_checks[value] = await registry.resolve_snapshot(cfg, value)
            except EngineConfigError as e:
                h.profile_checks[value] = str(e)
        c.profile = None if value == cfg.active_profile else value
    else:
        raise ValueError(f"unknown plan knob {knob!r}")


async def set_plan_choice(run_id: str, knob: str, value: str) -> Outcome:
    """A plan-card knob (length, depth, profile, recency; "next" cycles the
    last two): the card is re-rendered; nothing resumes."""
    async with _hold(run_id) as h:
        if h is None:
            return _not_found(run_id)
        if h.live:
            return _running(h)
        if h.gate_kind != PLAN:
            return Outcome("refused", f"Run `{run_id}` is not at its plan gate.", run_id)
        try:
            await _choose(h, knob, value)
        except ValueError as e:
            return Outcome("refused", f"⚠️ {e}", run_id)
        await _rerender_plan(h)
        return Outcome("updated", "", run_id)


async def approve_plan(run_id: str, *, actor: str = "telegram", length: str | None = None,
                       depth: str | None = None, profile: str | None = None) -> Outcome:
    """Approve: the RunConfig rebuilt for the choices (``rc.rebuild``) into the
    checkpoint and the library row — everything resolved first, so a refusal
    leaves the run at its gate — then the research drive."""
    async with _hold(run_id) as h:
        if h is None:
            return _not_found(run_id)
        if h.live:
            return _running(h)
        if h.gate_kind != PLAN:
            return Outcome("refused", f"Run `{run_id}` is not at its plan gate.", run_id)
        busy = _busy(exclude=run_id)
        if busy is not None:
            return busy
        h.starting = True                              # a second actor now gets already_running
        try:
            for knob, value in (("length", length), ("depth", depth), ("profile", profile)):
                if value is not None:
                    await _choose(h, knob, value)
            engines = await _chosen_engines(h)
            values = await _resumable(h, engines)
            old = require(values)
            rc = _rebuilt(h, old, engines)
            update: dict[str, Any] = {"length": rc.length, "run_config": rc.model_dump()}
            if rc.recency != old.recency:              # the scout decided recency once; research reuses it
                brief = ResearchBrief.model_validate(values.get("brief") or {})
                update["search_spend"] = {**(values.get("search_spend") or {}),
                                          "recency": resolve_recency(rc, brief)[0]}
            await h.graph.aupdate_state(h.cfg, update)
            lib = _library()
            if lib is not None and not await lib.set_run_config(run_id, rc.model_dump()):
                logger.warning("run %s has no library row; its run_config is in the checkpoint only", run_id)
        except Exception as e:
            h.starting = False
            return _refused(run_id, "approve", e)
        bus.publish(run_id, "gate", "", {"kind": PLAN, "actor": actor})
        await _edit_gate(h, "✅ approved from console" if actor == "webapp" else "_✅ Approved — continuing._")
        await _status(run_id, "running")
        _launch(h, Command(resume=True), stage="research")
        return Outcome("started", "", run_id)


async def edit_plan(run_id: str, feedback: str, *, actor: str = "telegram") -> Outcome:
    """Plan Edit: the feedback goes into the checkpoint as if intake just ran,
    then brief → scout re-drafts the plan to a new plan gate."""
    async with _hold(run_id) as h:
        if h is None:
            return _not_found(run_id)
        if h.live:
            return _running(h)
        if h.gate_kind != PLAN:
            return Outcome("refused", f"Run `{run_id}` is not at its plan gate.", run_id)
        if not feedback.strip():
            return Outcome("refused", "Please send some text.", run_id)
        busy = _busy(exclude=run_id)
        if busy is not None:
            return busy
        h.starting = True
        try:
            await _resumable(h)
            await h.graph.aupdate_state(h.cfg, {"plan_feedback": feedback.strip()}, as_node="intake")
        except Exception as e:
            h.starting = False
            return _refused(run_id, "re-plan", e)
        bus.publish(run_id, "gate", "", {"kind": PLAN, "actor": actor})
        await _edit_gate(h, "_✏️ Redrafting the plan with your changes…_")
        await _status(run_id, "planning")
        _launch(h, None, stage="brief")
        return Outcome("started", "", run_id)


async def await_reply(run_id: str, kind: str) -> Outcome:
    """Plan Edit / report Revise buttons: the chat's next text is the reply
    (``kind`` "plan_edit" | "revise_feedback"); the gate keeps its keyboard."""
    from . import research                          # lazy: research imports this module
    gate, prompt = _REPLY_PROMPTS[kind]
    async with _hold(run_id) as h:
        if h is None:
            return _not_found(run_id)
        if h.live:
            return _running(h)
        if h.gate_kind != gate:
            return Outcome("refused", f"Run `{run_id}` is not at its {gate} gate.", run_id)
        common._pending_reply[h.chat_id] = (kind, run_id)
        markup = (research._plan_keyboard(await _payload(h, (await h.graph.aget_state(h.cfg)).values or {}))
                  if gate == PLAN else research._report_keyboard(run_id))
        await _edit_gate(h, prompt, markup=markup)
        return Outcome("updated", "", run_id)


# ---------------------------------------------------------------------------
# the report gate
# ---------------------------------------------------------------------------

async def _present_report(h: RunHandle, values: dict) -> None:
    from . import research                          # lazy: research imports this module
    await _status(h.run_id, "awaiting_report", report_dir=(values.get("report_paths") or {}).get("folder"))
    msg, text, mode = await research._send_report_preview(_app.bot, h.chat_id, h.run_id, values)
    h.gate_msg = (msg.message_id, text, mode) if msg is not None else None
    h.gate_kind = REPORT


async def report_action(run_id: str, action: str, *, feedback: str | None = None,
                        actor: str = "telegram") -> Outcome:
    """Report gate: send · extend · revise (with ``feedback``) · cancel."""
    if action == "cancel":
        return await cancel_run(run_id, actor=actor)
    async with _hold(run_id) as h:
        if h is None:
            return _not_found(run_id)
        if h.live:
            return _running(h)
        if h.gate_kind != REPORT:
            return Outcome("refused", f"Run `{run_id}` is not at its report preview.", run_id)
        if action == "send":
            return await _deliver(h, actor)
        if action not in ("extend", "revise"):
            return Outcome("refused", f"⚠️ unknown report action {action!r}", run_id)
        if action == "revise" and not (feedback or "").strip():
            return Outcome("refused", "Please send some text.", run_id)
        busy = _busy(exclude=run_id)
        if busy is not None:
            return busy
        h.starting = True
        try:
            values = await _resumable(h)
            if action == "extend":
                if int(values.get("extend_rounds") or 0) >= MAX_EXTEND_ROUNDS:
                    raise ValueError(f"it already used its {MAX_EXTEND_ROUNDS} Extend rounds")
                update: dict[str, Any] = {"extend_requested": True}
            else:
                if int(values.get("revise_rounds") or 0) >= MAX_REVISE_ROUNDS:
                    raise ValueError(f"it already used its {MAX_REVISE_ROUNDS} Revise rounds")
                notes = [*(values.get("revision_notes") or []), f"USER REVISION REQUEST: {feedback.strip()}"]
                update = {"revision_requested": True, "revision_notes": notes}   # last-value channel: the full list
            await h.graph.aupdate_state(h.cfg, update)
        except Exception as e:
            h.starting = False
            return _refused(run_id, action, e)
        bus.publish(run_id, "gate", "", {"kind": REPORT, "actor": actor})
        await _edit_gate(h, ("🔎 Extending research — gathering more…" if action == "extend"
                             else "🔁 Revising with your notes…") + (" (from console)" if actor == "webapp" else ""))
        await _status(run_id, "running")
        _launch(h, Command(resume=True), stage="deliver")
        return Outcome("started", "", run_id)


async def _deliver(h: RunHandle, actor: str) -> Outcome:
    """Send: the report files to the chat, then the run is done. The engines
    snapshot must resolve first (pointer 6)."""
    try:
        await ensure_in_checkpoint(h.graph, h.cfg, engines=await registry.active_snapshot())
        values = (await h.graph.aget_state(h.cfg)).values or {}
    except Exception as e:
        return _refused(h.run_id, "send", e)
    paths = values.get("report_paths") or {}
    bus.publish(h.run_id, "gate", "", {"kind": REPORT, "actor": actor})
    await _edit_gate(h, "📤 Sent" + (" from console." if actor == "webapp" else "."))
    try:
        cap = f"Argus report — folder: `{paths.get('folder', '')}`"
        for key in ("md", "pdf"):
            if paths.get(key) and Path(paths[key]).exists():
                await _send_md_doc(_app.bot, h.chat_id, Path(paths[key]), caption=cap)
        await _say(h.chat_id, "✅ Delivered.")
    except Exception as e:
        logger.exception("run %s: sending the report failed", h.run_id)
        await _say(h.chat_id, f"⚠️ send failed: {e}")
    _forget(h)
    await _finalize(h, paths)
    return Outcome("delivered", "", h.run_id)


async def _finalize(h: RunHandle, paths: dict) -> None:
    """Terminal bookkeeping: status done, the report as a library asset, the run.md mirror."""
    await _status(h.run_id, "done", report_dir=paths.get("folder"))
    lib = _library()
    if lib is None:
        return
    try:
        if paths.get("md"):
            md = Path(paths["md"])
            await lib.add_asset(kind="report", path=str(md), title=h.topic,
                                size_bytes=md.stat().st_size if md.exists() else 0,
                                meta={"folder": paths.get("folder", ""), "pdf": paths.get("pdf", "")})
        run = await lib.get_run(h.run_id)
        if run and run.get("report_dir"):
            await asyncio.to_thread(mirror_run_md, run)
    except Exception:
        logger.exception("finalize (asset/mirror) failed for run %s", h.run_id)


# ---------------------------------------------------------------------------
# /continue
# ---------------------------------------------------------------------------

async def continue_run(row: dict, *, chat_id: int, appended: list[dict] | None = None,
                       on_report: Callable[[], Awaitable[None]] | None = None) -> Outcome:
    """/continue: a run at a gate is re-attached (its plan card or preview
    re-sent); a stopped run (cancelled, failed) resumes from its checkpoint;
    a finished run is forked into an Extend (with ``appended`` sources, if
    any, and no new searches); ``on_report`` runs at the next report gate."""
    run_id = row["run_id"]
    if row.get("mode") == "quick":
        return Outcome("refused", f"Run `{run_id}` was a quick /ask — nothing to continue.", run_id)
    h = common._inflight.get(run_id)
    if h is None:
        h = common._inflight[run_id] = RunHandle.of(row, chat_id)
    async with _hold(run_id) as held:
        if held is not h:
            return _not_found(run_id)
        if h.live:
            return _running(h)
        try:
            snap = await h.graph.aget_state(h.cfg)
            if snap.created_at is None or not snap.values:
                raise ValueError("no checkpoint found — it may predate the v2 upgrade or its checkpoint DB "
                                 "was deleted")
            nxt = tuple(snap.next or ())
            kind = _GATES.get(nxt)
            if kind is None:
                busy = _busy(exclude=run_id)
                if busy is not None:
                    if h.gate_kind is None:
                        _forget(h)
                    return busy
                h.starting = True
                values = await _resumable(h)
                if not nxt:
                    merged = [*(values.get("sources") or []), *(appended or [])]   # last-value channel: the full list
                    await h.graph.aupdate_state(h.cfg, {"sources": merged, "extend_requested": True,
                                                        "extend_rounds": 0, "revision_rounds": 0,
                                                        "append_only": bool(appended)}, as_node="deliver")
            else:
                await ensure_in_checkpoint(h.graph, h.cfg, engines=await registry.active_snapshot())
                values = (await h.graph.aget_state(h.cfg)).values or {}
                if kind == PLAN:
                    if h.choices is None:
                        h.choices = _default_choices(values)
                    await _present_plan(h, values)
                else:
                    await _present_report(h, values)
        except Exception as e:
            h.starting = False
            if h.gate_kind is None:
                _forget(h)
            return _refused(run_id, "continue", e)
        if kind is not None:
            return Outcome("attached", "", run_id)
        await _status(run_id, "running")
        if nxt:
            _launch(h, None, stage=nxt[0])
            return Outcome("started", f"▶️ Resuming run `{run_id}` from `{nxt[0]}`…", run_id)
        h.on_report = on_report
        _launch(h, None, stage="extend_prep")
        how = (f"with {len(appended)} appended source(s), no new searches" if appended
               else "with a fresh search pass")
        return Outcome("started", f"▶️ Continuing run `{run_id}` {how}…", run_id)


# ---------------------------------------------------------------------------
# the driver's view of a run
# ---------------------------------------------------------------------------

async def snapshot(run_id: str) -> dict | None:
    """``in_flight``, ``awaiting`` (the gate, only while no drive is live),
    ``next``, ``stage``, ``last_seq`` and the checkpoint ``values``; None
    for an unknown run."""
    h = common._inflight.get(run_id)
    if h is None:
        lib = _library()
        row = await lib.get_run(run_id) if lib is not None else None
        if row is None:
            return None
        h = RunHandle.of(row, row["chat_id"])         # an untracked view, not registered
    snap = await h.graph.aget_state(h.cfg)
    nxt = list(snap.next or ())
    return {"run_id": run_id, "in_flight": h.live, "awaiting": None if h.live else _GATES.get(tuple(nxt)),
            "next": nxt, "stage": h.stage, "started_at": h.started_at or None, "last_seq": bus.last_seq(run_id),
            "values": snap.values or {}}


__all__ = ["Outcome", "RunHandle", "bus", "init", "start_run", "approve_plan", "edit_plan", "await_reply",
           "set_plan_choice", "plan_payload", "report_action", "cancel_run", "continue_run", "snapshot",
           "console_url", "tracked", "drain_all"]
