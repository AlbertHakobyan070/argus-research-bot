"""The run driver (spec §2.5): every research drive and every gate action.

- **One live drive per run**: the in-flight registry (``common._inflight``,
  ``{run_id: RunHandle}``) holds each tracked run's handle with its own
  lock; every action takes the lock and checks the registry first, so a
  second actor gets ``already_running``. A run is tracked while it drives
  or rests at a gate; ``awaiting`` is its gate only while no drive is live
  (``snap.next`` alone is no gate signal). A restart empties the registry:
  an action on an untracked run resting at a gate attaches it first.
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
  running thread, only the wait for it); shutdown refuses new claims, lets
  the resumes claimed before it launch (their drives drain at once; a start
  is refused), drains every live drive — all within
  ``ARGUS_SHUTDOWN_DRAIN_S`` — then cancels.

Handlers (``research.py``, ``history.py``) and the console call these
functions and relay ``Outcome.text``; the gate messages (plan card, report
preview) belong to the driver, which edits them when a gate is acted on
from either UI.
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
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

from .. import llm
from ..config import get_settings
from ..engines import registry
from ..engines.registry import EngineConfigError
from ..engines.search import resolve_recency
from ..graph.nodes import MAX_EXTEND_ROUNDS, MAX_REVISE_ROUNDS
from ..graph.research import pass_mode
from ..graph.run_config import PRESETS, RunConfig, ensure_in_checkpoint, require
from ..graph.search_providers import exa_enabled
from ..graph.state import DEFAULT_LENGTH, VALID_LENGTHS, ResearchBrief
from ..library import mirror_run_md, new_run_id
from . import common
from .common import _safe_send, _send_md_doc, _set_run_status
from .progress import RunEventBus, TelegramCard, edit_text

logger = logging.getLogger("argus.bot")

bus = RunEventBus()
_app: Any = None                  # the Application (bot, graphs, library), set by init()
_sleep = asyncio.sleep            # the cancel deadline's clock (tests patch it)
_shutting_down = False            # drain_all began: no new claim (a claim made before may still launch)
_launch_closed = False            # drain_all stopped waiting for the claims made before it: nothing launches

PLAN, REPORT = "plan", "report"
_GATES = {("research",): PLAN, ("deliver",): REPORT}
# ("research",) is the plan gate only for a run that has not passed it: read through _gate, never alone
# A state write without as_node runs as the last node that ran (LangGraph reads versions_seen, which
# an as_node write never updates) and re-runs that node's edges. Each resting state the driver's
# as_node writes leave (the re-plan's intake; Send, the fork and an older bot's advance as deliver)
# → the one node that leads there: a later write on a run resting there passes it, so next stays.
_ROUTES = {("brief",): "intake", ("extend_prep",): "deliver", ("revise_prep",): "deliver", (): "deliver"}
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
    gate_msg: tuple[int, str | None, str | None] | None = None   # (id, text — None until rendered, parse mode)
    choices: Choices | None = None
    profile_checks: dict[str, Any] = field(default_factory=dict)   # profile → snapshot | problem text
    on_report: Callable[[], Awaitable[None]] | None = None
    announced: set[tuple[str, str]] = field(default_factory=set)   # (node, status) this drive's tasks published
    cancelled_by: str | None = None  # the actor of a cancel accepted while this drive was live
    attaching: bool = False          # registered by an action on an untracked run; attached under the lock
    route: str | None = None         # the node its launching action wrote as (re-plan, fork); a resume's: _ROUTES

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

def _gate(nxt: tuple, values: dict) -> str | None:
    """The gate a run resting at ``nxt`` waits at, or None. ``("research",)`` is
    the plan gate only while the run has not passed it — its research pass
    would be the first (``research.pass_mode``, the node's own test). An
    Extend, a gap wave or an append pass resting there (drained after
    ``extend_prep``, force-cancelled inside research) is a stopped run, which
    /continue resumes."""
    kind = _GATES.get(nxt)
    if kind == PLAN and nxt == ("research",) and pass_mode(values) != "first":
        return None
    return kind


def _cut(text: str, n: int = 32) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def _library():
    return _app.bot_data.get("library")


async def _status(run_id: str, status: str, *, report_dir: str | None = None) -> None:
    await _set_run_status(SimpleNamespace(application=_app), run_id, status, report_dir=report_dir)


async def _say(chat_id: int, text: str, *, mode: str | None = None, markup=None):
    return await _safe_send(_app.bot, chat_id, text, reply_markup=markup, parse_mode=mode)


def _not_found(run_id: str, *, quick: bool = False) -> Outcome:
    if quick:                                       # continue_run refuses a quick /ask: never point at /continue
        return Outcome("not_found", f"(run {run_id} was a quick /ask — nothing to cancel or continue)", run_id)
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


def _shutdown(run_id: str | None = None) -> Outcome:
    return Outcome("refused", "⏹ Argus is shutting down — try again after the restart", run_id)


def _busy(exclude: str | None = None) -> Outcome | None:
    """One executing run at a time: any live drive of another run."""
    for h in common._inflight.values():
        if h.run_id != exclude and h.live:
            return Outcome("busy", f"⏳ busy — run `{h.run_id}` (`{_cut(h.topic)}`) is at `{h.stage or 'start'}`; "
                                   "try again when it reaches a gate.", exclude)
    return None


def _claim(h: RunHandle) -> Outcome | None:
    """Claim the run for a drive, before the first await (a second actor then
    gets ``already_running``): refused once the shutdown began, or while
    another run drives."""
    if _shutting_down:
        return _shutdown(h.run_id)
    busy = _busy(exclude=h.run_id)
    if busy is not None:
        return busy
    h.starting = True
    return None


@contextlib.asynccontextmanager
async def _hold(run_id: str, what: str, *, message_id: int | None = None, card: str | None = None,
                row: dict | None = None, chat_id: int | None = None):
    """The run's handle with its lock held. An untracked run resting at a
    gate is attached first (§2.5: buttons minted before a restart keep
    working); the tapped message (``message_id``) becomes its gate message
    when it is that gate's card (``card``: the gate the tapped button
    belongs to). With ``row`` (``continue_run``) an untracked run is
    registered from it as it is: the caller re-attaches or resumes it. An
    actor that waited for the lock of a handle another actor then forgot (a
    failed attach, say) looks the run up again, once. Yields an ``Outcome``
    instead when there is nothing to act on: ``not_found`` (no library row,
    not resting at a gate, or forgotten again while this actor waited), or
    the attach's refusal (``what`` names the action)."""
    for again in (False, True):
        h = common._inflight.get(run_id)
        if h is None and row is not None:
            h = common._inflight[run_id] = RunHandle.of(row, chat_id)
        elif h is None:
            lib = _library()
            try:
                found = await lib.get_run(run_id) if lib is not None else None
            except Exception as e:
                yield _refused(run_id, what, e)
                return
            h = common._inflight.get(run_id)      # another actor may have registered it meanwhile
            if h is None and found is not None:
                h = common._inflight[run_id] = RunHandle.of(found, found["chat_id"])
                h.attaching = True
        if h is None:
            yield _not_found(run_id)
            return
        async with h.lock:
            if common._inflight.get(run_id) is not h:
                if not again:
                    continue                      # forgotten while this actor waited: look it up again
                yield _not_found(run_id, quick=h.quick)
                return
            if h.attaching:
                h.attaching = False
                if row is None:
                    try:
                        kind = await _attach(h, message_id, card)
                    except Exception as e:
                        _forget(h)
                        yield _refused(run_id, what, e)
                        return
                    if kind is None:
                        _forget(h)
                        yield _not_found(run_id, quick=h.quick)
                        return
            yield h
            return


async def _attach(h: RunHandle, message_id: int | None, card: str | None) -> str | None:
    """An untracked run resting at its gate: the checkpoint's gate, its
    RunConfig (``ensure_in_checkpoint``) and the default plan choices (the
    run's own engines profile among them); None when the run does not rest
    at a gate (nothing is written then). The tapped message becomes the gate
    message only when its button belongs to this gate (``card``)."""
    snap = await h.graph.aget_state(h.cfg)
    kind = _gate(tuple(snap.next or ()), snap.values) if snap.values else None
    if kind is None:
        return None
    active = await registry.active_snapshot()
    rc = await ensure_in_checkpoint(h.graph, h.cfg, engines=active)
    if kind == PLAN:
        _attached_choices(h, rc, active.profile)
    h.gate_kind = kind
    if message_id is not None and card == kind:     # its text is rendered from the payload at the first edit
        h.gate_msg = (message_id, None, ParseMode.MARKDOWN if kind == PLAN else ParseMode.HTML)
    logger.info("run %s: attached at its %s gate", h.run_id, kind)
    return kind


def _forget(h: RunHandle) -> None:
    if common._inflight.get(h.run_id) is h:
        del common._inflight[h.run_id]
    pending = common._pending_reply.get(h.chat_id)
    if pending is not None and pending[1] == h.run_id:
        del common._pending_reply[h.chat_id]


async def _edit_gate(h: RunHandle, note: str = "", *, markup=None) -> None:
    """Edit the gate message: its text plus ``note``; ``markup`` None removes
    the keyboard. A message attached from a tap is rendered from the payload
    first. "Message is not modified" is a success and a ``RetryAfter`` is
    waited out (``progress.edit_text``); a message that no longer exists is
    logged once and not edited again; another Telegram failure is logged,
    after a plain-text try (the run is unaffected)."""
    if h.gate_msg is None:
        return
    if h.gate_msg[1] is None:
        await _render_gate_text(h)
    mid, text, mode = h.gate_msg
    body = f"{text}\n\n{note}" if note else text
    for pm in dict.fromkeys((mode, None)):
        try:
            if not await edit_text(_app.bot, chat_id=h.chat_id, message_id=mid, text=body, reply_markup=markup,
                                   parse_mode=pm):
                logger.warning("run %s: the gate message %s no longer exists; it is not edited again",
                               h.run_id, mid)
                h.gate_msg = None
            return
        except TelegramError as e:
            logger.warning("run %s: editing the gate message (%s) failed: %s", h.run_id, pm or "plain", e)


async def _render_gate_text(h: RunHandle) -> None:
    from . import research                          # lazy: research imports this module
    values = (await h.graph.aget_state(h.cfg)).values or {}
    text = (research._format_plan(await _payload(h, values)) if h.gate_kind == PLAN
            else research._preview_bodies(values)[0])
    h.gate_msg = (h.gate_msg[0], text, h.gate_msg[2])


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
    (legacy injection, written as ``_ROUTES`` routes where the run rests, so
    it keeps ``next``) and this graph has its nodes. Returns the checkpoint's
    values; raises otherwise."""
    route = _ROUTES.get(tuple((await h.graph.aget_state(h.cfg)).next or ()))
    await ensure_in_checkpoint(h.graph, h.cfg, engines=engines or await registry.active_snapshot(), as_node=route)
    foreign = await _foreign_nodes(h)
    if foreign:
        raise ValueError(f"it was made by an older Argus graph (its checkpoint names {', '.join(foreign)}, "
                         "which this graph lacks) and cannot be resumed")
    return (await h.graph.aget_state(h.cfg)).values or {}


# ---------------------------------------------------------------------------
# /append sources (§2.4: pinned appends are always ingested)
# ---------------------------------------------------------------------------

async def _pending_appends(h: RunHandle) -> tuple[list[dict], list[dict]]:
    """The run's pending /append rows and the pinned sources they become."""
    from . import history                           # lazy: history imports this module
    lib = _library()
    return await history.pending_appends(lib, h.run_id) if lib is not None else ([], [])


def _appends_marker(h: RunHandle, pending: list[dict], appended: list[dict]):
    """The report-gate callback that marks the appends ingested / failed (the
    /continue fork's bookkeeping), its failure lines to the chat."""
    from . import history                           # lazy: history imports this module
    return history.appends_marker(_library(), h.graph, h.cfg, h.run_id, pending, appended,
                                  lambda text: _say(h.chat_id, text))


def _merged_sources(sources: list[dict] | None, appended: list[dict]) -> list[dict]:
    """``sources`` plus the appended sources it carries no pinned copy of yet."""
    from . import history                           # lazy: history imports this module
    return history.merged_sources(sources, appended)


async def _carried_appends(h: RunHandle, values: dict) -> tuple[list[dict], list[dict]]:
    """The pending /append rows (and their sources) the checkpoint already
    carries as pinned copies: an Extend or a fork merged them, then its pass
    was drained."""
    from . import history                           # lazy: history imports this module
    return history.carried_appends(values.get("sources"), *await _pending_appends(h))


async def _appends_notice(h: RunHandle, *, sent: bool = False, carried: int = 0) -> str:
    """The chat line for a resume or re-attach that leaves the run's pending
    /append sources unused: their count and the path that ingests them
    ("" when none are pending). ``carried``: the pending ones this resume
    ingests (its checkpoint carries them). Read while the action resolves."""
    lib = _library()
    pending = await lib.pending_sources(h.run_id) if lib is not None else []
    if len(pending) <= carried:
        return ""
    how = (f"/continue {h.run_id} ingests them" if sent
           else f"🔎 Extend at the report preview ingests them, or /continue {h.run_id} once the report is sent")
    return f"📎 Run `{h.run_id}` has {len(pending) - carried} pending appended source(s): {how}."


# ---------------------------------------------------------------------------
# drives
# ---------------------------------------------------------------------------

def _launch(h: RunHandle, resume_input: Any, *, stage: str, route: str | None = None) -> Outcome | None:
    """Start the run's next drive (None). Every caller claimed the run, and a
    claim is refused once the shutdown began, so a claim made before it
    launches — its drive drains at once — unless ``drain_all`` stopped
    waiting for it (refused then). ``start_run`` never calls it once the
    shutdown began. ``route``: the node the action's state write ran as
    (the re-plan's ``intake``, the fork's ``deliver``), or for a plain
    resume the one ``_ROUTES`` gives for where it rests — the cancel note's
    node while no node of this drive has finished."""
    h.starting = False
    if _launch_closed:
        # ponytail: a preparation that outlasts ARGUS_SHUTDOWN_DRAIN_S keeps its writes and never
        # launches; roll them back here if that bound is ever too short
        return _shutdown(h.run_id)
    if h.deadline is not None:
        h.deadline.cancel()                         # an earlier drive's deadline never outlives it
    h.control, h.deadline, h.gate_kind, h.cancelled_by = RunControl(), None, None, None
    h.stage, h.done_node, h.started_at, h.announced, h.route = stage, "", time.time(), set(), route
    pending = common._pending_reply.get(h.chat_id)
    if pending is not None and pending[1] == h.run_id:
        del common._pending_reply[h.chat_id]        # an Edit / Revise prompt this drive made moot
    h.card = TelegramCard(_app.bot, h.chat_id, h.run_id, bus,
                          title=f"🔄 Run {h.run_id} — {_cut(h.topic, 60)}")
    h.task = asyncio.create_task(_drive(h, resume_input), name=f"run {h.run_id}")
    logger.info("run %s: task started (%s)", h.run_id, stage)
    if _shutting_down:                              # claimed before drain_all began, which drains it
        _drain(h, "shutdown")
    return None


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
    except llm.QuotaExhausted as e:
        await _paused(h, e)
    except Exception as e:
        await _failed(h, e)
    finally:
        await _stop_deadline(h)                     # one armed after the stream returned
        await _record_spend(h)
        if h.gate_kind is not None and h.cancelled_by is not None:
            # a cancel accepted after the stream returned (the gate presentation, the report
            # callback, the spend write): no boundary is left to drain at — cancel at the gate
            await _cancel_at_gate(h, h.cancelled_by)
        bus.finish(h.run_id)
        if h.gate_kind is None:
            _forget(h)


async def _stream(h: RunHandle, resume_input: Any) -> None:
    """Drive to the next gate, the end or a drain; the nodes' custom events
    go to the bus as they arrive. Each task's stage transitions are published
    once (``_announce``): the driver's ``start`` per task and ``end`` per
    update, a node's own ``start`` / ``draining`` only if the driver's did not
    come first, the driver's ``end`` only if the node did not publish its own."""
    async for mode, chunk in h.graph.astream(resume_input, h.cfg, stream_mode=["tasks", "updates", "custom"],
                                             durability="sync", control=h.control):
        if mode == "custom":
            kind, node, data = chunk.get("kind") or "custom", chunk.get("node") or h.stage, chunk.get("data") or {}
            if kind == "stage" and data.get("status") in ("start", "draining", "end") \
                    and not _announce(h, node, data["status"]):
                continue
            bus.publish(h.run_id, kind, node, data)
        elif mode == "tasks" and "input" in chunk:            # a task starts (its result chunk is ignored)
            h.stage = chunk["name"]
            if _announce(h, h.stage, "start"):
                bus.publish(h.run_id, "stage", h.stage, {"status": "start", "summary": ""})
        elif mode == "updates":
            for node, delta in chunk.items():
                if node.startswith("__"):                       # __interrupt__ at a gate
                    continue
                h.done_node = node
                if _announce(h, node, "end"):
                    bus.publish(h.run_id, "stage", node, {"status": "end", "summary": _summary(node, delta or {})})
                h.announced = {k for k in h.announced if k[0] != node}    # its next task announces anew


def _announce(h: RunHandle, node: str, status: str) -> bool:
    """True the first time the running task of ``node`` reaches ``status``."""
    if (node, status) in h.announced:
        return False
    h.announced.add((node, status))
    return True


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
    """The drive stopped without ``GraphDrained``: at a gate (where a requested
    drain still wins) or at the end."""
    snap = await h.graph.aget_state(h.cfg)
    values, nxt = snap.values or {}, tuple(snap.next or ())
    kind = _gate(nxt, values)
    if kind is not None and h.control.drain_requested:
        # a gate interrupt fires before the drain boundary (interrupt_after scout):
        # the cancel still wins over presenting the gate
        await _drained(h, h.control.drain_reason)
    elif kind == PLAN:
        if not values.get("plan"):
            raise RuntimeError("the brief produced no plan")
        await h.card.stop("📋 Plan ready — review it below.")
        if h.choices is None:
            _attached_choices(h, require(values), (await registry.active_snapshot()).profile)
        await _present_plan(h, values)
    elif kind == REPORT:
        await h.card.stop("📝 Report ready — preview below.")
        await _present_report(h, values)
        if h.on_report is not None:
            callback, h.on_report = h.on_report, None
            try:
                await callback()
            except Exception as e:
                logger.exception("run %s: the report-gate callback failed", h.run_id)
                await _say(h.chat_id, f"⚠️ Could not record run {h.run_id}'s appended sources: {e}")
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
    """The drive drained (``GraphDrained``, or a drain that won over a gate).
    The note never moves ``next``: it is written as the last node this drive
    finished, else as ``route`` (the node its launching action wrote as; a
    plain resume's from where it rested), else as LangGraph infers it — the
    last node that ran, right for a resting state a real node produced.
    LangGraph's guess re-runs that node's edges, which took a re-plan, a fork
    or a resume of either, drained before its first node, back to its old
    gate. A fresh start drained before its first step rests at
    ``("__start__",)`` with only its input checkpoint: no node ran, so there
    is none to write the note as (LangGraph's "Ambiguous update") — no note,
    and the card says nothing was kept. The card's hint follows ``next``."""
    done = h.done_node
    note = f"cancelled after node {done} ({reason})" if done else f"cancelled before it started ({reason})"
    snap = await h.graph.aget_state(h.cfg)
    fresh = not done and h.route is None and tuple(snap.next or ()) == ("__start__",)   # only the input is saved
    if not fresh:
        await h.graph.aupdate_state(h.cfg, {"cancel_note": note}, as_node=done or h.route)
        snap = await h.graph.aget_state(h.cfg)
    nxt = () if fresh else tuple(snap.next or ())
    kind = _gate(nxt, snap.values or {})
    logger.info("run %s: %s", h.run_id, note)
    await _status(h.run_id, "cancelled")
    bus.publish(h.run_id, "stage", done or h.stage, {"status": "cancelled", "summary": note})
    head = f"⏹ Cancelled after {done}" if done else "⏹ Cancelled before it started"
    # a quick /ask is never told to /continue: continue_run refuses it
    hint = ("nothing was kept; start it again with /research" if not nxt
            else _drained_hint(h.run_id, nxt, kind))
    await h.card.stop(head + ("." if h.quick else f" — {hint}."))


def _drained_hint(run_id: str, nxt: tuple, kind: str | None) -> str:
    """What /continue does for a drained run resting at ``nxt`` (never empty:
    a drain leaves a task) — at gate ``kind`` (``_gate``) its card is
    re-attached, where the action is repeated; elsewhere the drive resumes."""
    if kind is not None:
        return f"/continue {run_id} re-attaches the {kind} card — repeat the action there"
    return f"/continue {run_id} resumes it from {nxt[0]}"


async def _abandoned(h: RunHandle) -> None:
    """``task.cancel()`` after the drain bound or the shutdown bound: the
    node's thread may still run and the node never wrote, so the checkpoint
    is the one before it (``next`` names that node). The note names the
    forced cancel and, written as ``_drained`` writes it, keeps ``next``;
    the card says what /continue then does. A failure to record the note is
    logged and said in the chat — the run still ends cancelled and
    ``_drive`` re-raises the ``CancelledError``."""
    node = h.stage
    note = f"force-cancelled during node {node} ({h.control.drain_reason or 'cancel'}); the node was abandoned"
    nxt = kind = None
    try:
        await h.graph.aupdate_state(h.cfg, {"cancel_note": note}, as_node=h.done_node or h.route)
        snap = await h.graph.aget_state(h.cfg)
        nxt = tuple(snap.next or ())
        kind = _gate(nxt, snap.values or {})
    except Exception as e:
        logger.error("run %s: could not record the cancel note", h.run_id, exc_info=e)
        await _say(h.chat_id, f"⚠️ Run {h.run_id}: could not record the cancel note: {e}")
    logger.info("run %s: %s", h.run_id, note)
    await _status(h.run_id, "cancelled")
    bus.publish(h.run_id, "stage", node, {"status": "cancelled", "summary": note})
    hint = "" if h.quick else f" — {_continue_hint(h.run_id, nxt, kind)}"     # continue_run refuses a quick /ask
    await h.card.stop(f"⏹ Force-cancelled during {node}{hint}.")


def _continue_hint(run_id: str, nxt: tuple | None, kind: str | None) -> str:
    """What /continue does for a run resting at ``nxt`` (None: unread), at gate
    ``kind`` (``_gate``) or none."""
    if nxt is None:
        return f"/continue {run_id} resumes it"
    if kind == PLAN:
        return f"/continue {run_id} re-attaches the plan; ✅ Approve reruns {nxt[0]}"
    if kind == REPORT:
        return f"/continue {run_id} re-attaches the report preview"
    return f"/continue {run_id} reruns {nxt[0]}" if nxt else f"/continue {run_id} extends it"


async def _failed(h: RunHandle, e: Exception) -> None:
    h.gate_kind = None
    logger.error("run %s: the drive failed at %s", h.run_id, h.stage, exc_info=e)
    bus.publish(h.run_id, "stage", h.stage, {"status": "error", "summary": f"{type(e).__name__}: {e}"})
    await h.card.stop(f"⚠️ Failed at {h.stage or 'start'}.")
    await _status(h.run_id, "error")
    await _say(h.chat_id, f"⚠️ Research failed: {e}")


async def _paused(h: RunHandle, e: llm.QuotaExhausted) -> None:
    """Every engine of a call is out of quota: the node raised, so the
    checkpoint is the one before it and /continue reruns it — never the
    hollow report skipping item after item would give."""
    h.gate_kind = None
    logger.warning("run %s: paused at %s: %s", h.run_id, h.stage, e)
    bus.publish(h.run_id, "stage", h.stage, {"status": "paused", "summary": str(e)})
    await h.card.stop(f"⏸ Paused at {h.stage or 'start'}: the model quota ran out.")
    await _status(h.run_id, "paused")
    if h.quick:                                     # continue_run refuses a quick /ask
        again = "ask again"
    else:
        snap = await h.graph.aget_state(h.cfg)
        nxt = tuple(snap.next or ())
        again = _continue_hint(h.run_id, nxt, _gate(nxt, snap.values or {}))
    await _say(h.chat_id, f"⏸ Run {h.run_id} paused: every model engine is out of quota ({e.role}). "
                          f"Once the quota resets, {again}.")


async def _record_spend(h: RunHandle) -> None:
    """``runs.exa_usd`` at every drive end (report gate, cancelled, error):
    the month-to-date total counts failed runs' spend too. The same write
    carries the brief's archetype once the brief exists. A failed write is
    visible but changes nothing else."""
    lib = _library()
    if lib is None:
        return
    try:
        values = (await h.graph.aget_state(h.cfg)).values or {}
        cols: dict[str, Any] = {"exa_usd": (values.get("search_spend") or {}).get("exa_usd")}
        if values.get("brief"):
            cols["archetype"] = ResearchBrief.model_validate(values["brief"]).archetype
        if not await lib.update_run(h.run_id, **cols):
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
    if _announce(h, h.stage, "draining"):
        bus.publish(h.run_id, "stage", h.stage, {"status": "draining", "summary": reason})


async def _drain_deadline(h: RunHandle, task: asyncio.Task, seconds: float) -> None:
    """The cancel's last resort, for the drive it was armed for only."""
    await _sleep(seconds)
    if not task.done():
        logger.warning("run %s: the drain did not finish within %g s; task.cancel() — it cannot stop the "
                       "node's running thread, only the wait for it", h.run_id, seconds)
        task.cancel()


async def _cancel_at_gate(h: RunHandle, actor: str) -> None:
    await _status(h.run_id, "cancelled")
    await _edit_gate(h, "❌ Cancelled" + (" from console." if actor == "webapp" else "."))
    _forget(h)


async def cancel_run(run_id: str, *, actor: str = "telegram", message_id: int | None = None,
                     card: str | None = None) -> Outcome:
    """A live drive drains (finishes its current step); a run at a gate is
    cancelled now. ``card``: the gate of the tapped card's button — a stale
    card of the other gate is refused, as its other buttons are."""
    async with _hold(run_id, "cancel", message_id=message_id, card=card) as h:
        if isinstance(h, Outcome):
            return h
        if card is not None and not h.live and h.gate_kind != card:
            return Outcome("refused", f"Run `{run_id}` is not at its "
                                      f"{'plan gate' if card == PLAN else 'report preview'}.", run_id)
        if h.live:
            if h.control.drain_requested:
                return Outcome("draining", f"⏹ Run `{run_id}` is already cancelling — finishing `{h.stage}`.", run_id)
            h.cancelled_by = actor
            _drain(h, "user cancel")
            h.deadline = asyncio.create_task(_drain_deadline(h, h.task, get_settings().cancel_drain_timeout_s),
                                             name=f"run {run_id} drain deadline")
            return Outcome("draining", f"⏹ Cancelling run `{run_id}` — finishing the current step (`{h.stage}`).",
                           run_id)
        await _cancel_at_gate(h, actor)
        return Outcome("cancelled", f"❌ Cancelled run `{run_id}`.", run_id)


async def drain_all(app) -> None:
    """POST_STOP hook: refuse new claims from now on; wait for the actions
    claimed before, which still prepare (each resume launches its drive,
    which drains at once; a start is refused at its launch), then drain
    every live drive — all within ``ARGUS_SHUTDOWN_DRAIN_S`` — then cancel
    what still runs. No action ends with its writes and no drive (a
    preparation past the bound aside)."""
    global _shutting_down, _launch_closed
    _shutting_down = True
    loop = asyncio.get_running_loop()
    bound = get_settings().shutdown_drain_s
    deadline = loop.time() + bound
    while any(h.starting for h in common._inflight.values()) and loop.time() < deadline:
        await asyncio.sleep(0.05)
    _launch_closed = True
    for h in common._inflight.values():
        if h.starting:
            logger.warning("run %s: still preparing %g s after the shutdown began; it will not launch",
                           h.run_id, bound)
    live = [h for h in list(common._inflight.values()) if h.task is not None and not h.task.done()]
    if not live:
        return
    for h in live:
        if not h.control.drain_requested:
            _drain(h, "shutdown")
    _, pending = await asyncio.wait([h.task for h in live], timeout=max(deadline - loop.time(), 0))
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
                    engines_profile: str | None = None, origin: str = "telegram") -> Outcome:
    """/research (``quick``: /ask) and ``POST /api/runs``: a new run and its
    first drive — to the plan gate, or (quick) to the answer. An
    ``engines_profile`` other than the active one is validated like the plan
    card's profile; it is the run's from the plan drive on and the card's
    profile choice."""
    if _app is None or ("quick_graph" if quick else "graph") not in _app.bot_data:
        return Outcome("refused", "⚠️ Bot not fully initialized (no graph) — restart the bot.")
    if _shutting_down:
        return _shutdown()
    busy = _busy()
    if busy is not None:
        return busy
    run_id = new_run_id()
    h = RunHandle(run_id=run_id, chat_id=chat_id, thread_id=f"tg:{chat_id}:{run_id}", topic=topic, quick=quick)
    h.starting = True                                  # claimed before the first await: a second start is busy
    common._inflight[run_id] = h
    async with h.lock:
        try:
            engines = active = await registry.active_snapshot()
            if engines_profile is not None:
                cfg = registry.current_config()
                if engines_profile not in cfg.profiles:
                    raise ValueError(f"unknown engines profile {engines_profile!r}")
                if engines_profile != cfg.active_profile:
                    engines = await _resolve_profile(h, engines_profile)
            rc = RunConfig.from_preset(length=length, depth=depth, engines=engines, overrides=overrides,
                                       origin=origin)
            h.choices = _default_choices(rc, active.profile)
            lib = _library()
            if lib is not None:
                await lib.create_run(run_id=run_id, thread_id=h.thread_id, chat_id=chat_id, topic=topic,
                                     length=rc.length, mode="quick" if quick else "deep",
                                     status="running" if quick else "planning", config_json=rc.model_dump_json(),
                                     depth=rc.depth, origin=rc.origin)
            state_in = {
                "thread_id": h.thread_id, "user_id": user_id, "user_request": topic, "length": rc.length,
                "run_config": rc.model_dump(), "messages": [], "plan": None, "sources": [], "fetched": [],
                "findings": [], "draft_md": "", "revision_notes": [], "revision_rounds": 0, "model_calls": [],
                "hitl": {"pending": False},
            }
            # a start never launches into the shutdown: a fresh thread drained before its first node
            # keeps no input and could not be resumed (the refusal below marks the row cancelled)
            refused = _shutdown(run_id) if _shutting_down else _launch(h, state_in, stage="intake")
        except Exception as e:
            _forget(h)
            if not isinstance(e, (EngineConfigError, ValueError)):
                logger.error("run %s: start failed", run_id, exc_info=e)
            return Outcome("refused", f"⚠️ Could not start the research: {e}")
        finally:
            h.starting = False
        if refused is not None:                        # the shutdown began while it prepared: never driven
            await _status(run_id, "cancelled")
            _forget(h)
            return refused
    return Outcome("started", "", run_id)


# ---------------------------------------------------------------------------
# the plan gate
# ---------------------------------------------------------------------------

def _default_choices(rc: RunConfig, active: str) -> Choices:
    """The plan-gate knobs ``rc`` was built with; its engines profile is None
    when it is ``active`` (the active profile's name)."""
    return Choices(length=rc.length, depth=None if rc.depth == RunConfig.default_depth(rc.length) else rc.depth,
                   profile=None if rc.engines.profile == active else rc.engines.profile, recency=rc.recency)


def _attached_choices(h: RunHandle, rc: RunConfig, active: str) -> None:
    """The knobs of a run attached at its plan gate (a tap or ``/continue``
    after a restart, a resume reaching the gate): those it was built with.
    Its own profile gone from the engines file is cached as its problem, as
    a cycled profile's would be (a static check): the card shows it, and
    Approve's live check refuses it until another profile is chosen."""
    h.choices = _default_choices(rc, active)
    name = h.choices.profile
    if name is not None and name not in h.profile_checks:
        try:
            registry.bindings(registry.current_config(), name)
        except EngineConfigError as e:
            h.profile_checks[name] = str(e)


def _rebuilt(c: Choices, rc: RunConfig, engines) -> RunConfig:
    """``rc`` for the plan-gate choices ``c``, through ``rebuild`` (console
    overrides survive); the recency knob is the ``recency`` override."""
    overrides = {k: v for k, v in rc.overrides_applied.items() if k != "recency"}
    if c.recency is not None:
        overrides["recency"] = c.recency
    return rc.model_copy(update={"overrides_applied": overrides}).rebuild(length=c.length, depth=c.depth,
                                                                           engines=engines)


async def _resolve_profile(h: RunHandle, name: str):
    """A non-active profile validated live (run start, Approve); the result, a
    failure's problem text included, is cached for the card's renders."""
    try:
        snap = await registry.resolve_snapshot(registry.current_config(), name)
    except EngineConfigError as e:
        h.profile_checks[name] = str(e)
        raise
    h.profile_checks[name] = snap
    return snap


async def _chosen_engines(h: RunHandle, c: Choices):
    """Approve's engines snapshot, validated now: the active profile's
    (``active_snapshot``), another one's live."""
    if c.profile is None:
        return await registry.active_snapshot()
    return await _resolve_profile(h, c.profile)


async def _payload(h: RunHandle, values: dict) -> dict:
    """The plan card's data (also the console's): plan, sources, knobs, Exa
    estimate. A render never validates live: the active profile's state is
    the cached snapshot, another profile's the result cached on the handle
    (none yet for a re-attached run's own profile: no problem shown)."""
    c, rc = h.choices, require(values)
    brief = ResearchBrief.model_validate(values.get("brief") or {})
    cfg = registry.current_config()
    profile = c.profile or cfg.active_profile
    if c.profile is None:
        problem = (None if registry.cached_snapshot() is not None
                   else "not validated yet — Approve validates it")
    else:
        check = h.profile_checks.get(profile)
        problem = check if isinstance(check, str) else None
    error = None
    try:
        preview = _rebuilt(c, rc, rc.engines)
    except ValueError as e:
        preview, error = rc, str(e)
    # pointer 9: the estimate prices searches even with Exa off — show it only for an enabled Exa
    exa_on = exa_enabled() and "exa" in preview.search.providers_enabled
    estimate = preview.exa_cost_estimate_usd(cfg.pricing.exa) if exa_on and error is None else None
    return {
        "run_id": h.run_id, "topic": h.topic, "plan": values.get("plan") or {},
        "sources": list(values.get("sources") or []),
        "archetype": brief.archetype if values.get("brief") else None,
        "length": c.length, "depth": c.depth or RunConfig.default_depth(c.length), "depth_auto": c.depth is None,
        "profile": profile, "profile_problem": problem, "profiles": list(cfg.profiles),
        "recency": c.recency,
        "recency_window": resolve_recency(rc.model_copy(update={"recency": c.recency}), brief)[0],
        "exa_enabled": exa_on, "exa_estimate_usd": estimate, "error": error,
    }


async def plan_payload(run_id: str) -> dict | None:
    """The plan card's data for a run resting at its plan gate, else None. An
    untracked run (after a restart) is read only: its library row and its
    checkpoint, with the choices an attach would give it (``_attached_choices``);
    nothing is written or registered. A legacy checkpoint (no ``run_config``)
    or a run not at the plan gate gives None."""
    h = common._inflight.get(run_id)
    if h is not None:
        if h.live or h.gate_kind != PLAN:
            return None
        return await _payload(h, (await h.graph.aget_state(h.cfg)).values or {})
    lib = _library()
    row = await lib.get_run(run_id) if lib is not None else None
    if row is None or row.get("mode") == "quick":
        return None
    h = RunHandle.of(row, row["chat_id"])             # a view, not registered
    snap = await h.graph.aget_state(h.cfg)
    values = snap.values or {}
    nxt = tuple(snap.next or ())
    if nxt != ("research",) or _gate(nxt, values) != PLAN or not values.get("run_config"):
        return None
    _attached_choices(h, require(values), registry.current_config().active_profile)
    return await _payload(h, values)


async def _present_plan(h: RunHandle, values: dict) -> None:
    from . import research                          # lazy: research imports this module
    payload = await _payload(h, values)
    text = research._format_plan(payload)
    # never silent: a card Telegram still refuses becomes a short one with the same buttons
    fallback = (f"📋 Research plan for run {h.run_id} is ready, but Telegram refused the full card. "
                "Approve, Edit or Cancel below; a knob tap re-renders it.")
    msg = await _safe_send(_app.bot, h.chat_id, text, reply_markup=research._plan_keyboard(payload),
                           parse_mode=ParseMode.MARKDOWN, fallback=fallback)
    h.gate_msg = (msg.message_id, text, ParseMode.MARKDOWN) if msg is not None else None
    await _status(h.run_id, "awaiting_plan")
    h.gate_kind = PLAN


async def _rerender_plan(h: RunHandle) -> None:
    from . import research                          # lazy: research imports this module
    payload = await _payload(h, (await h.graph.aget_state(h.cfg)).values or {})
    if h.gate_msg is not None:
        h.gate_msg = (h.gate_msg[0], research._format_plan(payload), h.gate_msg[2])
        await _edit_gate(h, markup=research._plan_keyboard(payload))


async def _choose(h: RunHandle, c: Choices, knob: str, value: str) -> None:
    """One knob onto ``c``; a bad value raises ValueError before anything changes."""
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
        if value == "next":                         # a profile gone from the engines file: from the first
            current = c.profile or cfg.active_profile
            value = names[(names.index(current) + 1) % len(names)] if current in names else names[0]
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


async def set_plan_choice(run_id: str, knob: str, value: str, *, message_id: int | None = None) -> Outcome:
    """A plan-card knob (length, depth, profile, recency; "next" cycles the
    last two): the card is re-rendered; nothing resumes."""
    async with _hold(run_id, "update", message_id=message_id, card=PLAN) as h:
        if isinstance(h, Outcome):
            return h
        if h.live:
            return _running(h)
        if h.gate_kind != PLAN:
            return Outcome("refused", f"Run `{run_id}` is not at its plan gate.", run_id)
        try:
            await _choose(h, h.choices, knob, value)
        except ValueError as e:
            return Outcome("refused", f"⚠️ {e}", run_id)
        await _rerender_plan(h)
        return Outcome("updated", "", run_id)


async def approve_plan(run_id: str, *, actor: str = "telegram", length: str | None = None,
                       depth: str | None = None, engines_profile: str | None = None,
                       message_id: int | None = None) -> Outcome:
    """Approve: the RunConfig rebuilt for the choices (``rc.rebuild``) into the
    checkpoint and the library row — everything resolved first (the console's
    knobs reach the choices only then), so a refusal leaves the run at its
    gate as it was — then the research drive."""
    async with _hold(run_id, "approve", message_id=message_id, card=PLAN) as h:
        if isinstance(h, Outcome):
            return h
        if h.live:
            return _running(h)
        if h.gate_kind != PLAN:
            return Outcome("refused", f"Run `{run_id}` is not at its plan gate.", run_id)
        refused = _claim(h)
        if refused is not None:
            return refused
        try:
            chosen = dataclasses.replace(h.choices)
            for knob, value in (("length", length), ("depth", depth), ("profile", engines_profile)):
                if value is not None:
                    await _choose(h, chosen, knob, value)
            engines = await _chosen_engines(h, chosen)
            values = await _resumable(h, engines)
            notice = await _appends_notice(h)
            old = require(values)
            rc = _rebuilt(chosen, old, engines)
            update: dict[str, Any] = {"length": rc.length, "run_config": rc.model_dump()}
            if rc.recency != old.recency:              # the scout decided recency once; research reuses it
                brief = ResearchBrief.model_validate(values.get("brief") or {})
                update["search_spend"] = {**(values.get("search_spend") or {}),
                                          "recency": resolve_recency(rc, brief)[0]}
            await h.graph.aupdate_state(h.cfg, update)
            lib = _library()
            if lib is not None and not await lib.set_run_config(run_id, rc.model_dump()):
                logger.warning("run %s has no library row; its run_config is in the checkpoint only", run_id)
            h.choices = chosen
            bus.publish(run_id, "gate", "", {"kind": PLAN, "actor": actor})
            await _edit_gate(h, "✅ approved from console" if actor == "webapp" else "_✅ Approved — continuing._")
            await _status(run_id, "running")
            return _launch(h, Command(resume=True), stage="research") or Outcome("started", notice, run_id)
        except Exception as e:
            return _refused(run_id, "approve", e)
        finally:
            h.starting = False


async def edit_plan(run_id: str, feedback: str, *, actor: str = "telegram") -> Outcome:
    """Plan Edit: the feedback goes into the checkpoint as if intake just ran,
    then brief → scout re-drafts the plan to a new plan gate."""
    async with _hold(run_id, "re-plan") as h:
        if isinstance(h, Outcome):
            return h
        if h.live:
            return _running(h)
        if h.gate_kind != PLAN:
            return Outcome("refused", f"Run `{run_id}` is not at its plan gate.", run_id)
        if not feedback.strip():
            return Outcome("refused", "Please send some text.", run_id)
        refused = _claim(h)
        if refused is not None:
            return refused
        try:
            await _resumable(h)
            notice = await _appends_notice(h)
            await h.graph.aupdate_state(h.cfg, {"plan_feedback": feedback.strip()}, as_node="intake")
            bus.publish(run_id, "gate", "", {"kind": PLAN, "actor": actor})
            await _edit_gate(h, "_✏️ Redrafting the plan with your changes…_")
            await _status(run_id, "planning")
            return _launch(h, None, stage="brief", route="intake") or Outcome("started", notice, run_id)
        except Exception as e:
            return _refused(run_id, "re-plan", e)
        finally:
            h.starting = False


async def await_reply(run_id: str, kind: str, *, message_id: int | None = None) -> Outcome:
    """Plan Edit / report Revise buttons: the chat's next text is the reply
    (``kind`` "plan_edit" | "revise_feedback"); the gate keeps its keyboard.
    A reply that would be refused (busy, Revise's round cap) is refused here,
    before it is armed."""
    from . import research                          # lazy: research imports this module
    gate, prompt = _REPLY_PROMPTS[kind]
    async with _hold(run_id, "edit" if gate == PLAN else "revise", message_id=message_id, card=gate) as h:
        if isinstance(h, Outcome):
            return h
        if h.live:
            return _running(h)
        if h.gate_kind != gate:
            return Outcome("refused", f"Run `{run_id}` is not at its {gate} gate.", run_id)
        busy = _busy(exclude=run_id)
        if busy is not None:
            return busy
        values = (await h.graph.aget_state(h.cfg)).values or {}
        if gate == REPORT and int(values.get("revise_rounds") or 0) >= MAX_REVISE_ROUNDS:
            return _refused(run_id, "revise", ValueError(f"it already used its {MAX_REVISE_ROUNDS} Revise rounds"))
        common._pending_reply[h.chat_id] = (kind, run_id)
        markup = (research._plan_keyboard(await _payload(h, values)) if gate == PLAN
                  else research._report_keyboard(run_id))
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
                        actor: str = "telegram", message_id: int | None = None) -> Outcome:
    """Report gate: send · extend · revise (with ``feedback``) · cancel. An
    Extend runs its normal pass with the run's pending /append sources merged
    in as pinned sources; the next report gate marks them."""
    if action == "cancel":
        return await cancel_run(run_id, actor=actor, message_id=message_id)
    async with _hold(run_id, action, message_id=message_id, card=REPORT) as h:
        if isinstance(h, Outcome):
            return h
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
        refused = _claim(h)
        if refused is not None:
            return refused
        try:
            values = await _resumable(h)
            pending: list[dict] = []
            appended: list[dict] = []
            notice = ""
            if action == "extend":
                if int(values.get("extend_rounds") or 0) >= MAX_EXTEND_ROUNDS:
                    raise ValueError(f"it already used its {MAX_EXTEND_ROUNDS} Extend rounds")
                update: dict[str, Any] = {"extend_requested": True, "revision_requested": False}   # see _deliver
                pending, appended = await _pending_appends(h)
                if appended:                               # last-value channel: the full list
                    update["sources"] = _merged_sources(values.get("sources"), appended)
            else:
                if int(values.get("revise_rounds") or 0) >= MAX_REVISE_ROUNDS:
                    raise ValueError(f"it already used its {MAX_REVISE_ROUNDS} Revise rounds")
                notice = await _appends_notice(h)
                notes = [*(values.get("revision_notes") or []), f"USER REVISION REQUEST: {feedback.strip()}"]
                update = {"revision_requested": True, "extend_requested": False,   # see _deliver
                          "revision_notes": notes}                                 # last-value channel: the full list
            await h.graph.aupdate_state(h.cfg, update)
            if pending:
                h.on_report = _appends_marker(h, pending, appended)
            bus.publish(run_id, "gate", "", {"kind": REPORT, "actor": actor})
            await _edit_gate(h, ("🔎 Extending research — gathering more…" if action == "extend"
                                 else "🔁 Revising with your notes…") + (" (from console)" if actor == "webapp" else ""))
            await _status(run_id, "running")
            return _launch(h, Command(resume=True), stage="deliver") or Outcome("started", notice, run_id)
        except Exception as e:
            return _refused(run_id, action, e)
        finally:
            h.starting = False


async def _deliver(h: RunHandle, actor: str) -> Outcome:
    """Send: the report files to the chat, then the run is finished — the
    checkpoint moves past the report gate (``as_node="deliver"`` → END), so a
    sent run never awaits and /continue extends it. The engines snapshot must
    resolve first (pointer 6). Every report-gate action writes both request
    flags: a flag left by a drive drained before ``deliver`` ran (a
    shutdown) never routes a later Send, Extend or Revise astray."""
    try:
        await ensure_in_checkpoint(h.graph, h.cfg, engines=await registry.active_snapshot())
        values = (await h.graph.aget_state(h.cfg)).values or {}
        notice = await _appends_notice(h, sent=True)
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
    try:
        await h.graph.aupdate_state(h.cfg, {"extend_requested": False, "revision_requested": False},
                                    as_node="deliver")              # Send ends the run: next == ()
    except Exception as e:
        return _refused(h.run_id, "finish", e)                       # it still rests at its report gate
    bus.finish(h.run_id)                            # after the gate event, so the buffer still expires
    _forget(h)
    await _finalize(h, paths)
    return Outcome("delivered", notice, h.run_id)


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
    any, and no new searches); ``on_report`` runs at the next report gate. A
    ``done`` run still resting at its report gate (sent by an older bot) is
    first moved past it, then forked. A resume ingests the appends its
    checkpoint already carries (an Extend or a fork drained after its merge)
    and marks them at its report gate. A re-attach or resume that leaves
    appends unused says so."""
    run_id = row["run_id"]
    if row.get("mode") == "quick":
        return Outcome("refused", f"Run `{run_id}` was a quick /ask — nothing to continue.", run_id)
    async with _hold(run_id, "continue", row=row, chat_id=chat_id) as h:
        if isinstance(h, Outcome):
            return h
        if h.live:
            return _running(h)
        try:
            snap = await h.graph.aget_state(h.cfg)
            if snap.created_at is None or not snap.values:
                raise ValueError("no checkpoint found — it may predate the v2 upgrade or its checkpoint DB "
                                 "was deleted")
            nxt = tuple(snap.next or ())
            kind = _gate(nxt, snap.values)
            sent = kind == REPORT and row.get("status") == "done"     # an older bot's Send left it at its gate
            if kind is None or sent:
                refused = _claim(h)
                if refused is not None:
                    if h.gate_kind is None:
                        _forget(h)
                    return refused
                values = await _resumable(h)
                notice = ""                                                       # a fork ingests them
                if nxt and not sent:
                    carried_pending, carried = await _carried_appends(h, values)
                    if carried_pending:
                        h.on_report = _appends_marker(h, carried_pending, carried)
                    notice = await _appends_notice(h, carried=len(carried_pending))
                if sent:
                    await h.graph.aupdate_state(h.cfg, {}, as_node="deliver")    # finished, as Send now does
                    nxt, kind = (), None
                if not nxt:
                    merged = _merged_sources(values.get("sources"), appended or [])   # last-value channel
                    await h.graph.aupdate_state(h.cfg, {"sources": merged, "extend_requested": True,
                                                        "extend_rounds": 0, "revision_rounds": 0,
                                                        "append_only": bool(appended)}, as_node="deliver")
                    h.on_report = on_report
                await _status(run_id, "running")
                refused = _launch(h, None, stage=nxt[0] if nxt else "extend_prep",
                                  route=_ROUTES.get(nxt) if nxt else "deliver")    # a resume: from where it rests
                if refused is not None:
                    _forget(h)
                    return refused
            else:
                active = await registry.active_snapshot()
                rc = await ensure_in_checkpoint(h.graph, h.cfg, engines=active)
                values = (await h.graph.aget_state(h.cfg)).values or {}
                notice = await _appends_notice(h)
                if kind == PLAN:
                    if h.choices is None:
                        _attached_choices(h, rc, active.profile)
                    await _present_plan(h, values)
                else:
                    await _present_report(h, values)
        except Exception as e:
            if h.gate_kind is None:
                _forget(h)
            return _refused(run_id, "continue", e)
        finally:
            h.starting = False
        if kind is not None:
            return Outcome("attached", notice, run_id)
        if nxt:
            return Outcome("started", "\n".join(filter(None, (f"▶️ Resuming run `{run_id}` from `{nxt[0]}`…",
                                                              notice))), run_id)
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
    awaiting = None if h.live else _gate(tuple(nxt), snap.values or {})
    return {"run_id": run_id, "in_flight": h.live, "awaiting": awaiting,
            "next": nxt, "stage": h.stage, "started_at": h.started_at or None, "last_seq": bus.last_seq(run_id),
            "values": snap.values or {}}


__all__ = ["Outcome", "RunHandle", "bus", "init", "start_run", "approve_plan", "edit_plan", "await_reply",
           "set_plan_choice", "plan_payload", "report_action", "cancel_run", "continue_run", "snapshot",
           "console_url", "tracked", "drain_all"]
