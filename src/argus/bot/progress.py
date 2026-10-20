"""The run event bus and the Telegram progress card (spec §2.5).

``RunEventBus`` keeps each run's events in a ring buffer; readers poll
``since()`` — the Telegram card every 3 s, the console every 2 s. Nothing is
pushed. The run driver (``runs.py``) publishes on the event-loop thread only
(the nodes' custom stream events as they arrive, plus its own ``stage``
events), so the bus needs no lock.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from telegram.error import TelegramError

logger = logging.getLogger("argus.bot")

_MAXLEN = 500        # events kept per run
_KEEP_S = 3600.0     # a finished run's buffer is dropped this long after finish()
_COUNTED = {"fetch": "fetched", "digest": "digested", "section": "sections", "judge": "judge batches"}


@dataclass(frozen=True)
class RunEvent:
    seq: int
    ts: float
    run_id: str
    kind: str
    node: str
    data: dict


class RunEventBus:
    def __init__(self, *, maxlen: int = _MAXLEN, keep_s: float = _KEEP_S,
                 clock: Callable[[], float] = time.time) -> None:
        self._events: dict[str, deque[RunEvent]] = {}
        self._seq: dict[str, int] = {}          # outlives a dropped buffer: a resumed run's seq keeps rising
        self._finished: dict[str, float] = {}
        self._maxlen, self._keep_s, self._clock = maxlen, keep_s, clock

    def publish(self, run_id: str, kind: str, node: str, data: dict) -> RunEvent:
        self._expire()
        self._finished.pop(run_id, None)
        seq = self._seq[run_id] = self._seq.get(run_id, 0) + 1
        event = RunEvent(seq, self._clock(), run_id, kind, node, dict(data))
        self._events.setdefault(run_id, deque(maxlen=self._maxlen)).append(event)
        return event

    def since(self, run_id: str, after: int, limit: int = 200) -> list[RunEvent]:
        self._expire()
        return [e for e in self._events.get(run_id, ()) if e.seq > after][:limit]

    def last_seq(self, run_id: str) -> int:
        return self._seq.get(run_id, 0)

    def finish(self, run_id: str) -> None:
        """The run's drive ended: keep its buffer for an hour, then drop it
        (a later publish keeps it)."""
        self._finished[run_id] = self._clock()

    def _expire(self) -> None:
        now = self._clock()
        for run_id in [r for r, t in self._finished.items() if now - t >= self._keep_s]:
            del self._finished[run_id]
            self._events.pop(run_id, None)


def _host(url: str | None) -> str:
    return urlsplit(url or "").hostname or (url or "")[:60]


def describe(e: RunEvent) -> str:
    """One line for an event: the card's "Last:" line and /status."""
    d = e.data
    ok = "✓" if d.get("ok") else "✗"
    match e.kind:
        case "stage":
            return f"{e.node or 'run'} {d.get('status', '')}" + (f" — {d['summary']}" if d.get("summary") else "")
        case "search":
            return f"search wave {d.get('wave')} ({d.get('mode')}): {d.get('queries')} queries → {d.get('hits')} hits"
        case "fetch":
            how = f" ({d['fetcher']}, {d.get('ms')} ms)" if d.get("fetcher") else ""
            return f"fetch {ok} {_host(d.get('url'))}{how}"
        case "digest":
            return f"digest [{d.get('source_id')}] {ok} relevance {d.get('relevance')}"
        case "rerank":
            return f"rerank wave {d.get('wave')}: kept {d.get('kept')} of {d.get('candidates')}"
        case "section":
            return f"section “{d.get('title')}” " + ("reused" if d.get("reused") else ok)
        case "analysis":
            return f"analysis: {d.get('takeaways')} takeaways"
        case "judge":
            batch = f" batch {d['batch']}" if d.get("batch") is not None else ""
            return f"judge {d.get('name')}{batch}: {d.get('verdict')}"
        case "degrade":
            return f"degraded: {d.get('role')} → {d.get('degradation')}"
        case "gate":
            return f"{d.get('kind')} gate acted on from {d.get('actor')}"
    return f"{e.kind}: {d}"


class TelegramCard:
    """One drive's live progress message: every ``interval`` seconds it reads
    the run's new events with ``since()`` and edits its message when the text
    changed. Plain text: topics and URLs break Telegram's markdown."""

    def __init__(self, bot, chat_id: int, run_id: str, bus: RunEventBus, *, title: str,
                 interval: float = 3.0) -> None:
        self.bot, self.chat_id, self.run_id, self.bus, self.title = bot, chat_id, run_id, bus, title
        self.interval = interval
        self.after = bus.last_seq(run_id)        # this drive's events only
        self.stage = self.last = self.final = self.text = ""
        self.cancelling = False
        self.counts: Counter = Counter()
        self.failed: Counter = Counter()
        self.message_id: int | None = None
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._poll(), name=f"card {self.run_id}")

    async def _poll(self) -> None:
        while True:
            await self.refresh()
            await asyncio.sleep(self.interval)

    async def stop(self, final: str) -> None:
        """Stop polling; the card keeps its counts under ``final``."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.wait([task])           # never swallows a cancel aimed at the caller
            if not task.cancelled() and task.exception() is not None:
                logger.error("run %s: the progress card failed", self.run_id, exc_info=task.exception())
        self.final = final
        await self.refresh()

    async def refresh(self) -> None:
        for e in self.bus.since(self.run_id, self.after, limit=_MAXLEN):
            self.after = e.seq
            self._apply(e)
        text = self.render()
        if text != self.text:
            await self._show(text)

    def _apply(self, e: RunEvent) -> None:
        if e.node:
            self.stage = e.node
        if e.kind == "stage" and e.data.get("status") == "draining":
            self.cancelling = True
        if e.kind in _COUNTED:
            self.counts[e.kind] += 1
            self.failed[e.kind] += e.data.get("ok") is False or bool(e.data.get("skipped"))
        self.last = describe(e)

    def render(self) -> str:
        counts = " · ".join(f"{_COUNTED[k]} {n}" + (f" ({self.failed[k]} ✗)" if self.failed[k] else "")
                            for k, n in self.counts.items())
        if self.final:
            return "\n".join(x for x in (self.title, self.final, counts) if x)
        lines = [self.title, f"Stage: {self.stage or 'starting'}"]
        lines += [counts] if counts else []
        lines += [f"Last: {self.last}"] if self.last else []
        lines += ["⏹ cancelling — finishing the current step"] if self.cancelling else []
        return "\n".join(lines)

    async def _show(self, text: str) -> None:
        try:
            if self.message_id is None:
                self.message_id = (await self.bot.send_message(chat_id=self.chat_id, text=text)).message_id
            else:
                await self.bot.edit_message_text(chat_id=self.chat_id, message_id=self.message_id, text=text)
        except TelegramError as e:              # retried on the next poll; the run goes on
            logger.warning("run %s: progress card update failed (%s)", self.run_id, e)
            return
        self.text = text


__all__ = ["RunEvent", "RunEventBus", "TelegramCard", "describe"]
