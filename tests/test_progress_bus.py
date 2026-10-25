"""The run event bus and the Telegram progress card (spec §2.5, WP-E1).

One consumption model: readers poll ``since()``. The bus keeps a ring buffer
per run (500 events) and drops a finished run's buffer an hour after
``finish()``; the card polls every 3 s and edits one message.
"""
from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest, RetryAfter, TelegramError

from argus.bot import progress
from argus.bot.progress import RunEventBus, TelegramCard, describe


class Clock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t


def _bot() -> MagicMock:
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=41))
    bot.edit_message_text = AsyncMock()
    return bot


# ---------------------------------------------------------------------------
# the bus
# ---------------------------------------------------------------------------

def test_seq_increments_per_run_and_since_returns_what_follows():
    bus = RunEventBus(clock=Clock())
    a1 = bus.publish("a", "stage", "intake", {"status": "end", "summary": ""})
    b1 = bus.publish("b", "fetch", "research", {"ok": True})
    a2 = bus.publish("a", "fetch", "research", {"ok": False})
    assert (a1.seq, a2.seq, b1.seq) == (1, 2, 1)
    assert [e.seq for e in bus.since("a", 0)] == [1, 2]
    assert [e.seq for e in bus.since("a", 1)] == [2]
    assert bus.since("a", 2) == []
    assert (bus.last_seq("a"), bus.last_seq("b"), bus.last_seq("unknown")) == (2, 1, 0)
    assert (a2.run_id, a2.kind, a2.node, a2.data, a2.ts) == ("a", "fetch", "research", {"ok": False}, 1_000.0)


def test_since_keeps_order_and_honours_the_limit():
    bus = RunEventBus()
    for i in range(10):
        bus.publish("a", "digest", "research", {"source_id": i})
    assert [e.seq for e in bus.since("a", 3, limit=4)] == [4, 5, 6, 7]
    assert len(bus.since("a", 0)) == 10                        # default limit 200


def test_since_returns_at_most_200_events_by_default():
    bus = RunEventBus()
    for i in range(250):
        bus.publish("a", "fetch", "research", {"i": i})
    assert [e.seq for e in bus.since("a", 0)] == list(range(1, 201))


def test_the_ring_buffer_keeps_the_last_500_events():
    bus = RunEventBus()
    for i in range(600):
        bus.publish("a", "fetch", "research", {"i": i})
    events = bus.since("a", 0, limit=1_000)
    assert len(events) == 500
    assert (events[0].seq, events[-1].seq) == (101, 600)
    assert bus.last_seq("a") == 600


def test_finish_keeps_the_buffer_an_hour_then_drops_it_and_seq_keeps_rising():
    clock = Clock()
    bus = RunEventBus(clock=clock)
    bus.publish("a", "stage", "scout", {"status": "end"})
    bus.finish("a")
    clock.t += 3_599
    assert [e.seq for e in bus.since("a", 0)] == [1]
    clock.t += 2
    assert bus.since("a", 0) == []
    assert bus.publish("a", "stage", "research", {"status": "start"}).seq == 2   # a resumed run


def test_a_publish_after_finish_revives_the_buffer():
    clock = Clock()
    bus = RunEventBus(clock=clock)
    bus.publish("a", "stage", "scout", {"status": "end"})
    bus.finish("a")                                 # the drive reached a gate
    clock.t += 60
    bus.publish("a", "gate", "", {"kind": "plan", "actor": "telegram"})   # the next drive
    clock.t += 7_200
    assert [e.seq for e in bus.since("a", 0)] == [1, 2]


def test_events_are_frozen_and_keep_their_own_data():
    bus = RunEventBus()
    data = {"ok": True}
    event = bus.publish("a", "fetch", "research", data)
    data["ok"] = False
    assert event.data == {"ok": True}
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.seq = 9                                               # type: ignore[misc]


@pytest.mark.parametrize("kind,node,data,expect", [
    ("stage", "research", {"status": "start", "summary": "first pass"}, "research start — first pass"),
    ("search", "research", {"wave": 2, "mode": "gap", "queries": 3, "hits": 17, "errors": 0},
     "search wave 2 (gap): 3 queries → 17 hits"),
    ("fetch", "research", {"url": "https://www.example.com/a?b=1", "ok": True, "fetcher": "native",
                           "ms": 812, "pinned": False}, "fetch ✓ www.example.com (native, 812 ms)"),
    ("fetch", "research", {"url": "https://x.org/p", "ok": False, "fetcher": "", "ms": 0, "pinned": True},
     "fetch ✗ x.org"),
    ("digest", "research", {"source_id": 4, "ok": True, "relevance": 5, "cached": False},
     "digest [4] ✓ relevance 5"),
    ("rerank", "research", {"wave": 1, "candidates": 40, "kept": 14}, "rerank wave 1: kept 14 of 40"),
    ("section", "compose", {"title": "Setup", "ok": True, "chars": 900, "reused": False}, "section “Setup” ✓"),
    ("section", "compose", {"title": "Setup", "ok": True, "chars": 900, "reused": True}, "section “Setup” reused"),
    ("analysis", "analyst", {"takeaways": 4, "has_disagreement": False, "has_scenarios": True},
     "analysis: 4 takeaways"),
    ("judge", "panel", {"name": "grounding", "batch": 2, "verdict": "1 flagged", "skipped": False},
     "judge grounding batch 2: 1 flagged"),
    ("judge", "panel", {"name": "coverage", "verdict": "pass", "skipped": False}, "judge coverage: pass"),
    ("degrade", "intake", {"role": "intake.classify", "degradation": "default_classification", "reason": "429"},
     "degraded: intake.classify → default_classification"),
    ("gate", "", {"kind": "plan", "actor": "webapp"}, "plan gate acted on from webapp"),
])
def test_describe_reads_every_event_kind(kind, node, data, expect):
    event = RunEventBus().publish("a", kind, node, data)
    assert describe(event) == expect


# ---------------------------------------------------------------------------
# the Telegram card
# ---------------------------------------------------------------------------

def test_the_card_polls_every_3_seconds_by_default():
    assert TelegramCard(_bot(), 7, "a", RunEventBus(), title="t").interval == 3.0


async def test_the_card_shows_the_stage_counts_last_event_and_cancelling():
    bus, bot = RunEventBus(), _bot()
    card = TelegramCard(bot, 7, "a", bus, title="🔄 Run a — topic")
    bus.publish("a", "stage", "research", {"status": "start", "summary": "first pass"})
    bus.publish("a", "fetch", "research", {"url": "https://example.com/x", "ok": True, "fetcher": "native",
                                           "ms": 812, "pinned": False})
    bus.publish("a", "fetch", "research", {"url": "https://bad.example/y", "ok": False, "fetcher": "",
                                           "ms": 0, "pinned": False})
    await card.refresh()
    bot.send_message.assert_awaited_once()
    text = bot.send_message.await_args.kwargs["text"]
    assert text.startswith("🔄 Run a — topic\n")
    assert "Stage: research" in text and "fetched 2 (1 ✗)" in text
    assert "Last: fetch ✗ bad.example" in text
    assert "cancelling" not in text

    bus.publish("a", "stage", "research", {"status": "draining", "summary": "user cancel"})
    await card.refresh()
    kw = bot.edit_message_text.await_args.kwargs
    assert (kw["chat_id"], kw["message_id"]) == (7, 41)
    assert "⏹ cancelling — finishing the current step" in kw["text"] == card.text

    await card.refresh()                                       # nothing new: no edit
    assert bot.edit_message_text.await_count == 1


async def test_the_card_shows_only_its_own_drive_events():
    bus, bot = RunEventBus(), _bot()
    bus.publish("a", "stage", "scout", {"status": "end", "summary": "an earlier drive"})
    card = TelegramCard(bot, 7, "a", bus, title="t")
    bus.publish("a", "stage", "research", {"status": "start", "summary": "first pass"})
    await card.refresh()
    assert "earlier drive" not in card.text and "Stage: research" in card.text


async def test_stop_ends_the_polling_and_leaves_the_final_line():
    bus, bot = RunEventBus(), _bot()
    card = TelegramCard(bot, 7, "a", bus, title="t", interval=60)
    card.start()
    await asyncio.sleep(0)
    bus.publish("a", "digest", "research", {"source_id": 1, "ok": True, "relevance": 4, "cached": False})
    await card.stop("📋 Plan ready — review it below.")
    assert card.text.splitlines()[1] == "📋 Plan ready — review it below."
    assert "digested 1" in card.text                           # the counts stay
    assert card._task is None


async def test_a_failed_send_is_logged_and_retried_on_the_next_poll(caplog):
    bus, bot = RunEventBus(), _bot()
    bot.send_message = AsyncMock(side_effect=[TelegramError("Flood control"), SimpleNamespace(message_id=5)])
    card = TelegramCard(bot, 7, "a", bus, title="t")
    with caplog.at_level("WARNING", logger="argus.bot"):
        await card.refresh()
    assert "progress card update failed (Flood control)" in caplog.text
    assert card.text == "" and card.message_id is None
    await card.refresh()
    assert card.message_id == 5 and card.text


async def _shown_card(bot) -> tuple[RunEventBus, TelegramCard]:
    bus = RunEventBus()
    card = TelegramCard(bot, 7, "a", bus, title="t")
    bus.publish("a", "stage", "research", {"status": "start", "summary": ""})
    await card.refresh()                                       # the card's message, sent
    bus.publish("a", "fetch", "research", {"url": "https://example.com/x", "ok": True})
    return bus, card


async def test_an_edit_that_changes_nothing_is_a_success(caplog):
    bot = _bot()
    bus, card = await _shown_card(bot)
    bot.edit_message_text = AsyncMock(side_effect=BadRequest("Message is not modified: specified new message "
                                                             "content and reply markup are exactly the same"))
    with caplog.at_level("WARNING", logger="argus.bot"):
        await card.refresh()
    assert "fetched 1" in card.text and "failed" not in caplog.text
    await card.refresh()                                       # nothing new: no second edit
    assert bot.edit_message_text.await_count == 1


async def test_a_card_edit_honours_retry_after(monkeypatch):
    waited: list[float] = []

    async def clock(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(progress, "_sleep", clock, raising=False)
    bot = _bot()
    bus, card = await _shown_card(bot)
    bot.edit_message_text = AsyncMock(side_effect=[RetryAfter(4), None])
    await card.refresh()
    assert waited == [4] and bot.edit_message_text.await_count == 2 and "fetched 1" in card.text


async def test_a_card_whose_message_is_gone_is_not_edited_again(caplog):
    bot = _bot()
    bus, card = await _shown_card(bot)
    bot.edit_message_text = AsyncMock(side_effect=BadRequest("Message to edit not found"))
    with caplog.at_level("WARNING", logger="argus.bot"):
        await card.refresh()
        bus.publish("a", "fetch", "research", {"url": "https://example.com/y", "ok": True})
        await card.refresh()
        await card.stop("✅ Finished.")
    assert bot.edit_message_text.await_count == 1
    assert caplog.text.count("no longer exists") == 1
