"""Real plan-edit and report-revise HITL loops (v3).

plan:edit → user reply becomes plan_feedback → brief re-drafts in-graph
→ pauses at the grounded plan gate again with a fresh plan.

report:revise → user reply appends to revision_notes + sets
revision_requested → deliver passes through → revise_prep → compose
(which injects the notes) → panel → report_builder → new preview.
Bounded by MAX_REVISE_ROUNDS.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from argus.bot import app as bot_app
from argus.bot import common as bot_common
from argus.bot import runs
from argus.bot.app import on_callback, on_text
from argus.config import Settings
from argus.engines import registry
from argus.graph import graph as graph_mod
from argus.graph import nodes as nodes_mod

from helpers import fake_engines_snapshot, make_run_config, settle_runs


# ---------------------------------------------------------------------------
# brief consumes plan_feedback
# ---------------------------------------------------------------------------


def test_brief_incorporates_plan_feedback(monkeypatch):
    from argus.graph import brief as brief_mod
    from helpers import fake_llm
    seen = {}

    def reply(system, human):
        seen["prompt"] = f"{system}\n{human}"
        return ('{"sub_questions": ['
                '{"q": "How does the revised security angle work?"},'
                '{"q": "What mitigations exist for it?"},'
                '{"q": "Which deployments were affected?"}], '
                '"must_have_keywords": ["topic"], '
                '"summary": "revised plan"}')

    fake_llm(monkeypatch, reply)

    out = brief_mod.brief_node({
        "run_config": make_run_config("short"),
        "user_request": "topic",
        "brief": {"summary": "old", "sub_questions": [
            {"q": "old q", "kind": "web"}], "must_have_keywords": ["old"]},
        "plan_feedback": "focus on the security angle please",
    })
    assert "security angle" in seen["prompt"], (
        "brief must feed the user's edit feedback into the prompt")
    assert "old q" in seen["prompt"], (
        "brief must show the previous brief so the LLM revises it")
    assert out["plan"]["summary"] == "revised plan"
    assert out.get("plan_feedback") == "", "feedback must be cleared after use"


# ---------------------------------------------------------------------------
# revise loop in the graph (real deliver/revise_prep/synth/router)
# ---------------------------------------------------------------------------

_PLAN = {"summary": "s", "sub_questions": ["q"], "planned_sources": [],
         "must_have_keywords": []}


def _stub_pipeline(monkeypatch, synth_calls: list):
    monkeypatch.setattr(graph_mod, "intake_node", lambda s: {"mode": "deep"})
    monkeypatch.setattr(graph_mod, "brief_node",
                        lambda s: {"plan": dict(_PLAN),
                                   "brief": {"sub_questions": [
                                       {"q": "q?", "kind": "web"}],
                                       "must_have_keywords": ["q"],
                                       "summary": "s"}})
    monkeypatch.setattr(graph_mod, "scout_node",
                        lambda s: {"sources": [{"kind": "web", "title": "W",
                                                "url": "https://w.ex/x"}]})
    monkeypatch.setattr(graph_mod, "research_node",
                        lambda s: {"fetched": [{"url": "https://w.ex/x",
                                                "title": "W", "excerpt": "x"}],
                                   "evidence": [], "coverage": {}})
    monkeypatch.setattr(graph_mod, "outline_node",
                        lambda s: {"outline": {"sections": []}})

    def compose(s):
        synth_calls.append(list(s.get("revision_notes") or []))
        return {"draft_md": "d", "findings": [], "sections": []}

    monkeypatch.setattr(graph_mod, "compose_node", compose)
    monkeypatch.setattr(graph_mod, "analyst_node", lambda s: {"analysis": None})
    monkeypatch.setattr(graph_mod, "panel_node",
                        lambda s: {"panel_verdict": {"verdict": "pass"},
                                   "review_verdict": {"verdict": "pass"}})
    monkeypatch.setattr(graph_mod, "route_after_panel",
                        lambda s: "report_builder")
    monkeypatch.setattr(graph_mod, "report_builder_node",
                        lambda s: {"report_paths": {"md": "r.md",
                                                    "folder": "f"}})
    # deliver_node, route_after_deliver, revise_prep_node stay REAL.


def _state(thread):
    return {"thread_id": thread, "user_id": 1, "user_request": "topic",
            "messages": [], "plan": None, "sources": [], "fetched": [],
            "findings": [], "draft_md": "", "revision_notes": [],
            "revision_rounds": 0, "model_calls": [], "hitl": {"pending": False}}


def test_revise_loops_through_compose_with_notes(monkeypatch):
    synth_calls: list = []
    _stub_pipeline(monkeypatch, synth_calls)
    g = graph_mod.build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t:rev1"}}
    g.invoke(_state("t:rev1"), config=cfg)      # → plan gate
    g.invoke(Command(resume=True), config=cfg)  # → report preview
    assert g.get_state(cfg).next == ("deliver",)
    assert len(synth_calls) == 1

    # User asks to revise with feedback.
    g.update_state(cfg, {"revision_requested": True,
                         "revision_notes": ["USER: add more detail"]})
    g.invoke(Command(resume=True), config=cfg)

    snap = g.get_state(cfg)
    assert snap.next == ("deliver",), (
        f"revise must loop back to a fresh preview, got {snap.next!r}")
    assert len(synth_calls) == 2, "compose must run again for the revision"
    assert synth_calls[1] == ["USER: add more detail"], (
        "the revision compose must see the user's feedback in revision_notes")
    assert int(snap.values.get("revise_rounds") or 0) == 1


def test_revise_is_bounded(monkeypatch):
    synth_calls: list = []
    _stub_pipeline(monkeypatch, synth_calls)
    g = graph_mod.build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t:rev2"}}
    g.invoke(_state("t:rev2"), config=cfg)
    g.invoke(Command(resume=True), config=cfg)

    # Hammer revise past the cap: each round re-requests.
    for _ in range(5):
        snap = g.get_state(cfg)
        if snap.next != ("deliver",):
            break
        g.update_state(cfg, {"revision_requested": True})
        g.invoke(Command(resume=True), config=cfg)
    rounds = int(g.get_state(cfg).values.get("revise_rounds") or 0)
    assert rounds <= nodes_mod.MAX_REVISE_ROUNDS, (
        f"revise must be capped at {nodes_mod.MAX_REVISE_ROUNDS}, got {rounds}")
    # Once the cap is hit, the run must be allowed to END.
    assert not g.get_state(cfg).next


# ---------------------------------------------------------------------------
# bot wiring: pending_reply registry + on_text routing
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean():
    bot_common._inflight.clear()
    bot_common._pending_reply.clear()
    yield
    bot_common._inflight.clear()
    bot_common._pending_reply.clear()


@pytest.fixture(autouse=True)
def _acl(monkeypatch):
    monkeypatch.setattr(bot_app, "_allowed", lambda _s, _u: True)


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    dummy = Settings(
        freellmapi_base_url="http://127.0.0.1:3001/v1",
        freellmapi_api_key="freellmapi-test", telegram_bot_token="123:test",
        telegram_allowed_user_id=1, reports_root=Path("r"),
        checkpoint_db=Path("c.sqlite"), library_db=Path("l.sqlite"),
        vault_root=Path("v"), media_root=Path("v/m"),
        transcripts_root=Path("v/t"), history_root=Path("v/h"),
        ffmpeg_path=None)
    monkeypatch.setattr(bot_app, "get_settings", lambda: dummy)


@pytest.fixture(autouse=True)
def _engines_resolved(monkeypatch):
    """Drive entries first await registry.active_snapshot() (v4 WP-A)."""
    monkeypatch.setattr(registry, "active_snapshot",
                        AsyncMock(return_value=fake_engines_snapshot()))


@pytest.fixture
async def driven(monkeypatch):
    """The real graph (stubbed nodes) driven by the run driver; no library."""
    synth_calls: list = []
    _stub_pipeline(monkeypatch, synth_calls)
    graph = graph_mod.build_graph(checkpointer=MemorySaver())
    ctx = _ctx()
    ctx.application.bot_data = {"graph": graph}
    monkeypatch.setattr(registry, "_state", registry._State())
    await registry.load_config()                 # the shipped engines file; no network
    await runs.init(ctx.application)
    yield SimpleNamespace(graph=graph, ctx=ctx, synth_calls=synth_calls)
    await settle_runs()


async def _at_gate(chat_id: int, *, report: bool = False) -> str:
    out = await runs.start_run(chat_id=chat_id, user_id=chat_id, topic="topic", length="short")
    await bot_common._inflight[out.run_id].task              # -> the plan gate
    if report:
        assert (await runs.approve_plan(out.run_id)).code == "started"
        await bot_common._inflight[out.run_id].task          # -> the report preview
    return out.run_id


def _cb(chat_id, data):
    q = MagicMock()
    q.data = data
    q.answer = AsyncMock()
    q.edit_message_text = AsyncMock()
    q.message = MagicMock()
    q.message.chat.id = chat_id
    q.message.text = "msg"
    u = MagicMock()
    u.effective_user.id = chat_id
    u.effective_chat.id = chat_id
    u.callback_query = q
    return u


def _ctx():
    app = MagicMock()
    app.bot = MagicMock()
    app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    app.bot.edit_message_text = AsyncMock()
    app.bot_data = {}
    ctx = MagicMock()
    ctx.application = app
    return ctx


def _text_update(chat_id, text):
    u = MagicMock()
    u.effective_user.id = chat_id
    u.effective_chat.id = chat_id
    u.message.text = text
    u.message.reply_text = AsyncMock()
    return u


async def test_plan_edit_tap_arms_pending_reply(driven):
    run_id = await _at_gate(5)
    await on_callback(_cb(5, f"plan:edit:{run_id}"), driven.ctx)
    assert bot_common._pending_reply.get(5) == ("plan_edit", run_id)


async def test_plan_edit_reply_triggers_replan(driven):
    run_id = await _at_gate(5)
    bot_common._pending_reply[5] = ("plan_edit", run_id)
    await on_text(_text_update(5, "focus on security"), driven.ctx)
    assert 5 not in bot_common._pending_reply, "pending reply consumed"
    await bot_common._inflight[run_id].task                  # the re-plan drive: brief -> scout
    snap = await driven.graph.aget_state(bot_common._inflight[run_id].cfg)
    assert snap.values["plan_feedback"] == "focus on security", "the reply is the brief's plan_feedback"
    assert tuple(snap.next) == ("research",), "the re-plan stops at the plan gate again"


async def test_revise_tap_arms_pending_reply_not_stub(driven):
    run_id = await _at_gate(6, report=True)
    await on_callback(_cb(6, f"report:revise:{run_id}"), driven.ctx)
    assert bot_common._pending_reply.get(6) == ("revise_feedback", run_id)


async def test_revise_reply_resumes_with_notes(driven):
    run_id = await _at_gate(6, report=True)
    cfg = bot_common._inflight[run_id].cfg
    await driven.graph.aupdate_state(cfg, {"revision_notes": ["prior note"]})
    bot_common._pending_reply[6] = ("revise_feedback", run_id)

    await on_text(_text_update(6, "tighten the intro"), driven.ctx)
    assert 6 not in bot_common._pending_reply
    await bot_common._inflight[run_id].task                  # deliver -> revise_prep -> compose -> ... -> preview

    notes = driven.synth_calls[-1]
    assert any("tighten the intro" in n for n in notes)
    assert "prior note" in notes, "revision_notes is a plain channel — must carry the FULL list"
    snap = await driven.graph.aget_state(cfg)
    assert tuple(snap.next) == ("deliver",) and snap.values["revise_rounds"] == 1


async def test_plain_text_without_pending_reply_still_does_url_paste():
    ctx = _ctx()
    upd = _text_update(7, "https://youtu.be/abc123DEF45")
    await on_text(upd, ctx)
    # No pending reply → falls through to the media-link path.
    upd.message.reply_text.assert_awaited()
