"""Phase 6a — real plan-edit and report-revise HITL loops.

plan:edit → user reply becomes plan_feedback → planner re-plans in-graph
→ pauses at the grounded plan gate again with a fresh plan.

report:revise → user reply appends to revision_notes + sets
revision_requested → deliver passes through → revise_prep → synthesizer
(which already injects the notes) → reviewer → report_builder → new
preview. Bounded by MAX_REVISE_ROUNDS.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from argus import bot as bot_mod
from argus.bot import _inflight, _pending_reply, on_callback, on_text
from argus.config import Settings
from argus.graph import graph as graph_mod
from argus.graph import nodes as nodes_mod


# ---------------------------------------------------------------------------
# planner consumes plan_feedback
# ---------------------------------------------------------------------------


def test_planner_incorporates_plan_feedback(monkeypatch):
    seen = {}

    class _FakeChat:
        def invoke(self, msgs):
            seen["prompt"] = "\n".join(getattr(m, "content", "") for m in msgs)
            return SimpleNamespace(
                content='{"sub_questions": ["revised"], '
                        '"planned_sources": [], "must_have_keywords": [], '
                        '"summary": "revised plan"}',
                response_metadata={}, usage_metadata={})

    monkeypatch.setattr(nodes_mod.llm, "chat_for_tier",
                        lambda *a, **k: _FakeChat())
    monkeypatch.setattr(nodes_mod.llm, "resolve_tier", lambda t: "m")

    out = nodes_mod.planner_node({
        "user_request": "topic",
        "plan": {"summary": "old", "sub_questions": ["old q"],
                 "planned_sources": []},
        "plan_feedback": "focus on the security angle please",
    })
    assert "security angle" in seen["prompt"], (
        "planner must feed the user's edit feedback into the prompt")
    assert "old q" in seen["prompt"], (
        "planner must show the previous plan so the LLM revises it")
    assert out["plan"]["summary"] == "revised plan"
    assert out.get("plan_feedback") == "", "feedback must be cleared after use"


# ---------------------------------------------------------------------------
# revise loop in the graph (real deliver/revise_prep/synth/router)
# ---------------------------------------------------------------------------

_PLAN = {"summary": "s", "sub_questions": ["q"], "planned_sources": [],
         "must_have_keywords": []}


def _stub_pipeline(monkeypatch, synth_calls: list):
    monkeypatch.setattr(graph_mod, "intake_node", lambda s: {"mode": "deep"})
    monkeypatch.setattr(graph_mod, "planner_node",
                        lambda s: {"plan": dict(_PLAN)})
    monkeypatch.setattr(graph_mod, "planner_reflect_node", lambda s: {})
    monkeypatch.setattr(graph_mod, "researcher_node",
                        lambda s: {"sources": [{"kind": "web", "title": "W",
                                                "url": "https://w.ex/x"}]})
    monkeypatch.setattr(graph_mod, "fetcher_node",
                        lambda s: {"fetched": [{"url": "https://w.ex/x",
                                                "title": "W", "excerpt": "x"}]})
    monkeypatch.setattr(graph_mod, "normalizer_node", lambda s: {})
    monkeypatch.setattr(graph_mod, "credibility_node", lambda s: {})
    monkeypatch.setattr(graph_mod, "filter_node", lambda s: {})

    def synth(s):
        synth_calls.append(list(s.get("revision_notes") or []))
        return {"draft_md": "d", "findings": []}

    monkeypatch.setattr(graph_mod, "synthesizer_node", synth)
    monkeypatch.setattr(graph_mod, "reviewer_node",
                        lambda s: {"review_verdict": {"verdict": "pass"}})
    monkeypatch.setattr(graph_mod, "route_after_review",
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


def test_revise_loops_through_synthesizer_with_notes(monkeypatch):
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
    assert len(synth_calls) == 2, "synthesizer must run again for the revision"
    assert synth_calls[1] == ["USER: add more detail"], (
        "the revision synthesis must see the user's feedback in revision_notes")
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
    _inflight.clear()
    _pending_reply.clear()
    yield
    _inflight.clear()
    _pending_reply.clear()


@pytest.fixture(autouse=True)
def _acl(monkeypatch):
    monkeypatch.setattr(bot_mod, "_allowed", lambda _s, _u: True)


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    dummy = Settings(
        freellmapi_base_url="http://127.0.0.1:3001/v1",
        freellmapi_api_key="freellmapi-test", telegram_bot_token="123:test",
        telegram_allowed_user_id=1, reports_root=Path("r"),
        checkpoint_db=Path("c.sqlite"), library_db=Path("l.sqlite"),
        vault_root=Path("v"), media_root=Path("v/m"),
        transcripts_root=Path("v/t"), history_root=Path("v/h"),
        ffmpeg_path=None, request_timeout_seconds=60.0, max_revision_rounds=3)
    monkeypatch.setattr(bot_mod, "get_settings", lambda: dummy)


def _seed(run_id="ab12cd34", chat_id=5, awaiting="plan_approval"):
    info = {"run_id": run_id, "chat_id": chat_id,
            "thread_id": f"tg:{chat_id}:{run_id}",
            "state": {"length": "short"}, "stage": "x", "length": "short",
            "awaiting": awaiting, "plan_text": "Plan text",
            "cfg": {"configurable": {"thread_id": f"tg:{chat_id}:{run_id}"}},
            "graph": MagicMock()}
    _inflight[run_id] = info
    return info


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


async def test_plan_edit_tap_arms_pending_reply():
    _seed(chat_id=5)
    ctx = _ctx()
    await on_callback(_cb(5, "plan:edit:ab12cd34"), ctx)
    assert _pending_reply.get(5) == ("plan_edit", "ab12cd34")


async def test_plan_edit_reply_triggers_replan(monkeypatch):
    info = _seed(chat_id=5)
    _pending_reply[5] = ("plan_edit", "ab12cd34")
    replan = AsyncMock()
    monkeypatch.setattr(bot_mod, "_replan_after_edit", replan)
    await on_text(_text_update(5, "focus on security"), _ctx())
    replan.assert_awaited_once()
    assert replan.await_args.args[2] == "focus on security"
    assert 5 not in _pending_reply, "pending reply consumed"


async def test_revise_tap_arms_pending_reply_not_stub():
    _seed(chat_id=6, awaiting="report_preview")
    ctx = _ctx()
    await on_callback(_cb(6, "report:revise:ab12cd34"), ctx)
    assert _pending_reply.get(6) == ("revise_feedback", "ab12cd34")


async def test_revise_reply_resumes_with_notes(monkeypatch):
    info = _seed(chat_id=6, awaiting="report_preview")
    info["cfg"] = {"configurable": {"thread_id": "tg:6:ab12cd34"}}
    graph = MagicMock()
    graph.aget_state = AsyncMock(return_value=SimpleNamespace(
        values={"revision_notes": ["prior note"]}))
    graph.aupdate_state = AsyncMock()
    info["graph"] = graph
    _pending_reply[6] = ("revise_feedback", "ab12cd34")
    resume = AsyncMock()
    monkeypatch.setattr(bot_mod, "_resume_after_plan", resume)

    await on_text(_text_update(6, "tighten the intro"), _ctx())

    graph.aupdate_state.assert_awaited_once()
    payload = graph.aupdate_state.await_args.args[1]
    assert payload["revision_requested"] is True
    assert any("tighten the intro" in n for n in payload["revision_notes"])
    assert "prior note" in payload["revision_notes"], (
        "revision_notes is a plain channel — must carry the FULL list")
    resume.assert_awaited_once()
    assert 6 not in _pending_reply


async def test_plain_text_without_pending_reply_still_does_url_paste():
    ctx = _ctx()
    upd = _text_update(7, "https://youtu.be/abc123DEF45")
    await on_text(upd, ctx)
    # No pending reply → falls through to the media-link path.
    upd.message.reply_text.assert_awaited()
