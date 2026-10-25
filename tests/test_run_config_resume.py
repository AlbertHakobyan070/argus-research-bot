"""v4 WP-D — every run carries its RunConfig in its checkpoint (spec §2.4, §6).

New runs (/research, /ask) set ``state["run_config"]``. Every resume entry
point (plan approve, plan edit, report send / extend / revise, /continue
re-attach and extend fork) first resolves the engines snapshot and calls
``ensure_in_checkpoint``, so a pre-v4 checkpoint gets the legacy defaults
visibly (Review Focus 1). Approve writes ``rc.rebuild(...)`` with the chosen
length to the checkpoint and the library, before anything else changes. A
resume that cannot resolve is refused in the chat and changes nothing (the
run driver, ``bot/runs.py``, WP-E1).

The real Argus graph (LLM and network nodes stubbed; gates, deliver,
extend_prep, revise_prep and the routers real) on a MemorySaver, a real
Library on tmp_path; the Telegram side is mocked.
"""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from argus.bot import app as bot_app
from argus.bot import common as bot_common
from argus.bot import history as bot_history
from argus.bot import research as bot_research
from argus.bot import runs
from argus.config import Settings
from argus.engines import registry
from argus.engines.registry import EngineConfigError
from argus.graph import graph as graph_mod
from argus.graph.run_config import ensure_in_checkpoint, require
from argus.graph.state import (
    ArgusState, EvidenceClaim, EvidenceNote, FetchedItem, Finding, OutlineSection,
    PanelVerdict, ResearchBrief)
from argus.library import Library
from helpers import fake_engines_snapshot, fake_llm, make_run_config, settle_runs

LEGACY = "legacy: run_config defaults applied (pre-v4 checkpoint)"
RUN_ID, CHAT = "ab12cd34", 5
THREAD = f"tg:{CHAT}:{RUN_ID}"
CFG = {"configurable": {"thread_id": THREAD}}

# What research_cmd put into a checkpoint before v4: no run_config.
_V3_STATE = {"thread_id": THREAD, "user_id": CHAT, "user_request": "topic", "length": "long",
             "messages": [], "plan": None, "sources": [], "fetched": [], "findings": [],
             "draft_md": "", "revision_notes": [], "revision_rounds": 0, "model_calls": [],
             "hitl": {"pending": False}}


@pytest.fixture(autouse=True)
async def _bot_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:test")
    dummy = dataclasses.replace(
        Settings.load(), telegram_allowed_user_id=CHAT,
        reports_root=tmp_path / "reports", checkpoint_db=tmp_path / "cp.sqlite",
        library_db=tmp_path / "lib.sqlite", vault_root=tmp_path / "vault",
        media_root=tmp_path / "vault/media", transcripts_root=tmp_path / "vault/transcripts",
        history_root=tmp_path / "vault/history")
    for mod in (bot_app, bot_research, bot_history):
        monkeypatch.setattr(mod, "get_settings", lambda: dummy)
        monkeypatch.setattr(mod, "_allowed", lambda _s, _uid: True)
    monkeypatch.setattr(registry, "_state", registry._State())
    await registry.load_config()                 # the plan card names the shipped profiles; no network
    monkeypatch.setattr(registry, "active_snapshot",
                        AsyncMock(return_value=fake_engines_snapshot()))
    bot_common._inflight.clear()
    bot_common._pending_reply.clear()
    yield
    await settle_runs()
    bot_common._inflight.clear()
    bot_common._pending_reply.clear()


@pytest.fixture
async def lib(tmp_path):
    lib = Library(tmp_path / "lib.sqlite")
    await lib.open()
    yield lib
    await lib.close()


@pytest.fixture
def graph(monkeypatch, tmp_path):
    report = {"md": str(tmp_path / "report.md"), "folder": str(tmp_path / "report")}
    monkeypatch.setattr(graph_mod, "intake_node", lambda s: {"mode": "deep"})
    monkeypatch.setattr(graph_mod, "brief_node", lambda s: {
        "plan": {"summary": "s", "sub_questions": ["q"], "planned_sources": []},
        "brief": {"sub_questions": [{"q": "q?", "kind": "web"}],
                  "must_have_keywords": ["q"], "summary": "s"},
        "plan_feedback": ""})
    monkeypatch.setattr(graph_mod, "scout_node", lambda s: {"sources": [
        {"kind": "web", "title": "W", "url": "https://w.example/x"}]})
    monkeypatch.setattr(graph_mod, "research_node", lambda s: {
        "fetched": [{"url": "https://w.example/x", "title": "W", "excerpt": "x"}],
        "evidence": [], "coverage": {}, "append_only": False})
    monkeypatch.setattr(graph_mod, "outline_node", lambda s: {"outline": {"sections": []}})
    monkeypatch.setattr(graph_mod, "compose_node",
                        lambda s: {"draft_md": "d", "findings": [], "sections": []})
    monkeypatch.setattr(graph_mod, "analyst_node", lambda s: {"analysis": None})
    monkeypatch.setattr(graph_mod, "panel_node", lambda s: {
        "panel_verdict": {"verdict": "pass"}, "review_verdict": {"verdict": "pass"}})
    monkeypatch.setattr(graph_mod, "route_after_panel", lambda s: "report_builder")
    monkeypatch.setattr(graph_mod, "report_builder_node", lambda s: {"report_paths": dict(report)})
    # extend_prep (real) only resets counters; the stubbed research runs the Extend wave (v4 WP-B)
    return graph_mod.build_graph(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Telegram doubles
# ---------------------------------------------------------------------------

def _update(text: str = "") -> MagicMock:
    u = MagicMock()
    u.effective_user.id = CHAT
    u.effective_chat.id = CHAT
    u.message.text = text
    u.message.reply_text = AsyncMock()
    return u


def _callback(data: str) -> MagicMock:
    q = MagicMock()
    q.data = data
    q.answer = AsyncMock()
    q.edit_message_text = AsyncMock()
    q.message = MagicMock()
    q.message.text = "msg"
    u = MagicMock()
    u.effective_user.id = CHAT
    u.effective_chat.id = CHAT
    u.callback_query = q
    return u


async def _ctx(graph, lib, args=()) -> MagicMock:
    """The handlers' context; the run driver drives on the same application."""
    app = MagicMock()
    app.bot = MagicMock()
    app.bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
    app.bot.edit_message_text = AsyncMock()
    app.bot.send_chat_action = AsyncMock()
    app.bot.send_document = AsyncMock()
    app.bot_data = {"graph": graph, "library": lib}
    ctx = MagicMock()
    ctx.application = app
    ctx.args = list(args)
    await runs.init(app)
    return ctx


def _sent(ctx) -> list[str]:
    return [str(c.kwargs.get("text", "")) for c in ctx.application.bot.send_message.await_args_list]


# ---------------------------------------------------------------------------
# runs at a gate
# ---------------------------------------------------------------------------

_STATUS = {"plan": "awaiting_plan", "report": "awaiting_report", "end": "done"}


async def _run_at(graph, lib, at: str, *, state: dict | None = None) -> None:
    """Drive a run to ``at`` (plan gate / report gate / END) and give it a library
    row: the v3 row (no v4 columns) for a pre-v4 state, else a v4 create_run."""
    state = dict(_V3_STATE) if state is None else state
    await graph.ainvoke(state, CFG)
    if at in ("report", "end"):
        await graph.ainvoke(Command(resume=True), CFG)
    if at == "end":
        await graph.ainvoke(Command(resume=True), CFG)
    rc = state.get("run_config")
    if rc is None:                              # the row as the v3 create_run wrote it
        db = lib._conn()
        ts = "2026-08-01T00:00:00+00:00"
        await db.execute(
            "INSERT INTO runs (run_id, thread_id, chat_id, topic, length, mode, status,"
            " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (RUN_ID, THREAD, CHAT, "topic", state.get("length") or "short", "deep",
             _STATUS[at], ts, ts))
        await db.commit()
    else:
        await lib.create_run(run_id=RUN_ID, thread_id=THREAD, chat_id=CHAT, topic="topic",
                             length=rc["length"], status=_STATUS[at], config_json=json.dumps(rc),
                             depth=rc["depth"], origin=rc["origin"])


async def _attach(graph, at: str, length: str = "long") -> runs.RunHandle:
    """The handle the run driver holds for a run resting at a gate (as a
    drive's end or /continue leaves it, minus /continue's own
    ensure_in_checkpoint — the injection is then the action's own); at the
    plan gate ``length`` is the chosen length (a Length tap)."""
    h = runs.RunHandle(run_id=RUN_ID, chat_id=CHAT, thread_id=THREAD, topic="topic")
    h.gate_kind = runs.PLAN if at == "plan" else runs.REPORT
    h.choices = runs.Choices(length=length)
    bot_common._inflight[RUN_ID] = h
    return h


def _refusal(ctx) -> str:
    """The one message an action sent: its refusal."""
    (text,) = _sent(ctx)
    assert text.startswith(f"⚠️ Could not ") and f"run `{RUN_ID}`" in text, text
    return text


def _approve(ctx):
    return bot_app.on_callback(_callback(f"plan:approve:{RUN_ID}"), ctx)


def _report(action):
    return lambda ctx: bot_app.on_callback(_callback(f"report:{action}:{RUN_ID}"), ctx)


def _reply(kind, text):
    def call(ctx):
        bot_common._pending_reply[CHAT] = (kind, RUN_ID)
        return bot_app.on_text(_update(text), ctx)
    return call


def _continue(ctx):
    ctx.args = [RUN_ID[:4]]
    return bot_history.continue_cmd(_update(), ctx)


# (name, where the run waits, attached in flight?, entry, next after the call)
RESUME_PATHS = [
    ("approve", "plan", True, _approve, ("deliver",)),
    ("plan_edit", "plan", True, _reply("plan_edit", "focus on security"), ("research",)),
    ("report_send", "report", True, _report("send"), ()),         # Send ends the run (Task 10 ruling)
    ("report_extend", "report", True, _report("extend"), ("deliver",)),
    ("report_revise", "report", True, _reply("revise_feedback", "tighten it"), ("deliver",)),
    ("continue_plan_gate", "plan", False, _continue, ("research",)),
    ("continue_report_gate", "report", False, _continue, ("deliver",)),
    ("continue_extend_fork", "end", False, _continue, ("deliver",)),
    # untracked at its gate, as after a restart: the action attaches the run first (§2.5, R-a)
    ("approve_untracked", "plan", False, _approve, ("deliver",)),
    ("plan_edit_untracked", "plan", False, _reply("plan_edit", "focus on security"), ("research",)),
    ("report_send_untracked", "report", False, _report("send"), ()),
    ("report_extend_untracked", "report", False, _report("extend"), ("deliver",)),
    ("report_revise_untracked", "report", False, _reply("revise_feedback", "tighten it"), ("deliver",)),
]


@pytest.mark.parametrize("name,at,attached,entry,nxt", RESUME_PATHS,
                         ids=[p[0] for p in RESUME_PATHS])
async def test_pre_v4_checkpoint_survives_every_resume_path(graph, lib, name, at, attached,
                                                           entry, nxt):
    await _run_at(graph, lib, at)
    assert "run_config" not in (await graph.aget_state(CFG)).values
    if attached:
        await _attach(graph, at)
    ctx = await _ctx(graph, lib)

    await entry(ctx)                                            # no exception
    await settle_runs()                                         # the drive it started, if any

    snap = await graph.aget_state(CFG)
    rc = require(snap.values)                     # valid, and its length agrees with the state's
    assert (rc.legacy, rc.origin, rc.length, rc.depth) == (True, "legacy", "long", "deep")
    assert snap.values["errors"].count(LEGACY) == 1             # recorded once, visibly
    assert tuple(snap.next) == nxt
    assert not [t for t in _sent(ctx) if t.startswith("⚠️")]


async def test_require_rejects_a_checkpoint_whose_length_disagrees_with_its_config(graph, lib):
    await _run_at(graph, lib, "plan")
    await ensure_in_checkpoint(graph, CFG, engines=fake_engines_snapshot())    # legacy, long
    await graph.aupdate_state(CFG, {"length": "short"})        # a writer that left run_config stale
    with pytest.raises(ValueError, match=r"run_config.*'long'.*'short'"):
        require((await graph.aget_state(CFG)).values)
    before = (await graph.aget_state(CFG)).config
    await _attach(graph, "plan", length="short")
    ctx = await _ctx(graph, lib)
    await _approve(ctx)                                         # resumes refuse it, in the chat
    assert "run_config" in _refusal(ctx)
    assert (await graph.aget_state(CFG)).config == before


async def test_a_pre_v4_checkpoint_resting_at_brief_keeps_next_through_the_injection(graph, lib, monkeypatch):
    """A v3 re-plan wrote its feedback as intake and stopped before brief ran:
    the checkpoint rests at ("brief",) with no run_config, and LangGraph would
    take an unattributed write for scout's, the last node that ran. The legacy
    injection is written as intake, so next stays at brief: /continue redrafts
    the plan with the edit and stops at the plan gate — research never runs
    without an Approve."""
    feedback, brief = [], graph_mod.brief_node
    monkeypatch.setattr(graph_mod, "brief_node", lambda s: feedback.append(s.get("plan_feedback")) or brief(s))
    graph = graph_mod.build_graph(checkpointer=graph.checkpointer)
    await _run_at(graph, lib, "plan")
    await graph.aupdate_state(CFG, {"plan_feedback": "focus on security"}, as_node="intake")
    await lib.set_run_status(RUN_ID, "cancelled")              # the restart reconcile's mark
    writes, update_state = [], graph.aupdate_state

    async def recording(cfg, values, *a, **kw):
        before = tuple((await graph.aget_state(cfg)).next)
        out = await update_state(cfg, values, *a, **kw)
        writes.append((sorted(values), before, tuple((await graph.aget_state(cfg)).next)))
        return out

    monkeypatch.setattr(graph, "aupdate_state", recording)
    ctx = await _ctx(graph, lib)
    await _continue(ctx)
    await settle_runs()

    assert writes == [(["errors", "run_config"], ("brief",), ("brief",))]     # the injection keeps next
    snap = await graph.aget_state(CFG)
    assert require(snap.values).legacy and snap.values["errors"].count(LEGACY) == 1
    assert feedback[1:] == ["focus on security"]               # the redraft saw the edit
    assert (tuple(snap.next), snap.values["fetched"]) == (("research",), [])  # at the plan gate, nothing fetched
    assert (await lib.get_run(RUN_ID))["status"] == "awaiting_plan"
    assert f"▶️ Resuming run `{RUN_ID}` from `brief`…" in _sent(ctx)


# ---------------------------------------------------------------------------
# Approve: rc.rebuild for the chosen length, written to checkpoint and library
# ---------------------------------------------------------------------------

async def test_approve_rebuilds_the_config_keeping_console_overrides(graph, lib):
    rc0 = make_run_config("short", **{"research.waves": 2})   # quick, with a console override
    await _run_at(graph, lib, "plan", state=dict(_V3_STATE, length="short", run_config=rc0))
    await _attach(graph, "plan", length="long")                # Length tap → depth quick → deep
    ctx = await _ctx(graph, lib)

    await _approve(ctx)
    await settle_runs()

    values = (await graph.aget_state(CFG)).values
    rc = require(values)
    assert (values["length"], rc.length, rc.depth, rc.preset_origin) == (
        "long", "long", "deep", "deep")
    assert rc.research.waves == 2                              # rebuild: from_preset gives deep's 4
    assert rc.research.fetch_per_wave == 16                    # the rest is deep's preset
    assert rc.overrides_applied == {"research.waves": 2}
    assert (rc.legacy, rc.origin) == (False, "telegram")
    assert LEGACY not in (values.get("errors") or [])
    row = await lib.get_run(RUN_ID)
    assert (row["length"], row["depth"]) == ("long", "deep")    # the stale-length bug, fixed
    assert await lib.get_run_config(RUN_ID) == values["run_config"]
    assert row["status"] == "awaiting_report"                  # and the run went on
    assert not [t for t in _sent(ctx) if t.startswith("⚠️")]


# ---------------------------------------------------------------------------
# Ordering: resolve + ensure before any mutation (Task 4 review finding)
# ---------------------------------------------------------------------------

# entry → (where the run waits, attached in flight?, callback data, a (kind, text) reply, or None for /continue)
_FAIL_ENTRIES = {"approve": ("plan", True, "plan:approve"),
                 "plan_edit": ("plan", True, ("plan_edit", "focus on security")),
                 "report_extend": ("report", True, "report:extend"),
                 "report_send": ("report", True, "report:send"),
                 "report_revise": ("report", True, ("revise_feedback", "tighten it")),
                 "continue_plan_gate": ("plan", False, None),
                 "continue_report_gate": ("report", False, None),
                 "continue_extend_fork": ("end", False, None)}


@pytest.mark.parametrize("failure", ["engines", "no_length"])
@pytest.mark.parametrize("entry", list(_FAIL_ENTRIES))
async def test_a_failed_resolution_leaves_the_run_unchanged(graph, lib, monkeypatch,
                                                            entry, failure):
    """The engines snapshot and ensure_in_checkpoint come before any change: a
    failure (the proxy down after a restart; a checkpoint ensure cannot repair)
    is refused in the chat and leaves the checkpoint, the library status, the
    run's handle and the keyboard as they were, so the run is never stranded
    as running."""
    at, attached, data = _FAIL_ENTRIES[entry]
    state = dict(_V3_STATE)
    if failure == "no_length":
        del state["length"]
    await _run_at(graph, lib, at, state=state)
    if failure == "engines":
        monkeypatch.setattr(registry, "active_snapshot", AsyncMock(side_effect=EngineConfigError(
            "role intake.classify: provider freellmapi: cannot list models")))
    if attached:
        await _attach(graph, at)
    before = await graph.aget_state(CFG)
    status = (await lib.get_run(RUN_ID))["status"]
    ctx = await _ctx(graph, lib)
    upd = _callback(f"{data}:{RUN_ID}") if isinstance(data, str) else None

    await (bot_app.on_callback(upd, ctx) if upd else _reply(*data)(ctx) if data else _continue(ctx))

    assert ("cannot list models" if failure == "engines" else "has no length") in _refusal(ctx)
    after = await graph.aget_state(CFG)
    assert (after.config, after.values) == (before.config, before.values)
    assert (await lib.get_run(RUN_ID))["status"] == status
    h = bot_common._inflight.get(RUN_ID)
    assert (h is None) if not attached else (not h.live and h.gate_kind is not None)
    ctx.application.bot.send_document.assert_not_awaited()
    ctx.application.bot.edit_message_text.assert_not_awaited()      # the keyboard stays
    if upd:
        upd.callback_query.edit_message_text.assert_not_awaited()


async def test_approve_that_cannot_rebuild_leaves_the_run_unchanged(graph, lib):
    """An override valid at quick does not fit the gap reserve of the chosen
    length's depth: rebuild raises before anything changed."""
    rc0 = make_run_config("short", **{"research.total_fetch_cap": 12})
    await _run_at(graph, lib, "plan", state=dict(_V3_STATE, length="short", run_config=rc0))
    await _attach(graph, "plan", length="long")
    before = await graph.aget_state(CFG)
    upd = _callback(f"plan:approve:{RUN_ID}")
    ctx = await _ctx(graph, lib)
    await bot_app.on_callback(upd, ctx)
    assert "total_fetch_cap" in _refusal(ctx)
    after = await graph.aget_state(CFG)
    assert after.config == before.config
    assert (await lib.get_run(RUN_ID))["status"] == "awaiting_plan"
    assert not bot_common._inflight[RUN_ID].live
    upd.callback_query.edit_message_text.assert_not_awaited()
    ctx.application.bot.edit_message_text.assert_not_awaited()


# ---------------------------------------------------------------------------
# New runs: /research and /ask start with their run_config
# ---------------------------------------------------------------------------

async def test_research_cmd_records_the_run_config_in_checkpoint_and_library(graph, lib):
    ctx = await _ctx(graph, lib, ["/length", "long", "some", "topic"])
    await bot_research.research_cmd(_update(), ctx)
    await settle_runs()                                         # the drive to the plan gate
    ((run_id, h),) = bot_common._inflight.items()
    values = (await graph.aget_state(h.cfg)).values
    rc = require(values)
    assert (rc.length, rc.depth, rc.origin, rc.legacy) == ("long", "deep", "telegram", False)
    assert rc.engines == fake_engines_snapshot()
    row = await lib.get_run(run_id)
    assert (row["length"], row["depth"], row["origin"], row["status"]) == (
        "long", "deep", "telegram", "awaiting_plan")
    assert await lib.get_run_config(run_id) == values["run_config"]


async def test_research_cmd_tells_the_chat_when_the_brief_is_exhausted(monkeypatch, lib):
    """``brief.draft`` must raise (spec §2.1): the drive reports it — the user
    is told, the run is marked ``error`` and leaves the in-flight registry."""
    monkeypatch.setattr(graph_mod, "intake_node", lambda s: {"mode": "deep"})
    fake_llm(monkeypatch, lambda system, human: "no JSON here")       # the real brief node, unparseable
    graph = graph_mod.build_graph(checkpointer=MemorySaver())
    ctx = await _ctx(graph, lib, ["some", "topic"])
    await bot_research.research_cmd(_update(), ctx)
    await settle_runs()
    failed = [t for t in _sent(ctx) if t.startswith("⚠️ Research failed: ")]
    assert failed == [t for t in failed if t.startswith("⚠️ Research failed: role brief.draft: every engine failed")]
    assert len(failed) == 1
    (row,) = await lib.list_runs(chat_id=CHAT)
    assert row["status"] == "error"
    assert bot_common._inflight == {}


async def test_ask_cmd_records_the_run_config_in_checkpoint_and_library(monkeypatch, lib):
    monkeypatch.setattr(graph_mod, "intake_node", lambda s: {"mode": "quick"})
    monkeypatch.setattr(graph_mod, "quick_answer_node", lambda s: {"quick_answer": "42"})
    quick = graph_mod.quick_answer_graph(checkpointer=MemorySaver())
    ctx = await _ctx(None, lib, ["what", "is", "x"])
    ctx.application.bot_data["quick_graph"] = quick
    await bot_research.ask_cmd(_update(), ctx)
    await settle_runs()                                         # the drive to the answer
    (row,) = await lib.list_runs(chat_id=CHAT)
    values = (await quick.aget_state({"configurable": {"thread_id": row["thread_id"]}})).values
    rc = require(values)
    assert (rc.length, rc.depth, rc.origin) == ("short", "quick", "telegram")
    assert (row["mode"], row["depth"], row["origin"], row["status"]) == (
        "quick", "quick", "telegram", "done")
    assert await lib.get_run_config(row["run_id"]) == values["run_config"]


async def test_bench_run_records_its_run_config_and_dumps_the_final_state(graph, monkeypatch, tmp_path):
    from bench import runner
    (tmp_path / "report.md").write_text("d", encoding="utf-8")      # WP-G: the runner copies the
    (tmp_path / "report").mkdir()                                    # report files report_builder wrote
    (tmp_path / "report" / "metadata.json").write_text("{}", encoding="utf-8")
    built = []

    def build(checkpointer):
        built.append(graph_mod.build_graph(checkpointer=checkpointer))
        return built[-1]

    monkeypatch.setattr(runner, "build_graph", build)
    rec = await runner._run_one({"id": "q900", "query": "topic"}, length="medium", depth="deep",
                                engines=fake_engines_snapshot(), timeout_s=60,
                                state_dir=tmp_path / "results")
    assert rec["status"] == "ok", rec
    snap = await built[0].aget_state({"configurable": {"thread_id": rec["thread_id"]}})
    assert snap.next == ()                          # past both static interrupts, to END
    assert (rec["depth"], rec["profile"]) == ("deep", "fake")
    state_path = tmp_path / "results" / "q900" / "state.json"
    assert rec["state_path"] == str(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    rc = require(state)                             # the full final state, run_config included
    assert (rc.length, rc.depth, rc.origin) == ("medium", "deep", "bench")
    assert (state["draft_md"], state["report_paths"]["md"]) == ("d", rec["report_path"])


async def test_bench_profile_switch_closes_its_throwaway_library(monkeypatch):
    """--profile switches in-process through a throwaway in-memory library; it
    is closed even when the switch fails (an open aiosqlite thread kept the
    process from exiting)."""
    from bench import runner
    opened = []

    class _Lib(runner.Library):
        async def open(self):
            opened.append(self)
            await super().open()

    monkeypatch.setattr(runner, "Library", _Lib)
    monkeypatch.setattr(registry, "_state", registry._State())
    with pytest.raises(EngineConfigError, match="unknown profile 'no-such'"):
        await runner._engines_snapshot("no-such")
    assert len(opened) == 1 and opened[0]._db is None


# ---------------------------------------------------------------------------
# The §2.4 state contract (graph/state.py is frozen after WP-D)
# ---------------------------------------------------------------------------

V4_STATE = {
    "run_config": make_run_config(),
    "search_spend": {"exa_searches": 3, "exa_text_pages": 24, "exa_contents_pages": 2,
                     "exa_usd": 0.05, "arxiv_queries": 1, "ddgs_queries": 4,
                     "github_queries": 0, "extend_allowances": 1,
                     "recency": {"window": "m"}, "dropped_undated": 2},
    "analysis": {"takeaways": [{"text": "t", "label": "high"}]},
    "tldr_md": "- a kept takeaway [1]",
    "coverage_gaps": [1, 2],
    "gap_history": [{"round": 1, "wave": 1, "sub_qs": [1], "new_strong": 0,
                     "source": "coverage"}],
    "gap_waves": 1,
    "compose_runs": 2,
    "cancel_note": "user cancel",
    "report_paths": {"md": "r.md", "pdf": None, "folder": "f", "evidence": "f/evidence.json"},
}


async def test_every_v4_state_key_survives_a_checkpoint():
    g = StateGraph(ArgusState)
    g.add_node("write", lambda s: dict(V4_STATE))
    g.add_edge(START, "write")
    g.add_edge("write", END)
    graph = g.compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "v4-keys"}}
    await graph.ainvoke({"user_request": "t"}, cfg)
    values = (await graph.aget_state(cfg)).values
    assert {k: values.get(k) for k in V4_STATE} == V4_STATE
    await graph.aupdate_state(cfg, {"gap_waves": 2, "coverage_gaps": []})   # last-value channels
    values = (await graph.aget_state(cfg)).values
    assert (values["gap_waves"], values["coverage_gaps"]) == (2, [])


V4_MODEL_DEFAULTS = [
    (FetchedItem(url="u"), {"canonical_url": "", "domain": "", "published": "", "fetcher": "",
                            "fetch_ms": 0, "chars": 0, "authority_prior": 1, "pinned": False}),
    (EvidenceClaim(text="c"), {"quote_verified": None}),
    (EvidenceNote(source_id=1, source_url="u"), {"source_type": "other", "authority": 1,
                                                 "cached": False}),
    (ResearchBrief(), {"archetype": "explainer", "time_sensitive": False, "recency_hint": None,
                       "is_academic_topic": False, "entities": []}),
    (OutlineSection(title="t"), {"note_ids": []}),
    (Finding(claim="c", citation_urls=["u"]), {"source_ids": [], "sentence": "",
                                               "status": "kept"}),
    (PanelVerdict(), {"next": "report_builder", "coverage_gaps": [], "wants_rewrite": False,
                      "judge_not_diverse": False, "uncited_specifics": [],
                      "skipped_judges": [], "judged": 0, "total": 0}),
]


@pytest.mark.parametrize("model,defaults", V4_MODEL_DEFAULTS,
                         ids=[type(m).__name__ for m, _ in V4_MODEL_DEFAULTS])
def test_v4_model_fields_default_so_pre_v4_dumps_still_load(model, defaults):
    dumped = model.model_dump()
    assert {k: dumped[k] for k in defaults} == defaults
    v3_dump = {k: v for k, v in dumped.items() if k not in defaults}
    assert type(model).model_validate(v3_dump) == model       # a v3 checkpoint's dict loads


@pytest.mark.parametrize("model,field,bad", [
    (Finding, "status", "dropped"), (PanelVerdict, "next", "deliver")])
def test_v4_literal_fields_reject_unknown_values(model, field, bad):
    base = {"claim": "c", "citation_urls": ["u"]} if model is Finding else {}
    with pytest.raises(ValueError, match=field):
        model.model_validate({**base, field: bad})
