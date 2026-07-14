"""/runs /append /continue mechanics (v3).

Covers, hermetically:
1. The extend fork on a FINISHED run: ``aupdate_state(as_node="deliver",
   extend_requested, append_only, FULL sources)`` → ``astream(None)``
   drives extend_prep→research→…→report_builder → pauses at the next
   report preview.
2. ``append_only`` skips the search wave (exactly the user's appended
   sources, no fresh searches).
3. The research fetch path's local_path branch (vault transcripts →
   file:/// FetchedItems; .md direct, .txt copied to a work .md,
   missing file → loud error).
4. Bot commands: /append queues run_sources; /continue forks a finished
   run with the merged FULL sources list; /continue re-attaches a
   plan-gate-paused run after a restart.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from argus import bot as bot_mod
from argus.bot import _inflight, append_cmd, continue_cmd
from argus.config import Settings
from argus.graph import graph as graph_mod
from argus.graph import nodes as nodes_mod
from argus.graph import research as research_mod
from argus.graph import search_providers as sp_mod


# ---------------------------------------------------------------------------
# graph-level: extend fork + append_only (REAL extend_prep/deliver/router)
# ---------------------------------------------------------------------------

_PLAN = {"summary": "s", "sub_questions": ["q"],
         "planned_sources": [{"kind": "search_result", "query": "q",
                              "target_url": None, "rationale": ""}],
         "must_have_keywords": []}


_BRIEF = {"sub_questions": [{"q": "q?", "kind": "web"}],
          "must_have_keywords": ["q"], "summary": "s",
          "success_criteria": []}


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the LLM/network nodes; keep deliver/extend_prep/routers REAL."""
    monkeypatch.setattr(graph_mod, "intake_node", lambda s: {"mode": "deep"})
    monkeypatch.setattr(graph_mod, "brief_node",
                        lambda s: {"plan": dict(_PLAN),
                                   "brief": dict(_BRIEF)})
    monkeypatch.setattr(graph_mod, "scout_node", lambda s: {"sources": [
        {"kind": "web", "title": "W", "url": "https://w.example/x",
         "summary": "", "source": "web-search"}]})
    # research stub: emulate fetch of every source (web url OR local
    # file:/// url) so the extend/append paths can assert on fetched.
    monkeypatch.setattr(graph_mod, "research_node", lambda s: {"fetched": [
        {"url": src.get("url", ""), "title": src.get("title", ""),
         "excerpt": "x"} for src in (s.get("sources") or [])],
        "evidence": [], "coverage": {}, "sources": s.get("sources") or [],
        "append_only": False})
    monkeypatch.setattr(graph_mod, "outline_node",
                        lambda s: {"outline": {"sections": []}})
    monkeypatch.setattr(graph_mod, "compose_node",
                        lambda s: {"draft_md": "d", "findings": [],
                                   "sections": []})
    monkeypatch.setattr(graph_mod, "panel_node",
                        lambda s: {"panel_verdict": {"verdict": "pass"},
                                   "review_verdict": {"verdict": "pass"}})
    monkeypatch.setattr(graph_mod, "route_after_panel",
                        lambda s: "report_builder")
    monkeypatch.setattr(graph_mod, "report_builder_node",
                        lambda s: {"report_paths": {"md": "r.md",
                                                    "folder": "f"}})
    # deliver_node, route_after_deliver, extend_prep_node stay REAL.


def _state(thread: str) -> dict:
    return {"thread_id": thread, "user_id": 1, "user_request": "topic",
            "messages": [], "plan": None, "sources": [], "fetched": [],
            "findings": [], "draft_md": "", "revision_notes": [],
            "revision_rounds": 0, "model_calls": [],
            "hitl": {"pending": False}}


def _drive_to_end(g, cfg) -> None:
    g.invoke(_state(cfg["configurable"]["thread_id"]), config=cfg)
    g.invoke(Command(resume=True), config=cfg)   # plan gate → preview
    g.invoke(Command(resume=True), config=cfg)   # preview → END
    snap = g.get_state(cfg)
    assert not snap.next, f"run should be finished, next={snap.next!r}"


def test_finished_run_extend_fork_append_only(monkeypatch):
    _stub_pipeline(monkeypatch)

    def _boom(*a, **kw):
        raise AssertionError(
            "append_only /continue must NOT run fresh searches")

    monkeypatch.setattr(research_mod, "followup_queries", _boom)
    monkeypatch.setattr(sp_mod, "run_query_wave", _boom)

    g = graph_mod.build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t:cont1"}}
    _drive_to_end(g, cfg)

    prev_sources = g.get_state(cfg).values.get("sources") or []
    appended = [{"kind": "local", "title": "My transcript",
                 "url": "file:///V/t.txt", "local_path": "V/t.txt",
                 "summary": "", "source": "appended-asset"}]
    merged = list(prev_sources) + appended

    # The /continue fork: position as if deliver just ran, then proceed.
    g.update_state(cfg, {"sources": merged, "extend_requested": True,
                         "extend_rounds": 0, "revision_rounds": 0,
                         "append_only": True}, as_node="deliver")
    g.invoke(None, config=cfg)

    snap = g.get_state(cfg)
    assert snap.next == ("deliver",), (
        f"extend pass must pause at the NEXT report preview, got "
        f"{snap.next!r}")
    vals = snap.values
    assert vals.get("extend_rounds") == 1
    assert vals.get("append_only") is False, (
        "research_node must consume the flag after the append pass")
    # The fetch pass saw the appended source.
    fetched_urls = [f.get("url") for f in vals.get("fetched") or []]
    assert "file:///V/t.txt" in fetched_urls


def test_finished_run_plain_continue_runs_fresh_search(monkeypatch):
    _stub_pipeline(monkeypatch)
    calls: list = []

    def fake_followups(brief, gaps, prior):
        calls.append(1)
        return ([{"query": "widened q", "provider": "ddgs",
                  "sub_qs": [0]}], [])

    def fake_wave(queries, **kw):
        return ([{"kind": "web", "title": "New",
                  "url": "https://new.example/y", "snippet": "",
                  "provider": "ddgs", "sub_qs": [0], "text": "",
                  "published": "", "source": "web-search",
                  "summary": ""}], [])

    monkeypatch.setattr(research_mod, "followup_queries", fake_followups)
    monkeypatch.setattr(sp_mod, "run_query_wave", fake_wave)

    g = graph_mod.build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t:cont2"}}
    _drive_to_end(g, cfg)

    prev_sources = g.get_state(cfg).values.get("sources") or []
    g.update_state(cfg, {"sources": list(prev_sources),
                         "extend_requested": True, "extend_rounds": 0,
                         "revision_rounds": 0, "append_only": False},
                   as_node="deliver")
    g.invoke(None, config=cfg)

    snap = g.get_state(cfg)
    assert snap.next == ("deliver",)
    assert calls, "plain /continue must run a fresh search wave"
    urls = [s.get("url") for s in snap.values.get("sources") or []]
    assert "https://new.example/y" in urls


# ---------------------------------------------------------------------------
# research fetch path: local_path branch
# ---------------------------------------------------------------------------


def _fetch(sources: list[dict]) -> tuple[list[dict], list[str]]:
    return research_mod._parallel_fetch(sources)


def test_fetch_ingests_local_md(tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# my notes", encoding="utf-8")
    fetched, errors = _fetch([{"kind": "local", "title": "Notes",
                               "local_path": str(md)}])
    assert len(fetched) == 1
    item = fetched[0]
    assert item["url"].startswith("file:///")
    assert item["markdown_path"] == str(md)
    assert item["title"] == "Notes"


def test_fetch_ingests_local_txt_via_work_copy(tmp_path):
    txt = tmp_path / "transcript.txt"
    txt.write_text("spoken words here", encoding="utf-8")
    fetched, errors = _fetch([{"kind": "local", "local_path": str(txt)}])
    assert len(fetched) == 1
    item = fetched[0]
    mp = Path(item["markdown_path"])
    assert mp.suffix == ".md" and mp.exists()
    assert "spoken words here" in mp.read_text(encoding="utf-8")
    assert str(tmp_path) not in str(mp), (
        "the .md work copy must NOT be written next to the vault file")
    assert "spoken words here" in item["excerpt"]


def test_fetch_missing_local_file_is_loud(tmp_path):
    fetched, errors = _fetch([{"kind": "local",
                               "local_path": str(tmp_path / "gone.txt")}])
    assert fetched == []
    assert any("missing" in e for e in errors)


# ---------------------------------------------------------------------------
# bot commands
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_inflight():
    _inflight.clear()
    yield
    _inflight.clear()


@pytest.fixture(autouse=True)
def _bypass_acl(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bot_mod, "_allowed", lambda _s, _uid: True)


@pytest.fixture(autouse=True)
def _dummy_settings(monkeypatch: pytest.MonkeyPatch):
    dummy = Settings(
        freellmapi_base_url="http://127.0.0.1:3001/v1",
        freellmapi_api_key="freellmapi-test",
        telegram_bot_token="123:test", telegram_allowed_user_id=1,
        reports_root=Path("reports"), checkpoint_db=Path("cp.sqlite"),
        library_db=Path("lib.sqlite"), vault_root=Path("vault"),
        media_root=Path("vault/media"),
        transcripts_root=Path("vault/transcripts"),
        history_root=Path("vault/history"), ffmpeg_path=None,
        request_timeout_seconds=60.0, max_revision_rounds=3,
    )
    monkeypatch.setattr(bot_mod, "get_settings", lambda: dummy)


class _FakeLib:
    def __init__(self, run: dict | None = None):
        self.run = run
        self.sources: list[tuple[str, str, str]] = []
        self.marked: list[tuple] = []
        self.pending: list[dict] = []
        self.statuses: list[tuple] = []

    async def resolve_run(self, chat_id, ref):
        return self.run

    async def add_run_source(self, run_id, ref, kind="url"):
        self.sources.append((run_id, ref, kind))
        self.pending.append({"ref": ref, "kind": kind, "status": "pending"})

    async def pending_sources(self, run_id):
        return list(self.pending)

    async def mark_sources(self, run_id, refs, status):
        self.marked.append((run_id, tuple(refs), status))

    async def get_assets(self, ids):
        return [{"asset_id": ids[0], "kind": "transcript", "title": "T",
                 "path": r"V:\t.txt", "bytes": 5, "meta": {}}]

    async def set_run_status(self, run_id, status, report_dir=None):
        self.statuses.append((run_id, status))
        return True

    async def list_runs(self, chat_id=None, limit=10):
        return [self.run] if self.run else []


def _update(chat_id=5):
    u = MagicMock()
    u.effective_user.id = chat_id
    u.effective_chat.id = chat_id
    u.message.reply_text = AsyncMock()
    return u


def _ctx(args, lib, graph=None):
    application = MagicMock()
    application.bot = MagicMock()
    application.bot.send_message = AsyncMock(
        return_value=SimpleNamespace(message_id=3))
    application.bot.edit_message_text = AsyncMock()
    application.bot.send_chat_action = AsyncMock()
    application.bot_data = {"library": lib}
    if graph is not None:
        application.bot_data["graph"] = graph
    ctx = MagicMock()
    ctx.application = application
    ctx.args = args
    return ctx


_RUN = {"run_id": "ab12cd34", "thread_id": "tg:5:ab12cd34", "chat_id": 5,
        "topic": "topic", "length": "short", "mode": "deep",
        "status": "done", "report_dir": None}


async def test_append_queues_urls_and_asset_refs():
    lib = _FakeLib(run=dict(_RUN))
    await append_cmd(_update(), _ctx(
        ["ab12", "https://example.com/a", "asset:7"], lib))
    kinds = {(ref, kind) for _, ref, kind in lib.sources}
    assert ("https://example.com/a", "url") in kinds
    assert ("asset:7", "asset") in kinds


class _FakeGraph:
    """Finished-run double: empty snap.next, records update/stream calls."""

    def __init__(self, values, nxt=()):
        self.values = values
        self.nxt = tuple(nxt)
        self.updates: list[tuple] = []

    async def aget_state(self, cfg):
        return SimpleNamespace(values=self.values, next=self.nxt)

    async def aupdate_state(self, cfg, payload, as_node=None):
        self.updates.append((payload, as_node))


async def test_continue_finished_run_forks_with_full_sources():
    lib = _FakeLib(run=dict(_RUN))
    lib.pending = [{"ref": "asset:7", "kind": "asset", "status": "pending"}]
    old_sources = [{"kind": "web", "title": "Old",
                    "url": "https://old.example/o"}]
    graph = _FakeGraph(values={"sources": old_sources, "length": "short"})

    async def fake_resume(ctx, info, resume_input=...):
        assert resume_input is None, (
            "a fork (no pending interrupt) must resume with None, not "
            "Command(resume=True)")
        info["awaiting"] = "report_preview"

    with patch.object(bot_mod, "_resume_after_plan",
                      side_effect=fake_resume) as rp:
        await continue_cmd(_update(), _ctx(["ab12"], lib, graph))

    assert rp.call_count == 1
    payload, as_node = graph.updates[-1]
    assert as_node == "deliver", "fork must be positioned after deliver"
    assert payload["extend_requested"] is True
    assert payload["append_only"] is True
    urls = [s["url"] for s in payload["sources"]]
    assert "https://old.example/o" in urls, (
        "sources is a last-value channel — the fork must carry the FULL "
        "merged list, never a delta")
    assert any(u.startswith("file:///") for u in urls)
    # Ingested bookkeeping happened after the successful resume.
    assert lib.marked and lib.marked[0][2] == "ingested"


async def test_continue_reattaches_plan_gate_paused_run():
    lib = _FakeLib(run=dict(_RUN))
    values = {"plan": {"summary": "s", "sub_questions": ["q"],
                       "planned_sources": []},
              "sources": [{"title": "W", "url": "https://w.example/x"}],
              "length": "long"}
    graph = _FakeGraph(values=values, nxt=("research",))
    update = _update()
    ctx = _ctx(["ab12"], lib, graph)

    await continue_cmd(update, ctx)

    assert "ab12cd34" in _inflight
    info = _inflight["ab12cd34"]
    assert info["awaiting"] == "plan_approval"
    assert info["length"] == "long"
    # Plan message re-rendered with the run-scoped keyboard.
    sends = ctx.application.bot.send_message.await_args_list
    markup = next((c.kwargs.get("reply_markup") for c in sends
                   if c.kwargs.get("reply_markup") is not None), None)
    assert markup is not None
    datas = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert all(d.endswith(":ab12cd34") for d in datas)


async def test_continue_unknown_run_says_so():
    lib = _FakeLib(run=None)
    update = _update()
    await continue_cmd(update, _ctx(["zz"], lib, _FakeGraph({})))
    text = str(update.message.reply_text.await_args_list[-1])
    assert "No unique run" in text


# ---------------------------------------------------------------------------
# Appended sources must SURVIVE ranking (live E2E caught them being cut)
# ---------------------------------------------------------------------------


def test_credibility_trusts_user_provided_local_files():
    """file:/// items are the user's own vault materials — they must not
    be scored like an unknown web domain (which lands below the floor
    and gets them dropped)."""
    from argus.graph.credibility import CREDIBILITY_FLOOR, score_fetched
    from argus.graph.state import FetchedItem

    item = FetchedItem(url="file:///V/transcripts/x.txt",
                       title="My transcript", markdown_path="w.md",
                       excerpt="whatever")
    scored = score_fetched([item], user_request="some topic")[0]
    assert (scored.credibility_score or 0) >= CREDIBILITY_FLOOR, (
        f"user-provided local evidence must score >= floor, got "
        f"{scored.credibility_score}")
    assert scored.credibility_flag in (None, "user_provided")


def test_triage_pins_appended_local_sources():
    """The live E2E (2026-07-10) showed an appended transcript being cut
    by top-N ranking — an explicitly appended source must always survive
    selection (v3: triage takes local_path sources unconditionally)."""
    from argus.graph.research import triage
    from argus.graph.state import ResearchBrief, SubQuestion
    brief = ResearchBrief(
        sub_questions=[SubQuestion(q="keyword topic?")],
        must_have_keywords=["keyword"])
    sources = [{
        "url": f"https://site{i}.example/x", "title": "keyword match topic",
        "snippet": "keyword " * 5, "sub_qs": [0],
    } for i in range(16)]
    sources.append({
        "title": "Appended transcript",
        "local_path": r"V:\transcripts\clip.txt",
        "snippet": "totally unrelated spoken words",
    })
    picked = triage(sources, set(), brief, cap=4)
    assert any(s.get("local_path") for s in picked), (
        "appended local sources must be pinned through triage")


def test_compose_pins_local_evidence_notes():
    """A user-appended note must be rendered to the writers even at low
    relevance, and must count as usable evidence."""
    from argus.graph.compose import render_notes
    notes = [{
        "source_id": i + 1, "source_url": f"https://s{i}.example/x",
        "title": "web", "sub_qs": [0], "relevance": 5, "stance": "supports",
        "claims": [{"text": "web claim", "quote": "", "confidence": "high"}],
    } for i in range(10)]
    notes.append({
        "source_id": 11, "source_url": "file:///V/transcripts/clip.txt",
        "title": "Appended transcript", "sub_qs": [], "relevance": 1,
        "stance": "background",
        "claims": [{"text": "spoken words claim", "quote": "",
                    "confidence": "medium"}],
    })
    block = render_notes(notes, [0], max_chars=2000)
    assert "[11]" in block, (
        "pinned local note must be rendered first, not crowded out")


def test_research_node_append_only_skips_search_waves(monkeypatch, tmp_path):
    """The REAL research_node must never run fresh search waves when
    append_only is set — it ingests the appended sources and clears the
    flag. (Regression: extend_prep used to clear the flag before
    research_node could see it.)"""
    md = tmp_path / "clip.md"
    md.write_text("appended transcript content " * 30, encoding="utf-8")

    def _boom(*a, **kw):
        raise AssertionError("append_only must NOT run search waves")

    monkeypatch.setattr(research_mod, "followup_queries", _boom)
    monkeypatch.setattr(research_mod, "run_query_wave", _boom)

    def fake_digest(item, source_id, brief, user_request):
        return ({"source_id": source_id, "source_url": item["url"],
                 "title": item["title"], "sub_qs": [],
                 "relevance": 3, "stance": "background",
                 "claims": [{"text": "c", "quote": "", "confidence":
                             "medium"}]}, None, "")

    monkeypatch.setattr(research_mod, "digest_one", fake_digest)
    monkeypatch.setattr(research_mod, "score_fetched",
                        lambda items, user_request: items)

    out = research_mod.research_node({
        "user_request": "topic",
        "append_only": True,
        "brief": {"sub_questions": [{"q": "q?", "kind": "web"}],
                  "must_have_keywords": ["topic"]},
        "sources": [{"kind": "local", "title": "Clip",
                     "local_path": str(md)}],
        "fetched": [], "evidence": [],
    })
    assert out["append_only"] is False, "flag must be consumed"
    assert any(f["url"].startswith("file:///") for f in out["fetched"])
    assert out["evidence"], "appended source must be digested"
