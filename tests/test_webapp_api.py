"""The Mini App API (v4 spec §2.7) on a fake run driver: the routes and JSON
shapes, the driver's outcome codes as HTTP (409 busy / already_running /
refused, 404 not_found), report-path containment, Send to chat, and the
engines routes on the real registry (provider definitions are not editable)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from argus.bot import common, runs
from argus.bot.runs import Outcome
from argus.engines import registry
from argus.graph.run_config import PRESETS
from argus.webapp import api, server

from test_runs_driver import CHAT, _at_plan_gate, _snap, env  # noqa: F401  (env: the real driver on a stubbed graph)
from webapp_fakes import TOKEN, USER_ID, FakeHandle, FakeLibrary, FakeRuns, auth, make_app, seed_run

ENGINES: dict = {
    "active_profile": "main",
    "providers": {
        "fixed": {"name": "fixed", "base_url": "http://fixed/v1", "api_key_env": "T_WEB_FIXED_KEY",
                  "models_endpoint": False, "known_models": ["k1", "k2"]},
        "local": {"name": "local", "base_url": "http://127.0.0.1:9/v1", "models_endpoint": False,
                  "known_models": ["l1"]},
    },
    "profiles": {
        "main": {"name": "main", "tiers": {t: {"provider": "fixed", "model": "k1"} for t in ("cheap", "strong", "judge")}},
        "other": {"name": "other", "tiers": {t: {"provider": "local", "model": "l1"} for t in ("cheap", "strong", "judge")}},
    },
    "pricing": {"exa": {"price_per_search_usd": 0.005, "price_per_page_usd": 0.001, "prices_as_of": "2026-10-01"}},
}


@pytest.fixture
def fake():
    return FakeRuns()


@pytest.fixture
def lib():
    return FakeLibrary()


@pytest.fixture
def app(fake, lib, tmp_path):
    return make_app(fake, library=lib, reports_root=tmp_path / "reports")


@pytest.fixture
async def client(app):
    async with TestClient(TestServer(app)) as c:
        yield c


async def _json(r, status=200):
    assert r.status == status, await r.text()
    return await r.json()


# --- /api/me ----------------------------------------------------------------

async def test_me(client, app, monkeypatch):
    app[api.DEPS].bot_data.update(webapp_public_url="https://a-b.trycloudflare.com", webapp_ok=True)
    monkeypatch.setattr(registry, "_state", registry._State())
    body = await _json(await client.get("/api/me", headers=auth()))
    assert body["user"] == {"id": USER_ID, "first_name": "Albert"}
    bot = body["bot"]
    assert (bot["allowed_user_id"], bot["public_url"], bot["console_ok"]) == (
        USER_ID, "https://a-b.trycloudflare.com", True)
    assert bot["active_profile"] is None and bot["version"]          # no engines config loaded here


# --- runs list and run view -------------------------------------------------

async def test_runs_list_uses_library_and_registry(client, app, fake):
    seed_run(fake, app, "plan0001", status="awaiting_plan", next_=("research",), created_at="2026-10-21T10:03")
    seed_run(fake, app, "live0001", status="running", live=True, stage="research", created_at="2026-10-21T10:02",
             archetype="forecast", exa_usd=0.12)
    seed_run(fake, app, "done0001", status="done", report_dir="/r/x", created_at="2026-10-21T10:01")
    seed_run(fake, app, "rept0001", status="awaiting_report", created_at="2026-10-21T10:00", tracked=False)
    body = await _json(await client.get("/api/runs?limit=20", headers=auth()))
    rows = {r["run_id"]: r for r in body["runs"]}
    assert list(rows) == ["plan0001", "live0001", "done0001", "rept0001"]
    assert set(rows["live0001"]) == {"run_id", "thread_id", "topic", "length", "depth", "archetype", "status",
                                     "origin", "created_at", "updated_at", "report_dir", "stage", "awaiting",
                                     "in_flight", "exa_usd"}
    assert (rows["live0001"]["in_flight"], rows["live0001"]["awaiting"], rows["live0001"]["stage"]) == (
        True, None, "research")
    assert (rows["live0001"]["archetype"], rows["live0001"]["exa_usd"]) == ("forecast", 0.12)
    assert (rows["plan0001"]["in_flight"], rows["plan0001"]["awaiting"]) == (False, "plan")
    assert rows["rept0001"]["awaiting"] == "report"                  # untracked after a restart: the library status
    assert (rows["done0001"]["awaiting"], rows["done0001"]["report_dir"]) == (None, "/r/x")
    assert len((await _json(await client.get("/api/runs?limit=2", headers=auth())))["runs"]) == 2


@pytest.mark.parametrize("limit", ["0", "x", "501"])
async def test_runs_list_bad_limit(client, limit):
    body = await _json(await client.get(f"/api/runs?limit={limit}", headers=auth()), 400)
    assert body["error"]["code"] == "bad_request"


async def test_run_view(client, app, fake):
    values = {"sources": [{}, {}, {}], "fetched": [{}, {}], "evidence": [{}], "findings": [],
              "coverage": {"0": 2}, "run_config": {"length": "short"}, "panel_verdict": {"judged": 3, "total": 3},
              "report_paths": {"md": "/x/report.md"}, "compose_runs": 1, "gap_waves": 2, "model_calls": [],
              "errors": []}
    seed_run(fake, app, "plan0001", status="awaiting_plan", next_=("research",), values=values)
    fake.plans["plan0001"] = {"run_id": "plan0001", "length": "short", "exa_estimate_usd": 0.03}
    fake.bus.publish("plan0001", "stage", "scout", {"status": "end"})
    fake.snapshots["plan0001"]["last_seq"] = fake.bus.last_seq("plan0001")
    body = await _json(await client.get("/api/runs/plan0001", headers=auth()))
    assert (body["run"]["run_id"], body["run"]["awaiting"], body["run"]["in_flight"]) == ("plan0001", "plan", False)
    snap = body["snapshot"]
    assert set(snap) == {"next", "counts", "coverage", "run_config", "engines_summary", "panel_verdict",
                         "report_paths", "plan", "compose_runs", "gap_waves"}
    assert snap["next"] == ["research"]
    assert snap["counts"] == {"sources": 3, "fetched": 2, "evidence": 1, "findings": 0}
    assert (snap["coverage"], snap["compose_runs"], snap["gap_waves"]) == ({"0": 2}, 1, 2)
    assert snap["plan"] == fake.plans["plan0001"]                    # the Telegram card's data
    assert snap["panel_verdict"] == {"judged": 3, "total": 3}
    assert "by_role" in snap["engines_summary"]
    assert body["last_seq"] == 1


async def test_unknown_run_is_404(client):
    for path in ("/api/runs/nope0001", "/api/runs/nope0001/events", "/api/runs/nope0001/report",
                 "/api/runs/nope0001/evidence"):
        body = await _json(await client.get(path, headers=auth()), 404)
        assert body["error"]["code"] == "not_found", path


async def test_events_since(client, app, fake):
    seed_run(fake, app, "live0001", live=True, stage="research")
    for i in range(3):
        fake.bus.publish("live0001", "fetch", "research", {"url": f"https://x/{i}", "ok": True})
    body = await _json(await client.get("/api/runs/live0001/events?after=1", headers=auth()))
    assert [e["seq"] for e in body["events"]] == [2, 3] and body["last_seq"] == 3
    assert body["events"][0]["kind"] == "fetch" and body["events"][0]["data"]["url"] == "https://x/1"
    assert (await _json(await client.get("/api/runs/live0001/events?after=-1", headers=auth()), 400))


# --- actions: the driver's outcome codes as HTTP ----------------------------

async def test_start_run(client, fake):
    body = await _json(await client.post("/api/runs", headers=auth(), json={
        "topic": "  gold outlook ", "length": "medium", "depth": "deep", "recency": "m",
        "engines_profile": "other", "overrides": {"research.waves": 2}}))
    assert body["run_id"] == "new00001" and body["outcome"] == "started"
    assert fake.calls == [("start_run", {"chat_id": USER_ID, "user_id": USER_ID, "topic": "gold outlook",
                                         "length": "medium", "depth": "deep", "engines_profile": "other",
                                         "overrides": {"research.waves": 2, "recency": "m"}, "origin": "webapp"})]


async def test_start_run_with_only_a_topic(client, fake):
    await _json(await client.post("/api/runs", headers=auth(), json={"topic": "x"}))
    assert fake.calls == [("start_run", {"chat_id": USER_ID, "user_id": USER_ID, "topic": "x",
                                         "overrides": {}, "origin": "webapp"})]


@pytest.mark.parametrize("payload", [{}, {"topic": "  "}, {"topic": 3}, {"topic": "x", "length": 5},
                                     {"topic": "x", "overrides": ["a"]}, [1]])
async def test_start_run_bad_body(client, fake, payload):
    body = await _json(await client.post("/api/runs", headers=auth(), json=payload), 400)
    assert body["error"]["code"] == "bad_request" and fake.calls == []


async def test_bad_json_is_400(client):
    r = await client.post("/api/runs", headers={**auth(), "Content-Type": "application/json"}, data="{nope")
    assert (await _json(r, 400))["error"]["code"] == "bad_request"


@pytest.mark.parametrize("code,status", [("busy", 409), ("already_running", 409), ("refused", 409),
                                         ("not_found", 404)])
async def test_outcome_codes_map_to_http(client, fake, code, status):
    text = f"text for {code}"
    for name in ("start_run", "approve_plan", "report_action", "cancel_run", "edit_plan"):
        fake.outcomes[name] = Outcome(code, text, "run00001")
    for path, payload in (("/api/runs", {"topic": "x"}), ("/api/runs/run00001/plan", {"action": "approve"}),
                          ("/api/runs/run00001/plan", {"action": "edit", "feedback": "narrower"}),
                          ("/api/runs/run00001/report", {"action": "extend"}), ("/api/runs/run00001/cancel", None)):
        body = await _json(await client.post(path, headers=auth(), json=payload), status)
        assert body["error"] == {"code": code, "message": text}, path


async def test_plan_actions_reach_the_driver_as_webapp(client, fake):
    fake.outcomes["cancel_run"] = Outcome("cancelled", "❌ Cancelled run `run00001`.", "run00001")
    a = await _json(await client.post("/api/runs/run00001/plan", headers=auth(), json={
        "action": "approve", "length": "long", "depth": "deep", "engines_profile": "other"}))
    e = await _json(await client.post("/api/runs/run00001/plan", headers=auth(),
                                      json={"action": "edit", "feedback": "only EU"}))
    c = await _json(await client.post("/api/runs/run00001/plan", headers=auth(), json={"action": "cancel"}))
    assert (a["outcome"], a["run_id"], e["outcome"]) == ("started", "run00001", "started")
    assert c == {"outcome": "cancelled", "message": "❌ Cancelled run `run00001`.", "run_id": "run00001"}
    assert fake.calls == [
        ("approve_plan", {"run_id": "run00001", "actor": "webapp", "length": "long", "depth": "deep",
                          "engines_profile": "other"}),
        ("edit_plan", {"run_id": "run00001", "feedback": "only EU", "actor": "webapp"}),
        ("cancel_run", {"run_id": "run00001", "actor": "webapp"})]


async def test_report_actions_reach_the_driver_as_webapp(client, fake):
    fake.outcomes["report_action"] = Outcome("delivered", "", "run00001")
    s = await _json(await client.post("/api/runs/run00001/report", headers=auth(), json={"action": "send"}))
    await _json(await client.post("/api/runs/run00001/report", headers=auth(),
                                  json={"action": "revise", "feedback": "tighter"}))
    d = await _json(await client.post("/api/runs/run00001/cancel", headers=auth()))
    assert s["outcome"] == "delivered" and d["outcome"] == "cancelled"
    assert fake.calls == [
        ("report_action", {"run_id": "run00001", "action": "send", "feedback": None, "actor": "webapp"}),
        ("report_action", {"run_id": "run00001", "action": "revise", "feedback": "tighter", "actor": "webapp"}),
        ("cancel_run", {"run_id": "run00001", "actor": "webapp"})]


@pytest.mark.parametrize("path,payload", [("/api/runs/run00001/plan", {"action": "fly"}),
                                          ("/api/runs/run00001/plan", {"action": "edit"}),
                                          ("/api/runs/run00001/report", {"action": "fly"}),
                                          ("/api/runs/run00001/report", {"action": "revise", "feedback": 3})])
async def test_bad_actions_are_400(client, fake, path, payload):
    assert (await _json(await client.post(path, headers=auth(), json=payload), 400))["error"]["code"] == "bad_request"
    assert fake.calls == []


async def test_unknown_outcome_code_is_an_error_not_a_success(client, fake):
    fake.outcomes["cancel_run"] = Outcome("mystery", "?", "run00001")
    assert (await _json(await client.post("/api/runs/run00001/cancel", headers=auth()), 500))["error"]["code"] == "internal"


# --- report files: containment, render, evidence, Send to chat ---------------

def _report(app, fake, run_id="done0001", *, folder=None, **files) -> Path:
    root = app[api.DEPS].reports_root
    folder = folder or root / run_id
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"folder": str(folder)}
    for key, (name, content) in files.items():
        (folder / name).write_bytes(content.encode() if isinstance(content, str) else content)
        paths[key] = str(folder / name)
    seed_run(fake, app, run_id, status="done", values={"report_paths": paths})
    return folder


async def test_report_html_is_rendered_and_sanitised(client, app, fake):
    _report(app, fake, md=("report.md", "# R\n\nClaim [1].\n\n<script>alert(1)</script>"))
    r = await client.get("/api/runs/done0001/report?format=html", headers=auth())
    assert r.status == 200 and r.content_type == "text/html"
    html = await r.text()
    assert "<h1>R</h1>" in html and 'data-n="1"' in html and "<script" not in html
    assert (await _json(await client.get("/api/runs/done0001/report?format=pdf", headers=auth()), 400))


async def test_evidence_is_served(client, app, fake):
    _report(app, fake, evidence=("evidence.json", json.dumps({"schema": 1, "findings": []})))
    body = await _json(await client.get("/api/runs/done0001/evidence", headers=auth()))
    assert body == {"schema": 1, "findings": []}


async def test_report_paths_outside_reports_root_are_refused(client, app, fake, tmp_path, caplog):
    outside = tmp_path / "elsewhere"
    _report(app, fake, folder=outside, md=("report.md", "secret"), evidence=("evidence.json", "{}"))
    for path in ("/api/runs/done0001/report", "/api/runs/done0001/evidence"):
        body = await _json(await client.get(path, headers=auth()), 404)
        assert "outside reports_root" in body["error"]["message"]
    body = await _json(await client.post("/api/runs/done0001/send", headers=auth(), json={"what": ["md"]}), 404)
    assert app[api.DEPS].bot.documents == []
    assert "outside reports_root" in caplog.text


async def test_traversal_in_a_registered_path_is_refused(client, app, fake):
    root = app[api.DEPS].reports_root
    (root / "x").mkdir(parents=True)
    (root.parent / "loot.md").write_text("loot")
    seed_run(fake, app, "done0001", status="done", values={"report_paths": {"md": str(root / "x" / ".." / ".." / "loot.md")}})
    assert (await _json(await client.get("/api/runs/done0001/report", headers=auth()), 404))


async def test_unregistered_or_missing_files_are_404(client, app, fake):
    folder = _report(app, fake, md=("report.md", "# R"))
    assert (await _json(await client.get("/api/runs/done0001/evidence", headers=auth()), 404))   # not registered
    (folder / "report.md").unlink()
    assert (await _json(await client.get("/api/runs/done0001/report", headers=auth()), 404))     # gone from disk


async def test_send_to_chat_sends_the_chosen_files(client, app, fake):
    _report(app, fake, md=("report.md", "# R"), pdf=("report.pdf", b"%PDF-1.4"), evidence=("evidence.json", "{}"))
    body = await _json(await client.post("/api/runs/done0001/send", headers=auth(),
                                         json={"what": ["md", "evidence", "md"]}))
    assert body["outcome"] == "sent" and body["sent"] == ["report.md", "evidence.json"]
    docs = app[api.DEPS].bot.documents
    assert [(d["chat_id"], d["filename"], d["data"]) for d in docs] == [
        (USER_ID, "report.md", b"# R"), (USER_ID, "evidence.json", b"{}")]
    assert fake.calls == []                                          # files only: no driver action, nothing moves


async def test_send_checks_every_file_before_sending_any(client, app, fake):
    _report(app, fake, md=("report.md", "# R"))
    body = await _json(await client.post("/api/runs/done0001/send", headers=auth(), json={"what": ["md", "pdf"]}), 404)
    assert "pdf" in body["error"]["message"] and app[api.DEPS].bot.documents == []


@pytest.mark.parametrize("payload", [{}, {"what": []}, {"what": "md"}, {"what": ["md", "metadata"]}])
async def test_send_bad_body(client, app, fake, payload):
    _report(app, fake, md=("report.md", "# R"))
    assert (await _json(await client.post("/api/runs/done0001/send", headers=auth(), json=payload), 400))


# --- engines and presets on the real registry --------------------------------

@pytest.fixture
async def engines(monkeypatch, tmp_path, lib):
    monkeypatch.setattr(registry, "_state", registry._State())
    monkeypatch.setenv("T_WEB_FIXED_KEY", "sk-test-not-shown")
    path = tmp_path / "engines.json"
    path.write_text(json.dumps(ENGINES), encoding="utf-8")
    await registry.load_config([path], library=lib)
    return path


async def test_get_engines(client, engines, lib):
    lib.exa_usd = 1.25
    body = await _json(await client.get("/api/engines", headers=auth()))
    assert set(body) == {"config", "active_profile", "snapshot", "validation", "exa_month_to_date_usd"}
    assert body["active_profile"] == "main" and body["exa_month_to_date_usd"] == 1.25
    providers = body["config"]["providers"]
    assert providers["fixed"]["api_key"] == {"env": "T_WEB_FIXED_KEY", "present": True}
    assert providers["local"]["api_key"] == {"env": None, "present": None}
    assert "api_key_env" not in providers["fixed"]
    assert "sk-test-not-shown" not in json.dumps(body)
    assert body["validation"] == {"ok": True, "errors": []}
    assert body["snapshot"]["profile"] == "main" and "warnings" in body["snapshot"]


async def test_get_engines_shows_a_failing_active_profile(client, engines, monkeypatch):
    monkeypatch.delenv("T_WEB_FIXED_KEY")
    body = await _json(await client.get("/api/engines", headers=auth()))
    assert body["snapshot"] is None and body["validation"]["ok"] is False
    assert "T_WEB_FIXED_KEY" in body["validation"]["errors"][0]
    assert body["config"]["providers"]["fixed"]["api_key"]["present"] is False


@pytest.mark.parametrize("payload", [{"providers": {"fixed": {"base_url": "http://evil/v1"}}},
                                     {"active_profile": "main", "pricing": {}},
                                     {"overrides": {"other": {"providers": {"x": {}}}}},
                                     {"overrides": {"other": {"tiers": {}, "base_url": "http://evil/v1"}}}])
async def test_put_engines_rejects_provider_definition_changes(client, engines, lib, payload):
    body = await _json(await client.put("/api/engines", headers=auth(), json=payload), 400)
    assert body["error"]["code"] == "bad_request" and "not editable" in body["error"]["message"]
    assert lib.settings == {} and registry.current_config().providers["fixed"].base_url == "http://fixed/v1"


async def test_put_engines_persists_per_profile_overrides(client, engines, lib):
    ov = {"tiers": {"judge": {"provider": "fixed", "model": "k2"}}}
    body = await _json(await client.put("/api/engines", headers=auth(), json={"overrides": {"other": ov}}))
    assert body["config"]["overrides"]["other"]["tiers"]["judge"]["model"] == "k2"
    assert json.loads(lib.settings["engines.overrides"])["other"]["tiers"]["judge"]["model"] == "k2"
    assert registry.library_overrides("other").tiers["judge"].model == "k2"
    # survives a reload from the library layer
    await registry.load_config([engines], library=lib)
    assert registry.current_config().overrides["other"].tiers["judge"].model == "k2"


async def test_put_engines_switches_profile_and_transcriber(client, engines, lib):
    # fix round 1 (item 4): one change per request, so the two switches are two PUTs
    body = await _json(await client.put("/api/engines", headers=auth(), json={"active_profile": "other"}))
    assert body["active_profile"] == "other" and body["validation"]["ok"] is True
    await _json(await client.put("/api/engines", headers=auth(), json={"transcriber": {"asr_engine": "openai_audio"}}))
    assert json.loads(lib.settings["engines.active_profile"]) == "other"
    assert json.loads(lib.settings["engines.transcriber"])["asr_engine"] == "openai_audio"


@pytest.mark.parametrize("payload", [
    {"active_profile": "other", "transcriber": {"asr_engine": "openai_audio"}},
    {"active_profile": "other", "overrides": {"other": {}}},
    {"overrides": {"other": {}}, "transcriber": {}},
    {"overrides": {"main": {}, "other": {}}},
    {},
    {"overrides": {}},
])
async def test_put_engines_is_one_change_per_request(client, engines, lib, payload):
    body = await _json(await client.put("/api/engines", headers=auth(), json=payload), 400)
    assert body["error"]["code"] == "bad_request"
    assert lib.settings == {} and registry.current_config().active_profile == "main"


# --- fix round 1: the fakes are bound to the real driver's signatures ----------

async def test_fake_runs_binds_each_call_against_the_real_driver():
    fake = FakeRuns()
    with pytest.raises(TypeError):
        await fake.approve_plan("run00001", actr="webapp")          # a kwarg the driver does not take
    with pytest.raises(TypeError):
        await fake.start_run(chat_id=1, user_id=1, topic="t", recency="m")
    with pytest.raises(TypeError):
        await fake.cancel_run("run00001", "webapp")                  # actor is keyword-only
    assert (await fake.edit_plan("run00001", "x", actor="webapp")).code == "started"
    assert fake.calls == [("edit_plan", {"run_id": "run00001", "feedback": "x", "actor": "webapp"})]


# --- fix round 1: blocking work leaves the event loop --------------------------

async def test_render_and_path_checks_run_in_threads(client, app, fake, monkeypatch):
    _report(app, fake, md=("report.md", "# R [1]"), evidence=("evidence.json", "{}"))
    seen = []
    real = asyncio.to_thread

    async def spy(fn, *a, **kw):
        seen.append(getattr(fn, "__name__", repr(fn)))
        return await real(fn, *a, **kw)

    monkeypatch.setattr(api.asyncio, "to_thread", spy)
    await _json(await client.get("/api/runs/done0001/evidence", headers=auth()))
    assert (await client.get("/api/runs/done0001/report", headers=auth())).status == 200
    assert "render_report" in seen
    assert seen.count(api._checked_path.__name__) == 2               # resolve() / is_file() off the loop


# --- fix round 1: the plan of an untracked run resting at its plan gate --------

async def test_untracked_plan_gate_run_shows_its_plan_read_only(env):
    run_id = await _at_plan_gate(env)
    tracked_plan = await runs.plan_payload(run_id)
    common._inflight.clear()                                         # the restart
    before = await _snap(env, run_id)
    edits = env.bot.edit_message_text.await_count
    deps = api.Deps(runs=runs, library=env.lib, bot=env.bot, bot_data={}, bot_token=TOKEN,
                    allowed_user_id=CHAT, max_age_s=3600, reports_root=env.tmp_path)
    async with TestClient(TestServer(server.build_app(deps))) as c:
        body = await _json(await c.get(f"/api/runs/{run_id}", headers=auth(CHAT)))
        assert body["run"]["awaiting"] == "plan"
        assert body["snapshot"]["plan"] == json.loads(json.dumps(tracked_plan))   # the card's data, as tracked
        assert run_id not in common._inflight                        # nothing registered
        after = await _snap(env, run_id)
        assert after.config["configurable"]["checkpoint_id"] == before.config["configurable"]["checkpoint_id"]
        assert env.bot.edit_message_text.await_count == edits             # no gate message touched
        # a later Approve still attaches the run and acts
        out = await _json(await c.post(f"/api/runs/{run_id}/plan", headers=auth(CHAT), json={"action": "approve"}))
        assert out["outcome"] == "started" and run_id in common._inflight
        await common._inflight[run_id].task
    assert (await env.lib.get_run(run_id))["status"] == "awaiting_report"


async def test_plan_payload_is_none_off_the_plan_gate_untracked(env):
    run_id = await _at_plan_gate(env)
    assert (await runs.approve_plan(run_id)).code == "started"
    await common._inflight[run_id].task                              # now at the report gate
    common._inflight.clear()
    assert await runs.plan_payload(run_id) is None
    assert await runs.plan_payload("nope0001") is None
    assert run_id not in common._inflight


async def test_put_engines_failure_is_400_with_the_registry_text(client, engines, lib, monkeypatch):
    monkeypatch.delenv("T_WEB_FIXED_KEY")
    registry._state.snapshot = None
    await registry.set_active_profile("other")
    body = await _json(await client.put("/api/engines", headers=auth(), json={"active_profile": "main"}), 400)
    assert body["error"]["code"] == "engine_config" and "T_WEB_FIXED_KEY" in body["error"]["message"]
    assert registry.current_config().active_profile == "other"              # nothing changed
    bad = await _json(await client.put("/api/engines", headers=auth(), json={
        "overrides": {"other": {"roles": {"no.such.role": {"primary": {"provider": "fixed", "model": "k1"}}}}}}), 400)
    assert "no.such.role" in bad["error"]["message"]


async def test_provider_models(client, engines):
    assert (await _json(await client.get("/api/engines/providers/fixed/models", headers=auth()))) == {
        "provider": "fixed", "models": ["k1", "k2"]}
    body = await _json(await client.get("/api/engines/providers/nope/models", headers=auth()), 400)
    assert "nope" in body["error"]["message"]


async def test_presets(client, engines):
    body = await _json(await client.get("/api/config/presets", headers=auth()))
    assert body["presets"] == json.loads(json.dumps(PRESETS))
    assert body["default_depth"] == {"tldr": "quick", "short": "quick", "medium": "standard", "long": "deep",
                                     "lecture": "deep"}
    assert body["exa_prices"]["price_per_search_usd"] == 0.005


async def test_every_api_route_requires_auth(client, app):
    for route in app.router.routes():
        path = route.resource.canonical
        if not path.startswith("/api"):
            continue
        url = path.replace("{id}", "run00001").replace("{name}", "fixed")
        r = await client.request(route.method, url)
        assert r.status == 401, (route.method, url)


async def test_index_until_the_frontend_exists(client, app, tmp_path):
    r = await client.get("/")
    assert r.status in (200, 404)                    # Task 12 ships webapp/static/index.html
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>Argus</title>", encoding="utf-8")
    (static / "app.js").write_text("export {}", encoding="utf-8")
    other = make_app(FakeRuns(), reports_root=tmp_path, static_dir=static)
    async with TestClient(TestServer(other)) as c:
        assert "Argus" in await (await c.get("/")).text()
        assert (await c.get("/static/app.js")).status == 200
        assert (await c.get("/static/")).status in (403, 404)       # no index listing
    missing = make_app(FakeRuns(), reports_root=tmp_path, static_dir=tmp_path / "none")
    async with TestClient(TestServer(missing)) as c:
        assert (await c.get("/")).status == 404


async def test_a_tracked_handle_without_a_row_is_not_listed(client, app, fake):
    fake.handles["ghost001"] = FakeHandle(live=True, stage="research")
    assert (await _json(await client.get("/api/runs", headers=auth())))["runs"] == []
