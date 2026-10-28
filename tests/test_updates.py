"""Backend updates (``argus.updates`` and ``/api/updates``): the fixed
component table reaches uv as a list (no request input), a dry run guards
every upgrade, a plan that changes another package is refused, uv's failure
is surfaced, and only the owner can call the routes."""
from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer

from argus import updates
from webapp_fakes import FakeRuns, auth, make_app

_DRY_DDGS = """Using Python 3.12.13 environment at: venv
Resolved 5 packages in 594ms
Would download 1 package
Would uninstall 1 package
Would install 1 package
 - ddgs==9.14.4
 + ddgs==9.16.1
"""
_DONE_DDGS = """Resolved 5 packages in 400ms
Uninstalled 1 package in 20ms
Installed 1 package in 50ms
 - ddgs==9.14.4
 + ddgs==9.16.1
"""
_DRY_NATIVE = """Would install 2 packages
 - yt-dlp==2026.10.26
 + yt-dlp==2026.11.2
 - brotli==1.2.0
 + brotli==1.2.1
"""


@pytest.fixture
def uv(monkeypatch):
    """A fake uv: ``replies`` in order; ``calls`` the argv lists it got."""
    state = {"replies": [], "calls": []}

    async def run(cmd):
        state["calls"].append(cmd)
        reply = state["replies"].pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply
    monkeypatch.setattr(updates, "_run", run)
    monkeypatch.setattr(updates.shutil, "which", lambda name: "C:/uv/uv.exe")
    monkeypatch.setattr(updates.sys, "base_prefix", "C:/base-python")    # running from a venv
    return state


async def test_an_upgrade_dry_runs_then_installs_only_its_own_packages(uv):
    uv["replies"] = [_DRY_DDGS, _DONE_DDGS]
    out = await updates.upgrade("ddgs")
    dry, real = uv["calls"]
    assert "--dry-run" in dry and "--dry-run" not in real
    assert real[:3] == ["C:/uv/uv.exe", "pip", "install"] and real[-1] == "ddgs"
    assert real[real.index("--upgrade-package") + 1] == "ddgs" and real.count("--upgrade-package") == 1
    assert out["changes"] == [{"op": "remove", "package": "ddgs", "version": "9.14.4"},
                              {"op": "install", "package": "ddgs", "version": "9.16.1"}]
    assert out["restart_needed"] is True


async def test_yt_dlp_moves_with_its_challenge_solver(uv):
    uv["replies"] = ["Would make no changes\n"]
    await updates.upgrade("yt-dlp")
    [dry] = uv["calls"]
    assert dry[-1] == "yt-dlp[default]"
    assert [dry[i + 1] for i, a in enumerate(dry) if a == "--upgrade-package"] == ["yt-dlp", "yt-dlp-ejs"]


async def test_nothing_to_change_installs_nothing(uv):
    uv["replies"] = ["Resolved 11 packages\nWould make no changes\n"]
    out = await updates.upgrade("yt-dlp")
    assert len(uv["calls"]) == 1 and out["changes"] == [] and out["output"] == "already up to date"


async def test_a_plan_that_changes_another_package_is_refused_before_installing(uv):
    uv["replies"] = [_DRY_NATIVE]
    with pytest.raises(updates.UpdateError) as e:
        await updates.upgrade("yt-dlp")
    assert e.value.code == "refused" and "brotli" in str(e.value) and "stop the bot" in str(e.value)
    assert len(uv["calls"]) == 1, "only the dry run ran"


async def test_without_uv_the_upgrade_fails_loudly(uv, monkeypatch):
    monkeypatch.setattr(updates.shutil, "which", lambda name: None)
    with pytest.raises(updates.UpdateError, match="uv is not on the bot's PATH"):
        await updates.upgrade("ddgs")


async def test_outside_a_venv_the_upgrade_is_refused(uv, monkeypatch):
    monkeypatch.setattr(updates.sys, "base_prefix", updates.sys.prefix)
    with pytest.raises(updates.UpdateError, match="not running from a virtual environment"):
        await updates.upgrade("ddgs")


async def test_uv_exit_code_is_reported(monkeypatch):
    monkeypatch.setattr(updates, "_TIMEOUT_S", 30)
    with pytest.raises(updates.UpdateError) as e:
        await updates._run([updates.sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
    assert e.value.code == "failed" and "code 3" in str(e.value) and "boom" in str(e.value)


async def test_status_reads_pypi_and_reports_its_failure(monkeypatch):
    async def latest(client, package):
        if package == "ddgs":
            raise RuntimeError("pypi down")
        return "2099.1.1"
    monkeypatch.setattr(updates, "latest", latest)
    by = {c["name"]: c for c in await updates.status()}
    assert by["yt-dlp"]["latest"] == "2099.1.1" and by["yt-dlp"]["latest_error"] is None
    assert [p["name"] for p in by["yt-dlp"]["packages"]] == ["yt-dlp", "yt-dlp-ejs"]
    assert by["ddgs"]["latest"] is None and by["ddgs"]["latest_error"] == "RuntimeError: pypi down"


# --- the routes -----------------------------------------------------------------------------------

@pytest.fixture
async def client(tmp_path):
    async with TestClient(TestServer(make_app(FakeRuns(), reports_root=tmp_path))) as c:
        yield c


async def test_only_the_owner_can_update(client, uv):
    r = await client.post("/api/updates/ddgs", headers=auth(user_id=7))
    assert r.status == 403 and uv["calls"] == []
    assert (await client.get("/api/updates", headers=auth(user_id=7))).status == 403


async def test_an_unknown_component_is_404_and_never_reaches_uv(client, uv):
    r = await client.post("/api/updates/pip%20install%20evil", headers=auth())
    assert r.status == 404 and (await r.json())["error"]["code"] == "not_found" and uv["calls"] == []


async def test_a_refused_update_is_409_with_the_reason(client, uv):
    uv["replies"] = [_DRY_NATIVE]
    r = await client.post("/api/updates/yt-dlp", headers=auth())
    body = await r.json()
    assert r.status == 409 and body["error"]["code"] == "refused" and "brotli" in body["error"]["message"]


async def test_a_successful_update_returns_the_changes_and_fresh_status(client, uv, monkeypatch):
    async def latest(client, package):
        return "9.16.1"
    monkeypatch.setattr(updates, "latest", latest)
    uv["replies"] = [_DRY_DDGS, _DONE_DDGS]
    r = await client.post("/api/updates/ddgs", headers=auth())
    body = await r.json()
    assert r.status == 200, body
    assert body["name"] == "ddgs" and body["restart_needed"] is True
    assert {c["name"] for c in body["components"]} == set(updates.COMPONENTS)
