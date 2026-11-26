"""Mini App initData auth (v4 spec §2.7): validation with reason codes, the
sign round-trip, field-name-only logging on 401/403, and the dev URL script."""
from __future__ import annotations

import hmac
import importlib.util
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, unquote, urlsplit

import httpx
import pytest
from aiohttp.test_utils import TestClient, TestServer
from yarl import URL

from argus.config import Settings
from argus.webapp import api
from argus.webapp.auth import AuthError, AuthUser, sign_init_data, validate_init_data

from webapp_fakes import SERVICE_TOKEN, TOKEN, USER_ID, FakeRuns, bearer, init_data, make_app, seed_run

ROOT = Path(__file__).resolve().parents[1]


def _validate(data: str, *, user: int = USER_ID, max_age: float = 3600, now: float | None = None) -> AuthUser:
    return validate_init_data(data, TOKEN, user, max_age, now=now)


def _code(data: str, **kw) -> str:
    with pytest.raises(AuthError) as e:
        _validate(data, **kw)
    return e.value.code


def test_valid_init_data_round_trips_to_the_user():
    user = _validate(init_data(first_name="Albert"))
    assert user == AuthUser(id=USER_ID, first_name="Albert")


def test_blank_values_are_kept_in_the_check_string():
    assert _validate(init_data(start_param="")).id == USER_ID      # signed with a blank field: still valid
    data = sign_init_data({"auth_date": int(time.time()), "user": {"id": USER_ID}, "chat_type": ""}, TOKEN)
    assert ("chat_type", "") in parse_qsl(data, keep_blank_values=True)
    assert _validate(data).id == USER_ID


def test_tampered_field_is_a_hash_mismatch():
    data = init_data().replace("AAH-test", "AAH-evil")
    assert _code(data) == "hash_mismatch"


def test_another_bots_token_is_a_hash_mismatch():
    assert _code(init_data(token="999:other-token")) == "hash_mismatch"


def test_non_ascii_hash_is_a_mismatch_not_a_crash():
    pairs = [(k, v) for k, v in parse_qsl(init_data(), keep_blank_values=True) if k != "hash"]
    data = "&".join(f"{k}={v}" for k, v in pairs) + "&hash=%C3%A9"
    assert _code(data) == "hash_mismatch"


def test_old_auth_date_is_expired():
    old = int(time.time()) - 3601
    assert _code(init_data(auth_date=old)) == "expired"
    assert _validate(init_data(auth_date=old), max_age=7200).id == USER_ID


def test_another_user_is_wrong_user():
    with pytest.raises(AuthError) as e:
        _validate(init_data(user_id=7))
    assert (e.value.code, e.value.status) == ("wrong_user", 403)


@pytest.mark.parametrize("drop", ["hash", "auth_date", "user"])
def test_missing_field(drop):
    fields = {"auth_date": int(time.time()), "user": {"id": USER_ID}}
    if drop == "hash":
        data = "&".join(f"{k}={v}" for k, v in parse_qsl(init_data(), keep_blank_values=True) if k != "hash")
    else:
        fields.pop(drop)
        data = sign_init_data(fields, TOKEN)
    with pytest.raises(AuthError) as e:
        _validate(data)
    assert (e.value.code, e.value.status) == ("missing_field", 401)


def test_auth_error_names_the_fields_present_never_values():
    with pytest.raises(AuthError) as e:
        _validate(init_data(user_id=7))
    assert set(e.value.fields) == {"auth_date", "query_id", "user", "hash"}


# --- through the server: status codes, and logs carry field names only ----

@pytest.fixture
async def client(tmp_path):
    async with TestClient(TestServer(make_app(FakeRuns(), reports_root=tmp_path))) as c:
        yield c


@pytest.mark.parametrize("header,status,code", [
    (None, 401, "missing_field"),
    ("Bearer xyz", 401, "missing_field"),
    ("tma " + init_data().replace("AAH-test", "AAH-evil"), 401, "hash_mismatch"),
    ("tma " + init_data(auth_date=int(time.time()) - 4000), 401, "expired"),
    ("tma " + init_data(user_id=7), 403, "wrong_user"),
])
async def test_api_refusals_carry_the_reason_code(client, caplog, header, status, code):
    caplog.set_level(logging.DEBUG)
    r = await client.get("/api/me", headers={"Authorization": header} if header else {})
    assert r.status == status
    assert (await r.json())["error"]["code"] == code
    logged = caplog.text
    assert code in logged
    if header and header.startswith("tma "):
        secret = header[4:]
        values = [v for _, v in parse_qsl(secret, keep_blank_values=True)]
        assert secret not in logged and "Albert" not in logged and "AAH-" not in logged
        assert not any(v in logged for v in values if len(v) > 6)       # the hash, the user JSON, auth_date
        assert "auth_date" in logged and "hash" in logged               # the field names
    assert TOKEN not in logged and TOKEN.split(":")[1] not in logged


def _webapp_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "argus.webapp"]


async def test_a_newline_in_a_field_name_cannot_forge_a_log_line(client, caplog):
    caplog.set_level(logging.DEBUG)
    header = "tma a%0AFAKE%20LOG%20LINE%20INFO%20root%3A%20injected=1&hash=x&auth_date=1"
    r = await client.get("/api/me", headers={"Authorization": header})
    assert r.status == 401
    [line] = _webapp_lines(caplog)
    assert "\n" not in line and "\r" not in line and "FAKE" not in line
    assert "hash, auth_date +1 other" in line


async def test_field_names_are_capped(client, caplog):
    caplog.set_level(logging.DEBUG)
    fields = "&".join(f"f{i}=1" for i in range(14)) + "&" + "x" * 40 + "=1"
    await client.get("/api/me", headers={"Authorization": f"tma {fields}"})
    [line] = _webapp_lines(caplog)
    assert "f0, f1, f2, f3, f4, f5, f6, f7, f8, f9 +5 other" in line and "x" * 33 not in line


async def test_a_newline_in_the_path_cannot_forge_a_log_line(client, caplog):
    caplog.set_level(logging.DEBUG)
    r = await client.get(URL("/api/%0Aforged-line", encoded=True))
    assert r.status == 401
    [line] = _webapp_lines(caplog)
    assert "\n" not in line and "'/api/\\nforged-line'" in line


async def test_valid_header_reaches_the_api(client):
    r = await client.get("/api/me", headers={"Authorization": f"tma {init_data()}"})
    assert r.status == 200
    assert (await r.json())["user"] == {"id": USER_ID, "first_name": "Albert"}


# --- scripts/webapp_dev_url.py ---------------------------------------------

def _dev_script():
    spec = importlib.util.spec_from_file_location("webapp_dev_url", ROOT / "scripts" / "webapp_dev_url.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _settings():
    return SimpleNamespace(telegram_bot_token=TOKEN, telegram_allowed_user_id=USER_ID, webapp_port=8787)


def test_dev_url_signs_for_the_allowed_user_and_never_prints_the_token(monkeypatch, capsys):
    mod = _dev_script()
    seen = []
    monkeypatch.setattr(mod, "get_settings", _settings)
    monkeypatch.setattr(mod.httpx, "get", lambda url, **kw: seen.append(url) or httpx.Response(404))
    assert mod.main(["--port", "9911"]) == 0
    out = capsys.readouterr()
    url = out.out.strip()
    assert seen == ["http://127.0.0.1:9911/"]
    assert url.startswith("http://127.0.0.1:9911/#tgWebAppData=")
    frag = dict(parse_qsl(urlsplit(url).fragment, keep_blank_values=True))
    assert (frag["tgWebAppVersion"], frag["tgWebAppPlatform"]) == ("9.0", "weba")
    assert "tgWebAppThemeParams" in frag
    assert set(dict(parse_qsl(frag["tgWebAppData"])).keys()) == {"user", "auth_date", "hash"}
    assert _validate(frag["tgWebAppData"]).id == USER_ID
    assert TOKEN not in out.out + out.err and TOKEN not in unquote(out.out)


def test_dev_url_exits_non_zero_when_the_server_is_down(monkeypatch, capsys):
    mod = _dev_script()
    monkeypatch.setattr(mod, "get_settings", _settings)

    def down(url, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(mod.httpx, "get", down)
    assert mod.main([]) != 0
    out = capsys.readouterr()
    assert out.out == "" and "8787" in out.err


# --- the service token: Authorization: Bearer, for other programs -------------

ALLOWED = [("POST", "/api/runs", {"topic": "x"}), ("GET", "/api/runs/svc00001", None),
           ("GET", "/api/runs/svc00001/events", None), ("GET", "/api/runs/svc00001/evidence", None),
           ("GET", "/api/runs/svc00001/report", None), ("POST", "/api/runs/svc00001/plan", {"action": "approve"}),
           ("POST", "/api/runs/svc00001/cancel", None)]


def _service_app(tmp_path, **kw):
    fake = FakeRuns()
    app = make_app(fake, reports_root=tmp_path, **kw)
    folder = tmp_path / "svc00001"
    folder.mkdir()
    (folder / "report.md").write_text("# R", encoding="utf-8")
    (folder / "evidence.json").write_text("{}", encoding="utf-8")
    seed_run(fake, app, "svc00001", status="awaiting_report", next_=("deliver",), origin="api",
             values={"report_paths": {"md": str(folder / "report.md"), "evidence": str(folder / "evidence.json")}})
    return fake, app


@pytest.fixture
async def service(tmp_path):
    fake, app = _service_app(tmp_path, service_token=SERVICE_TOKEN)
    async with TestClient(TestServer(app)) as c:
        yield SimpleNamespace(client=c, app=app, fake=fake)


async def _code_of(r) -> str:
    return (await r.json())["error"]["code"]


async def test_a_bearer_passes_on_every_allowlisted_route(service):
    for method, path, body in ALLOWED:
        r = await service.client.request(method, path, headers=bearer(), json=body)
        assert r.status == 200, (method, path, await r.text())
    assert [name for name, _ in service.fake.calls] == ["start_run", "approve_plan", "cancel_run"]
    assert service.fake.calls[0][1]["user_id"] == 0                    # the service principal


async def test_a_bearer_is_refused_on_every_other_route(service):
    allowed = {(m, p.replace("svc00001", "{id}")) for m, p, _ in ALLOWED}
    refused = set()
    for route in service.app.router.routes():
        path = route.resource.canonical
        if not path.startswith("/api") or (route.method, path) in allowed:
            continue
        url = path.replace("{id}", "svc00001").replace("{name}", "fixed")
        body = {"action": "send", "what": ["md"]} if route.method in ("POST", "PUT") else None
        r = await service.client.request(route.method, url, headers=bearer(), json=body)
        assert r.status == 401, (route.method, url)
        if route.method != "HEAD":
            assert await _code_of(r) == "not_allowed", (route.method, url)
        refused.add((route.method, path))
    assert {("POST", "/api/runs/{id}/report"), ("POST", "/api/runs/{id}/send"), ("GET", "/api/runs"),
            ("POST", "/api/runs/{id}/continue"), ("GET", "/api/me"), ("PUT", "/api/engines")} <= refused
    r = await service.client.get("/api/no-such-route", headers=bearer())
    assert r.status == 401 and await _code_of(r) == "not_allowed"
    assert service.fake.calls == [] and service.app[api.DEPS].bot.documents == []


@pytest.mark.parametrize("host", [None, "plain-river-quiet-hill.trycloudflare.com", "host.docker.internal:9999",
                                  "127.0.0.1:8787"])
async def test_a_bearer_on_another_host_is_401_as_if_the_feature_were_off(service, caplog, host):
    caplog.set_level(logging.DEBUG)
    headers = {**bearer(host=host), "X-Forwarded-Host": "host.docker.internal:8787"}   # never trusted
    r = await service.client.post("/api/runs", headers=headers, json={"topic": "x"})
    assert r.status == 401 and await _code_of(r) == "missing_field"      # the public side cannot tell
    assert "wrong_host" in caplog.text                                   # the server log can
    assert service.fake.calls == []


@pytest.mark.parametrize("configured", ["", None])
async def test_without_a_configured_token_every_bearer_is_401(tmp_path, configured):
    kw = {} if configured is None else {"service_token": configured}
    fake, app = _service_app(tmp_path, **kw)
    async with TestClient(TestServer(app)) as c:
        for method, path, body in ALLOWED:
            for token in (SERVICE_TOKEN, ""):
                r = await c.request(method, path, headers=bearer(token), json=body)
                assert r.status == 401, (method, path, token)
    assert fake.calls == []


@pytest.mark.parametrize("token", ["svc-wrong", SERVICE_TOKEN + "x", SERVICE_TOKEN[:-1], "", "tma"])
async def test_a_wrong_token_is_401_and_never_logged(service, caplog, token):
    caplog.set_level(logging.DEBUG)
    r = await service.client.post("/api/runs", headers=bearer(token), json={"topic": "x"})
    assert r.status == 401 and await _code_of(r) == "bad_token"
    assert service.fake.calls == []
    assert "bad_token" in caplog.text
    assert SERVICE_TOKEN not in caplog.text and (len(token) < 4 or token not in caplog.text)


async def test_the_token_is_compared_with_compare_digest(service, monkeypatch):
    seen = []
    real = hmac.compare_digest
    monkeypatch.setattr(hmac, "compare_digest", lambda a, b: seen.append((a, b)) or real(a, b))
    r = await service.client.post("/api/runs", headers=bearer(), json={"topic": "x"})
    assert r.status == 200
    assert (SERVICE_TOKEN.encode(), SERVICE_TOKEN.encode()) in seen
    seen.clear()
    r = await service.client.post("/api/runs", headers=bearer("svc-wrong"), json={"topic": "x"})
    assert r.status == 401 and (b"svc-wrong", SERVICE_TOKEN.encode()) in seen


async def test_a_tma_user_is_unchanged_beside_a_service_token(service):
    tma = {"Authorization": f"tma {init_data()}"}
    r = await service.client.get("/api/me", headers=tma)
    assert r.status == 200 and (await r.json())["user"] == {"id": USER_ID, "first_name": "Albert"}
    r = await service.client.post("/api/runs", headers=tma, json={"topic": "x"})
    assert r.status == 200
    assert (service.fake.calls[-1][1]["origin"], service.fake.calls[-1][1]["user_id"]) == ("webapp", USER_ID)


_SERVICE_ENV = ("ARGUS_SERVICE_TOKEN", "ARGUS_SERVICE_HOSTS", "ARGUS_SERVICE_ORIGIN")


@pytest.fixture
def service_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    for name in _SERVICE_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_service_settings_default_to_off(service_env):
    s = Settings.load()
    assert (s.service_token, s.service_hosts, s.service_origin) == ("", ("host.docker.internal:8787",), "api")


def test_service_settings_from_the_environment(service_env):
    service_env.setenv("ARGUS_SERVICE_TOKEN", f" {SERVICE_TOKEN} ")
    service_env.setenv("ARGUS_SERVICE_HOSTS", " Host.Docker.Internal:8787, 127.0.0.1:8787 ,, ")
    service_env.setenv("ARGUS_SERVICE_ORIGIN", "jobs_v2")
    s = Settings.load()
    assert (s.service_token, s.service_hosts, s.service_origin) == (
        SERVICE_TOKEN, ("host.docker.internal:8787", "127.0.0.1:8787"), "jobs_v2")
    assert SERVICE_TOKEN not in repr(s)


@pytest.mark.parametrize("origin", ["API", "a b", "x" * 33, "telegram", "webapp", "legacy"])
def test_a_bad_service_origin_is_refused(service_env, origin):
    service_env.setenv("ARGUS_SERVICE_ORIGIN", origin)
    with pytest.raises(ValueError, match="ARGUS_SERVICE_ORIGIN"):
        Settings.load()
