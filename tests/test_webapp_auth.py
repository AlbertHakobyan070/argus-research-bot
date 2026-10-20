"""Mini App initData auth (v4 spec §2.7): validation with reason codes, the
sign round-trip, field-name-only logging on 401/403, and the dev URL script."""
from __future__ import annotations

import importlib.util
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qsl, unquote, urlsplit

import httpx
import pytest
from aiohttp.test_utils import TestClient, TestServer

from argus.webapp.auth import AuthError, AuthUser, sign_init_data, validate_init_data

from webapp_fakes import TOKEN, USER_ID, FakeRuns, init_data, make_app

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
