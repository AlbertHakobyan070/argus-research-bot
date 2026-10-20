"""The cloudflared tunnel supervisor and the console lifecycle (v4 spec §2.7):
URL parsing from cloudflared's stderr, restart with exponential backoff on a
fake process, and ``server.start``/``stop`` on a fake bot — a failing console
logs, DMs and sets ``webapp_ok=False`` without raising into the bot."""
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from telegram import MenuButtonWebApp
from telegram.error import NetworkError

from argus.bot import app as bot_app
from argus.config import Settings
from argus.webapp import server, tunnel

from webapp_fakes import (FAILED_TUNNEL_STDERR, QUICK_TUNNEL_STDERR, TOKEN, USER_ID, FakeBot, FakeProcess)

URL = "https://plain-river-quiet-hill.trycloudflare.com"


# --- URL parsing --------------------------------------------------------------

def test_url_from_cloudflared_stderr():
    found = [u for u in map(tunnel.find_url, QUICK_TUNNEL_STDERR) if u]
    assert found == [URL]


def test_cloudflareds_own_api_host_is_not_the_tunnel_url():
    assert [tunnel.find_url(line) for line in FAILED_TUNNEL_STDERR] == [None]


def test_command_and_binary(monkeypatch):
    monkeypatch.delenv("ARGUS_CLOUDFLARED", raising=False)
    name = "cloudflared.exe" if os.name == "nt" else "cloudflared"
    assert tunnel.binary() == tunnel._ROOT / "tools" / "bin" / name
    monkeypatch.setenv("ARGUS_CLOUDFLARED", "/opt/cf/cloudflared")
    assert tunnel.binary() == Path("/opt/cf/cloudflared")
    assert tunnel.command(Path("cf"), 8787) == ["cf", "tunnel", "--url", "http://127.0.0.1:8787", "--no-autoupdate"]


# --- the supervisor on fake processes -----------------------------------------

class _Script:
    """``spawn`` hands out the queued processes, then parks forever; the fake
    ``sleep`` records each backoff delay; ``clock`` is advanced by hand."""

    def __init__(self, procs):
        self.procs = list(procs)
        self.spawned: list[list[str]] = []
        self.sleeps: list[float] = []
        self.now = 0.0
        self.parked = asyncio.Event()
        self.ups: list[str] = []
        self.downs: list[str] = []

    async def spawn(self, cmd):
        self.spawned.append(cmd)
        if not self.procs:
            self.parked.set()
            await asyncio.Event().wait()
        proc = self.procs.pop(0)
        if isinstance(proc, BaseException):
            raise proc
        return proc() if callable(proc) else proc

    async def sleep(self, s):
        self.sleeps.append(s)

    async def on_url(self, url):
        self.ups.append(url)

    async def on_down(self, reason):
        self.downs.append(reason)

    def supervisor(self, tmp_path, **kw):
        return tunnel.Supervisor(["cf"], on_url=self.on_url, on_down=self.on_down, spawn=self.spawn,
                                 sleep=self.sleep, clock=lambda: self.now, log_path=tmp_path / "logs" / "cf.log", **kw)


async def _until(cond, timeout=5.0):
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.01)


async def test_supervisor_reports_the_url_and_restarts_with_backoff(tmp_path):
    first = FakeProcess(QUICK_TUNNEL_STDERR)
    quick_exits = [FakeProcess(exit_code=1) for _ in range(11)]
    s = _Script([first, *quick_exits])
    sup = s.supervisor(tmp_path)
    sup.start()
    await _until(lambda: s.ups == [URL])
    first.exit(3)                                       # killed: the supervisor restarts it
    await s.parked.wait()
    await sup.stop()
    assert s.sleeps == [1, 2, 4, 8, 16, 32, 64, 128, 256, 300, 300, 300]   # capped at 5 min
    assert s.downs[0] == "cloudflared exited with code 3; restarting in 1 s"
    assert s.downs[1] == "cloudflared exited with code 1 before printing its URL; restarting in 2 s"
    assert len(s.spawned) == 13
    log = (tmp_path / "logs" / "cf.log").read_text(encoding="utf-8")
    assert URL in log and "Registered tunnel connection" in log          # stderr kept in the log


async def test_backoff_resets_after_a_stable_run(tmp_path):
    procs = [FakeProcess(exit_code=1), FakeProcess(exit_code=1), FakeProcess(QUICK_TUNNEL_STDERR),
             FakeProcess(exit_code=1)]
    s = _Script(procs)
    sup = s.supervisor(tmp_path)
    sup.start()
    await _until(lambda: s.ups == [URL])
    s.now += tunnel.STABLE_S                             # it ran a while before it died
    procs[2].exit(1)
    await s.parked.wait()
    await sup.stop()
    assert s.sleeps == [1, 2, 1, 2]


async def test_no_url_in_time_terminates_and_restarts(tmp_path):
    silent = FakeProcess(["INF Starting tunnel"])
    s = _Script([silent])
    sup = s.supervisor(tmp_path, url_timeout_s=0.05)
    sup.start()
    await s.parked.wait()
    await sup.stop()
    assert silent.terminated
    assert s.downs == ["cloudflared printed no URL within 0.05 s; restarting in 1 s"]


async def test_spawn_failure_is_reported_and_retried(tmp_path):
    s = _Script([FileNotFoundError(2, "No such file"), FakeProcess(QUICK_TUNNEL_STDERR)])
    sup = s.supervisor(tmp_path)
    sup.start()
    await _until(lambda: s.ups == [URL])
    await sup.stop()
    assert s.downs[0].startswith("cloudflared could not start (") and s.sleeps == [1]


async def test_stop_terminates_the_running_process(tmp_path):
    proc = FakeProcess(QUICK_TUNNEL_STDERR)
    s = _Script([proc])
    sup = s.supervisor(tmp_path)
    sup.start()
    await _until(lambda: s.ups == [URL])
    await sup.stop()
    await sup.stop()                                      # idempotent
    assert proc.terminated and s.downs == []


# --- server.start / stop on a fake bot ----------------------------------------

def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _settings(tmp_path, port, **kw) -> Settings:
    return Settings(freellmapi_base_url="http://127.0.0.1:9/v1", freellmapi_api_key="", telegram_bot_token=TOKEN,
                    telegram_allowed_user_id=USER_ID, reports_root=tmp_path / "reports",
                    checkpoint_db=tmp_path / "cp.sqlite", library_db=tmp_path / "lib.sqlite",
                    vault_root=tmp_path / "vault", media_root=tmp_path / "m", transcripts_root=tmp_path / "t",
                    history_root=tmp_path / "h", ffmpeg_path=None, webapp_port=port, **kw)


@pytest.fixture
def ptb(tmp_path, monkeypatch):
    """A PTB application stand-in and its settings; the tunnel off unless a test says so."""
    port = _free_port()
    monkeypatch.setattr(server, "get_settings", lambda: _settings(tmp_path, port))
    monkeypatch.setenv("ARGUS_WEBAPP_TUNNEL", "none")
    monkeypatch.delenv("ARGUS_WEBAPP_PUBLIC_URL", raising=False)
    monkeypatch.setattr(tunnel, "LOG_PATH", tmp_path / "logs" / "cloudflared.log")
    app = SimpleNamespace(bot=FakeBot(), bot_data={"library": None})
    app.port = port
    return app


async def _get(port, path="/api/me"):
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n".encode())
    data = await reader.read()
    writer.close()
    return data.decode()


async def test_start_local_only_then_stop(ptb, caplog):
    caplog.set_level("INFO")
    await server.start(ptb)
    try:
        assert (ptb.bot_data["webapp_ok"], ptb.bot_data["webapp_error"]) == (True, None)
        assert ptb.bot_data["webapp_public_url"] == f"http://127.0.0.1:{ptb.port}"
        assert f"webapp: listening on 127.0.0.1:{ptb.port}" in caplog.text
        assert (await _get(ptb.port)).startswith("HTTP/1.1 401")
        assert ptb.bot.menu_buttons == [] and ptb.bot.messages == []
    finally:
        await server.stop(ptb)
    await server.stop(ptb)                                # idempotent
    with pytest.raises(OSError):
        await _get(ptb.port)


async def test_a_failing_start_is_logged_dmd_and_shown_never_raised(ptb, caplog):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", ptb.port))
        busy.listen()
        await server.start(ptb)                           # must not raise into post_init
    assert ptb.bot_data["webapp_ok"] is False and ptb.bot_data["webapp_public_url"] is None
    assert ptb.bot_data["webapp_error"].startswith("server:")
    assert len(ptb.bot.messages) == 1 and ptb.bot.messages[0]["chat_id"] == USER_ID
    assert "Console" in ptb.bot.messages[0]["text"] and "webapp" in caplog.text
    await server.stop(ptb)


async def test_bad_tunnel_mode_is_a_visible_failure(ptb, monkeypatch):
    monkeypatch.setenv("ARGUS_WEBAPP_TUNNEL", "ngrok")
    await server.start(ptb)
    assert ptb.bot_data["webapp_ok"] is False and "ARGUS_WEBAPP_TUNNEL" in ptb.bot_data["webapp_error"]
    assert len(ptb.bot.messages) == 1
    await server.stop(ptb)


async def test_disabled_console_starts_nothing(ptb, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "get_settings", lambda: _settings(tmp_path, ptb.port, webapp_enabled=False))
    await server.start(ptb)
    assert "webapp_ok" not in ptb.bot_data and server._KEY not in ptb.bot_data
    await server.stop(ptb)


async def test_public_url_registers_the_menu_button(ptb, monkeypatch, caplog):
    caplog.set_level("INFO")
    monkeypatch.setenv("ARGUS_WEBAPP_PUBLIC_URL", "https://argus.example.org")
    await server.start(ptb)
    try:
        assert ptb.bot_data["webapp_public_url"] == "https://argus.example.org" and ptb.bot_data["webapp_ok"]
        [call] = ptb.bot.menu_buttons
        assert call["chat_id"] == USER_ID and isinstance(call["menu_button"], MenuButtonWebApp)
        assert (call["menu_button"].text, call["menu_button"].web_app.url) == ("Argus", "https://argus.example.org")
        assert f"menu button registered for chat {USER_ID}" in caplog.text
    finally:
        await server.stop(ptb)


async def test_missing_cloudflared_is_a_visible_failure_and_local_still_serves(ptb, monkeypatch, tmp_path):
    monkeypatch.setenv("ARGUS_WEBAPP_TUNNEL", "quick")
    monkeypatch.setenv("ARGUS_CLOUDFLARED", str(tmp_path / "no-such-cloudflared"))
    await server.start(ptb)
    try:
        assert ptb.bot_data["webapp_ok"] is False
        assert "cloudflared" in ptb.bot_data["webapp_error"] and "not found" in ptb.bot_data["webapp_error"]
        assert len(ptb.bot.messages) == 1
        assert (await _get(ptb.port)).startswith("HTTP/1.1 401")
    finally:
        await server.stop(ptb)


async def test_quick_tunnel_url_menu_button_restart_and_dm(ptb, monkeypatch, tmp_path, caplog):
    caplog.set_level("INFO")
    cf = tmp_path / "cloudflared.exe"
    cf.write_bytes(b"")
    monkeypatch.setenv("ARGUS_WEBAPP_TUNNEL", "quick")
    monkeypatch.setenv("ARGUS_CLOUDFLARED", str(cf))
    second = "https://other-name-two.trycloudflare.com"
    procs = [FakeProcess(QUICK_TUNNEL_STDERR), FakeProcess([f"INF |  {second}  |"])]
    spawned = []

    async def spawn(cmd):
        spawned.append(cmd)
        if not procs:
            await asyncio.Event().wait()
        return procs.pop(0)

    async def no_wait(s):
        pass

    first = procs[0]
    monkeypatch.setattr(tunnel, "spawn", spawn)
    monkeypatch.setattr(tunnel, "_sleep", no_wait)
    await server.start(ptb)
    try:
        assert ptb.bot_data["webapp_ok"] is False and "starting" in ptb.bot_data["webapp_error"]
        await _until(lambda: ptb.bot_data.get("webapp_public_url") == URL)
        assert (ptb.bot_data["webapp_ok"], ptb.bot_data["webapp_error"]) == (True, None)
        assert spawned[0] == [str(cf), "tunnel", "--url", f"http://127.0.0.1:{ptb.port}", "--no-autoupdate"]
        assert [b["menu_button"].web_app.url for b in ptb.bot.menu_buttons] == [URL]
        assert ptb.bot.messages == []                       # the first URL is not news
        assert f"tunnel: {URL}" in caplog.text
        first.exit(1)                                       # cloudflared killed
        await _until(lambda: ptb.bot_data.get("webapp_public_url") == second)
        assert [b["menu_button"].web_app.url for b in ptb.bot.menu_buttons] == [URL, second]
        texts = [m["text"] for m in ptb.bot.messages]
        assert len(texts) == 2 and "exited" in texts[0] and second in texts[1]
    finally:
        await server.stop(ptb)


@pytest.mark.parametrize("error", [NetworkError("boom"), RuntimeError("bug")])
async def test_menu_button_failure_is_visible(ptb, monkeypatch, error):
    monkeypatch.setenv("ARGUS_WEBAPP_PUBLIC_URL", "https://argus.example.org")
    ptb.bot.fail_menu = error
    await server.start(ptb)
    try:
        assert ptb.bot_data["webapp_ok"] is False and "menu button" in ptb.bot_data["webapp_error"]
        assert ptb.bot_data["webapp_public_url"] == "https://argus.example.org"   # the console itself is up
        assert len(ptb.bot.messages) == 1
    finally:
        await server.stop(ptb)


@pytest.mark.parametrize("module", ["argus.webapp.server", "argus.webapp.api"])
def test_the_webapp_imports_before_the_bot(module):
    """bot.app imports webapp.server at module level, so the webapp must not
    import the bot package at module level (a circular import otherwise)."""
    done = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": ""}, timeout=120)
    assert done.returncode == 0, done.stderr


def test_hooks_are_registered():
    assert server.start in bot_app.POST_INIT_HOOKS and server.stop in bot_app.POST_SHUTDOWN_HOOKS
    assert bot_app.POST_INIT_HOOKS.index(server.start) > 0      # after runs.init


def test_logs_dir_is_ignored():
    lines = (tunnel._ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "logs/" in lines
