"""Fakes for the Mini App backend (v4 WP-E2): a fake run driver, library, bot
and cloudflared process, and ``make_app(fake)`` — the real console app
(``webapp.server.build_app``) on top of them, with no live bot.

Task 12 reuses it to render-check the console locally::

    from webapp_fakes import FakeRuns, make_app, seed_run, init_data
    fake = FakeRuns()
    app = make_app(fake, reports_root=tmp)
    seed_run(fake, app, "r1", status="awaiting_plan", next_=("research",))
    web.run_app(app, host="127.0.0.1", port=8787)  # then open #tgWebAppData=<init_data()>
"""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from aiohttp import web

from argus.bot import runs
from argus.bot.progress import RunEventBus
from argus.bot.runs import Outcome
from argus.webapp import api, server
from argus.webapp.auth import sign_init_data

TOKEN = "123456:TEST-not-a-real-token"
USER_ID = 42
_GATES = {("research",): "plan", ("deliver",): "report"}


def init_data(user_id: int = USER_ID, *, auth_date: int | None = None, first_name: str = "Albert",
              token: str = TOKEN, **extra) -> str:
    """Signed initData for ``user_id`` (``auth_date`` defaults to now)."""
    fields = {"auth_date": int(time.time()) if auth_date is None else auth_date,
              "query_id": "AAH-test", "user": {"id": user_id, "first_name": first_name}, **extra}
    return sign_init_data(fields, token)


def auth(user_id: int = USER_ID, **kw) -> dict:
    """The ``Authorization`` header of a console request."""
    return {"Authorization": f"tma {init_data(user_id, **kw)}"}


def dev_fragment(**kw) -> str:
    """The URL fragment ``scripts/webapp_dev_url.py`` prints, for the fake's token."""
    return f"#tgWebAppData={quote(init_data(**kw), safe='')}&tgWebAppVersion=9.0&tgWebAppPlatform=weba"


@dataclass
class FakeHandle:
    """What the API reads of a ``RunHandle``."""
    live: bool = False
    stage: str = ""


class FakeRuns:
    """Stands in for ``bot/runs.py``. Every call is bound against the real
    driver function's signature first (``inspect.signature(...).bind``), so
    a kwarg the driver does not take fails the test as it would fail live.
    Each action is recorded in ``calls`` as ``(name, the arguments passed)``
    and answered with ``outcomes[name]`` (default: a success code).
    ``handles`` are the tracked runs, ``snapshots`` and ``plans`` what
    ``snapshot()`` / ``plan_payload()`` return; ``bus`` is a real
    ``RunEventBus``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.outcomes: dict[str, Outcome] = {}
        self.handles: dict[str, FakeHandle] = {}
        self.snapshots: dict[str, dict] = {}
        self.plans: dict[str, dict] = {}
        self.bus = RunEventBus()

    @staticmethod
    def _bind(name: str, args: tuple, kwargs: dict) -> dict:
        return dict(inspect.signature(getattr(runs, name)).bind(*args, **kwargs).arguments)

    def _act(self, name: str, args: tuple, kwargs: dict, default: Outcome) -> Outcome:
        self.calls.append((name, self._bind(name, args, kwargs)))
        return self.outcomes.get(name, default)

    def tracked(self, *args, **kw) -> dict:
        self._bind("tracked", args, kw)
        return dict(self.handles)

    async def snapshot(self, *args, **kw) -> dict | None:
        return self.snapshots.get(self._bind("snapshot", args, kw)["run_id"])

    async def plan_payload(self, *args, **kw) -> dict | None:
        return self.plans.get(self._bind("plan_payload", args, kw)["run_id"])

    async def start_run(self, *args, **kw) -> Outcome:
        return self._act("start_run", args, kw, Outcome("started", "", "new00001"))

    async def approve_plan(self, *args, **kw) -> Outcome:
        out = self._act("approve_plan", args, kw, None)
        return out or Outcome("started", "", self.calls[-1][1]["run_id"])

    async def edit_plan(self, *args, **kw) -> Outcome:
        out = self._act("edit_plan", args, kw, None)
        return out or Outcome("started", "", self.calls[-1][1]["run_id"])

    async def report_action(self, *args, **kw) -> Outcome:
        out = self._act("report_action", args, kw, None)
        call = self.calls[-1][1]
        return out or Outcome("delivered" if call["action"] == "send" else "started", "", call["run_id"])

    async def cancel_run(self, *args, **kw) -> Outcome:
        out = self._act("cancel_run", args, kw, None)
        run_id = self.calls[-1][1]["run_id"]
        return out or Outcome("cancelled", f"❌ Cancelled run `{run_id}`.", run_id)


class FakeLibrary:
    """The library calls the console makes, plus the settings kv the engines
    registry persists to; rows are plain dicts."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.settings: dict[str, str] = {}
        self.exa_usd = 0.0

    async def list_runs(self, *, chat_id: int | None = None, limit: int = 10) -> list[dict]:
        rows = sorted(self.rows.values(), key=lambda r: r["created_at"], reverse=True)
        return [r for r in rows if chat_id is None or r["chat_id"] == chat_id][:limit]

    async def get_run(self, run_id: str) -> dict | None:
        return self.rows.get(run_id)

    async def exa_usd_month_to_date(self) -> float:
        return self.exa_usd

    async def get_setting(self, key: str, default: str | None = None) -> str | None:
        return self.settings.get(key, default)

    async def set_setting(self, key: str, value: str) -> None:
        self.settings[key] = value


class FakeBot:
    """Records ``send_document`` (with the file's bytes), ``send_message`` and
    ``set_chat_menu_button``; ``fail_menu`` makes the last raise."""

    def __init__(self) -> None:
        self.documents: list[dict] = []
        self.messages: list[dict] = []
        self.menu_buttons: list[dict] = []
        self.fail_menu: Exception | None = None

    async def send_document(self, *, chat_id, document, filename, caption=""):
        self.documents.append({"chat_id": chat_id, "filename": filename, "caption": caption,
                               "data": document.read()})

    async def send_message(self, *, chat_id, text, reply_markup=None, parse_mode=None):
        self.messages.append({"chat_id": chat_id, "text": text})

    async def set_chat_menu_button(self, *, chat_id=None, menu_button=None):
        if self.fail_menu is not None:
            raise self.fail_menu
        self.menu_buttons.append({"chat_id": chat_id, "menu_button": menu_button})
        return True


def make_app(fake: FakeRuns, *, reports_root: Path, library: FakeLibrary | None = None,
             bot: FakeBot | None = None, bot_data: dict | None = None, max_age_s: float = 3600,
             static_dir: Path = server.STATIC_DIR) -> web.Application:
    """The real console app on the fakes; ``app[api.DEPS]`` holds them."""
    deps = api.Deps(runs=fake, library=library or FakeLibrary(), bot=bot or FakeBot(),
                    bot_data=bot_data if bot_data is not None else {"webapp_ok": True},
                    bot_token=TOKEN, allowed_user_id=USER_ID, max_age_s=max_age_s,
                    reports_root=Path(reports_root))
    return server.build_app(deps, static_dir=static_dir)


def seed_run(fake: FakeRuns, app: web.Application, run_id: str, *, status: str = "running",
             next_: tuple = (), live: bool = False, stage: str = "", values: dict | None = None,
             tracked: bool | None = None, **row) -> dict:
    """A library row plus the driver's view of it (``snapshot``'s formula:
    ``awaiting`` is the gate only while no drive is live)."""
    lib = app[api.DEPS].library
    lib.rows[run_id] = {
        "run_id": run_id, "thread_id": f"tg:{USER_ID}:{run_id}", "chat_id": USER_ID,
        "topic": row.pop("topic", f"topic of {run_id}"), "length": "short", "mode": "deep", "status": status,
        "report_dir": None, "created_at": row.pop("created_at", "2026-10-21T10:00:00+00:00"),
        "updated_at": "2026-10-21T10:05:00+00:00", "config_json": None, "depth": "quick",
        "archetype": None, "origin": "telegram", "exa_usd": None, **row}
    if tracked if tracked is not None else (live or bool(next_)):
        fake.handles[run_id] = FakeHandle(live=live, stage=stage)
    fake.snapshots[run_id] = {"run_id": run_id, "in_flight": live, "awaiting": None if live else _GATES.get(next_),
                              "next": list(next_), "stage": stage, "started_at": None,
                              "last_seq": fake.bus.last_seq(run_id), "values": values or {}}
    return lib.rows[run_id]


class FakeProcess:
    """A cloudflared stand-in: ``stderr`` is a real ``StreamReader`` fed with
    ``lines``; it runs until ``exit()`` (``terminate``/``kill`` exit it too)."""

    def __init__(self, lines: list[str] = (), *, exit_code: int | None = None) -> None:
        self.stderr = asyncio.StreamReader()
        for line in lines:
            self.stderr.feed_data(line.encode() + b"\n")
        self.returncode: int | None = None
        self.terminated = False
        self._exited = asyncio.Event()
        if exit_code is not None:
            self.exit(exit_code)

    def exit(self, code: int = 1) -> None:
        if self.returncode is None:
            self.returncode = code
            self.stderr.feed_eof()
            self._exited.set()

    async def wait(self) -> int:
        await self._exited.wait()
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.exit(-15)

    def kill(self) -> None:
        self.exit(-9)


# A quick tunnel's stderr as cloudflared prints it (the URL box), and a
# failed attempt that quotes cloudflared's own API host.
QUICK_TUNNEL_STDERR = [
    "2026-10-21T10:00:00Z INF Thank you for trying Cloudflare Tunnel. Doing so, without a Cloudflare account, "
    "is a quick way to experiment and try it out.",
    "2026-10-21T10:00:00Z INF Requesting new quick Tunnel on trycloudflare.com...",
    "2026-10-21T10:00:01Z INF +--------------------------------------------------------------------------------------------+",
    "2026-10-21T10:00:01Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |",
    "2026-10-21T10:00:01Z INF |  https://plain-river-quiet-hill.trycloudflare.com                                          |",
    "2026-10-21T10:00:01Z INF +--------------------------------------------------------------------------------------------+",
    "2026-10-21T10:00:02Z INF Registered tunnel connection connIndex=0 location=fra08 protocol=quic",
]
FAILED_TUNNEL_STDERR = [
    '2026-10-21T10:00:00Z ERR failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": '
    "dial tcp: lookup api.trycloudflare.com: no such host",
]
