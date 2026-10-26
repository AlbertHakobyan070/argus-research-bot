"""The console's JSON API (v4 spec §2.7): a thin layer over the run driver.

Every action calls ``bot/runs.py`` with ``actor="webapp"`` (the driver
takes the run's lock, claims it, attaches an untracked run resting at a
gate, and edits the Telegram gate message); its ``Outcome`` maps to HTTP:
``busy`` / ``already_running`` / ``refused`` → 409, ``not_found`` → 404,
anything else listed in ``_SUCCESS`` → 200 ``{outcome, message, run_id}``.
Views read the library and the driver (``tracked``, ``snapshot``,
``plan_payload``, the bus), never the graph per row. Report files are
served only through the run's ``report_paths``, contained in
``reports_root``. Every error is ``{"error": {"code", "message"}}``.
"""
from __future__ import annotations

import asyncio
import dataclasses
import functools
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import web
from pydantic import ValidationError
from telegram.error import TelegramError

from .. import __version__
from ..engines import registry
from ..engines.registry import EngineConfigError
from ..engines.schema import ProfileOverrides, TranscriberConfig
from ..graph.run_config import PRESETS, RunConfig
from ..graph.state import VALID_LENGTHS
from . import render
from .auth import AuthError, AuthUser, validate_init_data

logger = logging.getLogger("argus.webapp")


@dataclass
class Deps:
    """What the API runs on: the driver (``bot/runs.py``, or a fake), the
    library, the bot (Send to chat), the PTB ``bot_data`` (console health)
    and the auth settings."""
    runs: Any
    library: Any
    bot: Any
    bot_data: dict
    bot_token: str
    allowed_user_id: int
    max_age_s: float
    reports_root: Path
    settings: Any = None             # config.Settings: the vault roots the media routes write to and serve from


DEPS = web.AppKey("deps", Deps)
USER = web.RequestKey("user", AuthUser)

_HTTP = {"busy": 409, "already_running": 409, "refused": 409, "not_found": 404}
_SUCCESS = frozenset({"started", "updated", "attached", "delivered", "draining", "cancelled"})
_AWAITING = {"awaiting_plan": "plan", "awaiting_report": "report"}
_ROW_KEYS = ("run_id", "thread_id", "topic", "length", "depth", "archetype", "status", "origin", "created_at",
             "updated_at", "report_dir", "exa_usd")
_FILES = ("md", "pdf", "evidence")                    # report_paths keys the console may read or send
_ENGINES_KEYS = frozenset({"active_profile", "overrides", "transcriber"})
_OVERRIDE_KEYS = frozenset({"tiers", "roles"})
_AUTH_MESSAGES = {
    "missing_field": "missing or incomplete Telegram initData — open the console from the Argus menu button",
    "hash_mismatch": "initData signature mismatch — reopen the console from the Argus menu button",
    "expired": "session expired — reopen from the menu button",
    "wrong_user": "this console belongs to another Telegram user",
}
_FIELD_NAME = re.compile(r"[A-Za-z0-9_]{1,32}")
_MAX_LOGGED_FIELDS = 10
_dumps = functools.partial(json.dumps, default=str, ensure_ascii=False)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _error(status: int, code: str, message: str) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message}}, status=status, dumps=_dumps)


def _ok(data: Any) -> web.Response:
    return web.json_response(data, dumps=_dumps)


def _field_names(fields: tuple[str, ...]) -> str:
    """The initData field names for a log line: only plain names (a key is
    attacker-chosen, so no newline or other text can forge a line), at
    most ``_MAX_LOGGED_FIELDS``, the rest counted."""
    plain = [f for f in fields if _FIELD_NAME.fullmatch(f)][:_MAX_LOGGED_FIELDS]
    other = len(fields) - len(plain)
    return (", ".join(plain) + (f" +{other} other" if other else "")).strip() or "none"


def _bad(message: str) -> ApiError:
    return ApiError(400, "bad_request", message)


# ---------------------------------------------------------------------------
# middlewares
# ---------------------------------------------------------------------------

@web.middleware
async def errors(request: web.Request, handler) -> web.StreamResponse:
    """Every failure under ``/api`` as the error JSON: ours, an engines
    config refusal (400 with its text), a bad body, aiohttp's own (404,
    405), and anything else (500, logged with its traceback)."""
    try:
        return await handler(request)
    except ApiError as e:
        return _error(e.status, e.code, e.message)
    except EngineConfigError as e:
        return _error(400, "engine_config", str(e))
    except ValidationError as e:
        return _error(400, "bad_request", str(e))
    except web.HTTPException as e:
        if not request.path.startswith("/api") or e.status < 400:
            raise
        return _error(e.status, e.reason.lower().replace(" ", "_"), e.reason)
    except Exception:
        logger.exception("webapp: %s %r failed", request.method, request.path)
        return _error(500, "internal", "internal error — see the bot log")


@web.middleware
async def auth(request: web.Request, handler) -> web.StreamResponse:
    """``/api/*`` needs ``Authorization: tma <initData>``. A refusal logs the
    reason code and the names of the fields present — never a value, the
    header or the token. The source IP is never trusted (tunnel traffic
    arrives from loopback)."""
    if not request.path.startswith("/api"):
        return await handler(request)
    d = request.app[DEPS]
    scheme, _, init_data = request.headers.get("Authorization", "").partition(" ")
    try:
        if scheme != "tma" or not init_data:
            raise AuthError("missing_field")
        user = validate_init_data(init_data, d.bot_token, d.allowed_user_id, d.max_age_s)
    except AuthError as e:
        logger.warning("webapp: %s %r refused: %s (fields present: %s)", request.method, request.path, e.code,
                       _field_names(e.fields))
        return _error(e.status, e.code, _AUTH_MESSAGES[e.code])
    request[USER] = user
    return await handler(request)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _deps(request: web.Request) -> Deps:
    return request.app[DEPS]


async def _body(request: web.Request) -> dict:
    if not request.can_read_body:
        return {}
    try:
        body = await request.json()
    except ValueError as e:
        raise _bad(f"the body is not JSON ({e})") from None
    if not isinstance(body, dict):
        raise _bad("the body must be a JSON object")
    return body


def _opt(body: dict, key: str, kind: type = str):
    value = body.get(key)
    if value is not None and not isinstance(value, kind):
        raise _bad(f"{key} must be a {kind.__name__}")
    return value


def _int_query(request: web.Request, key: str, default: int, lo: int, hi: int) -> int:
    raw = request.query.get(key)
    try:
        value = default if raw is None else int(raw)
    except ValueError:
        raise _bad(f"{key} must be an integer") from None
    if not lo <= value <= hi:
        raise _bad(f"{key} must be between {lo} and {hi}")
    return value


def _outcome(out) -> web.Response:
    """A driver ``Outcome`` as HTTP (module docstring)."""
    if out.code in _HTTP:
        return _error(_HTTP[out.code], out.code, out.text)
    if out.code not in _SUCCESS:
        raise RuntimeError(f"run driver returned an unknown outcome {out.code!r}")
    return _ok({"outcome": out.code, "message": out.text, "run_id": out.run_id})


async def _row(d: Deps, run_id: str) -> dict:
    row = await d.library.get_run(run_id)
    if row is None:
        raise ApiError(404, "not_found", f"no run {run_id}")
    return row


def _run_json(row: dict, *, in_flight: bool, stage: str | None, awaiting: str | None) -> dict:
    return {**{k: row.get(k) for k in _ROW_KEYS}, "stage": stage or None, "awaiting": awaiting,
            "in_flight": in_flight}


def _checked_path(raw: str, root: Path) -> tuple[Path, bool, bool]:
    """``raw`` resolved (symlinks and ``..`` included), whether it lies in
    ``root``, whether it is a file (filesystem calls: run in a thread)."""
    path = Path(raw).resolve()
    return path, path.is_relative_to(root.resolve()), path.is_file()


async def _report_file(d: Deps, run_id: str, key: str) -> Path:
    """The run's registered ``report_paths[key]``, if it lies in reports_root and exists."""
    snap = await d.runs.snapshot(run_id)
    if snap is None:
        raise ApiError(404, "not_found", f"no run {run_id}")
    raw = (snap["values"].get("report_paths") or {}).get(key)
    if not raw:
        raise ApiError(404, "not_found", f"run {run_id} has no {key} file yet")
    path, inside, exists = await asyncio.to_thread(_checked_path, raw, d.reports_root)
    if not inside:
        logger.warning("webapp: run %r: its %s path is outside reports_root; refused", run_id, key)
        raise ApiError(404, "not_found", f"the {key} file of run {run_id} is outside reports_root")
    if not exists:
        raise ApiError(404, "not_found", f"the {key} file of run {run_id} is missing on disk ({path.name})")
    return path


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

async def me(request: web.Request) -> web.Response:
    d, user = _deps(request), request[USER]
    try:
        profile = registry.current_config().active_profile
    except EngineConfigError:
        profile = None                               # no engines config loaded: GET /api/engines says why
    return _ok({"user": {"id": user.id, "first_name": user.first_name},
                "bot": {"allowed_user_id": d.allowed_user_id, "active_profile": profile,
                        "public_url": d.bot_data.get("webapp_public_url") or None,
                        "console_ok": d.bot_data.get("webapp_ok") is True, "version": __version__}})


async def list_runs(request: web.Request) -> web.Response:
    """The library rows plus the in-flight registry: ``in_flight`` is a live
    drive, ``awaiting`` the library's gate status while none runs."""
    d = _deps(request)
    limit = _int_query(request, "limit", 20, 1, 200)
    tracked = d.runs.tracked()
    out = []
    for row in await d.library.list_runs(limit=limit):
        h = tracked.get(row["run_id"])
        live = bool(h is not None and h.live)
        out.append(_run_json(row, in_flight=live, stage=h.stage if h is not None else None,
                             awaiting=None if live else _AWAITING.get(row.get("status"))))
    return _ok({"runs": out})


async def get_run(request: web.Request) -> web.Response:
    d, run_id = _deps(request), request.match_info["id"]
    row = await _row(d, run_id)
    snap = await d.runs.snapshot(run_id)
    if snap is None:
        raise ApiError(404, "not_found", f"no run {run_id}")
    v = snap["values"]
    return _ok({
        "run": _run_json(row, in_flight=snap["in_flight"], stage=snap["stage"], awaiting=snap["awaiting"]),
        "snapshot": {
            "next": snap["next"],
            "counts": {k: len(v.get(k) or []) for k in ("sources", "fetched", "evidence", "findings")},
            "coverage": v.get("coverage") or {},
            "run_config": v.get("run_config"),
            "engines_summary": registry.summarize_calls(v.get("model_calls") or [], v.get("errors") or [],
                                                        exa_usd=row.get("exa_usd")),
            "panel_verdict": v.get("panel_verdict"),
            "report_paths": v.get("report_paths") or {},
            "plan": await d.runs.plan_payload(run_id),    # the plan card's data: same knobs, same Exa estimate
            "compose_runs": v.get("compose_runs") or 0,
            "gap_waves": v.get("gap_waves") or 0,
        },
        "pending_sources": await d.library.pending_sources(run_id),   # queued by /append or the media library
        "last_seq": snap["last_seq"],
    })


async def events(request: web.Request) -> web.Response:
    d, run_id = _deps(request), request.match_info["id"]
    after = _int_query(request, "after", 0, 0, 2**62)
    await _row(d, run_id)
    return _ok({"events": [dataclasses.asdict(e) for e in d.runs.bus.since(run_id, after)],
                "last_seq": d.runs.bus.last_seq(run_id)})


async def report(request: web.Request) -> web.Response:
    d, run_id = _deps(request), request.match_info["id"]
    if request.query.get("format", "html") != "html":
        raise _bad("format must be html")
    await _row(d, run_id)
    path = await _report_file(d, run_id, "md")
    text = await asyncio.to_thread(path.read_text, encoding="utf-8")
    html = await asyncio.to_thread(render.render_report, text)
    return web.Response(text=html, content_type="text/html", charset="utf-8")


async def evidence(request: web.Request) -> web.Response:
    d, run_id = _deps(request), request.match_info["id"]
    await _row(d, run_id)
    path = await _report_file(d, run_id, "evidence")
    return web.Response(body=await asyncio.to_thread(path.read_bytes), content_type="application/json")


async def send(request: web.Request) -> web.Response:
    """Send to chat: the bot sends the chosen report files to the run's chat
    (downloads cannot carry the Authorization header). Every file is checked
    before the first is sent; nothing about the run changes."""
    from ..bot.common import _send_md_doc            # lazy: bot.app imports the webapp
    d, run_id = _deps(request), request.match_info["id"]
    what = (await _body(request)).get("what")
    if not isinstance(what, list) or not what or any(w not in _FILES for w in what):
        raise _bad(f"what must be a non-empty list of {', '.join(_FILES)}")
    row = await _row(d, run_id)
    paths = [await _report_file(d, run_id, key) for key in dict.fromkeys(what)]
    sent: list[str] = []
    try:
        for path in paths:
            await _send_md_doc(d.bot, row["chat_id"], path, caption=f"Argus report {run_id} — {path.name}")
            sent.append(path.name)
    except TelegramError as e:
        logger.warning("webapp: run %r: send to chat failed after %d file(s): %s", run_id, len(sent), e)
        raise ApiError(502, "send_failed", f"⚠️ send failed after {len(sent)} file(s): {e}") from None
    return _ok({"outcome": "sent", "message": f"Sent {', '.join(sent)} to the chat.", "sent": sent})


async def start_run(request: web.Request) -> web.Response:
    d, body = _deps(request), await _body(request)
    topic = body.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        raise _bad("topic must be a non-empty string")
    knobs = {k: _opt(body, k) for k in ("length", "depth", "engines_profile")}
    overrides = dict(_opt(body, "overrides", dict) or {})
    recency = _opt(body, "recency")
    if recency is not None:
        overrides["recency"] = recency                   # an overridable root of RunConfig
    out = await d.runs.start_run(chat_id=d.allowed_user_id, user_id=request[USER].id, topic=topic.strip(),
                                 **{k: v for k, v in knobs.items() if v is not None}, overrides=overrides,
                                 origin="webapp")
    return _outcome(out)


async def plan_action(request: web.Request) -> web.Response:
    d, run_id, body = _deps(request), request.match_info["id"], await _body(request)
    action = body.get("action")
    if action == "approve":
        knobs = {k: _opt(body, k) for k in ("length", "depth", "engines_profile")}
        return _outcome(await d.runs.approve_plan(run_id, actor="webapp", **knobs))
    if action == "edit":
        feedback = _opt(body, "feedback")
        if feedback is None:
            raise _bad("edit needs feedback")
        return _outcome(await d.runs.edit_plan(run_id, feedback, actor="webapp"))
    if action == "cancel":
        return _outcome(await d.runs.cancel_run(run_id, actor="webapp"))
    raise _bad("action must be approve, edit or cancel")


async def report_action(request: web.Request) -> web.Response:
    d, run_id, body = _deps(request), request.match_info["id"], await _body(request)
    action = body.get("action")
    if action not in ("send", "extend", "revise", "cancel"):
        raise _bad("action must be send, extend, revise or cancel")
    feedback = _opt(body, "feedback")
    return _outcome(await d.runs.report_action(run_id, action, feedback=feedback, actor="webapp"))


async def cancel(request: web.Request) -> web.Response:
    d, run_id = _deps(request), request.match_info["id"]
    return _outcome(await d.runs.cancel_run(run_id, actor="webapp"))


async def continue_run(request: web.Request) -> web.Response:
    """Continue, as ``/continue``: a finished run is extended with its
    pending appended sources (append-only, no new searches) or a fresh
    search pass; a stopped one resumes; one at a gate is re-attached. The
    next report gate marks each append, naming failures in the chat."""
    from ..bot.common import _safe_send                     # lazy: bot.app imports the webapp
    from ..bot.history import appends_marker, pending_appends
    d, run_id = _deps(request), request.match_info["id"]
    row = await _row(d, run_id)
    graph = d.bot_data.get("graph")
    if graph is None:
        raise ApiError(409, "refused", "the bot is not fully initialized (no graph) — restart it")
    pending, appended = await pending_appends(d.library, run_id)

    async def reply(text: str) -> None:
        await _safe_send(d.bot, row["chat_id"], text, parse_mode=None)

    on_report = appends_marker(d.library, graph, {"configurable": {"thread_id": row["thread_id"]}}, run_id,
                               pending, appended, reply)
    return _outcome(await d.runs.continue_run(row, chat_id=row["chat_id"], appended=appended, on_report=on_report))


async def _engines_body(d: Deps) -> dict:
    cfg = registry.current_config()
    config = cfg.model_dump(mode="json")
    for prov in config["providers"].values():
        env = prov.pop("api_key_env")
        prov["api_key"] = {"env": env, "present": bool(os.environ.get(env)) if env else None}
    try:
        snap = (await registry.active_snapshot()).model_dump(mode="json")
        validation = {"ok": True, "errors": []}
    except EngineConfigError as e:
        snap, validation = None, {"ok": False, "errors": [str(e)]}
    mtd = await d.library.exa_usd_month_to_date() if d.library is not None else None
    return {"config": config, "active_profile": cfg.active_profile, "snapshot": snap, "validation": validation,
            "exa_month_to_date_usd": mtd}


async def get_engines(request: web.Request) -> web.Response:
    return _ok(await _engines_body(_deps(request)))


async def put_engines(request: web.Request) -> web.Response:
    """Per-profile tier/role overrides, the transcriber, the active profile —
    through ``registry.set_*`` (validated before it persists). One change
    per request (one part, overrides for one profile), so a request is
    applied whole or not at all. Provider definitions are file-only: any
    other key is refused before anything changes."""
    d, body = _deps(request), await _body(request)
    extra = sorted(set(body) - _ENGINES_KEYS)
    overrides = _opt(body, "overrides", dict) or {}
    for profile, ov in overrides.items():
        if not isinstance(ov, dict):
            raise _bad(f"overrides.{profile} must be an object")
        extra += [f"overrides.{profile}.{k}" for k in sorted(set(ov) - _OVERRIDE_KEYS)]
    if extra:
        raise _bad(f"not editable here: {', '.join(extra)} (provider definitions and pricing are file-only; "
                   "edit argus_engines.json)")
    parts = sorted(k for k in _ENGINES_KEYS if k in body)
    many_profiles = "overrides" in body and len(overrides) != 1
    if len(parts) != 1 or many_profiles:
        got = ", ".join(parts) or "nothing"
        if many_profiles:
            got += f" (overrides for {len(overrides)} profiles)"
        raise _bad("change one thing per request: overrides for one profile, the transcriber, or the "
                   f"active profile (got {got})")
    if "overrides" in body:
        [(profile, ov)] = overrides.items()
        await registry.set_overrides(profile, ProfileOverrides.model_validate(ov))
    elif "transcriber" in body:
        await registry.set_transcriber(TranscriberConfig.model_validate(body["transcriber"]))
    else:
        active = _opt(body, "active_profile")
        if active is None:
            raise _bad("active_profile must be a string")
        await registry.set_active_profile(active)
    return _ok(await _engines_body(d))


async def provider_models(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    return _ok({"provider": name, "models": await asyncio.to_thread(registry.live_models, name)})


async def presets(request: web.Request) -> web.Response:
    exa = registry.current_config().pricing.exa
    return _ok({"presets": PRESETS, "default_depth": {n: RunConfig.default_depth(n) for n in VALID_LENGTHS},
                "exa_prices": exa.model_dump(mode="json") if exa is not None else None})


def setup(app: web.Application) -> None:
    r = app.router
    r.add_get("/api/me", me)
    r.add_get("/api/runs", list_runs)
    r.add_post("/api/runs", start_run)
    r.add_get("/api/runs/{id}", get_run)
    r.add_get("/api/runs/{id}/events", events)
    r.add_get("/api/runs/{id}/report", report)
    r.add_get("/api/runs/{id}/evidence", evidence)
    r.add_post("/api/runs/{id}/send", send)
    r.add_post("/api/runs/{id}/plan", plan_action)
    r.add_post("/api/runs/{id}/report", report_action)
    r.add_post("/api/runs/{id}/cancel", cancel)
    r.add_post("/api/runs/{id}/continue", continue_run)
    r.add_get("/api/engines", get_engines)
    r.add_put("/api/engines", put_engines)
    r.add_get("/api/engines/providers/{name}/models", provider_models)
    r.add_get("/api/config/presets", presets)
