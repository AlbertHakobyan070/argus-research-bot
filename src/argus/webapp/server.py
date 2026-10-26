"""The console server and its lifecycle (v4 spec §2.7).

``start`` (a POST_INIT hook) serves the console on 127.0.0.1 and brings up
its public URL: ``ARGUS_WEBAPP_PUBLIC_URL`` if set, else a supervised
cloudflared quick tunnel (``ARGUS_WEBAPP_TUNNEL=quick``, the default), or
none (``none``: local only). Each new public URL is set as the chat's
``Argus`` menu button. The console is optional: any failure is logged,
DM'd to the allowed user and shown in ``/status`` (``bot_data``
``webapp_ok`` False, ``webapp_error``), and never raised into the bot.
``stop`` (a POST_SHUTDOWN hook, after the drive drain) is idempotent.

``bot_data`` keys (the contract with ``/status`` and ``runs.console_url``):
``webapp_public_url`` (None while there is none), ``webapp_ok``,
``webapp_error`` (None while ok).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from aiohttp import web
from telegram import MenuButtonWebApp, WebAppInfo
from telegram.error import TelegramError

from ..config import get_settings
from . import api, media_api, tunnel

logger = logging.getLogger("argus.webapp")

HOST = "127.0.0.1"
STATIC_DIR = Path(__file__).resolve().parent / "static"
CSP = ("default-src 'self'; script-src 'self' https://telegram.org; style-src 'self'; img-src 'self' data:; "
       "font-src 'self'; connect-src 'self'; "
       "frame-ancestors 'self' https://web.telegram.org https://*.telegram.org")
HEADERS = {"Content-Security-Policy": CSP, "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"}
DEFAULT_MAX_AGE_S = 3600
_KEY = "webapp_console"                  # bot_data: the running _Console


async def _security_headers(request: web.Request, response: web.StreamResponse) -> None:
    response.headers.update(HEADERS)
    # Revalidate (a cheap 304) so a webview never runs a stale app.js after an update; a
    # handler's own Cache-Control (the signed media files) stands.
    response.headers.setdefault("Cache-Control", "no-cache")


def build_app(deps: api.Deps, *, static_dir: Path = STATIC_DIR) -> web.Application:
    """The console app: ``GET /`` (``static/index.html``), ``/static/*`` (no
    listing, no symlinks), the API; the security headers on every response."""
    app = web.Application(middlewares=[api.errors, api.auth])
    app[api.DEPS] = deps
    app.on_response_prepare.append(_security_headers)
    index = static_dir / "index.html"

    async def home(request: web.Request) -> web.StreamResponse:
        if not index.is_file():
            raise web.HTTPNotFound(text="the console frontend is not built (webapp/static/index.html)")
        return web.FileResponse(index)

    app.router.add_get("/", home)
    if static_dir.is_dir():
        app.router.add_static("/static", static_dir, show_index=False, follow_symlinks=False)
    api.setup(app)
    media_api.setup(app)
    return app


class _Console:
    """One start: the runner, the tunnel supervisor, and the bot_data health keys."""

    def __init__(self, app: Any, settings) -> None:
        self.app, self.settings = app, settings
        self.bot_data: dict = app.bot_data
        self.runner: web.AppRunner | None = None
        self.supervisor: tunnel.Supervisor | None = None
        self.notified = False             # a "Console down" DM went out since the last good URL

    def _health(self, url: str | None, ok: bool, error: str | None) -> None:
        self.bot_data.update(webapp_public_url=url, webapp_ok=ok, webapp_error=error)

    async def _dm(self, text: str) -> None:
        from ..bot.common import _safe_send          # lazy: bot.app imports this module
        await _safe_send(self.app.bot, self.settings.telegram_allowed_user_id, text, parse_mode=None)

    async def failed(self, reason: str, *, url: str | None = None) -> None:
        """Logged, shown in /status, DM'd (once per run of failures)."""
        logger.error("webapp: console down — %s", reason)
        self._health(url, False, reason)
        if not self.notified:
            self.notified = True
            await self._dm(f"⚠️ Console down — {reason}")

    async def listen(self, port: int) -> None:
        deps = api.Deps(runs=_runs(), library=self.bot_data.get("library"), bot=self.app.bot, bot_data=self.bot_data,
                        bot_token=self.settings.telegram_bot_token,
                        allowed_user_id=self.settings.telegram_allowed_user_id,
                        max_age_s=float(os.environ.get("ARGUS_WEBAPP_INITDATA_MAX_AGE_S") or DEFAULT_MAX_AGE_S),
                        reports_root=Path(self.settings.reports_root), settings=self.settings)
        self.runner = web.AppRunner(build_app(deps), access_log=None)   # no request lines in the log
        await self.runner.setup()
        await web.TCPSite(self.runner, HOST, port).start()
        logger.info("webapp: listening on %s:%s", HOST, port)

    async def public_url(self, url: str) -> None:
        """A new public URL: health ok and the menu button re-registered; the
        first good URL after a "Console down" DM sends the "back up" DM."""
        was_down, self.notified = self.notified, False
        self._health(url, True, None)
        try:
            await self.app.bot.set_chat_menu_button(chat_id=self.settings.telegram_allowed_user_id,
                                                    menu_button=MenuButtonWebApp("Argus", WebAppInfo(url)))
        except Exception as e:             # TelegramError, or anything else: the console must not take the bot down
            if not isinstance(e, TelegramError):
                logger.exception("webapp: set_chat_menu_button failed")
            await self.failed(f"menu button not registered: {e}", url=url)
            return
        logger.info("menu button registered for chat %s", self.settings.telegram_allowed_user_id)
        if was_down:
            await self._dm(f"🌐 Console back up: {url}")

    async def tunnel_url(self, url: str) -> None:
        logger.info("tunnel: %s", url)
        await self.public_url(url)

    async def tunnel_down(self, reason: str) -> None:
        await self.failed(f"tunnel: {reason}")

    async def close(self) -> None:
        """The tunnel, then the server; each is closed even if the other fails."""
        first: Exception | None = None
        for close in (self.supervisor.stop if self.supervisor else None,
                      self.runner.cleanup if self.runner else None):
            if close is None:
                continue
            try:
                await close()
            except Exception as e:
                logger.exception("webapp: closing failed")
                first = first or e
        if first is not None:
            raise first


def _runs():
    from ..bot import runs                # lazy: bot.app imports this module
    return runs


async def start(app: Any) -> None:
    """POST_INIT hook: serve the console and bring up its public URL. Never raises."""
    s = get_settings()
    if not s.webapp_enabled:
        logger.info("webapp: disabled (ARGUS_WEBAPP_ENABLED=false)")
        return
    console = app.bot_data[_KEY] = _Console(app, s)
    mode = (os.environ.get("ARGUS_WEBAPP_TUNNEL") or "quick").strip().lower()
    if mode not in ("quick", "none"):
        await console.failed(f"ARGUS_WEBAPP_TUNNEL={mode!r} is not quick or none; the console was not started")
        return
    try:
        await console.listen(s.webapp_port)
    except Exception as e:
        logger.exception("webapp: the console server did not start")
        runner, console.runner = console.runner, None
        if runner is not None:
            try:
                await runner.cleanup()
            except Exception:
                logger.exception("webapp: cleaning up after the failed start failed")
        await console.failed(f"server: {e}")
        return
    public = (os.environ.get("ARGUS_WEBAPP_PUBLIC_URL") or "").strip()
    if public:
        await console.public_url(public)
    elif mode == "none":
        console._health(f"http://{HOST}:{s.webapp_port}", True, None)     # local only: no menu button
    else:
        cloudflared = tunnel.binary()
        if not cloudflared.is_file():
            await console.failed(f"tunnel: cloudflared not found at {cloudflared} (set ARGUS_CLOUDFLARED)")
            return
        console._health(None, False, "tunnel starting — waiting for the cloudflared URL")
        console.supervisor = tunnel.Supervisor(tunnel.command(cloudflared, s.webapp_port),
                                               on_url=console.tunnel_url, on_down=console.tunnel_down)
        console.supervisor.start()


async def stop(app: Any) -> None:
    """POST_SHUTDOWN hook: the tunnel and the server closed; idempotent."""
    console = app.bot_data.pop(_KEY, None)
    if console is not None:
        await console.close()
