"""The cloudflared quick tunnel and its supervisor (v4 spec §2.7).

cloudflared prints its ``https://<name>.trycloudflare.com`` URL on stderr.
A tail task copies every stderr line to ``logs/cloudflared.log``, takes the
first tunnel URL (within ``URL_TIMEOUT_S``) and keeps draining the pipe.
When the process exits (or prints no URL in time) the supervisor reports
it and starts a new one after an exponential backoff, capped at 5 min and
reset once a process ran ``STABLE_S``. The hostname changes on every start:
``on_url`` gets each new one. The binary is never downloaded here.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..config import _PROJECT_ROOT as _ROOT

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_API_URL = "https://api.trycloudflare.com"     # cloudflared's own API, quoted in its failure lines
URL_TIMEOUT_S = 60.0
BACKOFF_START_S = 1.0
BACKOFF_MAX_S = 300.0
STABLE_S = 60.0                                # a process that ran this long resets the backoff
_TERMINATE_WAIT_S = 5.0
LOG_PATH = _ROOT / "logs" / "cloudflared.log"
_sleep = asyncio.sleep                         # the backoff clock (tests patch it)


def find_url(line: str) -> str | None:
    """The tunnel URL in one stderr line, if any."""
    return next((u for u in URL_RE.findall(line) if u != _API_URL), None)


def binary() -> Path:
    """``ARGUS_CLOUDFLARED``, else ``tools/bin/cloudflared(.exe)`` in the repo."""
    raw = os.environ.get("ARGUS_CLOUDFLARED")
    if raw:
        return Path(raw).expanduser()
    return _ROOT / "tools" / "bin" / ("cloudflared.exe" if os.name == "nt" else "cloudflared")


def command(cloudflared: Path, port: int) -> list[str]:
    return [str(cloudflared), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]


async def spawn(cmd: list[str]) -> asyncio.subprocess.Process:
    extra = {} if os.name == "nt" else {"start_new_session": True}
    return await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.DEVNULL,
                                                stderr=asyncio.subprocess.PIPE, **extra)


async def _terminate(proc) -> None:
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), _TERMINATE_WAIT_S)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()


class Supervisor:
    """Runs ``cmd`` until ``stop()``. ``on_url(url)`` on each new tunnel URL;
    ``on_down(reason)`` each time a process is gone, before its backoff
    (the reason ends "; restarting in <s> s"). Neither may raise."""

    def __init__(self, cmd: list[str], *, on_url: Callable[[str], Awaitable[None]],
                 on_down: Callable[[str], Awaitable[None]], spawn: Callable | None = None,
                 sleep: Callable[[float], Awaitable[None]] | None = None,
                 clock: Callable[[], float] = time.monotonic, log_path: Path | None = None,
                 url_timeout_s: float = URL_TIMEOUT_S) -> None:
        self.cmd, self.on_url, self.on_down = cmd, on_url, on_down
        self._spawn, self._sleep, self._clock = spawn, sleep, clock
        self.log_path = log_path
        self.url_timeout_s = url_timeout_s
        self.proc = None
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.task = asyncio.create_task(self._run(), name="cloudflared supervisor")

    async def stop(self) -> None:
        """Idempotent: the loop is cancelled and the running process terminated."""
        task, self.task = self.task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self.proc is not None:
            await _terminate(self.proc)

    async def _run(self) -> None:
        delay = BACKOFF_START_S
        while True:
            started = self._clock()
            reason = await self._once()
            if self._clock() - started >= STABLE_S:
                delay = BACKOFF_START_S
            await self.on_down(f"{reason}; restarting in {delay:g} s")
            await (self._sleep or _sleep)(delay)
            delay = min(delay * 2, BACKOFF_MAX_S)

    async def _once(self) -> str:
        """One process, from its start to its exit: why it is gone."""
        try:
            self.proc = proc = await (self._spawn or spawn)(self.cmd)
        except OSError as e:
            return f"cloudflared could not start ({e})"
        url: asyncio.Future = asyncio.get_running_loop().create_future()
        tail = asyncio.create_task(self._tail(proc.stderr, url), name="cloudflared stderr")
        exited = asyncio.ensure_future(proc.wait())
        try:
            await asyncio.wait({url, exited}, timeout=self.url_timeout_s, return_when=asyncio.FIRST_COMPLETED)
            if url.done():
                await self.on_url(url.result())
                return f"cloudflared exited with code {await exited}"
            if exited.done():
                return f"cloudflared exited with code {exited.result()} before printing its URL"
            await _terminate(proc)
            return f"cloudflared printed no URL within {self.url_timeout_s:g} s"
        finally:
            exited.cancel()
            if proc.returncode is not None:           # gone: the tail reads its last lines to EOF
                await asyncio.wait({tail}, timeout=_TERMINATE_WAIT_S)
            if not tail.done():
                tail.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tail

    async def _tail(self, stream: asyncio.StreamReader, url: asyncio.Future) -> None:
        path = self.log_path or LOG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as log:      # ponytail: sync appends of short lines on the loop; to_thread if it shows
            while line := await stream.readline():
                log.write(line)
                log.flush()
                if not url.done() and (found := find_url(line.decode("utf-8", "replace"))):
                    url.set_result(found)
