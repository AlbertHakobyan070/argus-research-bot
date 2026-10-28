"""Backend updates (console Settings → Backend): the packages that break when
sites change, their installed and latest (PyPI) versions, and an upgrade into
the bot's own venv.

The components are a fixed table — no package name or argument from a request
reaches the command line. An upgrade runs ``uv pip install`` with
``--upgrade-package`` for the component's own packages only, after a dry run:
when the plan would change any other package it is refused (on Windows a
package whose native module the running bot has loaded cannot be replaced, and
uv leaves it half-removed), with the command to run while the bot is stopped.

yt-dlp runs as a subprocess per download (``media.py``), so its update applies
to the next download; ddgs is imported by the bot and needs a restart, shown
as ``restart_pending`` (the version on disk differs from the one loaded at
start).
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import re
import shutil
import sys
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Component:
    spec: str                    # the uv requirement installed
    packages: tuple[str, ...]    # upgraded together (the first is the one shown as "latest")
    restart: bool                # loaded by the bot: a restart applies the update
    note: str


COMPONENTS: dict[str, Component] = {
    "yt-dlp": Component("yt-dlp[default]", ("yt-dlp", "yt-dlp-ejs"), False,
                        "Media downloads; yt-dlp-ejs solves YouTube's JS challenge. Runs per download, "
                        "so an update applies to the next one."),
    "ddgs": Component("ddgs", ("ddgs",), True,
                      "Web search (DuckDuckGo and others). Loaded by the bot: restart it to apply an update."),
}

_PYPI = "https://pypi.org/pypi/{}/json"
_CHANGE = re.compile(r"^ ([+-]) ([A-Za-z0-9._-]+)==(\S+)$", re.MULTILINE)
_TIMEOUT_S = 300
_lock = asyncio.Lock()


class UpdateError(Exception):
    """An upgrade that did not happen: ``code`` is ``busy``, ``refused`` or ``failed``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def installed(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


_LOADED = {p: installed(p) for c in COMPONENTS.values() if c.restart for p in c.packages}   # at bot start


async def latest(client: httpx.AsyncClient, package: str) -> str:
    r = await client.get(_PYPI.format(package))
    r.raise_for_status()
    return r.json()["info"]["version"]


async def status() -> list[dict]:
    """Every component: its packages' installed versions, the latest release
    of its first package (or the error that kept it from being read), and
    whether a restart is pending."""
    async with httpx.AsyncClient(timeout=10) as client:
        found = await asyncio.gather(*(latest(client, c.packages[0]) for c in COMPONENTS.values()),
                                     return_exceptions=True)
    out = []
    for (name, c), new in zip(COMPONENTS.items(), found):
        versions = {p: installed(p) for p in c.packages}
        out.append({
            "name": name, "note": c.note, "restart": c.restart,
            "packages": [{"name": p, "installed": v} for p, v in versions.items()],
            "latest": None if isinstance(new, BaseException) else new,
            "latest_error": f"{type(new).__name__}: {new}" if isinstance(new, BaseException) else None,
            "restart_pending": c.restart and any(versions[p] != _LOADED.get(p) for p in c.packages),
        })
    return out


def _command(c: Component, *, dry_run: bool) -> list[str]:
    uv = shutil.which("uv")
    if uv is None:
        raise UpdateError("failed", "uv is not on the bot's PATH — install uv, or update by hand")
    if sys.prefix == sys.base_prefix:
        raise UpdateError("failed", "the bot is not running from a virtual environment; refusing to "
                                    f"install into {sys.executable}")
    cmd = [uv, "pip", "install", "--python", sys.executable, "--link-mode=copy"]
    for p in c.packages:
        cmd += ["--upgrade-package", p]
    return [*cmd, *(["--dry-run"] if dry_run else []), c.spec]


async def _run(cmd: list[str]) -> str:
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), _TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise UpdateError("failed", f"uv did not finish in {_TIMEOUT_S} s") from None
    text = out.decode("utf-8", "replace")
    if proc.returncode != 0:
        raise UpdateError("failed", f"uv exited with code {proc.returncode}: {text.strip()[-1500:]}")
    return text


def _norm(package: str) -> str:
    return re.sub(r"[-_.]+", "-", package).lower()


def _changes(text: str) -> list[dict]:
    return [{"op": "remove" if op == "-" else "install", "package": name, "version": v}
            for op, name, v in _CHANGE.findall(text)]


async def upgrade(name: str) -> dict:
    """Upgrade one component (``name`` a key of ``COMPONENTS``): ``{name,
    changes, restart_needed, output}``; ``UpdateError`` when another upgrade
    runs, the plan touches other packages, or uv fails."""
    c = COMPONENTS[name]
    if _lock.locked():
        raise UpdateError("busy", "another update is running")
    async with _lock:
        plan = _changes(await _run(_command(c, dry_run=True)))
        own = {_norm(p) for p in c.packages}
        other = sorted({ch["package"] for ch in plan if _norm(ch["package"]) not in own})
        if other:
            manual = " ".join(_command(c, dry_run=False)[1:])
            raise UpdateError("refused", f"this update would also change {', '.join(other)}, which the running "
                                         f"bot may have loaded; stop the bot and run: uv {manual}")
        if not plan:
            return {"name": name, "changes": [], "restart_needed": False, "output": "already up to date"}
        out = await _run(_command(c, dry_run=False))
        return {"name": name, "changes": _changes(out), "restart_needed": c.restart, "output": out.strip()[-1500:]}
