"""Argus media engine — download real platform media into the DS vault.

Phase 2 of the v2 rebuild. yt-dlp is a direct dependency of the argus
venv and is invoked as an *async* subprocess (``sys.executable -m
yt_dlp``) so downloads run in the background while the bot keeps
serving; progress is streamed via ``--progress-template`` lines.

Platforms: YouTube (watch/shorts/youtu.be), X/Twitter (status links),
Reddit (post / v.redd.it links), Instagram (reel/p links — usually
needs cookies; without them failures are surfaced honestly).

ffmpeg: resolved from ARGUS_FFMPEG → the imageio-ffmpeg bundled binary
→ PATH. Without ffmpeg yt-dlp cannot merge separate video+audio
streams, so quality falls back to single-file formats.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Literal

import httpx
from pydantic import BaseModel

logger = logging.getLogger("argus.media")

Platform = Literal["youtube", "x", "reddit", "instagram"]

# Command prefix for the downloader subprocess. Tests swap this for a
# stub script; production uses the venv's own yt-dlp module.
_YTDLP_PREFIX: list[str] = [sys.executable, "-m", "yt_dlp"]

DEFAULT_TIMEOUT_S = int(os.environ.get("ARGUS_MEDIA_TIMEOUT", "900"))
_CONCURRENCY = int(os.environ.get("ARGUS_MEDIA_CONCURRENCY", "2"))
_download_sem = asyncio.Semaphore(_CONCURRENCY)

_PROGRESS_TEMPLATE = "ARGUSP|%(progress._percent_str)s|%(progress._eta_str)s"
_PROGRESS_RE = re.compile(r"^ARGUSP\|\s*([\d.]+)%\|\s*(\S+)")

VALID_QUALITIES = ("auto", "min", "max")  # or a numeric pixel height


# ---------------------------------------------------------------------------
# URL routing
# ---------------------------------------------------------------------------

_PLATFORM_PATTERNS: list[tuple[Platform, re.Pattern]] = [
    ("youtube", re.compile(
        r"https?://(?:[\w-]+\.)?(?:youtube\.com/(?:watch\?|shorts/)|youtu\.be/)",
        re.I)),
    ("x", re.compile(
        r"https?://(?:[\w-]+\.)?(?:x|twitter)\.com/[^/\s]+/status/\d+", re.I)),
    ("reddit", re.compile(
        r"https?://(?:(?:[\w-]+\.)?reddit\.com/r/[^/\s]+/comments/"
        r"|redd\.it/|v\.redd\.it/)", re.I)),
    ("instagram", re.compile(
        r"https?://(?:[\w-]+\.)?instagram\.com/(?:reels?|p)/", re.I)),
]

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)


def detect_platform(url: str) -> Platform | None:
    """Map a URL onto a supported platform, or None."""
    for platform, pat in _PLATFORM_PATTERNS:
        if pat.search(url or ""):
            return platform
    return None


def extract_urls(text: str) -> list[str]:
    """Pull http(s) URLs out of free text — deduped, order-preserving,
    trailing punctuation stripped."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _URL_RE.finditer(text or ""):
        url = m.group(0).rstrip(".,;:!?")
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


# ---------------------------------------------------------------------------
# ffmpeg + quality
# ---------------------------------------------------------------------------


def resolve_ffmpeg() -> str | None:
    """ARGUS_FFMPEG env → imageio-ffmpeg bundled binary → PATH → None."""
    env = os.environ.get("ARGUS_FFMPEG")
    if env and Path(env).exists():
        return env
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def quality_format(quality: str, *, ffmpeg_available: bool) -> str:
    """Map the /quality setting onto a yt-dlp format string.

    ``auto`` caps at 1080p; ``min``/``max`` are the extremes; a bare
    number is a pixel-height cap. Without ffmpeg, merged (video+audio)
    formats are impossible, so single-file variants are used.
    """
    q = (quality or "auto").strip().lower()
    if q == "auto":
        if ffmpeg_available:
            return "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b"
        return "b[height<=1080][ext=mp4]/b[ext=mp4]/b"
    if q == "max":
        return "bv*+ba/b" if ffmpeg_available else "b[ext=mp4]/b"
    if q == "min":
        return "wv*+wa/w" if ffmpeg_available else "w[ext=mp4]/w"
    if q.isdigit():
        h = int(q)
        if ffmpeg_available:
            return f"bv*[height<={h}]+ba/b[height<={h}]/b"
        return f"b[height<={h}][ext=mp4]/b[height<={h}]/b"
    raise ValueError(
        f"quality must be auto|min|max|<pixel height>, got {quality!r}")


def parse_progress_line(line: str) -> tuple[float, str] | None:
    m = _PROGRESS_RE.match((line or "").strip())
    if not m:
        return None
    try:
        return float(m.group(1)), m.group(2)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


class MediaResult(BaseModel):
    ok: bool
    url: str = ""
    platform: str = ""
    path: str = ""
    info_json_path: str = ""
    title: str = ""
    media_id: str = ""
    size_bytes: int = 0
    duration_s: float | None = None
    error: str | None = None
    elapsed_s: float = 0.0


def _build_argv(url: str, platform: str, dest_root: Path, fmt: str,
                marker: Path, ffmpeg: str | None) -> list[str]:
    argv = [
        url,
        "-P", str(dest_root / platform),
        "-o", "%(upload_date>%Y%m%d|na)s_%(id)s_%(title).60B.%(ext)s",
        "--restrict-filenames",
        "--no-playlist",
        "--write-info-json",
        "--no-write-comments",
        "--newline",
        "--progress-template", _PROGRESS_TEMPLATE,
        "--print-to-file", "after_move:filepath", str(marker),
        "-f", fmt,
    ]
    if ffmpeg:
        argv += ["--ffmpeg-location", ffmpeg]
    cookies_file = os.environ.get("ARGUS_YTDLP_COOKIES")
    cookies_browser = os.environ.get("ARGUS_YTDLP_COOKIES_BROWSER")
    if cookies_file:
        argv += ["--cookies", cookies_file]
    elif cookies_browser:
        argv += ["--cookies-from-browser", cookies_browser]
    return argv


async def _run_streaming(cmd: list[str], *, timeout_s: int,
                         on_line: Callable[[str], None]) -> tuple[int, str]:
    """Run cmd, streaming stdout lines to ``on_line``. Returns
    (returncode, stderr_tail). Falls back to a thread-blocking run when
    the running loop can't spawn subprocesses (non-Proactor Windows
    loop) — progress lines are then delivered only at the end."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env)
    except NotImplementedError:
        logger.warning("event loop lacks subprocess support; running "
                       "downloader in a thread (no live progress)")
        import subprocess

        def _blocking() -> tuple[int, str]:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=timeout_s, env=env)
            for line in (p.stdout or "").splitlines():
                on_line(line)
            return p.returncode, (p.stderr or "")[-2000:]

        return await asyncio.to_thread(_blocking)

    stderr_buf: list[bytes] = []

    async def _pump_stdout():
        assert proc.stdout is not None
        async for raw in proc.stdout:
            on_line(raw.decode("utf-8", "replace").rstrip())

    async def _pump_stderr():
        assert proc.stderr is not None
        data = await proc.stderr.read()
        stderr_buf.append(data)

    try:
        await asyncio.wait_for(
            asyncio.gather(_pump_stdout(), _pump_stderr(), proc.wait()),
            timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, f"timed out after {timeout_s}s"
    stderr = b"".join(stderr_buf).decode("utf-8", "replace")
    return proc.returncode or 0, stderr[-2000:]


async def download_media(
    url: str, *, dest_root: Path | str, quality: str = "auto",
    on_progress: Callable[[float, str], None] | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> MediaResult:
    """Download one media URL into ``dest_root/<platform>/``.

    Returns a fail-soft :class:`MediaResult` — ``ok=False`` carries the
    downloader's stderr tail so the bot can surface WHY it failed
    (age-gate, login wall, dead link) instead of a generic error.
    """
    t0 = time.monotonic()
    platform = detect_platform(url)
    if platform is None:
        return MediaResult(
            ok=False, url=url,
            error=("unsupported platform — expected a YouTube / X / "
                   "Reddit / Instagram media link"))

    dest_root = Path(dest_root)
    plat_dir = dest_root / platform
    plat_dir.mkdir(parents=True, exist_ok=True)
    marker = plat_dir / f".argus_marker_{os.getpid()}_{time.monotonic_ns()}"

    ffmpeg = resolve_ffmpeg()
    fmt = quality_format(quality, ffmpeg_available=bool(ffmpeg))
    argv = _build_argv(url, platform, dest_root, fmt, marker, ffmpeg)
    cmd = [*_YTDLP_PREFIX, *argv]

    def _on_line(line: str) -> None:
        parsed = parse_progress_line(line)
        if parsed and on_progress is not None:
            try:
                on_progress(*parsed)
            except Exception:
                logger.exception("on_progress callback failed")

    try:
        async with _download_sem:
            rc, stderr = await _run_streaming(
                cmd, timeout_s=timeout_s, on_line=_on_line)
    except Exception as e:
        logger.exception("downloader subprocess failed to run")
        marker.unlink(missing_ok=True)
        return MediaResult(ok=False, url=url, platform=platform,
                           error=f"downloader failed to run: {e}",
                           elapsed_s=time.monotonic() - t0)

    final_path: Path | None = None
    if marker.exists():
        text = marker.read_text(encoding="utf-8", errors="replace").strip()
        marker.unlink(missing_ok=True)
        if text:
            # yt-dlp may append multiple lines on retries; last wins.
            candidate = Path(text.splitlines()[-1].strip())
            if candidate.exists():
                final_path = candidate

    if rc != 0 or final_path is None:
        err = stderr.strip() or f"yt-dlp exited {rc} without an output file"
        return MediaResult(ok=False, url=url, platform=platform,
                           error=err[-500:],
                           elapsed_s=time.monotonic() - t0)

    title, media_id, duration = "", "", None
    info_path = final_path.with_suffix(".info.json")
    if not info_path.exists():
        # merged outputs can differ in extension from the info json stem
        hits = sorted(final_path.parent.glob(final_path.stem + "*.info.json"))
        info_path = hits[0] if hits else info_path
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8",
                                                  errors="replace"))
            title = info.get("title") or ""
            media_id = str(info.get("id") or "")
            duration = info.get("duration")
        except Exception:
            logger.warning("could not parse info.json %s", info_path)

    return MediaResult(
        ok=True, url=url, platform=platform, path=str(final_path),
        info_json_path=str(info_path) if info_path.exists() else "",
        title=title, media_id=media_id,
        size_bytes=final_path.stat().st_size,
        duration_s=float(duration) if duration is not None else None,
        elapsed_s=time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Reddit search (public search.json — no auth, real UA, fail-soft)
# ---------------------------------------------------------------------------

_REDDIT_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) argus-research-bot/"
              "0.1 (personal research assistant)")


_REDDIT_POST_RE = re.compile(
    r"https?://(?:[\w-]+\.)?reddit\.com(/r/([^/\s]+)/comments/[^\s?#]+)", re.I)


async def reddit_search(query: str, *, limit: int = 8,
                        timeout: float = 15.0) -> list[dict[str, Any]]:
    """Search Reddit for VIDEO posts. Returns the same result shape as
    ``tools.youtube_search`` (title/url/channel/duration/views) so the
    bot's pool + keyboards work identically.

    Primary backend: Reddit's public ``search.json`` (no auth). Reddit
    403-blocks that from many networks, so on ANY failure we fall back
    to DuckDuckGo restricted to reddit post URLs — less precise (no
    is_video flag; yt-dlp reports non-video posts honestly at download
    time), but it works everywhere. Fail-soft: [] when both fail.
    """
    payload: dict | None = None
    try:
        async with httpx.AsyncClient(
                headers={"User-Agent": _REDDIT_UA},
                timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(
                "https://www.reddit.com/search.json",
                params={"q": query, "limit": 25, "sort": "relevance",
                        "raw_json": 1, "type": "link"})
            resp.raise_for_status()
            payload = resp.json()
    except Exception as e:
        logger.warning("reddit search.json failed (%s); trying ddgs fallback", e)

    if payload is not None:
        out: list[dict[str, Any]] = []
        for child in (payload.get("data", {}).get("children") or []):
            d = child.get("data") or {}
            if not d.get("is_video"):
                continue
            permalink = d.get("permalink") or ""
            if not permalink:
                continue
            duration = None
            media = d.get("media") or {}
            rv = media.get("reddit_video") or {}
            if rv.get("duration") is not None:
                duration = rv["duration"]
            out.append({
                "title": d.get("title") or "(untitled)",
                "url": f"https://www.reddit.com{permalink}",
                "channel": f"r/{d.get('subreddit', '?')}",
                "duration": duration,
                "views": d.get("score"),
                "platform": "reddit",
            })
            if len(out) >= limit:
                break
        return out

    # ddgs fallback — surface reddit post links found by web search.
    from .tools import ddgs_search
    try:
        hits = await asyncio.to_thread(
            ddgs_search, f"site:reddit.com {query} video", max_results=25)
    except Exception as e:
        logger.warning("reddit ddgs fallback failed: %s", e)
        return []
    out = []
    seen: set[str] = set()
    for h in hits or []:
        m = _REDDIT_POST_RE.search(h.get("url") or "")
        if not m:
            continue
        url = f"https://www.reddit.com{m.group(1)}"
        if url in seen:
            continue
        seen.add(url)
        out.append({
            "title": (h.get("title") or url)[:120],
            "url": url,
            "channel": f"r/{m.group(2)}",
            "duration": None,
            "views": None,
            "platform": "reddit",
        })
        if len(out) >= limit:
            break
    return out
