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
import contextlib
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

# Absolute ceiling for one download. A 72-minute X/Twitter HLS video is
# a legitimate ~20-minute download, so the old blanket 900s killed real
# work mid-stream (observed: a 4335s tweet died at fragment 871 of ~867,
# leaving a 129 MB .part orphan). The *stall* timeout below is what
# actually catches dead downloads; this cap is just a backstop.
DEFAULT_TIMEOUT_S = int(os.environ.get("ARGUS_MEDIA_TIMEOUT", "5400"))

# Kill a download that has produced no output at all for this long. This
# is the honest liveness signal — bytes are still arriving or they are
# not, regardless of how big the video is.
STALL_TIMEOUT_S = int(os.environ.get("ARGUS_MEDIA_STALL_TIMEOUT", "300"))

# yt-dlp downloads m3u8/DASH fragments serially by default, which is why
# HLS-only platforms (X in particular) crawled. Measured on a real
# 2-hour X video: 30.1 MB in 45s serial vs 135.9 MB with -N 8 — 4.5x.
_FRAGMENT_CONCURRENCY = int(os.environ.get("ARGUS_MEDIA_FRAGMENTS", "8"))

_CONCURRENCY = int(os.environ.get("ARGUS_MEDIA_CONCURRENCY", "2"))
_download_sem = asyncio.Semaphore(_CONCURRENCY)

_PROGRESS_TEMPLATE = "ARGUSP|%(progress._percent_str)s|%(progress._eta_str)s"
_PROGRESS_RE = re.compile(r"^ARGUSP\|\s*([\d.]+)%\|\s*(\S+)")

# How often the liveness watchdog checks for a stalled download.
_WATCHDOG_TICK_S = 5.0

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
    if env:
        # ~ so a macOS/Linux .env can say ARGUS_FFMPEG=~/bin/ffmpeg.
        expanded = Path(env).expanduser()
        if expanded.exists():
            return str(expanded)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return shutil.which("ffmpeg")


def _dimension_cap(short_side: int) -> str:
    """A resolution ceiling that is correct for BOTH orientations.

    yt-dlp's ``height`` filter silently assumes landscape. Capping
    height at 1080 rejects an ordinary 1080p *vertical* video (1080 wide
    by 1922 tall) and downgrades it to 480p-class — and vertical is most
    of what gets fetched from Reels, Shorts and X. Verified live on a
    1080x1222 tweet: ``height<=1080`` picked 720x814.

    "1080p" independent of orientation means the SHORT side is 1080,
    i.e. ``min(width, height)``. yt-dlp exposes that as ``res`` in -S
    sorting but offers no ``res`` format *filter*, so the ceiling here
    bounds both dimensions at 2x the short side — a hard cap admitting
    any aspect ratio up to 2:1 — and :func:`quality_sort` then picks the
    right rung inside it.
    """
    long_side = short_side * 2
    return f"[width<={long_side}][height<={long_side}]"


def quality_format(quality: str, *, ffmpeg_available: bool) -> str:
    """Map the /quality setting onto a yt-dlp format string.

    ``auto`` caps at 1080p; ``min``/``max`` are the extremes; a bare
    number is a pixel cap on the short side. Without ffmpeg, merged
    (video+audio) formats are impossible, so single-file variants are
    used. Pair with :func:`quality_sort` for the ``-S`` argument.
    """
    q = (quality or "auto").strip().lower()
    if q == "max":
        return "bv*+ba/b" if ffmpeg_available else "b[ext=mp4]/b"
    if q == "min":
        return "wv*+wa/w" if ffmpeg_available else "w[ext=mp4]/w"
    if q == "auto":
        short = 1080
    elif q.isdigit():
        short = int(q)
    else:
        raise ValueError(
            f"quality must be auto|min|max|<pixel height>, got {quality!r}")
    cap = _dimension_cap(short)
    if ffmpeg_available:
        # mp4/m4a first so the merge lands in a container Telegram plays
        # inline rather than a .webm it shows as a file attachment.
        return (f"bv*{cap}[ext=mp4]+ba[ext=m4a]/bv*{cap}+ba/b{cap}/b")
    return f"b{cap}[ext=mp4]/b{cap}/b"


def quality_sort(quality: str) -> str | None:
    """The ``-S`` argument pairing with :func:`quality_format`, or None.

    ``res`` is yt-dlp's min(width, height), so ``res:1080`` means "1080p"
    for portrait and landscape alike. ``min``/``max`` already express an
    extreme and need no sort.
    """
    q = (quality or "auto").strip().lower()
    if q == "auto":
        return "res:1080"
    if q.isdigit():
        return f"res:{int(q)}"
    return None


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


# Raw yt-dlp stderr is accurate but not actionable — "No video could be
# found in this tweet" doesn't tell you the fix is browser cookies. Each
# entry is (needle, hint); the first match wins.
_ERROR_HINTS: list[tuple[str, str]] = [
    ("no video could be found",
     "The post has no playable video, or it's protected/deleted. If you "
     "can see the video while logged in, give Argus your browser cookies: "
     "set ARGUS_YTDLP_COOKIES_BROWSER=chrome (or firefox/edge/safari)."),
    ("nsfw", "Age-restricted post — needs logged-in cookies. Set "
             "ARGUS_YTDLP_COOKIES_BROWSER=chrome (or firefox/edge/safari)."),
    ("sign in to confirm your age",
     "Age-restricted video — set ARGUS_YTDLP_COOKIES_BROWSER=chrome "
     "(or firefox/edge/safari) so yt-dlp can use your login."),
    ("sign in to confirm you're not a bot",
     "The platform is rate-limiting this IP. Set "
     "ARGUS_YTDLP_COOKIES_BROWSER=chrome (or firefox/edge/safari), or "
     "retry in a few minutes."),
    ("login required", "Login-walled post — set "
                       "ARGUS_YTDLP_COOKIES_BROWSER=chrome (or firefox/"
                       "edge/safari)."),
    ("private video", "This video is private — only a logged-in account "
                      "with access can fetch it (ARGUS_YTDLP_COOKIES_BROWSER)."),
    ("video unavailable", "The platform says this video is unavailable — "
                          "it may be deleted or region-blocked."),
    ("restricted video", "Region- or age-restricted — try cookies via "
                         "ARGUS_YTDLP_COOKIES_BROWSER."),
    ("requested format is not available",
     "No stream matched the current /quality setting. Try /quality auto."),
    ("unable to download webpage", "Network problem reaching the platform "
                                   "(DNS/proxy/offline?)."),
    ("unable to extract",
     "The platform changed its page layout and this yt-dlp can't parse it. "
     "Update the downloader: uv pip install -U --pre yt-dlp"),
    ("unsupported url",
     "yt-dlp has no extractor for this link. Update it "
     "(uv pip install -U --pre yt-dlp) or check the URL."),
    ("stalled —", "The download made no progress for several minutes — "
                  "usually a dropped connection. Re-send the link; yt-dlp "
                  "resumes from the partial file."),
    ("ceiling", "The download exceeded its time ceiling. Re-send the link "
                "to resume, or raise ARGUS_MEDIA_TIMEOUT."),
]


def explain_error(stderr: str) -> str:
    """Append an actionable hint to a raw downloader error, when we
    recognise it. The original stderr is always preserved."""
    low = (stderr or "").lower()
    for needle, hint in _ERROR_HINTS:
        if needle in low:
            return f"{stderr.strip()}\n\n💡 {hint}"
    return (stderr or "").strip()


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
                marker: Path, ffmpeg: str | None,
                sort: str | None = None) -> list[str]:
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
        # Parallel fragment fetch — the single biggest win for HLS/DASH
        # (X, and YouTube's DASH formats). Serial fragments are what made
        # long videos time out.
        "--concurrent-fragments", str(_FRAGMENT_CONCURRENCY),
        # Survive the transient 5xx/reset that a single fragment hits on a
        # 900-fragment stream instead of failing the whole download.
        "--retries", "10",
        "--fragment-retries", "20",
        "--retry-sleep", "linear=1::5",
        # Don't let one wedged socket burn the whole stall budget.
        "--socket-timeout", "30",
    ]
    if sort:
        argv += ["-S", sort]
    if ffmpeg:
        argv += ["--ffmpeg-location", ffmpeg]
    cookies_file = os.environ.get("ARGUS_YTDLP_COOKIES")
    cookies_browser = os.environ.get("ARGUS_YTDLP_COOKIES_BROWSER")
    if cookies_file:
        argv += ["--cookies", cookies_file]
    elif cookies_browser:
        argv += ["--cookies-from-browser", cookies_browser]
    return argv


def _kill_tree(pid: int) -> None:
    """Kill the downloader *and its children*.

    yt-dlp spawns ffmpeg to merge video+audio and to remux HLS. Killing
    only the parent leaves ffmpeg running, still writing into the vault
    minutes after we already reported failure to the user — which is how
    two fully-downloaded files (647 MB and 106 MB) ended up on disk with
    no library row and no Telegram delivery.
    """
    import subprocess

    if os.name == "nt":
        # No process groups; taskkill /T walks the child tree.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True)
        return
    import signal
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def _run_streaming(cmd: list[str], *, timeout_s: int,
                         on_line: Callable[[str], None],
                         stall_s: int = STALL_TIMEOUT_S) -> tuple[int, str]:
    """Run cmd, streaming stdout lines to ``on_line``. Returns
    (returncode, stderr_tail).

    Liveness is judged by *silence*, not by total elapsed time: the
    download is killed after ``stall_s`` with no output, or at the
    ``timeout_s`` backstop. A blanket wall-clock limit cannot tell a
    healthy 70-minute download from a wedged one.

    Falls back to a thread-blocking run when the running loop can't spawn
    subprocesses (non-Proactor Windows loop) — progress lines are then
    delivered only at the end, and only ``timeout_s`` applies.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    spawn_kwargs: dict[str, Any] = {}
    if os.name != "nt":
        # Own process group so _kill_tree can killpg the ffmpeg child.
        spawn_kwargs["start_new_session"] = True
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, env=env, **spawn_kwargs)
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
    started = time.monotonic()
    watch = {"last": started, "reason": ""}

    async def _pump_stdout():
        assert proc.stdout is not None
        async for raw in proc.stdout:
            watch["last"] = time.monotonic()
            on_line(raw.decode("utf-8", "replace").rstrip())

    async def _pump_stderr():
        assert proc.stderr is not None
        data = await proc.stderr.read()
        stderr_buf.append(data)

    async def _watchdog():
        while True:
            await asyncio.sleep(_WATCHDOG_TICK_S)
            now = time.monotonic()
            if now - watch["last"] > stall_s:
                watch["reason"] = (
                    f"stalled — no progress for {stall_s}s "
                    f"(after {int(now - started)}s of downloading)")
            elif now - started > timeout_s:
                watch["reason"] = f"hit the {timeout_s}s ceiling"
            else:
                continue
            logger.warning("killing download tree: %s", watch["reason"])
            _kill_tree(proc.pid)
            return

    watchdog = asyncio.create_task(_watchdog())
    try:
        await asyncio.gather(_pump_stdout(), _pump_stderr(), proc.wait())
    finally:
        watchdog.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watchdog

    if watch["reason"]:
        return 124, f"download {watch['reason']}"
    stderr = b"".join(stderr_buf).decode("utf-8", "replace")
    return proc.returncode or 0, stderr[-2000:]


async def download_media(
    url: str, *, dest_root: Path | str, quality: str = "auto",
    on_progress: Callable[[float, str], None] | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    stall_s: int = STALL_TIMEOUT_S,
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
    argv = _build_argv(url, platform, dest_root, fmt, marker, ffmpeg,
                       quality_sort(quality))
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
                cmd, timeout_s=timeout_s, on_line=_on_line,
                stall_s=stall_s)
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

    if final_path is None:
        err = stderr.strip() or f"yt-dlp exited {rc} without an output file"
        return MediaResult(ok=False, url=url, platform=platform,
                           error=explain_error(err)[-800:],
                           elapsed_s=time.monotonic() - t0)
    if rc != 0:
        # The file was moved into place, so the download itself finished —
        # a non-zero rc here is a post-step (or our own watchdog kill)
        # racing the finish. Keep the file instead of calling it a failure
        # and orphaning a complete video in the vault.
        logger.warning("yt-dlp exited %s but produced %s — keeping it",
                       rc, final_path)

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
