"""Media engine tests — platform routing, quality mapping, downloads.

Phase 2 of the v2 rebuild. Hermetic: ``download_media`` is exercised
against a stub script standing in for ``python -m yt_dlp`` (real
subprocess path, fake downloader), searches against mocked httpx.
"""
from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import time
from pathlib import Path

import pytest

from argus import media
from argus.media import (
    MediaResult,
    _build_argv,
    detect_platform,
    download_media,
    explain_error,
    extract_urls,
    parse_progress_line,
    quality_format,
    quality_sort,
    reddit_search,
)


# ---------------------------------------------------------------------------
# platform detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/watch?v=jNQXAC9IVRw", "youtube"),
    ("https://youtu.be/jNQXAC9IVRw", "youtube"),
    ("https://www.youtube.com/shorts/abc123DEF45", "youtube"),
    ("https://music.youtube.com/watch?v=abc123DEF45", "youtube"),
    ("https://x.com/user/status/1234567890", "x"),
    ("https://twitter.com/user/status/1234567890", "x"),
    ("https://mobile.twitter.com/user/status/1234567890", "x"),
    ("https://www.reddit.com/r/videos/comments/abc123/some_title/", "reddit"),
    ("https://old.reddit.com/r/videos/comments/abc123/t/", "reddit"),
    ("https://redd.it/abc123", "reddit"),
    ("https://v.redd.it/xyz789", "reddit"),
    ("https://www.instagram.com/reel/Cabc123/", "instagram"),
    ("https://www.instagram.com/reels/Cabc123/", "instagram"),
    ("https://www.instagram.com/p/Cabc123/", "instagram"),
    ("https://example.com/watch?v=abc", None),
    ("https://www.youtube.com/@somechannel", None),      # channel, not video
    ("https://x.com/user", None),                        # profile, not status
    ("not a url at all", None),
])
def test_detect_platform(url, expected):
    assert detect_platform(url) == expected


def test_extract_urls_finds_dedupes_and_keeps_order():
    text = ("check this https://youtu.be/abc123DEF45 and also "
            "https://x.com/u/status/99 again https://youtu.be/abc123DEF45, "
            "plus trailing-punct https://redd.it/zzz.")
    urls = extract_urls(text)
    assert urls == [
        "https://youtu.be/abc123DEF45",
        "https://x.com/u/status/99",
        "https://redd.it/zzz",
    ]


def test_extract_urls_empty_for_plain_text():
    assert extract_urls("no links here") == []


# ---------------------------------------------------------------------------
# quality → yt-dlp format mapping
# ---------------------------------------------------------------------------


def test_quality_auto_with_ffmpeg_merges_and_caps():
    fmt = quality_format("auto", ffmpeg_available=True)
    assert "+" in fmt, "with ffmpeg, auto should merge video+audio"
    assert "2160" in fmt, "auto caps both dimensions at 2x the 1080 short side"


def test_quality_auto_without_ffmpeg_is_single_file():
    fmt = quality_format("auto", ffmpeg_available=False)
    assert "+" not in fmt, "without ffmpeg yt-dlp cannot merge streams"
    assert "mp4" in fmt


def test_quality_max_and_min():
    assert quality_format("max", ffmpeg_available=True) == "bv*+ba/b"
    assert "+" not in quality_format("max", ffmpeg_available=False)
    lo = quality_format("min", ffmpeg_available=False)
    assert "w" in lo


def test_quality_numeric_height():
    fmt = quality_format("720", ffmpeg_available=True)
    assert "1440" in fmt and "+" in fmt


def test_quality_cap_does_not_reject_vertical_1080p():
    """Regression: ``height<=1080`` treats portrait video as oversized.

    A normal 1080p vertical clip is 1080x1922, and the old filter threw
    it away in favour of a 480p rung — verified live against a 1080x1222
    tweet, which downgraded to 720x814. The cap must bound the SHORT
    side, so both dimensions get 2x headroom.
    """
    fmt = quality_format("auto", ffmpeg_available=True)
    assert "[height<=1080]" not in fmt, (
        "a bare height cap silently downgrades vertical video")
    # 1080x1922 (portrait 1080p) must survive both bounds.
    assert "[width<=2160][height<=2160]" in fmt


def test_quality_sort_uses_res_for_orientation_independence():
    # `res` is yt-dlp's min(width, height) — the orientation-neutral one.
    assert quality_sort("auto") == "res:1080"
    assert quality_sort("720") == "res:720"
    assert quality_sort("max") is None, "max/min are already extremes"
    assert quality_sort("min") is None


def test_quality_rejects_garbage():
    with pytest.raises(ValueError):
        quality_format("potato", ffmpeg_available=True)


# ---------------------------------------------------------------------------
# progress parsing
# ---------------------------------------------------------------------------


def test_parse_progress_line():
    assert parse_progress_line("ARGUSP|  42.3%| 00:12") == (42.3, "00:12")
    assert parse_progress_line("ARGUSP|100.0%|00:00") == (100.0, "00:00")
    assert parse_progress_line("[download] something else") is None
    assert parse_progress_line("") is None


# ---------------------------------------------------------------------------
# download_media against a stub downloader (real subprocess, fake yt-dlp)
# ---------------------------------------------------------------------------

_STUB_OK = textwrap.dedent("""\
    import json, sys
    from pathlib import Path
    argv = sys.argv[1:]
    def _next(flag):
        i = argv.index(flag)
        return argv[i + 1]
    dest = Path(_next("-P"))
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "20260101_fake1_stub-video.mp4"
    out.write_bytes(b"x" * 4096)
    (dest / "20260101_fake1_stub-video.info.json").write_text(
        json.dumps({"title": "Stub video", "id": "fake1",
                    "duration": 61.5}), encoding="utf-8")
    # --print-to-file takes TWO values: [WHEN:]TEMPLATE FILE
    marker = Path(argv[argv.index("--print-to-file") + 2])
    print("ARGUSP|  10.0%|00:20", flush=True)
    print("ARGUSP| 100.0%|00:00", flush=True)
    marker.write_text(str(out), encoding="utf-8")
""")

_STUB_FAIL = textwrap.dedent("""\
    import sys
    print("ERROR: [twitter] NSFW tweet requires authentication",
          file=sys.stderr)
    sys.exit(1)
""")


def _install_stub(tmp_path: Path, monkeypatch, code: str) -> None:
    stub = tmp_path / "stub_ytdlp.py"
    stub.write_text(code, encoding="utf-8")
    monkeypatch.setattr(media, "_YTDLP_PREFIX",
                        [sys.executable, str(stub)])


async def test_download_media_happy_path(tmp_path, monkeypatch):
    _install_stub(tmp_path, monkeypatch, _STUB_OK)
    seen: list[float] = []

    r = await download_media(
        "https://youtu.be/abc123DEF45", dest_root=tmp_path / "vault",
        on_progress=lambda pct, eta: seen.append(pct))

    assert r.ok, r.error
    assert r.platform == "youtube"
    p = Path(r.path)
    assert p.exists() and p.suffix == ".mp4"
    assert "youtube" in p.parts, "file must land under <dest>/youtube/"
    assert r.size_bytes == 4096
    assert r.title == "Stub video"
    assert r.media_id == "fake1"
    assert r.duration_s == 61.5
    assert seen and seen[-1] == 100.0, "progress callback must fire"


async def test_download_media_failure_surfaces_stderr(tmp_path, monkeypatch):
    _install_stub(tmp_path, monkeypatch, _STUB_FAIL)
    r = await download_media("https://x.com/u/status/9",
                             dest_root=tmp_path / "vault")
    assert r.ok is False
    assert "NSFW" in (r.error or ""), (
        "the downloader's stderr must reach the user, not vanish")


async def test_download_media_rejects_unknown_platform(tmp_path):
    r = await download_media("https://example.com/x", dest_root=tmp_path)
    assert r.ok is False
    assert "platform" in (r.error or "").lower()


# ---------------------------------------------------------------------------
# download hardening: fragment concurrency, stall watchdog, salvage
# ---------------------------------------------------------------------------


def test_argv_requests_parallel_fragments_and_retries():
    """Serial m3u8 fragments are why long X videos timed out. Measured on
    a real 2-hour tweet: 30.1 MB/45s serial vs 135.9 MB/45s at -N 8."""
    argv = _build_argv("u", "x", Path("/tmp"), "bv*+ba", Path("/tmp/m"),
                       None, "res:1080")
    assert "--concurrent-fragments" in argv
    assert int(argv[argv.index("--concurrent-fragments") + 1]) > 1
    # A single flaky fragment out of ~900 must not kill the whole run.
    assert "--fragment-retries" in argv
    assert "--socket-timeout" in argv
    assert argv[argv.index("-S") + 1] == "res:1080"


def test_argv_omits_sort_when_not_applicable():
    argv = _build_argv("u", "x", Path("/tmp"), "bv*+ba", Path("/tmp/m"),
                       None, None)
    assert "-S" not in argv


# A downloader that prints one line then sleeps forever — the shape of a
# wedged connection. The watchdog must reap it via the stall timeout.
_STUB_STALL = textwrap.dedent("""\
    import sys, time
    print("ARGUSP|  5.0%|10:00", flush=True)
    time.sleep(600)
""")


async def test_stalled_download_is_killed_by_the_stall_watchdog(
        tmp_path, monkeypatch):
    """A blanket wall-clock limit cannot tell a healthy 70-minute
    download from a dead one. Silence can."""
    _install_stub(tmp_path, monkeypatch, _STUB_STALL)
    monkeypatch.setattr(media, "_WATCHDOG_TICK_S", 0.05)

    t0 = time.monotonic()
    r = await download_media("https://x.com/u/status/9",
                             dest_root=tmp_path / "vault",
                             timeout_s=600, stall_s=1)
    elapsed = time.monotonic() - t0

    assert r.ok is False
    assert elapsed < 30, (
        f"stalled download should die in ~1s, took {elapsed:.1f}s")
    assert "stall" in (r.error or "").lower()
    # and the user is told what to do about it
    assert "resume" in (r.error or "").lower()


async def test_stall_kill_reaps_grandchildren(tmp_path, monkeypatch):
    """yt-dlp spawns ffmpeg to merge. Killing only the parent left ffmpeg
    writing into the vault long after the user was told it failed."""
    beacon = tmp_path / "grandchild_survived.txt"
    stub = textwrap.dedent(f"""\
        import subprocess, sys, time
        # stand-in for the ffmpeg child yt-dlp spawns
        subprocess.Popen([sys.executable, "-c",
            "import time; time.sleep(6);"
            "open(r'{beacon}', 'w').write('alive')"])
        print("ARGUSP|  5.0%|10:00", flush=True)
        time.sleep(600)
    """)
    _install_stub(tmp_path, monkeypatch, stub)
    monkeypatch.setattr(media, "_WATCHDOG_TICK_S", 0.05)

    r = await download_media("https://x.com/u/status/9",
                             dest_root=tmp_path / "vault",
                             timeout_s=600, stall_s=1)
    assert r.ok is False
    await asyncio.sleep(8)  # outlive the grandchild's own sleep
    assert not beacon.exists(), (
        "grandchild survived the kill — it will keep writing into the "
        "vault after we already reported failure")


# Produces the output file, then exits non-zero (a post-processing step
# failing, or our own watchdog landing just after the file was moved).
_STUB_FILE_THEN_NONZERO = textwrap.dedent("""\
    import json, sys
    from pathlib import Path
    argv = sys.argv[1:]
    dest = Path(argv[argv.index("-P") + 1])
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "20260101_fake1_stub-video.mp4"
    out.write_bytes(b"y" * 2048)
    (dest / "20260101_fake1_stub-video.info.json").write_text(
        json.dumps({"title": "Salvaged", "id": "fake1", "duration": 9.0}),
        encoding="utf-8")
    Path(argv[argv.index("--print-to-file") + 2]).write_text(
        str(out), encoding="utf-8")
    sys.exit(1)
""")


async def test_completed_file_is_kept_even_when_ytdlp_exits_nonzero(
        tmp_path, monkeypatch):
    """Two fully-downloaded videos (647 MB and 106 MB) sat in the vault
    unregistered because a non-zero rc discarded a finished file."""
    _install_stub(tmp_path, monkeypatch, _STUB_FILE_THEN_NONZERO)
    r = await download_media("https://x.com/u/status/9",
                             dest_root=tmp_path / "vault")
    assert r.ok is True, r.error
    assert Path(r.path).exists()
    assert r.title == "Salvaged"
    assert r.size_bytes == 2048


# ---------------------------------------------------------------------------
# error hints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,needle", [
    ("ERROR: [twitter] No video could be found in this tweet", "cookies"),
    ("ERROR: Sign in to confirm your age", "cookies"),
    ("ERROR: Requested format is not available", "/quality auto"),
    ("download stalled — no progress for 300s", "resume"),
])
def test_explain_error_adds_an_actionable_hint(raw, needle):
    out = explain_error(raw)
    assert raw.strip() in out, "the raw downloader error must survive"
    assert needle.lower() in out.lower()


def test_explain_error_passes_unknown_errors_through():
    assert explain_error("ERROR: something novel") == "ERROR: something novel"
    assert explain_error("") == ""


# ---------------------------------------------------------------------------
# reddit search (mocked httpx)
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, params=None):
        return _FakeResp(self._payload)


def _reddit_payload():
    def child(i, is_video):
        return {"data": {
            "title": f"Post {i}",
            "permalink": f"/r/videos/comments/id{i}/post_{i}/",
            "subreddit": "videos",
            "score": 100 + i,
            "is_video": is_video,
            "media": ({"reddit_video": {"duration": 33}}
                      if is_video else None),
        }}
    return {"data": {"children": [
        child(1, True), child(2, False), child(3, True)]}}


async def test_reddit_search_filters_videos_and_maps_shape(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kw: _FakeClient(_reddit_payload()))
    results = await reddit_search("test query", limit=5)
    assert len(results) == 2, "non-video posts must be filtered out"
    r = results[0]
    assert r["title"] == "Post 1"
    assert r["url"].startswith("https://www.reddit.com/r/videos/comments/")
    assert r["channel"] == "r/videos"
    assert r["duration"] == 33


class _Boom:
    async def __aenter__(self):
        raise RuntimeError("403 Blocked")

    async def __aexit__(self, *a):
        return False


async def test_reddit_search_falls_back_to_ddgs_when_blocked(monkeypatch):
    """Reddit 403-blocks search.json from many networks (observed live
    2026-07-10). The ddgs fallback must surface reddit post links."""
    import httpx
    from argus import tools as tools_mod
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _Boom())
    monkeypatch.setattr(tools_mod, "ddgs_search", lambda q, **kw: [
        {"url": "https://www.reddit.com/r/videos/comments/abc1/cool_video/",
         "title": "Cool video", "snippet": ""},
        {"url": "https://www.reddit.com/r/videos/wiki/rules",  # not a post
         "title": "rules", "snippet": ""},
        {"url": "https://www.reddit.com/r/videos/comments/abc1/cool_video/?utm=1",
         "title": "dupe", "snippet": ""},
    ])
    results = await reddit_search("cool")
    assert len(results) == 1, "non-post URLs filtered, dupes collapsed"
    assert results[0]["url"].endswith("/comments/abc1/cool_video/")
    assert results[0]["channel"] == "r/videos"
    assert results[0]["platform"] == "reddit"


async def test_reddit_search_failsoft_when_both_backends_fail(monkeypatch):
    import httpx
    from argus import tools as tools_mod
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _Boom())

    def _boom_ddgs(q, **kw):
        raise RuntimeError("ddgs down")

    monkeypatch.setattr(tools_mod, "ddgs_search", _boom_ddgs)
    assert await reddit_search("q") == []
