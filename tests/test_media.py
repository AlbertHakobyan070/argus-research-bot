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
from pathlib import Path

import pytest

from argus import media
from argus.media import (
    MediaResult,
    detect_platform,
    download_media,
    extract_urls,
    parse_progress_line,
    quality_format,
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


def test_quality_auto_with_ffmpeg_merges_capped_1080():
    fmt = quality_format("auto", ffmpeg_available=True)
    assert "+" in fmt, "with ffmpeg, auto should merge video+audio"
    assert "1080" in fmt


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
    assert "720" in fmt and "+" in fmt


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
