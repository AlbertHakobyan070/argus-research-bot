"""Hermetic tests for the Phase 2 transcript pipeline.

Covers:
  - ``tools._vtt_to_text`` — VTT cue extraction, header stripping,
    multi-line cues, metadata lines, positioning tags.
  - ``tools.youtube_video_transcript`` — fail-soft on bad URL, fail-soft
    on no-vtt, success path. We monkeypatch ``_run_yt_dlp`` and the on-disk
    reads so the suite is hermetic (no network, no real yt-dlp invocations).
  - ``bot._parse_indices`` — comma/space separators, ranges, ``all``,
    bad tokens silently dropped, command prefix.
  - ``bot._pool_put`` / ``_pool_get`` — basic round-trip and TTL expiry.

No Telegram bot is started; we exercise the parsers and the tool layer
directly. The bot's glue (transcript_cmd) is shaped identically to other
handlers — if the tool + parser hold, the handler is wired by inspection.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from argus import bot, tools


# ---------------------------------------------------------------------------
# VTT stripping
# ---------------------------------------------------------------------------

_VTT_FIXTURE = (
    "WEBVTT\n"
    "Kind: captions\n"
    "Language: en\n"
    "\n"
    "00:00:01.200 --> 00:00:03.360\n"
    "All right, so here we are, in front of the\n"
    "elephants\n"
    "\n"
    "00:00:05.318 --> 00:00:07.974\n"
    "the cool thing about these guys is that they\n"
    "have really really long trunks\n"
    "\n"
    "00:00:12.616 --> 00:00:14.500\n"
    "<c.color>and</c> <00:00:12.616>that's cool\n"
)


def test_vtt_to_text_strips_headers_and_timings(tmp_path: Path):
    f = tmp_path / "x.en-en.vtt"
    f.write_text(_VTT_FIXTURE, encoding="utf-8")
    out = tools._vtt_to_text(f)
    # Three blocks, one line per block (multi-line joined with space).
    blocks = out.splitlines()
    assert len(blocks) == 3
    # Cue 1: multi-line collapsed
    assert blocks[0] == "All right, so here we are, in front of the elephants"
    # Cue 2: multi-line collapsed, no orphan whitespace
    assert blocks[1].startswith("the cool thing about these guys")
    assert blocks[1].endswith("have really really long trunks")
    # Cue 3: positioning tags stripped
    assert "and that's cool" in blocks[2]
    assert "<" not in blocks[2] and ">" not in blocks[2]


def test_vtt_to_text_returns_empty_for_missing_file(tmp_path: Path):
    assert tools._vtt_to_text(tmp_path / "nope.vtt") == ""


def test_vtt_to_text_handles_empty_file(tmp_path: Path):
    f = tmp_path / "empty.vtt"
    f.write_text("", encoding="utf-8")
    assert tools._vtt_to_text(f) == ""


def test_vtt_to_text_drops_note_style_blocks(tmp_path: Path):
    f = tmp_path / "n.vtt"
    f.write_text(
        "WEBVTT\n\n"
        "NOTE this is a comment\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "actual caption text\n",
        encoding="utf-8",
    )
    out = tools._vtt_to_text(f)
    assert "NOTE" not in out
    assert "actual caption text" in out


# ---------------------------------------------------------------------------
# youtube_video_transcript — hermetic (monkeypatch _run_yt_dlp)
# ---------------------------------------------------------------------------

_INFO_JSON = {
    "id": "jNQXAC9IVRw",
    "title": "Me at the zoo",
    "channel": "jawed",
    "duration": 19,
    "webpage_url": "https://www.youtube.com/watch?v=jNQXAC9IVRw",
}


def _fake_ytdlp_success(tmpdir: Path):
    """Build a fake subdirectory as if yt-dlp had just run: a vtt + an
    info.json + an extra non-vtt file (to confirm we filter)."""
    sub = tmpdir / "fake_out"
    sub.mkdir()
    (sub / "jNQXAC9IVRw.en-en.vtt").write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nhi there\n",
        encoding="utf-8",
    )
    (sub / "jNQXAC9IVRw.info.json").write_text(
        json.dumps(_INFO_JSON), encoding="utf-8")
    (sub / "other.txt").write_text("garbage", encoding="utf-8")
    return sub


def _fake_ytdlp_success(monkeypatch, tmp_path: Path, vtt_text: str | None = None,
                        info_json: dict | None = None):
    """Set up a fake tempfile.TemporaryDirectory that yields ``tmp_path``.

    ``tmp_path`` already exists (pytest fixture creates it), so the fake
    returns it as ``__enter__`` instead of mkdir'ing a sub-directory.
    Pre-populates ``.vtt`` and ``.info.json`` to mimic a successful yt-dlp.
    """
    vtt = vtt_text or (
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello world line one\n"
        "line two continues\n")
    info = info_json or _INFO_JSON

    (tmp_path / f"{info['id']}.en-en.vtt").write_text(vtt, encoding="utf-8")
    (tmp_path / f"{info['id']}.info.json").write_text(
        json.dumps(info), encoding="utf-8")

    class _FakeTD:
        def __init__(self, prefix=""):
            pass

        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, *a):
            return False

    return _FakeTD


def test_youtube_video_transcript_success(monkeypatch, tmp_path: Path):
    """Simulate the tempdir branch by patching TemporaryDirectory."""
    _FakeTD = _fake_ytdlp_success(monkeypatch, tmp_path)

    monkeypatch.setattr(tools.tempfile, "TemporaryDirectory", _FakeTD)

    rc_log: list = []
    def fake_run(args, **kw):
        rc_log.append(list(args))
        # Confirm -o template lives under our fake dir.
        assert any(a.startswith(str(tmp_path)) for a in args), \
            f"-o not under tmpdir: {args}"
        return (0, "", "")
    monkeypatch.setattr(tools, "_run_yt_dlp", fake_run)

    r = tools.youtube_video_transcript(
        "https://www.youtube.com/watch?v=jNQXAC9IVRw", timeout=10)
    assert r.ok, r.error
    assert r.title == "Me at the zoo"
    assert r.channel == "jawed"
    assert r.duration == 19
    assert r.language == "en-en"
    assert "Hello world" in r.transcript_text
    assert "line one line two continues" in r.transcript_text
    # transcript_bytes always populated when ok=True (lets the bot ship the
    # file without a tempdir dance), and suggested_filename looks sane.
    assert r.transcript_bytes
    assert r.transcript_bytes.decode("utf-8") == r.transcript_text
    assert r.suggested_filename.endswith(".txt")
    # Args sanity: --skip-download, --write-auto-subs, --sub-langs en.*
    args = rc_log[0]
    assert "--skip-download" in args
    assert "--write-auto-subs" in args
    assert any(a.startswith("--sub-langs=") for a in args)
    assert args[-1].startswith("https://")


# ---------------------------------------------------------------------------
# format=txt|srt mode (Phase 2.5, 2026-07-10)
# ---------------------------------------------------------------------------

_SRT_VTT_FIXTURE = (
    "WEBVTT\n"
    "Kind: captions\n"
    "Language: en\n"
    "\n"
    "00:00:01.200 --> 00:00:03.360\n"
    "All right, so here we are, in front of the\n"
    "elephants\n"
    "\n"
    "00:00:05.318 --> 00:00:07.974\n"
    "Welcome to the zoo.\n"
)

_SRT_TIMESTAMP_RE = re.compile(
    r"\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}"
)


def test_youtube_video_transcript_format_srt_preserves_timestamps(
    monkeypatch, tmp_path: Path,
):
    """format='srt' returns the raw .vtt body with timestamps intact.

    text_mode='txt' must NOT call _vtt_to_text when format='srt' is
    requested; the raw .vtt bytes are shipped verbatim so the user can
    see when each line was spoken.
    """
    _fake_ytdlp = _fake_ytdlp_success(
        monkeypatch, tmp_path, vtt_text=_SRT_VTT_FIXTURE)
    monkeypatch.setattr(tools.tempfile, "TemporaryDirectory", _fake_ytdlp)

    def fake_run(args, **kw):
        return (0, "", "")
    monkeypatch.setattr(tools, "_run_yt_dlp", fake_run)
    monkeypatch.setattr(tools, "_vtt_to_text", lambda _p: (
        pytest.fail("_vtt_to_text should NOT run under format='srt'")))

    r = tools.youtube_video_transcript(
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",
        format="srt", timeout=10)

    assert r.ok, r.error
    # Timestamps kept in both the in-memory text and the bytes.
    assert _SRT_TIMESTAMP_RE.search(r.transcript_text), (
        f"expected SRT-style timestamp in transcript_text, got: "
        f"{r.transcript_text!r}"
    )
    assert _SRT_TIMESTAMP_RE.search(r.transcript_bytes.decode("utf-8"))
    # The WEBVTT header line is preserved.
    assert "WEBVTT" in r.transcript_text
    # Suggested filename drops '.txt' for '.srt'.
    assert r.suggested_filename.endswith(".srt")


def test_youtube_video_transcript_default_format_is_txt(
    monkeypatch, tmp_path: Path,
):
    """No format= kwarg -> legacy 'txt' behaviour (timestamps stripped)."""
    _fake_ytdlp = _fake_ytdlp_success(
        monkeypatch, tmp_path, vtt_text=_SRT_VTT_FIXTURE)
    monkeypatch.setattr(tools.tempfile, "TemporaryDirectory", _fake_ytdlp)

    def fake_run(args, **kw):
        return (0, "", "")
    monkeypatch.setattr(tools, "_run_yt_dlp", fake_run)

    r = tools.youtube_video_transcript(
        "https://www.youtube.com/watch?v=jNQXAC9IVRw", timeout=10)
    assert r.ok, r.error
    # Timings stripped -> no SRT-style stamp survives in transcript_text.
    assert not _SRT_TIMESTAMP_RE.search(r.transcript_text), (
        f"timestamps leaked through txt-mode: {r.transcript_text!r}"
    )
    assert r.suggested_filename.endswith(".txt")


def test_youtube_video_transcript_format_invalid_raises():
    """Bad format -> ValueError, fail-fast before yt-dlp is invoked."""
    with pytest.raises(ValueError, match="format"):
        tools.youtube_video_transcript(
            "https://www.youtube.com/watch?v=jNQXAC9IVRw", format="json")


def test_parse_indices_ignores_format_flag():
    """`/transcript 2 format=srt` -> indices=[2], the flag is dropped."""
    indices = bot._parse_indices("2 format=srt")
    assert indices == [2]


def test_parse_indices_with_format_flag_and_range():
    """`/transcript 1-3 format=txt` -> indices=[1,2,3]."""
    indices = bot._parse_indices("1-3 format=txt")
    assert indices == [1, 2, 3]


def test_parse_indices_format_only_no_indices():
    """`/transcript format=srt` -> empty list (caller surfaces usage hint)."""
    indices = bot._parse_indices("format=srt")
    assert indices == []


def test_youtube_video_transcript_empty_url():
    r = tools.youtube_video_transcript("")
    assert not r.ok
    assert "empty" in (r.error or "").lower()


def test_youtube_video_transcript_non_http_url():
    r = tools.youtube_video_transcript("ftp://nope.example/x.mp4")
    assert not r.ok
    assert "url" in (r.error or "").lower()


def test_youtube_video_transcript_fails_when_no_vtt(monkeypatch, tmp_path: Path):
    """yt-dlp runs, no .vtt appears -> ok=False with explanation."""
    class _EmptyTD:
        def __init__(self, prefix=""):
            pass

        def __enter__(self):
            return str(tmp_path)

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(tools.tempfile, "TemporaryDirectory", _EmptyTD)
    monkeypatch.setattr(tools, "_run_yt_dlp",
                        lambda a, **kw: (0, "", ""))

    r = tools.youtube_video_transcript("https://www.youtube.com/watch?v=x")
    assert not r.ok
    assert "no .vtt" in (r.error or "").lower()


# ---------------------------------------------------------------------------
# Index parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("2",          [2]),
    ("2,4",        [2, 4]),
    ("2 4",        [2, 4]),
    ("2, 4",       [2, 4]),
    ("  1 , 3  ",  [1, 3]),
    ("1-3",        [1, 2, 3]),
    ("3-1",        [1, 2, 3]),
    ("all",        []),       # sentinel: caller expands
    ("*",          []),
    ("",           []),
    ("/transcript 2,4", [2, 4]),
    # Out-of-range isn't the parser's job; it only filters non-ints.
    ("2,9,gibberish,4", [2, 9, 4]),
    ("-1",         []),       # zero/negative dropped
    ("0",          []),
    ("abc,def",    []),
])
def test_parse_indices(text, expected):
    assert bot._parse_indices(text) == expected


# ---------------------------------------------------------------------------
# Pool round-trip + TTL
# ---------------------------------------------------------------------------

def test_pool_round_trip():
    bot._video_pool.clear()
    results = [{"url": "https://example/v=1", "title": "A"}]
    bot._pool_put("tg:1", results)
    assert bot._pool_get("tg:1") == results


def test_pool_ttl_expiry(monkeypatch):
    bot._video_pool.clear()
    bot._pool_put("tg:2", [{"url": "u", "title": "t"}])
    # Pretend 31 minutes have passed
    base = time.time()
    monkeypatch.setattr(bot.time, "time", lambda: base + 31 * 60)
    assert bot._pool_get("tg:2") is None
    # And the entry is now removed
    assert "tg:2" not in bot._video_pool


def test_pool_replaces_on_new_search():
    bot._video_pool.clear()
    bot._pool_put("tg:3", [{"url": "old", "title": "old"}])
    bot._pool_put("tg:3", [{"url": "new", "title": "new"}])
    res = bot._pool_get("tg:3")
    assert res and res[0]["url"] == "new"


def test_pool_isolates_threads():
    bot._video_pool.clear()
    bot._pool_put("tg:a", [{"url": "A", "title": "A"}])
    bot._pool_put("tg:b", [{"url": "B", "title": "B"}])
    assert bot._pool_get("tg:a")[0]["url"] == "A"
    assert bot._pool_get("tg:b")[0]["url"] == "B"


# ---------------------------------------------------------------------------
# rolling-caption duplication (v2 Phase 3 — Albert's 2026-07-10 screenshot)
# ---------------------------------------------------------------------------

_ROLLING_VTT = (
    "WEBVTT\n"
    "Kind: captions\n"
    "Language: en\n"
    "\n"
    "00:00:00.000 --> 00:00:02.000\n"
    "when you really need to, looking at\n"
    "\n"
    "00:00:02.000 --> 00:00:04.000\n"
    "when you really need to, looking at changes,\n"
    "\n"
    "00:00:04.000 --> 00:00:06.000\n"
    "changes, clicking around, going to\n"
    "\n"
    "00:00:06.000 --> 00:00:08.000\n"
    "changes, clicking around, going to definition.\n"
    "\n"
    "00:00:08.000 --> 00:00:10.000\n"
    "Probably the most exciting\n"
    "\n"
    "00:00:10.000 --> 00:00:12.000\n"
    "Probably the most exciting\n"
)


def test_vtt_to_text_collapses_rolling_caption_overlap(tmp_path: Path):
    """YouTube auto-captions repeat each cue's tail at the start of the
    next cue (rolling window). The txt transcript must collapse the
    overlap so prose reads once, not twice (live report 2026-07-10)."""
    f = tmp_path / "roll.vtt"
    f.write_text(_ROLLING_VTT, encoding="utf-8")
    out = tools._vtt_to_text(f)

    assert out.count("when you really need to, looking at") == 1, out
    assert out.count("changes, clicking around, going to") == 1, out
    assert out.count("Probably the most exciting") == 1, out
    # No word content may be lost either.
    assert "definition." in out


def test_vtt_to_text_keeps_distinct_consecutive_lines(tmp_path: Path):
    """Dedup must not eat genuinely distinct cues."""
    f = tmp_path / "ok.vtt"
    f.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\nfirst thought here\n\n"
        "00:00:02.000 --> 00:00:03.000\nsecond idea entirely\n",
        encoding="utf-8")
    out = tools._vtt_to_text(f)
    assert "first thought here" in out
    assert "second idea entirely" in out


# ---------------------------------------------------------------------------
# out_dir vault persistence (v2 Phase 3)
# ---------------------------------------------------------------------------


def test_youtube_video_transcript_persists_into_out_dir(monkeypatch,
                                                        tmp_path: Path):
    info = dict(_INFO_JSON)
    info["upload_date"] = "20050424"
    fake_src = tmp_path / "src"
    fake_src.mkdir()
    _FakeTD = _fake_ytdlp_success(monkeypatch, fake_src, info_json=info)
    monkeypatch.setattr(tools.tempfile, "TemporaryDirectory", _FakeTD)
    monkeypatch.setattr(tools, "_run_yt_dlp", lambda a, **k: (0, "", ""))

    vault = tmp_path / "vault" / "transcripts" / "youtube"
    r = tools.youtube_video_transcript(
        "https://www.youtube.com/watch?v=jNQXAC9IVRw", timeout=10,
        out_dir=vault)
    assert r.ok, r.error
    p = Path(r.transcript_path)
    assert p.parent == vault, "deliverable must persist into out_dir"
    assert p.exists()
    assert p.name.startswith("20050424_jNQXAC9IVRw_"), (
        f"vault filename must be <date>_<id>_<title>: {p.name}")
    assert p.name.endswith(".txt")
    assert "Me_at_the_zoo" in p.name
