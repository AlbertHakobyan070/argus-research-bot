"""Tests for the Phase 2 YouTube search tool + /video helpers.

Hermetic: yt-dlp is never actually invoked — we monkeypatch
``tools._run_yt_dlp`` to return canned ``--dump-json`` lines.
"""
from __future__ import annotations

from argus import tools
from argus.bot import _fmt_duration


# One JSON object per line, as `yt-dlp --dump-json --flat-playlist` emits.
_FAKE_YTDLP_OUT = "\n".join([
    '{"id": "aaa111", "title": "AI-Q Blueprint Deep Dive", '
    '"channel": "NVIDIA Developer", "duration": 605, "view_count": 12345}',
    '{"id": "bbb222", "title": "Shorts clip", "uploader": "SomeChan", '
    '"duration": 45}',
    'not-json-noise-line',
    '{"url": "https://www.youtube.com/watch?v=ccc333", '
    '"title": "Explicit URL entry", "channel": "C", "duration": 90}',
])


def test_youtube_search_parses_dump_json(monkeypatch):
    captured = {}

    def fake_run(args, **kw):
        captured["args"] = list(args)
        return (0, _FAKE_YTDLP_OUT, "")

    monkeypatch.setattr(tools, "_run_yt_dlp", fake_run)
    out = tools.youtube_search("nvidia ai-q", max_results=6)

    assert [v["url"] for v in out] == [
        "https://www.youtube.com/watch?v=aaa111",   # built from id
        "https://www.youtube.com/watch?v=bbb222",
        "https://www.youtube.com/watch?v=ccc333",   # explicit url kept
    ]
    assert out[0]["title"] == "AI-Q Blueprint Deep Dive"
    assert out[0]["channel"] == "NVIDIA Developer"
    assert out[0]["duration"] == 605
    assert out[1]["channel"] == "SomeChan"       # uploader fallback
    assert all(v["source"] == "youtube" for v in out)
    # The search term is the ytsearchN: spec passed to yt-dlp.
    assert captured["args"][0] == "ytsearch6:nvidia ai-q"


def test_youtube_search_shorts_biases_query(monkeypatch):
    captured = {}

    def fake_run(args, **kw):
        captured["args"] = list(args)
        return (0, "", "")

    monkeypatch.setattr(tools, "_run_yt_dlp", fake_run)
    tools.youtube_search("langgraph", max_results=3, shorts=True)
    assert captured["args"][0] == "ytsearch3:langgraph #shorts"


def test_youtube_search_empty_query_returns_empty(monkeypatch):
    monkeypatch.setattr(tools, "_run_yt_dlp",
                        lambda a, **k: (_ for _ in ()).throw(
                            AssertionError("should not shell out")))
    assert tools.youtube_search("   ") == []


def test_youtube_search_failure_is_soft(monkeypatch):
    monkeypatch.setattr(tools, "_run_yt_dlp", lambda a, **k: (1, "", "boom"))
    assert tools.youtube_search("x") == []


def test_youtube_search_caps_results(monkeypatch):
    monkeypatch.setattr(tools, "_run_yt_dlp",
                        lambda a, **k: (0, _FAKE_YTDLP_OUT, ""))
    assert len(tools.youtube_search("x", max_results=2)) == 2


def test_fmt_duration():
    assert _fmt_duration(45) == "0:45"
    assert _fmt_duration(605) == "10:05"
    assert _fmt_duration(3661) == "1:01:01"
    assert _fmt_duration(None) == ""
    assert _fmt_duration(0) == ""
    assert _fmt_duration("bad") == ""
