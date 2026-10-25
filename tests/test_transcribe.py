"""Unified transcription tests — the captions→ASR order from the engines
config, the ASR engine it selects, vault persistence, library registration,
and the error surfaces. Hermetic (the engines and downloads are faked)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from argus import transcribe as tr_mod
from argus.engines import registry
from argus.engines.registry import EngineConfigError
from argus.engines.schema import EnginesConfig
from argus.engines.transcribers import TranscribeError
from argus.library import Library
from argus.media import MediaResult
from argus.transcribe import transcribe_url

BLOCK = {"faster_whisper": {"model": "small", "device": "cpu", "compute_type": "int8"},
         "openai_audio": {"provider": "aud", "model": "w1", "chunk_seconds": 600, "language": None}}


def _cfg(**transcriber) -> EnginesConfig:
    tiers = {t: {"provider": "aud", "model": "m"} for t in ("cheap", "strong", "judge")}
    return EnginesConfig.model_validate({
        "active_profile": "p",
        "providers": {"aud": {"name": "aud", "base_url": "http://aud.test/v1", "api_key_env": "T_AUD_KEY"}},
        "profiles": {"p": {"name": "p", "tiers": tiers}},
        "transcriber": {**BLOCK, **transcriber}})


@pytest.fixture
async def lib(tmp_path):
    l = Library(tmp_path / "lib.sqlite")
    await l.open()
    yield l
    await l.close()


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(registry, "_state", registry._State())


def _vault(tmp_path) -> tuple[Path, Path]:
    return tmp_path / "vault" / "transcripts", tmp_path / "vault" / "media"


class FakeEngine:
    def __init__(self, backend: str, text: str = "hello from asr", error: Exception | None = None):
        self.backend, self.text, self.error, self.paths = backend, text, error, []

    async def transcribe(self, path: Path) -> tuple[str, str]:
        self.paths.append(path)
        if self.error:
            raise self.error
        return self.text, "en"


def _fake_asr(monkeypatch, engine: FakeEngine | None = None, error: Exception | None = None) -> list:
    seen = []

    def fake(t, cfg):
        seen.append(t)
        if error:
            raise error
        return engine or FakeEngine(f"{t.asr_engine}:x")
    monkeypatch.setattr(tr_mod, "asr_engine", fake)
    return seen


def _fake_download(monkeypatch, m_root: Path, calls: list | None = None):
    media_file = m_root / "x" / "20250101_id9_clip.mp4"

    async def fake(url, *, dest_root, quality="auto", **kw):
        if calls is not None:
            calls.append(url)
        media_file.parent.mkdir(parents=True, exist_ok=True)
        media_file.write_bytes(b"v" * 100)
        return MediaResult(ok=True, url=url, platform="x", path=str(media_file), title="An X clip",
                           media_id="id9", size_bytes=100, duration_s=15.0)
    monkeypatch.setattr(tr_mod, "download_media", fake)
    return media_file


def _fake_captions(monkeypatch, t_root: Path, ok: bool = True, calls: list | None = None):
    saved = t_root / "youtube" / "20250101_vid1_Title.txt"

    def fake(url, *, timeout=90, out_dir=None, **kw):
        if calls is not None:
            calls.append(url)
        if not ok:
            return SimpleNamespace(ok=False, error="no caption tracks", transcript_path=None,
                                   transcript_text="", language="", title="")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        saved.write_text("caption text", encoding="utf-8")
        return SimpleNamespace(ok=True, transcript_path=str(saved), transcript_text="caption text",
                               language="en", title="Title", error=None, duration=19)
    from argus import tools as tools_mod
    monkeypatch.setattr(tools_mod, "youtube_video_transcript", fake)
    return saved


async def test_unsupported_url_is_rejected(tmp_path):
    t_root, m_root = _vault(tmp_path)
    r = await transcribe_url("https://example.com/x", transcripts_root=t_root, media_root=m_root)
    assert r.ok is False and "platform" in (r.error or "")


async def test_youtube_routes_via_captions_and_registers(tmp_path, lib, monkeypatch):
    t_root, m_root = _vault(tmp_path)
    saved = _fake_captions(monkeypatch, t_root)
    seen = _fake_asr(monkeypatch)
    r = await transcribe_url("https://youtu.be/abc123DEF45", transcripts_root=t_root, media_root=m_root,
                             library=lib, engines=_cfg())
    assert r.ok and r.backend == "captions"
    assert Path(r.path) == saved
    assert seen == [], "captions served: no ASR engine is built"
    assets = await lib.list_assets(kind="transcript")
    assert len(assets) == 1 and assets[0]["meta"]["backend"] == "captions"


async def test_non_youtube_downloads_then_transcribes_with_the_configured_engine(tmp_path, lib, monkeypatch):
    t_root, m_root = _vault(tmp_path)
    media_file = _fake_download(monkeypatch, m_root)
    eng = FakeEngine("faster_whisper:small")
    seen = _fake_asr(monkeypatch, eng)
    r = await transcribe_url("https://x.com/u/status/9", transcripts_root=t_root, media_root=m_root,
                             library=lib, engines=_cfg())
    assert r.ok and r.backend == "faster_whisper:small"
    assert seen[0].asr_engine == "faster_whisper" and eng.paths == [media_file]
    p = Path(r.path)
    assert p.parent == t_root / "x" and p.stem == media_file.stem
    body = p.read_text(encoding="utf-8")
    assert "hello from asr" in body
    assert "# source: https://x.com/u/status/9" in body
    assert "# backend: faster_whisper:small (en)" in body
    kinds = {a["kind"] for a in (await lib.list_assets(kind="media")) + (await lib.list_assets(kind="transcript"))}
    assert kinds == {"media", "transcript"}
    assert (await lib.list_assets(kind="transcript"))[0]["meta"]["backend"] == "faster_whisper:small"


async def test_asr_reuses_already_downloaded_media(tmp_path, lib, monkeypatch):
    t_root, m_root = _vault(tmp_path)
    media_file = m_root / "reddit" / "20250101_r1_post.mp4"
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"v" * 50)
    url = "https://www.reddit.com/r/v/comments/r1/post/"
    await lib.add_asset(kind="media", platform="reddit", source_url=url, title="Post",
                        path=str(media_file), size_bytes=50)

    async def boom_download(*a, **kw):
        raise AssertionError("must reuse the library media, not re-download")
    monkeypatch.setattr(tr_mod, "download_media", boom_download)
    _fake_asr(monkeypatch, FakeEngine("faster_whisper:small", "reddit words"))
    r = await transcribe_url(url, transcripts_root=t_root, media_root=m_root, library=lib, engines=_cfg())
    assert r.ok and "reddit words" in Path(r.path).read_text(encoding="utf-8")


async def test_youtube_falls_back_to_asr_in_config_order_and_records_it(tmp_path, monkeypatch):
    t_root, m_root = _vault(tmp_path)
    _fake_captions(monkeypatch, t_root, ok=False)
    _fake_download(monkeypatch, m_root)
    _fake_asr(monkeypatch, FakeEngine("openai_audio:aud/w1"))
    r = await transcribe_url("https://youtu.be/abc123DEF45", transcripts_root=t_root, media_root=m_root,
                             engines=_cfg(asr_engine="openai_audio"))
    assert r.ok and r.backend == "captions→openai_audio:aud/w1"
    body = Path(r.path).read_text(encoding="utf-8")
    assert "# backend: captions→openai_audio:aud/w1 (en)" in body
    assert "# captions failed: no caption tracks" in body


async def test_youtube_asr_only_skips_captions(tmp_path, monkeypatch):
    t_root, m_root = _vault(tmp_path)
    caption_calls: list = []
    _fake_captions(monkeypatch, t_root, calls=caption_calls)
    _fake_download(monkeypatch, m_root)
    _fake_asr(monkeypatch, FakeEngine("faster_whisper:small"))
    r = await transcribe_url("https://youtu.be/abc123DEF45", transcripts_root=t_root, media_root=m_root,
                             engines=_cfg(youtube=["asr"]))
    assert r.ok and r.backend == "faster_whisper:small" and caption_calls == []


async def test_youtube_captions_only_never_falls_back(tmp_path, monkeypatch):
    t_root, m_root = _vault(tmp_path)
    _fake_captions(monkeypatch, t_root, ok=False)
    downloads: list = []
    _fake_download(monkeypatch, m_root, downloads)
    seen = _fake_asr(monkeypatch)
    r = await transcribe_url("https://youtu.be/abc123DEF45", transcripts_root=t_root, media_root=m_root,
                             engines=_cfg(youtube=["captions"]))
    assert r.ok is False and "no caption tracks" in r.error and "transcriber.youtube" in r.error
    assert downloads == [] and seen == []


async def test_asr_config_error_surfaces_before_any_download(tmp_path, monkeypatch):
    """The real engine, the openai_audio block and no key: the explicit
    configuration error names the env var, and nothing is downloaded."""
    monkeypatch.delenv("T_AUD_KEY", raising=False)
    t_root, m_root = _vault(tmp_path)
    downloads: list = []
    _fake_download(monkeypatch, m_root, downloads)
    r = await transcribe_url("https://x.com/u/status/9", transcripts_root=t_root, media_root=m_root,
                             engines=_cfg(asr_engine="openai_audio"))
    assert r.ok is False and "T_AUD_KEY is not set" in r.error and "transcriber" in r.error
    assert downloads == []


async def test_youtube_config_error_keeps_the_caption_failure(tmp_path, monkeypatch):
    t_root, m_root = _vault(tmp_path)
    _fake_captions(monkeypatch, t_root, ok=False)
    _fake_asr(monkeypatch, error=EngineConfigError("transcriber.faster_whisper is missing model"))
    r = await transcribe_url("https://youtu.be/abc123DEF45", transcripts_root=t_root, media_root=m_root,
                             engines=_cfg())
    assert r.ok is False and "no caption tracks" in r.error and "missing model" in r.error


async def test_transcribe_error_is_reported_not_raised(tmp_path, monkeypatch):
    t_root, m_root = _vault(tmp_path)
    _fake_download(monkeypatch, m_root)
    _fake_asr(monkeypatch, FakeEngine("openai_audio:aud/w1", error=TranscribeError("chunk 1/1: HTTP 500")))
    r = await transcribe_url("https://x.com/u/status/9", transcripts_root=t_root, media_root=m_root,
                             engines=_cfg(asr_engine="openai_audio"))
    assert r.ok is False and r.error == "ASR failed (openai_audio:aud/w1): chunk 1/1: HTTP 500"
    assert not (t_root / "x").exists(), "no transcript file on failure"


async def test_no_speech_is_never_a_silent_empty_transcript(tmp_path, monkeypatch):
    t_root, m_root = _vault(tmp_path)
    _fake_download(monkeypatch, m_root)
    _fake_asr(monkeypatch, FakeEngine("faster_whisper:small", text="  \n "))
    r = await transcribe_url("https://x.com/u/status/9", transcripts_root=t_root, media_root=m_root,
                             engines=_cfg())
    assert r.ok is False and "no speech" in r.error


async def test_without_engines_it_reads_the_registry_and_the_persisted_choice(tmp_path, lib, monkeypatch):
    """The bot's call (no ``engines=``): the registry's config, including a
    ``/engines transcriber`` choice persisted in the library."""
    p = tmp_path / "e.json"
    p.write_text(json.dumps(_cfg().model_dump()), encoding="utf-8")
    await registry.load_config([str(p)], library=lib)
    await registry.set_transcriber(registry.current_config().transcriber.model_copy(
        update={"asr_engine": "openai_audio"}))
    t_root, m_root = _vault(tmp_path)
    _fake_download(monkeypatch, m_root)
    seen = _fake_asr(monkeypatch)
    r = await transcribe_url("https://x.com/u/status/9", transcripts_root=t_root, media_root=m_root,
                             library=lib)
    assert r.ok and seen[0].asr_engine == "openai_audio"


@pytest.mark.parametrize("url, transcriber, expect, absent", [
    ("https://x.com/u/status/9", {}, ["faster_whisper:small", "downloads the model"], ["captions"]),
    ("https://x.com/u/status/9", {"asr_engine": "openai_audio"}, ["openai_audio:aud/w1"], ["downloads the model"]),
    ("https://youtu.be/abc123DEF45", {}, ["captions → faster_whisper:small"], []),
    ("https://youtu.be/abc123DEF45", {"youtube": ["captions"]}, ["captions"], ["faster_whisper", "speech-to-text"]),
])
async def test_bot_status_note_names_the_configured_backend(tmp_path, monkeypatch, url, transcriber, expect, absent):
    """Review Minor 6: the bot's "Transcribing…" note names what will run,
    not a hardcoded "whisper"."""
    from unittest.mock import AsyncMock, MagicMock
    from argus.bot import media as bot_media
    cfg = _cfg(**transcriber)

    async def engines_config(library):
        return cfg
    monkeypatch.setattr(bot_media, "engines_config", engines_config)
    monkeypatch.setattr(bot_media, "get_settings", lambda: SimpleNamespace(
        transcripts_root=tmp_path / "t", media_root=tmp_path / "m"))
    monkeypatch.setattr(bot_media, "transcribe_url", AsyncMock(return_value=tr_mod.TranscriptResult(
        ok=False, error="stop here")))
    app = MagicMock()
    app.bot.send_message = AsyncMock()
    app.bot_data = {}
    await bot_media._transcript_for_item(app, 1, {"url": url})
    note = app.bot.send_message.await_args_list[0].kwargs["text"]
    assert "whisper (" not in note and "via whisper" not in note
    for want in expect:
        assert want in note, note
    for no in absent:
        assert no not in note, note


async def test_bot_status_note_with_a_broken_config_still_lets_the_error_through(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from argus.bot import media as bot_media

    async def engines_config(library):
        raise EngineConfigError("engines file not found: x.json")
    monkeypatch.setattr(bot_media, "engines_config", engines_config)
    monkeypatch.setattr(bot_media, "get_settings", lambda: SimpleNamespace(
        transcripts_root=tmp_path / "t", media_root=tmp_path / "m"))
    monkeypatch.setattr(bot_media, "transcribe_url", AsyncMock(return_value=tr_mod.TranscriptResult(
        ok=False, error="transcriber configuration: engines file not found: x.json")))
    app = MagicMock()
    app.bot.send_message = AsyncMock()
    app.bot_data = {}
    await bot_media._transcript_for_item(app, 1, {"url": "https://x.com/u/status/9"})
    texts = [c.kwargs["text"] for c in app.bot.send_message.await_args_list]
    assert texts[0] == "🎙 Transcribing…"
    assert "engines file not found" in texts[-1]


async def test_without_a_loadable_config_the_error_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("ARGUS_ENGINES_FILE", str(tmp_path / "missing.json"))
    t_root, m_root = _vault(tmp_path)
    r = await transcribe_url("https://x.com/u/status/9", transcripts_root=t_root, media_root=m_root)
    assert r.ok is False and "engines file not found" in r.error
