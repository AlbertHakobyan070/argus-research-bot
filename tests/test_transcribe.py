"""Unified transcription tests — captions vs whisper routing, vault
persistence, library registration. Hermetic (whisper + downloads mocked)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from argus import transcribe as tr_mod
from argus.library import Library
from argus.media import MediaResult
from argus.transcribe import transcribe_url


@pytest.fixture
async def lib(tmp_path):
    l = Library(tmp_path / "lib.sqlite")
    await l.open()
    yield l
    await l.close()


def _vault(tmp_path) -> tuple[Path, Path]:
    return tmp_path / "vault" / "transcripts", tmp_path / "vault" / "media"


async def test_unsupported_url_is_rejected(tmp_path):
    t_root, m_root = _vault(tmp_path)
    r = await transcribe_url("https://example.com/x",
                             transcripts_root=t_root, media_root=m_root)
    assert r.ok is False and "platform" in (r.error or "")


async def test_youtube_routes_via_captions_and_registers(tmp_path, lib,
                                                         monkeypatch):
    t_root, m_root = _vault(tmp_path)
    saved = t_root / "youtube" / "20250101_vid1_Title.txt"

    def fake_captions(url, *, timeout=90, out_dir=None, **kw):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        saved.write_text("caption text", encoding="utf-8")
        return SimpleNamespace(ok=True, transcript_path=str(saved),
                               transcript_text="caption text",
                               language="en", title="Title", error=None,
                               duration=19)

    from argus import tools as tools_mod
    monkeypatch.setattr(tools_mod, "youtube_video_transcript", fake_captions)

    r = await transcribe_url("https://youtu.be/abc123DEF45",
                             transcripts_root=t_root, media_root=m_root,
                             library=lib)
    assert r.ok and r.backend == "captions"
    assert Path(r.path) == saved
    assets = await lib.list_assets(kind="transcript")
    assert len(assets) == 1
    assert assets[0]["meta"]["backend"] == "captions"


async def test_non_youtube_downloads_then_whispers(tmp_path, lib, monkeypatch):
    t_root, m_root = _vault(tmp_path)
    media_file = m_root / "x" / "20250101_id9_clip.mp4"

    async def fake_download(url, *, dest_root, quality="auto", **kw):
        media_file.parent.mkdir(parents=True, exist_ok=True)
        media_file.write_bytes(b"v" * 100)
        return MediaResult(ok=True, url=url, platform="x",
                           path=str(media_file), title="An X clip",
                           media_id="id9", size_bytes=100, duration_s=15.0)

    monkeypatch.setattr(tr_mod, "download_media", fake_download)
    monkeypatch.setattr(tr_mod, "asr_available", lambda: True)
    monkeypatch.setattr(tr_mod, "_whisper_transcribe_file",
                        lambda p: ("hello from whisper", "en"))

    r = await transcribe_url("https://x.com/u/status/9",
                             transcripts_root=t_root, media_root=m_root,
                             library=lib)
    assert r.ok and r.backend == "whisper"
    p = Path(r.path)
    assert p.parent == t_root / "x"
    assert p.stem == media_file.stem, (
        "transcript name must mirror the media file's vault name")
    body = p.read_text(encoding="utf-8")
    assert "hello from whisper" in body
    assert "# source: https://x.com/u/status/9" in body
    # Both the media file AND the transcript got registered.
    kinds = {a["kind"] for a in
             (await lib.list_assets(kind="media"))
             + (await lib.list_assets(kind="transcript"))}
    assert kinds == {"media", "transcript"}


async def test_whisper_reuses_already_downloaded_media(tmp_path, lib,
                                                       monkeypatch):
    t_root, m_root = _vault(tmp_path)
    media_file = m_root / "reddit" / "20250101_r1_post.mp4"
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"v" * 50)
    url = "https://www.reddit.com/r/v/comments/r1/post/"
    await lib.add_asset(kind="media", platform="reddit", source_url=url,
                        title="Post", path=str(media_file), size_bytes=50)

    async def boom_download(*a, **kw):
        raise AssertionError("must reuse the library media, not re-download")

    monkeypatch.setattr(tr_mod, "download_media", boom_download)
    monkeypatch.setattr(tr_mod, "asr_available", lambda: True)
    monkeypatch.setattr(tr_mod, "_whisper_transcribe_file",
                        lambda p: ("reddit words", "en"))

    r = await transcribe_url(url, transcripts_root=t_root, media_root=m_root,
                             library=lib)
    assert r.ok and r.backend == "whisper"
    assert "reddit words" in Path(r.path).read_text(encoding="utf-8")


async def test_asr_missing_is_an_honest_error(tmp_path, monkeypatch):
    t_root, m_root = _vault(tmp_path)
    monkeypatch.setattr(tr_mod, "asr_available", lambda: False)
    r = await transcribe_url("https://x.com/u/status/9",
                             transcripts_root=t_root, media_root=m_root)
    assert r.ok is False
    assert "faster-whisper" in (r.error or "")


async def test_silent_video_reports_no_speech(tmp_path, monkeypatch):
    t_root, m_root = _vault(tmp_path)

    async def fake_download(url, *, dest_root, quality="auto", **kw):
        f = m_root / "x" / "clip.mp4"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"v")
        return MediaResult(ok=True, url=url, platform="x", path=str(f),
                           size_bytes=1)

    monkeypatch.setattr(tr_mod, "download_media", fake_download)
    monkeypatch.setattr(tr_mod, "asr_available", lambda: True)
    monkeypatch.setattr(tr_mod, "_whisper_transcribe_file",
                        lambda p: ("", ""))
    r = await transcribe_url("https://x.com/u/status/9",
                             transcripts_root=t_root, media_root=m_root)
    assert r.ok is False and "no speech" in (r.error or "")
