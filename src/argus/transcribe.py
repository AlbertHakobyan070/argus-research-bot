"""Unified transcription into the DS vault.

Phase 3 of the v2 rebuild. One entry point, ``transcribe_url``:

- **YouTube** → caption extraction (fast, no media download) via
  ``tools.youtube_video_transcript`` with the vault ``out_dir``.
- **Everything else** (X / Reddit / Instagram — no caption tracks), or
  YouTube videos without captions → local **faster-whisper** ASR over
  the downloaded media file (reused from the library when the media was
  already fetched, downloaded otherwise).

Whisper runs on CPU (int8, model size via ARGUS_WHISPER_MODEL, default
"small"; first use downloads the model weights). Decoding happens
through PyAV, so no system ffmpeg is required for ASR itself.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .media import detect_platform, download_media

logger = logging.getLogger("argus.transcribe")

_ASR_LOCK = threading.Lock()
_ASR_MODEL: Any | None = None


class TranscriptResult(BaseModel):
    ok: bool
    source_url: str = ""
    platform: str = ""
    path: str = ""
    text: str = ""
    language: str = ""
    title: str = ""
    backend: str = ""          # "captions" | "whisper"
    error: str | None = None
    elapsed_s: float = 0.0


def asr_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _get_whisper_model():
    """Lazily load ONE WhisperModel per process (thread-safe). The first
    call downloads the model weights (hundreds of MB) — callers should
    tell the user that may take a while."""
    global _ASR_MODEL
    with _ASR_LOCK:
        if _ASR_MODEL is None:
            from faster_whisper import WhisperModel
            size = os.environ.get("ARGUS_WHISPER_MODEL", "small")
            logger.info("loading faster-whisper model %r (cpu/int8)…", size)
            _ASR_MODEL = WhisperModel(size, device="cpu",
                                      compute_type="int8")
        return _ASR_MODEL


def _whisper_transcribe_file(media_path: Path) -> tuple[str, str]:
    """Blocking ASR over one media file → (text, language). Runs inside
    ``asyncio.to_thread`` from the async entry point.

    Whisper segments can overlap-repeat just like YouTube rolling
    captions (observed live on short clips), so the same word-overlap
    dedup is applied to the segment stream."""
    from .tools import _dedup_rolling_captions
    model = _get_whisper_model()
    segments, info = model.transcribe(str(media_path), vad_filter=True)
    lines = [seg.text.strip() for seg in segments if seg.text.strip()]
    lines = _dedup_rolling_captions(lines)
    return "\n".join(lines), (getattr(info, "language", "") or "")


async def transcribe_url(
    url: str, *, transcripts_root: Path, media_root: Path,
    quality: str = "auto", library=None,
) -> TranscriptResult:
    """Produce a transcript for one media URL and persist it under
    ``transcripts_root/<platform>/``. Registers vault assets in the
    library when one is provided. Fail-soft result."""
    t0 = time.monotonic()
    platform = detect_platform(url)
    if platform is None:
        return TranscriptResult(
            ok=False, source_url=url,
            error="unsupported platform — YouTube / X / Reddit / Instagram "
                  "media links only")
    out_dir = Path(transcripts_root) / platform

    caption_err = ""
    if platform == "youtube":
        from .tools import youtube_video_transcript
        r = await asyncio.to_thread(
            youtube_video_transcript, url, timeout=90, out_dir=out_dir)
        if r.ok:
            result = TranscriptResult(
                ok=True, source_url=url, platform=platform,
                path=r.transcript_path or "", text=r.transcript_text,
                language=r.language, title=r.title, backend="captions",
                elapsed_s=time.monotonic() - t0)
            await _register(library, result)
            return result
        caption_err = r.error or "captions unavailable"
        logger.info("captions failed for %s (%s); trying ASR", url,
                    caption_err)

    # --- ASR path -----------------------------------------------------------
    if not asr_available():
        note = ("speech-to-text needs faster-whisper — install it with "
                "`uv pip install faster-whisper` (the [asr] extra)")
        if caption_err:
            note = f"captions failed ({caption_err}) and {note}"
        return TranscriptResult(ok=False, source_url=url, platform=platform,
                                error=note,
                                elapsed_s=time.monotonic() - t0)

    # Reuse already-downloaded media when the library knows about it.
    media_path: Path | None = None
    title = ""
    if library is not None:
        try:
            existing = await library.get_asset_by_source(url, "media")
            if existing and Path(existing["path"]).exists():
                media_path = Path(existing["path"])
                title = existing.get("title") or ""
        except Exception:
            logger.exception("library lookup failed; downloading fresh")

    if media_path is None:
        dl = await download_media(url, dest_root=media_root, quality=quality)
        if not dl.ok:
            return TranscriptResult(
                ok=False, source_url=url, platform=platform,
                error=f"media download for ASR failed: {dl.error}",
                elapsed_s=time.monotonic() - t0)
        media_path = Path(dl.path)
        title = dl.title
        if library is not None:
            try:
                await library.add_asset(
                    kind="media", platform=platform, source_url=url,
                    media_id=dl.media_id, title=title or url, path=dl.path,
                    size_bytes=dl.size_bytes, duration_s=dl.duration_s,
                    meta={"via": "transcribe"})
            except Exception:
                logger.exception("media asset registration failed")

    try:
        text, language = await asyncio.to_thread(
            _whisper_transcribe_file, media_path)
    except Exception as e:
        logger.exception("whisper ASR failed for %s", media_path)
        return TranscriptResult(
            ok=False, source_url=url, platform=platform,
            error=f"ASR failed: {e}", elapsed_s=time.monotonic() - t0)
    if not text.strip():
        return TranscriptResult(
            ok=False, source_url=url, platform=platform,
            error="ASR produced no speech (silent or music-only video?)",
            elapsed_s=time.monotonic() - t0)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{media_path.stem}.txt"
    header = (f"# {title or url}\n# source: {url}\n"
              f"# backend: whisper ({language or '?'})\n\n")
    out_path.write_text(header + text + "\n", encoding="utf-8")

    result = TranscriptResult(
        ok=True, source_url=url, platform=platform, path=str(out_path),
        text=text, language=language, title=title, backend="whisper",
        elapsed_s=time.monotonic() - t0)
    await _register(library, result)
    return result


async def _register(library, result: TranscriptResult) -> None:
    if library is None or not result.ok or not result.path:
        return
    try:
        p = Path(result.path)
        await library.add_asset(
            kind="transcript", platform=result.platform,
            source_url=result.source_url, title=result.title or None,
            path=result.path,
            size_bytes=p.stat().st_size if p.exists() else 0,
            meta={"backend": result.backend, "language": result.language})
    except Exception:
        logger.exception("transcript asset registration failed")
