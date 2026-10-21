"""Unified transcription into the DS vault.

One entry point, ``transcribe_url``, driven by the engines config's
``transcriber`` block (v4 WP-F: ``argus_engines.json``, with the
``/engines transcriber`` choice persisted under the library key
``engines.transcriber``):

- **YouTube** runs the steps of ``transcriber.youtube`` in order —
  ``captions`` (caption extraction via ``tools.youtube_video_transcript``,
  no media download) and ``asr`` — until one succeeds. Only the listed
  steps run. A step that failed before the one that served is recorded in
  ``backend`` (``captions→faster_whisper:small``) and in the file header.
- **Everything else** (X / Reddit / Instagram — no caption tracks) → ASR
  over the downloaded media file (reused from the library when the media
  was already fetched, downloaded otherwise).

ASR is the one engine ``transcriber.asr_engine`` names
(``engines/transcribers.py``): ``faster_whisper`` (local) or
``openai_audio`` (a provider's ``/audio/transcriptions``). There is no
cross-engine fallback. The engine is built before any download, so a
configuration error (a missing block, an unknown provider, a missing key)
is reported without fetching the media.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from pydantic import BaseModel

from .engines import registry
from .engines.registry import EngineConfigError
from .engines.schema import EnginesConfig
from .engines.transcribers import TranscribeError, asr_engine
from .media import detect_platform, download_media

logger = logging.getLogger("argus.transcribe")


class TranscriptResult(BaseModel):
    ok: bool
    source_url: str = ""
    platform: str = ""
    path: str = ""
    text: str = ""
    language: str = ""
    title: str = ""
    # "captions" | "faster_whisper:<model>" | "openai_audio:<provider>/<model>",
    # preceded by "<failed step>→" for each YouTube step that failed first
    backend: str = ""
    error: str | None = None
    elapsed_s: float = 0.0


async def engines_config(library) -> EnginesConfig:
    """The registry's config, loaded now when startup did not load it (a
    failing load raises the real problem)."""
    try:
        return registry.current_config()
    except EngineConfigError:
        return await registry.load_config(library=library)


async def transcribe_url(
    url: str, *, transcripts_root: Path, media_root: Path,
    quality: str = "auto", library=None, engines: EnginesConfig | None = None,
) -> TranscriptResult:
    """Produce a transcript for one media URL and persist it under
    ``transcripts_root/<platform>/``. Registers vault assets in the
    library when one is provided. ``engines`` defaults to the registry's
    config. Fail-soft result: every failure is spelled out in ``error``."""
    t0 = time.monotonic()
    platform = detect_platform(url)
    if platform is None:
        return TranscriptResult(
            ok=False, source_url=url,
            error="unsupported platform — YouTube / X / Reddit / Instagram "
                  "media links only")
    out_dir = Path(transcripts_root) / platform

    def fail(error: str) -> TranscriptResult:
        return TranscriptResult(ok=False, source_url=url, platform=platform, error=error,
                                elapsed_s=time.monotonic() - t0)

    try:
        cfg = engines if engines is not None else await engines_config(library)
    except EngineConfigError as e:
        return fail(f"transcriber configuration: {e}")

    failed: list[tuple[str, str]] = []          # (step label, why) of the steps that failed
    for step in (cfg.transcriber.youtube if platform == "youtube" else ["asr"]):
        if step == "captions":
            from .tools import youtube_video_transcript
            r = await asyncio.to_thread(
                youtube_video_transcript, url, timeout=90, out_dir=out_dir)
            if r.ok:
                result = TranscriptResult(
                    ok=True, source_url=url, platform=platform,
                    path=r.transcript_path or "", text=r.transcript_text,
                    language=r.language, title=r.title,
                    backend="→".join([*(label for label, _ in failed), "captions"]),
                    elapsed_s=time.monotonic() - t0)
                await _register(library, result)
                return result
            failed.append(("captions", r.error or "captions unavailable"))
            logger.info("captions failed for %s (%s)", url, failed[-1][1])
            continue
        result = await _asr(url, platform, out_dir, cfg, failed, media_root=media_root,
                            quality=quality, library=library, t0=t0)
        if result.ok:
            await _register(library, result)
            return result
        failed.append((result.backend or "asr", result.error or "ASR failed"))

    if not failed:
        return fail("transcriber.youtube lists no steps (captions, asr)")
    error = "; ".join(why if label != "captions" else f"captions failed ({why})"
                      for label, why in failed)
    if platform == "youtube" and "asr" not in cfg.transcriber.youtube:
        error += " and transcriber.youtube lists no asr step"
    return fail(error)


async def _asr(url: str, platform: str, out_dir: Path, cfg: EnginesConfig,
               failed: list[tuple[str, str]], *, media_root: Path, quality: str,
               library, t0: float) -> TranscriptResult:
    """One ASR step: build the configured engine (before any download), get
    the media, transcribe, write the vault file. ``ok=False`` carries why."""
    def fail(error: str, backend: str = "") -> TranscriptResult:
        return TranscriptResult(ok=False, source_url=url, platform=platform, backend=backend,
                                error=error, elapsed_s=time.monotonic() - t0)

    try:
        engine = asr_engine(cfg.transcriber, cfg)
    except EngineConfigError as e:
        return fail(f"transcriber configuration: {e}")

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
            return fail(f"media download for ASR failed: {dl.error}", engine.backend)
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
        text, language = await engine.transcribe(media_path)
    except TranscribeError as e:
        logger.warning("ASR failed for %s: %s", media_path, e)
        return fail(f"ASR failed ({engine.backend}): {e}", engine.backend)
    if not text.strip():                        # an engine must raise instead; never write an empty file
        return fail(f"ASR failed ({engine.backend}): no speech in the result", engine.backend)

    backend = "→".join([*(label for label, _ in failed), engine.backend])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{media_path.stem}.txt"
    header = (f"# {title or url}\n# source: {url}\n"
              f"# backend: {backend} ({language or '?'})\n"
              + "".join(f"# {label} failed: {why}\n" for label, why in failed) + "\n")
    out_path.write_text(header + text + "\n", encoding="utf-8")
    return TranscriptResult(
        ok=True, source_url=url, platform=platform, path=str(out_path),
        text=text, language=language, title=title, backend=backend,
        elapsed_s=time.monotonic() - t0)


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
