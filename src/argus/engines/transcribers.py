"""Speech-to-text engines (v4 WP-F, spec §3 WP-F).

``asr_engine(transcriber, cfg)`` builds the engine ``transcriber.asr_engine``
names from its block in ``argus_engines.json``:

- ``faster_whisper`` — ``{"model", "device", "compute_type"}`` → a local
  ``faster_whisper.WhisperModel`` (one loaded per process, reloaded when the
  block changes); ``backend = "faster_whisper:<model>"``.
- ``openai_audio`` — ``{"provider", "model", "chunk_seconds", "language"}`` →
  the provider's OpenAI-compatible ``{base_url}/audio/transcriptions``
  (multipart), over 16 kHz mono MP3 chunks of ``chunk_seconds`` whose texts
  are joined in order; ``backend = "openai_audio:<provider>/<model>"``.

The block is read at use time, never at startup: a missing or bad block, an
unknown provider or a missing key raises ``EngineConfigError`` naming it. A
failed request, a chunk over the upload limit or an all-silent result raises
``TranscribeError``. No engine falls back to another.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

import httpx

from ..media import extract_audio_chunks
from . import registry
from .registry import EngineConfigError
from .schema import EnginesConfig, TranscriberConfig

logger = logging.getLogger("argus.transcribe")

# The upload cap of the OpenAI and Groq transcription endpoints. Chunks are
# far below it at the shipped chunk_seconds (48 kbit/s MP3).
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
_FW_KEYS = ("model", "device", "compute_type")
_OA_KEYS = ("provider", "model", "chunk_seconds", "language")

_ASR_LOCK = threading.Lock()
_ASR_MODEL: tuple[tuple, Any] | None = None     # ((model, device, compute_type), WhisperModel)


class TranscribeError(RuntimeError):
    """A transcription that ran and failed: HTTP, size, unusable answer, no speech."""


def asr_available() -> bool:
    """Whether faster-whisper is installed, without importing it (the import
    costs seconds, and this runs on the event loop)."""
    return importlib.util.find_spec("faster_whisper") is not None


def _block(t: TranscriberConfig, name: str, keys: tuple[str, ...]) -> dict:
    block = getattr(t, name)
    missing = [k for k in keys if k not in block]
    if missing:
        raise EngineConfigError(f"transcriber.{name} is missing {', '.join(missing)} "
                                f"(the transcriber block in argus_engines.json)")
    return block


def asr_label(t: TranscriberConfig) -> str:
    """The configured ASR engine as its backend string, without building it
    ("?" for a key the block lacks): for display only."""
    if t.asr_engine == "faster_whisper":
        return f"faster_whisper:{t.faster_whisper.get('model', '?')}"
    return f"openai_audio:{t.openai_audio.get('provider', '?')}/{t.openai_audio.get('model', '?')}"


def asr_engine(t: TranscriberConfig, cfg: EnginesConfig) -> FasterWhisper | OpenAIAudio:
    if t.asr_engine == "faster_whisper":
        return FasterWhisper(_block(t, "faster_whisper", _FW_KEYS))
    return OpenAIAudio(_block(t, "openai_audio", _OA_KEYS), cfg)


def _no_speech(backend: str) -> TranscribeError:
    return TranscribeError(f"{backend} produced no speech (silent or music-only media?)")


class FasterWhisper:
    def __init__(self, block: dict):
        if not asr_available():
            raise EngineConfigError("speech-to-text needs faster-whisper — install it with "
                                    "`uv pip install faster-whisper` (the [asr] extra)")
        self.key = tuple(block[k] for k in _FW_KEYS)
        self.backend = f"faster_whisper:{block['model']}"

    async def transcribe(self, media_path: Path, *, language: str | None = None,
                         timestamps: bool = False) -> tuple[str, str]:
        try:
            text, language = await asyncio.to_thread(_whisper_file, Path(media_path), self.key,
                                                     language or None, timestamps)
        except Exception as e:                  # model load / decode failures: reported, with the cause chained
            logger.exception("faster-whisper failed on %s", media_path)
            raise TranscribeError(f"{type(e).__name__}: {e}") from e
        if not text.strip():
            raise _no_speech(self.backend)
        return text, language


def _whisper_model(key: tuple):
    """One WhisperModel per process (thread-safe); a changed block replaces
    it. The first load downloads the weights (hundreds of MB)."""
    global _ASR_MODEL
    with _ASR_LOCK:
        if _ASR_MODEL is None or _ASR_MODEL[0] != key:
            from faster_whisper import WhisperModel
            model, device, compute_type = key
            logger.info("loading faster-whisper model %r (%s/%s)…", model, device, compute_type)
            _ASR_MODEL = (key, WhisperModel(model, device=device, compute_type=compute_type))
        return _ASR_MODEL[1]


def stamp(seconds: float) -> str:
    """``[m:ss]`` (``[h:mm:ss]`` past an hour) for a timestamped transcript line."""
    s = int(seconds)
    h, m = divmod(s // 60, 60)
    return f"[{h}:{m:02d}:{s % 60:02d}]" if h else f"[{m}:{s % 60:02d}]"


def _whisper_file(media_path: Path, key: tuple, language: str | None = None,
                  timestamps: bool = False) -> tuple[str, str]:
    """Blocking ASR over one media file → (text, language). Whisper segments
    can overlap-repeat like rolling captions, so the same dedup applies (to
    plain text only: a timestamped line is kept with its time)."""
    from ..tools import _dedup_rolling_captions
    segments, info = _whisper_model(key).transcribe(str(media_path), vad_filter=True,
                                                    **({"language": language} if language else {}))
    if timestamps:
        lines = [f"{stamp(s.start)} {s.text.strip()}" for s in segments if s.text.strip()]
    else:
        lines = _dedup_rolling_captions([s.text.strip() for s in segments if s.text.strip()])
    return "\n".join(lines), (getattr(info, "language", "") or "")


def _client(timeout_s: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout_s)


class OpenAIAudio:
    def __init__(self, block: dict, cfg: EnginesConfig):
        name = block["provider"]
        if name not in cfg.providers:
            raise EngineConfigError(f"transcriber.openai_audio: provider {name!r} is not configured "
                                    f"(known: {', '.join(cfg.providers)})")
        chunk = block["chunk_seconds"]
        if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
            raise EngineConfigError(f"transcriber.openai_audio.chunk_seconds must be a positive integer, got {chunk!r}")
        self.prov = cfg.providers[name]
        self._key = registry.api_key(self.prov)          # raises, naming the env var, before any download
        self.url = registry.base_url(self.prov).rstrip("/") + "/audio/transcriptions"
        self.model, self.chunk_seconds, self.language = block["model"], chunk, block["language"]
        self.backend = f"openai_audio:{name}/{self.model}"

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key != "none" else text

    async def transcribe(self, media_path: Path, *, language: str | None = None,
                         timestamps: bool = False) -> tuple[str, str]:
        language = language or self.language
        with tempfile.TemporaryDirectory(prefix="argus_audio_") as tmp:
            try:
                chunks = await extract_audio_chunks(Path(media_path), Path(tmp),
                                                    chunk_seconds=self.chunk_seconds)
            except RuntimeError as e:
                raise TranscribeError(f"{self.backend}: audio extraction failed: {e}") from e
            n = len(chunks)
            for i, c in enumerate(chunks, 1):          # every size checked before the first upload
                size = c.stat().st_size
                if size > MAX_UPLOAD_BYTES:
                    raise TranscribeError(f"{self.backend}: chunk {i}/{n} is {size} bytes, over the "
                                          f"{MAX_UPLOAD_BYTES}-byte upload limit; lower "
                                          f"transcriber.openai_audio.chunk_seconds")
            texts = []
            async with _client(self.prov.timeout_s) as client:
                for i, c in enumerate(chunks, 1):
                    texts.append(await self._one(client, c, f"chunk {i}/{n}", language=language,
                                                 offset_s=(i - 1) * self.chunk_seconds if timestamps else None))
        text = "\n".join(t for t in texts if t)
        if not text:
            raise _no_speech(self.backend)
        return text, language or ""

    async def _one(self, client: httpx.AsyncClient, chunk: Path, where: str, *, language: str | None = None,
                   offset_s: float | None = None) -> str:
        """One chunk's text; with ``offset_s`` (timestamps asked) the provider's
        ``verbose_json`` segments as ``[m:ss] text`` lines, shifted by the
        chunk's start. A provider without verbose_json answers with an HTTP
        error, raised as such."""
        data = {"model": self.model, "response_format": "json" if offset_s is None else "verbose_json"}
        if language:
            data["language"] = language
        headers = {**self.prov.headers, "Authorization": f"Bearer {self._key}"}
        try:
            r = await client.post(self.url, headers=headers, data=data,
                                  files={"file": (chunk.name, chunk.read_bytes(), "audio/mpeg")})
        except httpx.HTTPError as e:
            failure = self._redact(f"{self.backend}: {where}: {type(e).__name__}: {e}")
        else:
            failure = None
        if failure is not None:                 # raised outside the handler: no __context__ holds the key
            raise TranscribeError(failure)
        if r.status_code >= 400:
            hint = "; lower transcriber.openai_audio.chunk_seconds" if r.status_code == 413 else ""
            raise TranscribeError(self._redact(f"{self.backend}: {where}: HTTP {r.status_code} "
                                               f"{r.text.strip()[:300]}{hint}"))
        try:
            body = r.json()
        except ValueError:
            body = None
        if offset_s is not None and isinstance(body, dict) and isinstance(body.get("segments"), list):
            return "\n".join(f"{stamp(offset_s + float(s.get('start') or 0))} {str(s.get('text') or '').strip()}"
                             for s in body["segments"] if isinstance(s, dict) and str(s.get("text") or "").strip())
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str):
            raise TranscribeError(self._redact(f"{self.backend}: {where}: unusable answer {r.text.strip()[:300]!r}"))
        return text.strip()
