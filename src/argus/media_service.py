"""Media library actions shared by the console (``webapp/media_api.py``) and
the chat (``bot/media.py``) — spec docs/superpowers/specs/2026-10-26-media-library-design.md.

Jobs: ``start_job`` runs one action as a background task and keeps its state
in memory (``JOBS``, the last ``MAX_JOBS``) for the console's Jobs tab to
poll. Results are library rows, so a restart loses only the progress view. A
failed job keeps its error; nothing is swallowed.

Items: a media asset (video or audio file) or a transcript. An mp3 made from
a file (``meta.derived_from``) and a transcript of a media asset
(``meta.media_asset_id``) are parts of that asset's item. Tags: the model's
suggestion lives in ``meta.suggest`` with ``meta.review = "pending"`` until
``accept`` writes ``meta.tags`` / ``meta.group`` (``review = "done"``).

Links: ``sign_link`` makes an HMAC-signed, expiring token for one asset's file
(``f`` inline, ``d`` download, ``t`` thumbnail); ``/files/{token}/<name>``
(``webapp/media_api.py``) serves it.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote

from . import media, media_tags
from .engines.registry import EngineConfigError
from .library import AUDIO_EXTS
from .llm import EngineError
from .transcribe import transcribe_file, transcribe_url

logger = logging.getLogger("argus.media")

SETTINGS_KEY = "media.settings"
DEFAULTS: dict[str, Any] = {"mp3_quality": "320", "language": "", "timestamps": False, "auto_tag": True,
                            "link_hours": 24}
VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi")
TEXT_EXTS = (".txt", ".md", ".srt", ".vtt")
MAX_JOBS = 100
_LANG = re.compile(r"[a-z]{2,3}")
_LINK_KINDS = ("f", "d", "t")


class MediaError(RuntimeError):
    """A media action that cannot be done, with the reason to show."""


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

async def load_settings(lib) -> dict[str, Any]:
    """The media settings: ``DEFAULTS`` under what was saved, and
    ``video_quality`` — the /quality setting (library key ``media_quality``)."""
    stored = await lib.get_json(SETTINGS_KEY, {}) or {}
    out = {**DEFAULTS, **{k: v for k, v in stored.items() if k in DEFAULTS}}
    out["video_quality"] = (await lib.get_setting("media_quality", "auto")) or "auto"
    return out


async def save_settings(lib, patch: dict) -> dict[str, Any]:
    """Validate every key of ``patch`` first (``ValueError`` naming the bad
    one), then persist; the new settings."""
    unknown = sorted(set(patch) - set(DEFAULTS) - {"video_quality"})
    if unknown:
        raise ValueError(f"unknown media setting(s): {', '.join(unknown)}")
    for key, value in patch.items():
        if key == "video_quality":
            media.quality_format(str(value), ffmpeg_available=True)          # raises with the accepted forms
        elif key == "mp3_quality" and value not in media.MP3_QUALITIES:
            raise ValueError(f"mp3_quality must be one of {', '.join(media.MP3_QUALITIES)}")
        elif key == "language" and not (value == "" or (isinstance(value, str) and _LANG.fullmatch(value))):
            raise ValueError("language must be a 2-3 letter code (en, ru, hy …) or empty for auto-detect")
        elif key in ("timestamps", "auto_tag") and not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
        elif key == "link_hours" and not (isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 168):
            raise ValueError("link_hours must be a whole number of hours from 1 to 168")
    if "video_quality" in patch:
        await lib.set_setting("media_quality", str(patch["video_quality"]).strip().lower())
    rest = {k: v for k, v in patch.items() if k != "video_quality"}
    if rest:
        await lib.set_json(SETTINGS_KEY, {**(await lib.get_json(SETTINGS_KEY, {}) or {}), **rest})
    return await load_settings(lib)


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    kind: str                       # fetch | convert | transcribe | tag | import
    title: str
    state: str = "running"          # running | done | failed
    pct: float | None = None        # None: no measurable progress (transcription, tagging)
    message: str = ""
    asset_id: int | None = None     # the item the job made or acted on
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def as_json(self) -> dict:
        return asdict(self)


JOBS: dict[str, Job] = {}
_TASKS: set[asyncio.Task] = set()


def start_job(kind: str, title: str, work: Callable[[Job], Awaitable[int | None]], *,
              then: Callable[[Job], Awaitable[None]] | None = None) -> Job:
    """Run ``work(job)`` in the background; its return is ``job.asset_id``.
    ``then(job)`` runs after it either way (the chat's reply)."""
    job = Job(id=uuid.uuid4().hex[:10], kind=kind, title=title[:200])
    JOBS[job.id] = job
    for jid in [j for j, v in JOBS.items() if v.state != "running"][:max(0, len(JOBS) - MAX_JOBS)]:
        JOBS.pop(jid)

    async def run() -> None:
        try:
            job.asset_id = await work(job)
            job.state, job.pct = "done", 100.0 if job.pct is not None else None
        except (MediaError, ValueError, EngineConfigError, EngineError) as e:   # expected: said, not a crash
            job.state, job.error = "failed", str(e)
            logger.warning("media job %s (%s) failed: %s", job.id, kind, e)
        except Exception as e:
            job.state, job.error = "failed", f"{type(e).__name__}: {e}"
            logger.exception("media job %s (%s) crashed", job.id, kind)
        finally:
            job.finished = time.time()
        if then is not None:
            try:
                await then(job)
            except Exception:
                logger.exception("media job %s: its follow-up failed", job.id)

    task = asyncio.get_running_loop().create_task(run())
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return job


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------

def media_type(asset: dict) -> str:
    """video | audio | transcript."""
    if asset["kind"] == "transcript":
        return "transcript"
    return "audio" if Path(asset["path"]).suffix.lower() in AUDIO_EXTS else "video"


async def require(lib, asset_id: int, kind: str | None = None) -> dict:
    """The asset, its file present on disk, or ``MediaError`` saying what is wrong."""
    a = await lib.get_asset(int(asset_id))
    if a is None or a["kind"] not in ("media", "transcript"):
        raise MediaError(f"no library item {asset_id}")
    if kind is not None and a["kind"] != kind:
        raise MediaError(f"item {asset_id} is a {media_type(a)}, not a {kind}")
    if not await asyncio.to_thread(Path(a["path"]).is_file):
        raise MediaError(f"the file of item {asset_id} is missing on disk ({Path(a['path']).name})")
    return a


async def register_file(lib, path: Path, *, platform: str, title: str, kind: str = "media",
                        source_url: str | None = None, meta: dict | None = None) -> int:
    """A file already in its vault folder as a library asset: its duration
    and (a video) a thumbnail beside it."""
    path = Path(path)
    meta = dict(meta or {})
    duration = None
    if kind == "media":
        duration = await media.media_duration(path)
        if path.suffix.lower() not in AUDIO_EXTS and "thumb" not in meta:
            thumb = path.with_suffix(".jpg")
            if await media.make_thumbnail(path, thumb):
                meta["thumb"] = str(thumb)
    return await lib.add_asset(kind=kind, platform=platform, source_url=source_url, title=title or path.name,
                               path=str(path), size_bytes=path.stat().st_size, duration_s=duration, meta=meta)


def _safe_name(name: str) -> str:
    stem = re.sub(r"[^\w.-]+", "_", Path(name).stem).strip("._")[:60] or "file"
    return f"{datetime.now():%Y%m%d_%H%M%S}_{stem}{Path(name).suffix.lower()}"


def import_kind(filename: str) -> str:
    """media | transcript for a file name, or ``MediaError`` for a type the library does not take."""
    ext = Path(filename).suffix.lower()
    if ext in TEXT_EXTS:
        return "transcript"
    if ext in VIDEO_EXTS or ext in AUDIO_EXTS:
        return "media"
    raise MediaError(f"{ext or 'a file without an extension'} is not a video, audio or transcript file "
                     f"(video {' '.join(VIDEO_EXTS)}; audio {' '.join(AUDIO_EXTS)}; text {' '.join(TEXT_EXTS)})")


async def import_file(lib, s, src: Path, *, filename: str, platform: str, title: str | None = None) -> int:
    """Move ``src`` (an upload's finished part, a chat download) into the
    vault — media under ``media_root/<platform>``, a transcript file under
    ``transcripts_root/<platform>`` — and register it (``title``: else the
    file name's stem)."""
    kind = import_kind(filename)
    root = Path(s.transcripts_root if kind == "transcript" else s.media_root) / platform
    dest = root / _safe_name(filename)
    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(shutil.move, str(src), str(dest))
    return await register_file(lib, dest, platform=platform, title=title or Path(filename).stem, kind=kind,
                               meta={"via": platform, "original_name": filename[:200],
                                     **({"imported": True} if kind == "transcript" else {})})


async def fetch(lib, s, url: str, job: Job, *, mode: str = "video", quality: str | None = None,
                mp3_quality: str | None = None, via: str = "console") -> int:
    """Download a link (any URL yt-dlp reads) as video at ``quality`` or as
    mp3 at ``mp3_quality`` (defaults: the media settings); the new asset."""
    if mode not in ("video", "mp3"):
        raise MediaError("mode must be video or mp3")
    cfg = await load_settings(lib)
    audio = (mp3_quality or cfg["mp3_quality"]) if mode == "mp3" else None
    quality = quality or cfg["video_quality"]
    job.pct, job.message = 0.0, "starting the download"

    def on_progress(pct: float, eta: str) -> None:
        job.pct, job.message = pct, f"{pct:.0f}% · ETA {eta}"

    r = await media.download_media(url, dest_root=s.media_root, quality=quality, on_progress=on_progress,
                                   audio=audio, any_url=True)
    if not r.ok:
        raise MediaError(r.error or "the download failed")
    meta: dict[str, Any] = {"via": via, "info_json": r.info_json_path}
    meta.update({"format": "mp3", "mp3_quality": audio} if audio else {"quality": quality})
    if r.thumb_path:
        meta["thumb"] = r.thumb_path
    job.message = f"saved · {r.size_bytes / 1048576:.1f} MB"
    return await lib.add_asset(kind="media", platform=r.platform, source_url=url, media_id=r.media_id,
                               title=r.title or url, path=r.path, size_bytes=r.size_bytes,
                               duration_s=r.duration_s, meta=meta)


async def convert(lib, asset_id: int, job: Job, *, quality: str | None = None) -> int:
    """The item's audio as an mp3 beside it (a part of the item); the new asset."""
    a = await require(lib, asset_id, "media")
    q = quality or (await load_settings(lib))["mp3_quality"]
    src = Path(a["path"])
    dest = src.with_name(f"{src.stem}.{q}.mp3")      # never the name of a link's own mp3 (<stem>_audio.mp3)
    job.pct, job.message = 0.0, f"converting to mp3 ({q})"

    def on_progress(pct: float, _eta: str) -> None:
        job.pct, job.message = pct, f"{pct:.0f}%"

    await media.convert_to_mp3(src, dest, quality=q, duration_s=a.get("duration_s"), on_progress=on_progress)
    job.message = f"mp3 ({q}) · {dest.stat().st_size / 1048576:.1f} MB"
    meta: dict[str, Any] = {"format": "mp3", "mp3_quality": q, "derived_from": a["asset_id"]}
    if a["meta"].get("thumb"):
        meta["thumb"] = a["meta"]["thumb"]
    return await lib.add_asset(kind="media", platform=a.get("platform"), source_url=a.get("source_url"),
                               media_id=a.get("media_id"), title=a.get("title"), path=str(dest),
                               size_bytes=dest.stat().st_size, duration_s=a.get("duration_s"), meta=meta)


async def transcribe(lib, s, asset_id: int, job: Job, *, language: str | None = None,
                     timestamps: bool | None = None) -> int:
    """A transcript of a media item (a part of it); the transcript asset.
    A YouTube item without timestamps goes through the transcriber's own
    order (captions first, when configured); everything else is ASR over
    the file, in ``language`` (None: the setting; "": detect)."""
    a = await require(lib, asset_id, "media")
    cfg = await load_settings(lib)
    language = cfg["language"] if language is None else language
    timestamps = cfg["timestamps"] if timestamps is None else timestamps
    job.message = "transcribing — speech-to-text can take a few minutes"
    url = a.get("source_url") or ""
    if a.get("platform") == "youtube" and url and not timestamps:
        r = await transcribe_url(url, transcripts_root=s.transcripts_root, media_root=s.media_root, library=lib)
        if not r.ok:
            raise MediaError(r.error or "the transcript failed")
        t = await lib.get_asset_by_source(url, "transcript")
        if t is None:
            raise MediaError("the transcript was written but not registered in the library (see the bot log)")
        await lib.update_asset_meta(t["asset_id"], {"media_asset_id": a["asset_id"]})
        return t["asset_id"]
    r = await transcribe_file(Path(a["path"]), transcripts_root=s.transcripts_root,
                              platform=a.get("platform") or "upload", title=a.get("title") or "",
                              source_url=url, language=language or None, timestamps=timestamps,
                              library=lib, media_asset_id=a["asset_id"])
    if not r.ok:
        raise MediaError(r.error or "the transcript failed")
    job.message = f"{r.backend} · {r.language or '?'}"
    return r.asset_id


def _read_text(path: str, limit: int = 20000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(limit)
    except OSError:
        return ""


def _description(meta: dict) -> str:
    info = meta.get("info_json")
    if not info or not Path(info).is_file():
        return ""
    try:
        return str(json.loads(Path(info).read_text(encoding="utf-8", errors="replace")).get("description") or "")
    except ValueError:
        return ""


async def tag(lib, asset_id: int) -> dict:
    """Ask for tags, a group, a gist and links; stored as the item's pending
    suggestion. The suggestion. Raises ``llm.EngineError`` when no engine answers."""
    a = await lib.get_asset(int(asset_id))
    if a is None:
        raise MediaError(f"no library item {asset_id}")
    transcript = ""
    if a["kind"] == "transcript":
        transcript = await asyncio.to_thread(_read_text, a["path"])
    else:
        for r in await lib.related_assets(a):
            if r["kind"] == "transcript":
                transcript = await asyncio.to_thread(_read_text, r["path"])
                if transcript:
                    break
    facets = await lib.asset_facets()
    sug = await media_tags.suggest(
        title=a.get("title") or Path(a["path"]).name, description=await asyncio.to_thread(_description, a["meta"]),
        transcript=transcript, url=a.get("source_url") or "",
        vocabulary=[t for t, _ in facets["tags"]], groups=[g for g, _ in facets["groups"]])
    await lib.update_asset_meta(a["asset_id"], {"suggest": None})       # replace, not merge, the old one
    await lib.update_asset_meta(a["asset_id"], {"suggest": sug, "review": "pending"})
    return sug


async def accept(lib, asset_id: int, *, tags: list, group: str | None) -> dict:
    """Confirm the item's tags and group (what the person chose, normalized); the updated asset."""
    if not isinstance(tags, list):
        raise ValueError("tags must be a list")
    norm = list(dict.fromkeys(t for t in (media_tags.normalize_tag(x) for x in tags) if t))[:20]
    group = (group or "").strip()[:48]
    row = await lib.update_asset_meta(int(asset_id), {"tags": norm, "group": group or None, "review": "done"})
    if row is None:
        raise MediaError(f"no library item {asset_id}")
    return row


def tag_job(lib, asset_id: int, title: str, *, then: Callable[[Job], Awaitable[None]] | None = None) -> Job:
    async def work(job: Job) -> int:
        job.message = "asking the model for tags"
        sug = await tag(lib, asset_id)
        job.message = " ".join(f"#{t}" for t in sug["tags"]) + (f" → {sug['group']}" if sug.get("group") else "")
        return asset_id
    return start_job("tag", f"Tags for {title}", work, then=then)


async def maybe_autotag(lib, asset_id: int | None, title: str, *,
                        then: Callable[[Job], Awaitable[None]] | None = None) -> Job | None:
    """A tag job for a newly saved primary item when ``auto_tag`` is on."""
    if asset_id is None or not (await load_settings(lib))["auto_tag"]:
        return None
    return tag_job(lib, asset_id, title, then=then)


async def delete(lib, s, asset_id: int) -> list[str]:
    """Delete an item — its file, info.json and thumbnail, and its parts —
    or one part (an mp3, a transcript: the parent's thumbnail stays). Files
    outside the Argus roots are never touched. The names removed from disk."""
    from .bot.delete import _safe_to_delete
    a = await lib.get_asset(int(asset_id))
    if a is None or a["kind"] not in ("media", "transcript"):
        raise MediaError(f"no library item {asset_id}")
    is_part = a["meta"].get("derived_from") is not None or a["meta"].get("media_asset_id") is not None
    rows = [a] if is_part else [a, *await lib.related_assets(a)]
    removed: list[str] = []
    for r in rows:
        derived = r["meta"].get("derived_from") is not None
        for f in dict.fromkeys(x for x in (r["path"], r["meta"].get("info_json"),
                                           None if derived else r["meta"].get("thumb")) if x):
            fp = Path(f)
            if fp.is_file() and _safe_to_delete(fp, s):
                await asyncio.to_thread(fp.unlink)
                removed.append(fp.name)
    await lib.delete_assets([r["asset_id"] for r in rows])
    return removed


# ---------------------------------------------------------------------------
# signed links
# ---------------------------------------------------------------------------

def _key(bot_token: str) -> bytes:
    return hmac.new(bot_token.encode(), b"argus-media-link-v1", hashlib.sha256).digest()


def _sig(bot_token: str, body: str) -> str:
    return hmac.new(_key(bot_token), body.encode(), hashlib.sha256).hexdigest()[:32]


def sign_link(bot_token: str, asset_id: int, kind: str = "f", hours: float = 24, *,
              now: float | None = None) -> str:
    if kind not in _LINK_KINDS:
        raise ValueError(f"link kind must be one of {_LINK_KINDS}")
    body = f"{int(asset_id)}.{int((time.time() if now is None else now) + hours * 3600)}.{kind}"
    return f"{body}.{_sig(bot_token, body)}"


def verify_link(bot_token: str, token: str, *, now: float | None = None) -> tuple[int, str]:
    """(asset_id, kind) of a valid token; ``MediaError`` for a forged,
    malformed or expired one (the same message for all, nothing to probe)."""
    parts = (token or "").split(".")
    ok = (len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit() and parts[2] in _LINK_KINDS
          and hmac.compare_digest(_sig(bot_token, ".".join(parts[:3])), parts[3])
          and int(parts[1]) >= (time.time() if now is None else now))
    if not ok:
        raise MediaError("this link is invalid or has expired — make a new one in the Argus console")
    return int(parts[0]), parts[2]


def link_url(public_url: str, token: str, filename: str) -> str:
    return f"{public_url.rstrip('/')}/files/{token}/{quote(filename)}"


def item_url(public_url: str, asset_id: int) -> str:
    """The console page of an item (a Mini App deep link)."""
    return f"{public_url.rstrip('/')}/#/media/item/{int(asset_id)}"


# ---------------------------------------------------------------------------
# send to chat
# ---------------------------------------------------------------------------

async def send_to_chat(bot, chat_id: int, asset: dict, *, bot_token: str, public_url: str | None,
                       hours: int) -> str:
    """The item's file in the chat when Telegram takes it ("sent"); else a
    private download link and an Open-in-Argus button ("link"). With no
    public URL for a link, ``MediaError`` saying so."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo

    from .bot.common import _TG_UPLOAD_CAP_BYTES
    p = Path(asset["path"])
    size = p.stat().st_size
    title = (asset.get("title") or p.name)[:1024]
    kind = media_type(asset)
    if size <= _TG_UPLOAD_CAP_BYTES:
        with p.open("rb") as f:
            if kind == "transcript":
                await bot.send_document(chat_id=chat_id, document=f, filename=p.name, caption=title)
            elif kind == "audio":
                await bot.send_audio(chat_id=chat_id, audio=f, filename=p.name, title=title[:64], caption=title,
                                     duration=int(asset["duration_s"]) if asset.get("duration_s") else None)
            else:
                try:
                    await bot.send_video(chat_id=chat_id, video=f, caption=title, supports_streaming=True)
                except Exception as e:                   # Telegram refused it as a video: the file as a document
                    logger.warning("send_video failed (%s); sending as a document", e)
                    f.seek(0)
                    await bot.send_document(chat_id=chat_id, document=f, filename=p.name, caption=title)
        return "sent"
    if not public_url:
        raise MediaError(f"the file is {size / 1048576:.0f} MB, over Telegram's bot upload cap, and the console "
                         "has no public URL for a download link (is the tunnel up? see /status)")
    token = sign_link(bot_token, asset["asset_id"], "d", hours)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("⬇️ Download", url=link_url(public_url, token, p.name))],
        [InlineKeyboardButton("📱 Open in Argus", web_app=WebAppInfo(item_url(public_url, asset["asset_id"])))]])
    await bot.send_message(chat_id=chat_id, reply_markup=kb, text=(
        f"📁 {title}\n{size / 1048576:.0f} MB — over Telegram's 50 MB bot upload cap, so here is a private "
        f"download link from your PC (valid {hours} h, or until the bot restarts)."))
    return "link"
