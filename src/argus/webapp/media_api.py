"""The console's media library API (spec docs/superpowers/specs/2026-10-26-media-library-design.md).

``/api/media/*`` sits behind the same initData auth and error JSON as the
rest of ``/api``; the actions themselves are ``media_service`` (shared with
the chat). Long actions (fetch, convert, transcribe, tag) answer at once with
a job the Jobs tab polls.

``/files/{token}/{name}`` is outside ``/api``: a ``<video>`` element or a
phone's download manager cannot send the Authorization header, so the token
(``media_service.sign_link``: asset, expiry, kind, HMAC) is the credential.
It serves only the asset's registered file (or its thumbnail), only inside
the Argus roots, with Range support (seeking).

Uploads are chunked so no request nears Cloudflare's ~100 MB body cap:
``POST /api/media/uploads`` opens one, ``PUT …/{uid}?offset=N`` appends a
chunk (in order), ``POST …/{uid}/finish`` moves it into the vault.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from aiohttp import web
from telegram.error import TelegramError

from .. import media, media_service as ms
from .api import DEPS, ApiError, _bad, _body, _int_query, _ok, _opt

logger = logging.getLogger("argus.webapp")

CHUNK_BYTES = 8 * 1024 * 1024
_MAX_CHUNK_BYTES = 16 * 1024 * 1024
_UPLOAD_TTL_S = 24 * 3600
_TAG_BATCH = 20
UPLOADS = web.AppKey("media_uploads", dict)


def _max_upload_bytes() -> int:
    return int(os.environ.get("ARGUS_MEDIA_UPLOAD_MAX_MB") or 4096) * 1024 * 1024


def _ctx(request: web.Request):
    d = request.app[DEPS]
    if d.library is None or d.settings is None:
        raise ApiError(409, "refused", "the media library is not available (no library or settings)")
    return d, d.library, d.settings


def _id(request: web.Request) -> int:
    raw = request.match_info["id"]
    if not raw.isdigit():
        raise _bad("the item id must be a number")
    return int(raw)


def _media_error(e: ms.MediaError) -> ApiError:
    return ApiError(404 if str(e).startswith("no library item") else 409, "refused", str(e))


async def _require(lib, asset_id: int, kind: str | None = None) -> dict:
    try:
        return await ms.require(lib, asset_id, kind)
    except ms.MediaError as e:
        raise _media_error(e) from None


def _links(d, a: dict, hours: int) -> dict:
    """Relative signed links (same origin; the page prefixes its origin when it needs an absolute one)."""
    name = Path(a["path"]).name
    out = {"file": ms.link_url("", ms.sign_link(d.bot_token, a["asset_id"], "f", hours), name),
           "download": ms.link_url("", ms.sign_link(d.bot_token, a["asset_id"], "d", hours), name)}
    if a["meta"].get("thumb"):
        out["thumb"] = ms.link_url("", ms.sign_link(d.bot_token, a["asset_id"], "t", hours), "thumb.jpg")
    return out


def _item(d, a: dict, hours: int) -> dict:
    meta = a.get("meta") or {}
    return {"id": a["asset_id"], "kind": a["kind"], "type": ms.media_type(a), "title": a.get("title") or Path(a["path"]).name,
            "platform": a.get("platform"), "source_url": a.get("source_url"), "bytes": a.get("bytes") or 0,
            "duration_s": a.get("duration_s"), "created_at": a.get("created_at"), "file_name": Path(a["path"]).name,
            "format": meta.get("format"), "quality": meta.get("quality") or meta.get("mp3_quality"),
            "tags": meta.get("tags") or [], "group": meta.get("group") or "", "review": meta.get("review"),
            "suggest": meta.get("suggest"), "derived_from": meta.get("derived_from"),
            "media_asset_id": meta.get("media_asset_id"), "backend": meta.get("backend"),
            "language": meta.get("language"), "imported": bool(meta.get("imported")),
            "links": _links(d, a, hours)}


async def _hours(lib) -> int:
    return int((await ms.load_settings(lib))["link_hours"])


def _job(job: ms.Job) -> web.Response:
    return _ok({"job": job.as_json()})


# ---------------------------------------------------------------------------
# library
# ---------------------------------------------------------------------------

async def list_items(request: web.Request) -> web.Response:
    d, lib, _ = _ctx(request)
    q = request.query
    kind = {"video": ("media", "video"), "audio": ("media", "audio"), "transcript": ("transcript", None)}
    if q.get("type") and q["type"] not in kind:
        raise _bad("type must be video, audio or transcript")
    k, mt = kind.get(q.get("type") or "", (None, None))
    try:
        rows, total = await lib.search_assets(
            q=(q.get("q") or "").strip() or None, kind=k, media_type=mt, tag=q.get("tag") or None,
            group=q.get("group") or None, review=q.get("review") or None,
            limit=_int_query(request, "limit", 30, 1, 200), offset=_int_query(request, "offset", 0, 0, 10**6))
    except ValueError as e:
        raise _bad(str(e)) from None
    hours = await _hours(lib)
    return _ok({"items": [_item(d, a, hours) for a in rows], "total": total, "facets": await lib.asset_facets()})


async def get_item(request: web.Request) -> web.Response:
    d, lib, _ = _ctx(request)
    a = await lib.get_asset(_id(request))
    if a is None or a["kind"] not in ("media", "transcript"):
        raise ApiError(404, "not_found", f"no library item {request.match_info['id']}")
    hours = await _hours(lib)
    parts = [_item(d, r, hours) for r in await lib.related_assets(a)]
    text = None
    if a["kind"] == "transcript":
        text = await asyncio.to_thread(ms._read_text, a["path"], 12000)
    present = await asyncio.to_thread(Path(a["path"]).is_file)
    return _ok({"item": _item(d, a, hours), "parts": parts, "text": text, "present": present,
                "public_url": d.bot_data.get("webapp_public_url") or None,
                "facets": await lib.asset_facets()})


async def delete_item(request: web.Request) -> web.Response:
    _, lib, s = _ctx(request)
    try:
        removed = await ms.delete(lib, s, _id(request))
    except ms.MediaError as e:
        raise _media_error(e) from None
    return _ok({"outcome": "deleted", "removed": removed,
                "message": f"Deleted {len(removed)} file(s) from the vault."})


async def facets(request: web.Request) -> web.Response:
    _, lib, _ = _ctx(request)
    return _ok(await lib.asset_facets())


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

async def probe(request: web.Request) -> web.Response:
    url = (await _body(request)).get("url")
    if not isinstance(url, str) or not url.strip().lower().startswith(("http://", "https://")):
        raise _bad("url must be an http(s) link")
    try:
        return _ok({"probe": await media.probe_url(url.strip())})
    except RuntimeError as e:
        raise ApiError(422, "probe_failed", str(e)) from None


async def fetch(request: web.Request) -> web.Response:
    _, lib, s = _ctx(request)
    body = await _body(request)
    url = body.get("url")
    if not isinstance(url, str) or not url.strip().lower().startswith(("http://", "https://")):
        raise _bad("url must be an http(s) link")
    mode = _opt(body, "mode") or "video"
    if mode not in ("video", "mp3"):
        raise _bad("mode must be video or mp3")
    quality, mp3_q = _opt(body, "quality"), _opt(body, "mp3_quality")
    if quality is not None:
        try:
            media.quality_format(quality, ffmpeg_available=True)
        except ValueError as e:
            raise _bad(str(e)) from None
    if mp3_q is not None and mp3_q not in media.MP3_QUALITIES:
        raise _bad(f"mp3_quality must be one of {', '.join(media.MP3_QUALITIES)}")
    url = url.strip()

    async def work(job: ms.Job) -> int:
        return await ms.fetch(lib, s, url, job, mode=mode, quality=quality, mp3_quality=mp3_q)

    async def then(job: ms.Job) -> None:
        if job.state == "done":
            await ms.maybe_autotag(lib, job.asset_id, job.title)

    title = _opt(body, "title") or url
    return _job(ms.start_job("fetch", f"{'MP3' if mode == 'mp3' else 'Video'} · {title}", work, then=then))


async def convert(request: web.Request) -> web.Response:
    _, lib, _ = _ctx(request)
    a = await _require(lib, _id(request), "media")
    q = _opt(await _body(request), "quality")
    if q is not None and q not in media.MP3_QUALITIES:
        raise _bad(f"quality must be one of {', '.join(media.MP3_QUALITIES)}")
    return _job(ms.start_job("convert", f"MP3 · {a.get('title') or a['asset_id']}",
                             lambda job: ms.convert(lib, a["asset_id"], job, quality=q)))


async def transcribe(request: web.Request) -> web.Response:
    _, lib, s = _ctx(request)
    a = await _require(lib, _id(request), "media")
    body = await _body(request)
    language, timestamps = _opt(body, "language"), _opt(body, "timestamps", bool)
    if language not in (None, "") and not ms._LANG.fullmatch(language):
        raise _bad("language must be a 2-3 letter code, or empty to detect it")

    async def then(job: ms.Job) -> None:
        if job.state == "done" and not (a["meta"].get("tags")):
            await ms.maybe_autotag(lib, a["asset_id"], a.get("title") or "")    # a transcript makes better tags

    return _job(ms.start_job("transcribe", f"Transcript · {a.get('title') or a['asset_id']}",
                             lambda job: ms.transcribe(lib, s, a["asset_id"], job, language=language,
                                                       timestamps=timestamps), then=then))


async def suggest_tags(request: web.Request) -> web.Response:
    _, lib, _ = _ctx(request)
    a = await lib.get_asset(_id(request))
    if a is None:
        raise ApiError(404, "not_found", f"no library item {request.match_info['id']}")
    return _job(ms.tag_job(lib, a["asset_id"], a.get("title") or str(a["asset_id"])))


async def tag_untagged(request: web.Request) -> web.Response:
    """A tag job for each of the newest untagged items, at most ``_TAG_BATCH``."""
    _, lib, _ = _ctx(request)
    rows, total = await lib.search_assets(review="untagged", limit=_TAG_BATCH)
    jobs = [ms.tag_job(lib, a["asset_id"], a.get("title") or str(a["asset_id"])).as_json() for a in rows]
    return _ok({"jobs": jobs, "untagged": total,
                "message": f"Asking for tags on {len(jobs)} of {total} untagged item(s)."})


async def put_tags(request: web.Request) -> web.Response:
    d, lib, _ = _ctx(request)
    body = await _body(request)
    try:
        row = await ms.accept(lib, _id(request), tags=body.get("tags"), group=_opt(body, "group"))
    except ValueError as e:
        raise _bad(str(e)) from None
    except ms.MediaError as e:
        raise _media_error(e) from None
    return _ok({"item": _item(d, row, await _hours(lib))})


async def send(request: web.Request) -> web.Response:
    d, lib, _ = _ctx(request)
    a = await _require(lib, _id(request))
    try:
        how = await ms.send_to_chat(d.bot, d.allowed_user_id, a, bot_token=d.bot_token,
                                    public_url=d.bot_data.get("webapp_public_url"), hours=await _hours(lib))
    except ms.MediaError as e:
        raise _media_error(e) from None
    except TelegramError as e:
        logger.warning("webapp: media %s: send to chat failed: %s", a["asset_id"], e)
        raise ApiError(502, "send_failed", f"⚠️ send failed: {e}") from None
    return _ok({"outcome": how, "message": "Sent to the chat." if how == "sent"
                else "Too big for Telegram — a download link was sent to the chat."})


async def attach(request: web.Request) -> web.Response:
    """Queue a transcript on a research run (the /append path); Continue on the run ingests it."""
    d, lib, _ = _ctx(request)
    a = await _require(lib, _id(request))
    if a["kind"] != "transcript":
        raise ApiError(409, "refused", "only a transcript can be added to a research run — transcribe it "
                                       "first, then attach the transcript")
    run_id = _opt(await _body(request), "run_id")
    if not run_id:
        raise _bad("run_id is required")
    run = await lib.get_run(run_id)
    if run is None:
        raise ApiError(404, "not_found", f"no run {run_id}")
    if run.get("mode") == "quick":
        raise ApiError(409, "refused", f"run {run_id} was a quick /ask — it takes no sources")
    await lib.add_run_source(run_id, f"asset:{a['asset_id']}", kind="asset")
    pending = await lib.pending_sources(run_id)
    when = {"done": "Continue the run to ingest it (an append-only pass, no new searches).",
            "awaiting_report": "Extend at the report preview ingests it.",
            "cancelled": "Continue the run: it resumes, then a later Continue ingests the sources.",
            "paused": "Continue the run: it resumes, then a later Continue ingests the sources."}
    return _ok({"outcome": "attached", "run_id": run_id, "pending": len(pending),
                "message": f"Added to run {run_id} ({len(pending)} pending). "
                           + when.get(run.get("status"), "It is ingested on the next Continue after this pass.")})


async def jobs(request: web.Request) -> web.Response:
    return _ok({"jobs": [j.as_json() for j in reversed(list(ms.JOBS.values()))]})


async def get_settings(request: web.Request) -> web.Response:
    _, lib, _ = _ctx(request)
    return _ok({"settings": await ms.load_settings(lib), "mp3_qualities": list(media.MP3_QUALITIES),
                "ffmpeg": bool(media.resolve_ffmpeg()), "upload_max_mb": _max_upload_bytes() // 1048576})


async def put_settings(request: web.Request) -> web.Response:
    _, lib, _ = _ctx(request)
    try:
        return _ok({"settings": await ms.save_settings(lib, await _body(request))})
    except ValueError as e:
        raise _bad(str(e)) from None


# ---------------------------------------------------------------------------
# uploads
# ---------------------------------------------------------------------------

def _parts_dir(s) -> Path:
    return Path(s.media_root) / ".uploads"


def _expire_uploads(app: web.Application) -> None:
    now = time.time()
    for uid, up in list(app[UPLOADS].items()):
        if now - up["touched"] > _UPLOAD_TTL_S:
            app[UPLOADS].pop(uid, None)
            Path(up["part"]).unlink(missing_ok=True)


async def upload_open(request: web.Request) -> web.Response:
    _, _, s = _ctx(request)
    body = await _body(request)
    name, size = body.get("name"), body.get("size")
    if not isinstance(name, str) or not name.strip():
        raise _bad("name must be the file's name")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise _bad("size must be the file's size in bytes")
    if size > _max_upload_bytes():
        raise ApiError(413, "too_large", f"the file is {size / 1048576:.0f} MB, over the "
                                         f"{_max_upload_bytes() // 1048576} MB upload limit (ARGUS_MEDIA_UPLOAD_MAX_MB)")
    try:
        ms.import_kind(name)
    except ms.MediaError as e:
        raise ApiError(415, "unsupported", str(e)) from None
    _expire_uploads(request.app)
    uid = uuid.uuid4().hex
    parts = _parts_dir(s)
    await asyncio.to_thread(parts.mkdir, parents=True, exist_ok=True)
    part = parts / f"{uid}.part"
    await asyncio.to_thread(part.write_bytes, b"")
    request.app[UPLOADS][uid] = {"name": Path(name).name[:200], "size": size, "received": 0, "part": str(part),
                                 "touched": time.time()}
    return _ok({"upload_id": uid, "chunk_size": CHUNK_BYTES})


def _upload(request: web.Request) -> tuple[str, dict]:
    uid = request.match_info["uid"]
    up = request.app[UPLOADS].get(uid)
    if up is None:
        raise ApiError(404, "not_found", "no such upload (it expired, finished, or the bot restarted) — start again")
    return uid, up


def _append(path: str, data: bytes) -> None:
    with open(path, "ab") as f:
        f.write(data)


async def upload_chunk(request: web.Request) -> web.Response:
    uid, up = _upload(request)
    offset = _int_query(request, "offset", -1, 0, 2**62)
    if offset != up["received"]:
        raise ApiError(409, "bad_offset", f"expected the chunk at offset {up['received']}, got {offset}")
    data = bytearray()
    while chunk := await request.content.read(1 << 20):
        data += chunk
        if len(data) > _MAX_CHUNK_BYTES:
            raise ApiError(413, "too_large", f"a chunk may be at most {_MAX_CHUNK_BYTES // 1048576} MB")
    if not data:
        raise _bad("the chunk is empty")
    if up["received"] + len(data) > up["size"]:
        raise _bad("the chunks add up to more than the declared size")
    await asyncio.to_thread(_append, up["part"], bytes(data))
    up["received"] += len(data)
    up["touched"] = time.time()
    return _ok({"received": up["received"], "size": up["size"]})


async def upload_finish(request: web.Request) -> web.Response:
    d, lib, s = _ctx(request)
    uid, up = _upload(request)
    if up["received"] != up["size"]:
        raise ApiError(409, "incomplete", f"received {up['received']} of {up['size']} bytes")
    body = await _body(request)
    want_tr, want_mp3 = bool(_opt(body, "transcribe", bool)), bool(_opt(body, "mp3", bool))
    request.app[UPLOADS].pop(uid, None)
    try:
        asset_id = await ms.import_file(lib, s, Path(up["part"]), filename=up["name"], platform="upload")
    except ms.MediaError as e:
        Path(up["part"]).unlink(missing_ok=True)
        raise ApiError(415, "unsupported", str(e)) from None
    a = await lib.get_asset(asset_id)
    title = a.get("title") or up["name"]
    started = []
    is_media = a["kind"] == "media"
    if is_media and want_mp3 and ms.media_type(a) == "video":
        started.append(ms.start_job("convert", f"MP3 · {title}", lambda job: ms.convert(lib, asset_id, job)))
    if is_media and want_tr:
        async def then(job: ms.Job) -> None:
            if job.state == "done":
                await ms.maybe_autotag(lib, asset_id, title)
        started.append(ms.start_job("transcribe", f"Transcript · {title}",
                                    lambda job: ms.transcribe(lib, s, asset_id, job), then=then))
    else:
        job = await ms.maybe_autotag(lib, asset_id, title)
        if job is not None:
            started.append(job)
    return _ok({"item": _item(d, a, await _hours(lib)), "jobs": [j.as_json() for j in started]})


async def upload_cancel(request: web.Request) -> web.Response:
    uid, up = _upload(request)
    request.app[UPLOADS].pop(uid, None)
    await asyncio.to_thread(Path(up["part"]).unlink, missing_ok=True)
    return _ok({"outcome": "cancelled"})


# ---------------------------------------------------------------------------
# signed files
# ---------------------------------------------------------------------------

def _inside_roots(path: Path, s) -> bool:
    roots = [Path(r).resolve() for r in (s.media_root, s.transcripts_root, s.history_root, s.reports_root)]
    rp = path.resolve()
    return any(rp != r and r in rp.parents for r in roots)


async def serve_file(request: web.Request) -> web.StreamResponse:
    d = request.app[DEPS]
    try:
        asset_id, kind = ms.verify_link(d.bot_token, request.match_info["token"])
    except ms.MediaError as e:
        raise web.HTTPForbidden(text=str(e)) from None
    if d.library is None or d.settings is None:
        raise web.HTTPServiceUnavailable(text="the media library is not available")
    a = await d.library.get_asset(asset_id)
    raw = (a or {}).get("meta", {}).get("thumb") if kind == "t" else (a or {}).get("path")
    if not raw:
        raise web.HTTPNotFound(text="this item no longer exists")
    path = Path(raw)
    if not await asyncio.to_thread(lambda: path.is_file() and _inside_roots(path, d.settings)):
        logger.warning("webapp: media %s: its file is missing or outside the Argus roots; refused", asset_id)
        raise web.HTTPNotFound(text="the file is missing on disk")
    headers = {"Content-Disposition": f"{'attachment' if kind == 'd' else 'inline'}; filename*=UTF-8''{quote(path.name)}",
               "Cache-Control": "private, max-age=3600"}
    if kind == "d":
        headers["Access-Control-Allow-Origin"] = "https://web.telegram.org"   # WebApp.downloadFile on Telegram Web
    return web.FileResponse(path, headers=headers)


def setup(app: web.Application) -> None:
    app[UPLOADS] = {}
    r = app.router
    r.add_get("/api/media", list_items)
    r.add_get("/api/media/facets", facets)
    r.add_get("/api/media/jobs", jobs)
    r.add_get("/api/media/settings", get_settings)
    r.add_put("/api/media/settings", put_settings)
    r.add_post("/api/media/probe", probe)
    r.add_post("/api/media/fetch", fetch)
    r.add_post("/api/media/tag-untagged", tag_untagged)
    r.add_post("/api/media/uploads", upload_open)
    r.add_put("/api/media/uploads/{uid}", upload_chunk)
    r.add_post("/api/media/uploads/{uid}/finish", upload_finish)
    r.add_delete("/api/media/uploads/{uid}", upload_cancel)
    r.add_get("/api/media/{id}", get_item)
    r.add_delete("/api/media/{id}", delete_item)
    r.add_post("/api/media/{id}/convert", convert)
    r.add_post("/api/media/{id}/transcribe", transcribe)
    r.add_post("/api/media/{id}/tag", suggest_tags)
    r.add_put("/api/media/{id}/tags", put_tags)
    r.add_post("/api/media/{id}/send", send)
    r.add_post("/api/media/{id}/attach", attach)
    r.add_get("/files/{token}/{name}", serve_file)
