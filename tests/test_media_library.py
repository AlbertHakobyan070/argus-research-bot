"""The media library (spec docs/superpowers/specs/2026-10-26-media-library-design.md): the library's
browse queries, media_service (links, settings, jobs, tags, delete), media_tags, the media API, the
chunked upload and the signed /files route."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from argus import media_service as ms
from argus import media_tags
from argus.library import Library
from webapp_fakes import TOKEN, FakeBot, FakeRuns, auth, make_app, media_settings


@pytest.fixture
async def lib(tmp_path):
    library = Library(tmp_path / "lib.sqlite")
    await library.open()
    yield library
    await library.close()


@pytest.fixture
def roots(tmp_path):
    s = media_settings(tmp_path / "vault")
    for r in (s.media_root, s.transcripts_root, s.history_root, s.reports_root):
        Path(r).mkdir(parents=True)
    return s


def _file(root: Path, rel: str, data: bytes = b"x" * 1000) -> Path:
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


async def _add(lib, path: Path, kind: str = "media", **kw) -> int:
    return await lib.add_asset(kind=kind, path=str(path), size_bytes=path.stat().st_size,
                               title=kw.pop("title", path.stem), **kw)


# ---------------------------------------------------------------------------
# library queries
# ---------------------------------------------------------------------------

async def test_primary_lists_items_and_parts_belong_to_their_item(lib, roots):
    video = await _add(lib, _file(roots.media_root, "youtube/talk.mp4"), platform="youtube", source_url="https://y/1")
    tr = await _add(lib, _file(roots.transcripts_root, "youtube/talk.txt"), kind="transcript",
                    meta={"media_asset_id": video})
    mp3 = await _add(lib, _file(roots.media_root, "youtube/talk.mp3"), meta={"derived_from": video, "format": "mp3"})
    same_url = await _add(lib, _file(roots.transcripts_root, "youtube/captions.txt"), kind="transcript",
                          source_url="https://y/1")
    alone = await _add(lib, _file(roots.transcripts_root, "upload/notes.txt"), kind="transcript")
    rows, total = await lib.search_assets()
    assert {r["asset_id"] for r in rows} == {video, alone} and total == 2
    parts = await lib.related_assets(await lib.get_asset(video))
    assert {r["asset_id"] for r in parts} == {tr, mp3, same_url}
    assert video in {r["asset_id"] for r in await lib.related_assets(await lib.get_asset(mp3))}
    rows, _ = await lib.search_assets(primary=False)
    assert len(rows) == 5


async def test_type_tag_group_review_filters_and_facets(lib, roots):
    a = await _add(lib, _file(roots.media_root, "x/a.mp4"))
    b = await _add(lib, _file(roots.media_root, "x/b.MP3"))
    c = await _add(lib, _file(roots.transcripts_root, "x/c.txt"), kind="transcript")
    await ms.accept(lib, a, tags=["RAG", "#LangGraph", "rag"], group="AI agents")
    await lib.update_asset_meta(b, {"suggest": {"tags": ["x"]}, "review": "pending"})
    ids = lambda rows: {r["asset_id"] for r in rows[0]}                       # noqa: E731
    assert ids(await lib.search_assets(media_type="audio")) == {b}
    assert ids(await lib.search_assets(media_type="video")) == {a}
    assert ids(await lib.search_assets(kind="transcript")) == {c}
    assert ids(await lib.search_assets(tag="langgraph")) == {a}
    assert ids(await lib.search_assets(group="AI agents")) == {a}
    assert ids(await lib.search_assets(review="pending")) == {b}
    assert ids(await lib.search_assets(review="untagged")) == {c}
    assert ids(await lib.search_assets(q="langgr")) == {a}                    # meta (tags) is searched
    assert ids(await lib.search_assets(q="100%")) == set()                     # LIKE wildcards are literal
    f = await lib.asset_facets()
    assert f == {"tags": [["langgraph", 1], ["rag", 1]], "groups": [["AI agents", 1]], "pending": 1}
    with pytest.raises(ValueError):
        await lib.search_assets(review="maybe")


async def test_meta_patch_merges_none_removes_and_a_redownload_keeps_tags(lib, roots):
    p = _file(roots.media_root, "x/a.mp4")
    a = await _add(lib, p, meta={"quality": "auto", "info_json": "i.json"})
    await ms.accept(lib, a, tags=["rag"], group="G")
    row = await lib.update_asset_meta(a, {"info_json": None, "thumb": "t.jpg"})
    assert row["meta"] == {"quality": "auto", "thumb": "t.jpg", "tags": ["rag"], "group": "G", "review": "done"}
    assert await lib.update_asset_meta(999, {"x": 1}) is None
    await _add(lib, p, meta={"quality": "max"})                                # the same file fetched again
    meta = (await lib.get_asset(a))["meta"]
    assert meta["tags"] == ["rag"] and meta["quality"] == "max"


# ---------------------------------------------------------------------------
# media_service
# ---------------------------------------------------------------------------

def test_signed_links_verify_and_refuse_forged_expired_or_malformed():
    tok = ms.sign_link(TOKEN, 7, "d", hours=1, now=1000)
    assert ms.verify_link(TOKEN, tok, now=1000) == (7, "d")
    body, sig = tok.rsplit(".", 1)
    for bad in (f"{body}.{'0' * 32}", tok.replace("7.", "8.", 1), tok.replace(".d.", ".f."), "nonsense", ""):
        with pytest.raises(ms.MediaError):
            ms.verify_link(TOKEN, bad, now=1000)
    with pytest.raises(ms.MediaError):
        ms.verify_link(TOKEN, tok, now=1000 + 3601)                              # expired
    with pytest.raises(ms.MediaError):
        ms.verify_link("999:other-token", tok, now=1000)                         # another bot's key
    with pytest.raises(ValueError):
        ms.sign_link(TOKEN, 7, "x")


async def test_settings_validate_every_key_before_saving(lib):
    assert (await ms.load_settings(lib)) == {**ms.DEFAULTS, "video_quality": "auto"}
    for bad in ({"mp3_quality": "999"}, {"language": "english"}, {"timestamps": "yes"}, {"link_hours": 0},
                {"link_hours": True}, {"video_quality": "huge"}, {"color": "red"}):
        with pytest.raises(ValueError):
            await ms.save_settings(lib, {"auto_tag": False, **bad})
    assert (await ms.load_settings(lib))["auto_tag"] is True                    # nothing saved by a refusal
    out = await ms.save_settings(lib, {"video_quality": "720", "mp3_quality": "v0", "language": "hy",
                                       "timestamps": True, "auto_tag": False, "link_hours": 48})
    assert out["video_quality"] == "720" and await lib.get_setting("media_quality") == "720"
    assert out["mp3_quality"] == "v0" and out["language"] == "hy" and out["link_hours"] == 48


async def test_jobs_record_success_and_failure():
    async def ok(job):
        job.pct = 10.0
        return 5

    async def boom(job):
        raise ms.MediaError("the download failed: no video")

    seen = []

    async def then(job):
        seen.append(job.state)

    j1, j2 = ms.start_job("fetch", "a", ok, then=then), ms.start_job("fetch", "b", boom, then=then)
    for _ in range(50):
        if j1.state != "running" and j2.state != "running":
            break
        await __import__("asyncio").sleep(0.01)
    assert (j1.state, j1.asset_id, j1.pct) == ("done", 5, 100.0)
    assert (j2.state, j2.error) == ("failed", "the download failed: no video")
    assert sorted(seen) == ["done", "failed"] and ms.JOBS[j1.id] is j1


async def test_delete_removes_item_parts_and_sidecars_but_never_outside_the_roots(lib, roots, tmp_path):
    v = _file(roots.media_root, "youtube/talk.mp4")
    thumb, info = _file(roots.media_root, "youtube/talk.jpg"), _file(roots.media_root, "youtube/talk.info.json", b"{}")
    a = await _add(lib, v, meta={"thumb": str(thumb), "info_json": str(info)})
    mp3 = await _add(lib, _file(roots.media_root, "youtube/talk.mp3"), meta={"derived_from": a, "thumb": str(thumb)})
    assert await ms.delete(lib, roots, mp3) == ["talk.mp3"]                     # a part: the parent's thumb stays
    assert thumb.exists() and await lib.get_asset(a) is not None
    tr = await _add(lib, _file(roots.transcripts_root, "youtube/talk.txt"), kind="transcript",
                    meta={"media_asset_id": a})
    removed = await ms.delete(lib, roots, a)
    assert sorted(removed) == ["talk.info.json", "talk.jpg", "talk.mp4", "talk.txt"]
    assert await lib.get_assets([a, tr]) == []
    outside = _file(tmp_path / "elsewhere", "keep.mp4")
    o = await _add(lib, outside)
    assert await ms.delete(lib, roots, o) == [] and outside.exists()           # the row goes, the file stays


def test_import_kind():
    assert ms.import_kind("Talk.MP4") == "media" and ms.import_kind("a.opus") == "media"
    assert ms.import_kind("notes.srt") == "transcript"
    with pytest.raises(ms.MediaError, match=r"\.exe"):
        ms.import_kind("setup.exe")


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------

def test_normalize_tag_and_extract_links():
    assert media_tags.normalize_tag("#Claude Code") == "claude-code"
    assert media_tags.normalize_tag("  RAG!! ") == "rag"
    assert media_tags.normalize_tag("---") == ""
    links = media_tags.extract_links(
        "Repo: github.com/anthropics/claude-code. Also https://github.com/orgs/acme/people and "
        "https://example.com/docs. and https://github.com/anthropics/claude-code.git")
    assert links == [{"url": "https://github.com/anthropics/claude-code", "kind": "repo", "label": "anthropics/claude-code"},
                     {"url": "https://example.com/docs", "kind": "link", "label": ""}]


def test_parse_refuses_an_answer_without_tags():
    with pytest.raises(ValueError):
        media_tags._parse('{"tags": [], "group": "x"}')
    out = media_tags._parse('```json\n{"tags": ["RAG", "rag", "Lang Graph"], "group": " AI agents ", '
                            '"summary": "s", "links": [{"url": "https://a.b", "label": "A"}, {"url": "ftp://no"}]}\n```')
    assert out == {"tags": ["rag", "lang-graph"], "group": "AI agents", "summary": "s",
                   "links": [{"url": "https://a.b", "label": "A"}]}


async def test_tag_stores_a_pending_suggestion_with_description_links(lib, roots, monkeypatch):
    info = _file(roots.media_root, "x/talk.info.json",
                 json.dumps({"description": "Code: https://github.com/acme/agent-kit"}).encode())
    a = await _add(lib, _file(roots.media_root, "x/talk.mp4"), title="Agents talk", meta={"info_json": str(info)})
    await ms.accept(lib, await _add(lib, _file(roots.media_root, "x/old.mp4")), tags=["rag"], group="LLM engineering")
    seen = {}

    async def snapshot():
        return SimpleNamespace(profile="test")

    def complete(role, messages, *, engines, validate, **kw):
        seen["role"], seen["prompt"] = role, messages[1].content
        return SimpleNamespace(parsed=validate('{"tags": ["AI Agents", "rag"], "group": "AI agents", '
                                               '"summary": "How to build agents.", "links": []}'),
                               records=[SimpleNamespace(served_model="m1")])

    monkeypatch.setattr(media_tags.registry, "active_snapshot", snapshot)
    monkeypatch.setattr(media_tags.llm, "complete", complete)
    sug = await ms.tag(lib, a)
    assert seen["role"] == "media.tag"
    assert '"rag"' in seen["prompt"] and "LLM engineering" in seen["prompt"]     # the vocabulary is offered
    assert "github.com/acme/agent-kit" in seen["prompt"]                        # the description is material
    assert sug["tags"] == ["ai-agents", "rag"] and sug["model"] == "m1"
    assert sug["links"][0] == {"url": "https://github.com/acme/agent-kit", "label": "acme/agent-kit", "kind": "repo"}
    meta = (await lib.get_asset(a))["meta"]
    assert meta["review"] == "pending" and meta["suggest"]["group"] == "AI agents" and "tags" not in meta


# ---------------------------------------------------------------------------
# the media API
# ---------------------------------------------------------------------------

@pytest.fixture
async def client(lib, roots, tmp_path):
    bot = FakeBot()
    app = make_app(FakeRuns(), reports_root=roots.reports_root, library=lib, bot=bot, settings=roots,
                   bot_data={"webapp_ok": True, "webapp_public_url": "https://console.example"})
    await ms.save_settings(lib, {"auto_tag": False})                           # no background tag jobs here
    async with TestClient(TestServer(app)) as c:
        c.bot, c.fake = bot, app[__import__("argus.webapp.api", fromlist=["DEPS"]).DEPS].runs
        yield c


async def test_list_and_item_with_signed_links_that_serve_ranges(client, lib, roots):
    a = await _add(lib, _file(roots.media_root, "youtube/talk.mp4", bytes(range(256)) * 4), platform="youtube")
    r = await client.get("/api/media", headers=auth())
    body = await r.json()
    assert r.status == 200 and body["total"] == 1 and body["items"][0]["type"] == "video"
    r = await client.get(f"/api/media/{a}", headers=auth())
    item = (await r.json())["item"]
    assert item["links"]["file"].startswith("/files/") and "thumb" not in item["links"]
    r = await client.get(item["links"]["file"], headers={"Range": "bytes=0-9"})   # no initData: the token is it
    assert r.status == 206 and await r.read() == bytes(range(10))
    r = await client.get(item["links"]["download"])
    assert r.status == 200 and r.headers["Content-Disposition"].startswith("attachment")
    assert r.headers["Access-Control-Allow-Origin"] == "https://web.telegram.org"
    forged = item["links"]["file"].replace("/files/", "/files/9")
    assert (await client.get(forged)).status == 403
    assert (await client.get("/api/media", headers={})).status == 401         # the API itself still needs initData


async def test_files_refuses_a_path_outside_the_roots(client, lib, tmp_path):
    a = await _add(lib, _file(tmp_path / "elsewhere", "secret.mp4"))
    tok = ms.sign_link(TOKEN, a, "f")
    assert (await client.get(f"/files/{tok}/secret.mp4")).status == 404


async def test_chunked_upload_of_a_transcript(client, lib, roots, monkeypatch):
    data = b"# my notes\nline one\n" * 1000
    r = await client.post("/api/media/uploads", json={"name": "Talk notes.md", "size": len(data)}, headers=auth())
    up = await r.json()
    uid = up["upload_id"]
    half = len(data) // 2
    assert (await client.put(f"/api/media/uploads/{uid}?offset=5", data=data[:half], headers=auth())).status == 409
    assert (await client.post(f"/api/media/uploads/{uid}/finish", json={}, headers=auth())).status == 409
    for off in (0, half):
        r = await client.put(f"/api/media/uploads/{uid}?offset={off}", data=data[off:off + half] if off == 0 else data[half:],
                             headers=auth())
        assert r.status == 200
    r = await client.post(f"/api/media/uploads/{uid}/finish", json={"transcribe": True}, headers=auth())
    body = await r.json()
    assert r.status == 200 and body["item"]["kind"] == "transcript" and body["jobs"] == []
    a = await lib.get_asset(body["item"]["id"])
    assert Path(a["path"]).read_bytes() == data and Path(a["path"]).parent == Path(roots.transcripts_root) / "upload"
    assert a["title"] == "Talk notes" and a["meta"]["imported"] is True
    assert list((Path(roots.media_root) / ".uploads").iterdir()) == []         # the part was moved
    r = await client.post("/api/media/uploads", json={"name": "setup.exe", "size": 10}, headers=auth())
    assert r.status == 415
    monkeypatch.setenv("ARGUS_MEDIA_UPLOAD_MAX_MB", "1")
    r = await client.post("/api/media/uploads", json={"name": "big.mp4", "size": 2 * 1048576}, headers=auth())
    assert r.status == 413


async def test_tags_attach_and_continue(client, lib, roots):
    v = await _add(lib, _file(roots.media_root, "x/talk.mp4"))
    t = await _add(lib, _file(roots.transcripts_root, "x/talk.txt", b"hello"), kind="transcript",
                   meta={"media_asset_id": v})
    r = await client.put(f"/api/media/{v}/tags", json={"tags": ["Claude Code", "rag"], "group": "Dev tooling"},
                         headers=auth())
    assert (await r.json())["item"]["tags"] == ["claude-code", "rag"]
    await lib.create_run(run_id="r1", thread_id="t1", chat_id=42, topic="agents", status="done",
                         config_json="{}", depth="quick", origin="telegram")
    r = await client.post(f"/api/media/{v}/attach", json={"run_id": "r1"}, headers=auth())
    assert r.status == 409                                                      # a video: transcribe it first
    r = await client.post(f"/api/media/{t}/attach", json={"run_id": "r1"}, headers=auth())
    body = await r.json()
    assert r.status == 200 and body["pending"] == 1 and "Continue the run" in body["message"]
    assert (await client.post(f"/api/media/{t}/attach", json={"run_id": "nope"}, headers=auth())).status == 404
    client.server.app[__import__("argus.webapp.api", fromlist=["DEPS"]).DEPS].bot_data["graph"] = object()
    r = await client.post("/api/runs/r1/continue", headers=auth())
    assert r.status == 200 and (await r.json())["outcome"] == "started"
    name, call = client.fake.calls[-1]
    assert name == "continue_run" and call["row"]["run_id"] == "r1"
    assert [s["local_path"] for s in call["appended"]] == [str(Path(roots.transcripts_root) / "x/talk.txt")]


async def test_send_a_small_file_and_a_link_for_a_big_one(client, lib, roots, monkeypatch):
    t = await _add(lib, _file(roots.transcripts_root, "x/notes.txt", b"hello"), kind="transcript")
    r = await client.post(f"/api/media/{t}/send", headers=auth())
    assert (await r.json())["outcome"] == "sent" and client.bot.documents[-1]["data"] == b"hello"
    monkeypatch.setattr("argus.bot.common._TG_UPLOAD_CAP_BYTES", 3)
    r = await client.post(f"/api/media/{t}/send", headers=auth())
    assert (await r.json())["outcome"] == "link" and "download link" in client.bot.messages[-1]["text"]


# ---------------------------------------------------------------------------
# the chat: files sent to the bot, the mx:* buttons
# ---------------------------------------------------------------------------

def _chat(monkeypatch, lib, roots, *, public="https://console.example"):
    from unittest.mock import AsyncMock, MagicMock

    from argus.bot import media as bot_media
    monkeypatch.setattr(bot_media, "_allowed", lambda _s, _uid: True)
    monkeypatch.setattr(bot_media, "get_settings", lambda: SimpleNamespace(
        telegram_bot_token=TOKEN, media_root=roots.media_root, transcripts_root=roots.transcripts_root))
    app = MagicMock()
    app.bot_data = {"library": lib, "webapp_public_url": public}
    app.bot.send_message = AsyncMock()
    ctx = MagicMock()
    ctx.application = app
    note = MagicMock()
    note.edit_text = AsyncMock()
    msg = MagicMock(spec=["video", "audio", "voice", "video_note", "document", "message_id", "chat_id", "caption",
                          "reply_text"])
    msg.video = msg.audio = msg.voice = msg.video_note = msg.document = None
    msg.message_id, msg.chat_id, msg.caption = 5, 42, None
    msg.reply_text = AsyncMock(return_value=note)
    update = MagicMock()
    update.effective_user.id = 42
    update.message = msg
    return bot_media, update, ctx, msg, note


async def test_a_file_over_the_bot_download_cap_points_to_the_console_upload(lib, roots, monkeypatch):
    bot_media, update, ctx, msg, _ = _chat(monkeypatch, lib, roots)
    msg.video = SimpleNamespace(file_name="lecture.mp4", file_size=300 * 1048576)
    await bot_media.handle_file(update, ctx)
    text, kw = msg.reply_text.await_args.args[0], msg.reply_text.await_args.kwargs
    assert "only up to 20 MB" in text
    assert kw["reply_markup"].inline_keyboard[0][0].web_app.url == "https://console.example/#/media/get"
    assert (await lib.search_assets())[1] == 0


async def test_a_document_of_another_type_is_refused(lib, roots, monkeypatch):
    bot_media, update, ctx, msg, _ = _chat(monkeypatch, lib, roots)
    msg.document = SimpleNamespace(file_name="tool.exe", file_size=10)
    await bot_media.handle_file(update, ctx)
    assert msg.reply_text.await_args.args[0].startswith("⚠️ Not saved")


async def test_a_sent_transcript_is_saved_with_its_buttons(lib, roots, monkeypatch):
    await ms.save_settings(lib, {"auto_tag": False})
    bot_media, update, ctx, msg, note = _chat(monkeypatch, lib, roots)

    class File:
        async def download_to_drive(self, custom_path):
            Path(custom_path).write_text("hello from the chat", encoding="utf-8")

    async def get_file():
        return File()

    msg.document = SimpleNamespace(file_name="Interview.srt", file_size=20, get_file=get_file)
    msg.caption = "Interview with the LangGraph team\nmore text"
    await bot_media.handle_file(update, ctx)
    rows, total = await lib.search_assets()
    assert total == 1 and rows[0]["kind"] == "transcript" and rows[0]["platform"] == "telegram"
    assert rows[0]["title"] == "Interview with the LangGraph team"
    assert Path(rows[0]["path"]).read_text(encoding="utf-8") == "hello from the chat"
    text, kb = note.edit_text.await_args.args[0], note.edit_text.await_args.kwargs["reply_markup"]
    assert f"asset:{rows[0]['asset_id']}" in text and "/append" in text
    labels = [b.text for row in kb.inline_keyboard for b in row]
    assert labels == ["🏷 Tags", "📱 Open"]                                    # a transcript: no mp3, no ASR


async def test_accept_in_chat_confirms_the_suggestion(lib, roots, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    bot_media, update, ctx, *_ = _chat(monkeypatch, lib, roots)
    a = await _add(lib, _file(roots.media_root, "x/clip.mp4"))
    await lib.update_asset_meta(a, {"review": "pending", "suggest": {"tags": ["rag", "evals"], "group": "LLM engineering"}})
    monkeypatch.setattr(bot_media, "_get_library", lambda _ctx: lib)
    update.callback_query = MagicMock()
    update.callback_query.message.chat.id = 42
    update.callback_query.edit_message_reply_markup = AsyncMock()
    await bot_media.handle_item_callback(update, ctx, "ok", str(a))
    meta = (await lib.get_asset(a))["meta"]
    assert meta["tags"] == ["rag", "evals"] and meta["group"] == "LLM engineering" and meta["review"] == "done"
    assert "✅ Tagged" in ctx.application.bot.send_message.await_args.kwargs["text"]


async def test_settings_route_refuses_bad_values(client):
    r = await client.put("/api/media/settings", json={"mp3_quality": "1000"}, headers=auth())
    assert r.status == 400 and "mp3_quality" in (await r.json())["error"]["message"]
    r = await client.get("/api/media/settings", headers=auth())
    assert (await r.json())["settings"]["auto_tag"] is False
