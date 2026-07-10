"""Library registry tests — SQLite-backed runs / assets / run_sources.

Phase 1 of the v2 rebuild: every research run and every vault asset
(media, transcript, report) is registered in ``argus_library.sqlite``
so runs can be listed (/status, /runs), resumed across bot restarts
(per-run thread ids), appended-to (/append), and deleted (/delete)
from Telegram.

Hermetic: each test gets a fresh DB under tmp_path. No network, no
Telegram, no LLM.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from argus.library import (
    ASSET_KINDS,
    Library,
    VALID_RUN_STATUSES,
    mirror_run_md,
    new_run_id,
)


@pytest.fixture
async def lib(tmp_path):
    l = Library(tmp_path / "lib.sqlite")
    await l.open()
    yield l
    await l.close()


async def _seed_run(lib: Library, *, run_id: str = "ab12cd34",
                    chat_id: int = 111, topic: str = "test topic") -> dict:
    return await lib.create_run(
        run_id=run_id,
        thread_id=f"tg:{chat_id}:{run_id}",
        chat_id=chat_id,
        topic=topic,
        length="short",
    )


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


def test_new_run_id_is_short_hex():
    a, b = new_run_id(), new_run_id()
    assert len(a) == 8 and len(b) == 8
    int(a, 16)  # must be valid hex
    assert a != b, "run ids must be unique"


async def test_create_and_get_run(lib: Library):
    created = await _seed_run(lib)
    assert created["run_id"] == "ab12cd34"
    assert created["status"] == "planning"
    got = await lib.get_run("ab12cd34")
    assert got is not None
    assert got["thread_id"] == "tg:111:ab12cd34"
    assert got["topic"] == "test topic"
    assert got["mode"] == "deep"
    assert got["created_at"] and got["updated_at"]


async def test_get_run_missing_returns_none(lib: Library):
    assert await lib.get_run("deadbeef") is None


async def test_thread_id_must_be_unique(lib: Library):
    await _seed_run(lib, run_id="aaaa0000")
    with pytest.raises(sqlite3.IntegrityError):
        await lib.create_run(
            run_id="bbbb1111",
            thread_id="tg:111:aaaa0000",  # duplicate thread
            chat_id=111,
            topic="x",
        )


async def test_resolve_run_exact_and_prefix(lib: Library):
    await _seed_run(lib, run_id="ab12cd34", chat_id=1)
    await _seed_run(lib, run_id="ef56ab78", chat_id=1)
    # Exact id resolves.
    r = await lib.resolve_run(1, "ab12cd34")
    assert r and r["run_id"] == "ab12cd34"
    # Unique prefix resolves.
    r = await lib.resolve_run(1, "ef")
    assert r and r["run_id"] == "ef56ab78"
    # Wrong chat does not resolve another chat's runs.
    assert await lib.resolve_run(2, "ab12") is None
    # Unknown prefix resolves to nothing.
    assert await lib.resolve_run(1, "zz") is None


async def test_resolve_run_ambiguous_prefix_returns_none(lib: Library):
    await _seed_run(lib, run_id="ab120000", chat_id=1)
    await _seed_run(lib, run_id="ab121111", chat_id=1)
    assert await lib.resolve_run(1, "ab12") is None, (
        "an ambiguous prefix must not silently pick one run"
    )


async def test_list_runs_newest_first_and_limit(lib: Library):
    for i in range(5):
        await _seed_run(lib, run_id=f"aaaa000{i}", topic=f"t{i}")
    runs = await lib.list_runs(chat_id=111, limit=3)
    assert len(runs) == 3
    assert [r["run_id"] for r in runs] == ["aaaa0004", "aaaa0003", "aaaa0002"]
    # Other chats see nothing.
    assert await lib.list_runs(chat_id=999) == []


async def test_set_run_status_valid_transition(lib: Library):
    await _seed_run(lib)
    assert await lib.set_run_status("ab12cd34", "awaiting_plan") is True
    got = await lib.get_run("ab12cd34")
    assert got["status"] == "awaiting_plan"


async def test_set_run_status_rejects_unknown_status(lib: Library):
    await _seed_run(lib)
    with pytest.raises(ValueError):
        await lib.set_run_status("ab12cd34", "totally_bogus")


async def test_set_run_status_missing_run_returns_false(lib: Library):
    assert await lib.set_run_status("deadbeef", "done") is False


async def test_set_run_status_can_attach_report_dir(lib: Library):
    await _seed_run(lib)
    await lib.set_run_status("ab12cd34", "done",
                             report_dir=r"A:\vault\history\run1")
    got = await lib.get_run("ab12cd34")
    assert got["report_dir"] == r"A:\vault\history\run1"


async def test_delete_run_cascades_run_sources(lib: Library):
    await _seed_run(lib)
    await lib.add_run_source("ab12cd34", "https://example.com/a")
    await lib.add_run_source("ab12cd34", "asset:5", kind="asset")
    assert len(await lib.pending_sources("ab12cd34")) == 2
    assert await lib.delete_run("ab12cd34") is True
    assert await lib.get_run("ab12cd34") is None
    assert await lib.pending_sources("ab12cd34") == []


# ---------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------


async def test_add_asset_and_get(lib: Library):
    aid = await lib.add_asset(
        kind="media", platform="youtube",
        source_url="https://youtu.be/x", media_id="x",
        title="A video", path=r"A:\vault\media\youtube\x.mp4",
        size_bytes=1234, duration_s=61.5, meta={"height": 1080},
    )
    assert isinstance(aid, int) and aid > 0
    rows = await lib.get_assets([aid])
    assert len(rows) == 1
    a = rows[0]
    assert a["kind"] == "media"
    assert a["platform"] == "youtube"
    assert a["bytes"] == 1234
    assert a["meta"] == {"height": 1080}


async def test_add_asset_rejects_unknown_kind(lib: Library):
    with pytest.raises(ValueError):
        await lib.add_asset(kind="weird", path="p")


async def test_add_asset_upserts_on_same_kind_and_path(lib: Library):
    aid1 = await lib.add_asset(kind="transcript", path="t.txt", title="v1")
    aid2 = await lib.add_asset(kind="transcript", path="t.txt", title="v2",
                               size_bytes=99)
    assert aid1 == aid2, "same (kind, path) must not create a duplicate row"
    a = (await lib.get_assets([aid1]))[0]
    assert a["title"] == "v2"
    assert a["bytes"] == 99


async def test_list_assets_filters_by_kind(lib: Library):
    await lib.add_asset(kind="media", path="m1.mp4")
    await lib.add_asset(kind="media", path="m2.mp4")
    await lib.add_asset(kind="transcript", path="t1.txt")
    media = await lib.list_assets(kind="media")
    assert {a["path"] for a in media} == {"m1.mp4", "m2.mp4"}
    everything = await lib.list_assets()
    assert len(everything) == 3


async def test_delete_assets(lib: Library):
    a1 = await lib.add_asset(kind="media", path="m1.mp4")
    a2 = await lib.add_asset(kind="media", path="m2.mp4")
    deleted = await lib.delete_assets([a1, a2])
    assert deleted == 2
    assert await lib.get_assets([a1, a2]) == []


# ---------------------------------------------------------------------------
# run_sources (append/continue backing store)
# ---------------------------------------------------------------------------


async def test_run_sources_pending_then_marked(lib: Library):
    await _seed_run(lib)
    await lib.add_run_source("ab12cd34", "https://example.com/a")
    await lib.add_run_source("ab12cd34", "https://example.com/b")
    pending = await lib.pending_sources("ab12cd34")
    assert {p["ref"] for p in pending} == {
        "https://example.com/a", "https://example.com/b"}
    await lib.mark_sources("ab12cd34", ["https://example.com/a"], "ingested")
    pending = await lib.pending_sources("ab12cd34")
    assert [p["ref"] for p in pending] == ["https://example.com/b"]


async def test_add_run_source_dedupes(lib: Library):
    await _seed_run(lib)
    await lib.add_run_source("ab12cd34", "https://example.com/a")
    await lib.add_run_source("ab12cd34", "https://example.com/a")
    assert len(await lib.pending_sources("ab12cd34")) == 1


async def test_mark_sources_rejects_unknown_status(lib: Library):
    await _seed_run(lib)
    await lib.add_run_source("ab12cd34", "https://example.com/a")
    with pytest.raises(ValueError):
        await lib.mark_sources("ab12cd34", ["https://example.com/a"], "nope")


# ---------------------------------------------------------------------------
# vault mirror (human-readable run.md next to the report)
# ---------------------------------------------------------------------------


def test_mirror_run_md_writes_into_report_dir(tmp_path):
    report_dir = tmp_path / "20260710_topic_short"
    report_dir.mkdir()
    run = {
        "run_id": "ab12cd34", "thread_id": "tg:111:ab12cd34",
        "chat_id": 111, "topic": "LangGraph vs LangChain",
        "length": "short", "mode": "deep", "status": "done",
        "report_dir": str(report_dir),
        "created_at": "2026-07-10T00:00:00+00:00",
        "updated_at": "2026-07-10T00:10:00+00:00",
    }
    out = mirror_run_md(run, sources=[
        {"ref": "https://example.com/a", "status": "ingested"},
    ])
    assert out == report_dir / "run.md"
    text = out.read_text(encoding="utf-8")
    assert "LangGraph vs LangChain" in text
    assert "ab12cd34" in text
    assert "tg:111:ab12cd34" in text
    assert "https://example.com/a" in text


def test_mirror_run_md_without_report_dir_returns_none(tmp_path):
    run = {"run_id": "x", "topic": "t", "report_dir": None}
    assert mirror_run_md(run) is None
