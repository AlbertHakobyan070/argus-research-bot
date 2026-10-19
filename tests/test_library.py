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

import json
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
from helpers import make_run_config


@pytest.fixture
async def lib(tmp_path):
    l = Library(tmp_path / "lib.sqlite")
    await l.open()
    yield l
    await l.close()


async def _seed_run(lib: Library, *, run_id: str = "ab12cd34",
                    chat_id: int = 111, topic: str = "test topic") -> dict:
    rc = make_run_config("short")
    return await lib.create_run(
        run_id=run_id,
        thread_id=f"tg:{chat_id}:{run_id}",
        chat_id=chat_id,
        topic=topic,
        length="short",
        config_json=json.dumps(rc), depth=rc["depth"], origin=rc["origin"],
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
            config_json=json.dumps(make_run_config()), depth="quick",
            origin="telegram",
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


# ---------------------------------------------------------------------------
# settings kv (backs /quality and future global toggles)
# ---------------------------------------------------------------------------


async def test_settings_roundtrip_and_default(lib: Library):
    assert await lib.get_setting("media_quality") is None
    assert await lib.get_setting("media_quality", "auto") == "auto"
    await lib.set_setting("media_quality", "720")
    assert await lib.get_setting("media_quality") == "720"
    await lib.set_setting("media_quality", "max")   # overwrite
    assert await lib.get_setting("media_quality") == "max"


# ---------------------------------------------------------------------------
# v4 WP-D: run config columns, the guarded migration, json settings, Exa spend
# ---------------------------------------------------------------------------

# The library schema as v3 shipped it (frozen copy: the migration test must
# keep testing a pre-v4 DB even when library._SCHEMA changes later).
_PRE_V4_SCHEMA = """
CREATE TABLE runs (
  run_id      TEXT PRIMARY KEY,
  thread_id   TEXT NOT NULL UNIQUE,
  chat_id     INTEGER NOT NULL,
  topic       TEXT NOT NULL,
  length      TEXT NOT NULL DEFAULT 'short',
  mode        TEXT NOT NULL DEFAULT 'deep',
  status      TEXT NOT NULL DEFAULT 'planning',
  report_dir  TEXT,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE INDEX idx_runs_chat ON runs(chat_id, created_at DESC);
CREATE TABLE assets (
  asset_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  kind       TEXT NOT NULL CHECK (kind IN ('media','transcript','report')),
  platform   TEXT, source_url TEXT, media_id TEXT, title TEXT,
  path       TEXT NOT NULL,
  bytes      INTEGER NOT NULL DEFAULT 0,
  duration_s REAL,
  meta_json  TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  UNIQUE (kind, path)
);
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE run_sources (
  run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  ref TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'url',
  status TEXT NOT NULL DEFAULT 'pending', added_at TEXT NOT NULL,
  ingested_at TEXT, PRIMARY KEY (run_id, ref)
);
"""
_V4_COLUMNS = {"config_json", "depth", "archetype", "origin", "exa_usd"}
_TABLES = ("runs", "assets", "settings", "run_sources")


def _pre_v4_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(_PRE_V4_SCHEMA)
    ts = "2026-08-01T10:00:00+00:00"
    con.execute("INSERT INTO runs VALUES ('old00001','tg:1:old00001',1,'old topic',"
                "'long','deep','done','A:/r',?,?)", (ts, ts))
    con.execute("INSERT INTO assets (kind, path, title, created_at) VALUES "
                "('report','A:/r/report.md','old topic',?)", (ts,))
    con.execute("INSERT INTO settings VALUES ('media_quality','max')")
    con.execute("INSERT INTO run_sources VALUES ('old00001','https://x.example','url',"
                "'pending',?,NULL)", (ts,))
    con.commit()
    con.close()


def _shape(path: Path) -> tuple[dict, set]:
    """(row count per table, runs column names) read with plain sqlite3."""
    con = sqlite3.connect(path)
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _TABLES}
    cols = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    con.close()
    return counts, cols


async def test_migration_adds_v4_columns_to_a_pre_v4_db_and_keeps_rows(tmp_path):
    db = tmp_path / "pre_v4.sqlite"
    _pre_v4_db(db)
    counts_before, cols_before = _shape(db)
    assert not (_V4_COLUMNS & cols_before)

    lib = Library(db)
    await lib.open()
    old = await lib.get_run("old00001")
    assert await lib.get_run_config("old00001") is None        # pre-v4 row: no config stored
    assert await lib.get_setting("media_quality") == "max"
    assert len(await lib.pending_sources("old00001")) == 1
    await lib.close()

    counts_after, cols_after = _shape(db)
    assert counts_after == counts_before                       # additive only
    assert cols_after == cols_before | _V4_COLUMNS
    assert (old["topic"], old["length"], old["status"], old["report_dir"]) == (
        "old topic", "long", "done", "A:/r")
    assert (old["config_json"], old["depth"], old["archetype"], old["exa_usd"]) == (
        None, None, None, None)
    assert old["origin"] == "telegram"                         # the column default

    again = Library(db)                                        # idempotent second open
    await again.open()
    await again.close()
    assert _shape(db) == (counts_before, cols_after)


async def test_fresh_db_gets_the_v4_columns(tmp_path):
    l = Library(tmp_path / "fresh.sqlite")
    await l.open()
    await l.close()
    assert _V4_COLUMNS <= _shape(tmp_path / "fresh.sqlite")[1]


async def test_create_run_stores_config_depth_origin(lib: Library):
    rc = make_run_config("medium", **{"research.waves": 2})
    run = await lib.create_run(run_id="cfg00001", thread_id="tg:1:cfg00001", chat_id=1,
                               topic="t", length="medium", config_json=json.dumps(rc),
                               depth=rc["depth"], origin="webapp")
    assert (run["depth"], run["origin"], run["archetype"], run["exa_usd"]) == (
        "standard", "webapp", None, None)
    assert json.loads(run["config_json"]) == rc
    assert await lib.get_run_config("cfg00001") == rc
    assert await lib.get_run_config("deadbeef") is None


@pytest.mark.parametrize("bad", ["[]", "not json", '"text"'])
async def test_create_run_rejects_config_json_that_is_not_an_object(lib: Library, bad):
    with pytest.raises(ValueError, match="config_json"):
        await lib.create_run(run_id="bad00001", thread_id="tg:1:bad00001", chat_id=1,
                             topic="t", config_json=bad, depth="quick", origin="telegram")
    assert await lib.get_run("bad00001") is None


async def test_set_run_config_rewrites_config_length_and_depth(lib: Library):
    """The stale-length bug: Approve with another length must update the row."""
    await _seed_run(lib)                                        # short / quick
    new = make_run_config("long", **{"research.waves": 2})
    assert await lib.set_run_config("ab12cd34", new) is True
    run = await lib.get_run("ab12cd34")
    assert (run["length"], run["depth"], run["origin"]) == ("long", "deep", "telegram")
    assert await lib.get_run_config("ab12cd34") == new
    assert await lib.set_run_config("deadbeef", new) is False


async def test_update_run_sets_known_columns_only(lib: Library):
    await _seed_run(lib)
    assert await lib.update_run("ab12cd34", archetype="news", exa_usd=0.25) is True
    run = await lib.get_run("ab12cd34")
    assert (run["archetype"], run["exa_usd"]) == ("news", 0.25)
    assert await lib.update_run("deadbeef", archetype="x") is False
    with pytest.raises(ValueError, match="run_id"):
        await lib.update_run("ab12cd34", run_id="other")
    with pytest.raises(ValueError, match="bogus"):
        await lib.update_run("ab12cd34", bogus=1)
    with pytest.raises(ValueError, match="status"):
        await lib.update_run("ab12cd34", status="totally_bogus")
    with pytest.raises(ValueError, match="column"):
        await lib.update_run("ab12cd34")


async def test_json_settings_roundtrip(lib: Library):
    assert await lib.get_json("engines.overrides") is None
    assert await lib.get_json("engines.overrides", {}) == {}
    value = {"main": {"roles": {"panel.grounding": {"primary": {"provider": "gem"}}}}}
    await lib.set_json("engines.overrides", value)
    assert await lib.get_json("engines.overrides") == value
    await lib.set_setting("broken", "{not json")
    with pytest.raises(json.JSONDecodeError):
        await lib.get_json("broken")


async def test_exa_usd_month_to_date_sums_this_months_runs(lib: Library):
    for i, usd in enumerate((0.10, 0.25, None, 1.00)):
        await _seed_run(lib, run_id=f"exa0000{i}")
        if usd is not None:
            await lib.update_run(f"exa0000{i}", exa_usd=usd)
    db = lib._conn()                                            # a run from an earlier month
    await db.execute("UPDATE runs SET created_at = '2000-01-15T00:00:00+00:00'"
                     " WHERE run_id = 'exa00003'")
    await db.commit()
    assert await lib.exa_usd_month_to_date() == pytest.approx(0.35)


async def test_exa_usd_month_to_date_is_zero_without_spend(lib: Library):
    assert await lib.exa_usd_month_to_date() == 0.0
