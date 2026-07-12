"""SQLite registry for Argus research runs and vault assets.

Phase 1 of the v2 rebuild. One aiosqlite connection, opened once at bot
startup (PTB ``post_init``) and closed at shutdown. Three tables:

- ``runs``         — one row per /research or /ask run. ``thread_id`` is
                     the per-run LangGraph checkpoint thread
                     (``tg:<chat>:<run8>``), so any run can be re-attached
                     after a bot restart and continued via /continue.
- ``assets``       — every file Argus writes into the DS vault (media,
                     transcript, report). ``UNIQUE(kind, path)`` so
                     re-downloading the same target upserts instead of
                     duplicating.
- ``run_sources``  — sources appended to a run (/append) awaiting the
                     next /continue. ``ref`` is a URL or ``asset:<id>``.

The DB itself lives in the project directory (NOT the synced Obsidian
vault — SQLite WAL + sync tooling is a corruption hazard). The vault
gets write-once files only; ``mirror_run_md`` writes the human-readable
``run.md`` sidecar into the run's report folder.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger("argus.library")

VALID_RUN_STATUSES: tuple[str, ...] = (
    "planning",         # /research accepted, planner running
    "awaiting_plan",    # paused at the plan-approval HITL gate
    "running",          # resumed, pipeline executing
    "awaiting_report",  # paused at the report-preview HITL gate
    "done",             # delivered (or report exists and run closed)
    "cancelled",
    "error",
)

ASSET_KINDS: tuple[str, ...] = ("media", "transcript", "report")

_SOURCE_STATUSES: tuple[str, ...] = ("pending", "ingested", "failed")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
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
CREATE INDEX IF NOT EXISTS idx_runs_chat ON runs(chat_id, created_at DESC);

CREATE TABLE IF NOT EXISTS assets (
  asset_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  kind       TEXT NOT NULL CHECK (kind IN ('media','transcript','report')),
  platform   TEXT,
  source_url TEXT,
  media_id   TEXT,
  title      TEXT,
  path       TEXT NOT NULL,
  bytes      INTEGER NOT NULL DEFAULT 0,
  duration_s REAL,
  meta_json  TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  UNIQUE (kind, path)
);
CREATE INDEX IF NOT EXISTS idx_assets_kind ON assets(kind, created_at DESC);

CREATE TABLE IF NOT EXISTS settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_sources (
  run_id      TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
  ref         TEXT NOT NULL,
  kind        TEXT NOT NULL DEFAULT 'url',
  status      TEXT NOT NULL DEFAULT 'pending',
  added_at    TEXT NOT NULL,
  ingested_at TEXT,
  PRIMARY KEY (run_id, ref)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """8-hex run id — short enough for Telegram callback data + typing."""
    return uuid.uuid4().hex[:8]


def _row_to_run(row: aiosqlite.Row) -> dict[str, Any]:
    return dict(row)


def _row_to_asset(row: aiosqlite.Row) -> dict[str, Any]:
    d = dict(row)
    try:
        d["meta"] = json.loads(d.pop("meta_json") or "{}")
    except json.JSONDecodeError:
        logger.warning("asset %s has corrupt meta_json", d.get("asset_id"))
        d["meta"] = {}
    return d


class Library:
    """Async registry over one aiosqlite connection.

    ``open()`` before use; all methods raise ``RuntimeError`` if called
    on an unopened library (a wiring bug, not a user error).
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        if self._db is not None:
            return
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(str(self.db_path))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.executescript(_SCHEMA)
        await db.commit()
        self._db = db

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Library not opened — call await lib.open() first")
        return self._db

    # -- runs ---------------------------------------------------------------

    async def create_run(self, *, run_id: str, thread_id: str, chat_id: int,
                         topic: str, length: str = "short",
                         mode: str = "deep",
                         status: str = "planning") -> dict[str, Any]:
        if status not in VALID_RUN_STATUSES:
            raise ValueError(f"unknown run status {status!r}")
        db = self._conn()
        ts = _now()
        await db.execute(
            "INSERT INTO runs (run_id, thread_id, chat_id, topic, length,"
            " mode, status, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, thread_id, chat_id, topic, length, mode, status, ts, ts),
        )
        await db.commit()
        run = await self.get_run(run_id)
        assert run is not None
        return run

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        db = self._conn()
        async with db.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)) as cur:
            row = await cur.fetchone()
        return _row_to_run(row) if row else None

    async def resolve_run(self, chat_id: int, ref: str) -> dict[str, Any] | None:
        """Resolve a (possibly partial) run id typed by the user.

        Chat-scoped. Returns the run only when the prefix matches exactly
        one of this chat's runs — an ambiguous prefix must never silently
        pick one.
        """
        ref = (ref or "").strip().lower()
        if not ref:
            return None
        db = self._conn()
        async with db.execute(
                "SELECT * FROM runs WHERE chat_id = ? AND run_id LIKE ?"
                " LIMIT 2", (chat_id, ref + "%")) as cur:
            rows = await cur.fetchall()
        if len(rows) != 1:
            return None
        return _row_to_run(rows[0])

    async def list_runs(self, *, chat_id: int | None = None,
                        limit: int = 10) -> list[dict[str, Any]]:
        db = self._conn()
        if chat_id is None:
            q = ("SELECT * FROM runs ORDER BY created_at DESC, rowid DESC"
                 " LIMIT ?")
            args: tuple = (limit,)
        else:
            q = ("SELECT * FROM runs WHERE chat_id = ?"
                 " ORDER BY created_at DESC, rowid DESC LIMIT ?")
            args = (chat_id, limit)
        async with db.execute(q, args) as cur:
            rows = await cur.fetchall()
        return [_row_to_run(r) for r in rows]

    async def list_runs_by_status(self, statuses: tuple[str, ...],
                                  limit: int = 50) -> list[dict[str, Any]]:
        """All runs currently in any of ``statuses`` (newest first).

        Used at startup to find runs orphaned by a crash/restart — those
        left in a non-terminal state (planning/awaiting_*/running) whose
        owning process is gone.
        """
        if not statuses:
            return []
        db = self._conn()
        marks = ",".join("?" * len(statuses))
        async with db.execute(
                f"SELECT * FROM runs WHERE status IN ({marks})"
                " ORDER BY created_at DESC, rowid DESC LIMIT ?",
                [*statuses, limit]) as cur:
            rows = await cur.fetchall()
        return [_row_to_run(r) for r in rows]

    async def set_run_status(self, run_id: str, status: str, *,
                             report_dir: str | None = None) -> bool:
        """Update a run's status (and optionally its report folder).

        Returns False when the run doesn't exist — callers on bot error
        paths must not crash the handler over a missing registry row.
        """
        if status not in VALID_RUN_STATUSES:
            raise ValueError(f"unknown run status {status!r}")
        db = self._conn()
        if report_dir is not None:
            cur = await db.execute(
                "UPDATE runs SET status = ?, report_dir = ?, updated_at = ?"
                " WHERE run_id = ?",
                (status, report_dir, _now(), run_id))
        else:
            cur = await db.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (status, _now(), run_id))
        await db.commit()
        return cur.rowcount > 0

    async def delete_run(self, run_id: str) -> bool:
        db = self._conn()
        cur = await db.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        await db.commit()
        return cur.rowcount > 0

    # -- assets ---------------------------------------------------------------

    async def add_asset(self, *, kind: str, path: str,
                        platform: str | None = None,
                        source_url: str | None = None,
                        media_id: str | None = None,
                        title: str | None = None,
                        size_bytes: int = 0,
                        duration_s: float | None = None,
                        meta: dict | None = None) -> int:
        if kind not in ASSET_KINDS:
            raise ValueError(f"unknown asset kind {kind!r}")
        db = self._conn()
        async with db.execute(
                "INSERT INTO assets (kind, platform, source_url, media_id,"
                " title, path, bytes, duration_s, meta_json, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(kind, path) DO UPDATE SET"
                "  platform=excluded.platform, source_url=excluded.source_url,"
                "  media_id=excluded.media_id, title=excluded.title,"
                "  bytes=excluded.bytes, duration_s=excluded.duration_s,"
                "  meta_json=excluded.meta_json"
                " RETURNING asset_id",
                (kind, platform, source_url, media_id, title, str(path),
                 int(size_bytes), duration_s,
                 json.dumps(meta or {}, ensure_ascii=False), _now())) as cur:
            row = await cur.fetchone()
        await db.commit()
        return int(row[0])

    async def get_assets(self, asset_ids: list[int]) -> list[dict[str, Any]]:
        if not asset_ids:
            return []
        db = self._conn()
        marks = ",".join("?" * len(asset_ids))
        async with db.execute(
                f"SELECT * FROM assets WHERE asset_id IN ({marks})"
                " ORDER BY asset_id", [int(i) for i in asset_ids]) as cur:
            rows = await cur.fetchall()
        return [_row_to_asset(r) for r in rows]

    async def get_asset_by_source(self, source_url: str,
                                  kind: str) -> dict[str, Any] | None:
        """Newest asset of ``kind`` registered for a source URL (lets the
        transcriber reuse an already-downloaded media file)."""
        if kind not in ASSET_KINDS:
            raise ValueError(f"unknown asset kind {kind!r}")
        db = self._conn()
        async with db.execute(
                "SELECT * FROM assets WHERE source_url = ? AND kind = ?"
                " ORDER BY asset_id DESC LIMIT 1", (source_url, kind)) as cur:
            row = await cur.fetchone()
        return _row_to_asset(row) if row else None

    async def list_assets(self, *, kind: str | None = None, limit: int = 20,
                          offset: int = 0) -> list[dict[str, Any]]:
        if kind is not None and kind not in ASSET_KINDS:
            raise ValueError(f"unknown asset kind {kind!r}")
        db = self._conn()
        if kind is None:
            q = ("SELECT * FROM assets ORDER BY created_at DESC,"
                 " asset_id DESC LIMIT ? OFFSET ?")
            args: tuple = (limit, offset)
        else:
            q = ("SELECT * FROM assets WHERE kind = ? ORDER BY created_at"
                 " DESC, asset_id DESC LIMIT ? OFFSET ?")
            args = (kind, limit, offset)
        async with db.execute(q, args) as cur:
            rows = await cur.fetchall()
        return [_row_to_asset(r) for r in rows]

    async def delete_assets(self, asset_ids: list[int]) -> int:
        if not asset_ids:
            return 0
        db = self._conn()
        marks = ",".join("?" * len(asset_ids))
        cur = await db.execute(
            f"DELETE FROM assets WHERE asset_id IN ({marks})",
            [int(i) for i in asset_ids])
        await db.commit()
        return cur.rowcount

    # -- settings kv (global toggles like /quality) ----------------------------

    async def get_setting(self, key: str, default: str | None = None) -> str | None:
        db = self._conn()
        async with db.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)) as cur:
            row = await cur.fetchone()
        return row[0] if row else default

    async def set_setting(self, key: str, value: str) -> None:
        db = self._conn()
        await db.execute(
            "INSERT INTO settings (key, value) VALUES (?,?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))
        await db.commit()

    # -- run_sources (append/continue) ----------------------------------------

    async def add_run_source(self, run_id: str, ref: str,
                             kind: str = "url") -> None:
        db = self._conn()
        await db.execute(
            "INSERT OR IGNORE INTO run_sources (run_id, ref, kind, status,"
            " added_at) VALUES (?,?,?,'pending',?)",
            (run_id, ref, kind, _now()))
        await db.commit()

    async def pending_sources(self, run_id: str) -> list[dict[str, Any]]:
        db = self._conn()
        async with db.execute(
                "SELECT * FROM run_sources WHERE run_id = ? AND"
                " status = 'pending' ORDER BY added_at, ref",
                (run_id,)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def mark_sources(self, run_id: str, refs: list[str],
                           status: str) -> None:
        if status not in _SOURCE_STATUSES:
            raise ValueError(f"unknown source status {status!r}")
        if not refs:
            return
        db = self._conn()
        marks = ",".join("?" * len(refs))
        ingested_at = _now() if status == "ingested" else None
        await db.execute(
            f"UPDATE run_sources SET status = ?, ingested_at = ?"
            f" WHERE run_id = ? AND ref IN ({marks})",
            [status, ingested_at, run_id, *refs])
        await db.commit()


# ---------------------------------------------------------------------------
# Vault mirror — human-readable run.md next to the report files.
# ---------------------------------------------------------------------------


def mirror_run_md(run: dict[str, Any], *,
                  sources: list[dict[str, Any]] | None = None) -> Path | None:
    """Write ``run.md`` into the run's report folder (sync — call via
    ``asyncio.to_thread`` from the bot). Returns the written path, or
    None when the run has no report folder yet (e.g. quick /ask runs).
    """
    folder = run.get("report_dir")
    if not folder:
        return None
    folder_p = Path(folder)
    folder_p.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Argus run — {run.get('topic', '(untitled)')}",
        "",
        f"- run_id: `{run.get('run_id')}`",
        f"- thread_id: `{run.get('thread_id')}`",
        f"- chat_id: `{run.get('chat_id')}`",
        f"- mode: {run.get('mode')} · length: {run.get('length')}",
        f"- status: {run.get('status')}",
        f"- created: {run.get('created_at')} · updated: {run.get('updated_at')}",
        f"- report folder: `{folder}`",
    ]
    if sources:
        lines += ["", "## Appended sources", ""]
        for s in sources:
            lines.append(f"- `{s.get('status', '?')}` — {s.get('ref', '')}")
    lines.append("")
    out = folder_p / "run.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
