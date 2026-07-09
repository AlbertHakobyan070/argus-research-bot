"""Hermetic tests for the persistent transcript cache ageing cleanup.

The /transcript command persists each transcript's plain text at
``tempfile.gettempdir()/argus_ytt_out/<video_id>.txt`` so it can be
attached to a Telegram chat. That directory grows unbounded across
restarts; ``cleanup_argus_ytt_cache()`` is invoked once at bot
startup and drops files older than ``max_age_seconds``.

These tests exercise the cleanup in isolation against a synthetic
tempdir (NOT the real ``%TEMP%/argus_ytt_out``) — the function
accepts an explicit ``root`` parameter for testability and so the
suite stays hermetic on Windows.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from argus.cache_cleanup import cleanup_argus_ytt_cache


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _touch(p: Path, mtime: float) -> Path:
    """Write ``p`` and set its mtime so ``age = now - mtime`` is reproducible."""
    p.write_text("x", encoding="utf-8")
    # two-step set to defeat filesystem timestamp resolution on Windows
    os.utime(p, (mtime, mtime))
    return p


# ---------------------------------------------------------------------------
# happy-path: deletes only files older than max_age_seconds
# ---------------------------------------------------------------------------


def test_deletes_files_older_than_max_age(tmp_path: Path):
    now = time.time()
    old = _touch(tmp_path / "old.txt", now - 48 * 3600)        # 2 days
    fresh = _touch(tmp_path / "fresh.txt", now - 60)           # 1 min
    borderline = _touch(tmp_path / "border.txt", now - 24 * 3600)  # exactly 24h

    # Pin `now` to the same wall-clock the test captured before
    # backdating mtimes — otherwise the function's per-call
    # ``time.time()`` drifts past the borderline between the touch
    # and the cutoff calculation.
    removed = cleanup_argus_ytt_cache(
        root=tmp_path, max_age_seconds=24 * 3600, now=now,
    )

    assert removed == 1
    assert not old.exists()
    assert fresh.exists()
    # borderline: 24h old file vs 24h cutoff. The cleanup uses
    # mtime < cutoff (strict less-than), so an exactly-24h file is kept.
    assert borderline.exists()


def test_keeps_recent_files(tmp_path: Path):
    now = time.time()
    a = _touch(tmp_path / "a.txt", now - 5)
    b = _touch(tmp_path / "b.txt", now - 100)

    removed = cleanup_argus_ytt_cache(root=tmp_path, max_age_seconds=3600)
    assert removed == 0
    assert a.exists() and b.exists()


def test_aggressive_cutoff_removes_everything(tmp_path: Path):
    now = time.time()
    _touch(tmp_path / "x.txt", now - 1)
    _touch(tmp_path / "y.txt", now - 10)

    removed = cleanup_argus_ytt_cache(root=tmp_path, max_age_seconds=0)
    # max_age_seconds=0 -> cutoff = now -> every file's mtime is < cutoff
    assert removed == 2
    assert not (tmp_path / "x.txt").exists()
    assert not (tmp_path / "y.txt").exists()


# ---------------------------------------------------------------------------
# directory-missing & empty roots are no-ops
# ---------------------------------------------------------------------------


def test_missing_root_directory_is_noop(tmp_path: Path):
    ghost = tmp_path / "does_not_exist"
    assert not ghost.exists()
    # Should not raise, should report zero deletions.
    assert cleanup_argus_ytt_cache(root=ghost) == 0


def test_empty_root_returns_zero(tmp_path: Path):
    assert cleanup_argus_ytt_cache(root=tmp_path) == 0


# ---------------------------------------------------------------------------
# non-file entries are ignored (subdirs, .info.json sidecars, etc.)
# ---------------------------------------------------------------------------


def test_subdirectories_are_not_pruned(tmp_path: Path):
    """We only age out *files*. A subdir under argus_ytt_out may carry
    bulk state we don't want to risk-traverse; the cleanup skips it."""
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "inside.txt").write_text("keep me", encoding="utf-8")
    os.utime(sub, (time.time() - 7 * 24 * 3600,) * 2)

    removed = cleanup_argus_ytt_cache(root=tmp_path, max_age_seconds=3600)
    assert removed == 0
    assert sub.exists()
    assert (sub / "inside.txt").exists()


def test_vtt_and_info_json_sidecars_are_aged_out(tmp_path: Path):
    """The cleanup operates on any file under the cache root,
    including .vtt sidecars and .info.json metadata files left from
    yt-dlp. Both flavours get aged out by the same rule."""
    now = time.time()
    _touch(tmp_path / "vid.en.vtt", now - 48 * 3600)
    _touch(tmp_path / "vid.info.json", now - 48 * 3600)
    _touch(tmp_path / "vid.fresh.vtt", now - 10)

    removed = cleanup_argus_ytt_cache(root=tmp_path, max_age_seconds=24 * 3600)
    assert removed == 2
    assert (tmp_path / "vid.fresh.vtt").exists()


# ---------------------------------------------------------------------------
# error-tolerance: a stale file disappears mid-scan (e.g. file removed
# concurrently by another process). The loop must not crash.
# ---------------------------------------------------------------------------


def test_missing_file_between_scan_and_unlink_is_swallowed(tmp_path: Path):
    """Race: an out-of-process cleanup removes a file after our stat.
    We should not raise — the cleanup is best-effort."""
    now = time.time()
    target = _touch(tmp_path / "ghost.txt", now - 48 * 3600)

    real_unlink = Path.unlink

    def flaky_unlink(self, *a, **kw):
        if self == target:
            raise FileNotFoundError(2, "No such file or directory", str(self))
        return real_unlink(self, *a, **kw)

    with patch.object(Path, "unlink", flaky_unlink):
        removed = cleanup_argus_ytt_cache(root=tmp_path, max_age_seconds=3600)
    assert removed == 1  # counted as removed even though unlink raised


# ---------------------------------------------------------------------------
# default-root: when no root is passed, the function targets the
# real ``%TEMP%/argus_ytt_out``. We don't want this test touching
# the real filesystem, so we monkeypatch tempfile.gettempdir.
# ---------------------------------------------------------------------------


def test_default_root_is_tempdir_argus_ytt_out(tmp_path: Path, monkeypatch):
    sentinel_dir = tmp_path / "argus_ytt_out"
    sentinel_dir.mkdir()
    now = time.time()
    old = _touch(sentinel_dir / "junk.txt", now - 48 * 3600)
    fresh = _touch(sentinel_dir / "kept.txt", now - 10)

    import tempfile as _tempfile
    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(tmp_path))

    removed = cleanup_argus_ytt_cache(max_age_seconds=24 * 3600)
    assert removed == 1
    assert not old.exists()
    assert fresh.exists()


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "age_seconds,max_age,expected_removed",
    [
        (10, 5, 1),       # fresh but past cutoff
        (5, 10, 0),       # within the window
        (3600, 60, 1),    # an hour old, 1-min cutoff
        (60, 3600, 0),    # 1-min old, 1-hour cutoff
    ],
)
def test_param_age_thresholds(tmp_path: Path, age_seconds, max_age, expected_removed):
    now = time.time()
    _touch(tmp_path / "f.txt", now - age_seconds)
    removed = cleanup_argus_ytt_cache(root=tmp_path, max_age_seconds=max_age)
    assert removed == expected_removed
