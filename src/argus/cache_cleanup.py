"""Persistent-cache ageing utilities.

The ``/transcript`` command persists each transcript's plain text at
``tempfile.gettempdir()/argus_ytt_out/<video_id>.txt`` so it can be
attached to a Telegram chat. That directory grows unbounded across
restarts; ``cleanup_argus_ytt_cache()`` is invoked once at bot
startup and drops files older than ``max_age_seconds``.

Lifecycle:

  * ``tools.youtube_video_transcript`` writes per-call temporary
    directories via :func:`tempfile.TemporaryDirectory` (``argus_ytt_*``)
    — those are auto-cleaned by the context-manager.
  * The *persistent* deliverable lives at ``argus_ytt_out/<id>.txt``
    and outlives the call. This module is responsible for trimming it.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def cleanup_argus_ytt_cache(
    *,
    max_age_seconds: int = 24 * 3600,
    root: Path | None = None,
    now: float | None = None,
) -> int:
    """Delete files under the ``argus_ytt_out`` cache older than ``max_age_seconds``.

    Args:
        max_age_seconds: Files whose ``mtime`` is strictly less than
            (``now`` - ``max_age_seconds``) are deleted. Default = 24h.
        root: Cache root to scan. Defaults to
            ``tempfile.gettempdir()/argus_ytt_out``. Exposed for tests
            so the suite doesn't touch the real OS tempdir.
        now: Override the wall-clock reference (Unix seconds). Defaults
            to ``time.time()``. Exposed for tests so the suite can pin
            the cutoff against files it has just backdated.

    Returns:
        Number of files actually deleted. Subdirectories are skipped
        (their presence is treated as opaque state we don't own).

    The cleanup is best-effort and exception-tolerant: missing files,
    permission errors, or a vanished target mid-loop are swallowed
    rather than propagated up to the bot's startup path.
    """
    cache_root = (
        Path(root) if root is not None
        else Path(tempfile.gettempdir()) / "argus_ytt_out"
    )
    if not cache_root.is_dir():
        # No cache yet (first boot, dir not yet created). Quiet no-op.
        return 0

    cutoff = (now if now is not None else time.time()) - max_age_seconds
    removed = 0
    try:
        entries = list(cache_root.iterdir())
    except OSError as exc:
        logger.warning("argus_ytt cache scan failed: %s", exc)
        return 0

    for entry in entries:
        # Skip directories — we only age out *files* under the cache.
        # yt-dlp shouldn't leave subdirs here, but a stray one (e.g.
        # an interrupted test) shouldn't cascade into a recursive wipe.
        if not entry.is_file():
            continue
        try:
            st = os.stat(entry)
            if st.st_mtime >= cutoff:
                continue
            os.remove(entry)
            removed += 1
        except FileNotFoundError:
            # Race: another process (or a previous bot run) already
            # removed it. Count it as removed and move on.
            removed += 1
        except OSError as exc:
            logger.debug(
                "argus_ytt cache: skipping %s: %s", entry.name, exc)
    return removed
