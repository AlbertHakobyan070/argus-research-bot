#!/usr/bin/env python
"""Reconcile the media vault with the library database.

The two can drift apart. A download that finished on disk but whose
process was killed before Argus registered it becomes invisible: it
doesn't show up in /delete, isn't counted, and gets silently
re-downloaded. Interrupted downloads also leave ``.part`` / ``.ytdl`` /
``*-Frag*.part`` litter behind, which yt-dlp uses to resume — so it is
reported here, never deleted unless you ask.

Usage
-----
    python scripts/reconcile_vault.py                 # report only
    python scripts/reconcile_vault.py --register      # adopt orphan files
    python scripts/reconcile_vault.py --prune-partials --older-than 7
    python scripts/reconcile_vault.py --remap OLD_ROOT NEW_ROOT

Also the first thing to run after moving the vault to another machine:
the report mode re-points nothing, but it tells you exactly what the new
vault holds and what the library thinks it holds. ``--remap`` then
rewrites the stored asset paths (``assets`` table only — each run's
``report_dir`` keeps the old root).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from argus.config import get_settings          # noqa: E402
from argus.library import Library              # noqa: E402

# Sidecars and scratch files — never registered as assets.
_NON_MEDIA_SUFFIXES = (".info.json", ".ytdl", ".part", ".temp", ".ytdl.part")
_PARTIAL_MARKERS = (".part", ".ytdl", ".temp")


def _is_media(p: Path) -> bool:
    name = p.name
    if name.startswith(".argus_marker_"):
        return False
    return not any(name.endswith(s) for s in _NON_MEDIA_SUFFIXES)


def _is_partial(p: Path) -> bool:
    name = p.name
    return (any(name.endswith(s) for s in _PARTIAL_MARKERS)
            or ".part-Frag" in name)


def _sidecar_info(media_path: Path) -> dict:
    """Find the .info.json yt-dlp wrote next to a media file. The merged
    output's extension can differ from the info json stem, so fall back
    to a stem glob."""
    direct = media_path.with_suffix(".info.json")
    if direct.exists():
        candidates = [direct]
    else:
        candidates = sorted(
            media_path.parent.glob(media_path.stem + "*.info.json"))
    for c in candidates:
        try:
            return json.loads(c.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
    return {}


def _looks_windows(root: str) -> bool:
    """True for 'A:\\...', 'A:/...' or a UNC '\\\\server\\share' root."""
    r = root.strip()
    return bool(r[1:2] == ":" or r.startswith("\\\\"))


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} GB"


async def _remap(old_root: str, new_root: str) -> int:
    """Rewrite stored asset paths after the vault moves between machines.

    Copying the vault from ``A:\\DS_Vault\\...`` to ``/Users/me/DS_Vault/...``
    leaves every library row pointing at a path that no longer exists, so
    /delete and /transcripts go blind. This rewrites the prefix and
    normalises the separators for the target root's OS.

    Only the ``assets`` table is rewritten. ``runs.report_dir`` keeps the
    old root, so /delete of such a run drops its registry row but cannot
    find (or remove) the moved report folder.
    """
    s = get_settings()
    lib = Library(s.library_db)
    await lib.open()
    try:
        db = lib._conn()
        async with db.execute(
                "SELECT asset_id, path FROM assets") as cur:
            rows = await cur.fetchall()

        old_norm = old_root.replace("\\", "/").rstrip("/")
        # Take the separator from the TARGET root, not from whichever OS
        # happens to run this — the remap is useful in both directions and
        # pathlib would silently rewrite "/Users/..." into "\Users\...".
        sep = "\\" if _looks_windows(new_root) else "/"
        new_base = new_root.replace("\\", "/").rstrip("/")

        changed = 0
        for asset_id, path in rows:
            unified = str(path).replace("\\", "/")
            if not unified.lower().startswith(old_norm.lower()):
                continue
            tail = unified[len(old_norm):].lstrip("/")
            new_path = f"{new_base}/{tail}".replace("/", sep)
            await db.execute("UPDATE assets SET path=? WHERE asset_id=?",
                             (new_path, asset_id))
            changed += 1
        await db.commit()
        print(f"remapped {changed} of {len(rows)} asset path(s)")
        print(f"  {old_root}  ->  {new_root}")
        if changed:
            print("Now run without --remap to confirm the files resolve.")
    finally:
        await lib.close()
    return 0


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--register", action="store_true",
                    help="add orphaned media files to the library")
    ap.add_argument("--prune-partials", action="store_true",
                    help="DELETE interrupted-download leftovers (this "
                         "destroys yt-dlp's ability to resume them)")
    ap.add_argument("--older-than", type=float, default=7.0, metavar="DAYS",
                    help="only prune leftovers older than this (default 7)")
    ap.add_argument("--remap", nargs=2, metavar=("OLD_ROOT", "NEW_ROOT"),
                    help="rewrite stored asset paths from OLD_ROOT to "
                         "NEW_ROOT (use after moving the vault to another "
                         "machine, e.g. --remap 'A:\\DS_Vault' "
                         "'/Users/me/DS_Vault')")
    args = ap.parse_args()

    if args.remap:
        return await _remap(*args.remap)

    s = get_settings()
    media_root = s.media_root
    print(f"vault media root : {media_root}")
    print(f"library database : {s.library_db}")
    if not media_root.exists():
        print("\nMedia root does not exist yet — nothing to reconcile.")
        return 0

    lib = Library(s.library_db)
    await lib.open()
    try:
        known: set[str] = set()
        rows = await lib.list_assets(kind="media", limit=100_000)
        for row in rows:
            known.add(str(Path(row["path"])))

        orphans: list[Path] = []
        partials: list[Path] = []
        missing: list[str] = []
        total_bytes = 0

        for path in sorted(media_root.rglob("*")):
            if not path.is_file():
                continue
            if _is_partial(path):
                partials.append(path)
                continue
            if not _is_media(path):
                continue
            total_bytes += path.stat().st_size
            if str(path) not in known:
                orphans.append(path)

        for known_path in sorted(known):
            if not Path(known_path).exists():
                missing.append(known_path)

        print(f"\non disk          : {len(orphans) + len(known) - len(missing)}"
              f" media files, {_human(total_bytes)}")
        print(f"in library       : {len(known)}")

        # -- orphaned complete files ---------------------------------
        if orphans:
            print(f"\n{len(orphans)} file(s) on disk but NOT in the library:")
            for p in orphans:
                print(f"   {_human(p.stat().st_size):>12}  "
                      f"{p.relative_to(media_root)}")
            if args.register:
                added = 0
                for p in orphans:
                    info = _sidecar_info(p)
                    await lib.add_asset(
                        kind="media",
                        platform=p.parent.name,
                        source_url=info.get("webpage_url") or "",
                        media_id=str(info.get("id") or ""),
                        title=info.get("title") or p.stem,
                        path=str(p),
                        size_bytes=p.stat().st_size,
                        duration_s=info.get("duration"),
                        meta={"registered_by": "reconcile_vault"})
                    added += 1
                print(f"   -> registered {added} file(s) in the library")
            else:
                print("   (re-run with --register to adopt them)")
        else:
            print("\nno orphaned media files.")

        # -- library rows whose file is gone --------------------------
        if missing:
            print(f"\n{len(missing)} library row(s) whose file is MISSING:")
            for m in missing:
                print(f"   {m}")
            print("   (delete these from the bot with /delete)")

        # -- interrupted-download litter ------------------------------
        if partials:
            now = time.time()
            print(f"\n{len(partials)} interrupted-download leftover(s):")
            prunable: list[Path] = []
            for p in partials:
                age_days = (now - p.stat().st_mtime) / 86400
                mark = ""
                if age_days >= args.older_than:
                    prunable.append(p)
                    mark = "  [prunable]"
                print(f"   {_human(p.stat().st_size):>12}  "
                      f"{age_days:5.1f}d  {p.relative_to(media_root)}{mark}")
            print("   yt-dlp resumes from these if you re-send the same link.")
            if args.prune_partials and prunable:
                freed = 0
                for p in prunable:
                    freed += p.stat().st_size
                    p.unlink()
                print(f"   -> deleted {len(prunable)} leftover(s), "
                      f"freed {_human(freed)}")
            elif prunable:
                print(f"   ({len(prunable)} older than {args.older_than}d — "
                      "re-run with --prune-partials to delete)")
        else:
            print("\nno interrupted-download leftovers.")
    finally:
        await lib.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
