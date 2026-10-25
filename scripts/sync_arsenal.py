"""Copy the arsenal web-ui kit files the console uses into the repo (v4 spec §3 WP-E2).

    ./venv/Scripts/python.exe scripts/sync_arsenal.py [--arsenal DIR] [--pair ledger] [--out DIR]

Copies ``FIXED_FILES`` plus the chosen pairing (``kits/web-ui/type/pair-<pair>.css``,
the faces it ``@import``s, and the font files and licences those faces and
``tokens/fonts.css`` reference) from the arsenal root to ``--out`` (default
``src/argus/webapp/static/vendor/arsenal``), mirroring the relative tree so the
kit's relative references keep working. The old snapshot is removed first.
Every relative ``@import`` / ``url()`` in a copied stylesheet must resolve to a
copied file, or nothing is written. Writes ``MANIFEST.json``
``{source_root, copied_at, pair, files: [{path, sha256}]}`` and a
``.gitattributes`` that keeps git from converting line endings. Without
``--arsenal`` the source is the existing manifest's ``source_root``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "src" / "argus" / "webapp" / "static" / "vendor" / "arsenal"
_COMPONENTS = ("core", "shell", "content", "overlay", "study", "layouts", "themes.gen")
FIXED_FILES: tuple[str, ...] = (
    "tokens/tokens.css", "tokens/base.css", "tokens/fonts.css",
    "kits/web-ui/ui.css", *(f"kits/web-ui/components/{c}.css" for c in _COMPONENTS),
    "kits/web-ui/ui.classic.js", "kits/web-ui/icons/icons.js", "kits/web-ui/motion/motion.css",
    "kits/web-ui/templates/app-shell.html", "kits/web-ui/templates/console.html",
)
# The snapshot is committed byte for byte (no line-ending conversion), so the
# sha256 sums hold on every checkout.
GITATTRIBUTES = "* -text\n"
_REF = re.compile(r"""@import\s+url\(\s*["']?([^"')]+)["']?\s*\)|url\(\s*["']?([^"')]+)["']?\s*\)""")


def _refs(css: str) -> list[str]:
    """The relative references of a stylesheet (``@import url()`` and ``url()``)."""
    out = []
    for m in _REF.finditer(css):
        ref = (m.group(1) or m.group(2)).strip()
        if not re.match(r"^(data:|[a-z][a-z0-9+.-]*://|/|#)", ref, re.I):
            out.append(ref)
    return out


def _resolve(rel: str, ref: str) -> str:
    """``ref`` (relative to the file ``rel``) as a path relative to the arsenal root."""
    path = posixpath.normpath(posixpath.join(posixpath.dirname(rel), ref))
    if path.startswith(".."):
        raise ValueError(f"{rel}: {ref} points outside the arsenal root")
    return path


def file_list(arsenal: Path, pair: str) -> list[str]:
    """The fixed files, the pairing, its faces, and the fonts (with each font
    folder's LICENSE.txt) the faces and ``tokens/fonts.css`` reference."""
    pair_css = f"kits/web-ui/type/pair-{pair}.css"
    files = [*FIXED_FILES, pair_css]
    for rel in files:
        if not (arsenal / rel).is_file():
            raise FileNotFoundError(f"{arsenal / rel} (listed in scripts/sync_arsenal.py) does not exist")
    faces = [_resolve(pair_css, r) for r in _refs((arsenal / pair_css).read_text(encoding="utf-8"))]
    fonts: list[str] = []
    for css in (*faces, "tokens/fonts.css"):
        if not (arsenal / css).is_file():
            raise FileNotFoundError(f"{arsenal / css} (imported by {pair_css}) does not exist")
        fonts += [_resolve(css, r) for r in _refs((arsenal / css).read_text(encoding="utf-8"))]
    licences = [posixpath.join(posixpath.dirname(f), "LICENSE.txt") for f in fonts]
    every = sorted(set(files) | set(faces) | set(fonts) | set(licences))
    missing = [f for f in every if not (arsenal / f).is_file()]
    if missing:
        raise FileNotFoundError(f"referenced by the kit but missing in {arsenal}: {', '.join(missing)}")
    return every


def _check_closure(arsenal: Path, files: list[str]) -> None:
    """Every relative reference of a copied stylesheet is itself copied."""
    have = set(files)
    for rel in files:
        if rel.endswith(".css"):
            for ref in _refs((arsenal / rel).read_text(encoding="utf-8")):
                if _resolve(rel, ref) not in have:
                    raise ValueError(f"{rel} references {ref} ({_resolve(rel, ref)}), which is not in the snapshot "
                                     "— add it to FIXED_FILES in scripts/sync_arsenal.py")


def sync(arsenal: Path, pair: str, out: Path) -> dict:
    files = file_list(arsenal, pair)
    _check_closure(arsenal, files)
    if out.exists():
        shutil.rmtree(out)
    entries = []
    for rel in files:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(arsenal / rel, dest)
        entries.append({"path": rel, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()})
    manifest = {"source_root": arsenal.as_posix(), "copied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "pair": pair, "files": entries}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    (out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8", newline="\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arsenal", type=Path, help="the arsenal root (default: the manifest's source_root)")
    parser.add_argument("--pair", default="ledger", help="the type pairing (kits/web-ui/type/pair-<name>.css)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the snapshot root")
    args = parser.parse_args(argv)
    arsenal = args.arsenal
    if arsenal is None:
        manifest = args.out / "MANIFEST.json"
        if not manifest.is_file():
            raise SystemExit(f"no {manifest} to take the source from: pass --arsenal DIR")
        arsenal = Path(json.loads(manifest.read_text(encoding="utf-8"))["source_root"])
    result = sync(arsenal.resolve(), args.pair, args.out)
    print(f"copied {len(result['files'])} files from {result['source_root']} (pair {args.pair}) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
