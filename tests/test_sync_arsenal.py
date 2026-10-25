"""scripts/sync_arsenal.py (v4 WP-E2): the fixed file list mirrored under the
snapshot root with a sha256 MANIFEST.json, on a fixture arsenal tree; and the
committed snapshot plus the console page it serves (no inline script, every
local reference present)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "argus" / "webapp" / "static"
_spec = importlib.util.spec_from_file_location("sync_arsenal", ROOT / "scripts" / "sync_arsenal.py")
sync = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def kit(tmp_path) -> Path:
    """A minimal arsenal tree: every fixed file, a pairing with two faces, their fonts and licences."""
    root = tmp_path / "arsenal"
    for rel in sync.FIXED_FILES:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"/* {rel} */\n", encoding="utf-8")
    (root / "kits/web-ui/ui.css").write_text('@import url("components/core.css");\n', encoding="utf-8")
    (root / "tokens/fonts.css").write_text(
        '@font-face{src:url(../vendor/fonts/web/sans/sans-400.woff2) format("woff2")}\n', encoding="utf-8")
    type_dir = root / "kits/web-ui/type"
    (type_dir / "faces").mkdir(parents=True)
    (type_dir / "pair-ledger.css").write_text('@import url("faces/sans.css");\n@import url("faces/mono.css");\n'
                                              ":root{--body:Sans}\n", encoding="utf-8")
    (type_dir / "pair-other.css").write_text('@import url("faces/mono.css");\n', encoding="utf-8")
    for face, files in {"sans": ("sans-400.woff2", "sans-700.woff2"), "mono": ("mono-400.woff2",)}.items():
        css = "".join(f'@font-face{{src:url(../../../../vendor/fonts/web/{face}/{f}) format("woff2")}}\n'
                      for f in files)
        (type_dir / "faces" / f"{face}.css").write_text(css, encoding="utf-8")
        fonts = root / "vendor/fonts/web" / face
        fonts.mkdir(parents=True)
        (fonts / "LICENSE.txt").write_text(f"OFL {face}\n", encoding="utf-8")
        for f in files:
            (fonts / f).write_bytes(f.encode() * 3)
    (root / "vendor/fonts/web/unused").mkdir(parents=True)
    (root / "vendor/fonts/web/unused/unused.woff2").write_bytes(b"x")
    return root


def test_mirrors_the_fixed_list_with_a_sha256_manifest(kit, tmp_path):
    out = tmp_path / "snap"
    assert sync.main(["--arsenal", str(kit), "--out", str(out)]) == 0
    manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"source_root", "copied_at", "pair", "files"}
    assert manifest["pair"] == "ledger" and manifest["source_root"] == kit.as_posix()
    paths = [f["path"] for f in manifest["files"]]
    assert paths == sorted(paths)
    expected = {*sync.FIXED_FILES, "kits/web-ui/type/pair-ledger.css", "kits/web-ui/type/faces/sans.css",
                "kits/web-ui/type/faces/mono.css", "vendor/fonts/web/sans/sans-400.woff2",
                "vendor/fonts/web/sans/sans-700.woff2", "vendor/fonts/web/mono/mono-400.woff2",
                "vendor/fonts/web/sans/LICENSE.txt", "vendor/fonts/web/mono/LICENSE.txt"}
    assert set(paths) == expected
    for f in manifest["files"]:
        copied = out / f["path"]
        assert copied.read_bytes() == (kit / f["path"]).read_bytes()      # same relative tree
        assert f["sha256"] == _sha(copied)
    on_disk = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert on_disk == expected | {"MANIFEST.json", ".gitattributes"}      # nothing unlisted (no unused font)
    assert (out / ".gitattributes").read_text(encoding="utf-8") == "* -text\n"      # sums hold on every checkout


def test_another_pairing_copies_only_its_faces(kit, tmp_path):
    out = tmp_path / "snap"
    sync.main(["--arsenal", str(kit), "--out", str(out), "--pair", "other"])
    paths = {f["path"] for f in json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))["files"]}
    assert "kits/web-ui/type/pair-other.css" in paths and "kits/web-ui/type/faces/mono.css" in paths
    assert "kits/web-ui/type/faces/sans.css" not in paths and "kits/web-ui/type/pair-ledger.css" not in paths
    assert "vendor/fonts/web/sans/sans-400.woff2" in paths                # tokens/fonts.css still references it


def test_a_resync_replaces_the_old_snapshot(kit, tmp_path):
    out = tmp_path / "snap"
    out.mkdir()
    (out / "stale.css").write_text("old", encoding="utf-8")
    sync.main(["--arsenal", str(kit), "--out", str(out)])
    assert not (out / "stale.css").exists()


def test_the_default_source_is_the_manifests(kit, tmp_path):
    out = tmp_path / "snap"
    sync.main(["--arsenal", str(kit), "--out", str(out)])
    (kit / "tokens/base.css").write_text("/* changed */\n", encoding="utf-8")
    assert sync.main(["--out", str(out)]) == 0
    assert (out / "tokens/base.css").read_text(encoding="utf-8") == "/* changed */\n"


def test_no_source_and_no_manifest_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="--arsenal"):
        sync.main(["--out", str(tmp_path / "snap")])


def test_a_missing_listed_file_raises(kit, tmp_path):
    (kit / "kits/web-ui/ui.classic.js").unlink()
    with pytest.raises(FileNotFoundError, match="ui.classic.js"):
        sync.main(["--arsenal", str(kit), "--out", str(tmp_path / "snap")])


def test_an_unknown_pairing_raises(kit, tmp_path):
    with pytest.raises(FileNotFoundError, match="pair-nope.css"):
        sync.main(["--arsenal", str(kit), "--out", str(tmp_path / "snap"), "--pair", "nope"])


def test_a_reference_outside_the_snapshot_raises(kit, tmp_path):
    (kit / "kits/web-ui/ui.css").write_text('@import url("components/missing.css");\n', encoding="utf-8")
    with pytest.raises(ValueError, match="components/missing.css"):
        sync.main(["--arsenal", str(kit), "--out", str(tmp_path / "snap")])


# --- the committed snapshot and the console page ------------------------------

def test_the_committed_snapshot_matches_its_manifest():
    snap = STATIC / "vendor" / "arsenal"
    manifest = json.loads((snap / "MANIFEST.json").read_text(encoding="utf-8"))
    listed = {f["path"]: f["sha256"] for f in manifest["files"]}
    on_disk = {p.relative_to(snap).as_posix() for p in snap.rglob("*") if p.is_file()} - {"MANIFEST.json", ".gitattributes"}
    assert on_disk == set(listed)
    assert all(_sha(snap / p) == h for p, h in listed.items())


def test_the_console_has_no_inline_script_and_its_references_exist():
    for page in STATIC.rglob("*.html"):
        if "vendor" in page.relative_to(STATIC).parts:
            continue                                     # the kit's own templates are reference copies
        html = page.read_text(encoding="utf-8")
        assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), page    # every script is a file
        assert not re.search(r"\son[a-z]+\s*=", html, re.I), page              # no inline handlers
        assert not re.search(r"\sstyle\s*=", html, re.I), page                 # style-src 'self'
        for ref in re.findall(r'(?:src|href)="(/static/[^"#?]+)"', html):
            assert (STATIC / ref.removeprefix("/static/")).is_file(), ref
    for js in STATIC.rglob("*.js"):
        if "vendor" in js.relative_to(STATIC).parts:
            continue
        for ref in re.findall(r'from "(\.{1,2}/[^"]+)"', js.read_text(encoding="utf-8")):
            assert (js.parent / ref).resolve().is_file(), (js, ref)
