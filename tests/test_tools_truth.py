"""Truthiness tests for argus.tools fetch wrappers.

These tests target the silent-empty-report bug observed in
t_7f2b625c: the fetch wrappers (snatch_url, crawl_url,
normalize_to_markdown) used to return ok=True whenever the intel-stack
subprocess exited with rc==0, even when the path printed on stdout did
not actually exist on disk. The T5 fix computes ``ok`` from real file
existence for each wrapper, and these tests prove:

1. **Happy path** — real rc=0 + real existing path -> ok=True + populated fields.
2. **Silent-empty lie (the bug)** — rc=0 + NON-existent or non-parseable
   "path" in stdout -> ok=False (NOT ok=True with markdown_path=None,
   which is what the old code did).
3. **Parser correctness per tool** — each stdout format gets the
   right extraction:
     - snatch.py: a single JSON line with ``{"folder": "..."}``.
     - crawl.py: a single JSON line with ``{"folder": "..."}``.
     - article_convert.py: a single ``md: A:\\path\\file.md`` line.
4. **Folder path vs file path** — snatch/crawl want a *directory*
   containing .md files; normalize wants the .md file directly.
5. **Subprocess failure** — rc!=0 -> ok=False + error populated.
6. **Multi-line stdout** — when a tool prints status banners before
   the JSON, parsing still finds the JSON.

Each test patches ``subprocess.run`` (called by ``argus.tools._run_script``)
so it never actually shells out — fully hermetic, no network, no
intel-stack dependencies, runs in <100 ms.

Marker: ``@pytest.mark.unit`` (default test set). These do NOT need
``ARGUS_E2E=1`` because they never touch the real subprocesses.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


# Make sure the test does not pick up a leaked PYTHONPATH from CI.
@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)


class _FakeCompleted:
    """Stand-in for ``subprocess.CompletedProcess`` with .returncode/.stdout/.stderr."""

    def __init__(self, rc: int, out: str = "", err: str = "") -> None:
        self.returncode = rc
        self.stdout = out
        self.stderr = err


@pytest.fixture
def fake_subprocess(monkeypatch: pytest.MonkeyPatch):
    """Patch ``subprocess.run`` with a callable the test can drive.

    Yields an object the test sets ``.rc`` / ``.stdout`` / ``.stderr``
    on. The wrapper invokes the real ``_run_script`` -> ``subprocess.run``
    path, so the parse step is exercised exactly as in production.
    """
    state: dict[str, Any] = {"rc": 0, "out": "", "err": ""}
    calls: list[list[str]] = []

    def _fake(cmd, **kwargs):  # mirrors subprocess.run signature loosely
        calls.append(list(cmd))
        return _FakeCompleted(state["rc"], state["out"], state["err"])

    monkeypatch.setattr(subprocess, "run", _fake)
    return type("F", (), {"state": state, "calls": calls})()


# ----------------------------------------------------------------------
# snatch_url — json stdout
# ----------------------------------------------------------------------

def test_snatch_happy_path(tmp_path: Path, fake_subprocess) -> None:
    from argus.tools import snatch_url

    folder = tmp_path / "snatched_dir"
    folder.mkdir()
    md = folder / "page.md"
    md.write_text("# Page Title\n\nbody\n", encoding="utf-8")
    fake_subprocess.state["rc"] = 0
    fake_subprocess.state["out"] = json.dumps(
        {"ok": True, "kind": "articles", "folder": str(folder)}
    )

    r = snatch_url("https://example.com/x", kind="auto")
    assert r.ok is True, f"ok should be True; error={r.error!r}"
    assert r.markdown_path == str(md)
    assert r.title == "Page Title"
    assert r.folder == str(folder)
    assert r.error is None


def test_snatch_rc0_but_json_garbage_returns_not_ok(fake_subprocess) -> None:
    """Reproduces the silent-empty-report bug from t_7f2b625c.

    Subprocess exits 0, stdout is *almost* JSON but malformed so the
    folder key is missing. The old wrapper returned ok=True with
    folder=literal-garbage. The new wrapper must return ok=False.
    """
    from argus.tools import snatch_url

    fake_subprocess.state["rc"] = 0
    # Looks JSON-ish but missing the folder field entirely.
    fake_subprocess.state["out"] = json.dumps({"ok": True, "kind": "papers"})

    r = snatch_url("https://example.com/y")
    assert r.ok is False
    assert r.markdown_path is None


def test_snatch_rc0_but_folder_path_does_not_exist(fake_subprocess) -> None:
    """Another face of the bug.

    JSON parses cleanly, claims ``folder``: a path, but the path does
    not exist on disk (e.g. user cleaned up tmp between runs, network
    mount is gone, etc.). The new wrapper must report ok=False because
    there is no real markdown to read.
    """
    from argus.tools import snatch_url

    ghost = "/a/ghost/dir/that/never/existed"
    fake_subprocess.state["rc"] = 0
    fake_subprocess.state["out"] = json.dumps(
        {"ok": True, "folder": ghost, "kind": "articles"}
    )

    r = snatch_url("https://example.com/z")
    assert r.ok is False
    assert r.markdown_path is None


def test_snatch_subprocess_failure(fake_subprocess) -> None:
    """Non-zero rc -> ok=False with stderr captured."""
    from argus.tools import snatch_url

    fake_subprocess.state["rc"] = 1
    fake_subprocess.state["err"] = "boom: connection refused"

    r = snatch_url("https://example.com/q")
    assert r.ok is False
    assert "connection refused" in (r.error or "")


# ----------------------------------------------------------------------
# crawl_url — json stdout (same shape as snatch)
# ----------------------------------------------------------------------

def test_crawl_happy_path(tmp_path: Path, fake_subprocess) -> None:
    from argus.tools import crawl_url

    folder = tmp_path / "crawled"
    folder.mkdir()
    (folder / "a.md").write_text("# A\n", encoding="utf-8")
    (folder / "b.md").write_text("# B\n", encoding="utf-8")
    fake_subprocess.state["rc"] = 0
    fake_subprocess.state["out"] = json.dumps(
        {"ok": True, "folder": str(folder), "pages": 2, "failed": 0}
    )

    r = crawl_url("https://docs.example.com/", max_pages=3)
    assert r.ok is True
    assert r.markdown_path and Path(r.markdown_path).exists()
    # up to max_pages returned in `pages`
    assert 1 <= len(r.pages) <= 3


def test_crawl_rc0_but_folder_ghost(fake_subprocess) -> None:
    """Same T5 fix applied to crawl_url."""
    from argus.tools import crawl_url

    fake_subprocess.state["rc"] = 0
    fake_subprocess.state["out"] = json.dumps(
        {"ok": True, "folder": "/nonexistent/ghost/", "pages": 0}
    )

    r = crawl_url("https://docs.example.com/")
    assert r.ok is False
    assert r.markdown_path is None


def test_crawl_multiline_stdout_with_banner(tmp_path: Path,
                                              fake_subprocess) -> None:
    """crawl.py may print status banners BEFORE the final JSON line.

    Old parser grabbed the *last* line containing ``\\`` (so it grabbed
    a banner like ``[INFO] scraping 3/10 pages``) and treated it as the
    folder. New parser scans all lines and picks the one that parses as
    JSON.
    """
    from argus.tools import crawl_url

    folder = tmp_path / "crawled2"
    folder.mkdir()
    (folder / "x.md").write_text("# X\n", encoding="utf-8")

    banner_lines = [
        "[INFO] crawl.py starting",
        "[INFO] scraping 3 pages",
        json.dumps({"ok": True, "folder": str(folder), "pages": 1}),
    ]
    fake_subprocess.state["rc"] = 0
    fake_subprocess.state["out"] = "\n".join(banner_lines)

    r = crawl_url("https://docs.example.com/")
    assert r.ok is True
    assert r.markdown_path and Path(r.markdown_path).exists()


# ----------------------------------------------------------------------
# normalize_to_markdown — prefixed-path stdout
# ----------------------------------------------------------------------

def test_normalize_happy_path(tmp_path: Path, fake_subprocess) -> None:
    from argus.tools import normalize_to_markdown

    md = tmp_path / "article.md"
    md.write_text("# Hello\n\nWorld\n", encoding="utf-8")
    fake_subprocess.state["rc"] = 0
    fake_subprocess.state["out"] = f"md: {md}"

    r = normalize_to_markdown("https://example.com/article")
    assert r.ok is True
    assert r.markdown_path == str(md)
    assert r.title == "Hello"
    assert "World" in r.markdown_text


def test_normalize_rc0_but_md_prefix_with_ghost_path(fake_subprocess) -> None:
    """Reproduces the original silent-empty-report bug.

    Subprocess exits 0, prints ``md: A:\\nonexistent\\path.md``,
    but the file was never written. Old code computed
    ``folder = literal_line``, ran ``Path(folder).rglob('*.md')``
    which returned nothing, and returned ok=True with markdown_path=None.
    New code must return ok=False because the file does not exist.
    """
    from argus.tools import normalize_to_markdown

    fake_subprocess.state["rc"] = 0
    fake_subprocess.state["out"] = "md: A:\\Hermes\\Downloads\\articles\\ghost_xyz\\file.md"

    r = normalize_to_markdown("https://example.com/ghost")
    assert r.ok is False, (
        f"ok must be False when the .md file is missing; got {r.ok!r} "
        f"with markdown_path={r.markdown_path!r}"
    )
    assert r.markdown_path is None


def test_normalize_falls_back_to_pdf_prefix(tmp_path: Path,
                                              fake_subprocess) -> None:
    """If ``--md-only`` is False the script might emit a pdf: prefix.

    The new parser prefers md: but will fall back to pdf: if no md: is
    present. This documents that behavior — current default is
    md_only=True so md: is the canonical case, but the helper handles
    the other case explicitly rather than silently swallowing it.
    """
    from argus.tools import normalize_to_markdown

    pdf = tmp_path / "article.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake pdf\n")
    fake_subprocess.state["rc"] = 0
    fake_subprocess.state["out"] = f"pdf: {pdf}"

    r = normalize_to_markdown("https://example.com/p", md_only=False)
    # pdf fallback: we got the pdf path back (not markdown_path == None).
    assert r.markdown_path == str(pdf)


def test_normalize_subprocess_failure(fake_subprocess) -> None:
    from argus.tools import normalize_to_markdown

    fake_subprocess.state["rc"] = 2
    fake_subprocess.state["err"] = "article_convert: invalid URL"

    r = normalize_to_markdown("not a url")
    assert r.ok is False
    assert "invalid URL" in (r.error or "")


# ----------------------------------------------------------------------
# helper-level unit checks (no subprocess patching needed)
# ----------------------------------------------------------------------

def test_parse_json_field_finds_last_json_line() -> None:
    from argus.tools import _parse_json_field

    s = 'noise\n{ "first": 1, "folder": "A:\\\\one" }\nmore noise\n{ "folder": "A:\\\\two" }\n'
    assert _parse_json_field(s, "folder") == "A:\\two"


def test_parse_json_field_returns_empty_on_no_json() -> None:
    from argus.tools import _parse_json_field

    assert _parse_json_field("nothing json here", "folder") == ""


def test_parse_json_field_returns_empty_on_missing_key() -> None:
    from argus.tools import _parse_json_field

    assert _parse_json_field('{"ok": true}', "folder") == ""


def test_parse_article_convert_path_strips_md_prefix(tmp_path: Path) -> None:
    from argus.tools import _parse_article_convert_path

    md = tmp_path / "x.md"
    md.write_text("# x\n", encoding="utf-8")
    out = f"md: {md}"
    assert _parse_article_convert_path(out) == str(md)


def test_parse_article_convert_path_returns_none_when_missing() -> None:
    from argus.tools import _parse_article_convert_path

    out = "md: A:\\Hermes\\Downloads\\articles\\never_existed_xyz\\ghost.md"
    # Path doesn't exist; parser returns None.
    assert _parse_article_convert_path(out) is None


# ----------------------------------------------------------------------
# T6: fetcher_node + researcher_node fallback paths
# ----------------------------------------------------------------------

def test_fetch_skips_source_with_empty_url(fake_subprocess, monkeypatch):
    """T6 fix, v3 shape: a source without a URL must be skipped, not
    crash — each skipped source records an error string."""
    from argus.graph.research import _parallel_fetch

    sources = [
        {"kind": "paper", "title": "X", "url": ""},
        {"kind": "paper", "title": "Y"},  # missing 'url'
        {"kind": "search_result", "title": "Z", "url": "site:arxiv.org foo"},
    ]
    fetched, errors = _parallel_fetch(sources)
    assert fetched == [], (
        f"empty/non-http URLs must not be fetched; got {fetched!r}"
    )
    assert len(errors) >= 2, (
        f"each skipped source must record an error; got {errors!r}"
    )


def test_arxiv_hits_returns_list_for_valid_query(monkeypatch):
    """arXiv provider parses Atom entries (hermetic httpx stub)."""
    from argus.graph import search_providers as sp

    class _R:
        text = (
            '<entry><title>Real paper</title>'
            '<id>http://arxiv.org/abs/2503.16581v1</id>'
            '<summary>An actual abstract.</summary></entry>'
        )
        def raise_for_status(self): pass

    class _C:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, *a, **kw): return _R()

    monkeypatch.setattr(sp.httpx, "Client", lambda **kw: _C())

    items, err = sp.arxiv_hits("test query")
    assert err == ""
    assert isinstance(items, list)
    assert len(items) == 1
    assert items[0]["url"] == "http://arxiv.org/abs/2503.16581v1"
    assert items[0]["title"] == "Real paper"


def test_arxiv_hits_handles_empty_query():
    """Empty / None query must NOT raise."""
    from argus.graph import search_providers as sp
    assert sp.arxiv_hits("") == ([], "")
    assert sp.arxiv_hits(None) == ([], "")


def _make_empty_harvest():
    from argus.tools import HarvestReport
    return HarvestReport(folder="", items=[], raw_stdout="", duration_s=0.0)


# ----------------------------------------------------------------------
# Phase 2: scrapling stealth fetch fallback
# ----------------------------------------------------------------------

def test_snatch_url_passes_stealth_and_transcript_flags(monkeypatch):
    """snatch_url(stealth=True, transcript=True) must forward --stealth and
    --transcript to snatch.py; the defaults must not include them."""
    from argus import tools
    captured: dict = {}

    def fake_run_script(script, args, **kw):
        captured["script"] = script
        captured["args"] = list(args)
        return (0, '{"ok": true, "folder": "A:\\\\nope"}', "")

    monkeypatch.setattr(tools, "_run_script", fake_run_script)

    tools.snatch_url("https://x.example", kind="auto",
                     stealth=True, transcript=True)
    assert captured["script"] == "snatch.py"
    assert "--stealth" in captured["args"]
    assert "--transcript" in captured["args"]

    tools.snatch_url("https://x.example", kind="auto")
    assert "--stealth" not in captured["args"]
    assert "--transcript" not in captured["args"]


def test_fetch_retries_with_stealth_when_normal_fetch_fails(monkeypatch):
    """When snatch (plain) + crawl both fail, the research fetch path
    retries once with Scrapling stealth. A success there must land in
    fetched (bot-walled site recovery)."""
    from argus.graph import research as research_mod
    from argus.tools import SnatchResult, CrawlResult

    stealth_flags: list[bool] = []

    def fake_snatch(url, *a, **kw):
        stealth_flags.append(bool(kw.get("stealth")))
        if kw.get("stealth"):
            return SnatchResult(ok=True, folder="A:\\f",
                                markdown_path="A:\\f\\ok.md",
                                title="Stealth OK", url=url)
        return SnatchResult(ok=False, url=url, error="403 bot wall")

    def fake_crawl(url, *a, **kw):
        return CrawlResult(ok=False, error="403", duration_s=0.0)

    monkeypatch.setattr(research_mod, "snatch_url", fake_snatch)
    monkeypatch.setattr(research_mod, "crawl_url", fake_crawl)

    fetched, errors = research_mod._parallel_fetch(
        [{"url": "https://walled.example", "kind": "blog",
          "title": "W", "summary": ""}])
    assert True in stealth_flags, (
        f"stealth retry must fire after snatch+crawl fail; calls={stealth_flags!r}"
    )
    assert len(fetched) == 1
    assert fetched[0]["url"] == "https://walled.example"
    assert not errors, errors
