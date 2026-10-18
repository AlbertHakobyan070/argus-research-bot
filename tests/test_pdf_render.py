"""Regression tests for the ReportLab PDF renderer (2026-07-16 redesign).

Covers three concrete bugs found in a live-generated report
("Bitcoin market trends", Long mode) and fixed alongside the "old
money" editorial redesign:

1. Double title — the folder-name slug ("Bitcoin_market_trends") was
   rendered as its own H1 above the markdown body's real title
   ("Bitcoin market trends"), which is written by
   ``render_title_block``. Fixed by no longer drawing a title
   separately from the markdown body's own ``# ...`` line.
2. Fragmented quality-summary card — ``render_title_block`` emits a
   bare ``>`` (no trailing space) as a blank blockquote-continuation
   line, which the old ``line.lstrip().startswith("> ")`` check missed,
   splitting the "Quality summary" callout into two disconnected boxes
   with a stray literal ">" rendered as plain text between them.
3. Invisible emoji glyphs — headings containing emoji (e.g.
   ``## ⚠️ Unverified / flagged claims``) rendered as missing-glyph
   boxes in every available PDF text font.

No PDF-parsing library is a project dependency, so these tests assert
on renderer behaviour that's directly observable without one: the
function completes without raising, produces a non-trivial file, and
the pure helper functions (``_strip_emoji``, ``_register_report_fonts``)
behave correctly in isolation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from argus.tools import _register_report_fonts, _strip_emoji, markdown_to_pdf


# ---------------------------------------------------------------------------
# _strip_emoji — the PDF-only emoji removal pass
# ---------------------------------------------------------------------------

def test_strip_emoji_removes_warning_sign_and_variation_selector():
    out = _strip_emoji("⚠️ Unverified / flagged claims")
    assert out == "Unverified / flagged claims"


def test_strip_emoji_removes_leading_emoji_from_heading():
    out = _strip_emoji("## ⚠️ Unverified / flagged claims")
    assert out == "## Unverified / flagged claims"


def test_strip_emoji_leaves_plain_text_untouched():
    text = "Bitcoin's price has shown volatility [10][16]."
    assert _strip_emoji(text) == text


def test_strip_emoji_leaves_markdown_structure_untouched():
    """Emoji stripping must not eat structural markers (#, -, >, |)."""
    assert _strip_emoji("- a bullet point") == "- a bullet point"
    assert _strip_emoji("> a blockquote") == "> a blockquote"
    assert _strip_emoji("| a | table |") == "| a | table |"


# ---------------------------------------------------------------------------
# _register_report_fonts — bundled fonts must actually load
# ---------------------------------------------------------------------------

def test_report_fonts_register_to_bundled_files_not_base14():
    """A missing/broken font file must not silently fall back without
    trace — this asserts the REAL bundled fonts are what's active in
    this environment (base14 fallback is a separate degrade path, not
    the expected state of a checked-out repo)."""
    fonts = _register_report_fonts()
    assert fonts["display"] == "Cardo-Regular"
    assert fonts["body"] == "CrimsonText-Regular"
    assert fonts["caps"] == "CormorantSC-SemiBold"


def test_bundled_font_files_exist_on_disk():
    from argus.tools import FONTS_DIR
    for name in ("Cardo-Regular", "Cardo-Bold", "Cardo-Italic",
                 "CrimsonText-Regular", "CrimsonText-Italic",
                 "CrimsonText-SemiBold", "CrimsonText-Bold",
                 "CrimsonText-BoldItalic", "CormorantSC-SemiBold"):
        assert (FONTS_DIR / f"{name}.ttf").exists(), name


# ---------------------------------------------------------------------------
# markdown_to_pdf — end-to-end render smoke tests
# ---------------------------------------------------------------------------

_SAMPLE_MD = """# Sample Research Report

_Mode: **Long** • 2026-07-16T00:00:00+04:00 • findings: 2 • sources: 2 • revisions: 1_

> **Quality summary**
>
> Overall relevancy: [HIGH] high
> Reviewer flagged 1 unsupported claim(s).

---

## TL;DR

A short summary with a citation [1].

> WARNING: **Limited direct evidence:** only
> 1 of 3 credible sources
> mention the queried entity.

## A Section Heading

Body prose for the section [1].

```
raw code line one
raw code line two
```

## Sources

1. [Example Source](https://example.com/a)

---

## ⚠️ Unverified / flagged claims

_Flagged findings preserved for audit._

- (id=f0) An unsupported claim [1]
"""


def test_markdown_to_pdf_renders_without_raising(tmp_path):
    out = tmp_path / "report.pdf"
    markdown_to_pdf(_SAMPLE_MD, str(out), title="Sample Research Report")
    assert out.exists()
    # A near-empty file would indicate the renderer silently produced a
    # blank/broken document rather than the multi-page report above.
    assert out.stat().st_size > 20_000


def test_markdown_to_pdf_handles_title_page_and_body_h1_without_raising(tmp_path):
    """Regression for the double-title bug: passing a topic string that
    differs from the body's own leading '# ...' line (as the real
    report_builder now does — folder slug vs. raw user_request) must
    not raise, and must not require any special-casing in the caller."""
    out = tmp_path / "report.pdf"
    md = "# Bitcoin market trends\n\nBody text with a citation [1].\n"
    markdown_to_pdf(md, str(out), title="Bitcoin market trends")
    assert out.exists()
    assert out.stat().st_size > 5_000


def test_markdown_to_pdf_survives_missing_font_dir(tmp_path, monkeypatch):
    """The bundled-font lookup must degrade to base14, never crash a
    report delivery, if the assets directory is unavailable.

    monkeypatch reverts both attributes (including the font-role cache)
    to their pre-test values on teardown, so this cannot leak a base14
    fallback into later tests.
    """
    import argus.tools as tools_mod
    monkeypatch.setattr(tools_mod, "_REPORT_FONT_ROLES", None)
    monkeypatch.setattr(tools_mod, "FONTS_DIR", tmp_path / "does-not-exist")
    fonts = tools_mod._register_report_fonts()
    assert fonts["display"] == "Times-Bold"
    assert fonts["body"] == "Times-Roman"


# ---------------------------------------------------------------------------
# No hidden PDF fallback (v4 WP-0b): ReportLab is the only renderer.
# ---------------------------------------------------------------------------

def test_markdown_to_pdf_raises_without_chromium_fallback(tmp_path, monkeypatch):
    """When ReportLab fails, ``markdown_to_pdf`` re-raises. It must not try
    the intel-stack Chromium renderer.

    A canary ``_common.py`` stands in for the intel-stack helper, so a
    regression (the fallback coming back) fails here by name instead of
    launching a real browser.
    """
    import argus.tools as tools_mod

    (tmp_path / "_common.py").write_text(
        "def markdown_to_pdf(*a, **kw):\n"
        "    raise AssertionError('Chromium fallback was invoked')\n",
        encoding="utf-8")
    monkeypatch.setattr(tools_mod, "INTEL_STACK_DIR", tmp_path)

    def boom(*a, **kw):
        raise RuntimeError("reportlab exploded")

    monkeypatch.setattr(tools_mod, "_render_pdf_reportlab", boom)

    with pytest.raises(RuntimeError, match="reportlab exploded"):
        markdown_to_pdf("# t\n\nbody\n", str(tmp_path / "r.pdf"), title="t")


def test_report_builder_records_pdf_failure(tmp_path, monkeypatch):
    """A PDF failure is never hidden: report_builder_node reports the PDF as
    missing and puts the exception in the run's ``errors``. The markdown
    report is still produced."""
    import types

    from argus import config as config_mod
    from argus.graph import nodes as nodes_mod

    monkeypatch.setattr(config_mod, "get_settings",
                        lambda: types.SimpleNamespace(reports_root=tmp_path))

    def boom(*a, **kw):
        raise RuntimeError("reportlab exploded")

    monkeypatch.setattr(nodes_mod, "markdown_to_pdf", boom)

    out = nodes_mod.report_builder_node({
        "user_request": "pdf failure probe", "length": "short",
        "draft_md": "# probe\n\nBody text.\n",
    })

    assert out["report_paths"]["pdf"] is None
    assert out["hitl"]["ctx"]["paths"]["pdf"] is None
    errors = out.get("errors") or []
    assert len(errors) == 1 and errors[0].startswith("pdf: "), errors
    assert "reportlab exploded" in errors[0]
    assert Path(out["report_paths"]["md"]).exists()


def test_report_builder_returns_pdf_path_on_success(tmp_path, monkeypatch):
    """Success path of the same node: the real ReportLab render yields a PDF
    path on disk and adds nothing to ``errors``."""
    import types

    from argus import config as config_mod
    from argus.graph import nodes as nodes_mod

    monkeypatch.setattr(config_mod, "get_settings",
                        lambda: types.SimpleNamespace(reports_root=tmp_path))

    out = nodes_mod.report_builder_node({
        "user_request": "pdf success probe", "length": "short",
        "draft_md": "# probe\n\nBody text.\n",
    })

    pdf = Path(out["report_paths"]["pdf"])
    assert pdf.exists() and pdf.stat().st_size > 5_000
    assert out["hitl"]["ctx"]["paths"]["pdf"] == str(pdf)
    assert "errors" not in out
