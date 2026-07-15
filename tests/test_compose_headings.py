"""Regression tests for the section-heading duplication bug (2026-07-16).

Live evidence (Bitcoin market trends report, Long mode): every single
section in the delivered report opened with its own heading repeated
verbatim as the first line of the writer's output — e.g. the assembled
markdown read ``## Historical Trends and Volatility`` (added by
compose_node) immediately followed by the model's own
``Historical Trends and Volatility`` line. The SECTION_SYSTEM prompt
already told the model not to do this ("do NOT repeat the section
heading") — proxy-routed weak models disregarded it on every section,
so the fix is structural (strip in code), not another prompt tweak.
"""
from __future__ import annotations

from argus.graph.compose import _normalize_heading_text, _strip_duplicate_heading


def test_strips_markdown_heading_duplicate():
    md = "## Historical Trends and Volatility\n\nBitcoin's price..."
    out = _strip_duplicate_heading(md, "Historical Trends and Volatility")
    assert out == "Bitcoin's price..."


def test_strips_bold_duplicate():
    md = "**Historical Trends and Volatility**\n\nBitcoin's price..."
    out = _strip_duplicate_heading(md, "Historical Trends and Volatility")
    assert out == "Bitcoin's price..."


def test_strips_plain_text_duplicate():
    md = "Historical Trends and Volatility\nBitcoin's price..."
    out = _strip_duplicate_heading(md, "Historical Trends and Volatility")
    assert out == "Bitcoin's price..."


def test_strips_h3_level_case_insensitive_with_trailing_punctuation():
    md = "### historical trends and volatility.\n\nBody text."
    out = _strip_duplicate_heading(md, "Historical Trends and Volatility")
    assert out == "Body text."


def test_strips_at_most_one_leading_blank_line_after_removal():
    md = "## Historical Trends and Volatility\n\n\nBitcoin's price..."
    out = _strip_duplicate_heading(md, "Historical Trends and Volatility")
    assert out == "Bitcoin's price..."


def test_leaves_genuine_prose_untouched():
    """A section that legitimately opens with a sentence sharing a few
    words with the title must NOT be stripped — only a near-exact echo
    of the title triggers removal."""
    md = ("Bitcoin's price has shown volatility since inception, driven "
          "by market trends.")
    out = _strip_duplicate_heading(md, "Historical Trends and Volatility")
    assert out == md


def test_leaves_body_untouched_when_no_title():
    md = "Some prose here."
    assert _strip_duplicate_heading(md, "") == md


def test_leaves_empty_body_untouched():
    assert _strip_duplicate_heading("", "Any Title") == ""


def test_normalize_heading_text_strips_markers_and_casefolds():
    assert _normalize_heading_text("## Historical Trends") == \
        _normalize_heading_text("**historical trends**") == \
        _normalize_heading_text("Historical Trends.") == \
        "historical trends"
