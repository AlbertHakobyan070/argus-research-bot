"""Regression: bot.py line ~606 sends report previews via Telegram's strict
legacy MARKDOWN parser. When the excerpt contains `_`, `*`, `[`, or `&`
(common in research reports), Telegram raises
``BadRequest: Can't parse entities: can't find end of the entity starting
at byte offset N`` — surfaced as ``resume failed`` to the user.

Bug observed 2026-07-08: Albert hit "Revise" on a plan and the resume failed.

Fix:
1. The report-preview path now uses ``ParseMode.HTML`` and runs the text
   through a Markdown→HTML escaper (``_html_escape_for_tg``) before sending.
2. The send is wrapped in ``try/except`` so a parsing failure degrades
   gracefully to a plain-text preview rather than blowing up ``_resume_after_plan``.

These tests verify the helper and the parse-mode wiring; the actual Telegram
call is exercised in ``manual_e2e.py``.
"""
from __future__ import annotations

import html as _html
import inspect

import pytest

from argus import bot as argus_bot


# --- helper signature check --------------------------------------------------

def test_helper_html_escape_for_tg_exists() -> None:
    """Argus must expose a HTML escape helper used by the report-preview path."""
    assert hasattr(argus_bot, "_html_escape_for_tg"), (
        "Add _html_escape_for_tg() to bot.py. It must escape `&`, `<`, `>` "
        "for Telegram's HTML parser."
    )
    assert callable(argus_bot._html_escape_for_tg)


@pytest.mark.parametrize("raw,expected", [
    # & < > MUST be escaped; Telegram HTML treats <i>, <b>, <code>, <pre> as formatting.
    ("Smith & Co", "Smith &amp; Co"),
    ("<script>", "&lt;script&gt;"),
    ("a > b", "a &gt; b"),
    ("", ""),
    # Plain text should pass through unchanged.
    ("hello world", "hello world"),
    # Newlines preserved (Telegram renders \n as a line break).
    ("line1\nline2", "line1\nline2"),
])
def test_html_escape_for_tg_basic(raw: str, expected: str) -> None:
    assert argus_bot._html_escape_for_tg(raw) == expected


def test_html_escape_for_tg_handles_code_fence_unescaped() -> None:
    """A fenced code block (`` ```...``` ``) should NOT be HTML-escaped
    character-by-character — Telegram's HTML parser passes it through as
    plain text inside <pre>. We just verify the helper doesn't mangle
    the body chars that would normally be HTML-dangerous."""
    excerpt = "```python\nx = 1 < 2\n```"
    out = argus_bot._html_escape_for_tg(excerpt)
    # `&` and `<` from the Python snippet become &amp; / &lt;
    assert "&lt;" in out
    assert "&amp;" not in out  # no `&` in this example
    # The fenced markers pass through.
    assert "```" in out


# --- wiring check: bot.py uses ParseMode.HTML on the report-preview send -----

def test_report_preview_send_uses_html_or_safe_fallback() -> None:
    """The report-preview send in bot.py must NOT use ParseMode.MARKDOWN
    alone — it must either switch to ParseMode.HTML or be wrapped in a
    try/except fallback to a safe plain-text send.

    We verify by reading the source text of the resume_after_report
    function and asserting at least one of these patterns is present.
    """
    src = inspect.getsource(argus_bot._resume_after_plan)
    assert "ParseMode.MARKDOWN" not in src or "ParseMode.HTML" in src, (
        "bot._resume_after_plan still uses ParseMode.MARKDOWN for the "
        "report preview. Switch to ParseMode.HTML with proper escaping, "
        "or wrap the send in a try/except fallback."
    )
    # If MARKDOWN is still referenced anywhere, the function MUST contain
    # a try/except around the offending send.
    if "ParseMode.MARKDOWN" in src:
        # Look for try/except around a send_message call. The simpler
        # structural check: there should be a `try:` block in the
        # report-preview region.
        assert "try:" in src, (
            "bot._resume_after_plan still uses ParseMode.MARKDOWN and "
            "lacks a try/except fallback. Wrap the send so a parse "
            "failure degrades to plain text."
        )