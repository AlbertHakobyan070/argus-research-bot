"""Telegram bot smoke test — only verifies imports/wiring without connecting."""
from __future__ import annotations

import pytest

from argus.bot import (
    HELP_TEXT, build_application, _format_plan, _plan_keyboard,
)


def test_help_text_nonempty():
    assert "Argus" in HELP_TEXT
    assert "/research" in HELP_TEXT
    assert "/ask" in HELP_TEXT


def test_format_plan_basic():
    plan = {
        "summary": "Investigate X",
        "sub_questions": ["What is X?", "Why does X matter?"],
        "planned_sources": [
            {"kind": "paper", "query": "X", "target_url": None,
             "rationale": "primary source"},
            {"kind": "official_doc", "query": "", "target_url":
             "https://example.com/x", "rationale": "docs"},
        ],
    }
    text = _format_plan(plan)
    assert "Research plan" in text
    assert "Investigate X" in text
    assert "What is X?" in text
    assert "paper" in text
    assert "example.com/x" in text


def test_plan_keyboard_has_buttons():
    kb = _plan_keyboard()
    labels = [b.text for row in kb.inline_keyboard for b in row]
    assert any("Approve" in l for l in labels)
    assert any("Cancel" in l for l in labels)


def test_build_application_requires_token(monkeypatch):
    """If the env token is removed, build_application must raise."""
    import importlib
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "")
    # Invalidate cached settings.
    config_mod = importlib.import_module("argus.config")
    config_mod._cached = None
    with pytest.raises(RuntimeError):
        build_application()
    # restore
    config_mod._cached = None