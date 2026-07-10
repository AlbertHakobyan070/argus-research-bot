"""Telegram bot smoke test — only verifies imports/wiring without connecting."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from telegram.constants import ParseMode

from argus.bot import (
    HELP_TEXT, build_application, _format_plan, _plan_keyboard,
    _safe_send, _safe_edit_cb, _stream_progress, video_cmd,
)


class _FakeBot:
    """Bot double that mimics Telegram rejecting markdown with unbalanced
    entities (underscores in paths) but accepting plain text."""

    def __init__(self):
        self.sent: list[tuple[str, object]] = []
        self._mid = 100

    async def send_message(self, chat_id=None, text="", reply_markup=None,
                           parse_mode=None, **kw):
        if parse_mode == ParseMode.MARKDOWN and "_" in text:
            raise RuntimeError("Can't parse entities: byte offset 113")
        self._mid += 1
        self.sent.append((text, parse_mode))
        return SimpleNamespace(message_id=self._mid)

    async def edit_message_text(self, chat_id=None, message_id=None, text="",
                                parse_mode=None, **kw):
        if parse_mode == ParseMode.MARKDOWN and "_" in text:
            raise RuntimeError("Can't parse entities: byte offset 113")
        return SimpleNamespace(message_id=message_id)


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
            {"kind": "blog", "query": "metacognitive RL transformers",
             "target_url": "https://blog.ought.com/fabricated-x",
             "rationale": "adjacent commentary"},
        ],
    }
    text = _format_plan(plan)
    assert "Research plan" in text
    assert "Investigate X" in text
    assert "What is X?" in text
    assert "paper" in text
    # query takes precedence — the fabricated blog URL must NOT be shown
    # raw, even when it's the only hint we have:
    assert "metacognitive RL transformers" in text
    assert "blog.ought.com/fabricated-x" not in text


def test_format_plan_prefers_query_over_target_url():
    """Planner target_urls are LLM-invented (knowledge-cutoff fantasy) and
    are ignored by the researcher subgraph. The plan preview must show
    the search intent (query) and never the fabricated URL."""
    plan = {
        "summary": "Investigate X",
        "sub_questions": [],
        "planned_sources": [
            {"kind": "blog", "query": "metacognitive RL transformers",
             "target_url": "https://blog.ought.com/fabricated-x",
             "rationale": "adjacent"},
        ],
    }
    text = _format_plan(plan)
    # The fabricated URL the researcher will never use must not appear
    # anywhere in the plan preview.
    assert "blog.ought.com/fabricated-x" not in text
    # The query it WOULD search for must.
    assert "metacognitive RL transformers" in text


def test_format_plan_never_shows_planner_urls():
    """Even when a planner source has ONLY a target_url (no query), the
    URL must NOT be shown — planner URLs are guesses, and displaying
    them is exactly the 'garbage sources' Albert reported (2026-07-10:
    thelema.org/officialdocs, github.com/otto-xyz/crowley…). Real URLs
    come from the live-search results section instead."""
    plan = {
        "summary": "",
        "sub_questions": [],
        "planned_sources": [
            {"kind": "official_doc", "query": "", "target_url":
             "https://example.com/x", "rationale": "docs"},
        ],
    }
    text = _format_plan(plan)
    assert "example.com/x" not in text, (
        "planner-invented URLs must never reach the plan preview")


def test_format_plan_lists_live_found_sources():
    """When real search results are passed in, the preview must list
    them (title + URL) so the user approves ACTUAL sources."""
    plan = {"summary": "s", "sub_questions": ["q"], "planned_sources": []}
    sources = [
        {"title": "Real repo", "url": "https://github.com/real/repo",
         "kind": "repo"},
        {"title": "Real paper", "url": "https://arxiv.org/abs/2501.00001",
         "kind": "paper"},
    ]
    text = _format_plan(plan, sources=sources)
    assert "Found sources" in text
    assert "https://github.com/real/repo" in text
    assert "Real paper" in text
    assert "(2" in text or "2," in text or "2 " in text  # count visible


def test_format_plan_warns_when_live_search_found_nothing():
    plan = {"summary": "s", "sub_questions": ["q"], "planned_sources": []}
    text = _format_plan(plan, sources=[])
    assert "no sources" in text.lower(), (
        "an empty live-search result must be called out loudly at the "
        "gate — approving would fetch nothing")


def test_format_plan_no_query_no_url_says_live_search():
    plan = {
        "summary": "",
        "sub_questions": [],
        "planned_sources": [
            {"kind": "search", "rationale": "broad"},
        ],
    }
    text = _format_plan(plan)
    assert "live search" in text.lower()


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


# --- markdown-parse resilience (the "Can't parse entities" / "resume failed"
#     bug hit live on 2026-07-09: report folder paths with underscores broke
#     legacy-markdown sends and crashed the resume) ------------------------------

async def test_safe_send_falls_back_to_plain_on_markdown_error():
    bot = _FakeBot()
    msg = await _safe_send(bot, 1, "A:/reports/2026_metacognitive_rl_transformers")
    assert msg is not None, "message must still be sent"
    assert bot.sent[-1][1] is None, "should have retried as plain text"


async def test_stream_progress_survives_underscore_path():
    """The exact crash: '📝 Report ready: <folder with underscores>'."""
    bot = _FakeBot()
    last = await _stream_progress(
        bot, 1, "📝 Report ready: A:/r/20260709_topic_long", {"message_id": None})
    assert last["message_id"] is not None
    assert bot.sent[-1][1] is None  # plain fallback used


async def test_safe_edit_cb_falls_back_to_plain():
    calls: list[object] = []

    class _Q:
        async def edit_message_text(self, text, reply_markup=None, parse_mode=None):
            if parse_mode == ParseMode.MARKDOWN and "_" in text:
                raise RuntimeError("Can't parse entities")
            calls.append(parse_mode)

    await _safe_edit_cb(_Q(), "plan with _underscores_ and *stars*")
    assert calls and calls[-1] is None  # plain fallback used


def test_help_mentions_video_command():
    assert "/video" in HELP_TEXT


def test_report_keyboard_has_extend_button():
    from argus.bot import _report_keyboard
    labels = [b.text for row in _report_keyboard().inline_keyboard for b in row]
    assert any("Extend" in l for l in labels)
    assert any("Send" in l for l in labels)