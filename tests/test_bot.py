"""Telegram bot smoke test — only verifies imports/wiring without connecting."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from telegram.constants import ParseMode

from argus.bot.app import build_application
from argus.bot.common import (
    HELP_TEXT, _safe_send, _safe_edit_cb, _stream_progress,
)
from argus.bot.media import video_cmd
from argus.bot.research import _format_plan, _plan_keyboard


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
    for item in ("/engines transcriber", "/status", "/cancel", "/continue", "menu button"):
        assert item in HELP_TEXT, item


def test_help_text_is_compact():
    """The in-chat /help must stay short — the verbose guide lives in
    docs/help.md (sent by /help full). Guard against it creeping back."""
    assert len(HELP_TEXT) < 900, (
        f"HELP_TEXT is {len(HELP_TEXT)} chars — keep it terse; put detail "
        "in docs/help.md")
    assert "full" in HELP_TEXT.lower(), "must point users at /help full"


def test_help_md_exists_and_covers_commands():
    from pathlib import Path
    from argus.bot.common import _HELP_MD_PATH
    assert _HELP_MD_PATH.exists(), "docs/help.md must ship with the bot"
    body = Path(_HELP_MD_PATH).read_text(encoding="utf-8")
    for cmd in ("/research", "/find", "/fetch", "/quality", "/transcripts",
                "/runs", "/append", "/continue", "/delete", "/status", "/cancel",
                "/engines", "/engines transcriber", "menu button"):
        assert cmd in body, f"help.md must document {cmd}"


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
    text = _format_plan({"plan": plan})
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
    text = _format_plan({"plan": plan})
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
    text = _format_plan({"plan": plan})
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
    text = _format_plan({"plan": plan, "sources": sources})
    assert "Found sources" in text
    assert "https://github.com/real/repo" in text
    assert "Real paper" in text
    assert "(2" in text or "2," in text or "2 " in text  # count visible


def test_format_plan_warns_when_live_search_found_nothing():
    plan = {"summary": "s", "sub_questions": ["q"], "planned_sources": []}
    text = _format_plan({"plan": plan, "sources": []})
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
    text = _format_plan({"plan": plan})
    assert "live search" in text.lower()


def test_plan_keyboard_has_buttons():
    kb = _plan_keyboard({})
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


def test_help_mentions_media_commands():
    # /find replaced /video as the primary media-search entry point.
    assert "/find" in HELP_TEXT
    assert "/fetch" in HELP_TEXT


def test_report_keyboard_has_extend_button():
    from argus.bot.research import _report_keyboard
    labels = [b.text for row in _report_keyboard().inline_keyboard for b in row]
    assert any("Extend" in l for l in labels)
    assert any("Send" in l for l in labels)


def test_help_lists_engines():
    """/engines (WP-A) belongs in the compact help (pointer 11)."""
    assert "/engines" in HELP_TEXT


def test_plan_keyboard_rows_and_run_scoping():
    """Lengths (✅ on the chosen), depths (✅ on the chosen), the cycling
    ⚙ profile and 🕒 recency buttons, then Approve / Edit / Cancel — every
    button scoped to the run (WP-E1, spec §3)."""
    kb = _plan_keyboard({"run_id": "ab12cd34", "length": "long", "depth": "exhaustive",
                         "profile": "max-quality", "profile_problem": "ANTHROPIC_API_KEY is not set",
                         "recency": "w"})
    rows = [[(b.text, b.callback_data) for b in row] for row in kb.inline_keyboard]
    assert [d for _, d in rows[0] + rows[1]] == [f"len:{m}:ab12cd34" for m in
                                                ("tldr", "short", "medium", "long", "lecture")]
    assert [t for t, _ in rows[0] + rows[1] if "✅" in t] == ["Long ✅"]
    assert [d for _, d in rows[2]] == [f"dep:{d}:ab12cd34" for d in ("quick", "standard", "deep", "exhaustive",
                                                                     "auto")]      # Task 10 review: an Auto button
    assert [t for t, _ in rows[2] if "✅" in t] == ["Exhaustive ✅"]
    assert rows[3] == [("⚙ max-quality ⚠", "prof:next:ab12cd34"), ("🕒 past week", "rec:next:ab12cd34")]
    assert [d for _, d in rows[4]] == ["plan:approve:ab12cd34", "plan:edit:ab12cd34", "plan:cancel:ab12cd34"]
    assert not any("console" in t.lower() for row in rows for t, _ in row)     # no console button (spec §3)


@pytest.mark.parametrize("payload,shown", [
    ({"exa_enabled": True, "exa_estimate_usd": 0.054}, "Exa: ≈ $0.054"),
    ({"exa_enabled": True, "exa_estimate_usd": None}, "Exa: n/a"),            # no pricing configured
    ({"exa_enabled": False, "exa_estimate_usd": None}, "Exa: off"),
])
def test_format_plan_shows_the_exa_estimate_only_for_an_enabled_exa(payload, shown):
    text = _format_plan({"plan": {"summary": "s"}, **payload})
    assert shown in text
    assert ("≈" in text) is (payload["exa_estimate_usd"] is not None)


@pytest.mark.parametrize("recency,window,shown", [
    (None, {"window": "m", "start_date": "2026-09-20"}, "Recency: auto — past month (since 2026-09-20)"),
    (None, None, "Recency: auto — no filter"),
    ("d", {"window": "d", "start_date": "2026-10-19"}, "Recency: past day (since 2026-10-19)"),
    ("any", None, "Recency: any date (no filter)"),
])
def test_format_plan_shows_archetype_recency_depth_and_profile(recency, window, shown):
    text = _format_plan({"plan": {"summary": "s"}, "archetype": "forecast", "recency": recency,
                         "recency_window": window, "length": "medium", "depth": "standard", "depth_auto": True,
                         "profile": "free-proxy", "profile_problem": None})
    assert text.splitlines()[0] == "📋 *Research plan*  ·  mode: Medium  ·  depth: standard (auto)"
    assert "Archetype: forecast" in text and shown in text and "Engines: free-proxy" in text


def test_the_auto_depth_button_carries_the_check_while_depth_follows_the_length():
    rows = _plan_keyboard({"run_id": "ab12cd34", "length": "long", "depth": "deep", "depth_auto": True}).inline_keyboard
    assert [b.text for b in rows[2] if "✅" in b.text] == ["Auto ✅"]


def test_banners_are_read_only_from_the_title_block():
    from argus.bot.research import _quality_lines
    md = ("# topic\n\n> ⚠ Degraded run — 3 of 9 sources could not be digested (429)\n\n## Findings\n\n"
          "> ⚠ a quoted warning inside the body\n")
    lines = _quality_lines({}, md)
    assert [line for line in lines if line.startswith("⚠")] == [
        "⚠ Degraded run — 3 of 9 sources could not be digested (429)"]


async def test_the_preview_quality_line_is_html_escaped(tmp_path):
    """A banner or a degradation note holding ``<`` or ``&`` reaches the HTML
    preview escaped (the head is HTML, not raw text)."""
    from argus.bot.research import _send_report_preview
    md = tmp_path / "report.md"
    md.write_text("# t\n\n> ⚠ Degraded run — 3 < 5 sources & the rest skipped\n\n## Findings\n\nx\n",
                  encoding="utf-8")
    bot = _FakeBot()
    values = {"report_paths": {"md": str(md), "folder": str(tmp_path)},
              "errors": ["[r1] skip: research.digest https://w.example/y?a=1&b=<2> (429)"]}
    msg, text, mode = await _send_report_preview(bot, 5, "ab12cd34", values)
    head = text.split("<pre>")[0]
    assert mode == ParseMode.HTML
    assert "⚠ Degraded run — 3 &lt; 5 sources &amp; the rest skipped" in head
    assert "https://w.example/y?a=1&amp;b=&lt;2&gt; (429)" in head
    assert "3 < 5" not in head and "<2>" not in head


async def test_a_raising_stop_hook_still_closes_the_saver_and_library():
    """PTB skips ``shutdown()`` and ``post_shutdown`` when ``post_stop``
    raises (pointer 1, the POST_STOP half): ``_post_stop`` runs every hook,
    logs each failure and stores the first; ``_post_shutdown`` closes the
    library and the saver, then re-raises it."""
    from unittest.mock import AsyncMock, MagicMock
    from argus.bot import app as bot_app

    ran: list[str] = []

    async def bad(app):
        ran.append("bad")
        raise RuntimeError("drain failed")

    async def good(app):
        ran.append("good")

    app = SimpleNamespace(bot_data={"library": MagicMock(close=AsyncMock()),
                                    "saver_cm": MagicMock(__aexit__=AsyncMock())})
    stop, shutdown = bot_app.POST_STOP_HOOKS[:], bot_app.POST_SHUTDOWN_HOOKS[:]
    bot_app.POST_STOP_HOOKS[:], bot_app.POST_SHUTDOWN_HOOKS[:] = [bad, good], [good]
    try:
        await bot_app._post_stop(app)                          # returns: PTB goes on to shutdown()
        assert ran == ["bad", "good"]
        with pytest.raises(RuntimeError, match="drain failed"):
            await bot_app._post_shutdown(app)
    finally:
        bot_app.POST_STOP_HOOKS[:], bot_app.POST_SHUTDOWN_HOOKS[:] = stop, shutdown
    assert ran == ["bad", "good", "good"]
    app.bot_data["library"].close.assert_awaited_once()
    app.bot_data["saver_cm"].__aexit__.assert_awaited_once()


async def test_hook_runner_runs_every_hook_and_reraises_the_first_failure():
    """Pointer 1: ``_run_hooks`` runs every hook even when one raises and
    re-raises the first failure; a raising shutdown hook never skips the
    library and saver close. (A raising *stop* hook would skip PTB's
    ``shutdown()`` — that path is the test above.)"""
    from unittest.mock import AsyncMock, MagicMock
    from argus.bot import app as bot_app

    ran: list[str] = []

    async def bad(app):
        ran.append("bad")
        raise RuntimeError("tunnel stop failed")

    async def worse(app):
        ran.append("worse")
        raise ValueError("second failure")

    async def good(app):
        ran.append("good")

    app = SimpleNamespace(bot_data={"library": MagicMock(close=AsyncMock()),
                                    "saver_cm": MagicMock(__aexit__=AsyncMock())})
    with pytest.raises(RuntimeError, match="tunnel stop failed"):
        await bot_app._run_hooks([bad, worse, good], app)
    assert ran == ["bad", "worse", "good"]

    ran.clear()
    original = bot_app.POST_SHUTDOWN_HOOKS[:]
    bot_app.POST_SHUTDOWN_HOOKS[:] = [bad, good]
    try:
        with pytest.raises(RuntimeError, match="tunnel stop failed"):
            await bot_app._post_shutdown(app)
    finally:
        bot_app.POST_SHUTDOWN_HOOKS[:] = original
    assert ran == ["bad", "good"]
    app.bot_data["library"].close.assert_awaited_once()
    app.bot_data["saver_cm"].__aexit__.assert_awaited_once()


def test_the_run_driver_hooks_are_registered():
    from argus.bot import app as bot_app, runs
    assert runs.init in bot_app.POST_INIT_HOOKS and runs.drain_all in bot_app.POST_STOP_HOOKS

# --- live check 12, item 1: the plan card always fits Telegram and parses --------

_TG_LIMIT = 4096


def _tg_len(text: str) -> int:
    """Telegram counts UTF-16 code units."""
    return len(text.encode("utf-16-le")) // 2


def _tg_legacy_markdown(text: str) -> str:
    """Telegram's legacy Markdown parser (tdlib ``parse_markdown``), strict: the
    parsed text, or ValueError where Telegram answers "Can't parse entities".
    ``\\`` escapes only ``_ * ` [``; inside an entity nothing is escaped and the
    entity ends at its first end character; a link's URL ends at its first ``)``."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n and text[i + 1] in "_*`[":
            out.append(text[i + 1])
            i += 2
            continue
        if c not in "_*`[":
            out.append(c)
            i += 1
            continue
        begin, end, pre = i, "]" if c == "[" else c, False
        i += 1
        if c == "`" and text[i:i + 2] == "``":
            pre, i = True, i + 2
        while i < n and (text[i] != end or (pre and text[i + 1:i + 3] != "``")):
            out.append(text[i])
            i += 1
        if i >= n:
            raise ValueError(f"Can't find end of the entity starting at offset {begin}")
        if c == "[" and text[i + 1:i + 2] == "(":
            close = text.find(")", i + 2)
            if close < 0:
                raise ValueError(f"Can't find end of a URL at offset {i + 2}")
            i = close
        i += 3 if pre else 1
    return "".join(out)


def test_the_strict_checker_rejects_the_live_failure():
    """The 05:18:22 card: an escaped ``_`` inside an italic entity ends it early."""
    with pytest.raises(ValueError, match="Can't find end"):
        _tg_legacy_markdown("- `web` — _LangGraph \\_\\_interrupt\\_\\_ field stream mode_")
    assert _tg_legacy_markdown("a \\_b\\_ *c* [d](https://e.example/f_(g)") == "a _b_ c d"


def _long_plan_payload() -> dict:
    nasty = "LangGraph __interrupt__ *stream* mode [beta] `code` snake_case ] ) "
    plan = {"summary": "Summary: " + nasty * 3,
            "sub_questions": [f"Q{i}: " + nasty * 8 for i in range(7)],
            "planned_sources": [{"kind": "web_search`x", "query": f"intent {i}: " + nasty * 3} for i in range(14)]}
    sources = [{"title": f"T{i} " + nasty * 2, "url": f"https://en.wikipedia.org/wiki/Foo_(bar)_{i}?q=a_b"}
               for i in range(40)]
    return {"plan": plan, "sources": sources, "run_id": "abcd1234", "length": "medium", "depth": "standard",
            "depth_auto": True, "archetype": "how_to", "recency": None, "recency_window": None,
            "profile": "free_proxy", "profile_problem": "role panel_grounding: choose *a* model",
            "exa_enabled": True, "exa_estimate_usd": 0.1234, "error": "bad_[thing]"}


def test_a_long_plan_card_fits_telegram_and_its_markdown_parses():
    """7 long sub-questions, 14 intents, 40 sources, entity characters in every
    model-written field: the card fits 4096 (measured on the source, which is
    never shorter than the parsed text) and parses; the header, the knobs and
    the buttons survive, and every cut list says how much it left out."""
    payload = _long_plan_payload()
    text = _format_plan(payload)
    assert _tg_len(text) <= _TG_LIMIT, _tg_len(text)
    parsed = _tg_legacy_markdown(text)
    assert parsed.splitlines()[0] == "📋 Research plan  ·  mode: Medium  ·  depth: standard (auto)"
    for part in ("Archetype: how_to", "Recency: auto", "Engines: free_proxy", "role panel_grounding",
                 "Exa: ≈ $0.123", "bad_[thing]"):
        assert part in parsed, part
    assert all(f"Q{i}:" in parsed for i in range(7)), "the sub-questions are cut last"
    assert "__interrupt__" in parsed, "model text keeps its characters where it is not cut"
    assert "more" in parsed and "…and" in parsed
    labels = [b.text for row in _plan_keyboard(payload).inline_keyboard for b in row]
    assert any("Approve" in t for t in labels) and any("Edit" in t for t in labels)


def test_a_plan_card_with_a_gate_note_still_fits():
    """The gate edits append a note (approved, redrafting, the Edit prompt)."""
    from argus.bot import runs
    text = _format_plan(_long_plan_payload())
    for _gate, note in runs._REPLY_PROMPTS.values():
        assert _tg_len(f"{text}\n\n{note}") <= _TG_LIMIT
    _tg_legacy_markdown(f"{text}\n\n{runs._REPLY_PROMPTS['plan_edit'][1]}")


def test_a_short_plan_card_is_not_cut():
    payload = {"plan": {"summary": "s", "sub_questions": ["a_b?", "c*d?"],
                        "planned_sources": [{"kind": "web", "query": "x_y"}]},
               "sources": [{"title": "T_1", "url": "https://a.example/p_(1)"}]}
    text = _format_plan(payload)
    parsed = _tg_legacy_markdown(text)
    assert "a_b?" in parsed and "c*d?" in parsed and "x_y" in parsed and "T_1" in parsed
    assert "more" not in parsed


class _RejectingBot:
    """Rejects every send of a long text (Markdown and plain), accepts a short one."""

    def __init__(self):
        self.sent: list[tuple[str, object, object]] = []

    async def send_message(self, chat_id=None, text="", reply_markup=None, parse_mode=None, **kw):
        if len(text) > 200:
            raise RuntimeError("Message is too long")
        self.sent.append((text, parse_mode, reply_markup))
        return SimpleNamespace(message_id=7)


async def test_a_send_that_still_fails_is_never_silent_and_keeps_the_buttons():
    bot, markup = _RejectingBot(), _plan_keyboard({"run_id": "abcd1234"})
    msg = await _safe_send(bot, 1, "x" * 5000, reply_markup=markup, fallback="📋 Plan ready — tap below.")
    assert msg is not None
    assert bot.sent == [("📋 Plan ready — tap below.", None, markup)]
    msg = await _safe_send(bot, 1, "y" * 5000, reply_markup=markup)
    assert msg is not None and bot.sent[-1][2] is markup
    assert bot.sent[-1][0].startswith("⚠️") and "Message is too long" in bot.sent[-1][0]


def _preview_values(tmp_path, note_chars: int) -> dict:
    md = tmp_path / "report.md"
    md.write_text("# t\n\n## A\n\n" + "Body & <text> _x_. " * 400, encoding="utf-8")
    errors = [f"[r1] skip: research.digest https://x.example/{i} ({'429 ' * (note_chars // 4)})" for i in range(8)]
    return {"report_paths": {"md": str(md), "folder": str(tmp_path), "pdf": None}, "length": "medium",
            "errors": errors, "panel_verdict": {"judged": 3, "total": 3}, "findings": []}


@pytest.mark.parametrize("note_chars", [50, 900])
def test_the_report_preview_always_fits_telegram(tmp_path, note_chars):
    """Long degradation notes (error text) once pushed the preview's head past
    4096 even with no excerpt left."""
    from argus.bot.research import _preview_bodies
    html_body, plain_body = _preview_bodies(_preview_values(tmp_path, note_chars))
    for body in (html_body, plain_body):
        assert _tg_len(body) + 200 <= _TG_LIMIT, _tg_len(body)     # room for a gate note
    assert "Quality: judged 3/3" in html_body


async def test_a_report_preview_that_cannot_be_sent_still_carries_its_buttons(tmp_path):
    from argus.bot.research import _send_report_preview
    bot = _RejectingBot()
    msg, text, _mode = await _send_report_preview(bot, 1, "abcd1234", _preview_values(tmp_path, 50))
    assert msg is not None
    (sent, _pm, markup), = bot.sent
    assert sent.startswith("⚠️") and markup is not None
    assert [b.callback_data for row in markup.inline_keyboard for b in row][0] == "report:send:abcd1234"


async def test_a_long_markdown_message_goes_out_cut_on_its_plain_retry():
    """A long /ask answer: its Markdown send fails, the plain retry fits Telegram."""
    sent: list[str] = []

    class _Bot:
        async def send_message(self, chat_id=None, text="", reply_markup=None, parse_mode=None, **kw):
            if parse_mode is not None or _tg_len(text) > _TG_LIMIT:
                raise RuntimeError("Message is too long")
            sent.append(text)
            return SimpleNamespace(message_id=1)

    answer = "🧠 " + "a long answer with *markdown* " * 300
    assert await _safe_send(_Bot(), 1, answer) is not None
    (text,) = sent
    assert _tg_len(text) <= _TG_LIMIT and text.endswith("…(cut)") and text.startswith("🧠 a long answer")


# --- live-check fix round 2: the cut always terminates --------------------------

def _within(seconds: float, fn):
    """``fn()`` on a daemon thread; its result, or AssertionError if it is still running."""
    import threading
    out: list = []
    t = threading.Thread(target=lambda: out.append(fn()), daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), f"still running after {seconds} s"
    return out[0]


def test_tg_cut_terminates_and_fits_for_any_limit():
    from argus.bot.common import _tg_cut

    def every():
        for text in ("abcdefghij", "😀" * 12, "x" * 5000):
            for limit in range(-5, 11):
                cut = _tg_cut(text, limit)
                assert _tg_len(cut) <= max(limit, 0), (text[:5], limit, cut)
        return True
    assert _within(5, every)


def test_an_escape_heavy_preview_head_returns_within_the_budget(tmp_path):
    """Review Important 1: 8 notes of 300 "&" each (5x on escaping) drove the
    head-cut limit negative and _tg_cut looped forever on the event loop."""
    from argus.bot.research import _PREVIEW_BUDGET, _preview_bodies
    values = _preview_values(tmp_path, 50)
    values["errors"] = [f"[r1] skip: research.digest https://x.example/{i}?{'&' * 300} (429)" for i in range(8)]
    html_body, plain_body = _within(10, lambda: _preview_bodies(values))
    assert _tg_len(html_body) <= _PREVIEW_BUDGET and _tg_len(plain_body) <= _PREVIEW_BUDGET
    assert html_body.startswith("📝 *Report preview*")
