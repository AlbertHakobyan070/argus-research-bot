"""The bot token never reaches a log line, and ARGUS_LOG_HTTP silences httpx's request lines.

Telegram puts the token in every API URL and httpx logs each request at INFO,
so `HTTP Request: POST https://api.telegram.org/bot<token>/getUpdates` used to
land, token included, in the console and in any file it is redirected to.

Hermetic: synthetic tokens only (the real value from .env is removed from the
environment for every test), no network (httpx runs on a MockTransport).
`_configure_logging()` runs on a handler-less root logger inside
`_clean_logging()`, which puts the root's handlers and levels back afterwards,
so other tests' caplog keeps working."""
from __future__ import annotations

import contextlib
import io
import logging
import re

import httpx
import pytest

from argus.bot import app as bot_app

TOKEN = "123456789:AAEexampleexampleexampleexample_x1"      # synthetic, token-shaped
SECRET = TOKEN.split(":")[1]
URL = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
LOGGERS = ["httpx", "telegram.ext.Application", "argus.bot", "apscheduler.scheduler"]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    # argus.config loaded the real .env into os.environ when it was imported:
    # neither the real token nor a real knob may take part in these tests.
    for name in ("TELEGRAM_BOT_TOKEN", "ARGUS_LOG", "ARGUS_LOG_HTTP"):
        monkeypatch.delenv(name, raising=False)


@contextlib.contextmanager
def _clean_logging():
    """A handler-less root and unset httpx/httpcore levels, so basicConfig acts
    as at process start. pytest keeps its capture handlers on the root, which
    makes basicConfig a no-op, and puts them back at the start of the test body,
    so a fixture cannot empty the root: this has to run inside the test.
    Handlers and levels are put back on exit."""
    root = logging.getLogger()
    quiet = [logging.getLogger(name) for name in ("httpx", "httpcore")]
    handlers, level, levels = root.handlers[:], root.level, [lg.level for lg in quiet]
    root.handlers[:] = []
    for lg in quiet:
        lg.setLevel(logging.NOTSET)
    try:
        yield
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers[:] = handlers
        root.setLevel(level)
        for lg, lvl in zip(quiet, levels):
            lg.setLevel(lvl)


def _configured() -> io.StringIO:
    """Run _configure_logging() and point the handler it installed at a buffer."""
    bot_app._configure_logging()
    (handler,) = logging.getLogger().handlers
    out = io.StringIO()
    handler.setStream(out)
    return out


def _post_through_httpx() -> None:
    """One real httpx request on a MockTransport: httpx itself writes the log line."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True, "result": []}))
    with httpx.Client(transport=transport) as client:
        client.post(URL)


# ---- the token is redacted ----------------------------------------------------------------


@pytest.mark.parametrize("name", LOGGERS)
def test_the_token_in_a_url_is_redacted_on_every_logger(name):
    with _clean_logging():
        out = _configured()
        logging.getLogger(name).info('HTTP Request: %s %s "%s %d %s"', "POST", URL, "HTTP/1.1", 200, "OK")
        text = out.getvalue()
    assert "https://api.telegram.org/bot<redacted>/getUpdates" in text
    assert TOKEN not in text and SECRET not in text


def test_a_real_httpx_request_line_is_redacted():
    with _clean_logging():
        out = _configured()
        _post_through_httpx()
        text = out.getvalue()
    assert "HTTP Request: POST https://api.telegram.org/bot<redacted>/getUpdates" in text
    assert TOKEN not in text and SECRET not in text


def test_an_exception_message_and_its_traceback_are_redacted():
    with _clean_logging():
        out = _configured()
        try:
            try:
                raise ConnectionError(f"cannot reach {URL}")
            except ConnectionError as exc:
                raise RuntimeError("polling failed") from exc
        except RuntimeError:
            logging.getLogger("argus.bot").exception("poll of %s failed", URL)
        text = out.getvalue()
    message, _, traceback = text.partition("Traceback")
    assert "poll of https://api.telegram.org/bot<redacted>/getUpdates failed" in message
    assert "ConnectionError: cannot reach https://api.telegram.org/bot<redacted>/getUpdates" in traceback
    assert TOKEN not in text and SECRET not in text


def test_stack_text_is_redacted_too():
    stack = f'Stack (most recent call last):\n  File "poll.py", line 3, in poll\n    get("{URL}")'
    record = logging.LogRecord("argus.bot", logging.WARNING, __file__, 1, "slow", None, None, sinfo=stack)
    with _clean_logging():
        _configured()
        (handler,) = logging.getLogger().handlers
        text = handler.format(record)
    assert 'get("https://api.telegram.org/bot<redacted>/getUpdates")' in text
    assert TOKEN not in text and SECRET not in text


def test_the_configured_token_is_redacted_without_the_bot_prefix(monkeypatch):
    # python-telegram-bot's InvalidToken message carries the bare token.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    with _clean_logging():
        out = _configured()
        logging.getLogger("telegram.ext.Application").error("The token `%s` was rejected by the server.", TOKEN)
        text = out.getvalue()
    assert "The token `<redacted>` was rejected by the server." in text
    assert TOKEN not in text and SECRET not in text


@pytest.mark.parametrize("token_env", [None, ""])
def test_ordinary_lines_are_left_alone(monkeypatch, token_env):
    # An empty TELEGRAM_BOT_TOKEN must not turn into str.replace("", ...) on every character.
    if token_env is not None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token_env)
    line = "bot_data ready at 12:34:56; fetched 1234:5678 of 99"
    with _clean_logging():
        out = _configured()
        logging.getLogger("argus.bot").info(line)
        text = out.getvalue()
    assert text.endswith(f"argus.bot: {line}\n")


# ---- the old level and format, one handler -------------------------------------------------


def test_the_level_and_format_are_the_old_ones(monkeypatch):
    monkeypatch.setenv("ARGUS_LOG", "DEBUG")
    with _clean_logging():
        out = _configured()
        level = logging.getLogger().level
        logging.getLogger("argus.bot").debug("hello")
        text = out.getvalue()
    assert level == logging.DEBUG
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} DEBUG argus\.bot: hello\n", text)


def test_configuring_twice_does_not_stack_handlers_or_formatters():
    with _clean_logging():
        out = _configured()
        (handler,) = logging.getLogger().handlers
        bot_app._configure_logging()
        assert logging.getLogger().handlers == [handler]
        logging.getLogger("argus.bot").info("poll %s", URL)
        text = out.getvalue()
    assert text.count("\n") == 1 and text.count("bot<redacted>") == 1


def test_main_configures_logging_before_it_builds_the_bot(monkeypatch):
    class Stop(Exception):
        pass

    seen = []

    def build():
        seen.append("build")
        raise Stop

    monkeypatch.setattr(bot_app, "_configure_logging", lambda: seen.append("logging"))
    monkeypatch.setattr(bot_app, "build_application", build)      # main() stops here: no bot, no polling
    with pytest.raises(Stop):
        bot_app.main()
    assert seen == ["logging", "build"]


# ---- ARGUS_LOG_HTTP -----------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "1", "true", "yes", "on"])
def test_request_logging_is_on_by_default(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("ARGUS_LOG_HTTP", value)
    with _clean_logging():
        bot_app._configure_logging()
        levels = [logging.getLogger(name).getEffectiveLevel() for name in ("httpx", "httpcore")]
    assert levels == [logging.INFO, logging.INFO]


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_the_knob_off_sets_httpx_and_httpcore_to_warning(monkeypatch, value):
    monkeypatch.setenv("ARGUS_LOG_HTTP", value)
    monkeypatch.setenv("ARGUS_LOG", "DEBUG")        # httpcore only talks at DEBUG: the knob must win
    with _clean_logging():
        bot_app._configure_logging()
        levels = [logging.getLogger(name).getEffectiveLevel() for name in ("httpx", "httpcore")]
    assert levels == [logging.WARNING, logging.WARNING]


def test_with_the_knob_off_request_lines_stop_but_warnings_still_show(monkeypatch):
    monkeypatch.setenv("ARGUS_LOG_HTTP", "0")
    with _clean_logging():
        out = _configured()
        _post_through_httpx()
        logging.getLogger("httpx").warning("pool is slow for %s", URL)
        text = out.getvalue()
    assert "HTTP Request" not in text
    assert "pool is slow for https://api.telegram.org/bot<redacted>/getUpdates" in text


def test_a_garbage_knob_value_raises_before_logging_is_touched(monkeypatch):
    monkeypatch.setenv("ARGUS_LOG_HTTP", "maybe")
    with _clean_logging():
        with pytest.raises(ValueError, match="ARGUS_LOG_HTTP"):
            bot_app._configure_logging()
        assert logging.getLogger().handlers == []
