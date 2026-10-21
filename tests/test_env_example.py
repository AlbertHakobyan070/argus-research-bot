"""`.env.example` parses as it reads (WP-H fix round 1).

`cp .env.example .env` is the documented setup, and python-dotenv (what
`argus.config` loads `.env` with) reads `KEY=   # comment` as the value
"# comment" when the value is blank. So every comment sits on its own line,
and a key — active or commented out, ready to uncomment — carries nothing
after its value."""
from __future__ import annotations

import io
import re
from pathlib import Path

from dotenv import dotenv_values

EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"
PLACEHOLDERS = {"FREELLMAPI_BASE_URL"}            # the only active key with a value


def _uncommented() -> str:
    """Every commented-out ``# KEY=…`` line as the user would uncomment it."""
    lines = EXAMPLE.read_text(encoding="utf-8").splitlines()
    return "\n".join(re.sub(r"^#\s?", "", ln) for ln in lines if re.match(r"^#\s?[A-Z][A-Z0-9_]*=", ln))


def test_no_parsed_value_is_comment_text():
    for source, values in (("active", dotenv_values(EXAMPLE)),
                           ("uncommented", dotenv_values(stream=io.StringIO(_uncommented())))):
        assert values, source
        bad = {k: v for k, v in values.items() if v and v.lstrip().startswith("#")}
        assert not bad, f"{source} keys parse a comment as their value: {sorted(bad)}"


def test_blank_keys_parse_empty():
    values = dotenv_values(EXAMPLE)
    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USER_ID", "FREELLMAPI_API_KEY", "EXA_API_KEY"):
        assert key in values, key
    assert {k: v for k, v in values.items() if k not in PLACEHOLDERS and v != ""} == {}
    optional = dotenv_values(stream=io.StringIO(_uncommented()))
    assert {k: v for k, v in optional.items() if v and " " in v} == {}, "a value with trailing text"
