"""Robust JSON parsing for weak-model output — shared by every v3 node.

Extracted from nodes.py (v2) where these helpers were hardened against
the 2026-07-12 empty-report bug: proxy-rerouted weak models write valid
findings but break the surrounding JSON (literal newlines inside string
values, fences, truncation). The v3 engine keeps every structured LLM
output SMALL, but still routes all of them through this parser.
"""
from __future__ import annotations

import json
import re


def repair_json(t: str) -> str:
    """Escape literal control chars inside JSON string values.

    The #1 cause of parse failures: a model writes a long string value
    with LITERAL newlines/tabs, which is invalid JSON. We walk the text
    tracking whether we're inside a quoted string and escape raw
    \\n \\r \\t so ``json.loads`` can recover the object.
    """
    out: list[str] = []
    in_str = False
    escaped = False
    for ch in t:
        if in_str:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                in_str = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return "".join(out)


def parse_json_obj(text: str, *, min_keys: int = 1,
                   require_any: tuple[str, ...] | None = None) -> dict:
    """Best-effort JSON parse; tolerates ```json fences and embedded JSON.

    Strategy:
      1. Strip outer fences and try direct parse (raw, then repaired).
      2. Balanced-brace match — outermost valid JSON object (raw, then
         repaired), respecting ``require_any``.
      3. Fall back to a ```json fenced block.

    ``min_keys`` rejects trivially-empty objects like ``{}``.
    ``require_any`` (e.g. ``("findings","no_evidence")``) forces the
    parser to accept ONLY an object containing at least one of those
    top-level keys — so a truncated/newline-broken response can't cause
    it to latch onto an inner object.
    """
    raw = text or ""

    def _accept(d) -> dict | None:
        if not isinstance(d, dict):
            return None
        if len(d) < min_keys:
            return None
        if require_any and not any(k in d for k in require_any):
            return None
        return d

    def _try(candidate: str) -> dict | None:
        for variant in (candidate, repair_json(candidate)):
            try:
                d = _accept(json.loads(variant))
                if d is not None:
                    return d
            except Exception:
                continue
        return None

    # 1. Direct (after fence stripping), raw then repaired.
    t = raw.strip()
    t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()
    d = _try(t)
    if d is not None:
        return d
    # 2. Balanced braces — every top-level '{...}' span, outermost first.
    s = t
    i = 0
    while i < len(s):
        if s[i] == "{":
            depth = 0
            for j in range(i, len(s)):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        d = _try(s[i:j + 1])
                        if d is not None:
                            return d
                        break
            i += 1
        else:
            i += 1
    # 3. Try json-fenced block.
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL |
                  re.IGNORECASE)
    if m:
        d = _try(m.group(1))
        if d is not None:
            return d
    raise ValueError(
        f"no useful JSON object found in: {raw[:200]!r} "
        f"(required >= {min_keys} keys"
        + (f", any of {require_any}" if require_any else "") + ")"
    )


def salvage_objects(text: str, *, required_keys: tuple[str, ...]) -> list[dict]:
    """Extract objects with the given keys from unrecoverable JSON.

    Scans for ``{...}`` spans that contain every key in ``required_keys``
    and returns them in order. Last-resort so real content the model
    produced isn't thrown away because its wrapper JSON broke.
    """
    out: list[dict] = []
    s = text or ""
    i = 0
    while i < len(s):
        if s[i] == "{":
            depth = 0
            for j in range(i, len(s)):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        span = s[i:j + 1]
                        for variant in (span, repair_json(span)):
                            try:
                                obj = json.loads(variant)
                            except Exception:
                                continue
                            if (isinstance(obj, dict)
                                    and all(obj.get(k) for k in required_keys)):
                                out.append(obj)
                            break
                        i = j
                        break
        i += 1
    return out


def salvage_findings(text: str) -> list[dict]:
    """v2-compatible wrapper: finding objects have claim + citation_urls."""
    return salvage_objects(text, required_keys=("claim", "citation_urls"))


__all__ = ["repair_json", "parse_json_obj", "salvage_objects",
           "salvage_findings"]
