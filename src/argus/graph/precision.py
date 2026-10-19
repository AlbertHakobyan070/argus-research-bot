"""Uncited specifics (v4 spec §2.8, WP-C): the deterministic precision scan.

``scan(md)`` lists the sentences that state a number, an amount, a year or a
date without an ``[n]`` citation. It feeds ``panel_verdict.uncited_specifics``
and the report's "## Uncited specifics" appendix only: the body is never
marked inline, because the scan is a heuristic ("3 regions" is a specific to
it). No LLM call (``panel.precision_mode`` is ``deterministic`` or ``off``).
"""
from __future__ import annotations

import re

from ..quarantine import sentences
from .assemble import CITE_RE

_SPECIFIC = re.compile(r"\d|[$€£¥₹]")


def scan(md: str) -> list[str]:
    """The uncited sentences of ``md`` with a digit or a currency sign, in
    order and without repeats (headings, code, tables and quotes hold none)."""
    out: list[str] = []
    for s in sentences(md):
        if _SPECIFIC.search(s) and not CITE_RE.search(s) and s not in out:
            out.append(s)
    return out


def render_appendix(items: list[str]) -> str:
    """The "## Uncited specifics" appendix ("" without items)."""
    if not items:
        return ""
    return "\n\n".join([
        "## Uncited specifics",
        "_Sentences that state a number, an amount or a date without a citation. A heuristic scan: "
        "the body is not marked, and a figure here may be common knowledge._",
        "\n".join(f"- {' '.join(s.split())}" for s in items)])
