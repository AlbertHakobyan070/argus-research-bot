"""P5 - grounding check: does the evidence actually support the queried entity?

Why this exists (bug observed 2026-07-09, ``/research GLM 5.2``):
The synthesizer will always write a report. ``verify_citations``
checks URLs were *fetched*, never that the fetched content
*actually mentions or supports claims about the queried entity*.
For rare or non-existent entities (a bleeding-edge / possibly-
nonexistent model name) the top web hits are content farms and
adjacent-topic pages. The LLM bridges the gap by inventing.

This module answers one question: does a threshold number of credible
fetched sources actually mention the entity the user asked about?

If NO, the synthesizer gets a strict grounding warning injected into
its prompt, and the report carries a top-of-page banner so the reader
is not deceived.

v4 (spec §3 WP-B): the entities are the brief's ``entities`` — full
phrases, never single query tokens (baseline b5, a nonexistent entity,
passed v3's gate because any one of six generic request tokens counted;
only 1 of 28 documents named it). A phrase is matched normalised and
case-insensitive (a possessive 's dropped) in the fetched document's full
text, or spelled without its spaces over a run of whole tokens ("CrewAI" ↔
"Crew AI"); a source counts as credible when its digest note is relevant
(relevance ≥ 3) and not of authority 0. No LLM; the only I/O is reading the
fetched markdown files.
Until the brief writes ``entities`` (WP-I) there is nothing to check and
every run is grounded.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


# Default minimum credible-mention count required for a topic to be
# considered "grounded". Below this, the synthesizer gets the warning.
# Lowered 3 -> 2 (2026-07-12 depth rebalance): threshold-3 over-fired on
# normal well-covered topics (only a couple of sources clear the floor
# after fetch failures), triggering the strict-conservatism warning that
# produced shallow, hedged reports. 2 still catches genuinely thin /
# non-existent entities (0-1 credible mentions) — the GLM-5.2 case.
DEFAULT_GROUNDING_THRESHOLD = 2
_MIN_RELEVANCE = 3

# Everything that is not a letter or digit becomes a space — except a dot
# between two digits, so "5.2" stays one token ("GLM 5" never matches "GLM 5.2").
_SEPARATORS = re.compile(r"(?!(?<=\d)\.(?=\d))[\W_]")
_POSSESSIVE = re.compile(r"(?<=\w)['’]s\b")          # "Bitcoin's price" → "bitcoin price"


def normalise(text: str) -> str:
    return " ".join(_SEPARATORS.sub(" ", _POSSESSIVE.sub("", (text or "").lower())).split())


def _phrase_pattern(phrase: str) -> re.Pattern:
    """A normalised phrase as a run of whole tokens of a space-padded
    normalised text, its spaces optional: "crew ai" matches "crew ai" and
    "crewai", "crewai" matches both too; never part of a token."""
    return re.compile("(?<= )" + " ?".join(map(re.escape, phrase.replace(" ", ""))) + "(?= )")


# compose checks grounding once per section: each document is read and
# normalised once. ponytail: keyed by (path, mtime_ns, size) — a same-size
# rewrite within one mtime tick would be missed; fetched files are named by
# URL hash and written once per fetch.
@lru_cache(maxsize=256)
def _document(path: str, _mtime_ns: int, _size: int) -> str:
    return " " + normalise(Path(path).read_text(encoding="utf-8", errors="replace")) + " "


def _text_of(item: dict) -> str:
    """The fetched document's full text, normalised and space-padded; the
    title and excerpt when it has no file on disk."""
    path = item.get("markdown_path")
    if path and os.path.isfile(path):
        st = os.stat(path)
        return _document(path, st.st_mtime_ns, st.st_size)
    return " " + normalise(f"{item.get('title') or ''} {item.get('excerpt') or ''}") + " "


@dataclass
class GroundingResult:
    """Outcome of check_grounding."""
    grounded: bool
    entity_tokens: list          # the normalised entity phrases looked for
    mention_count: int           # credible sources naming the least-supported entity
    n_credible: int
    threshold: int
    sample_evidence_titles: list = field(default_factory=list)

    def __repr__(self):
        return (
            f"GroundingResult(grounded={self.grounded!r}, "
            f"mentions={self.mention_count}/{self.n_credible}, "
            f"threshold={self.threshold}, "
            f"tokens={self.entity_tokens!r})"
        )


def check_grounding(state, threshold=DEFAULT_GROUNDING_THRESHOLD):
    """Grounding verdict for the current pipeline state (``brief``,
    ``fetched``, ``evidence``).

    Credible sources are the fetched items whose evidence note has
    relevance ≥ 3 and authority ≥ 1 (notes without an ``authority`` count
    as 1). Each entity phrase must occur in at least ``threshold`` of them;
    ``mention_count`` is the count of the least-supported entity. No
    entities → nothing to check → grounded.
    """
    entities = [p for p in (normalise(e) for e in ((state.get("brief") or {}).get("entities") or [])) if p]
    entities = list(dict.fromkeys(entities))
    if not entities:
        return GroundingResult(grounded=True, entity_tokens=[], mention_count=0,
                               n_credible=0, threshold=threshold)
    by_url = {f.get("url"): f for f in state.get("fetched") or []}
    credible = []
    for note in state.get("evidence") or []:
        item = by_url.get(note.get("source_url"))
        if (item is not None and int(note.get("relevance") or 0) >= _MIN_RELEVANCE
                and int(note.get("authority", 1)) >= 1):
            credible.append(item)
    patterns = {e: _phrase_pattern(e) for e in entities}       # compiled once, reused per document
    counts = {e: 0 for e in entities}
    samples: list[str] = []
    for item in credible:
        text = _text_of(item)
        named = [e for e, pattern in patterns.items() if pattern.search(text)]
        for e in named:
            counts[e] += 1
        if named and len(samples) < 3:
            samples.append(item.get("title") or item.get("url") or "")
    mention_count = min(counts.values())
    return GroundingResult(
        grounded=mention_count >= threshold, entity_tokens=entities,
        mention_count=mention_count, n_credible=len(credible),
        threshold=threshold, sample_evidence_titles=samples,
    )


def grounding_warning_prompt(result, user_request):
    """Return a strict-conservatism prompt-inject for the synthesizer
    when NOT grounded. Empty when grounded.
    """
    if result.grounded:
        return ""
    lines = [
        "",
        "",
        "=== GROUNDING WARNING ===",
        f"The user asked about: {user_request!r}",
        f"Entities looked for: {result.entity_tokens!r}",
        (
            f"Only {result.mention_count} of {result.n_credible} credible "
            "fetched sources mention the entity directly (threshold: "
            f"{result.threshold}). The web is thin on this topic."
        ),
        "",
        "You MUST be strictly conservative:",
        "- State 'insufficient direct evidence' rather than bridging gaps.",
        "- Do NOT invent numbers, model sizes, dates, or paper authors.",
        "- If you must mention specifics, mark them as speculative.",
        "- Cross-reference at least 2 independent credible sources before stating a number as fact.",
        "===========================",
        "",
    ]
    return chr(10).join(lines)


def grounding_banner(result):
    """Return a one-line banner to prepend to the synthesized report
    when not grounded. Empty when grounded.
    """
    if result.grounded:
        return ""
    return (
        "> WARNING: **Limited direct evidence:** only" + chr(10)
        + f"> {result.mention_count} of {result.n_credible} credible sources" + chr(10)
        + "> mention the queried entity. Treat factual claims as" + chr(10)
        + "> conservative hypotheses - the web is thin on this topic." + chr(10) + chr(10)
    )


__all__ = [
    "check_grounding",
    "normalise",
    "GroundingResult",
    "grounding_warning_prompt",
    "grounding_banner",
    "DEFAULT_GROUNDING_THRESHOLD",
]
