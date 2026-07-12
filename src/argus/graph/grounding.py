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

Pure (no I/O, no LLM). Module-level so it is unit-testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# Default minimum credible-mention count required for a topic to be
# considered "grounded". Below this, the synthesizer gets the warning.
# Lowered 3 -> 2 (2026-07-12 depth rebalance): threshold-3 over-fired on
# normal well-covered topics (only a couple of sources clear the floor
# after fetch failures), triggering the strict-conservatism warning that
# produced shallow, hedged reports. 2 still catches genuinely thin /
# non-existent entities (0-1 credible mentions) — the GLM-5.2 case.
DEFAULT_GROUNDING_THRESHOLD = 2

_GROUNDING_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "into", "this", "that",
    "are", "was", "were", "how", "why", "what", "when", "where",
    "model", "system", "approach", "method", "paper", "work",
    "tldr", "short", "medium", "long", "lecture",
    "research", "report",
})

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _tokens(text: str) -> list[str]:
    """Tokenise text in source order (preserves first-seen position).
    Returns a list, not a set, because extract_entity_tokens needs
    deterministic ordering for "first meaningful word stays first".
    """
    return _TOKEN_RE.findall((text or "").lower())


def extract_entity_tokens(text, stop=None):
    """Pull meaningful signal tokens from a free-form user request.
    Returns a deduped list of tokens in first-seen order.
    """
    stop = stop or _GROUNDING_STOPWORDS
    out: list[str] = []
    seen: set[str] = set()
    # Iterate the regex-match list directly (preserves source order)
    # instead of the helper-returned list/dedup on demand.
    for t in _TOKEN_RE.findall((text or "").lower()):
        if t in stop or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _has_any_token(haystack, tokens):
    hlow = (haystack or "").lower()
    return any(t in hlow for t in tokens)


@dataclass
class GroundingResult:
    """Outcome of check_grounding."""
    grounded: bool
    entity_tokens: list
    mention_count: int
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


def check_grounding(state, threshold=DEFAULT_GROUNDING_THRESHOLD, credibility_floor=0.35):
    """Compute a grounding verdict for the current pipeline state.

    state: dict exposing user_request and fetched (FetchedItem list or model_dump() dicts).
    threshold: minimum credible-mentions required to count as grounded. Default 3.
    credibility_floor: items below this score are NOT counted (noise filter). Default 0.4.
    """
    user_request = state.get("user_request") or ""
    fetched = state.get("fetched") or []
    tokens = extract_entity_tokens(user_request)

    if not tokens:
        return GroundingResult(
            grounded=True, entity_tokens=[], mention_count=0,
            n_credible=0, threshold=threshold,
        )

    n_credible = 0
    mention_count = 0
    samples = []

    for it in fetched:
        if isinstance(it, dict):
            title = it.get("title") or ""
            excerpt = it.get("excerpt") or ""
            cred = it.get("credibility_score") or 0.0
        else:
            title = getattr(it, "title", "") or ""
            excerpt = getattr(it, "excerpt", "") or ""
            cred = getattr(it, "credibility_score", 0.0) or 0.0
        if cred < credibility_floor:
            continue
        n_credible += 1
        if _has_any_token(title + " " + excerpt, tokens):
            mention_count += 1
            if len(samples) < 3:
                samples.append(title)

    grounded = mention_count >= threshold
    return GroundingResult(
        grounded=grounded, entity_tokens=tokens,
        mention_count=mention_count, n_credible=n_credible,
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
        f"Entity tokens looked for: {result.entity_tokens!r}",
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
    "extract_entity_tokens",
    "GroundingResult",
    "grounding_warning_prompt",
    "grounding_banner",
    "DEFAULT_GROUNDING_THRESHOLD",
]
