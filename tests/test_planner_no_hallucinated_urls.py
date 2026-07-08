"""Regression: planner must not invent URLs.

Bug observed 2026-07-08: when the strong tier fell back to llama-3.1-8b-instant,
the planner emitted plausible-looking but fake URLs (e.g.
`github.com/transformers-metacognition`, `2207.12345`).
Fix: PLANNER_SYSTEM prompt now bans invented URLs and forces kind=search_result
when no exact URL is known. This file proves the prompt enforces the rule by
regex-parsing it (no LLM in the test — that's `test_planner_node_integration`).
"""
import re
import pytest

from argus.graph.nodes import PLANNER_SYSTEM


# --- substring checks --------------------------------------------------------

REQUIRED_RULES = [
    "Do NOT invent URLs",
    "Do NOT guess arXiv IDs",
    "kind: \"search_result\"",
    "Verify URLs by reasoning",
    "arxiv.org",
    "github.com",
]


@pytest.mark.parametrize("rule", REQUIRED_RULES)
def test_planner_prompt_bans_invented_urls(rule: str) -> None:
    """The planner system prompt must explicitly forbid URL invention."""
    assert rule in PLANNER_SYSTEM, (
        f"PLANNER_SYSTEM prompt is missing required URL-integrity rule: {rule!r}. "
        f"Without it, the planner will fabricate plausible-looking links that "
        f"the researcher cannot resolve. Add an explicit ban to PLANNER_SYSTEM "
        f"in src/argus/graph/nodes.py."
    )


# --- structural checks -------------------------------------------------------

URL_PATTERN = re.compile(r"https?://[^\s\"'`>]+|github\.com/[A-Za-z0-9_.-]+|arxiv\.org/\S+")


def test_planner_prompt_keeps_legitimate_url_examples() -> None:
    """The prompt should keep EXAMPLES of legitimate primary-source URLs so the
    planner knows what 'real' looks like. We assert at least 3 example URLs."""
    urls = URL_PATTERN.findall(PLANNER_SYSTEM)
    assert len(urls) >= 3, (
        f"PLANNER_SYSTEM should show >=3 example primary-source URLs, found {len(urls)}: {urls}"
    )


def test_planner_prompt_mentions_search_result_fallback() -> None:
    """When the planner doesn't know the exact URL, the prompt must direct it
    to emit kind='search_result' with a query (no target_url)."""
    lowered = PLANNER_SYSTEM.lower()
    # Both 'search_result' and a fallback signal must appear (close enough
    # in the prompt — we tolerate some intervening text).
    assert "search_result" in lowered, (
        "PLANNER_SYSTEM must mention 'search_result' as the safe fallback."
    )
    assert any(signal in lowered for signal in (
        "fall back to", "fallback to", "use search_result", "use kind: \"search_result\"",
        "fall back to `search_result`",
    )), (
        "PLANNER_SYSTEM must direct the planner to fall back to kind='search_result' "
        "(with a query, no target_url) when it doesn't know an exact URL."
    )


def test_planner_prompt_no_longer_advertises_target_url_as_required() -> None:
    """The OLD prompt said `target_url: 'https://... (only if you have a
    specific URL)'` — which models interpret as 'fill in a URL.' The new
    prompt must make target_url truly optional and steer toward search_result."""
    # Old phrasing would say something like "target_url" without the
    # "use search_result instead" guidance. Check the prompt now steers
    # away from speculative URL filling.
    assert "target_url" not in PLANNER_SYSTEM or re.search(
        r"target_url.*only.*when.*you.*know|target_url.*never.*guess",
        PLANNER_SYSTEM, re.I | re.S,
    ), (
        "PLANNER_SYSTEM still presents target_url as optional-with-default. "
        "Replace with explicit 'use kind='search_result' if you don't know "
        "the exact URL' guidance."
    )