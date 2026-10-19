"""v4 WP-I — report archetypes are plain data (spec §3 WP-I).

Every archetype the brief can pick has a sectioned variant ((title, focus)
pairs the outline offers inside the length's section budget), a flat variant
(the tldr/short emphases) and its analysis headings; only forecast and
decision add the scenarios section.
"""
from __future__ import annotations

import pytest

from argus.graph import archetypes
from argus.graph.archetypes import TEMPLATES

ARCHETYPES = {"explainer", "definition", "comparison", "forecast", "sota_review", "howto",
              "news_update", "decision"}


def test_the_archetype_set_and_its_default():
    assert set(TEMPLATES) == ARCHETYPES
    assert archetypes.DEFAULT_ARCHETYPE in TEMPLATES


@pytest.mark.parametrize("name", sorted(ARCHETYPES))
def test_every_archetype_has_both_variants_and_non_empty_focus(name):
    t = TEMPLATES[name]
    assert set(t) == {"sectioned", "flat", "analyst_sections"}
    assert 3 <= len(t["sectioned"]) <= 6
    assert all(title.strip() and focus.strip() for title, focus in t["sectioned"])
    assert len({title for title, _ in t["sectioned"]}) == len(t["sectioned"]), "titles are distinct"
    assert set(t["flat"]) == {"body_emphasis", "analyst_emphasis"}
    assert all(text.strip() for text in t["flat"].values())


@pytest.mark.parametrize("name", sorted(ARCHETYPES))
def test_analysis_headings_open_with_takeaways_and_scenarios_only_for_forecast_and_decision(name):
    heads = list(TEMPLATES[name]["analyst_sections"])
    assert heads[0] == archetypes.TAKEAWAYS and heads[-1] == archetypes.OPEN_QUESTIONS
    assert {archetypes.AGREEMENT, archetypes.IMPLICATIONS} <= set(heads)
    assert (archetypes.SCENARIOS in heads) == (name in {"forecast", "decision"})


def test_the_module_is_data_only():
    """No machinery (spec §3 WP-I): nothing in the module is callable."""
    assert [k for k, v in vars(archetypes).items() if not k.startswith("__") and callable(v)] == []
