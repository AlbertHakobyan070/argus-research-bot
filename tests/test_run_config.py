"""v4 WP-0 — RunConfig models, presets, budget helpers (spec §2.4)."""
import operator
from datetime import datetime
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from argus.graph.run_config import (
    PRESETS, RunConfig, ensure_in_checkpoint, require)
from argus.graph.state import VALID_LENGTHS
from conftest import fake_engines_snapshot, make_run_config


def _rc(depth=None, length="medium", **overrides) -> RunConfig:
    return RunConfig.from_preset(length=length, depth=depth,
                                 engines=fake_engines_snapshot(),
                                 overrides=overrides, origin="bench")


# The §2.4 table, column by column.
TABLE = {
    #            waves fpw  cap  gap/w gapmax subq strong chunks minrel
    "quick":      (1, 10,  12,  0,  0, 2, 1, 1, 2),
    "standard":   (3, 14,  48,  8,  2, 3, 2, 2, 3),
    "deep":       (4, 16,  78, 10,  3, 4, 3, 3, 3),
    "exhaustive": (5, 20, 108, 12,  3, 5, 3, 4, 3),
}
EXA = {"quick": (6, 12), "standard": (16, 40), "deep": (26, 64),
       "exhaustive": (36, 96)}
PANEL = {"quick": (True, False, 0), "standard": (True, True, 1),
         "deep": (True, True, 1), "exhaustive": (True, True, 2)}
# Parenthesised split in the table: (first pass, reserve) for fetches and
# Exa searches.
SPLIT = {"quick": ((12, 0), (6, 0)), "standard": ((32, 16), (12, 4)),
         "deep": ((48, 30), (20, 6)), "exhaustive": ((72, 36), (30, 6))}


def test_presets_cover_every_depth():
    assert set(PRESETS) == set(TABLE)


@pytest.mark.parametrize("depth", list(TABLE))
def test_preset_values_match_spec_table(depth):
    rc = _rc(depth)
    r = rc.research
    assert (r.waves, r.fetch_per_wave, r.total_fetch_cap, r.gap_fetch_per_wave,
            r.gap_waves_max, r.per_subq_target, r.coverage_min_strong,
            r.digest_chunks_long_doc, r.writer_min_relevance) == TABLE[depth]
    assert (rc.search.exa.max_searches,
            rc.search.exa.max_contents_pages) == EXA[depth]
    assert (rc.panel.grounding, rc.panel.coverage,
            rc.panel.max_compose_revisions) == PANEL[depth]
    assert rc.panel.precision_mode == "deterministic"
    assert rc.compose.analyst is True
    assert (rc.depth, rc.preset_origin, rc.legacy) == (depth, depth, False)


@pytest.mark.parametrize("depth", list(TABLE))
def test_budget_helpers_first_pass_is_total_minus_reserve(depth):
    rc = _rc(depth)
    (first, reserve), (exa_first, exa_reserve) = SPLIT[depth]
    assert rc.gap_reserve() == reserve
    assert rc.first_pass_fetch_cap() == rc.research.total_fetch_cap - rc.gap_reserve()
    assert rc.first_pass_fetch_cap() == first
    assert rc.exa_search_reserve() == exa_reserve
    assert rc.search.exa.max_searches - rc.exa_search_reserve() == exa_first


def test_exa_cost_estimate_uses_configured_prices():
    # standard: 16 searches · $0.007 + (16 · 8 results + 40 pages) · $0.001
    assert _rc("standard").exa_cost_estimate_usd() == pytest.approx(0.28)
    doubled = _rc("standard", **{"search.exa.price_per_search_usd": 0.014,
                                 "search.exa.price_per_page_usd": 0.002})
    assert doubled.exa_cost_estimate_usd() == pytest.approx(0.56)


def test_default_depth():
    expected = {"tldr": "quick", "short": "quick", "medium": "standard",
                "long": "deep", "lecture": "deep"}
    assert {n: RunConfig.default_depth(n) for n in VALID_LENGTHS} == expected
    with pytest.raises(ValueError, match="unknown length"):
        RunConfig.default_depth("epic")


def test_from_preset_defaults_depth_from_length():
    rc = RunConfig.from_preset(length="long", engines=fake_engines_snapshot(),
                               origin="telegram")
    assert (rc.length, rc.depth, rc.preset_origin) == ("long", "deep", "deep")
    assert rc.origin == "telegram"
    datetime.fromisoformat(rc.created_at)
    with pytest.raises(ValueError, match="unknown depth"):
        _rc("ultra")


def test_quick_preset_turns_analyst_off_only_for_tldr():
    assert _rc(length="tldr").compose.analyst is False
    assert _rc(length="short").compose.analyst is True
    assert _rc(length="tldr", **{"compose.analyst": True}).compose.analyst is True


def test_from_preset_overrides_by_dotted_key():
    rc = _rc("standard", **{"research.waves": 2, "search.exa.num_results": 5,
                            "panel.coverage": False, "fetch.concurrency": 2,
                            "search.source_mix_min": {"paper": 2},
                            "recency": "m"})
    assert (rc.research.waves, rc.search.exa.num_results, rc.panel.coverage,
            rc.fetch.concurrency, rc.recency) == (2, 5, False, 2, "m")
    assert rc.search.source_mix_min == {"paper": 2}
    assert rc.research.fetch_per_wave == 14      # untouched preset value
    assert rc.preset_origin == "standard"


@pytest.mark.parametrize("key", ["research.wavez", "research", "depth", "length",
                                 "legacy", "engines.profile", "search.exa.nope"])
def test_unknown_override_key_rejected(key):
    with pytest.raises(ValueError, match="unknown run_config override"):
        RunConfig.from_preset(length="medium", depth="standard",
                              engines=fake_engines_snapshot(),
                              overrides={key: 1}, origin="bench")


@pytest.mark.parametrize("key,value,match", [
    ("research.waves", "many", "waves"),                    # type
    ("research.waves", -1, "research.waves must be >= 0"),
    ("search.source_mix_min", {"repo": -2}, "source_mix_min.repo must be >= 0"),
    ("compose.concurrency", 0, "compose.concurrency must be >= 1"),
    ("research.digest_concurrency", 0, "digest_concurrency must be >= 1"),
    ("panel.rewrite_flag_ratio", 1.5, "rewrite_flag_ratio must be <= 1"),
    ("research.quote_verify_fuzzy", 1.1, "quote_verify_fuzzy must be <= 1"),
    # standard reserves 2 gap waves × 8 fetches and 2 × 2 Exa searches.
    ("research.total_fetch_cap", 10, "total_fetch_cap"),
    ("search.exa.max_searches", 3, "max_searches"),
])
def test_override_bounds(key, value, match):
    with pytest.raises(ValueError, match=match):
        _rc("standard", **{key: value})


def test_legacy_is_standard_preset_marked_legacy():
    rc = RunConfig.legacy(engines=fake_engines_snapshot())
    assert (rc.legacy, rc.origin, rc.depth, rc.preset_origin) == (
        True, "legacy", "standard", "standard")
    assert rc.length == "medium"     # the length whose default depth is standard
    assert rc.research == _rc("standard").research
    # Field and constructor share the name: instances read the field.
    assert _rc().legacy is False
    assert RunConfig.model_fields["legacy"].default is False


def test_require_missing_or_invalid_raises_naming_run_config():
    with pytest.raises(ValueError, match="run_config"):
        require({})
    with pytest.raises(ValueError, match="run_config"):
        require({"run_config": None})
    with pytest.raises(ValueError, match="run_config"):
        require({"run_config": {"length": "short"}})


def test_require_round_trips_make_run_config():
    rc = require({"run_config": make_run_config()})
    assert (rc.length, rc.depth, rc.engines.profile) == ("short", "quick", "fake")
    rc = require({"run_config": make_run_config("medium", **{"research.waves": 1})})
    assert (rc.depth, rc.research.waves) == ("standard", 1)


class _S(TypedDict, total=False):
    x: int
    run_config: dict
    errors: Annotated[list[str], operator.add]


def _tiny_graph():
    g = StateGraph(_S)
    g.add_node("step", lambda s: {"x": 1})
    g.add_edge(START, "step")
    g.add_edge("step", END)
    return g.compile(checkpointer=MemorySaver())


async def test_ensure_in_checkpoint_injects_legacy_once():
    graph = _tiny_graph()
    cfg = {"configurable": {"thread_id": "pre-v4"}}
    await graph.ainvoke({"x": 0}, cfg)               # checkpoint without run_config

    rc = await ensure_in_checkpoint(graph, cfg, engines=fake_engines_snapshot())
    values = (await graph.aget_state(cfg)).values
    assert rc.legacy is True
    assert require(values) == rc
    assert values["errors"] == [
        "legacy: run_config defaults applied (pre-v4 checkpoint)"]

    again = await ensure_in_checkpoint(graph, cfg, engines=fake_engines_snapshot())
    assert again == rc
    assert (await graph.aget_state(cfg)).values["errors"] == values["errors"]
