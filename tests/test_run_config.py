"""v4 WP-0 — RunConfig models, presets, budget helpers (spec §2.4); WP-D —
ensure_in_checkpoint on the real ArgusState, and the Settings knobs of §7."""
import dataclasses
import functools
import operator
from datetime import datetime
from typing import Annotated, TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from argus import config as config_mod
from argus.engines import registry
from argus.engines.schema import ExaPrices
from argus.graph.run_config import (
    PRESETS, ExaConfig, RunConfig, SearchConfig, ensure_in_checkpoint, require)
from argus.graph.state import VALID_LENGTHS, ArgusState
from helpers import fake_engines_snapshot, make_run_config


def _rc(depth=None, length="medium", **overrides) -> RunConfig:
    return RunConfig.from_preset(length=length, depth=depth,
                                 engines=fake_engines_snapshot(),
                                 overrides=overrides, origin="bench")


def _knob(rc: RunConfig, key: str):
    return functools.reduce(getattr, key.split("."), rc)


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
# The bounds written inline in the §2.4 knob classes: (key, floor, ceiling),
# None where the spec gives no inline bound (every numeric knob is >= 0).
BOUNDS = [
    ("search.exa.max_searches", None, 60),
    ("search.exa.max_contents_pages", None, 200),
    ("search.exa.num_results", 1, 20),
    ("search.arxiv.max_queries_per_run", None, 6),
    ("fetch.concurrency", 1, 8),
    ("research.waves", 1, 6),
    ("research.fetch_per_wave", 1, 30),
    ("research.total_fetch_cap", None, 150),
    ("research.gap_fetch_per_wave", None, 20),
    ("research.per_subq_target", 1, 8),
    ("research.coverage_min_strong", 0, 5),
    ("research.gap_waves_max", None, 3),
    ("research.digest_chunks_long_doc", 1, 6),
    ("research.digest_concurrency", 1, 8),
    ("research.writer_min_relevance", 0, 5),
    ("compose.concurrency", 1, 6),
    ("compose.section_min_notes", 1, 5),
    ("panel.judge_batch_size", 4, 20),
    ("panel.max_compose_revisions", None, 3),
    ("panel.rewrite_flag_ratio", None, 1),
]
_PRICES = ExaPrices(price_per_search_usd=0.007, price_per_page_usd=0.001,
                    prices_as_of="2026-10-18")


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


def test_exa_cost_estimate_uses_the_given_prices():
    # searches · search price + /contents pages · page price (inline search text is not billed)
    exa = _rc("standard").search.exa
    expected = exa.max_searches * _PRICES.price_per_search_usd + exa.max_contents_pages * _PRICES.price_per_page_usd
    assert _rc("standard").exa_cost_estimate_usd(_PRICES) == pytest.approx(expected)
    assert _rc("standard").exa_cost_estimate_usd(_PRICES) == pytest.approx(0.152)   # 16 · 0.007 + 40 · 0.001
    doubled = ExaPrices(price_per_search_usd=0.014, price_per_page_usd=0.002,
                        prices_as_of="2026-10-18")
    assert _rc("standard").exa_cost_estimate_usd(doubled) == pytest.approx(2 * expected)


def test_exa_cost_estimate_without_prices_is_none():
    assert _rc("standard").exa_cost_estimate_usd(None) is None   # shown as "n/a"


def test_fix_round_knob_defaults_and_removed_fields():
    rc = _rc("standard")
    assert rc.fetch.max_bytes == 25_000_000
    assert rc.panel.judge_batch_size == 10
    assert rc.overrides_applied == {}
    assert not ({"price_per_search_usd", "price_per_page_usd", "prices_as_of"}
                & set(ExaConfig.model_fields))
    assert "source_mix_min" not in SearchConfig.model_fields
    assert "archetype_override" not in RunConfig.model_fields
    with pytest.raises(ValueError, match="precision_mode"):
        _rc("standard", **{"panel.precision_mode": "llm"})


def test_default_depth():
    expected = {"tldr": "quick", "short": "quick", "medium": "standard",
                "long": "deep", "lecture": "deep"}
    assert {n: RunConfig.default_depth(n) for n in VALID_LENGTHS} == expected
    with pytest.raises(ValueError, match="unknown length"):
        RunConfig.default_depth("epic")


def test_from_preset_defaults_depth_from_length():
    rc = RunConfig.from_preset(length="long", engines=fake_engines_snapshot(),
                               overrides=None, origin="telegram")
    assert (rc.length, rc.depth, rc.preset_origin) == ("long", "deep", "deep")
    assert rc.origin == "telegram"
    assert rc.overrides_applied == {}
    datetime.fromisoformat(rc.created_at)
    with pytest.raises(ValueError, match="unknown depth"):
        _rc("ultra")


def test_quick_preset_turns_analyst_off_only_for_tldr():
    assert _rc(length="tldr").compose.analyst is False
    assert _rc(length="short").compose.analyst is True
    assert _rc(length="tldr", **{"compose.analyst": True}).compose.analyst is True


def test_from_preset_overrides_by_dotted_key():
    overrides = {"research.waves": 2, "search.exa.num_results": 5,
                 "panel.coverage": False, "fetch.concurrency": 2, "recency": "m"}
    rc = _rc("standard", **overrides)
    assert (rc.research.waves, rc.search.exa.num_results, rc.panel.coverage,
            rc.fetch.concurrency, rc.recency) == (2, 5, False, 2, "m")
    assert rc.research.fetch_per_wave == 14      # untouched preset value
    assert rc.preset_origin == "standard"
    assert rc.overrides_applied == overrides


@pytest.mark.parametrize("key", ["research.wavez", "research", "depth", "length",
                                 "legacy", "engines.profile", "search.exa.nope",
                                 "overrides_applied", "archetype_override",
                                 "search.source_mix_min",
                                 "search.exa.price_per_search_usd"])
def test_unknown_override_key_rejected(key):
    with pytest.raises(ValueError, match="unknown run_config override"):
        RunConfig.from_preset(length="medium", depth="standard",
                              engines=fake_engines_snapshot(),
                              overrides={key: 1}, origin="bench")


@pytest.mark.parametrize("key,value,match", [
    ("research.waves", "many", "waves"),                    # type
    ("research.digest_head_chars", -1, "research.digest_head_chars must be >= 0"),
    ("search.arxiv.min_interval_s", -0.5, "min_interval_s must be >= 0"),
    # standard reserves 2 gap waves × 8 fetches and 2 × 2 Exa searches.
    ("research.total_fetch_cap", 10, "total_fetch_cap"),
    ("search.exa.max_searches", 3, "max_searches"),
])
def test_override_bounds(key, value, match):
    with pytest.raises(ValueError, match=match):
        _rc("standard", **{key: value})


@pytest.mark.parametrize("key,floor,ceiling", BOUNDS)
def test_inline_floors_and_ceilings(key, floor, ceiling):
    step = 0.01 if isinstance(_knob(_rc("standard"), key), float) else 1     # a float bound: just past it
    if floor is not None:
        assert _knob(_rc("standard", **{key: floor}), key) == floor
        with pytest.raises(ValueError, match=f"{key} must be >= {floor}"):
            _rc("standard", **{key: floor - step})
    if ceiling is not None:
        assert _knob(_rc("standard", **{key: ceiling}), key) == ceiling
        with pytest.raises(ValueError, match=f"{key} must be <= {ceiling}"):
            _rc("standard", **{key: ceiling + step})


def test_for_legacy_checkpoint_follows_the_checkpoint_length():
    rc = RunConfig.for_legacy_checkpoint(engines=fake_engines_snapshot(),
                                         length="long")
    assert (rc.length, rc.depth, rc.preset_origin) == ("long", "deep", "deep")
    assert (rc.legacy, rc.origin) == (True, "legacy")
    assert rc.research == _rc("deep", length="long").research
    short = RunConfig.for_legacy_checkpoint(engines=fake_engines_snapshot(),
                                            length="short")
    assert (short.depth, short.legacy) == ("quick", True)
    # The field keeps its name and default; no constructor is attached under it.
    assert _rc().legacy is False
    assert RunConfig.model_fields["legacy"].default is False
    assert not callable(getattr(RunConfig, "legacy", None))


def test_rebuild_reapplies_overrides_applied():
    overrides = {"research.waves": 2, "panel.coverage": False, "recency": "m"}
    rc = _rc("standard", **overrides)
    other = fake_engines_snapshot().model_copy(update={"profile": "other"})
    new = rc.rebuild(length="long", depth="exhaustive", engines=other)
    assert (new.length, new.depth, new.preset_origin) == (
        "long", "exhaustive", "exhaustive")
    assert (new.research.waves, new.panel.coverage, new.recency) == (2, False, "m")
    assert new.research.fetch_per_wave == 20     # exhaustive's preset, not standard's
    assert new.overrides_applied == overrides
    assert (new.engines.profile, new.origin, new.legacy) == ("other", "bench", False)
    # depth=None → default_depth(length); the overrides still apply.
    quick = rc.rebuild(length="short", engines=fake_engines_snapshot())
    assert (quick.depth, quick.research.waves, quick.recency) == ("quick", 2, "m")


def test_rebuild_keeps_the_legacy_mark():
    rc = RunConfig.for_legacy_checkpoint(engines=fake_engines_snapshot(),
                                         length="medium")
    new = rc.rebuild(length="long", engines=fake_engines_snapshot())
    assert (new.length, new.depth, new.legacy, new.origin) == (
        "long", "deep", True, "legacy")


def test_require_missing_or_invalid_raises_naming_run_config():
    with pytest.raises(ValueError, match="run_config"):
        require({})
    with pytest.raises(ValueError, match="run_config"):
        require({"run_config": None})
    with pytest.raises(ValueError, match="run_config"):
        require({"run_config": {"length": "short"}})


def test_require_rejects_a_config_whose_length_differs_from_the_state():
    raw = make_run_config("short")
    assert require({"run_config": raw, "length": "short"}).length == "short"
    assert require({"run_config": raw, "length": None}).length == "short"  # length not set
    with pytest.raises(ValueError, match=r"run_config.*'short'.*'medium'"):
        require({"run_config": raw, "length": "medium"})


def test_a_stored_run_config_with_the_removed_quote_knob_still_loads():
    """WP-B fix round 3: ``research.quote_verify_fuzzy`` is gone (verbatim quotes).
    A stored run_config that still carries it loads, the key ignored; an override
    naming it is an unknown key."""
    raw = make_run_config("medium")
    raw["research"]["quote_verify_fuzzy"] = 0.9
    rc = require({"run_config": raw})
    assert "quote_verify_fuzzy" not in type(rc.research).model_fields
    assert not hasattr(rc.research, "quote_verify_fuzzy")
    with pytest.raises(ValueError, match="quote_verify_fuzzy"):
        _rc("standard", **{"research.quote_verify_fuzzy": 0.8})


def test_require_round_trips_make_run_config():
    rc = require({"run_config": make_run_config()})
    assert (rc.length, rc.depth, rc.engines.profile) == ("short", "quick", "fake")
    rc = require({"run_config": make_run_config("medium", **{"research.waves": 1})})
    assert (rc.depth, rc.research.waves) == ("standard", 1)


def test_require_loads_a_checkpointed_config_naming_the_retired_legacy_roles():
    """WP-H removed the Phase 1–2 ``legacy.<tier>`` roles; a run_config
    checkpointed before still names them and must resume."""
    data = make_run_config()
    for tier in ("cheap", "strong", "judge"):
        role = f"legacy.{tier}"
        data["engines"]["roles"][role] = {
            "role": role, "tier": tier, "primary": {"provider": "fake", "model": "fake-model"},
            "fallbacks": [], "on_exhausted": "raise", "retry_on_truncation": True}
    rc = require({"run_config": data, "length": "short"})
    assert set(rc.engines.roles) == set(fake_engines_snapshot().roles)
    assert not any(r.startswith("legacy.") for r in rc.engines.roles)


class _S(TypedDict, total=False):
    x: int
    length: str
    run_config: dict
    errors: Annotated[list[str], operator.add]


def _tiny_graph():
    g = StateGraph(_S)
    g.add_node("step", lambda s: {"x": 1})
    g.add_edge(START, "step")
    g.add_edge("step", END)
    return g.compile(checkpointer=MemorySaver())


async def _checkpoint_id(graph, cfg):
    return (await graph.aget_state(cfg)).config["configurable"].get("checkpoint_id")


async def test_ensure_in_checkpoint_injects_legacy_once():
    graph = _tiny_graph()
    cfg = {"configurable": {"thread_id": "pre-v4"}}
    await graph.ainvoke({"x": 0, "length": "long"}, cfg)   # no run_config

    rc = await ensure_in_checkpoint(graph, cfg, engines=fake_engines_snapshot())
    values = (await graph.aget_state(cfg)).values
    assert (rc.legacy, rc.origin, rc.length, rc.depth) == (True, "legacy", "long", "deep")
    assert require(values) == rc
    assert values["errors"] == [
        "legacy: run_config defaults applied (pre-v4 checkpoint)"]

    written = await _checkpoint_id(graph, cfg)
    again = await ensure_in_checkpoint(graph, cfg, engines=fake_engines_snapshot())
    assert again == rc
    assert await _checkpoint_id(graph, cfg) == written          # nothing written
    assert (await graph.aget_state(cfg)).values["errors"] == values["errors"]


async def test_ensure_in_checkpoint_refuses_a_thread_that_never_ran():
    graph = _tiny_graph()
    cfg = {"configurable": {"thread_id": "never-ran"}}
    with pytest.raises(ValueError, match="no checkpoint for thread 'never-ran'"):
        await ensure_in_checkpoint(graph, cfg, engines=fake_engines_snapshot())
    snap = await graph.aget_state(cfg)
    assert (snap.created_at, snap.values) == (None, {})         # nothing written


async def test_ensure_in_checkpoint_refuses_a_checkpoint_without_length():
    graph = _tiny_graph()
    cfg = {"configurable": {"thread_id": "no-length"}}
    await graph.ainvoke({"x": 0}, cfg)
    before = await _checkpoint_id(graph, cfg)
    with pytest.raises(ValueError, match="thread 'no-length'.*length"):
        await ensure_in_checkpoint(graph, cfg, engines=fake_engines_snapshot())
    assert await _checkpoint_id(graph, cfg) == before           # nothing written
    assert "run_config" not in (await graph.aget_state(cfg)).values


async def test_ensure_in_checkpoint_on_argus_state_keeps_the_injected_config():
    """aupdate_state silently drops a key ArgusState does not declare: the
    injection must survive in a checkpoint of the real state schema."""
    g = StateGraph(ArgusState)
    g.add_node("step", lambda s: {"draft_md": ""})
    g.add_edge(START, "step")
    g.add_edge("step", END)
    graph = g.compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "argus-pre-v4"}}
    await graph.ainvoke({"user_request": "t", "length": "long"}, cfg)

    rc = await ensure_in_checkpoint(graph, cfg, engines=fake_engines_snapshot())
    values = (await graph.aget_state(cfg)).values
    assert values["run_config"] == rc.model_dump()
    assert (rc.legacy, rc.length, rc.depth) == (True, "long", "deep")
    assert values["errors"] == ["legacy: run_config defaults applied (pre-v4 checkpoint)"]
    assert await ensure_in_checkpoint(graph, cfg, engines=fake_engines_snapshot()) == rc
    assert (await graph.aget_state(cfg)).values["errors"] == values["errors"]


# ---------------------------------------------------------------------------
# Settings knobs added by WP-D (spec §7): read by the run driver (WP-E1) and
# the console (WP-E2)
# ---------------------------------------------------------------------------

_KNOB_ENV = ("ARGUS_WEBAPP_ENABLED", "ARGUS_WEBAPP_PORT", "ARGUS_ENGINES_FILE",
             "ARGUS_ENGINES_LOCAL_FILE", "ARGUS_CANCEL_DRAIN_TIMEOUT_S", "ARGUS_SHUTDOWN_DRAIN_S")


@pytest.fixture
def _bare_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:test")
    for name in _KNOB_ENV:
        monkeypatch.delenv(name, raising=False)


def test_settings_v4_knob_defaults(_bare_env):
    s = config_mod.Settings.load()
    root = config_mod._PROJECT_ROOT
    assert (s.webapp_enabled, s.webapp_port) == (True, 8787)
    # the engines files have one reader, the registry (WP-H removed the dead Settings copy)
    assert registry.default_paths() == [root / "argus_engines.json", root / "argus_engines.local.json"]
    assert (s.cancel_drain_timeout_s, s.shutdown_drain_s) == (300, 120)
    fields = {f.name for f in dataclasses.fields(config_mod.Settings)}
    assert not fields & {"max_revision_rounds", "engines_file", "engines_local_file"}


def test_settings_v4_knobs_from_env(_bare_env, monkeypatch, tmp_path):
    monkeypatch.setenv("ARGUS_WEBAPP_ENABLED", "false")
    monkeypatch.setenv("ARGUS_WEBAPP_PORT", "9999")
    monkeypatch.setenv("ARGUS_ENGINES_FILE", "conf/engines.json")        # relative → repo root
    monkeypatch.setenv("ARGUS_ENGINES_LOCAL_FILE", str(tmp_path / "local.json"))
    monkeypatch.setenv("ARGUS_CANCEL_DRAIN_TIMEOUT_S", "30")
    monkeypatch.setenv("ARGUS_SHUTDOWN_DRAIN_S", "7.5")
    s = config_mod.Settings.load()
    assert (s.webapp_enabled, s.webapp_port) == (False, 9999)
    assert registry.default_paths() == [config_mod._PROJECT_ROOT / "conf" / "engines.json",
                                        tmp_path / "local.json"]
    assert (s.cancel_drain_timeout_s, s.shutdown_drain_s) == (30, 7.5)


@pytest.mark.parametrize("value", ["1", "yes", "ON", "True"])
def test_settings_webapp_enabled_accepts_true_spellings(_bare_env, monkeypatch, value):
    monkeypatch.setenv("ARGUS_WEBAPP_ENABLED", value)
    assert config_mod.Settings.load().webapp_enabled is True


def test_settings_webapp_enabled_rejects_garbage(_bare_env, monkeypatch):
    monkeypatch.setenv("ARGUS_WEBAPP_ENABLED", "maybe")
    with pytest.raises(ValueError, match="ARGUS_WEBAPP_ENABLED"):
        config_mod.Settings.load()
