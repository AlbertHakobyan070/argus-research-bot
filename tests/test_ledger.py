"""engines/ledger.SpendLedger (v4 spec §2.4, §2.8): one lock-guarded ledger
for every research budget — atomic check-and-count under a thread pool, the
first-pass / gap / Extend split from the RunConfig, and the snapshot that
round-trips through ``state["search_spend"]``."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from argus.engines.ledger import SpendLedger
from argus.graph.run_config import RunConfig

from helpers import make_run_config


def _rc(length="medium", **overrides) -> RunConfig:
    return RunConfig.model_validate(make_run_config(length, **overrides))


def test_take_never_overshoots_under_a_4_thread_pool():
    ledger = SpendLedger({"fetch_first": 137})
    with ThreadPoolExecutor(max_workers=4) as ex:
        granted = list(ex.map(lambda _: ledger.take("fetch_first"), range(1000)))
    assert sum(granted) == 137, "exactly the cap is granted, never more"
    assert ledger.snapshot()["fetch_first"] == 137
    assert ledger.remaining("fetch_first") == 0


def test_counters_and_usd_lose_no_updates_under_a_4_thread_pool():
    ledger = SpendLedger({})

    def work(_):
        ledger.count("exa_text_pages")
        ledger.count("arxiv_queries", 2)
        ledger.add_usd(0.001)

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(work, range(2000)))
    snap = ledger.snapshot()
    assert snap["exa_text_pages"] == 2000
    assert snap["arxiv_queries"] == 4000
    assert snap["exa_usd"] == pytest.approx(2.0)


def test_take_n_is_all_or_nothing():
    ledger = SpendLedger({"exa_contents_pages": 5})
    assert ledger.take("exa_contents_pages", 3) is True
    assert ledger.take("exa_contents_pages", 3) is False, "3 more would overshoot: nothing is counted"
    assert ledger.take("exa_contents_pages", 2) is True
    assert ledger.remaining("exa_contents_pages") == 0


def test_for_run_splits_first_pass_gap_reserve_and_extend():
    rc = _rc("medium")                      # standard: 48 = 32 + 16 fetches, 16 = 12 + 4 searches
    ledger = SpendLedger.for_run(rc, None)
    assert ledger.remaining("fetch_first") == rc.first_pass_fetch_cap() == 32
    assert ledger.remaining("fetch_gap") == rc.gap_reserve() == 16
    assert ledger.remaining("exa_search_first") == 12
    assert ledger.remaining("exa_search_gap") == rc.exa_search_reserve() == 4
    assert ledger.remaining("exa_contents_pages") == 40
    assert ledger.remaining("fetch_extend") == 0, "no Extend yet, no Extend allowance"
    assert ledger.take("exa_search_extend") is False

    ledger.grant_extend()                   # a user Extend: fetch_per_wave + max_searches // waves
    assert ledger.remaining("fetch_extend") == rc.research.fetch_per_wave == 14
    assert ledger.remaining("exa_search_extend") == 16 // 3
    assert ledger.snapshot()["extend_allowances"] == 1


def test_quick_preset_has_no_gap_reserve():
    ledger = SpendLedger.for_run(_rc("short"), None)
    assert ledger.remaining("fetch_first") == 12
    assert ledger.take("fetch_gap") is False
    assert ledger.take("exa_search_gap") is False


def test_snapshot_round_trips_through_search_spend():
    rc = _rc("medium")
    first = SpendLedger.for_run(rc, None)
    for _ in range(5):
        assert first.take("fetch_first")
    assert first.take("exa_search_first", 3)
    first.grant_extend()
    assert first.take("fetch_extend", 2)
    first.count("exa_text_pages", 7)
    first.count("dropped_undated")
    first.add_usd(0.25)
    first.set_recency({"window": "m", "start_date": "2026-09-19"})
    snap = first.snapshot()
    assert {"exa_searches", "exa_text_pages", "exa_contents_pages", "exa_usd", "arxiv_queries",
            "ddgs_queries", "github_queries", "extend_allowances", "recency",
            "dropped_undated"} <= set(snap), "the spec §2.4 keys"
    assert snap["exa_searches"] == 3

    again = SpendLedger.for_run(rc, snap)
    assert again.remaining("fetch_first") == 32 - 5
    assert again.remaining("exa_search_first") == 12 - 3
    assert again.remaining("fetch_extend") == 14 - 2, "the allowance granted earlier is kept"
    assert again.snapshot() == snap


def test_unknown_kinds_raise():
    ledger = SpendLedger({"fetch_first": 1})
    with pytest.raises(ValueError, match="fetch_bogus"):
        ledger.take("fetch_bogus")
    with pytest.raises(ValueError, match="fetch_bogus"):
        ledger.count("fetch_bogus")
    with pytest.raises(ValueError, match="exa_text_pages"):
        ledger.take("exa_text_pages")      # an uncapped counter has no cap to take from


def test_count_on_a_capped_kind_is_never_refused_and_shares_the_counter():
    """Pinned sources count what they use (spec §2.8): an exa_contents page past the
    cap is counted, and it reduces what take() has left for unpinned pages."""
    ledger = SpendLedger({"exa_contents_pages": 2})
    ledger.count("exa_contents_pages")
    assert ledger.remaining("exa_contents_pages") == 1
    ledger.count("exa_contents_pages", 2)                  # past the cap: still counted
    assert ledger.snapshot()["exa_contents_pages"] == 3
    assert ledger.take("exa_contents_pages") is False, "the shared counter left nothing for unpinned pages"
