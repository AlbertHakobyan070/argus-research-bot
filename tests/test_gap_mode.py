"""research_node passes beyond the first (v4 spec §2.4, §2.5, §3 WP-B):

- gap mode — one wave restricted to ``coverage_gaps``, ``gap_fetch_per_wave``
  fetches and ≤ 2 Exa searches, one ``gap_history`` entry (full list
  written), ``gap_waves += 1``, ``coverage_gaps`` cleared;
- the Extend pass — one fresh allowance, ``research.extend_querygen``, a
  ``mode="extend"`` wave, ``fetch_extend`` spent (``skip_extend_wave`` on
  degrade, said so in errors);
- the between-item drain (Review Focus 5, node side);
- the custom events.

Hermetic: the search wave, the per-source fetch and ``llm.complete`` are
replaced where research looks them up."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.runtime import Runtime, RunControl

from argus import llm
from argus.graph import research as research_mod
from argus.graph.state import FetchedItem

from helpers import make_run_config

REAL_FETCH_ONE_SOURCE = research_mod.fetch_one_source
BRIEF = {"sub_questions": [{"q": "What is X?", "kind": "web"}, {"q": "How does Y work?", "kind": "web"}],
         "must_have_keywords": ["x", "y"], "summary": "s"}
DOC = "Y works by pausing the graph and resuming it with a value. " * 20


def _rec(role: str, item: str = "", ok: bool = True) -> llm.CallRecord:
    return llm.CallRecord(role=role, tier="cheap", profile="fake", provider="fake", requested_model="m",
                          served_model="m" if ok else "", routed_via="", ok=ok, item=item)


def _digest_json(relevance=4):
    return json.dumps({"relevance": relevance, "stance": "supports", "source_type": "blog",
                       "claims": [{"text": "Y pauses the graph", "quote": "pausing the graph",
                                   "confidence": "high"}]})


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Fakes for the three seams; ``fake.answers[role]`` scripts llm.complete."""
    class F:
        answers: dict = {"research.digest": lambda human: _digest_json()}
        roles: list[tuple[str, str]] = []
        waves: list[dict] = []
        wave_hits: list[dict] = []
        fetches: list[str] = []
        srcs: list[dict] = []
        on_fetch = None
        on_wave = None

    def complete(role, messages, *, engines, validate=None, item="", **kw):
        F.roles.append((role, item))
        out = F.answers[role](messages[-1].content)
        if isinstance(out, BaseException):
            raise out
        rec = _rec(role, item)
        return llm.LLMResult(text=out, parsed=validate(out) if validate else llm.NOT_VALIDATED,
                             record=rec, records=[rec])

    def wave(queries, **kw):
        F.waves.append({"queries": queries, **kw})
        if F.on_wave:
            F.on_wave()
        return list(F.wave_hits), []

    def fetch(src, **kw):
        F.fetches.append(src["url"])
        F.srcs.append(src)
        if F.on_fetch:
            F.on_fetch(src)
        md = tmp_path / f"doc{len(F.fetches)}.md"
        md.write_text(DOC, encoding="utf-8")
        return FetchedItem(url=src["url"], title=src.get("title", ""), markdown_path=str(md),
                           excerpt=DOC[:600], sub_qs=list(src.get("sub_qs") or []),
                           provider="ddgs", fetcher="native").model_dump(), []

    monkeypatch.setattr(research_mod.llm, "complete", complete)
    monkeypatch.setattr(research_mod, "run_query_wave", wave)
    monkeypatch.setattr(research_mod, "fetch_one_source", fetch)
    return F


def _src(url, sub_q):
    return {"url": url, "title": f"About {url}", "snippet": "x y", "kind": "blog", "provider": "ddgs",
            "sub_qs": [sub_q]}


def _prior(tmp_path, n=2, sub_q=0):
    """``n`` sources already fetched and digested strongly for ``sub_q``."""
    fetched, evidence = [], []
    for i in range(n):
        md = tmp_path / f"prior{i}.md"
        md.write_text(DOC, encoding="utf-8")
        url = f"https://a{i}.example/p"
        fetched.append(FetchedItem(url=url, title="a", markdown_path=str(md), sub_qs=[sub_q]).model_dump())
        evidence.append({"source_id": i + 1, "source_url": url, "title": "a", "sub_qs": [sub_q],
                         "relevance": 4, "stance": "supports", "claims": []})
    return fetched, evidence


def _state(tmp_path, *, length="medium", overrides=None, **kw):
    fetched, evidence = _prior(tmp_path)
    state = {"user_request": "x and y", "brief": dict(BRIEF), "length": length,
             "run_config": make_run_config(length, **{"research.rerank": False, **(overrides or {})}),
             "sources": [_src(f["url"], 0) for f in fetched], "fetched": fetched, "evidence": evidence,
             "queries": [], "research_rounds": 1, "coverage": {"0": 2, "1": 0}}
    state.update(kw)
    return state


# ---------------------------------------------------------------------------
# gap mode
# ---------------------------------------------------------------------------

def test_gap_wave_targets_coverage_gaps_and_records_history(fake, tmp_path):
    fake.answers["research.followup_querygen"] = lambda human: json.dumps(
        {"queries": [{"sub_q": 1, "q": "y mechanism explained", "provider": "web"}]})
    fake.wave_hits = [_src("https://new.example/g", 1)]
    state = _state(tmp_path, coverage_gaps=[1], gap_waves=0, gap_history=[])
    state["sources"] += [_src(f"https://b{i}.example/p", 1) for i in range(12)]
    state["sources"] += [_src(f"https://c{i}.example/p", 0) for i in range(3)]

    out = research_mod.research_node(state)

    new = out["fetched"][2:]
    assert len(new) == 8, "gap_fetch_per_wave (standard) fetches"
    assert all(f["sub_qs"] == [1] for f in new), "only sources for the gap sub-question"
    (w,) = fake.waves
    assert w["mode"] == "gap" and [q["query"] for q in w["queries"]] == ["y mechanism explained"]
    assert out["gap_waves"] == 1 and out["coverage_gaps"] == []
    assert out["gap_history"] == [{"round": 1, "wave": 1, "sub_qs": [1], "new_strong": 8, "source": "coverage",
                                   "new_strong_by_subq": {"1": 8}}]
    assert out["search_spend"]["fetch_gap"] == 8
    assert [r for r, _ in fake.roles].count("research.followup_querygen") == 1
    assert out["coverage"]["1"] == 8


def test_gap_history_is_written_in_full_and_the_source_can_be_the_judge(fake, tmp_path):
    fake.answers["research.followup_querygen"] = lambda human: json.dumps(
        {"queries": [{"sub_q": 0, "q": "x more", "provider": "web"}]})
    first = {"round": 1, "wave": 1, "sub_qs": [1], "new_strong": 0, "source": "coverage",
             "new_strong_by_subq": {"1": 0}}
    state = _state(tmp_path, coverage_gaps=[0], gap_waves=1, gap_history=[first],
                   outline={"sections": [{"title": "X", "sub_qs": [0]}, {"title": "Y", "sub_qs": [1]}]})
    state["sources"] += [_src("https://d0.example/p", 0)]
    out = research_mod.research_node(state)
    assert out["gap_history"][0] == first, "gap_history is a last-value channel: the full list"
    second = out["gap_history"][1]
    assert (second["wave"], second["sub_qs"], second["source"]) == (2, [0], "judge"), (
        "sub-question 0 already had its strong notes and a section: the coverage judge asked")
    assert out["gap_waves"] == 2


def test_gap_wave_refused_at_gap_waves_max(fake, tmp_path):
    state = _state(tmp_path, coverage_gaps=[1], gap_waves=2, gap_history=[])
    state["sources"] += [_src("https://b0.example/p", 1)]
    out = research_mod.research_node(state)
    assert "research: gap wave refused — gap_waves_max (2) reached" in out["errors"]
    assert fake.fetches == [] and fake.waves == []
    assert out["coverage_gaps"] == []


# ---------------------------------------------------------------------------
# Review Focus 5 (node side) — the between-item drain
# ---------------------------------------------------------------------------

def test_research_stops_between_items_on_drain(fake, tmp_path):
    control = RunControl()
    fake.on_fetch = lambda src: control.request_drain("user cancel") if len(fake.fetches) == 2 else None
    state = _state(tmp_path, overrides={"fetch.concurrency": 1}, research_rounds=0,
                   fetched=[], evidence=[], coverage={})
    state["sources"] = [_src(f"https://s{i}.example/p", i % 2) for i in range(6)]

    out = research_mod.research_node(state, Runtime(control=control))

    assert fake.fetches == ["https://s0.example/p", "https://s1.example/p"], "no fetch after the drain"
    assert len(out["fetched"]) == 2 and len(out["evidence"]) == 2, "what was fetched is still digested"
    assert "cancelled: research stopped after 2/6 fetches in wave 1 (user cancel)" in out["errors"]
    assert fake.waves == [], "the remaining waves are skipped"
    assert out["search_spend"]["fetch_first"] == 2, "only submitted fetches spend the budget"
    assert out["research_rounds"] == 1


def test_research_emits_custom_events(fake, tmp_path):
    events: list[dict] = []
    control = RunControl()
    fake.on_fetch = lambda src: control.request_drain("shutdown")
    state = _state(tmp_path, overrides={"fetch.concurrency": 1}, research_rounds=0,
                   fetched=[], evidence=[], coverage={})
    state["sources"] = [_src("https://s0.example/p", 0), _src("https://s1.example/p", 1)]
    research_mod.research_node(state, Runtime(control=control, stream_writer=events.append))
    kinds = [(e["kind"], e["node"]) for e in events]
    assert kinds[0] == ("stage", "research") and events[0]["data"]["status"] == "start"
    assert ("fetch", "research") in kinds and ("digest", "research") in kinds
    fetch = next(e["data"] for e in events if e["kind"] == "fetch")
    assert fetch == {"url": "https://s0.example/p", "ok": True, "fetcher": "native", "ms": fetch["ms"],
                     "pinned": False}
    digest = next(e["data"] for e in events if e["kind"] == "digest")
    assert digest == {"source_id": 1, "ok": True, "relevance": 4, "cached": False}
    assert any(e["kind"] == "stage" and e["data"]["status"] == "draining" for e in events)
    assert events[-1]["kind"] == "stage" and events[-1]["data"]["status"] == "end"


# ---------------------------------------------------------------------------
# the Extend pass (a plain /continue or the report's Extend button)
# ---------------------------------------------------------------------------

def test_plain_continue_extend_pass_runs_a_fresh_search_wave(fake, tmp_path):
    fake.answers["research.extend_querygen"] = lambda human: json.dumps(
        {"queries": [{"sub_q": 0, "q": "x from another angle", "provider": "web"},
                     {"sub_q": 1, "q": "y adjacent aspects", "provider": "web"}]})
    fake.answers["research.followup_querygen"] = lambda human: json.dumps({"queries": []})
    fake.wave_hits = [_src("https://new.example/1", 0), _src("https://new.example/2", 1)]
    spend = {"fetch_first": 32, "exa_search_first": 12, "exa_searches": 12}
    state = _state(tmp_path, extend_rounds=1, search_spend=spend)

    out = research_mod.research_node(state)

    assert fake.waves[0]["mode"] == "extend"
    assert [q["query"] for q in fake.waves[0]["queries"]] == ["x from another angle", "y adjacent aspects"]
    assert ("research.extend_querygen", "") in fake.roles
    assert out["search_spend"]["extend_allowances"] == 1
    urls = {f["url"] for f in out["fetched"]}
    assert {"https://new.example/1", "https://new.example/2"} <= urls
    assert out["search_spend"]["fetch_extend"] == len(out["fetched"]) - 2
    assert out["search_spend"]["fetch_first"] == 32, "the first-pass counters are untouched"


def test_extend_querygen_degrade_skips_the_wave_and_says_so(fake, tmp_path):
    attempt = _rec("research.extend_querygen", ok=False)
    fake.answers["research.extend_querygen"] = lambda human: llm.EngineDegraded(
        "role research.extend_querygen: every engine failed after 1 attempt (last: RateLimitError: 429)",
        role="research.extend_querygen", attempts=[attempt], degradation="skip_extend_wave")
    fake.answers["research.followup_querygen"] = lambda human: json.dumps({"queries": []})
    state = _state(tmp_path, extend_rounds=1)
    state["sources"] += [_src("https://left.example/1", 1)]       # left unfetched by the first pass

    out = research_mod.research_node(state)

    assert fake.waves == [], "skip_extend_wave: no search"
    assert ("[r2] degrade: research.extend_querygen → skip_extend_wave (role research.extend_querygen: "
            "every engine failed after 1 attempt (last: RateLimitError: 429))") in out["errors"]
    assert attempt.model_dump() in out["model_calls"], "the attempts are kept"
    assert "https://left.example/1" in {f["url"] for f in out["fetched"]}, "the allowance still fetches leftovers"
    assert out["search_spend"]["extend_allowances"] == 1


def test_append_only_pass_ingests_pinned_sources_without_searching(fake, tmp_path, monkeypatch):
    monkeypatch.setattr(research_mod, "fetch_one_source", REAL_FETCH_ONE_SOURCE)   # the local-file branch, no network
    md = tmp_path / "clip.md"
    md.write_text(DOC, encoding="utf-8")
    state = _state(tmp_path, append_only=True, extend_rounds=1,
                   search_spend={"fetch_first": 32})
    state["sources"] += [{"kind": "local", "title": "Clip", "local_path": str(md),
                          "url": "file:///" + str(md).replace("\\", "/"), "source": "appended-asset"},
                         _src("https://web.example/z", 1)]
    out = research_mod.research_node(state)
    assert fake.waves == [] and out["append_only"] is False
    assert [f["url"] for f in out["fetched"][2:]] == ["file:///" + str(md).replace("\\", "/")]
    assert out["fetched"][2]["pinned"] is True
    assert "https://web.example/z" not in fake.fetches, "exactly the appended sources"


# ---------------------------------------------------------------------------
# WP-B review, fix round 1
# ---------------------------------------------------------------------------

def _appended(url):
    return {"kind": "search_result", "title": url, "url": url, "summary": "", "source": "appended-url"}


def test_a_pinned_copy_wins_the_dedupe_in_an_append_only_pass(fake, tmp_path):
    """C1 (the reviewer's pinned probe): the appended URL duplicates a scout source
    the first pass left unfetched. At 98ebc5b the unpinned copy won the dedupe and
    the append pass fetched nothing, silently."""
    state = _state(tmp_path, append_only=True, extend_rounds=1, search_spend={"fetch_first": 32})
    state["sources"] += [_src("https://www.blog.example/post/", 0), _appended("https://blog.example/post")]
    out = research_mod.research_node(state)
    assert fake.fetches == ["https://blog.example/post"]
    assert fake.srcs[0]["source"] == "appended-url", "fetched as the pinned copy"
    assert not [e for e in out.get("errors", []) if "blog.example" in e]
    assert len(out["fetched"]) == 3


def test_in_an_extend_pass_the_pinned_copy_is_neither_reranked_nor_capped(fake, tmp_path):
    prompts: list[str] = []

    def rerank(human):
        prompts.append(human)
        return json.dumps({"scores": [{"id": 1, "score": 5}, {"id": 2, "score": 4}]})

    fake.answers["research.extend_querygen"] = lambda human: json.dumps({"queries": []})
    fake.answers["research.rerank"] = rerank
    state = _state(tmp_path, extend_rounds=1,
                   overrides={"research.rerank": True, "research.fetch_per_wave": 1})
    state["sources"] += [_src("https://www.blog.example/post/", 1), _appended("https://blog.example/post"),
                         _src("https://other.example/1", 1), _src("https://other.example/2", 1)]

    out = research_mod.research_node(state)

    assert prompts and not any("blog.example" in p for p in prompts), "the pinned URL is never reranked"
    assert fake.fetches == ["https://blog.example/post", "https://other.example/1"]
    assert fake.srcs[0]["source"] == "appended-url"
    assert out["search_spend"]["fetch_extend"] == 1, "the cap (fetch_per_wave 1) counts only the reranked pick"


def test_a_malformed_url_never_fails_the_node(fake, tmp_path):
    """D1: canonical_url raises on a malformed URL; research drops the source (from
    the state or a search hit) with an errors line naming it and carries on."""
    fake.answers["research.extend_querygen"] = lambda human: json.dumps(
        {"queries": [{"sub_q": 1, "q": "y more", "provider": "web"}]})
    fake.answers["research.followup_querygen"] = lambda human: json.dumps({"queries": []})
    fake.wave_hits = [{"url": "http://bad.example:99999/x", "title": "bad", "sub_qs": [1]},
                      _src("https://good.example/1", 1)]
    state = _state(tmp_path, extend_rounds=1)
    state["sources"] += [{"url": "http://[::1/broken", "title": "broken", "sub_qs": [1]}]

    out = research_mod.research_node(state)

    assert "https://good.example/1" in fake.fetches
    assert "research: dropped source with a malformed URL 'http://[::1/broken' (Invalid IPv6 URL)" in out["errors"]
    assert any(e.startswith("research: dropped source with a malformed URL 'http://bad.example:99999/x'")
               for e in out["errors"])
    assert {s["url"] for s in out["sources"]}.isdisjoint({"http://[::1/broken", "http://bad.example:99999/x"})


def test_no_follow_up_once_the_fetch_budget_is_spent(fake, tmp_path):
    """D2: wave 1 spends the first pass's last fetches; coverage gaps remain, but no
    follow-up querygen and no search run (at 98ebc5b both did, then fetched nothing)."""
    fake.answers["research.followup_querygen"] = lambda human: json.dumps(
        {"queries": [{"sub_q": 1, "q": "y more", "provider": "web"}]})
    state = _state(tmp_path, research_rounds=0, fetched=[], evidence=[], coverage={},
                   search_spend={"fetch_first": 30})                        # standard: first-pass cap 32
    state["sources"] = [_src(f"https://s{i}.example/p", i % 2) for i in range(6)]

    out = research_mod.research_node(state)

    assert len(fake.fetches) == 2 and out["search_spend"]["fetch_first"] == 32
    assert "research.followup_querygen" not in [r for r, _ in fake.roles]
    assert fake.waves == []


@pytest.mark.parametrize("pass_", ["extend opening search", "first-pass follow-up search"])
def test_no_rerank_after_a_drain(fake, tmp_path, pass_):
    """D3 (extends the drain test): a drain requested during a search — the next
    step is a rerank call, and none runs; no fetch starts."""
    control = RunControl()
    fake.answers["research.rerank"] = lambda human: json.dumps({"scores": [{"id": 1, "score": 5}]})
    fake.answers["research.extend_querygen"] = lambda human: json.dumps(
        {"queries": [{"sub_q": 1, "q": "y more", "provider": "web"}]})
    fake.answers["research.followup_querygen"] = fake.answers["research.extend_querygen"]
    fake.wave_hits = [_src("https://new.example/1", 1)]
    fake.on_wave = lambda: control.request_drain("user cancel")
    if pass_ == "extend opening search":
        state = _state(tmp_path, extend_rounds=1, overrides={"research.rerank": True})
        fetched_before = []
    else:
        state = _state(tmp_path, research_rounds=0, fetched=[], evidence=[], coverage={},
                       overrides={"research.rerank": True})
        state["sources"] = [_src("https://s0.example/p", 0), _src("https://s1.example/p", 1)]
        fetched_before = ["https://s0.example/p", "https://s1.example/p"]           # wave 1, before the drain

    out = research_mod.research_node(state, Runtime(control=control))

    assert len(fake.waves) == 1
    reranks = [r for r, _ in fake.roles].count("research.rerank")
    assert reranks == (0 if pass_ == "extend opening search" else 1), "no rerank call after the drain"
    assert fake.fetches == fetched_before
    assert any(e.startswith("cancelled: research stopped") for e in out["errors"])


@pytest.mark.parametrize("pass_", ["extend", "gap"])
def test_a_drain_before_the_pass_costs_no_querygen_call(fake, tmp_path, pass_):
    """Fix round 3: an Extend or a gap pass reads the drain before its querygen, so a
    cancel already requested costs no LLM call. (At 66c7377 the querygen ran first,
    and only run_waves' first check stopped the pass.)"""
    control = RunControl()
    control.request_drain("user cancel")
    for role in ("research.extend_querygen", "research.followup_querygen"):
        fake.answers[role] = lambda human: json.dumps({"queries": [{"sub_q": 1, "q": "y more", "provider": "web"}]})
    state = (_state(tmp_path, extend_rounds=1) if pass_ == "extend"
             else _state(tmp_path, coverage_gaps=[1], gap_waves=0, gap_history=[]))

    out = research_mod.research_node(state, Runtime(control=control))

    assert fake.roles == [], "no LLM call"
    assert fake.waves == [] and fake.fetches == []
    assert any(e.startswith("cancelled: research stopped") for e in out["errors"])
    if pass_ == "gap":
        assert out["gap_waves"] == 0, "no gap wave ran"


def test_a_drain_abandons_the_fetches_still_in_flight(fake, tmp_path):
    """Live check 12: one fetch ran 4.5 min after the drain, the 300 s drain cap
    fired task.cancel() during the digest, and the pass was lost. Once a drain is
    requested, research stops waiting on the fetches still in flight at once
    ("abandoned on cancel"), submits nothing new and digests what was fetched.
    The drain lands while the node waits on the blocked fetch alone."""
    import threading
    import time

    control, release = RunControl(), threading.Event()

    def on_fetch(src):
        if src["url"].startswith("https://slow"):
            release.wait(60)                       # blocks until the test ends (bounded: a regression fails, never hangs)
        else:                                      # the cancel arrives after this fetch returned
            threading.Timer(0.3, control.request_drain, args=("user cancel",)).start()

    fake.on_fetch = on_fetch
    state = _state(tmp_path, overrides={"fetch.concurrency": 2}, research_rounds=0,
                   fetched=[], evidence=[], coverage={})
    state["sources"] = [_src("https://slow.example/p", 0), _src("https://fast.example/p", 1),
                        _src("https://later.example/p", 0)]
    try:
        t0 = time.monotonic()
        out = research_mod.research_node(state, Runtime(control=control))
        elapsed = time.monotonic() - t0
    finally:
        release.set()
    assert elapsed < 3, f"research waited {elapsed:.1f} s on a fetch abandoned by the cancel"
    assert sorted(f["url"] for f in out["fetched"]) == ["https://fast.example/p", "https://later.example/p"]
    assert len(out["evidence"]) == 2, "the fetched items are still digested"
    assert "fetch https://slow.example/p failed: abandoned on cancel" in out["errors"], out["errors"]
    assert "cancelled: research stopped after 3/3 fetches in wave 1 (user cancel)" in out["errors"]
    assert sorted(fake.fetches) == ["https://fast.example/p", "https://later.example/p", "https://slow.example/p"]


def test_a_fetch_finished_as_the_drain_is_read_is_kept(fake, tmp_path):
    """Review Minor 1: a fetch that completes after wait() returned but before
    the drain is read was recorded "abandoned on cancel" and its item dropped."""
    import threading
    import time

    from argus.engines.ledger import SpendLedger
    from argus.graph.run_config import FetchConfig

    go = threading.Event()
    fake.on_fetch = lambda src: go.wait(10)          # running until the drain check lets it finish
    reads: list[bool] = []

    def stop() -> bool:
        if reads:                                    # the drain check after the poll timed out
            go.set()
            time.sleep(0.3)                          # the fetch returns: its future is done
            reads.append(True)
            return True
        reads.append(False)                          # the submit check
        return False

    fetched, errors = research_mod._parallel_fetch(
        [_src("https://done.example/p", 0)], cfg=FetchConfig(), ledger=SpendLedger({}), exa_on=False, stop=stop)
    assert [f["url"] for f in fetched] == ["https://done.example/p"], errors
    assert not any("abandoned on cancel" in e for e in errors), errors
