"""v4 WP-G — bench v2: deterministic metrics, compare, the result layout, queries.

Every metric of spec §3 WP-G on fixture runs (state.json and report.md, plus
evidence.json / record.json where a metric reads them), and on a v3-shaped
state.json with the keys of the orchestrator's baseline
(bench/results/v3-baseline/*/state.json): the v4-only metrics are "n/a" there
— never 0, never a crash — and every metric v3 data supports is computed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.runtime import RunControl
from langgraph.types import Command

from bench import compare, metrics, runner
from helpers import fake_engines_snapshot, make_run_config

NA = metrics.NA
ROOT = Path(__file__).resolve().parents[1]

# Metrics that need a v4 field (or v4 data) a v3 state never has.
V4_ONLY = {"fetchers_used", "gap_history", "gap_waves", "compose_runs", "compose_budget_ok",
           "quote_verified_ratio", "judged_ratio", "routed_via_captured_ratio",
           "degradations_count", "exa_spend", "takeaways_present",
           "takeaways_with_confidence_ratio", "disagreement_section_present_when_conflict",
           "scenarios_present_for_forecast"}


# ---------------------------------------------------------------------------
# Fixtures: a v3-shaped run (the baseline's keys) and the same run in v4 shape
# ---------------------------------------------------------------------------

def _src(url, published=""):
    return {"url": url, "title": "t", "snippet": "", "kind": "search_result", "provider": "exa",
            "source": "exa", "sub_qs": [0], "summary": "", "text": "", "published": published}


def _fetched(url, **kw):          # a v3 FetchedItem dict: no v4 keys
    return {"url": url, "title": "t", "markdown_path": "x.md", "section": "search_result",
            "excerpt": "", "relevance_score": 0.0, "credibility_score": None,
            "credibility_flag": None, "sub_qs": [0], "provider": "exa", **kw}


def _note(sid, url, relevance, n_claims, stance="supports"):     # a v3 EvidenceNote dict
    return {"source_id": sid, "source_url": url, "title": "t", "sub_qs": [0],
            "relevance": relevance, "stance": stance,
            "claims": [{"text": f"c{i}", "quote": "q", "confidence": "high"} for i in range(n_claims)]}


def _finding(fid, section, claim):   # a v3 Finding dict
    return {"claim": claim, "citation_urls": ["https://github.com/o/r"], "confidence": "high",
            "section": section, "id": fid}


def _call(tier, served):             # a v3 model_calls record: no role, ok or routed_via
    return {"tier": tier, "requested_model": tier, "served_model": served, "served_provider": "",
            "prompt_tokens": 10, "completion_tokens": 5, "extra": {}}


V3_STATE = {
    "thread_id": "bench-q010-0000", "user_id": 0, "user_request": "topic", "mode": "deep",
    "length": "medium", "plan": {},
    "brief": {"sub_questions": [{"q": "a", "kind": "web"}], "must_have_keywords": [],
              "summary": "", "success_criteria": []},
    "messages": [{"role": "user", "content": "topic", "ts": "2026-10-18T21:19:56.675588+04:00"},
                 {"role": "assistant", "content": "Research: 4 sources read"}],
    "sources": [_src("https://www.example.com/a/", "2026-07-16T13:55:23.000Z"),
                _src("https://arxiv.org/abs/2401.12345"),
                _src("https://github.com/o/r", "2026-10-08T00:00:00.000Z")],
    "fetched": [_fetched("https://www.example.com/a/"),
                _fetched("http://example.com/a#intro"),           # the same page again
                _fetched("https://arxiv.org/pdf/2401.12345v2", provider="arxiv", section="paper"),
                _fetched("https://github.com/o/r")],
    "evidence": [_note(1, "https://www.example.com/a/", 5, 2),
                 _note(3, "https://arxiv.org/pdf/2401.12345v2", 2, 3, stance="background"),
                 _note(4, "https://github.com/o/r", 4, 1)],
    "coverage": {"0": 2, "1": 0, "2": 1},
    "findings": [
        _finding("f0", "Alpha", "LangGraph pauses a graph at an interrupt and saves a checkpoint."),
        _finding("f1", "Beta", "LangGraph pauses the graph at an interrupt and saves a checkpoint [4]."),
        _finding("f2", "Beta", "Temporal resumes a workflow without re-running finished activities."),
        _finding("f3", "Alpha", "The resume value travels in Command(resume=...).")],
    "panel_verdict": {"verdict": "revise", "judge_verdicts": {"grounding": "revise"}, "notes": [],
                      "flagged_finding_ids": ["f1", "f2"], "unsupported_claims": ["x", "y"],
                      "fabrication_flags": [], "revise_sections": ["Beta"]},
    "revision_rounds": 3, "research_rounds": 2,
    "model_calls": ([_call("cheap", "openai/gpt-oss-120b")] * 3
                    + [_call("strong", "gemini-2.5-flash")] * 2
                    + [_call("judge", "openai/gpt-oss-20b")]),
    "errors": ["fetch https://dl.acm.org/doi/10.1/x failed: all tools returned not-ok (incl. stealth)",
               "arxiv: HTTPStatusError: Client error '429 Unknown Error'",
               "panel: precision judge failed (no useful JSON object found in: 'We need to scan' "
               "(required >= 1 keys, any of ('verdict',)))"],
    "report_paths": {"folder": "x", "md": "x/report.md", "pdf": "x/report.pdf"},
    "hitl": {"pending": False}, "append_only": False,
}

# A v3 report: 【】 markers left unnormalised, one [n] without a Sources entry,
# a code span, and the v3 flagged appendix below Sources with claim-local labels.
V3_REPORT = """# topic

_Mode: **Medium** • findings: 4 • sources: 4 • revisions: 3_

## Alpha

LangGraph pauses a graph at an interrupt [1][4]. Resumes run the node again【2†L1-L4】 and `x[0]` is code.

## Beta

Temporal does not re-run activities【3】 [5].

## Sources

1. [A](https://example.com/a)
4. [R](https://github.com/o/r)

---

## ⚠️ Unverified / flagged claims

- (id=f1) LangGraph pauses the graph at an interrupt. [1] [2]
"""


def _run(state, *, report=None, evidence=None, record=None, summary=None, metadata=None):
    return {"qid": "q1", "state": state, "report": report, "evidence": evidence,
            "metadata": metadata, "record": record, "summary": summary}


def _v3_run():
    return _run(copy.deepcopy(V3_STATE), report=V3_REPORT, summary={"elapsed_s": 258.6})


def _rec(role, served, *, requested="m", ok=True, routed="groq/x", error=""):   # a v4 CallRecord dict
    return {"role": role, "tier": "cheap", "profile": "fake", "provider": "fake",
            "requested_model": requested, "served_model": served if ok else "",
            "routed_via": routed if ok else "", "served_provider": "", "attempt": 1,
            "fallback_rank": 0, "ok": ok, "error": error, "latency_ms": 5, "prompt_tokens": 1,
            "completion_tokens": 1, "reasoning_tokens": 0, "finish_reason": "stop",
            "json_mode": False, "retry_reason": "", "item": "", "cached": False, "extra": {}}


def _v4_state() -> dict:
    s = copy.deepcopy(V3_STATE)
    s["run_config"] = make_run_config(length="medium")      # standard: gap_waves_max 2, max_compose_revisions 1
    for f, fetcher, published in zip(s["fetched"], ["exa_text", "exa_text", "native", ""],
                                     ["", "", "2026-10-01T00:00:00Z", ""]):
        f.update(canonical_url=metrics.canonical_url(f["url"]), domain="", published=published,
                 fetcher=fetcher, fetch_ms=10, chars=100, authority_prior=1, pinned=False)
    for n, verified in zip(s["evidence"], [[True, False], [True, None, None], [None]]):
        n.update(source_type="other", authority=1, cached=False)
        for c, v in zip(n["claims"], verified):
            c["quote_verified"] = v
    s["evidence"][1]["stance"] = "contradicts"             # a conflict: the disagreement metric applies
    for f, status in zip(s["findings"], ["kept", "unlocated", "cut", "kept"]):
        f.update(source_ids=[4], sentence=f["claim"], status=status)
    s["panel_verdict"].update(next="report_builder", coverage_gaps=[], wants_rewrite=False,
                              judge_not_diverse=False, uncited_specifics=[], skipped_judges=[],
                              judged=4, total=4)
    s["brief"].update(archetype="forecast", time_sensitive=True, recency_hint="m",
                      is_academic_topic=False, entities=[])
    s.update(
        compose_runs=2, revision_rounds=0, gap_waves=1,
        gap_history=[{"round": 1, "wave": 1, "sub_qs": [1], "new_strong": 2, "source": "coverage"}],
        search_spend={"exa_searches": 4, "exa_text_pages": 30, "exa_contents_pages": 2,
                      "exa_usd": 0.06, "arxiv_queries": 1, "ddgs_queries": 3, "github_queries": 0,
                      "extend_allowances": 0, "recency": None, "dropped_undated": 0},
        analysis={"takeaways": [{"text": "a", "source_ids": [1], "confidence": "high", "reasons": []},
                                {"text": "b", "source_ids": [4], "confidence": "medium", "reasons": []},
                                {"text": "c", "source_ids": [], "confidence": "", "reasons": []}],
                  "stance_map": {}, "has_disagreement_section": True, "has_scenarios": False},
        model_calls=[
            _rec("research.digest", "llama-3.3-70b"),
            _rec("research.digest", "", ok=False,
                 error="unparseable: ValueError: no useful JSON object found in: 'x'"),
            _rec("research.digest", "llama-3.3-70b", routed=""),
            _rec("compose.section", "gemini-2.5-pro", requested="gemini-2.5-pro", routed="google/g"),
            _rec("panel.grounding", "gpt-oss-120b", routed="groq/gpt-oss")],
        errors=[
            "skip: research.digest https://x (role research.digest: every engine failed after 2 "
            "attempts (last: unparseable: ValueError: no useful JSON object found in: 'x'))",
            "degrade: research.rerank → keyword_triage (429)",
            "panel: coverage judge failed (no useful JSON object found in: 'x' (required >= 1 keys))",
            "fetch: https://10.0.0.1/x failed (refused: private address; rungs tried: native)"])
    return s


V4_REPORT = """# topic

_Engines: fake · reroute 75 % · judge diversity ok · judged 4/4 · warnings 0_

## TL;DR

_TL;DR — compiled from the report's own cited, judged findings_

- LangGraph pauses a graph at an interrupt [1][4].

## Alpha

LangGraph pauses a graph at an interrupt and saves a checkpoint [4]. See `cfg[0]`.

## Analysis

Takeaway [1].

## Flagged but not located (still in body)

- LangGraph pauses the graph at an interrupt [4].

## Sources

1. [A](https://example.com/a)
4. [R](https://github.com/o/r)
"""

V4_VALUES = {
    "unique_sources": 3, "dupe_ratio": 0.25, "fetched_success_ratio": 0.8,
    "fetchers_used": {"(none)": 1, "exa_text": 2, "native": 1}, "pinned_ingested_ratio": NA,
    "evidence_strong_ratio": 0.667, "coverage_min": 0, "coverage_mean": 1.0,
    "gap_history": [{"round": 1, "wave": 1, "sub_qs": [1], "new_strong": 2, "source": "coverage"}],
    "gap_waves": 1, "compose_runs": 2, "rewrite_rounds": 0, "compose_budget_ok": True,
    "quote_verified_ratio": 0.667, "claims_per_primary_source": 2.0,
    "cite_markers_unresolved": 0, "sources_block_present": True, "flagged_claims_in_body": 1,
    "cross_section_duplicate_ratio": 0.667, "judged_ratio": 1.0, "calls_total": 5,
    "calls_by_role": {"compose.section": 1, "panel.grounding": 1, "research.digest": 3,
                      "research.rerank": 0},
    "reroute_rate": 0.75, "routed_via_captured_ratio": 0.75, "judge_writer_same_family": False,
    "degradations_count": 2, "json_parse_failures": 2,
    "exa_spend": {"searches": 4, "text_pages": 30, "contents_pages": 2, "usd": 0.06},
    "wall_clock_s": 301.5, "median_source_age_days": 17,
    "takeaways_present": True, "takeaways_with_confidence_ratio": 0.667,
    "disagreement_section_present_when_conflict": True, "scenarios_present_for_forecast": False,
}


# ---------------------------------------------------------------------------
# Every metric, on a v3-shaped run and on a v4 run
# ---------------------------------------------------------------------------

def test_a_v3_shaped_run_computes_what_v3_data_supports_and_says_na_for_the_rest():
    m = metrics.compute(_v3_run())
    assert list(m) == list(metrics.METRICS)                  # every metric, in the spec's order
    assert {k for k, v in m.items() if v == NA} == V4_ONLY | {"pinned_ingested_ratio"}
    assert m == {
        **{k: NA for k in V4_ONLY}, "pinned_ingested_ratio": NA,     # nothing was appended
        "unique_sources": 3, "dupe_ratio": 0.25, "fetched_success_ratio": 0.8,
        "evidence_strong_ratio": 0.667, "coverage_min": 0, "coverage_mean": 1.0,
        "rewrite_rounds": 3, "claims_per_primary_source": 2.0,
        "cite_markers_unresolved": 3, "sources_block_present": True,
        "flagged_claims_in_body": 2,                    # no status in v3: every flag counts (unlocated)
        "cross_section_duplicate_ratio": 0.5, "calls_total": 6,
        "calls_by_role": {"legacy.cheap": 3, "legacy.judge": 1, "legacy.strong": 2},
        "reroute_rate": 1.0, "judge_writer_same_family": False, "json_parse_failures": 1,
        "wall_clock_s": 258.6, "median_source_age_days": 52.0}


def test_a_v4_run_computes_every_metric():
    m = metrics.compute(_run(_v4_state(), report=V4_REPORT, record={"duration_s": 301.5}))
    assert m == V4_VALUES
    assert list(m) == list(metrics.METRICS)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,canonical", [
    ("https://www.Example.com/a/", "https://example.com/a"),
    ("http://example.com:80/a#frag", "https://example.com/a"),
    ("https://example.com:8443/a", "https://example.com:8443/a"),
    ("https://example.com/a?utm_source=x&id=3&ref=hn&fbclid=1&gclid=2", "https://example.com/a?id=3"),
    ("https://export.arxiv.org/abs/2401.12345", "https://arxiv.org/abs/2401.12345"),
    ("https://arxiv.org/pdf/2401.12345v2", "https://arxiv.org/abs/2401.12345"),
    ("https://arxiv.org/html/2401.12345v1", "https://arxiv.org/abs/2401.12345"),
    ("https://ar5iv.labs.arxiv.org/html/2401.12345", "https://arxiv.org/abs/2401.12345"),
    ("https://huggingface.co/papers/2401.12345", "https://arxiv.org/abs/2401.12345"),
    ("https://huggingface.co/meta-llama/Llama-3", "https://huggingface.co/meta-llama/Llama-3"),
    ("file:///A:/vault/note.md", "file:///A:/vault/note.md"),
])
def test_canonical_url_follows_the_spec_rules(url, canonical):
    assert metrics.canonical_url(url) == canonical


def test_source_counts_on_an_empty_or_missing_fetch_list():
    assert metrics.compute(_run({"fetched": []}))["unique_sources"] == 0
    for state in ({"fetched": []}, {}):
        m = metrics.compute(_run(state))
        assert (m["dupe_ratio"], m["fetched_success_ratio"]) == (NA, NA)
    assert metrics.compute(_run({}))["unique_sources"] == NA


def test_fetched_success_ratio_counts_failed_fetch_lines_of_both_formats():
    state = {"fetched": [{"url": "https://a.example"}],
             "errors": ["fetch https://b.example failed: timeout", "[r2] fetch: https://c.example failed (x)",
                        "fetcher crashed", "digest: https://a.example failed (429)"]}
    assert metrics.compute(_run(state))["fetched_success_ratio"] == 0.333


def test_pinned_ingested_ratio_counts_appended_and_pinned_sources():
    state = {"sources": [{"url": "https://a.example", "source": "appended-url"},
                         {"url": "file:///A:/v/t.md", "source": "appended-asset", "local_path": "A:/v/t.md"},
                         {"url": "https://c.example", "source": "exa"}],
             "fetched": [{"url": "https://a.example"}, {"url": "https://d.example", "pinned": True},
                         {"url": "https://c.example"}]}
    # requested: a (appended), the asset (appended, not fetched), d (pinned in fetched)
    assert metrics.compute(_run(state))["pinned_ingested_ratio"] == 0.667


def test_evidence_and_coverage_metrics_need_their_fields():
    m = metrics.compute(_run({"evidence": [], "coverage": {}}))
    assert (m["evidence_strong_ratio"], m["coverage_min"], m["coverage_mean"]) == (NA, NA, NA)
    assert metrics.compute(_run({}))["claims_per_primary_source"] == NA
    no_primary = {"evidence": [_note(1, "https://blog.example/x", 5, 4)]}
    assert metrics.compute(_run(no_primary))["claims_per_primary_source"] == NA


# ---------------------------------------------------------------------------
# Compose bookkeeping (orchestrator ruling 1: §2.4's per-round invariant)
# ---------------------------------------------------------------------------

def test_a_v4_run_without_gap_waves_counts_zero_and_without_compose_runs_is_na():
    """The post-WP-D reference run: run_config present, no gap mode yet and a v3
    compose that does not count its runs."""
    state = {"run_config": make_run_config(length="medium"), "revision_rounds": 2}
    m = metrics.compute(_run(state))
    assert (m["gap_waves"], m["gap_history"]) == (0, [])
    assert (m["compose_runs"], m["compose_budget_ok"], m["rewrite_rounds"]) == (NA, NA, 2)


def _budget(**kw):
    state = {"run_config": make_run_config(length="medium"),   # gap_waves_max 2, max_compose_revisions 1
             "compose_runs": 2, "revision_rounds": 0, "gap_waves": 1,
             "gap_history": [{"round": 1, "wave": 1, "sub_qs": [1], "new_strong": 2, "source": "coverage"}]}
    state.update(kw)
    return metrics.compute(_run({k: v for k, v in state.items() if v is not ...}))["compose_budget_ok"]


@pytest.mark.parametrize("change,ok", [
    ({}, True),
    ({"compose_runs": 3}, False),                                # one compose too many for this round
    ({"compose_runs": 3, "revision_rounds": 1}, True),           # a panel rewrite
    ({"compose_runs": 4, "revision_rounds": 2}, False),          # over max_compose_revisions
    ({"compose_runs": 4, "gap_waves": 3, "gap_history": [         # over gap_waves_max
        {"round": 1, "wave": w, "sub_qs": [1], "new_strong": 1, "source": "coverage"} for w in (1, 2, 3)]},
     False),
    # a user Revise reset the per-round counters: round 2 has no gap wave of its own
    ({"revise_rounds": 1, "compose_runs": 1}, True),
    ({"revise_rounds": 1, "compose_runs": 3}, False),            # the stale Phase D formula's count
    # an Extend round with its own gap wave
    ({"extend_rounds": 1, "gap_waves": 2, "compose_runs": 2, "gap_history": [
        {"round": 1, "wave": 1, "sub_qs": [1], "new_strong": 1, "source": "coverage"},
        {"round": 2, "wave": 1, "sub_qs": [0], "new_strong": 1, "source": "judge"}]}, True),
    ({"compose_runs": ...}, NA),
    ({"run_config": ...}, NA),
    ({"revision_rounds": ...}, NA),
])
def test_compose_budget_ok_is_the_per_round_invariant(change, ok):
    assert _budget(**change) == ok


# ---------------------------------------------------------------------------
# Evidence quality
# ---------------------------------------------------------------------------

def test_quote_verified_ratio_is_na_when_no_quote_was_checked():
    state = copy.deepcopy(_v4_state())
    for n in state["evidence"]:
        for c in n["claims"]:
            c["quote_verified"] = None
    assert metrics.compute(_run(state))["quote_verified_ratio"] == NA


# ---------------------------------------------------------------------------
# Report body: citation markers, the Sources block
# ---------------------------------------------------------------------------

def _markers(body: str, sources: str = "1. [A](https://a.example)\n") -> int:
    return metrics.compute(_run({}, report=f"# t\n\n{body}\n\n## Sources\n\n{sources}"))["cite_markers_unresolved"]


@pytest.mark.parametrize("marker", ["【1】", "［1］", "〔1〕", "【1†L1-L4】", "[1, 1]", "[1-1]",
                                    "[1–1]", "[^1]", "[Source 1]"])
def test_every_unnormalised_marker_form_counts_as_unresolved(marker):
    assert _markers(f"A claim {marker}.") == 1


def test_canonical_markers_resolve_against_the_sources_entries():
    assert _markers("A [1]. B [1][1].") == 0
    assert _markers("A [1]. B [2].") == 1
    assert _markers("A [7].", sources="[7] [Seven](https://s.example)\n") == 0
    assert _markers("Code `a[2]` and\n\n```\nb[3]\n```\n") == 0
    assert _markers("【注】 is not a citation.") == 0


def test_without_a_sources_heading_every_marker_is_unresolved():
    m = metrics.compute(_run({}, report="# t\n\nA [1]. B【2】.\n"))
    assert (m["cite_markers_unresolved"], m["sources_block_present"]) == (2, False)


@pytest.mark.parametrize("report,present", [
    ("# t\n\nA [1].\n\n## Sources\n\n1. [A](https://a.example)\n", True),
    ("# t\n\nA [1].\n\n## References\n\n[1] A https://a.example\n", True),
    ("# t\n\nA.\n\n## Sources\n\n---\n", False),
    ("# t\n\nA.\n", False),
])
def test_sources_block_present_needs_a_heading_with_entries(report, present):
    assert metrics.compute(_run({}, report=report))["sources_block_present"] is present


def test_report_metrics_are_na_without_a_report():
    m = metrics.compute(_run({}))
    assert (m["cite_markers_unresolved"], m["sources_block_present"]) == (NA, NA)


# ---------------------------------------------------------------------------
# Flags, duplicates, judging
# ---------------------------------------------------------------------------

def test_flagged_claims_in_body_counts_every_flag_not_cut():
    state = _v4_state()
    assert metrics.compute(_run(state))["flagged_claims_in_body"] == 1          # f1 unlocated, f2 cut
    state["findings"][1]["status"] = "kept"                                     # flagged but never cut
    assert metrics.compute(_run(state))["flagged_claims_in_body"] == 1
    state["findings"][2]["status"] = "unlocated"
    assert metrics.compute(_run(state))["flagged_claims_in_body"] == 2
    state["panel_verdict"]["flagged_finding_ids"] = ["f1", "f1", "f99"]          # a repeat, an unknown id
    assert metrics.compute(_run(state))["flagged_claims_in_body"] == 1
    assert metrics.compute(_run({"findings": []}))["flagged_claims_in_body"] == NA


def test_evidence_json_findings_and_degradations_take_precedence():
    findings = [{**f, "status": "cut"} for f in _v4_state()["findings"]]
    evidence = {"findings": findings, "degradations": [{"round": 1, "text": "degrade: x"}],
                "analysis": {"takeaways": [], "has_disagreement_section": False, "has_scenarios": True}}
    m = metrics.compute(_run(_v4_state(), evidence=evidence))
    assert (m["flagged_claims_in_body"], m["cross_section_duplicate_ratio"]) == (0, NA)
    assert m["degradations_count"] == 1
    assert (m["takeaways_present"], m["takeaways_with_confidence_ratio"]) == (False, NA)
    assert (m["disagreement_section_present_when_conflict"], m["scenarios_present_for_forecast"]) == (
        False, True)


def test_cross_section_duplicates_need_two_sections():
    state = {"findings": [_finding("f0", "Report", "A claim."), _finding("f1", "Report", "A claim.")]}
    assert metrics.compute(_run(state))["cross_section_duplicate_ratio"] == NA


@pytest.mark.parametrize("pv,ratio", [({"judged": 3, "total": 4}, 0.75), ({"judged": 0, "total": 0}, NA),
                                      ({"verdict": "pass"}, NA), (None, NA)])
def test_judged_ratio(pv, ratio):
    assert metrics.compute(_run({"panel_verdict": pv}))["judged_ratio"] == ratio


# ---------------------------------------------------------------------------
# Engines and provenance
# ---------------------------------------------------------------------------

def test_degradations_count_reads_the_facade_lines_with_round_tags():
    state = _v4_state()
    state["errors"] = ["[r2] skip: compose.section Beta (429)", "degrade: [r1] outline.plan → x (y)",
                       "legacy: run_config defaults applied (pre-v4 checkpoint)", "arxiv: 429",
                       "fetch https://x failed: y"]
    assert metrics.compute(_run(state))["degradations_count"] == 3


def test_engine_metrics_are_na_without_model_calls():
    m = metrics.compute(_run({}))
    assert {k: m[k] for k in ("calls_total", "calls_by_role", "reroute_rate", "routed_via_captured_ratio",
                              "judge_writer_same_family", "degradations_count", "json_parse_failures")} == {
        "calls_total": NA, "calls_by_role": NA, "reroute_rate": NA, "routed_via_captured_ratio": NA,
        "judge_writer_same_family": NA, "degradations_count": NA, "json_parse_failures": NA}


def test_wall_clock_prefers_the_run_record_over_the_v3_capture():
    assert metrics.compute(_run({}, record={"duration_s": 12.5},
                                summary={"elapsed_s": 99.0}))["wall_clock_s"] == 12.5
    assert metrics.compute(_run({}))["wall_clock_s"] == NA


def test_source_age_needs_a_run_date_and_a_dated_source():
    state = copy.deepcopy(V3_STATE)
    for s in state["sources"]:
        s["published"] = ""
    assert metrics.compute(_run(state))["median_source_age_days"] == NA
    del state["messages"]
    assert metrics.compute(_run(state))["median_source_age_days"] == NA


def test_insight_proxies_apply_only_where_they_mean_something():
    state = _v4_state()
    state["evidence"][1]["stance"] = "background"           # no conflict among the notes
    state["brief"]["archetype"] = "explainer"
    m = metrics.compute(_run(state))
    assert (m["disagreement_section_present_when_conflict"], m["scenarios_present_for_forecast"]) == (NA, NA)
    assert m["takeaways_present"] is True


# ---------------------------------------------------------------------------
# Files: load_run, collect, the metrics CLI
# ---------------------------------------------------------------------------

def _write_run(qdir: Path, state: dict, *, report=None, evidence=None, record=None, summary=None) -> None:
    qdir.mkdir(parents=True)
    (qdir / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    if report is not None:
        (qdir / "report.md").write_text(report, encoding="utf-8")
    for name, obj in (("evidence", evidence), ("record", record), ("summary", summary)):
        if obj is not None:
            (qdir / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")


def test_load_run_reads_the_result_layout(tmp_path):
    _write_run(tmp_path / "q010", V3_STATE, report=V3_REPORT, summary={"elapsed_s": 258.6})
    run = metrics.load_run(tmp_path / "q010")
    assert (run["qid"], run["report"], run["summary"]) == ("q010", V3_REPORT, {"elapsed_s": 258.6})
    assert (run["evidence"], run["record"], run["metadata"]) == (None, None, None)
    assert metrics.compute(run) == metrics.compute(_v3_run())


def test_load_run_names_a_broken_file(tmp_path):
    _write_run(tmp_path / "q1", {})
    (tmp_path / "q1" / "evidence.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence.json: not valid JSON"):
        metrics.load_run(tmp_path / "q1")
    with pytest.raises(FileNotFoundError, match="no state.json"):
        metrics.load_run(tmp_path / "q2")


def test_metrics_cli_prints_one_row_per_metric_and_a_mean(tmp_path, capsys):
    _write_run(tmp_path / "q010", V3_STATE, report=V3_REPORT, summary={"elapsed_s": 258.6})
    _write_run(tmp_path / "q001", _v4_state(), report=V4_REPORT, record={"duration_s": 301.5})
    (tmp_path / "reports").mkdir()                              # not a query: no state.json
    assert metrics.main(["--results", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "| metric | q001 | q010 | mean |"
    rows = {line.split(" | ")[0].lstrip("| "): line for line in out.splitlines()[2:]}
    assert list(rows) == list(metrics.METRICS)
    assert rows["compose_runs"] == "| compose_runs | 2 | n/a | 2 |"
    assert rows["sources_block_present"] == "| sources_block_present | yes | yes | 1 |"
    assert rows["wall_clock_s"] == "| wall_clock_s | 301.5 | 258.6 | 280.05 |"
    assert rows["calls_by_role"].endswith("| legacy.cheap: 3, legacy.judge: 1, legacy.strong: 2 | - |")


def test_metrics_cli_refuses_a_folder_without_runs(tmp_path):
    with pytest.raises(SystemExit, match="no <qid>/state.json"):
        metrics.main(["--results", str(tmp_path)])


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def test_compare_renders_a_delta_table_over_the_shared_queries(tmp_path, capsys):
    _write_run(tmp_path / "a" / "q010", V3_STATE, report=V3_REPORT, summary={"elapsed_s": 258.6})
    _write_run(tmp_path / "a" / "q001", V3_STATE, report=V3_REPORT)
    _write_run(tmp_path / "b" / "q010", _v4_state(), report=V4_REPORT, record={"duration_s": 301.5})
    _write_run(tmp_path / "b" / "q013", _v4_state())
    assert compare.main(["--a", str(tmp_path / "a"), "--b", str(tmp_path / "b")]) == 0
    out = capsys.readouterr().out
    rows = {line.split(" | ")[0].lstrip("| "): line for line in out.splitlines() if line.startswith("| ")}
    assert rows["metric"] == "| metric | q010 |"
    assert rows["flagged_claims_in_body"] == "| flagged_claims_in_body | 2 -> 1 (-1) |"
    assert rows["cite_markers_unresolved"] == "| cite_markers_unresolved | 3 -> 0 (-3) |"
    assert rows["wall_clock_s"] == "| wall_clock_s | 258.6 -> 301.5 (+42.9) |"
    assert rows["compose_budget_ok"] == "| compose_budget_ok | n/a -> yes |"
    assert rows["sources_block_present"] == "| sources_block_present | yes -> yes |"
    assert "only in a: q001" in out and "only in b: q013" in out


def test_compare_refuses_sets_with_no_shared_query(tmp_path):
    _write_run(tmp_path / "a" / "q010", V3_STATE)
    _write_run(tmp_path / "b" / "q011", V3_STATE)
    with pytest.raises(SystemExit, match="no query in both"):
        compare.main(["--a", str(tmp_path / "a"), "--b", str(tmp_path / "b")])


# ---------------------------------------------------------------------------
# The runner's result layout (spec §3 WP-G) and its astream arguments (§1.1)
# ---------------------------------------------------------------------------

class _FakeGraph:
    """Pauses at the plan gate and at the report gate, then ends; keeps every
    astream call's input and keyword arguments."""

    def __init__(self, values):
        self.values, self.calls, self.gates = values, [], 2

    async def astream(self, inp, **kw):
        self.calls.append((inp, kw))
        if self.gates:
            self.gates -= 1
            yield ("updates", {"__interrupt__": ()})
            return
        yield ("custom", {"kind": "stage"})
        yield ("updates", {"deliver": {"hitl": {"pending": False}}})

    async def aget_state(self, cfg):
        return SimpleNamespace(values=self.values, created_at="2026-10-19T00:00:00+00:00")


def _report_folder(tmp_path, *, evidence=True) -> dict:
    folder = tmp_path / "vault" / "20261019_topic_short"
    folder.mkdir(parents=True)
    (folder / "report.md").write_text("# report\n", encoding="utf-8")
    (folder / "metadata.json").write_text('{"engines": {}}', encoding="utf-8")
    paths = {"folder": str(folder), "md": str(folder / "report.md"), "pdf": None}
    if evidence:
        (folder / "evidence.json").write_text('{"schema": 1}', encoding="utf-8")
        paths["evidence"] = str(folder / "evidence.json")
    return paths


async def test_runner_writes_the_bench_result_layout(monkeypatch, tmp_path):
    values = {"draft_md": "d", "fetched": [], "findings": [], "report_paths": _report_folder(tmp_path)}
    fake = _FakeGraph(values)
    monkeypatch.setattr(runner, "build_graph", lambda checkpointer: fake)
    rec = await runner._run_one({"id": "q9", "query": "topic"}, length="short", depth=None,
                                engines=fake_engines_snapshot(), timeout_s=None,
                                state_dir=tmp_path / "res", origin="live_check",
                                overrides={"recency": "m"})
    qdir = tmp_path / "res" / "q9"
    assert rec["status"] == "ok", rec
    assert json.loads((qdir / "record.json").read_text(encoding="utf-8")) == rec
    assert json.loads((qdir / "state.json").read_text(encoding="utf-8")) == values
    assert (qdir / "report.md").read_text(encoding="utf-8") == "# report\n"
    assert json.loads((qdir / "metadata.json").read_text(encoding="utf-8")) == {"engines": {}}
    assert json.loads((qdir / "evidence.json").read_text(encoding="utf-8")) == {"schema": 1}
    # three drives (start, past the plan gate, past the report gate), one RunControl for the run
    assert len(fake.calls) == 3
    (first, kw0), *resumes = fake.calls
    assert isinstance(kw0["control"], RunControl)
    assert all(kw["durability"] == "sync" and kw["stream_mode"] == ["updates", "custom"]
               and kw["control"] is kw0["control"] for _, kw in fake.calls)
    assert all(isinstance(inp, Command) and inp.resume is True for inp, _ in resumes)
    rc = first["run_config"]
    assert (rc["origin"], rc["recency"], rc["overrides_applied"]) == ("live_check", "m", {"recency": "m"})


async def test_runner_copies_no_report_when_the_run_made_none(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "build_graph", lambda checkpointer: _FakeGraph({"draft_md": ""}))
    rec = await runner._run_one({"id": "q9", "query": "topic"}, length="short", depth=None,
                                engines=fake_engines_snapshot(), timeout_s=60, state_dir=tmp_path)
    assert (rec["status"], json.loads((tmp_path / "q9" / "state.json").read_text())) == ("ok", {"draft_md": ""})
    assert sorted(p.name for p in (tmp_path / "q9").iterdir()) == ["record.json", "state.json"]


async def test_runner_records_a_report_folder_without_metadata_as_an_error(monkeypatch, tmp_path):
    paths = _report_folder(tmp_path, evidence=False)
    Path(paths["folder"], "metadata.json").unlink()
    monkeypatch.setattr(runner, "build_graph",
                        lambda checkpointer: _FakeGraph({"report_paths": paths}))
    rec = await runner._run_one({"id": "q9", "query": "topic"}, length="short", depth=None,
                                engines=fake_engines_snapshot(), timeout_s=60, state_dir=tmp_path / "res")
    assert rec["status"] == "error" and "metadata.json" in rec["error"]
    assert json.loads((tmp_path / "res" / "q9" / "record.json").read_text())["status"] == "error"


# ---------------------------------------------------------------------------
# queries.jsonl (orchestrator ruling 2)
# ---------------------------------------------------------------------------

def test_queries_q011_to_q016_are_appended_in_the_record_shape():
    lines = (ROOT / "bench" / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(set(r) == {"id", "query", "expected_domains", "difficulty"} for r in rows)
    assert [(r["id"], r["query"]) for r in rows[-6:]] == [
        ("q011", "gold price outlook for the next month"),
        ("q012", "Bitcoin market trends this quarter"),
        ("q013", "Quantum Lattice Transformer by Nebulon AI benchmark results"),
        ("q014", "Helixon-9 protein design model release notes"),
        ("q015", "psychological and social factors behind doomscrolling"),
        ("q016", "Mediterranean vs Nordic diet for longevity — which is better supported")]
    assert all((r["expected_domains"], r["difficulty"]) == ([], "") for r in rows[-6:])
