"""v4 WP-G — scripts/live_check.py: argument parsing and every acceptance check.

The live check replaces tests/manual_e2e.py: one graph-direct run through
``bench.runner._run_one`` (origin "live_check", the engines resolved first),
the bench result layout, then the acceptance table of spec §3 WP-G; it exits
1 when any check fails. A check whose field the run does not have is "n/a",
never a pass by default. No live call: ``_run_one`` and the engines snapshot
are faked.
"""
from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from argus.engines import registry
from bench import metrics, runner
from helpers import fake_engines_snapshot, make_run_config

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("live_check", ROOT / "scripts" / "live_check.py")
live_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live_check)

NA, PASS, FAIL = metrics.NA, "PASS", "FAIL"


@pytest.fixture(autouse=True)
def _reports_root_is_restored():
    """``_main`` sets ARGUS_REPORTS_ROOT in os.environ: no test may leave it changed."""
    before = os.environ.get("ARGUS_REPORTS_ROOT")
    yield
    assert os.environ.get("ARGUS_REPORTS_ROOT") == before


def _no_reports_root(monkeypatch):
    """Unset ARGUS_REPORTS_ROOT for the test. setenv first: deleting an unset
    variable records nothing, and the value _main sets would leak."""
    monkeypatch.setenv("ARGUS_REPORTS_ROOT", "x")
    monkeypatch.delenv("ARGUS_REPORTS_ROOT")


def _r(state=None, *, report=None, evidence=None, metadata=None, record=None):
    return {"qid": "q1", "state": state or {}, "report": report, "evidence": evidence,
            "metadata": metadata, "record": record, "summary": None}


# ---------------------------------------------------------------------------
# Arguments and the query
# ---------------------------------------------------------------------------

def test_parse_args_defaults():
    a = live_check.parse_args([])
    assert (a.query, a.length, a.depth, a.profile, a.recency, a.expect_banner) == (
        "q010", "medium", None, None, None, False)
    assert a.out == Path(tempfile.gettempdir()) / "argus_live_check"


def test_parse_args_takes_every_option():
    a = live_check.parse_args(["--query", "gold price outlook for the next month", "--length", "short",
                               "--depth", "standard", "--profile", "free-proxy", "--recency", "m",
                               "--out", "x", "--expect-banner"])
    assert (a.query, a.length, a.depth, a.profile, a.recency, a.out, a.expect_banner) == (
        "gold price outlook for the next month", "short", "standard", "free-proxy", "m", Path("x"), True)


@pytest.mark.parametrize("argv", [["--length", "quick"], ["--depth", "huge"], ["--recency", "q"]])
def test_parse_args_refuses_unknown_choices(argv):
    with pytest.raises(SystemExit):
        live_check.parse_args(argv)


def test_resolve_query_by_id_by_text_or_ad_hoc():
    assert live_check.resolve_query("q013")["query"] == (
        "Quantum Lattice Transformer by Nebulon AI benchmark results")
    assert live_check.resolve_query("gold price outlook for the next month")["id"] == "q011"
    assert live_check.resolve_query("What is X? (2026)") == {"id": "what_is_x_2026",
                                                            "query": "What is X? (2026)"}
    with pytest.raises(SystemExit, match="unknown bench query id 'q099'"):
        live_check.resolve_query("q099")         # a typo must not start a live run on "q099"


# ---------------------------------------------------------------------------
# Every check, passing and failing
# ---------------------------------------------------------------------------

LC = live_check
GROUNDING_ON = make_run_config(length="medium")
GROUNDING_OFF = make_run_config(length="medium", **{"panel.grounding": False})
EVIDENCE = {**{k: {} for k in LC.EVIDENCE_KEYS}, "schema": 1, "compose_runs": 1, "revision_rounds": 0,
            "gap_waves": 0, "revise_rounds": 0, "extend_rounds": 0, "degraded": False,
            **{k: [] for k in ("gap_history", "sources", "notes", "findings", "degradations", "errors")}}


@pytest.mark.parametrize("check,run,m,status", [
    (LC.check_run_completed, _r(record={"status": "ok", "duration_s": 3.0, "error": None}), {}, PASS),
    (LC.check_run_completed, _r(record={"status": "error", "duration_s": 3.0, "error": "boom"}), {}, FAIL),
    (LC.check_run_completed, _r(), {}, FAIL),
    (LC.check_fetched, _r({"fetched": [{"url": "u"}]}), {}, PASS),
    (LC.check_fetched, _r({"fetched": []}), {}, FAIL),
    (LC.check_evidence, _r({"evidence": [{"source_id": 1}]}), {}, PASS),
    (LC.check_evidence, _r({}), {}, FAIL),
    (LC.check_cited_findings, _r({"findings": [{"citation_urls": ["u"]}]}), {}, PASS),
    (LC.check_cited_findings, _r({"findings": [{"citation_urls": [], "source_ids": [2]}]}), {}, PASS),
    (LC.check_cited_findings, _r({"findings": [{"citation_urls": []}]}), {}, FAIL),
    # only kept findings count (no status: v3, kept); cut and unlocated ones never do
    (LC.check_cited_findings, _r({"findings": [{"citation_urls": ["u"], "status": "kept"}]}), {}, PASS),
    (LC.check_cited_findings, _r({"findings": [{"citation_urls": ["u"], "status": "cut"}]}), {}, FAIL),
    (LC.check_cited_findings, _r({"findings": [{"citation_urls": ["u"], "status": "unlocated"}]}), {}, FAIL),
    (LC.check_few_findings, _r({"findings": [{}] * 3}), {}, PASS),
    (LC.check_few_findings, _r({"findings": [{}] * 4}), {}, FAIL),
    (LC.check_few_findings, _r({"findings": [{"status": "kept"}] * 3 + [{"status": "cut"}] * 5}), {}, PASS),
    (LC.check_few_findings, _r({"findings": [{"status": "kept"}] * 4 + [{"status": "cut"}]}), {}, FAIL),
    (LC.check_report_size, _r(report="x" * 1001), {}, PASS),
    (LC.check_report_size, _r(report="x" * 1000), {}, FAIL),
    (LC.check_report_size, _r(), {}, FAIL),
    (LC.check_sources_block, _r(), {"sources_block_present": True}, PASS),
    (LC.check_sources_block, _r(), {"sources_block_present": False}, FAIL),
    (LC.check_sources_block, _r(), {"sources_block_present": NA}, NA),
    (LC.check_cite_markers, _r(), {"cite_markers_unresolved": 0}, PASS),
    (LC.check_cite_markers, _r(), {"cite_markers_unresolved": 21}, FAIL),
    (LC.check_cite_markers, _r(), {"cite_markers_unresolved": NA}, NA),
    (LC.check_flagged_in_body, _r(), {"flagged_claims_in_body": 0}, PASS),
    (LC.check_flagged_in_body, _r(), {"flagged_claims_in_body": 4}, FAIL),
    (LC.check_flagged_in_body, _r(), {"flagged_claims_in_body": NA}, NA),
    (LC.check_judged, _r({"panel_verdict": {"judged": 4, "total": 4}}), {}, PASS),
    (LC.check_judged, _r({"panel_verdict": {"judged": 3, "total": 4}}), {}, FAIL),
    # v4 defaults with nothing counted although findings were due for judging (the v3 panel)
    (LC.check_judged, _r({"panel_verdict": {"judged": 0, "total": 0}, "findings": [{}],
                          "run_config": GROUNDING_ON}), {}, FAIL),
    (LC.check_judged, _r({"panel_verdict": {"judged": 0, "total": 0}, "findings": [],
                          "run_config": GROUNDING_ON}), {}, NA),
    (LC.check_judged, _r({"panel_verdict": {"judged": 0, "total": 0}, "findings": [{}],
                          "run_config": GROUNDING_OFF}), {}, NA),
    (LC.check_judged, _r({"panel_verdict": {"verdict": "pass"}, "findings": [{}]}), {}, NA),   # v3 shape
    (LC.check_compose_budget, _r(), {"compose_budget_ok": True}, PASS),
    (LC.check_compose_budget, _r(), {"compose_budget_ok": False}, FAIL),
    (LC.check_compose_budget, _r(), {"compose_budget_ok": NA}, NA),
    (LC.check_evidence_json, _r(evidence=EVIDENCE), {}, PASS),
    (LC.check_evidence_json, _r(evidence={**EVIDENCE, "schema": 2}), {}, FAIL),
    (LC.check_evidence_json, _r(evidence={k: v for k, v in EVIDENCE.items() if k != "panel"}), {}, FAIL),
    (LC.check_evidence_json, _r(evidence={**EVIDENCE, "sources": [{"markdown_path": "A:/x.md"}]}), {}, FAIL),
    (LC.check_evidence_json, _r(evidence={**EVIDENCE, "errors": ["markdown_path missing"]}), {}, PASS),
    (LC.check_evidence_json, _r(evidence=[1]), {}, FAIL),
    (LC.check_evidence_json, _r(), {}, NA),
    (LC.check_metadata_engines, _r(metadata={"engines": {"calls": 3}}), {}, PASS),
    (LC.check_metadata_engines, _r(metadata={"topic": "t"}), {}, FAIL),
    (LC.check_metadata_engines, _r(), {}, FAIL),
    (LC.check_no_failure_text, _r(report="# t\n\nThe immune response was suppressed [1].\n"), {}, PASS),
    (LC.check_no_failure_text, _r(report="- **Class:** `synthesis_no_evidence`\n"), {}, FAIL),
    (LC.check_no_failure_text, _r(report="(no sources fetched; report suppressed)"), {}, FAIL),
    (LC.check_no_failure_text, _r(), {}, NA),
    (LC.check_banner_absent, _r(report="# t\n\nbody\n"), {}, PASS),
    (LC.check_banner_absent, _r(report="> WARNING: **Limited direct evidence:** only\n"), {}, FAIL),
    (LC.check_banner_absent, _r(), {}, NA),
    (LC.check_banner_present, _r(report="> WARNING: **Limited direct evidence:** only\n"), {}, PASS),
    (LC.check_banner_present, _r(report="# t\n\nbody\n"), {}, FAIL),
    (LC.check_banner_present, _r(), {}, NA),
])
def test_each_check(check, run, m, status):
    got, detail = check(run, m)
    assert got == status, detail
    assert isinstance(detail, str) and detail


def test_check_details_name_what_was_measured():
    assert LC.check_run_completed(_r(record={"status": "error", "duration_s": 3.0, "error": "boom"}), {}) == (
        FAIL, "error in 3.0 s (boom)")
    assert LC.check_judged(_r({"panel_verdict": {"judged": 0, "total": 0}, "findings": [{}] * 37,
                               "run_config": GROUNDING_ON}), {}) == (
        FAIL, "judged 0 of total 0 with 37 findings: the panel recorded no judging")
    assert LC.check_evidence_json(_r(evidence={"schema": 1}), {})[1].startswith("missing run, run_config")
    findings = [{"citation_urls": ["u"], "status": "kept"}, {"citation_urls": [], "status": "kept"},
                {"citation_urls": ["u"], "status": "cut"}]
    assert LC.check_cited_findings(_r({"findings": findings}), {}) == (PASS, "1 of 2 kept")
    assert LC.check_few_findings(_r({"findings": findings}), {}) == (PASS, "2 kept")


def test_the_check_list_follows_expect_banner():
    names = [n for n, _ in LC.checks(expect_banner=False)]
    assert names == ["run completed", "fetched > 0", "evidence > 0", "findings with citations > 0",
                     "report.md > 1000 chars", "sources_block_present", "cite_markers_unresolved == 0",
                     "flagged_claims_in_body == 0", "judged == total", "compose_budget_ok",
                     "evidence.json valid", "metadata.json has engines",
                     "no synthesis_no_evidence / report suppressed text", "grounding banner absent"]
    banner = dict(LC.checks(expect_banner=True))
    assert banner["findings <= 3"] is LC.check_few_findings
    assert banner["grounding banner present"] is LC.check_banner_present
    assert "findings with citations > 0" not in banner and "grounding banner absent" not in banner


def test_render_and_exit_code():
    rows = [("fetched > 0", PASS, "28"), ("compose_budget_ok", NA, "field not in this run")]
    assert LC.exit_code(rows) == 0
    out = LC.render(rows)
    assert "| fetched > 0 | PASS | 28 |" in out and "OVERALL: PASS (n/a: compose_budget_ok)" in out
    rows.append(("judged == total", FAIL, "3/4"))
    assert LC.exit_code(rows) == 1
    assert "OVERALL: FAIL (failed: judged == total; n/a: compose_budget_ok)" in LC.render(rows)


# ---------------------------------------------------------------------------
# main: the engines first, one _run_one, the checks decide the exit status
# ---------------------------------------------------------------------------

def _passing_state() -> dict:
    return {
        "user_request": "topic", "length": "medium", "run_config": make_run_config(length="medium"),
        "messages": [{"role": "user", "content": "topic", "ts": "2026-10-19T10:00:00+00:00"}],
        "sources": [], "fetched": [{"url": "https://example.com/a", "title": "A", "fetcher": "native"}],
        "evidence": [{"source_id": 1, "source_url": "https://example.com/a", "relevance": 4,
                      "stance": "supports", "claims": [{"text": "c", "quote": "q", "confidence": "high",
                                                        "quote_verified": True}]}],
        "findings": [{"id": "f0", "claim": "c", "citation_urls": ["https://example.com/a"],
                      "section": "A", "source_ids": [1], "sentence": "c", "status": "kept"}],
        "panel_verdict": {"verdict": "pass", "flagged_finding_ids": [], "judged": 1, "total": 1},
        "compose_runs": 1, "revision_rounds": 0, "model_calls": [], "errors": []}


PASSING_REPORT = ("# topic\n\n## A\n\n" + "LangGraph pauses a run at an interrupt [1]. " * 30
                  + "\n\n## Sources\n\n1. [A](https://example.com/a)\n")


def _write(qdir: Path, *, report: str) -> None:
    qdir.mkdir(parents=True)
    for name, obj in (("state", _passing_state()), ("evidence", EVIDENCE), ("metadata", {"engines": {}}),
                      ("record", {"status": "ok", "duration_s": 245.0, "error": None})):
        (qdir / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")
    (qdir / "report.md").write_text(report, encoding="utf-8")


@pytest.mark.parametrize("report,code", [(PASSING_REPORT, 0),
                                         (PASSING_REPORT.replace("[1]. ", "【1】. ", 1), 1)],
                         ids=["passing", "cjk-marker-fails"])
async def test_main_resolves_the_engines_runs_once_and_exits_by_the_checks(
        monkeypatch, tmp_path, capsys, report, code):
    snapshot = fake_engines_snapshot()
    monkeypatch.setattr(registry, "active_snapshot", AsyncMock(return_value=snapshot))
    _no_reports_root(monkeypatch)
    seen = {}

    async def fake_run_one(query_obj, **kw):
        seen.update(kw, query=query_obj, reports_root=os.environ["ARGUS_REPORTS_ROOT"])
        _write(kw["state_dir"] / query_obj["id"], report=report)
        return {"status": "ok"}

    monkeypatch.setattr(runner, "_run_one", fake_run_one)
    args = live_check.parse_args(["--query", "q010", "--length", "medium", "--recency", "m",
                                  "--out", str(tmp_path)])
    assert await live_check._main(args) == code
    registry.active_snapshot.assert_awaited_once()
    assert seen["engines"] is snapshot
    assert (seen["origin"], seen["overrides"], seen["timeout_s"], seen["length"], seen["depth"]) == (
        "live_check", {"recency": "m"}, None, "medium", None)
    assert seen["state_dir"] == tmp_path
    assert seen["query"]["query"] == "What is the LangGraph interrupt mechanism and how do agents resume?"
    assert seen["reports_root"] == str(tmp_path / "q010" / "reports")      # never the vault
    out = capsys.readouterr().out
    assert ("OVERALL: PASS" if code == 0 else "OVERALL: FAIL (failed: cite_markers_unresolved == 0") in out
    assert "| metric | q010 | mean |" in out                               # the run's bench metrics too


async def test_main_without_a_checkpoint_reports_the_run_and_fails(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(registry, "active_snapshot", AsyncMock(return_value=fake_engines_snapshot()))
    _no_reports_root(monkeypatch)

    async def fake_run_one(query_obj, **kw):
        return {"status": "error", "error": "EngineError('role legacy.cheap: every engine failed')"}

    monkeypatch.setattr(runner, "_run_one", fake_run_one)
    assert await live_check._main(live_check.parse_args(["--out", str(tmp_path)])) == 1
    assert "no checkpoint" in capsys.readouterr().out
