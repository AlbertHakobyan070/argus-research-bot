"""Worked numbers (graph/calc.py): the model proposes calculations, the code
checks every input against the finding it cites (or takes it as a labelled
assumption), evaluates the formula without ``eval`` and report_builder renders
the kept ones — a calculation whose figure was cut goes with it.

Run with:
    PYTHONPATH='' ./venv/Scripts/python.exe -m pytest tests/test_calc.py -q
"""
from __future__ import annotations

import json
import math

import pytest

import test_cut_not_flag as cnf
from argus.graph import calc
from argus.graph.insight import analyst_node

from helpers import fake_llm, make_run_config

FINDINGS = [
    {"id": "f0", "claim": "The agentic research platform market is $952 million in 2025.", "source_ids": [2],
     "section": "Market", "status": "kept"},
    {"id": "f1", "claim": "It grows at a 30.3% CAGR to 2032.", "source_ids": [2, 5], "section": "Market",
     "status": "kept"},
    {"id": "f2", "claim": "Exa charges 5 USD per 1,000 searches.", "source_ids": [4], "section": "Costs",
     "status": "kept"},
    {"id": "f3", "claim": "Telegram bots are free to host on Telegram's side.", "source_ids": [6],
     "section": "Costs", "status": "kept"},
]
PROPOSAL = {"calculations": [
    {"label": "Market size in 2028", "formula": "size * (1 + cagr) ** 3", "unit": "USD",
     "inputs": [{"name": "size", "value": 952000000, "unit": "USD", "finding": "f0"},
                {"name": "cagr", "value": 0.303, "unit": "per year", "finding": "f1"}],
     "note": "three more years at the stated growth rate"},
    {"label": "Search cost per report", "formula": "price / 1000 * searches", "unit": "USD",
     "inputs": [{"name": "price", "value": 5, "unit": "USD", "finding": "f2"},
                {"name": "searches", "value": 40, "unit": "searches", "assumption": "a long report's searches"}],
     "note": ""},
    {"label": "Made-up figure", "formula": "x * 2",
     "inputs": [{"name": "x", "value": 777, "finding": "f0"}]},
    {"label": "Sneaky", "formula": "__import__('os').system('echo hi')", "inputs": [
        {"name": "a", "value": 1, "assumption": "one"}]},
]}


# --- the evaluator ----------------------------------------------------------------------------

def test_evaluate_does_arithmetic_over_names_and_a_few_functions():
    assert calc.evaluate("a * (1 + g) ** 2 - 5 % 3", {"a": 100, "g": 0.1}) == pytest.approx(119.0)
    assert calc.evaluate("round(max(a, b) / 3, 2) + abs(-1)", {"a": 10, "b": 4}) == pytest.approx(4.33)


@pytest.mark.parametrize("formula", ["__import__('os').system('x')", "a.real", "[a]", "a if a else 1", "lambda: 1",
                                     "b * 2", "2 ** 1000", "1 / 0", "True + 1", "max(a, key=1)", "open('f')"])
def test_evaluate_refuses_anything_but_arithmetic(formula):
    with pytest.raises(ValueError):
        calc.evaluate(formula, {"a": 3})


def test_stated_numbers_read_scale_words_percentages_and_thousands():
    s = calc.stated_numbers("$952 million in 2025, a 30.3% CAGR, 1,200 users and 6.49bn later")
    for v in (952, 9.52e8, 2025, 30.3, 0.303, 1200, 6.49, 6.49e9):
        assert any(math.isclose(v, x, rel_tol=1e-9) for x in s), v


def test_wants_numbers():
    assert calc.wants_numbers("Research the profitability … with calculations")
    assert calc.wants_numbers("What would hosting it cost per month?")
    assert not calc.wants_numbers("How does LangGraph pause a graph?")


# --- checking a proposal ----------------------------------------------------------------------

def test_check_keeps_cited_figures_and_labelled_assumptions_and_drops_the_rest():
    by_id = {f["id"]: f for f in FINDINGS}
    good, why = calc.check(PROPOSAL["calculations"][0], by_id)
    assert why == "" and good["result"] == pytest.approx(952e6 * 1.303 ** 3)
    assert [i["source_ids"] for i in good["inputs"]] == [[2], [2, 5]]
    mixed, _ = calc.check(PROPOSAL["calculations"][1], by_id)
    assert mixed["result"] == pytest.approx(0.2) and mixed["inputs"][1]["assumption"] == "a long report's searches"
    assert calc.check(PROPOSAL["calculations"][2], by_id) == (None, "input x: 777 is not stated in finding f0")
    none, why = calc.check(PROPOSAL["calculations"][3], by_id)
    assert none is None and "not allowed" in why
    assert calc.check({"label": "x", "formula": "a", "inputs": [{"name": "a", "value": 1}]}, by_id)[1] == \
        "input a: neither a finding nor an assumption"


# --- the node -------------------------------------------------------------------------------

def _state(request, findings=FINDINGS, **over):
    rc = make_run_config("medium", **{"compose.analyst": False, **over})
    return {"user_request": request, "run_config": rc, "findings": [dict(f) for f in findings],
            "brief": {"sub_questions": [{"q": "q?"}]}}


def test_the_analyst_node_adds_checked_worked_numbers(monkeypatch):
    seen = fake_llm(monkeypatch, lambda system, human: json.dumps(PROPOSAL))
    out = analyst_node(_state("Estimate the market and the cost per report, with calculations"))
    assert [c["label"] for c in out["calculations"]] == ["Market size in 2028", "Search cost per report"]
    assert seen[0][0] == calc.CALC_SYSTEM and "f0: The agentic research platform market" in seen[0][1]
    assert "f3" not in seen[0][1]                                # a finding stating no number is not offered
    assert [c["role"] for c in out["model_calls"]] == ["compose.calc"]
    assert any("dropped 'Made-up figure' — input x: 777 is not stated" in e for e in out["errors"])


@pytest.mark.parametrize("request_, over", [("How does LangGraph pause a graph?", {}),
                                            ("Estimate the cost per report", {"compose.calculations": "off"})])
def test_no_calculations_unless_asked_and_on(monkeypatch, request_, over):
    seen = fake_llm(monkeypatch, lambda system, human: pytest.fail("no calc call"))
    out = analyst_node(_state(request_, **over))
    assert "calculations" not in out and seen == []


def test_a_run_whose_snapshot_predates_the_role_skips_with_a_note(monkeypatch):
    fake_llm(monkeypatch, lambda system, human: pytest.fail("no calc call"))
    state = _state("Estimate the cost per report")
    del state["run_config"]["engines"]["roles"][calc.ROLE]
    out = analyst_node(state)
    assert "calculations" not in out
    assert out["errors"] == ["[r1] calc: skipped — this run's engines snapshot has no compose.calc binding"]


def test_a_degraded_calc_omits_the_section_visibly(monkeypatch):
    fake_llm(monkeypatch, lambda system, human: "not json at all")
    out = analyst_node(_state("Estimate the cost per report"))
    assert "calculations" not in out
    assert any(e.startswith("[r1] degrade: compose.calc → omit_worked_numbers") for e in out["errors"])


# --- the report ---------------------------------------------------------------------------

def test_render_appendix_tables_the_kept_and_drops_a_calculation_whose_figure_was_cut():
    by_id = {f["id"]: f for f in FINDINGS}
    calcs = [calc.check(c, by_id)[0] for c in PROPOSAL["calculations"][:2]]
    md, dropped = calc.render_appendix(calcs, {"f0", "f2", "f3"})        # f1 (the CAGR) was cut
    assert dropped == ["Market size in 2028"]
    assert md.startswith("## Worked numbers") and "Search cost per report" in md and "Market size" not in md
    assert "price = 5 USD [4]; searches = 40 searches (assumption: a long report's searches)" in md
    assert "**0.2** USD" in md
    assert calc.render_appendix(calcs[:1], {"f0"}) == ("", ["Market size in 2028"])


def test_report_builder_places_worked_numbers_and_cites_their_sources(monkeypatch, tmp_path):
    calcs = [{"label": "Kept one", "formula": "a * 2", "unit": "USD", "note": "doubling", "result": 4.0,
              "inputs": [{"name": "a", "value": 2.0, "unit": "USD", "finding": "f0", "source_ids": [1]}]},
             {"label": "Lost one", "formula": "b", "unit": "", "note": "", "result": 1.0,
              "inputs": [{"name": "b", "value": 1.0, "unit": "", "finding": "f1", "source_ids": [2]}]}]
    out, md, _, evidence, _ = cnf._report(monkeypatch, tmp_path, cnf._state(calculations=calcs))
    assert "## Worked numbers" in md and md.index("## Usage") < md.index("## Worked numbers") < md.index("## Sources")
    assert "Kept one" in md and "Lost one" not in md.split("## Degradation notes")[0]
    assert "[r1] calc: dropped 'Lost one' — a figure it uses was cut" in out["errors"]
    assert [c["label"] for c in evidence["calculations"]] == ["Kept one", "Lost one"]
