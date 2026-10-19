"""v4 WP-C — graph/precision.py: the deterministic uncited-specifics scan (spec §2.8).

``scan(md)`` lists the sentences that state a number, amount, year or date
without an ``[n]`` citation. It feeds the report's "## Uncited specifics"
appendix and ``panel_verdict.uncited_specifics`` only: the body is never
marked inline (the scan is a heuristic).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from argus import config as config_mod
from argus.graph import nodes as nodes_mod
from argus.graph.panel import panel_node
from argus.graph.precision import scan
from helpers import fake_llm, make_run_config


def test_scan_finds_uncited_sentences_with_numbers_amounts_or_dates():
    md = ("Revenue grew 40% in 2024. It was confirmed by the filing [1]. Prices hit $5 per unit [2]. "
          "The fee is €3 for members. Nothing numeric here.\n\n"
          "- The plan costs 12 dollars.\n- Cited item 7 [3].\n\n"
          "Revenue grew 40% in 2024.")
    assert scan(md) == ["Revenue grew 40% in 2024.", "The fee is €3 for members.", "The plan costs 12 dollars."]


def test_scan_skips_headings_code_and_tables():
    md = ("### Results for 2024\n\n| year | value |\n|---|---|\n| 2024 | 7 |\n\n"
          "```\nx = 42\n```\n\nPlain prose without figures.")
    assert scan(md) == []


def test_a_list_item_is_reported_without_its_marker():
    assert scan("1. The model has 7 billion parameters.\n2. It is open [1].") == [
        "The model has 7 billion parameters."]


FETCHED = [{"url": "https://a.example/p", "title": "A"}, {"url": "https://b.example/p", "title": "B"}]
BRIEF = {"sub_questions": [{"q": "What is X?", "kind": "web"}], "summary": "s"}
BODY = "X shipped in 2024 to 3 regions. X is a graph runtime [1]. X pauses at interrupts [2]."


def _state(**over) -> dict:
    state = {
        "thread_id": "tg:1:abcd1234", "user_request": "What is X?", "length": "medium",
        "run_config": make_run_config("medium"), "brief": BRIEF, "fetched": FETCHED, "evidence": [],
        "outline": {"sections": [{"title": "Overview", "sub_qs": [0], "note_ids": [1, 2]}],
                    "changed_sections": ["Overview"]},
        "sections": [{"title": "Overview", "md": BODY, "note_ids": [1, 2]}],
        "findings": [
            {"id": "f0", "claim": "X is a graph runtime", "citation_urls": [FETCHED[0]["url"]],
             "section": "Overview", "source_ids": [1], "sentence": "X is a graph runtime [1].", "status": "kept"},
            {"id": "f1", "claim": "X pauses at interrupts", "citation_urls": [FETCHED[1]["url"]],
             "section": "Overview", "source_ids": [2], "sentence": "X pauses at interrupts [2].", "status": "kept"}],
        "panel_verdict": {"verdict": "pass", "judged": 2, "total": 2},
        "model_calls": [], "errors": []}
    state.update(over)
    return state


def _report(monkeypatch, tmp_path, state) -> tuple[dict, str]:
    monkeypatch.setattr(config_mod, "get_settings", lambda: SimpleNamespace(reports_root=tmp_path))
    monkeypatch.setattr(nodes_mod, "markdown_to_pdf", lambda *a, **k: None)
    out = nodes_mod.report_builder_node(state)
    return out, Path(out["report_paths"]["md"]).read_text(encoding="utf-8")


def test_the_report_lists_them_in_an_appendix_and_never_marks_the_body(monkeypatch, tmp_path):
    _, md = _report(monkeypatch, tmp_path, _state())
    body, appendix = md.split("## Uncited specifics", 1)
    assert "X shipped in 2024 to 3 regions." in body, "the sentence stays in the body"
    assert "[uncited]" not in md and "uncited]" not in body
    assert "- X shipped in 2024 to 3 regions." in appendix.split("## Sources")[0]
    assert md.index("## Uncited specifics") < md.index("## Sources")


def test_precision_off_means_no_appendix(monkeypatch, tmp_path):
    state = _state(run_config=make_run_config("medium", **{"panel.precision_mode": "off"}))
    _, md = _report(monkeypatch, tmp_path, state)
    assert "## Uncited specifics" not in md


def test_the_panel_records_uncited_specifics(monkeypatch):
    seen = fake_llm(monkeypatch, lambda system, human: '{"flagged": [], "notes": []}')
    out = panel_node(_state(run_config=make_run_config("medium", **{"panel.coverage": False})))
    assert out["panel_verdict"]["uncited_specifics"] == ["X shipped in 2024 to 3 regions."]
    assert len(seen) == 1, "one grounding batch, no precision LLM call"
    off = panel_node(_state(run_config=make_run_config(
        "medium", **{"panel.coverage": False, "panel.precision_mode": "off"})))
    assert off["panel_verdict"]["uncited_specifics"] == []
    assert json.dumps(off)                              # plain data