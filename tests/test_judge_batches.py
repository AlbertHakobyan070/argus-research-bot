"""v4 WP-C — batched grounding judging (spec §2.8 R1, Review Focus 6).

The grounding judge runs in batches of at most ``panel.judge_batch_size``
findings, grouped by section; each batch prompt carries exactly the notes
its findings cite (at most 3 claims each, quotes in full up to 250 chars)
and stays within a 24 000-char budget — an oversized batch is halved.
``judged/total`` is recorded; a batch that is skipped, fails or returns
unparseable JSON leaves its findings unjudged — never passed — and the
report says so in the "⚠ Not fact-checked" banner.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from argus import config as config_mod
from argus.engines.llm_openai import DriverError
from argus.graph import nodes as nodes_mod
from argus.graph.panel import GROUNDING_SYSTEM, panel_node
from helpers import fake_llm, make_run_config

BUDGET = 24_000
_FINDING = re.compile(r"^\[(f\d+)\] ", re.M)
_NOTE = re.compile(r"^\[(\d+)\] ", re.M)


def _note(i: int, claims: list[tuple[str, str]]) -> dict:
    return {"source_id": i, "source_url": f"https://s{i}.example/p", "title": f"Source {i}", "sub_qs": [0],
            "relevance": 4, "stance": "supports",
            "claims": [{"text": t, "quote": q, "confidence": "high", "quote_verified": True} for t, q in claims]}


def _findings(sizes: dict[str, int], n_notes: int) -> list[dict]:
    out, k = [], 0
    for section, n in sizes.items():
        for _ in range(n):
            ids = [k % n_notes + 1, (k + 3) % n_notes + 1]
            out.append({"id": f"f{k}", "claim": f"claim {k}", "citation_urls": [], "section": section,
                        "source_ids": ids, "sentence": f"Finding {k} about topic {k} " + "".join(f"[{i}]" for i in ids),
                        "status": "kept"})
            k += 1
    return out


def _state(findings, evidence, **overrides) -> dict:
    fetched = [{"url": f"https://s{i}.example/p", "title": f"Source {i}"} for i in range(1, len(evidence) + 1)]
    sections = list(dict.fromkeys(f["section"] for f in findings))
    return {"thread_id": "tg:1:abcd1234", "user_request": "topic", "length": "medium",
            "run_config": make_run_config("medium", **{"panel.coverage": False, **overrides}),
            "brief": {"sub_questions": [{"q": "q?", "kind": "web"}], "summary": "s"},
            "coverage": {"0": 5}, "fetched": fetched, "evidence": evidence,
            "outline": {"sections": [{"title": s, "sub_qs": [0], "note_ids": []} for s in sections],
                        "changed_sections": []},
            "sections": [{"title": s, "md": f"Body of {s} [1].", "note_ids": []} for s in sections],
            "findings": findings, "model_calls": [], "errors": []}


def _judge(flag=lambda fid: False):
    def reply(system, human):
        assert system == GROUNDING_SYSTEM
        ids = _FINDING.findall(human)
        return json.dumps({"flagged": [{"id": i, "reason": "not in the quote"} for i in ids if flag(i)],
                           "notes": []})
    return reply


def test_every_finding_judged_once_with_its_own_notes(monkeypatch):
    evidence = [_note(i, [(f"note {i} claim {j}", f"quote {i}.{j} " + "q" * 40) for j in range(4)])
                for i in range(1, 9)]
    findings = _findings({"Alpha": 12, "Beta": 7, "Gamma": 4}, n_notes=8)
    seen = fake_llm(monkeypatch, _judge(lambda fid: fid in {"f3", "f20"}))
    out = panel_node(_state(findings, evidence, **{"panel.judge_batch_size": 5}))

    prompts = [m[1] for m in seen]
    batches = [_FINDING.findall(p) for p in prompts]
    judged_ids = [i for b in batches for i in b]
    assert sorted(judged_ids, key=lambda i: int(i[1:])) == [f["id"] for f in findings], "each exactly once"
    assert all(len(b) <= 5 for b in batches), "batch size from run_config"
    by_id = {f["id"]: f for f in findings}
    for b in batches:
        secs = [by_id[i]["section"] for i in b]
        assert secs == sorted(secs, key=["Alpha", "Beta", "Gamma"].index), "grouped by section, in order"
    for prompt, b in zip(prompts, batches):
        cited = {n for i in b for n in by_id[i]["source_ids"]}
        assert {int(n) for n in _NOTE.findall(prompt)} == cited, "exactly the batch's cited notes"
        for n in cited:
            claims = [c for c in evidence[n - 1]["claims"] if c["text"] in prompt]
            assert len(claims) == 3, "three claims per cited note"
            assert all(f'"{c["quote"]}"' in prompt for c in claims), "quotes in full"
        assert len(GROUNDING_SYSTEM) + len(prompt) <= BUDGET
    pv = out["panel_verdict"]
    assert (pv["judged"], pv["total"]) == (23, 23)
    assert pv["flagged_finding_ids"] == ["f3", "f20"] and pv["skipped_judges"] == []


def test_an_oversized_batch_is_halved_until_it_fits(monkeypatch):
    evidence = [_note(i, [("long claim " + "x" * 1400, "q" * 250)] * 3) for i in range(1, 9)]
    findings = _findings({"Alpha": 10}, n_notes=8)
    seen = fake_llm(monkeypatch, _judge())
    out = panel_node(_state(findings, evidence))
    batches = [_FINDING.findall(m[1]) for m in seen]
    assert len(batches) > 1, "a 10-finding batch over the budget was split"
    assert all(len(GROUNDING_SYSTEM) + len(m[1]) <= BUDGET for m in seen)
    assert sorted(i for b in batches for i in b) == sorted(f["id"] for f in findings)
    assert (out["panel_verdict"]["judged"], out["panel_verdict"]["total"]) == (10, 10)


def test_a_single_finding_over_the_budget_is_unjudged(monkeypatch):
    evidence = [_note(1, [("c", "q")])]
    findings = _findings({"Alpha": 2}, n_notes=1)
    findings[1]["sentence"] = "An enormous sentence " + "y" * BUDGET + " [1]"
    seen = fake_llm(monkeypatch, _judge())
    out = panel_node(_state(findings, evidence))
    assert [_FINDING.findall(m[1]) for m in seen] == [["f0"]]
    pv = out["panel_verdict"]
    assert (pv["judged"], pv["total"]) == (1, 2)
    assert pv["skipped_judges"] == ["grounding batch 2: input over the 24000-char budget"]
    assert out["errors"] == ["[r1] skip: panel.grounding batch 2 (input over the 24000-char budget)"]


@pytest.mark.parametrize("failure", ["skipped", "unparseable"])
def test_a_failed_batch_is_unjudged_never_passed(monkeypatch, tmp_path, failure):
    evidence = [_note(i, [(f"claim {i}", f"quote {i}")]) for i in range(1, 5)]
    findings = _findings({"Alpha": 4, "Beta": 4}, n_notes=4)

    def reply(system, human):
        if "[f5]" in human:
            if failure == "skipped":                      # a 503, not a 429: an exhausted quota pauses the run
                raise DriverError("provider fake: 503 overloaded", code=503)
            return "the judge rambles without JSON"
        return _judge(lambda fid: True)(system, human)

    monkeypatch.setattr("argus.llm._sleep", lambda s: None)
    fake_llm(monkeypatch, reply)
    out = panel_node(_state(findings, evidence, **{"panel.judge_batch_size": 4}))
    pv = out["panel_verdict"]
    assert (pv["judged"], pv["total"]) == (4, 8)
    assert pv["flagged_finding_ids"] == ["f0", "f1", "f2", "f3"], "only the judged batch can flag"
    assert len(pv["skipped_judges"]) == 1 and pv["skipped_judges"][0].startswith("grounding batch 2: ")
    skip = [e for e in out["errors"] if e.startswith("[r1] skip: panel.grounding ")]
    assert len(skip) == 1 and "every engine failed" in skip[0]

    monkeypatch.setattr(config_mod, "get_settings", lambda: SimpleNamespace(reports_root=tmp_path))
    monkeypatch.setattr(nodes_mod, "markdown_to_pdf", lambda *a, **k: None)
    state = {**_state(findings, evidence, **{"panel.judge_batch_size": 4}), **out,
             "errors": out["errors"], "model_calls": out["model_calls"]}
    rb = nodes_mod.report_builder_node(state)
    md = Path(rb["report_paths"]["md"]).read_text(encoding="utf-8")
    reason = "503" if failure == "skipped" else "unparseable"
    assert f"> ⚠ Not fact-checked — 4 of 8 findings were not judged ({reason})" in md
    meta = json.loads((Path(rb["report_paths"]["folder"]) / "metadata.json").read_text(encoding="utf-8"))
    assert meta["degraded"] is True


def test_no_batch_judged_means_the_judge_failed(monkeypatch, tmp_path):
    evidence = [_note(1, [("c", "q")])]
    findings = _findings({"Alpha": 2}, n_notes=1)
    monkeypatch.setattr("argus.llm._sleep", lambda s: None)
    fake_llm(monkeypatch, lambda system, human: (_ for _ in ()).throw(
        DriverError("provider fake: 503 overloaded", code=503)))
    out = panel_node(_state(findings, evidence))
    pv = out["panel_verdict"]
    assert (pv["judged"], pv["total"]) == (0, 2)
    assert pv["flagged_finding_ids"] == [] and pv["judge_verdicts"]["grounding"] == "unjudged"

    monkeypatch.setattr(config_mod, "get_settings", lambda: SimpleNamespace(reports_root=tmp_path))
    monkeypatch.setattr(nodes_mod, "markdown_to_pdf", lambda *a, **k: None)
    rb = nodes_mod.report_builder_node({**_state(findings, evidence), **out})
    md = Path(rb["report_paths"]["md"]).read_text(encoding="utf-8")
    assert "> ⚠ Not fact-checked — the grounding judge failed (503)" in md


def test_grounding_off_judges_nothing_and_says_so(monkeypatch, tmp_path):
    seen = fake_llm(monkeypatch, _judge())
    evidence = [_note(1, [("c", "q")])]
    findings = _findings({"Alpha": 2}, n_notes=1)
    out = panel_node(_state(findings, evidence, **{"panel.grounding": False}))
    assert seen == [] and (out["panel_verdict"]["judged"], out["panel_verdict"]["total"]) == (0, 0)
    monkeypatch.setattr(config_mod, "get_settings", lambda: SimpleNamespace(reports_root=tmp_path))
    monkeypatch.setattr(nodes_mod, "markdown_to_pdf", lambda *a, **k: None)
    rb = nodes_mod.report_builder_node({**_state(findings, evidence, **{"panel.grounding": False}), **out})
    md = Path(rb["report_paths"]["md"]).read_text(encoding="utf-8")
    assert "> ⚠ Not fact-checked — the grounding judge is off" in md
    meta = json.loads((Path(rb["report_paths"]["folder"]) / "metadata.json").read_text(encoding="utf-8"))
    assert meta["degraded"] is False, "the judge was switched off, not lost"


@pytest.mark.parametrize("entry", [{"id": "[f1]"}, {"id": " f1 "}, {"id": "F1"}, {"id": 1}, {"id": "1"},
                                   {"finding_id": "f1"}, "[f1]", "f1"])
def test_the_judges_ids_are_normalised(monkeypatch, entry):
    evidence = [_note(1, [("c", "q")])]
    findings = _findings({"Alpha": 2}, n_notes=1)
    item = {**entry, "reason": "not in the quote"} if isinstance(entry, dict) else entry
    fake_llm(monkeypatch, lambda system, human: json.dumps({"flagged": [item], "notes": []}))
    pv = panel_node(_state(findings, evidence))["panel_verdict"]
    assert pv["flagged_finding_ids"] == ["f1"] and (pv["judged"], pv["total"]) == (2, 2)


def test_a_reply_naming_no_finding_of_its_batch_is_unparseable_and_unjudged(monkeypatch, tmp_path):
    evidence = [_note(1, [("c", "q")])]
    findings = _findings({"Alpha": 2}, n_notes=1)
    seen = fake_llm(monkeypatch, lambda system, human: json.dumps({"flagged": [{"id": "f7", "reason": "x"}],
                                                                   "notes": []}))
    out = panel_node(_state(findings, evidence))
    pv = out["panel_verdict"]
    assert (pv["judged"], pv["total"]) == (0, 2) and pv["flagged_finding_ids"] == [], "never read as a pass"
    assert len(seen) == 2 and seen[1][-1] == "Reply with only the JSON object."
    monkeypatch.setattr(config_mod, "get_settings", lambda: SimpleNamespace(reports_root=tmp_path))
    monkeypatch.setattr(nodes_mod, "markdown_to_pdf", lambda *a, **k: None)
    rb = nodes_mod.report_builder_node({**_state(findings, evidence), **out})
    md = Path(rb["report_paths"]["md"]).read_text(encoding="utf-8")
    assert "> ⚠ Not fact-checked — the grounding judge failed (unparseable)" in md


def test_grounding_is_counted_over_its_last_panel_pass(monkeypatch, tmp_path):
    evidence = [_note(i, [(f"claim {i}", f"quote {i}")]) for i in range(1, 5)]
    findings = _findings({"Alpha": 4, "Beta": 4}, n_notes=4)
    down = {"on": True}

    def reply(system, human):
        if down["on"]:
            raise DriverError("provider fake: 503 overloaded", code=503)
        return _judge()(system, human)

    monkeypatch.setattr("argus.llm._sleep", lambda s: None)
    fake_llm(monkeypatch, reply)
    state = _state(findings, evidence, **{"panel.judge_batch_size": 4})
    first = panel_node(state)
    assert first["panel_verdict"]["judged"] == 0 and len(first["panel_verdict"]["skipped_judges"]) == 2
    down["on"] = False                                     # judged after a gap wave
    state = {**state, **first}
    second = panel_node(state)
    assert (second["panel_verdict"]["judged"], second["panel_verdict"]["total"]) == (8, 8)
    state = {**state, **second, "errors": first["errors"] + second["errors"],
             "model_calls": first["model_calls"] + second["model_calls"]}
    monkeypatch.setattr(config_mod, "get_settings", lambda: SimpleNamespace(reports_root=tmp_path))
    monkeypatch.setattr(nodes_mod, "markdown_to_pdf", lambda *a, **k: None)
    rb = nodes_mod.report_builder_node(state)
    md = Path(rb["report_paths"]["md"]).read_text(encoding="utf-8")
    assert "⚠ Degraded run" not in md and "⚠ Not fact-checked" not in md
    assert "skip: panel.grounding batch 1" in md.split("## Degradation notes")[1], "the first pass's skips: notes"


def test_the_judge_events(monkeypatch):
    from langgraph.runtime import Runtime
    evidence = [_note(1, [("c", "q")])]
    findings = _findings({"Alpha": 2}, n_notes=1)
    fake_llm(monkeypatch, _judge(lambda fid: fid == "f1"))
    events: list = []
    panel_node(_state(findings, evidence), Runtime(stream_writer=events.append))
    assert [e for e in events if e["kind"] == "judge"] == [
        {"kind": "judge", "node": "panel", "data": {"name": "grounding", "batch": 1, "verdict": "1 flagged",
                                                     "skipped": False}}]