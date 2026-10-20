"""v4 WP-C — cut-not-flag (spec §2.8 B1) and the report_builder around it.

A flagged finding is located in its section (or the analysis) by its
verbatim ``sentence``: normalised containment first, else a ``difflib``
ratio ≥ 0.85 over a window of 1–3 adjacent sentences. Whole sentences are
cut, never a fragment; a paragraph or list item left empty is dropped. A
finding whose citations are all unknown ids is cut without a judge. A flag
that cannot be located stays in the body, is listed as such and counts as
in-body for the metric. Around it: the deterministic TL;DR after the cuts,
the "⚠ Not fact-checked" / "⚠ Degraded run" banners, the title-block lines,
the degradation notes by round and ``metadata.json``.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from argus import config as config_mod
from argus.graph import nodes as nodes_mod
from argus.graph.insight import ANALYSIS_LABEL
from argus.quarantine import cut, locate, sentences
from bench import metrics
from helpers import make_run_config

# The live probe's f6 (q010 medium, 2026-10-18): a condensed restatement of two
# adjacent sentences (ratio 0.94 over the pair, 0.58 / 0.68 on either alone).
F6_PARAGRAPH = (
    "The architectural integration occurs at the runtime level: LangGraph’s execution engine treats "
    "`GraphInterrupt` as a first-class control signal, distinct from exceptions. When thrown, it triggers a "
    "state transition where the graph’s current node is marked as paused, and the `resume` value is associated "
    "with the task’s execution context. Upon resumption, the graph’s scheduler uses the `Command` object to "
    "inject the `resume` value back into the node’s input stream, effectively continuing execution as if the "
    "interruption never occurred—though only for the interrupted node and its dependent branches [3]. This "
    "design ensures partial state preservation: while prior nodes retain their outputs, subsequent nodes are "
    "re-evaluated with the updated context, avoiding redundant recomputation of unchanged steps [2]. The "
    "mechanism does not support nested interruptions or hierarchical rollback (e.g., resuming subgraphs from "
    "arbitrary points), but it enables atomic interruption of individual nodes or linear workflow segments.")
F6_SENTENCE = (
    "When thrown, `GraphInterrupt` triggers a state transition where the graph’s current node is marked as "
    "paused, and the `resume` value is associated with the task’s execution context; upon resumption, the "
    "scheduler injects the `resume` value back into the node’s input stream, effectively continuing execution "
    "as if the interruption never occurred—though only for the interrupted node and its dependent branches.")


# ---------------------------------------------------------------------------
# The locator and the cutter
# ---------------------------------------------------------------------------

def test_containment_locates_the_sentence():
    md = "Alpha holds [1]. Beta was measured at 4.01 points [2]. Gamma follows."
    assert locate("beta was measured at 4.01 points", md) == [1]
    assert locate("Beta was measured at 4.01 points [2].", md) == [1]


def test_the_probe_f6_is_located_over_a_two_sentence_window():
    idx = locate(F6_SENTENCE, F6_PARAGRAPH)
    assert idx == [1, 2]
    out = cut(F6_PARAGRAPH, idx)
    assert out.startswith("The architectural integration occurs") and "When thrown" not in out
    assert "Upon resumption" not in out and "This design ensures partial state preservation" in out


def test_an_unrelated_text_is_not_located():
    assert locate("Quantum lattices grow on trees in the spring", F6_PARAGRAPH) is None
    assert locate("", F6_PARAGRAPH) is None


def test_cuts_take_whole_sentences_and_leave_no_fragment():
    md = ("Intro sentence [1]. GLM-5.2 is the latest flagship LLM from Z.ai with a 1M-token context "
          "window [8]. Closing sentence [2].")
    idx = locate("latest flagship LLM from Z.ai", md)              # a fragment of the sentence
    assert idx == [1]
    assert cut(md, idx) == "Intro sentence [1]. Closing sentence [2]."


def test_a_paragraph_or_item_left_empty_is_dropped_and_its_heading_with_it():
    md = ("First paragraph [1].\n\nOnly sentence here [2].\n\n- Item one [1].\n- Item two [2].\n\n"
          "### Details\n\nThe detail sentence [3].\n\n### Kept\n\nKept text [1].")
    texts = sentences(md)
    drop = [texts.index(t) for t in ("Only sentence here [2].", "Item two [2].", "The detail sentence [3].")]
    assert cut(md, drop) == "First paragraph [1].\n\n- Item one [1].\n\n### Kept\n\nKept text [1]."


def test_an_untouched_body_stays_byte_identical():
    md = "Line one [1].\nLine two continues it.\n\n\n```\ncode. Not prose.\n```\n| a | b |\n"
    assert cut(md, []) == md


def test_headings_code_and_tables_hold_no_cuttable_sentence():
    md = "### A heading.\n\n```\nx = 1. y = 2.\n```\n\n| t. | u. |\n\nProse one. Prose two."
    assert sentences(md) == ["Prose one.", "Prose two."]


def test_abbreviations_and_decimals_do_not_split_a_sentence():
    md = "The U.S. agency, e.g. the FDA, cited 4.5 points vs. 3.2 [1]. Dr. Smith agreed [2]."
    assert sentences(md) == ["The U.S. agency, e.g. the FDA, cited 4.5 points vs. 3.2 [1].", "Dr. Smith agreed [2]."]


# ---------------------------------------------------------------------------
# report_builder: the cuts, the appendices and the metric
# ---------------------------------------------------------------------------

FETCHED = [{"url": f"https://s{i}.example/p", "title": f"Source {i}"} for i in range(1, 4)]
BRIEF = {"sub_questions": [{"q": "How does it pause?", "kind": "web"}, {"q": "Who uses it?", "kind": "web"}],
         "summary": "s"}
MECHANISM = ("LangGraph pauses a run at an interrupt [1]. The pause keeps the state in a checkpoint [2]. "
             "Some say it was released in 1999 [9].\n\n" + F6_PARAGRAPH)
USAGE = ("- Teams use it for approvals [1].\n- It powers review loops [2].\n\n"
         "### Details\n\nThe resume value is passed back with a command [3].")


def _f(fid, section, sentence, ids, claim=None):
    return {"id": fid, "claim": claim or sentence.rstrip("."), "citation_urls": [], "confidence": "high",
            "section": section, "source_ids": ids, "sentence": sentence, "status": "kept"}


FINDINGS = [
    _f("f0", "Mechanism", "LangGraph pauses a run at an interrupt [1].", [1]),
    _f("f1", "Mechanism", "The pause keeps the state in a checkpoint [2].", [2]),
    _f("f2", "Mechanism", "Some say it was released in 1999 [9].", [9]),
    _f("f3", "Mechanism", F6_SENTENCE, [3], claim="GraphInterrupt pauses the node; resume re-injects the value"),
    _f("f4", "Usage", "It powers review loops [2].", [2]),
    _f("f5", "Usage", "A claim the body never states [1].", [1]),
    _f("f6", "Usage", "The resume value is passed back with a command [3].", [3]),
    _f("f7", "Usage", "Teams use it for approvals [1].", [1]),
]
FLAGGED = ["f1", "f3", "f4", "f5", "f6"]


def _state(**over) -> dict:
    state = {
        "thread_id": "tg:1:abcd1234", "user_request": "How does LangGraph pause?", "length": "medium",
        "run_config": make_run_config("medium", **{"compose.analyst": False}), "brief": BRIEF,
        "fetched": FETCHED, "evidence": [], "coverage": {"0": 2, "1": 3},
        "outline": {"sections": [{"title": "Mechanism", "sub_qs": [0], "note_ids": [1, 2, 3]},
                                 {"title": "Usage", "sub_qs": [1], "note_ids": [1, 2, 3]}],
                    "changed_sections": []},
        "sections": [{"title": "Mechanism", "md": MECHANISM, "note_ids": [1, 2, 3]},
                     {"title": "Usage", "md": USAGE, "note_ids": [1, 2, 3]}],
        "findings": [dict(f) for f in FINDINGS],
        "panel_verdict": {"verdict": "pass", "flagged_finding_ids": list(FLAGGED), "judged": 8, "total": 8,
                          "judge_verdicts": {"grounding": "revise",
                                             "flag_reasons": {i: f"reason {i}" for i in FLAGGED}}},
        "model_calls": [], "errors": [], "gap_waves": 1, "compose_runs": 2, "revision_rounds": 0,
        "search_spend": {"exa_usd": 0.042, "recency": None}}
    state.update(over)
    return state


def _report(monkeypatch, tmp_path, state, *, pdf=lambda *a, **k: None):
    monkeypatch.setattr(config_mod, "get_settings", lambda: SimpleNamespace(reports_root=tmp_path))
    monkeypatch.setattr(nodes_mod, "markdown_to_pdf", pdf)
    out = nodes_mod.report_builder_node(state)
    folder = Path(out["report_paths"]["folder"])
    md = (folder / "report.md").read_text(encoding="utf-8")
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    evidence = json.loads(Path(out["report_paths"]["evidence"]).read_text(encoding="utf-8"))
    run = {"qid": "t", "state": {**state, **out, "errors": state["errors"] + out.get("errors", [])},
           "evidence": evidence, "report": md, "metadata": meta, "record": None, "summary": None}
    return out, md, meta, evidence, run


def test_flagged_and_unknown_id_findings_are_cut_and_unlocated_ones_listed(monkeypatch, tmp_path):
    out, md, _, evidence, run = _report(monkeypatch, tmp_path, _state())
    status = {f["id"]: f["status"] for f in out["findings"]}
    assert status == {"f0": "kept", "f1": "cut", "f2": "cut", "f3": "cut", "f4": "cut", "f5": "unlocated",
                      "f6": "cut", "f7": "kept"}
    assert [f["status"] for f in evidence["findings"]] == list(status.values())
    body = md.split("## ⚠ Flagged claims")[0]
    for gone in ("The pause keeps the state", "released in 1999", "When thrown", "Upon resumption",
                 "It powers review loops", "The resume value is passed back", "### Details"):
        assert gone not in body
    assert "LangGraph pauses a run at an interrupt [1]." in body and "- Teams use it for approvals [1]." in body
    assert "This design ensures partial state preservation" in body
    cut_part = md.split("## ⚠ Flagged claims (removed from the body)")[1].split("## ⚠ Flagged but not located")[0]
    assert "The pause keeps the state in a checkpoint [2]. — reason f1" in cut_part
    assert "Some say it was released in 1999. — unknown source id (cites 9)" in cut_part
    assert "When thrown, it triggers a state transition" in cut_part and "reason f3" in cut_part
    located = md.split("## ⚠ Flagged but not located (still in body)")[1].split("## Sources")[0]
    assert "A claim the body never states [1] — reason f5" in located
    assert metrics.flagged_claims_in_body(run) == 1, "the unlocated flag counts as in-body"
    assert metrics.cite_markers_unresolved(run) == 0, "the unknown [9] left with its sentence"
    sources = md.split("## Sources")[1].split("\n---")[0]          # the block, not the footer's timestamp
    assert re.findall(r"^(\d+)\. ", sources, re.M) == ["1", "2", "3"], "no Sources entry for the unknown 9"


def test_a_section_emptied_by_the_cuts_is_dropped(monkeypatch, tmp_path):
    state = _state(sections=[{"title": "Mechanism", "md": MECHANISM, "note_ids": [1]},
                             {"title": "Lonely", "md": "Only a flagged sentence here [2].", "note_ids": [2]}],
                   findings=[dict(FINDINGS[0]), _f("f1", "Lonely", "Only a flagged sentence here [2].", [2])],
                   panel_verdict={"verdict": "pass", "flagged_finding_ids": ["f1"], "judged": 2, "total": 2})
    _, md, *_ = _report(monkeypatch, tmp_path, state)
    assert "## Lonely" not in md and "## Mechanism" in md


def test_the_tldr_is_built_after_the_cuts_from_the_first_kept_finding_of_each_section(monkeypatch, tmp_path):
    out, md, _, evidence, _ = _report(monkeypatch, tmp_path, _state())
    tldr = ("## TL;DR\n\n_TL;DR — compiled from the report's own cited, judged findings_\n\n"
            "- LangGraph pauses a run at an interrupt [1].\n- Teams use it for approvals [1].")
    assert tldr in md and md.index("## TL;DR") < md.index("## Mechanism")
    assert out["tldr_md"] == tldr.split("\n\n", 1)[1]
    assert evidence["tldr"] == {"source": "sections", "items": ["LangGraph pauses a run at an interrupt [1].",
                                                                "Teams use it for approvals [1]."]}


ANALYSIS_MD = ("_Analysis — interpretation by Argus._\n\n### Key takeaways\n\n"
               "- It pauses at interrupts [1] — confidence **high** (2 independent sources)\n"
               "- It was built in 1999 [2] — confidence **low** (1 independent source)\n"
               "- It resumes with a command [3] — confidence **medium** (1 independent source)")
ANALYSIS = {"md": ANALYSIS_MD, "stance_map": {}, "has_disagreement_section": False, "has_scenarios": False,
            "archetype": "explainer",
            "takeaways": [{"text": "It pauses at interrupts [1]", "source_ids": [1], "confidence": "high",
                           "reasons": ["2 independent sources"]},
                          {"text": "It was built in 1999 [2]", "source_ids": [2], "confidence": "low",
                           "reasons": ["1 independent source"]},
                          {"text": "It resumes with a command [3]", "source_ids": [3], "confidence": "medium",
                           "reasons": ["1 independent source"]}]}


def test_the_tldr_takes_the_kept_takeaways_when_the_analyst_ran(monkeypatch, tmp_path):
    findings = [dict(FINDINGS[0]), _f("f1", "Analysis", "It was built in 1999 [2]", [2]),
                _f("f2", "Analysis", "It pauses at interrupts [1]", [1])]
    state = _state(run_config=make_run_config("medium"), analysis=ANALYSIS, findings=findings,
                   panel_verdict={"verdict": "pass", "flagged_finding_ids": ["f1"], "judged": 3, "total": 3})
    out, md, _, evidence, _ = _report(monkeypatch, tmp_path, state)
    assert ("- It pauses at interrupts [1] — confidence **high**\n"
            "- It resumes with a command [3] — confidence **medium**") in out["tldr_md"]
    assert "built in 1999" not in out["tldr_md"] and "built in 1999" not in md.split("## ⚠ Flagged claims")[0]
    assert evidence["tldr"]["source"] == "takeaways"
    assert md.index("## Usage") < md.index("## Analysis") < md.index("## Sources")


def test_the_tldr_is_omitted_and_recorded_when_nothing_is_kept(monkeypatch, tmp_path):
    findings = [dict(FINDINGS[1]), dict(FINDINGS[4])]
    state = _state(findings=findings,
                   panel_verdict={"verdict": "pass", "flagged_finding_ids": ["f1", "f4"], "judged": 2, "total": 2})
    out, md, _, evidence, _ = _report(monkeypatch, tmp_path, state)
    assert "## TL;DR" not in md and out["tldr_md"] is None
    assert "[r1] tldr: omitted — no kept takeaway or finding" in out["errors"]
    assert evidence["tldr"] == {"source": "omitted", "items": []}
    assert "- [r1] tldr: omitted — no kept takeaway or finding" in md.split("## Degradation notes")[1]


def test_the_lecture_tldr_heading(monkeypatch, tmp_path):
    state = _state(length="lecture", run_config=make_run_config("lecture", **{"compose.analyst": False}))
    _, md, *_ = _report(monkeypatch, tmp_path, state)
    assert "## Executive TL;DR\n\n_TL;DR — compiled" in md


# ---------------------------------------------------------------------------
# Banners, the degraded mark, the title block and the degradation notes
# ---------------------------------------------------------------------------

def _digests(ok: int, skipped: int) -> tuple[list[dict], list[str]]:
    calls = [{"role": "research.digest", "tier": "cheap", "profile": "fake", "provider": "fake",
              "requested_model": "m", "served_model": "m", "routed_via": "", "ok": True, "item": f"https://o{i}"}
             for i in range(ok)]
    errors = [f"[r1] skip: research.digest https://x{i}.example/p (role research.digest: every engine failed "
              "after 3 attempts (last: DriverError: provider freellmapi: 429 Too Many Requests))"
              for i in range(skipped)]
    return calls, errors


def test_the_degraded_run_banner_at_half_the_items(monkeypatch, tmp_path):
    calls, errors = _digests(14, 14)
    _, md, meta, evidence, _ = _report(monkeypatch, tmp_path, _state(model_calls=calls, errors=errors))
    assert "> ⚠ Degraded run — 14 of 28 sources could not be digested (429)" in md
    assert md.index("⚠ Degraded run") < md.index("## TL;DR")
    assert meta["degraded"] is True and evidence["degraded"] is True


def test_below_half_is_not_a_degraded_run(monkeypatch, tmp_path):
    calls, errors = _digests(15, 13)
    _, md, meta, *_ = _report(monkeypatch, tmp_path, _state(model_calls=calls, errors=errors))
    assert "⚠ Degraded run" not in md and "⚠ Not fact-checked" not in md
    assert meta["degraded"] is False


def test_metadata_carries_the_run_provenance(monkeypatch, tmp_path):
    calls, errors = _digests(15, 13)
    out, _, meta, evidence, _ = _report(monkeypatch, tmp_path, _state(model_calls=calls, errors=errors))
    assert meta["evidence_path"] == out["report_paths"]["evidence"]
    assert meta["exa_usd"] == 0.042 and meta["engines"]["exa_usd"] == 0.042
    assert meta["run_config"]["depth"] == "standard" and "roles" not in meta["run_config"]["engines"]
    assert meta["degradations"] == errors
    assert meta["engines"]["by_role"]["research.digest"]["skipped"] == 13


def test_the_title_block_lines(monkeypatch, tmp_path):
    writer = {"role": "compose.section", "tier": "strong", "profile": "fake", "provider": "fake",
              "requested_model": "m", "served_model": "m", "routed_via": "", "ok": True}
    state = _state(model_calls=[writer], search_spend={"exa_usd": 0.0, "recency": {"window": "m",
                                                                                     "start_date": "2026-09-20"}})
    _, md, *_ = _report(monkeypatch, tmp_path, state)
    head = md.split("---")[0]
    assert "_Engines: fake · reroute 0 % · judge diversity ok · judged 8/8 · warnings 0_" in head
    assert "_Recency: past month (sources published since 2026-09-20)_" in head
    assert "_Research: 3 sources · coverage min 2 / mean 2.5 · gap waves 1 · compose runs 2_" in head


def test_the_title_block_says_when_the_judge_shares_the_writers_family(monkeypatch, tmp_path):
    pv = {**_state()["panel_verdict"], "judge_not_diverse": True}
    _, md, *_ = _report(monkeypatch, tmp_path, _state(panel_verdict=pv))
    assert "· judge diversity ⚠ ·" in md.split("---")[0]


def test_degradation_notes_show_this_round_and_count_the_earlier_ones(monkeypatch, tmp_path):
    errors = ["[r1] skip: research.digest https://x.example/p (role research.digest: every engine failed after "
              "3 attempts (last: DriverError: 429))",
              "[r1] degrade: outline.plan → one_section_per_subquestion_roundrobin_notes (boom)",
              "fetch: https://y.example failed (HTTP 404)",
              "legacy: run_config defaults applied (pre-v4 checkpoint)",
              "analysis: dropped text before the first heading",
              "[r2] skip: compose.section Usage (role compose.section: every engine failed after 1 attempt "
              "(last: DriverError: boom))"]
    writer = {"role": "compose.section", "tier": "strong", "profile": "fake", "provider": "fake",
              "requested_model": "m", "served_model": "m", "routed_via": "", "ok": True}
    out, md, meta, evidence, run = _report(monkeypatch, tmp_path,
                                           _state(errors=errors, revise_rounds=1, model_calls=[writer]))
    notes = md.split("## Degradation notes")[1]
    assert md.index("## Sources") < md.index("## Degradation notes")
    assert "- [r2] legacy: run_config defaults applied (pre-v4 checkpoint)" in notes
    assert "- [r2] skip: compose.section Usage" in notes
    assert "- earlier rounds: r1 — 2 notes" in notes
    assert "outline.plan" not in notes and "fetch:" not in notes and "analysis:" not in notes
    assert evidence["degradations"] == [{"round": 1, "text": errors[0].removeprefix("[r1] ")},
                                        {"round": 1, "text": errors[1].removeprefix("[r1] ")},
                                        {"round": 2, "text": errors[3]},
                                        {"round": 2, "text": errors[5].removeprefix("[r2] ")}], (
        "the round lives in round, the text without its [r<k>] tag")
    assert metrics.degradations_count(run) == 4 == metrics.degradations_count({**run, "evidence": None})


def test_a_pdf_failure_is_an_error_line_and_a_degradation_note(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("reportlab exploded")
    out, md, meta, evidence, _ = _report(monkeypatch, tmp_path, _state(), pdf=boom)
    assert out["report_paths"]["pdf"] is None
    assert out["errors"] == ["[r1] pdf: RuntimeError: reportlab exploded"]
    assert "- [r1] pdf: RuntimeError: reportlab exploded" in md.split("## Degradation notes")[1]
    assert evidence["errors"][-1] == "[r1] pdf: RuntimeError: reportlab exploded"


@pytest.mark.parametrize("pv,banner", [
    ({"judged": 0, "total": 8, "skipped_judges": ["grounding batch 1: 429"],
      "judge_verdicts": {"grounding": "unjudged", "grounding_batches": 1}},
     "> ⚠ Not fact-checked — the grounding judge failed (429)"),
    ({"judged": 5, "total": 8, "skipped_judges": ["grounding batch 2: unparseable"],
      "judge_verdicts": {"grounding": "pass", "grounding_batches": 2}},
     "> ⚠ Not fact-checked — 3 of 8 findings were not judged (unparseable)"),
])
def test_the_not_fact_checked_banner(monkeypatch, tmp_path, pv, banner):
    _, md, meta, *_ = _report(monkeypatch, tmp_path, _state(panel_verdict={"verdict": "pass", **pv}))
    assert banner in md and meta["degraded"] is True


# ---------------------------------------------------------------------------
# Fix round 1: location, collateral and twin cuts, unknown ids, unjudged
# sections, per-item roles, the Analysis
# ---------------------------------------------------------------------------

def test_a_flagged_sentence_stated_twice_leaves_the_body(monkeypatch, tmp_path):            # reviewer B's P2
    twice = "LangGraph pauses a run at an interrupt [1]. It powers review loops [2]. " \
            "LangGraph pauses a run at an interrupt [1]."
    state = _state(sections=[{"title": "Mechanism", "md": twice, "note_ids": [1, 2]}],
                   findings=[_f("f0", "Mechanism", "LangGraph pauses a run at an interrupt [1].", [1]),
                             _f("f1", "Mechanism", "It powers review loops [2].", [2])],
                   panel_verdict={"verdict": "pass", "flagged_finding_ids": ["f0"], "judged": 2, "total": 2})
    _, md, _, _, run = _report(monkeypatch, tmp_path, state)
    assert "LangGraph pauses a run" not in md.split("## ⚠ Flagged claims")[0]
    assert metrics.flagged_claims_in_body(run) == 0


_spec = importlib.util.spec_from_file_location("live_check", Path(__file__).resolve().parents[1] / "scripts"
                                               / "live_check.py")
live_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live_check)


def test_a_finding_stated_in_a_cut_sentence_is_cut_everywhere(monkeypatch, tmp_path):     # reviewer B's P4
    s = "Model A has 70B parameters and was released in May [1]."
    state = _state(sections=[{"title": "Mechanism", "md": f"{s} It is open weights [2].", "note_ids": [1, 2]}],
                   findings=[_f("f0", "Mechanism", s, [1], claim="Model A has 70B parameters"),
                             _f("f1", "Mechanism", s, [1], claim="Model A came out in May"),
                             _f("f2", "Mechanism", "It is open weights [2].", [2])],
                   panel_verdict={"verdict": "pass", "flagged_finding_ids": ["f0"], "judged": 3, "total": 3,
                                  "judge_verdicts": {"grounding": "revise", "flag_reasons": {"f0": "no such size"}}})
    out, md, meta, evidence, run = _report(monkeypatch, tmp_path, state)
    assert [f["status"] for f in evidence["findings"]] == ["cut", "cut", "kept"]
    assert "• findings: 1 •" in md and meta["n_findings"] == 1 and len(live_check._kept(run)) == 1
    assert "- **Mechanism:** Model A has 70B parameters and was released in May [1]. — no such size; stated in a " \
           "cut sentence (f0)" in md


def test_a_cut_claim_never_reaches_the_tldr_through_its_kept_analysis_twin(monkeypatch, tmp_path):   # live q013
    analysis_md = ("_Analysis — interpretation by Argus._\n\n### Key takeaways\n\n"
                   "- QISA has a 2.6x longer inference time [1] — confidence **high** (2 independent sources)\n"
                   "- It resumes with a command [3] — confidence **medium** (1 independent source)")
    analysis = {**ANALYSIS, "md": analysis_md, "takeaways": [
        {"text": "QISA has a 2.6x longer inference time [1]", "source_ids": [1], "confidence": "high",
         "reasons": ["2 independent sources"]},
        {"text": "It resumes with a command [3]", "source_ids": [3], "confidence": "medium",
         "reasons": ["1 independent source"]}]}
    body = "QISA has a 2.6x longer inference time than GPT-4 [1]. Teams use it for approvals [2]."
    state = _state(run_config=make_run_config("medium"), analysis=analysis,
                   sections=[{"title": "Report", "md": body, "note_ids": [1, 2, 3]}],
                   findings=[_f("f0", "Report", "QISA has a 2.6x longer inference time than GPT-4 [1].", [1]),
                             _f("f1", "Report", "Teams use it for approvals [2].", [2]),
                             _f("f2", "Analysis", "QISA has a 2.6x longer inference time [1]", [1]),
                             _f("f3", "Analysis", "It resumes with a command [3]", [3])],
                   panel_verdict={"verdict": "pass", "flagged_finding_ids": ["f0"], "judged": 4, "total": 4,
                                  "judge_verdicts": {"grounding": "revise",
                                                     "flag_reasons": {"f0": "no such model"}}})
    out, md, _, evidence, _ = _report(monkeypatch, tmp_path, state)
    assert [f["status"] for f in evidence["findings"]] == ["cut", "kept", "cut", "kept"]
    assert "2.6x" not in out["tldr_md"] and "It resumes with a command [3]" in out["tldr_md"]
    assert "2.6x" not in md.split("## ⚠ Flagged claims")[0]
    assert "- **Analysis:** QISA has a 2.6x longer inference time [1] — confidence **high** (2 independent " \
           "sources) — near-duplicate of cut f0" in md


def test_an_unknown_id_beside_known_ones_is_stripped_and_recorded(monkeypatch, tmp_path):
    findings = [dict(f) for f in FINDINGS]
    findings[7] = _f("f7", "Usage", "Teams use it for approvals [1][9].", [1, 9])
    usage = USAGE.replace("approvals [1].", "approvals [1][9].")
    state = _state(sections=[{"title": "Mechanism", "md": MECHANISM, "note_ids": [1, 2, 3]},
                             {"title": "Usage", "md": usage, "note_ids": [1, 2, 3]}], findings=findings)
    out, md, _, _, run = _report(monkeypatch, tmp_path, state)
    assert "- Teams use it for approvals [1]." in md and "approvals [1][9]" not in md
    assert '[r1] cite: dropped unknown source id [9] in section "Usage"' in out["errors"]
    assert metrics.cite_markers_unresolved(run) == 0


def test_sections_citing_sources_without_findings_are_named_not_fact_checked(monkeypatch, tmp_path):
    skip = ("[r1] skip: compose.findings_extract Usage (role compose.findings_extract: every engine failed after "
            "3 attempts (last: DriverError: provider fake: 429 Too Many Requests))")
    state = _state(run_config=make_run_config("medium"), analysis=ANALYSIS, findings=[], errors=[skip],
                   panel_verdict={"verdict": "pass", "judged": 0, "total": 0})
    _, md, meta, evidence, _ = _report(monkeypatch, tmp_path, state)
    assert ('> ⚠ Not fact-checked — section "Mechanism" was not fact-checked (no findings extracted); '
            'section "Usage" was not fact-checked (429); '
            'section "Analysis" was not fact-checked (no findings extracted)') in md
    assert meta["degraded"] is True and evidence["degraded"] is True


def test_one_skipped_coverage_review_is_a_note_not_a_degraded_run(monkeypatch, tmp_path):
    line = ("[r1] skip: panel.coverage report (role panel.coverage: every engine failed after 3 attempts "
            "(last: DriverError: provider fake: 429 Too Many Requests))")
    _, md, meta, evidence, _ = _report(monkeypatch, tmp_path, _state(errors=[line]))
    assert "⚠ Degraded run" not in md and meta["degraded"] is False and evidence["degraded"] is False
    assert f"- {line}" in md.split("## Degradation notes")[1]


def test_an_outline_section_titled_analysis_keeps_its_own_body(monkeypatch, tmp_path):
    own = "The section's own reading of the usage [2]."
    sections = [{"title": "Mechanism", "md": MECHANISM, "note_ids": [1, 2, 3]},
                {"title": "Analysis", "md": own, "note_ids": [2]}]
    _, md, *_ = _report(monkeypatch, tmp_path, _state(sections=sections, findings=[
        dict(FINDINGS[0]), _f("f1", "Analysis", own, [2])], panel_verdict={"verdict": "pass", "judged": 2,
                                                                              "total": 2}))
    assert md.count(f"## Analysis\n\n{own}") == 1, "no analyst ran: the section is not repeated as the analysis"
    _, md, *_ = _report(monkeypatch, tmp_path / "analyst", _state(
        sections=sections, run_config=make_run_config("medium"), analysis=ANALYSIS,
        findings=[dict(FINDINGS[0]), _f("f1", "Analysis", "It pauses at interrupts [1]", [1])],
        panel_verdict={"verdict": "pass", "judged": 2, "total": 2}))
    assert md.count(own) == 1 and md.count("It resumes with a command [3]") == 2, (
        "the section keeps its body; the analysis is rendered once (and in the TL;DR)")


def test_an_analysis_left_with_its_label_only_is_dropped(monkeypatch, tmp_path):
    analysis = {**ANALYSIS, "takeaways": [ANALYSIS["takeaways"][1]],
                "md": f"{ANALYSIS_LABEL}\n\n### Key takeaways\n\n"
                      "- It was built in 1999 [2] — confidence **low** (1 independent source)"}
    state = _state(run_config=make_run_config("medium"), analysis=analysis,
                   findings=[dict(FINDINGS[0]), _f("f1", "Analysis", "It was built in 1999 [2]", [2])],
                   panel_verdict={"verdict": "pass", "flagged_finding_ids": ["f1"], "judged": 2, "total": 2})
    out, md, *_ = _report(monkeypatch, tmp_path, state)
    assert "## Analysis" not in md and ANALYSIS_LABEL not in md
    assert "analysis: every takeaway cut — section dropped" in out["errors"]


# ---------------------------------------------------------------------------
# The run driver: runs.exa_usd after the report preview (spec §2.4)
# ---------------------------------------------------------------------------

async def test_the_run_driver_writes_runs_exa_usd_after_the_preview(monkeypatch, caplog):
    from argus.bot import research as bot_research
    order: list[str] = []

    class Graph:
        def astream(self, inp, config=None, stream_mode=None):
            async def gen():
                yield {"report_builder": {"report_paths": {"folder": "f", "md": "f/report.md"}}}
            return gen()

        async def aget_state(self, cfg):
            return SimpleNamespace(values={"report_paths": {"folder": "f", "md": "f/report.md"},
                                           "search_spend": {"exa_usd": 0.042}})

    async def update_run(run_id, **cols):
        order.append(f"update_run {run_id} {cols}")
        return False                                             # no library row

    async def preview(ctx, info):
        order.append("preview")

    async def nothing(*a, **k):
        return {}

    lib = SimpleNamespace(update_run=update_run)
    monkeypatch.setattr(bot_research, "_ensure_run_config", nothing)
    monkeypatch.setattr(bot_research, "_stream_progress", nothing)
    monkeypatch.setattr(bot_research, "_set_run_status", nothing)
    monkeypatch.setattr(bot_research, "_send_report_preview", preview)
    monkeypatch.setattr(bot_research, "_get_library", lambda ctx: lib)
    info = {"run_id": "ab12cd34", "chat_id": 5, "graph": Graph(), "cfg": {}, "length": "medium"}
    with caplog.at_level("WARNING", logger="argus.bot"):
        await bot_research._resume_after_plan(SimpleNamespace(application=SimpleNamespace(bot=None)), info)
    assert order == ["preview", "update_run ab12cd34 {'exa_usd': 0.042}"]
    assert "run ab12cd34 has no library row; its exa_usd is in the checkpoint only" in caplog.text