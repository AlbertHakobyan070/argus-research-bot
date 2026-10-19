"""v4 WP-C — compose (spec §3 WP-C): allocation-only notes, the reuse rule,
``compose_runs``, the skipped-writer merge (R11), the dedupe pass, the
revision-notes order, the flat body and the extractor's verbatim sentence.

A section writer sees only its allocated notes (``note_ids``) plus the
pinned ones, filtered by ``writer_min_relevance`` and ``quote_verified is
not False``. Compose rewrites only ``panel_verdict.revise_sections`` (a
panel rewrite) or ``outline.changed_sections`` (fresh, gap wave, extend,
user revise) — new sections always; the rest keep md, ``note_ids`` and
findings (re-ided). A first write that is skipped merges its notes into its
neighbour (the previous section, else the next), written once more; a
skipped re-write keeps the previous version; every section skipped raises.
"""
from __future__ import annotations

import json
import re

import pytest
from langgraph.runtime import Runtime

from argus import llm
from argus.engines.llm_openai import DriverError
from argus.graph.compose import (EXTRACT_SYSTEM, FLAT_SYSTEM, SECTION_SYSTEM, compose_node, extract_findings,
                                 render_notes)
from argus.graph.nodes import revise_prep_node
from argus.graph.run_config import RunConfig
from helpers import fake_llm, make_run_config

WORDS = {1: "graphs pause at interrupts", 2: "checkpoints persist the state", 3: "resume takes a command",
         4: "the user clip mentions approvals", 5: "weak evidence", 6: "subgraphs restart from the top"}
FETCHED = [{"url": f"https://s{i}.example/p", "title": f"S{i}"} for i in range(1, 7)]
FETCHED[3] = {"url": "file:///V/clip.txt", "title": "My clip", "pinned": True}


def _note(i, subqs, relevance=4, claims=None):
    return {"source_id": i, "source_url": FETCHED[i - 1]["url"], "title": FETCHED[i - 1]["title"],
            "sub_qs": subqs, "relevance": relevance, "stance": "supports",
            "claims": claims or [{"text": f"fact {i}", "quote": f"quote {i}", "confidence": "high",
                                  "quote_verified": True}]}


NOTES = [_note(1, [0]),
         _note(2, [0], claims=[{"text": "fact 2", "quote": "q2", "confidence": "high", "quote_verified": True},
                               {"text": "BAD CLAIM 2", "quote": "nope", "confidence": "high",
                                "quote_verified": False}]),
         _note(3, [1]), _note(4, [], relevance=1), _note(5, [0], relevance=1), _note(6, [2])]
BRIEF = {"sub_questions": [{"q": "A?", "kind": "web"}, {"q": "B?", "kind": "web"}, {"q": "C?", "kind": "web"}],
         "summary": "s"}
OUTLINE = {"sections": [{"title": "A", "focus": "", "sub_qs": [0], "note_ids": [1, 2, 5]},
                        {"title": "B", "focus": "", "sub_qs": [1], "note_ids": [3]},
                        {"title": "C", "focus": "", "sub_qs": [2], "note_ids": [6]}],
           "changed_sections": ["A", "B", "C"]}


def _sentence(i: int) -> str:
    return f"{WORDS[i].capitalize()} [{i}]."


PRIOR_SECTIONS = [{"title": "A", "md": f"{_sentence(1)} {_sentence(2)}", "note_ids": [1, 2, 5]},
                  {"title": "B", "md": _sentence(3), "note_ids": [3]},
                  {"title": "C", "md": _sentence(6), "note_ids": [6]}]


def _finding(fid, section, i, status="kept"):
    return {"id": fid, "claim": WORDS[i].capitalize(), "citation_urls": [FETCHED[i - 1]["url"]],
            "confidence": "high", "section": section, "source_ids": [i], "sentence": _sentence(i), "status": status}


PRIOR_FINDINGS = [_finding("f0", "A", 1), _finding("f1", "A", 2, "cut"), _finding("f2", "B", 3),
                  _finding("f3", "C", 6), _finding("f4", "Analysis", 1)]


def _state(**over) -> dict:
    state = {"thread_id": "tg:1:abcd1234", "user_request": "A, B and C", "length": "medium",
             "run_config": make_run_config("medium"), "brief": BRIEF, "evidence": NOTES, "fetched": FETCHED,
             "outline": OUTLINE, "coverage": {"0": 2, "1": 1, "2": 1}, "model_calls": [], "errors": []}
    state.update(over)
    return state


def _later(**over) -> dict:
    """A state after a first compose: its sections, findings and compose_runs."""
    return _state(**{"sections": PRIOR_SECTIONS, "findings": PRIOR_FINDINGS, "compose_runs": 1,
                     "outline": {**OUTLINE, "changed_sections": []}, **over})


def _llm(monkeypatch, *, fail=lambda title, human: False, write=None, extract=None):
    def reply(system, human):
        if system.startswith(SECTION_SYSTEM):
            title = re.search(r"^Section: (.+)$", human, re.M).group(1)
            if fail(title, human):
                raise DriverError("provider fake: boom")
            if write and title in write:
                return write[title]
            return " ".join(_sentence(int(i)) for i in re.findall(r"^\[(\d+)\] ", human, re.M))
        if system == EXTRACT_SYSTEM:
            title = re.match(r"Section: (.+)", human).group(1)
            if extract and title in extract:
                return extract[title]
            md = human.split("\n\n", 1)[1].rsplit("\n\nReturn the findings JSON.", 1)[0]
            return json.dumps({"findings": [
                {"claim": re.sub(r"\s*\[\d+\]", "", s).rstrip("."), "sentence": s,
                 "source_ids": [int(x) for x in re.findall(r"\[(\d+)\]", s)], "confidence": "high"}
                for s in re.split(r"(?<=\.)\s+", md) if s.strip()]})
        raise AssertionError(f"unexpected system prompt {system[:40]!r}")
    return fake_llm(monkeypatch, reply)


def _writes(seen) -> list[tuple[str, str]]:
    return [(re.search(r"^Section: (.+)$", m[1], re.M).group(1), m[1]) for m in seen
            if m[0].startswith(SECTION_SYSTEM)]


# ---------------------------------------------------------------------------
# Allocation-only notes
# ---------------------------------------------------------------------------

def test_each_writer_sees_only_its_allocated_notes_and_the_pinned_one(monkeypatch):
    seen = _llm(monkeypatch)
    compose_node(_state())
    prompts = dict(_writes(seen))
    ids = {t: sorted(int(i) for i in re.findall(r"^\[(\d+)\] ", p, re.M)) for t, p in prompts.items()}
    assert ids == {"A": [1, 2, 4], "B": [3, 4], "C": [4, 6]}, "5 is below writer_min_relevance; 4 is pinned"
    assert "fact 2" in prompts["A"] and "BAD CLAIM 2" not in prompts["A"], "a failed quote never reaches a writer"


def test_render_notes_puts_pinned_notes_first():
    block = render_notes(NOTES, [1, 2], pinned=frozenset({4}))
    assert block.index("[4] ") < block.index("[1] ") and "[3] " not in block


# ---------------------------------------------------------------------------
# The reuse rule and compose_runs
# ---------------------------------------------------------------------------

def test_a_panel_rewrite_rewrites_only_the_revise_sections(monkeypatch):
    seen = _llm(monkeypatch)
    out = compose_node(_later(panel_verdict={"next": "compose", "revise_sections": ["B"],
                                             "notes": ["B: quote the evidence"]}))
    writes = _writes(seen)
    assert [t for t, _ in writes] == ["B"] and "- B: quote the evidence" in writes[0][1]
    assert out["compose_runs"] == 2
    secs = {s["title"]: s for s in out["sections"]}
    assert secs["A"] == PRIOR_SECTIONS[0] and secs["C"] == PRIOR_SECTIONS[2], "reused: md and note_ids"
    assert [(f["id"], f["section"], f["claim"], f["status"]) for f in out["findings"]] == [
        ("f0", "A", "Graphs pause at interrupts", "kept"), ("f1", "A", "Checkpoints persist the state", "kept"),
        ("f2", "B", "The user clip mentions approvals", "kept"), ("f3", "B", "Resume takes a command", "kept"),
        ("f4", "C", "Subgraphs restart from the top", "kept")], (
        "reused findings keep their text, reset to kept; B's rewrite renders the pinned note first")
    extracted = [m[1] for m in seen if m[0] == EXTRACT_SYSTEM]
    assert len(extracted) == 1 and extracted[0].startswith("Section: B")


def test_the_changed_sections_are_rewritten_after_a_gap_wave_or_an_extend(monkeypatch):
    seen = _llm(monkeypatch)
    state = _later(panel_verdict={"next": "research", "revise_sections": ["B"]},
                   outline={**OUTLINE, "changed_sections": ["C"]})
    compose_node(state)
    assert [t for t, _ in _writes(seen)] == ["C"]


def test_a_new_section_is_always_written(monkeypatch):
    seen = _llm(monkeypatch)
    outline = {"sections": [*OUTLINE["sections"], {"title": "D", "sub_qs": [2], "note_ids": [6]}],
               "changed_sections": []}
    out = compose_node(_later(outline=outline))
    assert [t for t, _ in _writes(seen)] == ["D"]
    assert [s["title"] for s in out["sections"]] == ["A", "B", "C", "D"]


def test_a_user_revise_rewrites_every_section_with_the_newest_note_first(monkeypatch):
    seen = _llm(monkeypatch)
    state = _later(revision_notes=["first ask", "second ask"], panel_verdict={"next": "report_builder"})
    state = {**state, **revise_prep_node(state)}
    assert state["outline"]["changed_sections"] == ["A", "B", "C"] and state["compose_runs"] == 0
    out = compose_node(state)
    writes = _writes(seen)
    assert sorted(t for t, _ in writes) == ["A", "B", "C"]
    assert all("REVISION NOTES (must address):\n- second ask\n- first ask\n" in p for _, p in writes)
    assert out["compose_runs"] == 1


def test_revision_notes_are_the_users_then_the_panels_capped_at_eight(monkeypatch):
    seen = _llm(monkeypatch)
    compose_node(_later(revision_notes=["u1", "u2"],
                        panel_verdict={"next": "compose", "revise_sections": ["A"],
                                       "notes": [f"p{i}" for i in range(1, 10)]}))
    (_, prompt), = _writes(seen)
    notes = prompt.split("REVISION NOTES (must address):\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert notes == ["- u2", "- u1", "- p1", "- p2", "- p3", "- p4", "- p5", "- p6"]


# ---------------------------------------------------------------------------
# Skipped writers (R11 merge) and the ALL_SKIPPED_RAISES rule
# ---------------------------------------------------------------------------

def test_a_skipped_first_write_merges_its_notes_into_the_previous_section(monkeypatch):
    seen = _llm(monkeypatch, fail=lambda title, human: title == "B")
    out = compose_node(_state())
    assert [s["title"] for s in out["sections"]] == ["A", "C"]
    a = next(s for s in out["sections"] if s["title"] == "A")
    assert a["note_ids"] == [1, 2, 5, 3] and "[3]" in a["md"], "A was written once more, with B's notes"
    assert [t for t, _ in _writes(seen)].count("A") == 2
    assert any(e.startswith("[r1] skip: compose.section B (role compose.section: every engine failed")
               for e in out["errors"])


def test_a_skipped_first_section_merges_into_the_next(monkeypatch):
    _llm(monkeypatch, fail=lambda title, human: title == "A")
    out = compose_node(_state())
    assert [s["title"] for s in out["sections"]] == ["B", "C"]
    assert out["sections"][0]["note_ids"] == [3, 1, 2, 5]


def test_when_the_merged_rewrite_is_skipped_too_the_neighbour_keeps_its_first_version(monkeypatch):
    _llm(monkeypatch, fail=lambda title, human: title == "B" or (title == "A" and "\n[3] " in human))
    out = compose_node(_state())
    a = out["sections"][0]
    assert [s["title"] for s in out["sections"]] == ["A", "C"]
    assert a["note_ids"] == [1, 2, 5] and "[3]" not in a["md"]
    assert '[r1] compose: section "B" dropped — notes 3 not written (its merge into "A" was skipped)' in out["errors"]
    assert sum(e.startswith("[r1] skip: compose.section ") for e in out["errors"]) == 2


def test_every_section_skipped_raises(monkeypatch):
    _llm(monkeypatch, fail=lambda title, human: True)
    with pytest.raises(llm.EngineError, match="every section"):
        compose_node(_state())


def test_a_skipped_rewrite_keeps_the_previous_version(monkeypatch):
    _llm(monkeypatch, fail=lambda title, human: title == "B")
    out = compose_node(_later(panel_verdict={"next": "compose", "revise_sections": ["B"], "notes": []}))
    assert out["sections"] == PRIOR_SECTIONS
    assert [f["claim"] for f in out["findings"] if f["section"] == "B"] == ["Resume takes a command"]


def test_an_empty_reply_is_a_skipped_write(monkeypatch):
    _llm(monkeypatch, write={"C": "  short "})
    out = compose_node(_state())
    assert [s["title"] for s in out["sections"]] == ["A", "B"]
    assert "[r1] skip: compose.section C (empty reply)" in out["errors"]


# ---------------------------------------------------------------------------
# The dedupe pass
# ---------------------------------------------------------------------------

def test_dedupe_cuts_a_later_near_duplicate_in_a_rewritten_section(monkeypatch):
    _llm(monkeypatch, write={"B": "Graphs pause at interrupts here [1]. Resume takes a command [3]."})
    out = compose_node(_later(panel_verdict={"next": "compose", "revise_sections": ["B"], "notes": []}))
    secs = {s["title"]: s for s in out["sections"]}
    assert secs["B"]["md"] == "Resume takes a command [3]."
    assert secs["A"] == PRIOR_SECTIONS[0], "the first occurrence wins; a reused section stays byte-identical"
    assert [f["claim"] for f in out["findings"] if f["section"] == "B"] == ["Resume takes a command"]


def test_dedupe_keeps_a_duplicate_whose_sentence_backs_another_finding(monkeypatch):
    md = "Graphs pause at interrupts and resume takes a command [1][3]."
    extract = json.dumps({"findings": [
        {"claim": "Graphs pause at interrupts", "sentence": md, "source_ids": [1]},
        {"claim": "Resume takes a command from the user", "sentence": md, "source_ids": [3]}]})
    _llm(monkeypatch, write={"B": md}, extract={"B": extract})
    out = compose_node(_later(panel_verdict={"next": "compose", "revise_sections": ["B"], "notes": []}))
    assert next(s for s in out["sections"] if s["title"] == "B")["md"] == md
    assert len([f for f in out["findings"] if f["section"] == "B"]) == 2


def test_dedupe_can_be_switched_off(monkeypatch):
    _llm(monkeypatch, write={"B": "Graphs pause at interrupts here [1]. Resume takes a command [3]."})
    out = compose_node(_later(run_config=make_run_config("medium", **{"compose.dedupe_pass": False}),
                              panel_verdict={"next": "compose", "revise_sections": ["B"], "notes": []}))
    assert len([f for f in out["findings"] if f["section"] == "B"]) == 2


# ---------------------------------------------------------------------------
# The extractor
# ---------------------------------------------------------------------------

def test_the_extractor_returns_the_verbatim_sentence_and_its_ids(monkeypatch):
    reply = json.dumps({"findings": [
        {"claim": "Rates rose", "sentence": "Rates rose sharply 【2】.", "source_ids": [5], "confidence": "high"},
        {"claim": "Ghost", "sentence": "Ghost fact [9].", "source_ids": [9]},
        {"claim": "No ids", "sentence": "No citation here.", "source_ids": []},
        {"claim": "Model ids only", "sentence": "", "source_ids": [1, "x", 1]},
        "not a finding"]})
    fake_llm(monkeypatch, lambda system, human: reply)
    rc = RunConfig.model_validate(make_run_config("medium"))
    findings, calls, errors = extract_findings("A", "Rates rose sharply [2].", FETCHED, engines=rc.engines, rnd=2)
    assert findings == [
        {"claim": "Rates rose", "citation_urls": [FETCHED[1]["url"]], "confidence": "high", "section": "A",
         "source_ids": [2], "sentence": "Rates rose sharply [2].", "status": "kept"},
        {"claim": "Ghost", "citation_urls": [], "confidence": "medium", "section": "A", "source_ids": [9],
         "sentence": "Ghost fact [9].", "status": "kept"},
        {"claim": "Model ids only", "citation_urls": [FETCHED[0]["url"]], "confidence": "medium", "section": "A",
         "source_ids": [1], "sentence": "", "status": "kept"}]
    assert [c["role"] for c in calls] == ["compose.findings_extract"] and errors == []


def test_a_skipped_extraction_is_declared(monkeypatch):
    fake_llm(monkeypatch, lambda system, human: (_ for _ in ()).throw(DriverError("provider fake: boom")))
    rc = RunConfig.model_validate(make_run_config("medium"))
    findings, calls, errors = extract_findings("Analysis", "x [1].", FETCHED, engines=rc.engines, rnd=3)
    assert findings == [] and [c["ok"] for c in calls] == [False]
    assert errors == ["[r3] skip: compose.findings_extract Analysis (role compose.findings_extract: every engine "
                      "failed after 1 attempt (last: DriverError: provider fake: boom))"]


def test_every_extraction_skipped_raises(monkeypatch):
    _llm(monkeypatch, extract={t: "no json at all" for t in "ABC"})
    with pytest.raises(llm.EngineError, match="extraction"):
        compose_node(_state())


def test_writer_markers_are_normalised_before_extraction(monkeypatch):
    seen = _llm(monkeypatch, write={"A": "Graphs pause at interrupts 【1】. Checkpoints persist [2, 1]."})
    out = compose_node(_state())
    a = next(s for s in out["sections"] if s["title"] == "A")
    assert a["md"] == "Graphs pause at interrupts [1]. Checkpoints persist [2][1]."
    assert "【" not in next(m[1] for m in seen if m[0] == EXTRACT_SYSTEM and m[1].startswith("Section: A"))


# ---------------------------------------------------------------------------
# The flat body, no evidence, events
# ---------------------------------------------------------------------------

def test_the_flat_writer_writes_a_body_without_tldr_or_key_findings(monkeypatch):
    assert "TL;DR" not in FLAT_SYSTEM and "Key Findings" not in FLAT_SYSTEM
    seen = fake_llm(monkeypatch, lambda system, human: (
        "Graphs pause at interrupts [1]. Resume takes a command [3]." if system.startswith(FLAT_SYSTEM)
        else json.dumps({"findings": [{"claim": "Graphs pause", "sentence": "Graphs pause at interrupts [1].",
                                       "source_ids": [1]}]})))
    out = compose_node(_state(length="short", run_config=make_run_config("short"),
                              outline={"sections": [], "changed_sections": []}))
    assert [s["title"] for s in out["sections"]] == ["Report"]
    assert out["draft_md"].startswith("## Findings\n\nGraphs pause at interrupts [1].")
    assert [f["section"] for f in out["findings"]] == ["Report"] and out["compose_runs"] == 1
    assert sorted(int(i) for i in re.findall(r"^\[(\d+)\] ", seen[0][1], re.M)) == [1, 2, 3, 4, 6]


def test_a_flat_writer_failure_raises(monkeypatch):
    fake_llm(monkeypatch, lambda system, human: (_ for _ in ()).throw(DriverError("provider fake: boom")))
    with pytest.raises(llm.EngineError):
        compose_node(_state(length="short", run_config=make_run_config("short"),
                            outline={"sections": [], "changed_sections": []}))


def test_no_usable_note_is_the_classified_no_evidence_draft(monkeypatch):
    fake_llm(monkeypatch, lambda system, human: pytest.fail("no writer without evidence"))
    out = compose_node(_state(evidence=[_note(1, [0], relevance=1)], compose_runs=0))
    assert "synthesis_no_evidence" in out["draft_md"] and out["findings"] == [] and out["sections"] == []
    assert out["compose_runs"] == 1


def test_section_events(monkeypatch):
    _llm(monkeypatch, fail=lambda title, human: title == "C")
    events: list = []
    compose_node(_later(panel_verdict={"next": "compose", "revise_sections": ["B", "C"], "notes": []}),
                 Runtime(stream_writer=events.append))
    rows = {e["data"]["title"]: e["data"] for e in events if e["kind"] == "section"}
    assert rows["A"] == {"title": "A", "ok": True, "chars": len(PRIOR_SECTIONS[0]["md"]), "reused": True}
    assert rows["B"]["ok"] is True and rows["B"]["reused"] is False
    assert rows["C"] == {"title": "C", "ok": False, "chars": len(PRIOR_SECTIONS[2]["md"]), "reused": True}, (
        "a skipped re-write: the write failed, the previous version stays")
    assert all(e["node"] == "compose" for e in events)