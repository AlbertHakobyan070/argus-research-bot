"""v4 WP-C — compose (spec §3 WP-C): allocation-only notes, the reuse rule,
``compose_runs``, the skipped-writer merge (R11), the dedupe pass, the
revision-notes order, the flat body and the extractor's verbatim sentence.

A section writer sees only its allocated notes (``note_ids``) plus the
pinned ones, filtered by ``writer_min_relevance`` and ``quote_verified is
not False``, within 24 000 chars (the rest recorded). Compose rewrites only
``panel_verdict.revise_sections`` (a panel rewrite) or
``outline.changed_sections`` (fresh, gap wave, extend, user revise) — new
sections always; the rest keep md, ``note_ids`` and findings (re-ided), and
one citing ``[n]`` without findings is extracted again. A pre-v4 outline is
allocated first. A first write that is skipped merges into its neighbour (the
previous section, else the next: sub-questions, notes, focus), written once
more, and the merge sticks for the round; a skipped re-write keeps the
previous version; every section skipped raises. A section titled "Analysis"
is renamed "Evidence analysis" (the title is the analyst's); a section the
dedupe pass empties merges into its neighbour in the outline; footnote
definitions in writer output are removed (a ``cite:`` line).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.runtime import Runtime

from argus import config as config_mod
from argus import llm
from argus.engines.llm_openai import DriverError
from argus.graph import nodes as nodes_mod
from argus.graph.compose import (EXTRACT_SYSTEM, FLAT_SYSTEM, OUTLINE_SYSTEM, SECTION_SYSTEM, compose_node,
                                 extract_findings, render_notes)
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


def _big(i: int, relevance: int) -> dict:
    """A note whose block is ~7 000 chars: three fit the 24 000-char writer budget, four do not."""
    return _note(i, [0], relevance=relevance,
                 claims=[{"text": f"claim {i} " + "x" * 7000, "confidence": "high", "quote_verified": True}])


BIG = [_big(1, 3), _big(2, 5), _big(3, 4), _big(4, 1), _big(6, 3)]          # 4 is the pinned local clip


def test_render_notes_keeps_24000_chars_dropping_the_lowest_relevance_first():
    block = render_notes(BIG, [1, 2, 3, 6])
    assert [int(i) for i in re.findall(r"^\[(\d+)\] ", block, re.M)] == [4, 2, 3], (
        "the pinned note first, then by relevance — no longer the silent 7 000-char cap")


def test_notes_over_the_writer_budget_are_recorded(monkeypatch):
    seen = _llm(monkeypatch)
    outline = {"sections": [{"title": "A", "sub_qs": [0, 1, 2], "note_ids": [1, 2, 3, 6]}], "changed_sections": ["A"]}
    out = compose_node(_state(evidence=BIG, outline=outline))
    (_, prompt), = _writes(seen)
    assert sorted(int(i) for i in re.findall(r"^\[(\d+)\] ", prompt, re.M)) == [2, 3, 4]
    assert '[r1] compose: section "A" — notes [1, 6] over the writer budget' in out["errors"]


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


LEGACY_OUTLINE = {"sections": [{"title": t, "focus": "", "sub_qs": [k]} for k, t in enumerate("ABC")]}
LEGACY_SECTIONS = [{"title": "A", "md": _sentence(1)}, {"title": "B", "md": _sentence(3)},
                   {"title": "C", "md": _sentence(6)}]


def test_a_revise_on_a_pre_v4_checkpoint_allocates_the_notes_first(monkeypatch):
    seen = _llm(monkeypatch)
    state = _state(outline=LEGACY_OUTLINE, sections=LEGACY_SECTIONS, findings=[], compose_runs=1,
                   revision_notes=["make it shorter"], panel_verdict={"next": "report_builder"})
    state = {**state, **revise_prep_node(state)}
    out = compose_node(state)
    ids = {t: sorted(int(i) for i in re.findall(r"^\[(\d+)\] ", p, re.M)) for t, p in _writes(seen)}
    assert ids == {"A": [1, 2, 4], "B": [3, 4], "C": [4, 6]}, "every writer gets its sub-questions' notes"
    assert [s["note_ids"] for s in out["sections"]] == [[1, 2, 4], [3], [6]]
    assert [s["note_ids"] for s in out["outline"]["sections"]] == [[1, 2, 4], [3], [6]]


def test_a_reused_section_citing_sources_without_findings_is_re_extracted(monkeypatch):
    seen = _llm(monkeypatch)
    out = compose_node(_later(findings=[f for f in PRIOR_FINDINGS if f["section"] != "B"],
                              panel_verdict={"next": "compose", "revise_sections": ["C"], "notes": []}))
    assert sorted(m[1].split("\n", 1)[0] for m in seen if m[0] == EXTRACT_SYSTEM) == ["Section: B", "Section: C"]
    assert [t for t, _ in _writes(seen)] == ["C"]
    assert next(s for s in out["sections"] if s["title"] == "B") == PRIOR_SECTIONS[1], "reused, byte-identical"
    assert [f["claim"] for f in out["findings"] if f["section"] == "B"] == ["Resume takes a command"]


def _report_md(monkeypatch, tmp_path, state) -> tuple[str, dict]:
    monkeypatch.setattr(config_mod, "get_settings", lambda: SimpleNamespace(reports_root=tmp_path))
    monkeypatch.setattr(nodes_mod, "markdown_to_pdf", lambda *a, **k: None)
    out = nodes_mod.report_builder_node(state)
    folder = Path(out["report_paths"]["folder"])
    return (folder / "report.md").read_text(encoding="utf-8"), json.loads(
        (folder / "metadata.json").read_text(encoding="utf-8"))


def _judged(state: dict, out: dict) -> dict:
    """``out`` applied to ``state`` as the graph would, every finding judged and passed."""
    n = len(out["findings"])
    return {**state, **out, "errors": state["errors"] + out["errors"],
            "model_calls": state["model_calls"] + out["model_calls"],
            "panel_verdict": {"verdict": "pass", "judged": n, "total": n, "flagged_finding_ids": []}}


def test_a_skipped_extraction_is_named_in_the_banner_until_a_later_compose_re_extracts_it(monkeypatch, tmp_path):
    _llm(monkeypatch, extract={"B": "no json at all"})
    first = _judged(_state(), compose_node(_state()))
    assert not [f for f in first["findings"] if f["section"] == "B"]
    md, meta = _report_md(monkeypatch, tmp_path, first)
    assert '> ⚠ Not fact-checked — section "B" was not fact-checked (unparseable)' in md
    assert meta["degraded"] is True

    _llm(monkeypatch)
    later = compose_node({**first, "panel_verdict": {"next": "compose", "revise_sections": ["C"], "notes": []}})
    b = next(s for s in later["sections"] if s["title"] == "B")
    assert b == next(s for s in first["sections"] if s["title"] == "B"), "reused, byte-identical"
    assert [f["claim"] for f in later["findings"] if f["section"] == "B"] == [
        "The user clip mentions approvals", "Resume takes a command"]
    md, meta = _report_md(monkeypatch, tmp_path / "later", _judged(first, later))
    assert "⚠ Not fact-checked" not in md and "⚠ Degraded run" not in md and meta["degraded"] is False


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


def _subqs(prompt: str) -> list[str]:
    return re.search(r"Sub-questions to answer here:\n(.*?)\nTarget", prompt, re.S).group(1).splitlines()


def test_the_merged_rewrite_carries_the_skipped_sections_sub_questions_and_focus(monkeypatch):
    outline = {"sections": [{**s, "focus": f"focus {s['title']}"} for s in OUTLINE["sections"]],
               "changed_sections": ["A", "B", "C"]}
    seen = _llm(monkeypatch, fail=lambda title, human: title == "B")
    out = compose_node(_state(outline=outline))
    merged = [p for t, p in _writes(seen) if t == "A"][-1]
    assert _subqs(merged) == ["- A?", "- B?"] and "Focus: focus A; focus B\n" in merged
    assert out["outline"] == {"sections": [
        {"title": "A", "focus": "focus A; focus B", "sub_qs": [0, 1], "note_ids": [1, 2, 5, 3]},
        {"title": "C", "focus": "focus C", "sub_qs": [2], "note_ids": [6]}], "changed_sections": ["A", "C"]}, (
        "the merge sticks: B is merged into A and removed from the outline")


def _next(state: dict, out: dict, **over) -> dict:
    """The state a later compose of the round sees: this compose's output applied."""
    return {**state, **{k: out[k] for k in ("sections", "findings", "compose_runs", "outline") if k in out},
            **over}


@pytest.mark.parametrize("later", ["a panel rewrite of C", "a gap wave changing only C"])
@pytest.mark.parametrize("skipped", ["A", "B"])
def test_the_merge_sticks_for_the_round(monkeypatch, skipped, later):
    fail = {skipped}
    seen = _llm(monkeypatch, fail=lambda title, human: title in fail)
    first = compose_node(_state())
    fail.clear()
    seen.clear()
    if later == "a panel rewrite of C":
        over = {"panel_verdict": {"next": "compose", "revise_sections": ["C"], "notes": []}}
    else:                                   # the extend-mode outline: only C's notes changed
        over = {"panel_verdict": {"next": "research"},
                "outline": {**first.get("outline", OUTLINE), "changed_sections": ["C"]}}
    out = compose_node(_next(_state(), first, **over))
    assert [t for t, _ in _writes(seen)] == ["C"], "the merged-away section is not written again"
    assert skipped not in [s["title"] for s in out["sections"]]
    for i in (1, 2, 3):
        assert out["draft_md"].count(_sentence(i)) == 1, f"note {i}'s sentence once"


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


def test_dedupe_drops_a_rewritten_section_it_empties(monkeypatch):
    _llm(monkeypatch, write={"B": "Graphs pause at interrupts here [1]. Checkpoints persist the state here [2]."})
    out = compose_node(_later(panel_verdict={"next": "compose", "revise_sections": ["B"], "notes": []}))
    assert [s["title"] for s in out["sections"]] == ["A", "C"] and "## B" not in out["draft_md"]
    assert ('[r1] compose: section "B" dropped — the dedupe pass cut every sentence (each repeats an earlier '
            'section)') in out["errors"]


def test_dedupe_keeps_a_near_duplicate_citing_other_sources(monkeypatch):
    md = "Graphs pause at interrupts here [3]. Resume takes a command [3]."
    _llm(monkeypatch, write={"B": md})
    out = compose_node(_later(panel_verdict={"next": "compose", "revise_sections": ["B"], "notes": []}))
    assert next(s for s in out["sections"] if s["title"] == "B")["md"] == md, (
        "[3] is not within the earlier finding's [1]: a different source, not a duplicate")


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


def test_the_extractor_intersects_the_models_ids_with_the_sentences(monkeypatch):
    reply = json.dumps({"findings": [
        {"claim": "X holds", "sentence": "X holds [1], and Y holds [2].", "source_ids": [1]},
        {"claim": "Y holds", "sentence": "X holds [1], and Y holds [2].", "source_ids": [2, 7]}]})
    fake_llm(monkeypatch, lambda system, human: reply)
    rc = RunConfig.model_validate(make_run_config("medium"))
    findings, _, _ = extract_findings("A", "X holds [1], and Y holds [2].", FETCHED, engines=rc.engines, rnd=1)
    assert [f["source_ids"] for f in findings] == [[1], [2]]


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


def test_a_too_short_flat_reply_raises_after_the_retry(monkeypatch):
    seen = fake_llm(monkeypatch, lambda system, human: (
        "  short [1] " if system.startswith(FLAT_SYSTEM)
        else json.dumps({"findings": [{"claim": "s", "sentence": "short [1]", "source_ids": [1]}]})))
    with pytest.raises(llm.EngineError, match="unparseable"):
        compose_node(_state(length="short", run_config=make_run_config("short"),
                            outline={"sections": [], "changed_sections": []}))
    assert len(seen) == 2, "the sections' 40-char rule, through the validate hook: one retry, then raise"


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


# ---------------------------------------------------------------------------
# Fix round 2: the reserved "Analysis" title, a section the dedupe empties,
# footnote definitions, the flat writer's markdown nudge
# ---------------------------------------------------------------------------

def test_a_section_titled_analysis_is_renamed_evidence_analysis(monkeypatch):
    assert "Do NOT title a section TL;DR, Sources or Analysis" in " ".join(OUTLINE_SYSTEM.split())
    outline = {"sections": [OUTLINE["sections"][0], {**OUTLINE["sections"][1], "title": " analysis "},
                            OUTLINE["sections"][2]], "changed_sections": ["A", " analysis ", "C"]}
    seen = _llm(monkeypatch)
    out = compose_node(_state(outline=outline))
    assert sorted(t for t, _ in _writes(seen)) == ["A", "C", "Evidence analysis"]
    assert [s["title"] for s in out["sections"]] == ["A", "Evidence analysis", "C"]
    assert "Evidence analysis" in {f["section"] for f in out["findings"]} and "## Analysis" not in out["draft_md"]
    assert out["outline"] == {"sections": [OUTLINE["sections"][0],
                                           {**OUTLINE["sections"][1], "title": "Evidence analysis"},
                                           OUTLINE["sections"][2]],
                              "changed_sections": ["A", "Evidence analysis", "C"]}, (
        "compose returns the renamed outline, as the merge does")


ANALYST = {"md": "_Analysis — interpretation by Argus._\n\n### Key takeaways\n\n"
                 "- Graphs pause at interrupts [1] — confidence **high** (2 independent sources)",
           "takeaways": [{"text": "Graphs pause at interrupts [1]", "source_ids": [1], "confidence": "high",
                          "reasons": ["2 independent sources"]}],
           "stance_map": {}, "has_disagreement_section": False, "has_scenarios": False, "archetype": "explainer"}
ANALYST_FINDING = {"claim": "Graphs pause at interrupts", "citation_urls": [], "confidence": "high",
                   "section": "Analysis", "source_ids": [1], "sentence": "Graphs pause at interrupts [1]",
                   "status": "kept"}


def _with_analysis(state: dict, out: dict, flagged=()) -> dict:
    findings = [*out["findings"], {**ANALYST_FINDING, "id": f"f{len(out['findings'])}"}]
    return {**_judged(state, out), "analysis": ANALYST, "findings": findings,
            "panel_verdict": {"verdict": "pass", "judged": len(findings), "total": len(findings),
                              "flagged_finding_ids": [f["id"] for f in findings if flagged and flagged(f)]}}


def test_the_renamed_section_is_judged_cut_and_stripped_like_any_other(monkeypatch, tmp_path):
    outline = {"sections": [OUTLINE["sections"][0], {**OUTLINE["sections"][1], "title": "Analysis"},
                            OUTLINE["sections"][2]], "changed_sections": None}
    state = _state(outline=outline)
    body = "Resume takes a command [3]. The user clip mentions approvals [4][9]."
    _llm(monkeypatch, write={"Evidence analysis": body})
    out = compose_node(state)
    md, _ = _report_md(monkeypatch, tmp_path, _with_analysis(
        state, out, flagged=lambda f: f["section"] == "Evidence analysis" and "[3]" in f["sentence"]))
    shown = md.split("## ⚠ Flagged claims")[0]
    assert shown.count("\n## Analysis\n") == 1 and "\n## Evidence analysis\n" in shown, "one Analysis: the analyst's"
    assert "Resume takes a command" not in shown, "its flagged sentence is cut, as in any section"
    assert "The user clip mentions approvals [4]." in shown and "[9]" not in shown, "its unknown id is stripped"

    _llm(monkeypatch, write={"Evidence analysis": body}, extract={"Evidence analysis": "no json at all"})
    out = compose_node(state)
    md, meta = _report_md(monkeypatch, tmp_path / "unjudged", _with_analysis(state, out))
    assert 'section "Evidence analysis" was not fact-checked (unparseable)' in md and meta["degraded"] is True, (
        "the analyst's own findings no longer hide the section's")


def test_a_section_the_dedupe_empties_leaves_the_outline(monkeypatch):               # re-review open point (d)
    twice = "Graphs pause at interrupts here [1]. Checkpoints persist the state here [2]."     # A's two findings
    seen = _llm(monkeypatch, write={"B": twice})
    first = compose_node(_later(panel_verdict={"next": "compose", "revise_sections": ["B"], "notes": []}))
    assert [t for t, _ in _writes(seen)] == ["B"], "merged into A in the outline only: A is not written again"
    assert first["outline"] == {"sections": [
        {"title": "A", "focus": "", "sub_qs": [0, 1], "note_ids": [1, 2, 5, 3]},
        {"title": "C", "focus": "", "sub_qs": [2], "note_ids": [6]}], "changed_sections": []}
    seen.clear()
    out = compose_node(_next(_later(), first,
                             panel_verdict={"next": "compose", "revise_sections": ["C"], "notes": []}))
    assert [t for t, _ in _writes(seen)] == ["C"], "a later compose of the round does not write B again"
    assert [s["title"] for s in out["sections"]] == ["A", "C"]


def test_footnote_definitions_in_a_sections_reply_are_removed_and_recorded(monkeypatch):
    _llm(monkeypatch, write={"A": "Graphs pause at interrupts [^1]. Checkpoints persist the state [2].\n\n"
                                  "[^1]: https://s1.example/p\n[^2]: The second source"})
    out = compose_node(_state())
    assert next(s for s in out["sections"] if s["title"] == "A")["md"] == (
        "Graphs pause at interrupts [1]. Checkpoints persist the state [2].")
    assert [e for e in out["errors"] if " cite: " in e] == [
        '[r1] cite: removed footnote definition [^1] in section "A"',
        '[r1] cite: removed footnote definition [^2] in section "A"']


def test_footnote_definitions_in_the_flat_reply_are_removed_and_recorded(monkeypatch):
    body = "Graphs pause at interrupts [^1], and resume takes a command [3]."
    fake_llm(monkeypatch, lambda system, human: (
        f"{body}\n\n[^1]: https://s1.example/p" if system.startswith(FLAT_SYSTEM)
        else json.dumps({"findings": [{"claim": "Graphs pause", "sentence": "Graphs pause at interrupts [1], and "
                                       "resume takes a command [3].", "source_ids": [1, 3]}]})))
    out = compose_node(_state(length="short", run_config=make_run_config("short"),
                              outline={"sections": [], "changed_sections": []}))
    assert out["sections"][0]["md"] == "Graphs pause at interrupts [1], and resume takes a command [3]."
    assert '[r1] cite: removed footnote definition [^1] in section "Report"' in out["errors"]


def test_a_too_short_flat_reply_is_retried_for_markdown_never_json(monkeypatch):     # re-review probe (item 20)
    body = "X is true, as the source states it plainly [1]. More text here."

    def reply(system, human):
        if not system.startswith(FLAT_SYSTEM):
            return json.dumps({"findings": [{"claim": "X is true", "sentence": "X is true, as the source states "
                                             "it plainly [1].", "source_ids": [1]}]})
        nudge = seen[-1][2:]                        # this attempt's retry message, if any
        if not nudge:
            return ""                               # too short: the validate hook's retry follows
        return json.dumps({"report": body}) if "JSON" in nudge[0] else body     # a model does as it is told

    seen = fake_llm(monkeypatch, reply)
    out = compose_node(_state(length="short", run_config=make_run_config("short"),
                              outline={"sections": [], "changed_sections": []}))
    retry = [m for m in seen if m[0].startswith(FLAT_SYSTEM)][1]
    assert retry[-1] == "Reply with only the complete markdown text."
    assert out["sections"][0]["md"] == body, "the body is the markdown, never a JSON object"


def test_a_json_reply_is_never_the_flat_body(monkeypatch):                          # re-review probe (item 20)
    replies = iter(["", '{"report": "X is true, as the source states it plainly [1]. More text here."}'])
    seen = fake_llm(monkeypatch, lambda system, human: next(replies) if system.startswith(FLAT_SYSTEM)
                    else pytest.fail("no extraction without a body"))
    with pytest.raises(llm.EngineError, match="unparseable"):
        compose_node(_state(length="short", run_config=make_run_config("short"),
                            outline={"sections": [], "changed_sections": []}))
    assert len(seen) == 2, "the JSON object fails the validate hook as the empty reply did: then the role raises"