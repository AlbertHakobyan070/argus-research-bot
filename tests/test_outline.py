"""v4 WP-I — the outline node (spec §3 WP-I, §2.8 allocation).

Fresh mode: sections with ``note_ids`` from ``outline.plan``, then the
deterministic repair — unknown ids dropped, a note in at most 2 sections,
every usable note allocated (most shared sub-questions; ties rotate), a
section under ``section_min_notes`` merged into its neighbour (the previous
one, else the next), never dropped. Extend mode: titles are immutable, at most
one new section and only for a sub-question that was starved before the wave,
``changed_sections`` names what changed. ``outline.plan`` degrades to the
deterministic outline with the same allocation. The real ``llm.complete`` runs
on a fake driver.
"""
from __future__ import annotations

import json

import pytest

from argus.graph import compose, outline
from argus.graph.outline import EXTEND_SYSTEM, OUTLINE_SYSTEM, outline_node

from helpers import fake_llm, make_run_config

SUBS = ["What is X?", "How does X work?", "Where is X used?", "What are X's limits?"]


def _brief(subs=SUBS, archetype="explainer") -> dict:
    return {"sub_questions": [{"q": q, "kind": "web"} for q in subs], "must_have_keywords": ["x"],
            "summary": "s", "success_criteria": ["explains X"], "archetype": archetype}


def _note(sid, sub_qs, *, relevance=4, verified=True, url=None) -> dict:
    return {"source_id": sid, "source_url": url or f"https://s{sid}.example/p", "title": f"S{sid}",
            "sub_qs": sub_qs, "relevance": relevance, "stance": "supports",
            "claims": [{"text": f"claim {sid}", "quote": "q", "confidence": "high", "quote_verified": verified}],
            "source_type": "other", "authority": 1, "cached": False}


EVIDENCE = [_note(1, [0]), _note(2, [0]), _note(3, [1]), _note(4, [1]),
            _note(5, [2]), _note(6, [2]), _note(7, [3]), _note(8, [3])]


def _state(length="medium", *, evidence=EVIDENCE, subs=SUBS, archetype="explainer", rc=None, **extra) -> dict:
    return {"user_request": "explain X", "length": length, "run_config": rc or make_run_config(length),
            "brief": _brief(subs, archetype), "evidence": evidence,
            "fetched": [{"url": n["source_url"], "title": n["title"], "pinned": False} for n in evidence],
            **extra}


def _sections(*specs) -> str:
    return json.dumps({"sections": [{"title": t, "focus": f, "sub_qs": s, "note_ids": n}
                                    for t, f, s, n in specs]})


def _layout(out) -> list[tuple]:
    return [(s["title"], s["sub_qs"], s["note_ids"]) for s in out["outline"]["sections"]]


def _assert_valid(out, usable_ids):
    held = [i for s in out["outline"]["sections"] for i in s["note_ids"]]
    assert set(held) == set(usable_ids), "every usable note is allocated, nothing else"
    assert all(held.count(i) <= 2 for i in usable_ids), "a note sits in at most 2 sections"
    assert all(len(s["note_ids"]) == len(set(s["note_ids"])) for s in out["outline"]["sections"])


# ---------------------------------------------------------------------------
# Flat lengths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("length", ["tldr", "short"])
def test_flat_lengths_need_no_outline(monkeypatch, length):
    seen = fake_llm(monkeypatch, lambda system, human: pytest.fail("no outline call for a flat length"))
    out = outline_node(_state(length))
    assert out["outline"] == {"sections": [], "changed_sections": []} and seen == []


def test_a_single_subquestion_brief_needs_no_outline(monkeypatch):
    seen = fake_llm(monkeypatch, lambda system, human: pytest.fail("no outline call"))
    out = outline_node(_state(subs=["What is X?"], evidence=[_note(1, [0])]))
    assert out["outline"]["sections"] == [] and seen == []


def test_compose_keeps_a_reexport_of_the_moved_node():
    assert compose.outline_node is outline.outline_node


# ---------------------------------------------------------------------------
# Fresh mode
# ---------------------------------------------------------------------------

def test_fresh_prompt_carries_the_archetype_shape_the_budget_and_the_note_index(monkeypatch):
    seen = fake_llm(monkeypatch, lambda system, human: _sections(
        ("Basics", "f", [0, 1], [1, 2, 3, 4]), ("Uses", "f", [2, 3], [5, 6, 7, 8])))
    outline_node(_state(archetype="comparison"))
    (system, human), = [m[:2] for m in seen]
    assert system == OUTLINE_SYSTEM
    assert "at most 4 sections" in human
    assert "comparison" in human and "Comparison criteria" in human, "the archetype's section shape"
    assert "[3] s3.example · relevance 4 · sub-questions 1 — claim 3" in human, "the note index"
    assert "(1) How does X work? — 2 notes, 2 strong" in human


def test_a_lecture_keeps_its_parts_and_applies_the_archetype_focus_inside(monkeypatch):
    seen = fake_llm(monkeypatch, lambda system, human: _sections(
        ("Part I — Basics", "f", [0, 1], [1, 2, 3, 4]), ("Part II — Uses", "f", [2, 3], [5, 6, 7, 8])))
    outline_node(_state("lecture"))
    human = seen[0][1]
    assert "Part I" in human and "Part IV" in human and "How it works" in human
    assert "at most 7 sections" in human


def test_fresh_allocation_is_repaired(monkeypatch):
    """Unknown id 99 and the duplicate 2 dropped; note 5's third section
    dropped; the unallocated note 8 goes to the section sharing its
    sub-question; "Uses" is left with one note and merges into the previous
    section — its sub-question, notes and focus move, its title goes."""
    fake_llm(monkeypatch, lambda system, human: _sections(
        ("Basics", "what X is", [0], [1, 2, 2, 99, 5]),
        ("Mechanism", "how", [1], [3, 4, 5]),
        ("Uses", "where", [2], [5, 6]),
        ("Limits", "limits", [3], [7])))
    out = outline_node(_state())
    _assert_valid(out, range(1, 9))
    assert _layout(out) == [("Basics", [0], [1, 2, 5]), ("Mechanism", [1, 2], [3, 4, 5, 6]),
                            ("Limits", [3], [7, 8])]
    assert out["outline"]["sections"][1]["focus"] == "how; where"
    assert out["outline"]["changed_sections"] == ["Basics", "Mechanism", "Limits"]
    assert [c["role"] for c in out["model_calls"]] == ["outline.plan"] and out["errors"] == []


def test_the_first_section_when_starved_merges_into_the_next(monkeypatch):
    fake_llm(monkeypatch, lambda system, human: _sections(
        ("Lone", "a", [0], [1]), ("Rest", "b", [1, 2, 3], [3, 4, 5, 6, 7, 8])))
    out = outline_node(_state(rc=make_run_config("medium", **{"compose.section_min_notes": 3})))
    assert _layout(out) == [("Rest", [1, 2, 3, 0], [3, 4, 5, 6, 7, 8, 1, 2])]


def test_merging_never_drops_the_last_section(monkeypatch):
    """Two usable notes for three sections: A (first) merges into the next,
    then the empty C into the previous — everything ends in one section."""
    fake_llm(monkeypatch, lambda system, human: _sections(
        ("A", "a", [0], [1]), ("B", "b", [1], [3]), ("C", "c", [2, 3], [])))
    out = outline_node(_state(evidence=[_note(1, [0]), _note(2, [1], relevance=1), _note(3, [1])]))
    assert _layout(out) == [("B", [1, 0, 2, 3], [3, 1])]


def test_the_section_budget_of_the_length_wins(monkeypatch):
    fake_llm(monkeypatch, lambda system, human: _sections(
        *[(t, "f", [k % 4], ids) for k, (t, ids) in enumerate(
            [("A", [1, 2]), ("B", [3, 4]), ("C", [5, 6]), ("D", [7, 8]), ("E", [1, 3]), ("F", [2, 4])])]))
    out = outline_node(_state())
    assert [s["title"] for s in out["outline"]["sections"]] == ["A", "B", "C", "D"]


def test_only_notes_a_writer_can_use_are_allocated(monkeypatch):
    """Below ``writer_min_relevance`` or every quote failed → not allocated;
    an appended local file or a pinned source always is."""
    evidence = [*EVIDENCE, _note(9, [0], relevance=1), _note(10, [1], verified=False),
                _note(11, [2], relevance=1, url="file:///vault/notes.md"), _note(12, [3], relevance=0)]
    state = _state(evidence=evidence)
    state["fetched"][11]["pinned"] = True
    seen = fake_llm(monkeypatch, lambda system, human: _sections(
        ("Basics", "f", [0, 1], [1, 2, 3, 4, 9, 10]), ("Uses", "f", [2, 3], [5, 6, 7, 8])))
    out = outline_node(state)
    _assert_valid(out, [*range(1, 9), 11, 12])
    human = seen[0][1]
    assert "[11] local" in human and "[12]" in human and "[9]" not in human and "[10]" not in human


def test_fresh_degrade_is_the_deterministic_outline_with_its_notes_allocated(monkeypatch):
    seen = fake_llm(monkeypatch, lambda system, human: "Sections: background, mechanism, uses.")
    out = outline_node(_state())
    assert len(seen) == 2, "the nudge retry, then the degradation"
    assert _layout(out) == [("What is X", [0], [1, 2]), ("How does X work", [1], [3, 4]),
                            ("Where is X used", [2], [5, 6]), ("What are X's limits", [3], [7, 8])]
    (line,) = out["errors"]
    assert line.startswith("[r1] degrade: outline.plan → one_section_per_subquestion_roundrobin_notes "
                           "(role outline.plan: every engine failed after 2 attempts")
    assert [c["ok"] for c in out["model_calls"]] == [False, False]


@pytest.mark.parametrize("sub_qs,expected", [
    ([[0, 1]] * 5, [[1, 3, 5], [2, 4]]),                           # every note tied
    ([[0, 1], [0], [0, 1], [0], [0, 1]], [[1, 2, 4, 5], [3]]),     # ties 1, 3, 5 alternate; 2 and 4 are no tie
])
def test_ties_rotate_across_the_tied_sections(monkeypatch, sub_qs, expected):
    fake_llm(monkeypatch, lambda system, human: "no JSON here")
    evidence = [_note(i, s) for i, s in enumerate(sub_qs, 1)]
    rc = make_run_config("medium", **{"compose.section_min_notes": 1})
    out = outline_node(_state(subs=SUBS[:2], evidence=evidence, rc=rc))
    assert [s["note_ids"] for s in out["outline"]["sections"]] == expected


def test_a_repeated_title_merges_into_its_first_occurrence(monkeypatch):
    """Case and whitespace aside; sub-questions and notes unioned in order, the
    first focus kept, and the duplicate does not count against the budget."""
    fake_llm(monkeypatch, lambda system, human: _sections(
        ("How X works", "the mechanism", [0, 1], [1, 2, 3]), ("Uses", "where", [2], [5, 6]),
        ("how  x   WORKS", "again", [1, 3], [3, 4, 7]), ("Limits", "limits", [3], [7, 8]),
        ("Outlook", "next", [2], [5, 6])))
    out = outline_node(_state())
    _assert_valid(out, range(1, 9))
    assert _layout(out) == [("How X works", [0, 1, 3], [1, 2, 3, 4, 7]), ("Uses", [2], [5, 6]),
                            ("Limits", [3], [7, 8]), ("Outlook", [2], [5, 6])]
    assert out["outline"]["sections"][0]["focus"] == "the mechanism"
    assert out["outline"]["changed_sections"] == ["How X works", "Uses", "Limits", "Outlook"]


def test_the_round_tag_counts_user_rounds(monkeypatch):
    fake_llm(monkeypatch, lambda system, human: "no JSON here")
    out = outline_node(_state(revise_rounds=1, extend_rounds=1))
    assert out["errors"][0].startswith("[r3] degrade: outline.plan")


# ---------------------------------------------------------------------------
# Extend mode
# ---------------------------------------------------------------------------

def _existing(*specs) -> dict:
    return {"sections": [{"title": t, "focus": f"{t} focus", "sub_qs": s, "note_ids": n} for t, s, n in specs],
            "changed_sections": []}


OLD = _existing(("Basics", [0, 1], [1, 2, 3, 4]), ("Uses", [2, 3], [5, 6, 7, 8]))


def test_extend_keeps_titles_and_assigns_only_the_new_notes(monkeypatch):
    reply = {"assign": [{"note_id": 9, "section": "basics"}, {"note_id": 9, "section": "Uses"},
                        {"note_id": 9, "section": "Basics"}, {"note_id": 1, "section": "Uses"},
                        {"note_id": 77, "section": "Uses"}, {"note_id": 10, "section": "Nope"}],
             "new_section": None}
    seen = fake_llm(monkeypatch, lambda system, human: json.dumps(reply))
    evidence = [*EVIDENCE, _note(9, [1]), _note(10, [3])]
    out = outline_node(_state(evidence=evidence, outline=OLD))
    assert seen[0][0] == EXTEND_SYSTEM
    human = seen[0][1]
    assert "[9] s9.example" in human and "[10] s10.example" in human
    assert "[1] s1.example" not in human, "only the new notes are indexed"
    assert '"Basics"' in human
    assert _layout(out) == [("Basics", [0, 1], [1, 2, 3, 4, 9]), ("Uses", [2, 3], [5, 6, 7, 8, 9, 10])]
    assert out["outline"]["changed_sections"] == ["Basics", "Uses"]
    assert [c["role"] for c in out["model_calls"]] == ["outline.plan"]


def test_extend_merges_a_repeated_existing_title_and_lists_it_once(monkeypatch):
    """An outline written before titles were deduplicated: its repeated title
    merges into the first occurrence, so the assignments land there, not all
    on the last duplicate, and ``changed_sections`` names the title once."""
    dup = _existing(("Basics", [0], [1, 2]), ("Uses", [2, 3], [5, 6, 7, 8]), ("Basics", [1], [3, 4]))
    reply = {"assign": [{"note_id": 9, "section": "  BASICS "}, {"note_id": 10, "section": "Basics"}],
             "new_section": None}
    fake_llm(monkeypatch, lambda system, human: json.dumps(reply))
    out = outline_node(_state(evidence=[*EVIDENCE, _note(9, [1]), _note(10, [0])], outline=dup))
    assert _layout(out) == [("Basics", [0, 1], [1, 2, 3, 4, 9, 10]), ("Uses", [2, 3], [5, 6, 7, 8])]
    assert out["outline"]["sections"][0]["focus"] == "Basics focus"
    assert out["outline"]["changed_sections"] == ["Basics"]


def test_extend_with_no_new_note_changes_nothing(monkeypatch):
    seen = fake_llm(monkeypatch, lambda system, human: pytest.fail("no call without new notes"))
    out = outline_node(_state(evidence=[*EVIDENCE, _note(9, [1], relevance=1)], outline=OLD))
    assert out["outline"] == OLD and seen == []


SUBS5 = [*SUBS, "Who deploys X?"]


def _extend_state(new_notes, outline_=OLD):
    return _state(subs=SUBS5, evidence=[*EVIDENCE, *new_notes], outline=outline_)


def test_extend_may_open_one_section_for_a_starved_subquestion(monkeypatch):
    reply = {"assign": [], "new_section": {"title": "Deployment", "focus": "who deploys X",
                                           "sub_qs": [4], "note_ids": [9, 10]}}
    fake_llm(monkeypatch, lambda system, human: json.dumps(reply))
    out = outline_node(_extend_state([_note(9, [4]), _note(10, [4])]))
    assert _layout(out)[-1] == ("Deployment", [4], [9, 10])
    assert [s["title"] for s in out["outline"]["sections"]] == ["Basics", "Uses", "Deployment"]
    assert out["outline"]["changed_sections"] == ["Deployment"]


@pytest.mark.parametrize("new_section", [
    {"title": "More basics", "focus": "f", "sub_qs": [0], "note_ids": [9, 10]},   # sub-question 0 was not starved
    {"title": "basics", "focus": "f", "sub_qs": [4], "note_ids": [9, 10]},        # an existing title
])
def test_extend_rejects_a_new_section_the_rules_do_not_allow(monkeypatch, new_section):
    fake_llm(monkeypatch, lambda system, human: json.dumps({"assign": [], "new_section": new_section}))
    out = outline_node(_extend_state([_note(9, [0, 4]), _note(10, [3, 4])]))
    assert _layout(out) == [("Basics", [0, 1], [1, 2, 3, 4, 9]), ("Uses", [2, 3], [5, 6, 7, 8, 10])]


def test_a_starved_new_section_merges_into_the_last_existing_one(monkeypatch):
    reply = {"assign": [{"note_id": 10, "section": "Basics"}],
             "new_section": {"title": "Deployment", "focus": "who", "sub_qs": [4], "note_ids": [9]}}
    fake_llm(monkeypatch, lambda system, human: json.dumps(reply))
    out = outline_node(_extend_state([_note(9, [4]), _note(10, [0])]))
    assert _layout(out) == [("Basics", [0, 1], [1, 2, 3, 4, 10]), ("Uses", [2, 3, 4], [5, 6, 7, 8, 9])]
    assert out["outline"]["changed_sections"] == ["Basics", "Uses"]


def test_extend_degrade_allocates_the_new_notes_by_overlap(monkeypatch):
    fake_llm(monkeypatch, lambda system, human: "Put 9 under Basics.")
    out = outline_node(_extend_state([_note(9, [1]), _note(10, [3])]))
    assert _layout(out) == [("Basics", [0, 1], [1, 2, 3, 4, 9]), ("Uses", [2, 3], [5, 6, 7, 8, 10])]
    assert out["outline"]["changed_sections"] == ["Basics", "Uses"]
    assert out["errors"][0].startswith("[r1] degrade: outline.plan → one_section_per_subquestion_roundrobin_notes")
