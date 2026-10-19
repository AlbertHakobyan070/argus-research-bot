"""Tests for src/argus/quarantine.py — the v4 cut-not-flag machinery (WP-C).

The sentence splitter, the locator (containment, then a 1–3-sentence
windowed fuzzy match), the whole-sentence cutter and ``apply_cuts``, which
turns the panel's flags and the unknown-id rule into finding statuses, cut
bodies and the two appendices. Report-level behaviour is in
tests/test_cut_not_flag.py.

Run with:
    PYTHONPATH='' ./venv/Scripts/python.exe -m pytest tests/test_quarantine.py -q
"""
from __future__ import annotations

from argus.quarantine import (apply_cuts, cut, locate, render_cut_appendix, render_unlocated_appendix,
                              sentences)


def test_sentences_end_at_a_terminator_before_a_capital_a_digit_or_markup():
    md = "One is here. Two follows! 3 starts with a digit? **Bold** opens the next. last one"
    assert sentences(md) == ["One is here.", "Two follows!", "3 starts with a digit?",
                             "**Bold** opens the next. last one"]


def test_trailing_citation_markers_stay_with_their_sentence():
    assert sentences("A holds. [1] B holds [2]. C holds.") == ["A holds. [1]", "B holds [2].", "C holds."]


def test_list_items_and_continuation_lines():
    md = "- First item. Second sentence\n  continues here [1].\n- Next item [2]."
    assert sentences(md) == ["First item.", "Second sentence\n  continues here [1].", "Next item [2]."]


def test_locate_prefers_containment_and_maps_to_whole_sentences():
    md = "Alpha holds [1]. The beta value is 4.5 points [2]. Gamma follows."
    assert locate("beta value is 4.5", md) == [1]
    assert locate("Alpha holds [1]. The beta value", md) == [0, 1], "a span over two sentences"
    assert locate("Something else entirely", md) is None


def test_cut_rebuilds_only_the_touched_block():
    md = "Keep this.\nAnd this line.\n\nDrop me [1]. Keep me [2].\n\n- item [3]."
    assert cut(md, [2]) == "Keep this.\nAnd this line.\n\nKeep me [2].\n\n- item [3]."


FINDINGS = [
    {"id": "f0", "claim": "Alpha holds", "section": "S", "source_ids": [1], "sentence": "Alpha holds [1].",
     "status": "cut"},
    {"id": "f1", "claim": "Beta is 4.5", "section": "S", "source_ids": [2],
     "sentence": "The beta value is 4.5 points [2].", "status": "kept"},
    {"id": "f2", "claim": "Ghost", "section": "S", "source_ids": [7], "sentence": "Gamma follows [7].",
     "status": "kept"},
    {"id": "f3", "claim": "Nowhere to be found", "section": "S", "source_ids": [1], "sentence": "",
     "status": "kept"},
    {"id": "f4", "claim": "Delta", "section": "Other", "source_ids": [1], "sentence": "Delta [1].",
     "status": "kept"},
]
MDS = {"S": "Alpha holds [1]. The beta value is 4.5 points [2]. Gamma follows [7].", "T": "Untouched [1]."}
FLAGS = {"f1": "the quote says 3.5", "f3": "invented", "f4": "nope"}


def test_apply_cuts_sets_the_statuses_and_cuts_the_bodies():
    findings, mds, cut_rows, unlocated = apply_cuts([dict(f) for f in FINDINGS], MDS, FLAGS, n_sources=3)
    assert [f["status"] for f in findings] == ["kept", "cut", "cut", "unlocated", "unlocated"], (
        "a stale status is recomputed; a finding of a section without a body cannot be located")
    assert mds == {"S": "Alpha holds [1].", "T": "Untouched [1]."}
    assert cut_rows == [
        {"section": "S", "text": "The beta value is 4.5 points [2].", "reason": "the quote says 3.5"},
        {"section": "S", "text": "Gamma follows.", "reason": "unknown source id (cites 7)"}]
    assert unlocated == [{"section": "S", "text": "Nowhere to be found", "reason": "invented"},
                         {"section": "Other", "text": "Delta", "reason": "nope"}]


def test_a_finding_without_flags_is_kept_and_the_body_untouched():
    findings, mds, cut_rows, unlocated = apply_cuts([{**FINDINGS[0], "status": "kept"}], MDS, {}, n_sources=3)
    assert findings[0]["status"] == "kept" and mds == MDS and cut_rows == unlocated == []


def test_two_flags_on_one_sentence_make_one_appendix_line():
    findings = [dict(FINDINGS[1]), {**FINDINGS[1], "id": "f9", "claim": "Beta again"}]
    _, _, cut_rows, _ = apply_cuts(findings, MDS, {"f1": "wrong", "f9": "also wrong"}, n_sources=3)
    assert cut_rows == [{"section": "S", "text": "The beta value is 4.5 points [2].", "reason": "wrong; also wrong"}]


def test_the_appendices():
    assert render_cut_appendix([]) == "" and render_unlocated_appendix([]) == ""
    cut_md = render_cut_appendix([{"section": "S", "text": "Gamma follows.", "reason": "unknown source id (cites 7)"}])
    assert cut_md.startswith("## ⚠ Flagged claims (removed from the body)\n\n_")
    assert cut_md.endswith("\n\n- **S:** Gamma follows. — unknown source id (cites 7)")
    loose = render_unlocated_appendix([{"section": "S", "text": "Nowhere", "reason": "invented"}])
    assert loose.startswith("## ⚠ Flagged but not located (still in body)\n\n_")
    assert loose.endswith("\n\n- **S:** Nowhere — invented")