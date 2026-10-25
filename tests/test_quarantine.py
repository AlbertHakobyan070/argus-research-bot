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

import re
from decimal import Decimal
from difflib import SequenceMatcher

import pytest

from argus import quarantine
from argus.graph import compose as compose_mod
from argus.graph import research
from argus.graph.report_builder_helpers import build_tldr
from argus.quarantine import (apply_cuts, cut, locate, render_cut_appendix, render_unlocated_appendix,
                              sentences, strip_unknown_ids)


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


def _f(fid, sentence, ids, claim=None, section="S"):
    return {"id": fid, "claim": claim or sentence.rstrip("."), "section": section, "source_ids": ids,
            "sentence": sentence, "status": "kept"}


def test_a_flagged_sentence_stated_twice_is_cut_at_every_occurrence():               # reviewer B's P2
    md = "X is the fastest model on the board [1]. Other text here [2]. X is the fastest model on the board [1]."
    findings, mds, rows, _ = apply_cuts([_f("f1", "X is the fastest model on the board [1].", [1])], {"S": md},
                                        {"f1": "unsupported"}, 3)
    assert findings[0]["status"] == "cut" and mds["S"] == "Other text here [2]."
    assert rows == [{"section": "S", "text": "X is the fastest model on the board [1].", "reason": "unsupported"}]


def test_a_whole_sentence_match_comes_before_containment():                          # reviewer B's P3
    md = "Analysts said prices rose sharply in March [1]. Prices rose [2]."
    assert locate("Prices rose [2].", md) == [1]
    findings, mds, _, _ = apply_cuts([_f("f1", "Prices rose [2].", [2]),
                                      _f("f2", "Analysts said prices rose sharply in March [1].", [1])],
                                     {"S": md}, {"f1": "unsupported"}, 3)
    assert [f["status"] for f in findings] == ["cut", "kept"]
    assert mds["S"] == "Analysts said prices rose sharply in March [1]."


def test_a_finding_stated_in_a_cut_sentence_is_cut_too():                             # reviewer B's P4
    s = "Model A has 70B parameters and was released in May [1]."
    findings, mds, rows, _ = apply_cuts([_f("f1", s, [1], claim="Model A has 70B parameters"),
                                         _f("f2", s, [1], claim="Model A came out in May"),
                                         _f("f3", "It is open weights [2].", [2])],
                                        {"S": f"{s} It is open weights [2]."}, {"f1": "no such size"}, 3)
    assert [f["status"] for f in findings] == ["cut", "cut", "kept"] and mds["S"] == "It is open weights [2]."
    assert rows == [{"section": "S", "text": s, "reason": "no such size; stated in a cut sentence (f1)"}]


def test_a_kept_near_duplicate_of_a_cut_finding_takes_the_stricter_verdict():
    findings, mds, rows, _ = apply_cuts(
        [_f("f1", "QISA runs 2.6x longer inference time [1].", [1]),
         _f("f2", "QISA has a 2.6x longer inference time [2].", [2], section="Analysis"),
         _f("f3", "Other models are faster [3].", [3], section="Analysis")],
        {"S": "QISA runs 2.6x longer inference time [1].",
         "Analysis": "QISA has a 2.6x longer inference time [2]. Other models are faster [3]."},
        {"f1": "no such model"}, 3)
    assert [f["status"] for f in findings] == ["cut", "cut", "kept"]
    assert mds == {"S": "", "Analysis": "Other models are faster [3]."}
    assert rows[1] == {"section": "Analysis", "text": "QISA has a 2.6x longer inference time [2].",
                       "reason": "near-duplicate of cut f1"}


def test_a_cut_sentence_never_reaches_the_tldr():                                      # reviewer B's P1
    s1 = "Revenue grew 30 percent in 2024 and margins doubled across the group [1]."
    s3 = "Revenue grew 31 percent in 2024 and margins doubled across the group [2]."
    findings, mds, _, _ = apply_cuts([_f("f1", s1, [1]), _f("f2", s1, [1]), _f("f3", s3, [2])],
                                     {"S": f"{s1} {s3}"}, {"f1": "the quote says 3 percent"}, 3)
    items, _ = build_tldr(analysis=None, analysis_md="", sections=[{"title": "S", "md": mds["S"]}],
                          findings=findings)
    assert not any("30 percent" in i for i in items)
    # and by containment only: a kept finding whose sentence is gone never enters by a fuzzy match
    kept = [{**_f("f2", s1, [1]), "status": "kept"}]
    assert build_tldr(analysis=None, analysis_md="", sections=[{"title": "S", "md": s3}],
                      findings=kept) == ([], "omitted")


def test_an_unknown_id_beside_known_ones_is_stripped():
    md = "Alpha holds [2][9]. Beta holds [9]. Gamma [1] and delta [12] hold.\n\n- Item [3] [9]."
    assert strip_unknown_ids(md, 3) == (
        "Alpha holds [2]. Beta holds [9]. Gamma [1] and delta hold.\n\n- Item [3].", [9, 12])
    assert strip_unknown_ids("Untouched [1].", 3) == ("Untouched [1].", [])


def test_the_appendices():
    assert render_cut_appendix([]) == "" and render_unlocated_appendix([]) == ""
    cut_md = render_cut_appendix([{"section": "S", "text": "Gamma follows.", "reason": "unknown source id (cites 7)"}])
    assert cut_md.startswith("## ⚠ Flagged claims (removed from the body)\n\n_")
    assert cut_md.endswith("\n\n- **S:** Gamma follows. — unknown source id (cites 7)")
    loose = render_unlocated_appendix([{"section": "S", "text": "Nowhere", "reason": "invented"}])
    assert loose.startswith("## ⚠ Flagged but not located (still in body)\n\n_")
    assert loose.endswith("\n\n- **S:** Nowhere — invented")


# ---------------------------------------------------------------------------
# Fix round 2: the re-review's probes O2–O7
# ---------------------------------------------------------------------------

def test_a_finding_still_stated_after_the_cut_stays_kept():                           # re-review O2
    md = "Model X tops the board on MMLU, analysts say [1]. In May, Model X tops the board on MMLU again [2]."
    findings, mds, rows, _ = apply_cuts([_f("f1", "Model X tops the board on MMLU, analysts say [1].", [1]),
                                         _f("f2", "Model X tops the board on MMLU", [2], claim="Model X leads MMLU")],
                                        {"S": md}, {"f1": "no such analysts"}, 3)
    assert [f["status"] for f in findings] == ["cut", "kept"], "f2's text is still in the body"
    assert mds["S"] == "In May, Model X tops the board on MMLU again [2]."
    assert rows == [{"section": "S", "text": "Model X tops the board on MMLU, analysts say [1].",
                     "reason": "no such analysts"}]


@pytest.mark.parametrize("copy", ["| Alpha holds [2][9]. |\n|---|", "### Alpha holds [2][9].",
                                  "```\nAlpha holds [2][9].\n```"])
def test_the_unknown_id_strip_edits_the_sentence_where_it_stands(copy):              # re-review O5
    assert strip_unknown_ids(f"{copy}\n\nAlpha holds [2][9].", 3) == (f"{copy}\n\nAlpha holds [2].", [9])


def test_a_twin_carries_the_same_number_tokens():                                     # re-review O7
    findings, mds, _, _ = apply_cuts(
        [_f("f1", "Revenue grew 30 percent in 2024 across the group [1].", [1]),
         _f("f2", "Revenue grew 31 percent in 2024 across the group [2].", [2], section="T"),
         _f("f3", "QISA needs 2.6x longer inference [1].", [1], section="U"),
         _f("f4", "QISA needs 2.7x longer inference [3].", [3], section="T")],
        {"S": "Revenue grew 30 percent in 2024 across the group [1].",
         "T": "Revenue grew 31 percent in 2024 across the group [2]. QISA needs 2.7x longer inference [3].",
         "U": "QISA needs 2.6x longer inference [1]."}, {"f1": "the source says 3", "f3": "no such model"}, 3)
    assert [f["status"] for f in findings] == ["cut", "kept", "cut", "kept"], "a different number, a different claim"
    assert mds["T"] == "Revenue grew 31 percent in 2024 across the group [2]. QISA needs 2.7x longer inference [3]."


def test_twins_share_compose_dedupe_normalisation():
    findings, *_ = apply_cuts([_f("f1", "QISA needs 2.6x longer inference [1].", [1]),
                               _f("f2", "ＱＩＳＡ needs ２.６x longer inference [2].", [2], section="T")],
                              {"S": "QISA needs 2.6x longer inference [1].",
                               "T": "ＱＩＳＡ needs ２.６x longer inference [2]."}, {"f1": "no such model"}, 3)
    assert [f["status"] for f in findings] == ["cut", "cut"], "NFKC and casefold: the full-width claim is the same"
    assert not hasattr(compose_mod, "_norm"), "one normalisation, quarantine.claim_norm"
    assert quarantine.claim_norm("ＱＩＳＡ Grew [3] 2.6x") == "qisa grew 2 6x"
    assert quarantine._FOLD == research._QUOTE_FOLD, "quote verification's fold, copied verbatim"
    assert quarantine._NEGATORS == research._NEGATORS, "quote verification's negators, copied verbatim"


TAKEAWAY_MD = ("_Analysis — label._\n\n### Key takeaways\n\n"
               "- QISA needs 2.6x longer inference [1] — confidence **high** (2 independent sources)")


def _tldr(findings, mds, flagged):
    out, cut_mds, _, _ = apply_cuts(findings, {"Analysis": TAKEAWAY_MD, **mds}, flagged, 5)
    analysis = {"takeaways": [{"text": "QISA needs 2.6x longer inference [1]", "confidence": "high"}]}
    return ([f["status"] for f in out],
            build_tldr(analysis=analysis, analysis_md=cut_mds["Analysis"], sections=[], findings=out))


def test_a_takeaway_enters_the_tldr_only_when_its_analysis_finding_is_kept():        # re-review O3, O4
    # O3: the analysis twin of a cut finding, its sentence a paraphrase found nowhere: unlocated
    statuses, tldr = _tldr([_f("f0", "QISA needs 2.6x longer inference than GPT-4 [1].", [1], section="Report",
                               claim="QISA needs 2.6x longer inference"),
                            _f("f1", "QISA is said to need 2.6 times the inference time", [1], section="Analysis",
                               claim="QISA needs 2.6x longer inference")],
                           {"Report": "QISA needs 2.6x longer inference than GPT-4 [1]. Other [2]."},
                           {"f0": "no such model"})
    assert statuses == ["cut", "unlocated"] and tldr == ([], "omitted")
    # O4: the flagged takeaway finding itself, unlocated
    statuses, tldr = _tldr([_f("f1", "A paraphrase nobody wrote", [1], section="Analysis",
                               claim="QISA needs 2.6x longer inference")], {}, {"f1": "unsupported"})
    assert statuses == ["unlocated"] and tldr == ([], "omitted")
    # no analysis finding at all: nothing judged it
    assert _tldr([], {}, {})[1] == ([], "omitted")
    # kept: its finding states the takeaway
    statuses, tldr = _tldr([_f("f1", "QISA needs 2.6x longer inference [1]", [1], section="Analysis")], {}, {})
    assert statuses == ["kept"] and tldr == (["QISA needs 2.6x longer inference [1] — confidence **high**"],
                                             "takeaways")

# ---------------------------------------------------------------------------
# Fix round 3: twins by numeric value and a shared source, the negation guard (spec R1 at 9b7e012)
# ---------------------------------------------------------------------------

# The re-review's probes (probe_misc.py; its first six are twin_formats.py's reformatted restatements)
PROBE_PAIRS = [
    ("Revenue grew 30% in 2024 across the group [1]", "Revenue grew 30 percent in 2024 across the group [1]", True),
    ("QISA improves the character error rate by 15.5x [4]", "QISA improves the character error rate by 15.5× [4]",
     True),
    ("The model has 70B parameters and ships in 2025 [2]", "The model has 70 billion parameters and ships in 2025 [2]",
     True),
    ("QISA needs 2.6x longer inference than GPT-4 [1]", "QISA needs 2.6x longer inference [1]", True),
    ("The service costs $1,200 per month for teams [3]", "The service costs $1200 per month for teams [3]", True),
    ("Latency fell from 120 ms to 80 ms after the upgrade [1]", "Latency fell from 120ms to 80ms after the upgrade [1]",
     True),
    ("Revenue grew 30 percent in 2024 across the group [1]", "Revenue grew 31 percent in 2024 across the group [2]",
     False),
    ("QISA needs 2.6x longer inference [1].", "QISA needs 2.7x longer inference [3].", False),
    ("Accuracy rose to 0.5% on the test [1]", "Accuracy rose to 5% on the test [1]", False),
]


def _ids(text: str) -> list[int]:
    return [int(i) for i in re.findall(r"\[(\d+)\]", text)]


@pytest.mark.parametrize("a,b,twin", PROBE_PAIRS)
def test_twins_compare_numeric_values_not_their_format(a, b, twin):                  # re-review 2, probes
    assert quarantine.twins(a, b, _ids(a), _ids(b)) is twin


def test_a_number_attached_to_a_name_is_no_value():
    assert quarantine._values("GPT-4, H2 and v0.2 cost $1,200 or 1,200.50 at 2.50x [3]") == {
        Decimal("1200"), Decimal("1200.5"), Decimal("2.5")}, "names out, thousands separators out, by value"


def test_a_claim_and_its_negation_are_never_twins():                                 # re-review 2
    a, b = "GPT-5 is not open source under any license [3]", "GPT-5 is open source under any license [3]"
    assert SequenceMatcher(None, quarantine.claim_norm(a), quarantine.claim_norm(b), autojunk=False).ratio() >= 0.85
    assert not quarantine.twins(a, b, [3], [3])
    assert not quarantine.twins("QISA doesn’t need 2.6x longer inference [1]", "QISA needs 2.6x longer inference [1]",
                                [1], [1]), "the fold deletes the apostrophe: doesn’t → doesnt, a negator"

# ---------------------------------------------------------------------------
# Fix round 4: the negation guard on rule (i) only; a hyphenated range read as both its bounds (spec R1 at c9804ad)
# ---------------------------------------------------------------------------

# Live q013 (69a2a4b run, accept-t9f2-q013): the Report f0 and its Analysis restatement f8, which adds a caveat with a
# negator — the same values {78.3, 45.7, 11.5}, the same source [2], similarity 0.57. Verbatim from its evidence.json.
CAVEATED_F0 = "QiT-B reports 78.3% ImageNet-1K top-1 accuracy with 45.7M parameters and 11.5 GFLOPs."
CAVEATED_F8 = ("The only exact figures found belong to QiT-B, a Quantum-Inspired Transformer, which reports 78.3% "
               "ImageNet-1K top-1 accuracy with 45.7M parameters and 11.5 GFLOPs, and these should not be attributed "
               "to Nebulon AI.")


def test_a_caveated_restatement_twins_by_its_values_and_source():                      # re-review 3, live q013
    assert SequenceMatcher(None, quarantine.claim_norm(CAVEATED_F0), quarantine.claim_norm(CAVEATED_F8),
                           autojunk=False).ratio() < 0.85
    assert quarantine._negated(CAVEATED_F8) and not quarantine._negated(CAVEATED_F0)
    assert quarantine.twins(CAVEATED_F0, CAVEATED_F8, [2], [2]), "rule (ii) ignores negators"
    assert not quarantine.twins(CAVEATED_F0, CAVEATED_F8, [2], [9]), "rule (ii) needs a shared source"


@pytest.mark.parametrize("text,values", [
    ("10-20%", {10, 20}),
    ("2019-2023", {2019, 2023}),
    ("750–1,000", {750, 1000}),
    ("$4,217–4,235", {4217, 4235}),
])
def test_a_hyphenated_range_carries_both_its_bounds(text, values):                     # re-review 3
    assert quarantine._values(text) == {Decimal(v) for v in values}


# The re-review's range and name probes (candidate_b_refined.py): a different bound makes a different claim; a
# number after a name's hyphen is still no value
RANGE_PAIRS = [
    ("The method gives a 10-20% gain on retrieval [2]", "The method gives a 10-25% gain on retrieval [2]", False),
    ("Qwen2.5-72B scores 86% on MMLU [3]", "Qwen2.5 scores 86% on MMLU [3]", True),
    ("Llama-3-70B scores 82% on MMLU [2]", "Llama-3 scores 82% on MMLU [2]", True),
]


@pytest.mark.parametrize("a,b,twin", RANGE_PAIRS)
def test_twins_read_range_bounds_but_never_a_names_number(a, b, twin):                 # re-review 3
    assert quarantine.twins(a, b, _ids(a), _ids(b)) is twin
