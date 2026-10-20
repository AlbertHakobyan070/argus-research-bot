"""v4 WP-I — the analyst, the stance map and the R13 confidence label
(spec §3 WP-I, §2.8 R12/R13).

The analyst reads the per-section findings, the stance map, the coverage and
the note index — never the section bodies — and writes plain markdown under
fixed headings. Post-processing is deterministic and tested here: uncited
takeaways move to the open questions, the short variant is compacted, each
takeaway gets its confidence label, findings are extracted over the analysis
(section "Analysis"). The real ``llm.complete`` runs on a fake driver; the
findings extractor (compose's ``extract_findings``, WP-C) is faked where it is
looked up, and run for real once (``compose.findings_extract``).
"""
from __future__ import annotations

import json

import pytest
from langgraph.runtime import Runtime

from argus.engines.llm_openai import DriverError
from argus.graph import compose, insight
from argus.graph.insight import (ANALYSIS_LABEL, ANALYST_SYSTEM, analyst_node, confidence_label,
                                 stance_map)
from argus.graph.state import ResearchBrief, SubQuestion

from helpers import fake_llm, make_run_config

FETCHED = [
    {"url": "https://docs.alpha.com/a", "title": "Alpha docs", "published": "2026-09-01"},        # 1
    {"url": "https://blog.alpha.com/b", "title": "Alpha blog", "published": ""},                  # 2: same registrable domain
    {"url": "https://www.beta.org/c", "title": "Beta report", "published": "2026-08-15T09:00:00Z"},  # 3
    {"url": "https://farm.example/d", "title": "Farm page", "published": ""},                    # 4: authority 0
    {"url": "https://gamma.net/e", "title": "Gamma post", "published": ""},                      # 5
    {"url": "https://delta.io/f", "title": "Delta critique", "published": ""},                   # 6: contradicts
]
CLAIM = "Model X reaches 90 percent accuracy on the reasoning benchmark"
COST = "Running the service costs 2 dollars per hour"      # shares no non-stop word with TAKEAWAY


def _n(sid, *, authority=1, verified=True, text=CLAIM, sub_qs=(0,), stance="supports") -> dict:
    return {"source_id": sid, "source_url": FETCHED[sid - 1]["url"], "title": FETCHED[sid - 1]["title"],
            "sub_qs": list(sub_qs), "relevance": 4, "stance": stance,
            "claims": [{"text": text, "quote": "q", "confidence": "high", "quote_verified": verified}],
            "source_type": "other", "authority": authority, "cached": False}


NOTES = [_n(1, authority=2), _n(2, authority=2), _n(3, verified=False), _n(4, authority=0),
         _n(5, text=COST, sub_qs=(1,))]
CONTRA = _n(6, text="Model X accuracy claims do not replicate", stance="contradicts")
BRIEF = ResearchBrief(sub_questions=[SubQuestion(q="How accurate is Model X?"),
                                     SubQuestion(q="What does Model X cost?")],
                      success_criteria=["states the accuracy"], archetype="explainer")
QUIET = stance_map(BRIEF, NOTES)
CONTESTED = stance_map(BRIEF, [*NOTES, CONTRA])
TAKEAWAY = "Model X reaches about 90 percent accuracy on reasoning tasks [1][3]"


# ---------------------------------------------------------------------------
# stance_map
# ---------------------------------------------------------------------------

def test_stance_map_counts_each_subquestion():
    brief = ResearchBrief(sub_questions=[SubQuestion(q="a"), SubQuestion(q="b")])
    notes = [{"source_id": 1, "sub_qs": [0], "stance": "supports"},
             {"source_id": 5, "sub_qs": [0, 1], "stance": "contradicts"},
             {"source_id": 3, "sub_qs": [1], "stance": "mixed"},
             {"source_id": 4, "sub_qs": [1, 7], "stance": "background"},
             {"source_id": 2, "sub_qs": [1], "stance": "contradicts"}]
    assert stance_map(brief, notes) == {
        "0": {"supports": 1, "contradicts": 1, "mixed": 0, "background": 0, "contradicting_sources": [5]},
        "1": {"supports": 0, "contradicts": 2, "mixed": 1, "background": 1, "contradicting_sources": [2, 5]}}


# ---------------------------------------------------------------------------
# confidence_label — the R13 table
# ---------------------------------------------------------------------------

def _verified(sid, text, **kw):
    return _n(sid, text=text, verified=True, **kw)


R13 = [
    # 1. D = 2 (alpha.com, beta.org), V (note 1), A = 2 → high
    ([1, 3], NOTES, TAKEAWAY, QUIET,
     ("high", ["2 independent sources", "quote verified", "authority 2/3"])),
    # 2. registrable-domain collapse: docs.alpha.com + blog.alpha.com are one domain → not high
    ([1, 2], NOTES, TAKEAWAY, QUIET,
     ("medium", ["1 independent source", "quote verified", "authority 2/3"])),
    # 3. overlap-based V: the verified claim shares < 2 non-stop words with the takeaway
    ([1, 3], NOTES, "Enterprise pricing remains unclear [1][3]", QUIET,
     ("medium", ["2 independent sources", "no verified quote", "authority 2/3"])),
    # 4. overlap-based V: only the unverified claim overlaps; the verified one is off-topic
    ([3, 5], NOTES, TAKEAWAY, QUIET,
     ("low", ["2 independent sources", "no verified quote", "authority 1/3"])),
    # 5. contradiction cap: high conditions, but [6] contradicts on the takeaway's sub-question
    ([1, 3], NOTES, TAKEAWAY, CONTESTED,
     ("medium", ["2 independent sources", "quote verified", "authority 2/3", "contradicted by [6]"])),
    # 6. authority-0 exclusion: the content farm is not an independent source
    ([1, 4], NOTES, TAKEAWAY, QUIET,
     ("medium", ["1 independent source", "quote verified", "authority 2/3"])),
    # 7. D ≥ 2 and V with authority 1 → medium
    ([3, 5], [*NOTES[:4], _verified(5, CLAIM, sub_qs=(1,))], TAKEAWAY, QUIET,
     ("medium", ["2 independent sources", "quote verified", "authority 1/3"])),
    # 8. one unverified source of authority 1 → low
    ([3], NOTES, TAKEAWAY, QUIET,
     ("low", ["1 independent source", "no verified quote", "authority 1/3"])),
    # 9. authority-0 exclusion from V: only the content farm's quote is verified → no V
    ([1, 4], [_n(1, authority=2, verified=False), *NOTES[1:]], TAKEAWAY, QUIET,
     ("low", ["1 independent source", "no verified quote", "authority 2/3"])),
]


@pytest.mark.parametrize("ids,notes,text,smap,expected", R13)
def test_confidence_label_r13_table(ids, notes, text, smap, expected):
    assert confidence_label(ids, FETCHED, notes, text, smap) == expected


def test_the_contradiction_cap_only_lowers_a_high_label_and_reads_the_cited_subquestions():
    assert confidence_label([3], FETCHED, NOTES, TAKEAWAY, CONTESTED) == (
        "low", ["1 independent source", "no verified quote", "authority 1/3", "contradicted by [6]"])
    # [5] sits on sub-question 1, where nothing contradicts
    assert "contradicted by [6]" not in confidence_label([5], FETCHED, NOTES, COST, CONTESTED)[1]


def test_unknown_or_noteless_ids_count_as_no_source():
    assert confidence_label([99], FETCHED, NOTES, TAKEAWAY, QUIET) == ("low", ["no known source cited"])
    assert confidence_label([6], FETCHED, NOTES, TAKEAWAY, QUIET) == ("low", ["no known source cited"])


# ---------------------------------------------------------------------------
# analyst_node
# ---------------------------------------------------------------------------

SECTIONS = [{"title": "Accuracy", "md": "BODY-SENTINEL accuracy prose [1][3]."},
            {"title": "Cost", "md": "Cost prose [5]."}]


def _finding(fid, claim, section, ids) -> dict:
    return {"id": fid, "claim": claim, "citation_urls": [FETCHED[i - 1]["url"] for i in ids],
            "confidence": "high", "section": section, "source_ids": ids, "sentence": claim, "status": "kept"}


FINDINGS = [_finding("f0", "Model X reaches 90 percent accuracy", "Accuracy", [1, 3]),
            _finding("f1", "Model X costs 2 dollars per hour", "Cost", [5])]
OUTLINE = {"sections": [{"title": "Accuracy", "focus": "", "sub_qs": [0], "note_ids": [1, 3]},
                        {"title": "Cost", "focus": "", "sub_qs": [1], "note_ids": [5]}],
           "changed_sections": ["Accuracy", "Cost"]}


def _state(length="medium", *, rc=None, findings=FINDINGS, evidence=NOTES, archetype="explainer", **extra):
    return {"user_request": "How good is Model X?", "length": length, "run_config": rc or make_run_config(length),
            "brief": BRIEF.model_copy(update={"archetype": archetype}).model_dump(), "evidence": evidence,
            "fetched": FETCHED, "coverage": {"0": 2, "1": 1}, "sections": SECTIONS, "findings": findings,
            "outline": OUTLINE, **extra}


REPLY = """### Key takeaways

- Model X reaches about 90 percent accuracy on reasoning tasks [1][3]
- Model X costs 2 dollars per hour to run [5]
- Model X will dominate the market next year

### Where sources agree and disagree

The sources agree on the accuracy figure [1][3].

### Implications

Teams can trial it cheaply [5].

### Open questions and limits

- Long-term costs are not reported.
"""


@pytest.fixture
def extracted(monkeypatch):
    """compose's findings extractor, faked where insight looks it up."""
    calls: list[tuple[str, str]] = []

    def fake(title, md, fetched, *, engines, rnd):
        calls.append((title, md))
        return [{"claim": "Model X reaches about 90 percent accuracy", "citation_urls": [FETCHED[0]["url"]],
                 "confidence": "high", "section": title, "source_ids": [1],
                 "sentence": "Model X reaches about 90 percent accuracy on reasoning tasks [1][3]",
                 "status": "kept"}], [], []
    monkeypatch.setattr(insight, "extract_findings", fake)
    return calls


@pytest.mark.parametrize("rc", [make_run_config("tldr"), make_run_config("medium", **{"compose.analyst": False})])
def test_no_analysis_for_tldr_or_when_the_analyst_is_off(monkeypatch, rc):
    seen = fake_llm(monkeypatch, lambda system, human: pytest.fail("no analyst call"))
    stale = [*FINDINGS, _finding("f2", "an old analysis claim", "Analysis", [1])]
    out = analyst_node(_state(rc["length"], rc=rc, findings=stale))
    assert out == {"analysis": None, "findings": FINDINGS} and seen == []


def test_no_analysis_without_findings(monkeypatch):
    seen = fake_llm(monkeypatch, lambda system, human: pytest.fail("no analyst call"))
    assert analyst_node(_state(findings=[])) == {"analysis": None} and seen == []


def test_the_analyst_reads_findings_stance_map_coverage_and_the_note_index(monkeypatch, extracted):
    seen = fake_llm(monkeypatch, lambda system, human: REPLY)
    analyst_node(_state(evidence=[*NOTES, CONTRA]))
    system, human = seen[0][:2]
    assert system == ANALYST_SYSTEM
    assert "## Accuracy\n- Model X reaches 90 percent accuracy [1][3]" in human
    assert "## Cost\n- Model X costs 2 dollars per hour [5]" in human
    assert "(0) How accurate is Model X? — supports 4, contradicts 1 (sources [6]), mixed 0, background 0" in human
    assert "(0) 2 · (1) 1" in human
    assert "[1] Alpha docs — alpha.com · authority 2 · published 2026-09-01" in human
    assert "[3] Beta report — beta.org · authority 1 · published 2026-08-15" in human
    assert "[6] Delta critique — delta.io" in human, "the contradicting source is indexed"
    assert "[2] Alpha blog" not in human and "[4] Farm page" not in human, "only cited or contradicting notes"
    assert "BODY-SENTINEL" not in human, "never the section bodies"
    for head in ("### Key takeaways", "### Where sources agree and disagree", "### Implications",
                 "### Open questions and limits"):
        assert head in human
    assert "### Scenarios and indicators" not in human, "explainer: no scenarios"


def test_takeaways_are_parsed_labelled_and_uncited_ones_relocated(monkeypatch, extracted):
    fake_llm(monkeypatch, lambda system, human: REPLY)
    out = analyst_node(_state())
    analysis = out["analysis"]
    assert [(t["text"], t["source_ids"], t["confidence"]) for t in analysis["takeaways"]] == [
        ("Model X reaches about 90 percent accuracy on reasoning tasks [1][3]", [1, 3], "high"),
        ("Model X costs 2 dollars per hour to run [5]", [5], "low")]
    assert analysis["takeaways"][0]["reasons"] == ["2 independent sources", "quote verified", "authority 2/3"]
    md = analysis["md"]
    assert md.startswith(ANALYSIS_LABEL)
    assert ("- Model X reaches about 90 percent accuracy on reasoning tasks [1][3] — confidence **high** "
            "(2 independent sources, quote verified, authority 2/3)") in md
    takeaways_part, open_part = md.split("### Key takeaways")[1].split("### Open questions and limits")
    assert "dominate the market" not in takeaways_part
    assert "- Long-term costs are not reported.\n- (uncited) Model X will dominate the market next year" in open_part
    assert (analysis["has_disagreement_section"], analysis["has_scenarios"]) == (True, False)
    assert analysis["stance_map"] == QUIET and analysis["archetype"] == "explainer"
    assert out["errors"] == [] and [c["role"] for c in out["model_calls"]] == ["compose.analyst"]


def test_uncited_takeaways_open_the_questions_section_when_it_is_missing(monkeypatch, extracted):
    reply = "### Key takeaways\n\n- Cited one [1]\n- An uncited guess\n"
    fake_llm(monkeypatch, lambda system, human: reply)
    md = analyst_node(_state())["analysis"]["md"]
    assert md.endswith("### Open questions and limits\n\n- (uncited) An uncited guess")


def test_findings_are_extracted_over_the_unlabelled_analysis_and_merged(monkeypatch, extracted):
    fake_llm(monkeypatch, lambda system, human: REPLY)
    stale = [*FINDINGS, _finding("f2", "an old analysis claim", "Analysis", [1])]
    out = analyst_node(_state(findings=stale))
    (title, md), = extracted
    assert title == "Analysis"
    assert "confidence **" not in md and ANALYSIS_LABEL not in md, "labels are not claims"
    assert "- Model X reaches about 90 percent accuracy on reasoning tasks [1][3]" in md
    assert [(f["id"], f["section"]) for f in out["findings"]] == [
        ("f0", "Accuracy"), ("f1", "Cost"), ("f2", "Analysis")]
    assert out["findings"][2]["claim"] == "Model X reaches about 90 percent accuracy"


def test_the_analysis_findings_come_from_the_real_extractor(monkeypatch):
    """Pins the call contract of compose's ``extract_findings`` (WP-C): the real
    extractor on ``compose.findings_extract`` over a fake driver, with the
    verbatim sentence, its source ids and the kept status."""
    extract = json.dumps({"findings": [{"claim": "Model X reaches about 90 percent accuracy",
                                        "sentence": "Model X reaches about 90 percent accuracy on reasoning "
                                                    "tasks [1][3]",
                                        "source_ids": [1, 3], "confidence": "high"}]})
    seen = fake_llm(monkeypatch, lambda system, human: REPLY if system == ANALYST_SYSTEM else extract)
    out = analyst_node(_state())
    assert [m[0] for m in seen] == [ANALYST_SYSTEM, compose.EXTRACT_SYSTEM]
    assert seen[1][1].startswith("Section: Analysis\n\n### Key takeaways\n\n- Model X reaches about 90 percent")
    assert out["findings"][2:] == [{"id": "f2", "claim": "Model X reaches about 90 percent accuracy",
                                    "citation_urls": [FETCHED[0]["url"], FETCHED[2]["url"]],
                                    "confidence": "high", "section": "Analysis", "source_ids": [1, 3],
                                    "sentence": "Model X reaches about 90 percent accuracy on reasoning tasks [1][3]",
                                    "status": "kept"}]
    assert [c["role"] for c in out["model_calls"]] == ["compose.analyst", "compose.findings_extract"]
    assert out["errors"] == []


def test_a_skipped_analysis_extraction_is_a_declared_skip(monkeypatch):
    def reply(system, human):
        if system == ANALYST_SYSTEM:
            return REPLY
        raise DriverError("provider fake: boom")
    fake_llm(monkeypatch, reply)
    out = analyst_node(_state(revise_rounds=1))
    assert out["analysis"] is not None and [f["section"] for f in out["findings"]] == ["Accuracy", "Cost"]
    assert out["errors"] == ["[r2] skip: compose.findings_extract Analysis (role compose.findings_extract: every "
                             "engine failed after 1 attempt (last: DriverError: provider fake: boom))"]


def test_list_markers_are_normalised_before_the_takeaways_are_parsed(monkeypatch, extracted):
    fake_llm(monkeypatch, lambda system, human: "### Key takeaways\n\n- Model X is accurate [1, 3]\n")
    takeaways = analyst_node(_state())["analysis"]["takeaways"]
    assert [(t["text"], t["source_ids"]) for t in takeaways] == [("Model X is accurate [1][3]", [1, 3])], (
        "a [1, 2] takeaway is cited, not relocated as uncited")


def test_a_label_only_analysis_takes_omit_analysis(monkeypatch, extracted):
    fake_llm(monkeypatch, lambda system, human: "### Key takeaways\n\n### Open questions and limits\n")
    out = analyst_node(_state())
    assert out["analysis"] is None and "draft_md" not in out and extracted == []
    assert out["errors"] == ["[r1] degrade: compose.analyst → omit_analysis (no analysis content)"]


def test_no_extraction_when_the_analysis_cites_nothing(monkeypatch, extracted):
    fake_llm(monkeypatch, lambda system, human: "### Key takeaways\n\n- A guess\n")
    out = analyst_node(_state())
    assert extracted == [] and [f["id"] for f in out["findings"]] == ["f0", "f1"]


def test_the_interim_draft_has_the_analysis_and_no_tldr(monkeypatch, extracted):
    fake_llm(monkeypatch, lambda system, human: REPLY)
    draft = analyst_node(_state())["draft_md"]
    assert draft.startswith("## Accuracy\n\nBODY-SENTINEL")
    assert draft.index("## Cost") < draft.index("## Analysis\n\n" + ANALYSIS_LABEL) < draft.index("## Sources")
    assert "TL;DR" not in draft
    assert "5. [Gamma post](https://gamma.net/e)" in draft


SHORT_REPLY = "\n".join([
    "### Key takeaways", *[f"- takeaway {i} [{1 + i % 2}]" for i in range(1, 7)],
    "### Where sources agree and disagree", "They agree [1].",
    "### Scenarios and indicators", "Base case [1].",
    "### Implications", "Do it [1].",
    "### Open questions and limits", *[f"- question {i}" for i in range(1, 6)]])


def _parts(md: str) -> list[str]:
    return [line[4:] for line in md.splitlines() if line.startswith("### ")]


def test_the_short_variant_is_compacted(monkeypatch, extracted):
    seen = fake_llm(monkeypatch, lambda system, human: SHORT_REPLY)
    out = analyst_node(_state("short"))
    analysis = out["analysis"]
    human = seen[0][1]
    assert "### Implications" not in human and "### Where sources agree and disagree" not in human
    assert "at most 4" in human and "at most 3" in human
    assert [t["text"] for t in analysis["takeaways"]] == [f"takeaway {i} [{1 + i % 2}]" for i in range(1, 5)]
    assert _parts(analysis["md"]) == ["Key takeaways", "Open questions and limits"]
    assert analysis["md"].count("- question") == 3
    assert (analysis["has_disagreement_section"], analysis["has_scenarios"]) == (False, False)
    assert out["errors"] == ["analysis: short variant kept 4 of 6 takeaways, 3 of 5 open questions"]


def test_the_short_cap_counts_relocated_uncited_takeaways_as_open_questions(monkeypatch, extracted):
    reply = ("### Key takeaways\n\n- One [1]\n- Two [3]\n- Guess a\n- Guess b\n- Guess c\n\n"
             "### Open questions and limits\n\n- Q1\n- Q2\n")
    fake_llm(monkeypatch, lambda system, human: reply)
    out = analyst_node(_state("short"))
    assert out["analysis"]["md"].endswith("- Q1\n- Q2\n- (uncited) Guess a")
    assert out["errors"] == ["analysis: short variant kept 2 of 2 takeaways, 3 of 5 open questions"]


def test_the_short_variant_records_nothing_when_nothing_is_trimmed(monkeypatch, extracted):
    reply = "### Key takeaways\n\n- One [1]\n\n### Open questions and limits\n\n- Q1\n"
    fake_llm(monkeypatch, lambda system, human: reply)
    assert analyst_node(_state("short"))["errors"] == []


def test_the_short_variant_keeps_the_disagreement_when_sources_conflict(monkeypatch, extracted):
    seen = fake_llm(monkeypatch, lambda system, human: SHORT_REPLY)
    analysis = analyst_node(_state("short", evidence=[*NOTES, CONTRA]))["analysis"]
    assert "### Where sources agree and disagree" in seen[0][1]
    assert _parts(analysis["md"]) == ["Key takeaways", "Where sources agree and disagree", "Open questions and limits"]
    assert analysis["has_disagreement_section"] is True


def test_the_short_variant_keeps_scenarios_for_a_forecast(monkeypatch, extracted):
    seen = fake_llm(monkeypatch, lambda system, human: SHORT_REPLY)
    analysis = analyst_node(_state("short", archetype="forecast"))["analysis"]
    assert "### Scenarios and indicators" in seen[0][1]
    assert _parts(analysis["md"]) == ["Key takeaways", "Scenarios and indicators", "Open questions and limits"]
    assert analysis["has_scenarios"] is True


def test_sectioned_lengths_keep_every_section(monkeypatch, extracted):
    fake_llm(monkeypatch, lambda system, human: SHORT_REPLY)
    analysis = analyst_node(_state("long", archetype="decision"))["analysis"]
    assert len(analysis["takeaways"]) == 6 and analysis["md"].count("- question") == 5
    assert _parts(analysis["md"]) == ["Key takeaways", "Where sources agree and disagree",
                                      "Scenarios and indicators", "Implications", "Open questions and limits"]


def test_heading_variants_are_recognised(monkeypatch, extracted):
    reply = ("## Key Takeaways:\n\n1. Cited [1]\n\n#### Where sources agree & disagree\n\nThey agree [1].\n"
             "### Key takeaways\n\n• Second [3]\n")                  # a repeated heading continues the first
    fake_llm(monkeypatch, lambda system, human: reply)
    analysis = analyst_node(_state())["analysis"]
    assert [t["text"] for t in analysis["takeaways"]] == ["Cited [1]", "Second [3]"]
    assert _parts(analysis["md"]) == ["Key takeaways", "Where sources agree and disagree"]


@pytest.mark.parametrize("heading", ["### 1. Key takeaways", "## 2) Key takeaways", "**Key takeaways**",
                                     "**Key takeaways:**", "**Key takeaways**:", "### Takeaways"])
def test_numbered_bold_and_short_takeaways_headings_are_recognised(monkeypatch, extracted, heading):
    reply = f"{heading}\n\n- Cited [1]\n\n### Open questions and limits\n\n- Q1\n"
    fake_llm(monkeypatch, lambda system, human: reply)
    out = analyst_node(_state())
    assert [t["text"] for t in out["analysis"]["takeaways"]] == ["Cited [1]"]
    assert _parts(out["analysis"]["md"]) == ["Key takeaways", "Open questions and limits"]
    assert out["errors"] == []


@pytest.mark.parametrize("before", [
    "Okay, the user wants an analysis. Let me think…\n\n",
    "### Thinking\n\nThe sources mostly agree, so lead with accuracy.\n\n",
    "Okay, the user wants an analysis. Let me think…\n\n### Thinking\n\nLead with accuracy.\n\n",
])
def test_text_before_the_first_fixed_heading_is_dropped_and_recorded(monkeypatch, extracted, before):
    fake_llm(monkeypatch, lambda system, human: before + REPLY)
    out = analyst_node(_state())
    md = out["analysis"]["md"]
    assert md.startswith(ANALYSIS_LABEL + "\n\n### Key takeaways\n\n")
    assert "Let me think" not in md and "Thinking" not in md and "Lead with accuracy" not in md
    assert out["errors"] == ["analysis: dropped text before the first heading"]
    (_, plain), = extracted
    assert "Let me think" not in plain and "Thinking" not in plain, "nor extracted as findings"


def test_other_headings_after_the_first_fixed_one_stay(monkeypatch, extracted):
    reply = "### Key takeaways\n\n- Cited [1]\n\n### Caveats\n\nSmall samples [3].\n"
    fake_llm(monkeypatch, lambda system, human: reply)
    out = analyst_node(_state())
    assert _parts(out["analysis"]["md"]) == ["Key takeaways", "Caveats"]
    assert "Small samples [3]." in out["analysis"]["md"] and out["errors"] == []


@pytest.mark.parametrize("reply", [
    "Model X looks accurate [1].",
    "### Thinking\n\nModel X looks accurate [1].\n\n### Summary\n\n- It is accurate [3]\n",
    "**Summary**\n\n- It is accurate [3]\n",                       # bold, but not a fixed heading
])
def test_a_reply_without_the_fixed_headings_omits_the_analysis_visibly(monkeypatch, extracted, reply):
    fake_llm(monkeypatch, lambda system, human: reply)
    events: list = []
    stale = [*FINDINGS, _finding("f2", "an old analysis claim", "Analysis", [1])]
    out = analyst_node(_state(findings=stale), Runtime(stream_writer=events.append))
    assert out["analysis"] is None and out["findings"] == FINDINGS and "draft_md" not in out
    assert out["errors"] == ["[r1] degrade: compose.analyst → omit_analysis (no analysis headings)"]
    assert events == [{"kind": "degrade", "node": "analyst",
                       "data": {"role": "compose.analyst", "degradation": "omit_analysis",
                                "reason": "no analysis headings"}}]
    assert extracted == [] and [c["role"] for c in out["model_calls"]] == ["compose.analyst"]


def test_a_missing_takeaways_heading_is_recorded_and_the_text_kept(monkeypatch, extracted):
    reply = "### Where sources agree and disagree\n\nModel X looks accurate [1]."
    fake_llm(monkeypatch, lambda system, human: reply)
    out = analyst_node(_state())
    assert out["errors"] == ["analysis: takeaways heading missing"]
    assert out["analysis"]["takeaways"] == [] and "Model X looks accurate [1]." in out["analysis"]["md"]


@pytest.mark.parametrize("body", ["- A guess\n- Another guess\n", "Prose without bullets [1].\n"])
def test_a_takeaways_part_without_a_cited_bullet_is_recorded(monkeypatch, extracted, body):
    fake_llm(monkeypatch, lambda system, human: f"### Key takeaways\n\n{body}")
    out = analyst_node(_state())
    assert out["analysis"]["takeaways"] == []
    assert out["errors"] == ["analysis: no cited takeaways"]


def test_an_empty_reply_omits_the_analysis_visibly(monkeypatch, extracted):
    fake_llm(monkeypatch, lambda system, human: "  ")
    out = analyst_node(_state())
    assert out["analysis"] is None
    assert out["errors"] == ["[r1] degrade: compose.analyst → omit_analysis (empty reply)"]
    assert "draft_md" not in out and [c["role"] for c in out["model_calls"]] == ["compose.analyst"]


def test_degrade_omits_the_analysis_and_records_it(monkeypatch, extracted):
    def boom(system, human):
        raise DriverError("provider fake: boom")
    fake_llm(monkeypatch, boom)
    events: list = []
    stale = [*FINDINGS, _finding("f2", "an old analysis claim", "Analysis", [1])]
    out = analyst_node(_state(findings=stale, revise_rounds=1), Runtime(stream_writer=events.append))
    assert out["analysis"] is None and out["findings"] == FINDINGS and "draft_md" not in out
    assert out["errors"] == ["[r2] degrade: compose.analyst → omit_analysis (role compose.analyst: every engine "
                             "failed after 1 attempt (last: DriverError: provider fake: boom))"]
    assert [(c["role"], c["ok"]) for c in out["model_calls"]] == [("compose.analyst", False)]
    assert events == [{"kind": "degrade", "node": "analyst",
                       "data": {"role": "compose.analyst", "degradation": "omit_analysis",
                                "reason": out["errors"][0].split(" (", 1)[1][:-1]}}]


def test_the_analysis_event(monkeypatch, extracted):
    fake_llm(monkeypatch, lambda system, human: REPLY)
    events: list = []
    analyst_node(_state(), Runtime(stream_writer=events.append))
    assert events == [{"kind": "analysis", "node": "analyst",
                       "data": {"takeaways": 2, "has_disagreement": True, "has_scenarios": False}}]


def test_the_analysts_footnote_definitions_are_removed_and_recorded(monkeypatch, extracted):   # re-review (e)
    reply = REPLY.replace("reasoning tasks [1][3]", "reasoning tasks [^1][3]") + "\n[^1]: https://docs.alpha.com/a\n"
    fake_llm(monkeypatch, lambda system, human: reply)
    out = analyst_node(_state())
    md = out["analysis"]["md"]
    assert "[^1]" not in md and "https://docs.alpha.com/a" not in md
    assert out["analysis"]["takeaways"][0]["text"] == (
        "Model X reaches about 90 percent accuracy on reasoning tasks [1][3]")
    assert out["errors"] == ['[r1] cite: removed footnote definition [^1] in section "Analysis"'], "as compose does"
