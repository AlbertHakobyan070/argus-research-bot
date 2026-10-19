"""v4 WP-0 / WP-C — graph/assemble.py (spec §2.8).

The two EXPECTED strings were captured from compose_node's inline
assembly at commit 012a756 (before the helpers moved), by driving the
real sectioned path with the writer/TL;DR/extraction LLM calls stubbed.
"""
import pytest

from argus.graph.assemble import (
    assemble_draft, normalize_cite_markers, strip_fences)

SECTIONS = [
    {"title": "Background",
     "md": "Alpha was first described in 2019 [1]. Later work refined it [3]."},
    {"title": "Current state",
     "md": "Two vendors ship it today [2][3].\n\n### Details\n\n"
           "An uncited aside, and a stale id [9]."},
]
FETCHED = [
    {"url": "https://example.org/alpha", "title": "Alpha: the original paper"},
    {"url": "https://example.com/vendors", "title": ""},
    {"url": "https://docs.example.net/guide",
     "title": "A very long title " + "x" * 100},
]

_SOURCES = (
    "## Sources\n\n"
    "1. [Alpha: the original paper](https://example.org/alpha)\n"
    "2. [https://example.com/vendors](https://example.com/vendors)\n"
    "3. [A very long title " + "x" * 72 + "](https://docs.example.net/guide)\n")
EXPECTED_NO_TLDR = (
    "## Background\n\nAlpha was first described in 2019 [1]. Later work "
    "refined it [3].\n\n## Current state\n\nTwo vendors ship it today "
    "[2][3].\n\n### Details\n\nAn uncited aside, and a stale id [9].\n\n"
    + _SOURCES)
EXPECTED_WITH_TLDR = "## TL;DR\n\nAlpha is a thing [1].\n\n" + EXPECTED_NO_TLDR


def test_assembly_identical_to_pre_split_output():
    assert assemble_draft(tldr=None, sections=SECTIONS, analysis_md=None,
                          fetched=FETCHED, flat=False) == EXPECTED_NO_TLDR


def test_assembly_with_tldr_identical_to_pre_split_output():
    assert assemble_draft(tldr="  Alpha is a thing [1].  ", sections=SECTIONS,
                          analysis_md=None, fetched=FETCHED,
                          flat=False) == EXPECTED_WITH_TLDR


def test_blank_tldr_and_analysis_are_omitted():
    assert assemble_draft(tldr="  ", sections=SECTIONS, analysis_md="\n",
                          fetched=FETCHED, flat=False) == EXPECTED_NO_TLDR


def test_analysis_follows_sections_and_feeds_sources():
    md = assemble_draft(tldr="T [1].", sections=SECTIONS[:1],
                        analysis_md="Takeaway [2].", fetched=FETCHED, flat=False)
    order = [md.index(h) for h in
             ("## TL;DR", "## Background", "## Analysis", "## Sources")]
    assert order == sorted(order)
    assert "2. [https://example.com/vendors]" in md   # cited only in Analysis


def test_tldr_heading_is_configurable():
    # Lecture passes "Executive TL;DR", the heading compose writes for it today.
    md = assemble_draft(tldr="Alpha is a thing [1].", sections=SECTIONS,
                        analysis_md=None, fetched=FETCHED, flat=False,
                        tldr_heading="Executive TL;DR")
    assert md == ("## Executive TL;DR\n\nAlpha is a thing [1].\n\n"
                  + EXPECTED_NO_TLDR)


def test_appendices_render_before_sources_and_feed_them():
    md = assemble_draft(
        tldr=None, sections=SECTIONS[:1], analysis_md="Takeaway [1].",
        appendices=["## Uncited specifics\n\n- It grew 40% in 2024 [2].\n",
                    "  ",                                   # blank: omitted
                    "## Flagged claims\n\n- A cut claim [9]."],
        fetched=FETCHED, flat=False)
    # [2] is cited only in an appendix and still gets its Sources entry.
    assert md == (
        "## Background\n\nAlpha was first described in 2019 [1]. Later work "
        "refined it [3].\n\n## Analysis\n\nTakeaway [1].\n\n"
        "## Uncited specifics\n\n- It grew 40% in 2024 [2].\n\n"
        "## Flagged claims\n\n- A cut claim [9].\n\n" + _SOURCES)


def test_flat_renders_findings_heading():
    md = assemble_draft(tldr="T [1].", sections=[{"title": "Report",
                                                  "md": "Body [3]."}],
                        analysis_md="A [1].", fetched=FETCHED, flat=True)
    assert md == ("## TL;DR\n\nT [1].\n\n## Findings\n\nBody [3].\n\n"
                  "## Analysis\n\nA [1].\n\n## Sources\n\n"
                  "1. [Alpha: the original paper](https://example.org/alpha)\n"
                  "3. [A very long title " + "x" * 72
                  + "](https://docs.example.net/guide)\n")


@pytest.mark.parametrize("raw,clean", [
    ("```markdown\nHello [1]\n```", "Hello [1]"),
    ("```MD\nHello\n```  ", "Hello"),
    ("  plain text ", "plain text"),
])
def test_strip_fences(raw, clean):
    assert strip_fences(raw) == clean


@pytest.mark.parametrize("raw,clean", [
    ("Rates rose 【4】.", "Rates rose [4]."),
    ("Rates rose【2†L1-L4】.", "Rates rose[2]."),
    ("Rates rose ［3］ and 〔5〕.", "Rates rose [3] and [5]."),
    ("Rates rose 【1, 2】.", "Rates rose [1][2]."),
    ("Rates rose [1, 2].", "Rates rose [1][2]."),
    ("Rates rose [1; 2; 7].", "Rates rose [1][2][7]."),
    ("Rates rose [1-3].", "Rates rose [1][2][3]."),
    ("Rates rose [4–6].", "Rates rose [4][5][6]."),
    ("Rates rose [^2].", "Rates rose [2]."),
    ("Rates rose [Source 3].", "Rates rose [3]."),
    ("Rates rose [sources 3, 4].", "Rates rose [3][4]."),
])
def test_normalize_cite_markers_table(raw, clean):
    assert normalize_cite_markers(raw) == clean


@pytest.mark.parametrize("md", [
    "Rates rose [2][5] and [12].",
    "An array `[1, 2]` and a fence:\n```\nx = 【4】\n```\n",
    "A wide span [1-40] is not a citation range.",
    "Years [2019, 2020] are not source ids.",
])
def test_normalize_cite_markers_leaves_the_rest(md):
    assert normalize_cite_markers(md) == md
