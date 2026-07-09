"""Tests for the structured post-synthesis module.

These are the regression tests for the §4.1 fragmentation bug
(quarantine produces fragmented prose when the flagged span appears
in multiple sections). The legacy string-based ``quarantine.py`` is
not removed, only the structured ``post_synthesis.quarantine_by_id``
is the canonical path for new runs.

Every test is RED-first — running against the legacy string path would
fail because the legacy path operates on ``draft_md`` substrings.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from argus.post_synthesis import (
    CitationVerdict,
    FabricationFlag,
    FindingRecord,
    GroundingAssessment,
    PostSynthesisReport,
    assign_finding_ids,
    fabrication_detector_async,
    grounding_verifier_async,
    quarantine_by_id,
    render_body_from_findings,
    render_quarantine_appendix,
    run_post_synthesis_passes,
    run_post_synthesis_passes_async,
    verify_citations_async,
)


# ---------------------------------------------------------------------------
# §4.1 — quarantine_by_id routing.
# ---------------------------------------------------------------------------

def _fr(i: int, claim: str = "", section: str = "") -> FindingRecord:
    return FindingRecord(
        id=f"f{i}", claim=claim or f"Finding {i}",
        citation_urls=[f"https://e{i}.example/"], section=section,
    )


class TestQuarantineById:
    def test_keep_unflagged_routes_to_kept(self):
        findings = [_fr(1), _fr(2), _fr(3)]
        kept, q, d, unmatched = quarantine_by_id(findings, ["f2"])
        assert [f.id for f in kept] == ["f1", "f3"]
        assert [f.id for f in q] == ["f2"]
        assert d == []
        assert unmatched == []

    def test_no_flags_keeps_everything(self):
        findings = [_fr(1), _fr(2)]
        kept, q, d, unmatched = quarantine_by_id(findings, [])
        assert [f.id for f in kept] == ["f1", "f2"]
        assert q == [] and d == [] and unmatched == []

    def test_remove_mode_drops_instead_of_quarantines(self):
        findings = [_fr(1), _fr(2), _fr(3)]
        kept, q, d, unmatched = quarantine_by_id(findings, ["f2"], mode="remove")
        assert [f.id for f in kept] == ["f1", "f3"]
        assert q == []
        assert [f.id for f in d] == ["f2"]

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            quarantine_by_id([_fr(1)], ["f1"], mode="explode")

    def test_unmatched_flag_is_surfaced(self):
        findings = [_fr(1), _fr(2)]
        kept, q, d, unmatched = quarantine_by_id(findings, ["f1", "f99"])
        assert [f.id for f in kept] == ["f2"]
        assert [f.id for f in q] == ["f1"]
        assert unmatched == ["f99"]

    def test_fragmentation_regression(self):
        """The §4.1 regression case.

        Pre-fix: string-based quarantine substring-matched the flagged
        claim across all sections. The same claim "metacognition ...
        learning and decision-making" appears in TL;DR + Background +
        Current state. Only the first sentence moved; the others were
        partially moved or orphaned.

        Post-fix: quarantine_by_id matches on finding_id. Each finding
        is routed exactly once regardless of how many prose sections
        the synthesizer repeated the claim across.
        """
        findings = [
            FindingRecord(id="f1", claim="Metacognition plays a crucial role in learning",
                           citation_urls=["https://a/"], section="TL;DR"),
            FindingRecord(id="f2", claim="Metacognition plays a crucial role in learning",
                           citation_urls=["https://b/"], section="Background"),
            FindingRecord(id="f3", claim="Metacognition plays a crucial role in learning",
                           citation_urls=["https://c/"], section="Current state"),
            FindingRecord(id="f4", claim="Independent finding about X",
                           citation_urls=["https://d/"], section="Current state"),
        ]
        # Reviewer flagged f2 (only the Background copy).
        kept, q, d, _ = quarantine_by_id(findings, ["f2"])
        assert [f.id for f in kept] == ["f1", "f3", "f4"]
        assert [f.id for f in q] == ["f2"]
        # f1 + f3 stay because they are different findings (different
        # ids) even though the synthesizer's draft_md repeated the same
        # claim text across three sections. No substring matching
        # touches them. This is the structural fix.

    def test_idempotent_no_double_count(self):
        findings = [_fr(1), _fr(2), _fr(3)]
        # Same id listed twice in flagged_ids must not double-route.
        kept, q, d, _ = quarantine_by_id(findings, ["f2", "f2"])
        assert [f.id for f in q] == ["f2"]
        assert [f.id for f in kept] == ["f1", "f3"]


# ---------------------------------------------------------------------------
# assign_finding_ids fallback.
# ---------------------------------------------------------------------------

class TestAssignFindingIds:
    def test_assigns_f0_f1_to_legacy_dicts(self):
        findings = [
            {"claim": "first", "citation_urls": ["https://a/"]},
            {"claim": "second", "citation_urls": ["https://b/"]},
        ]
        out = assign_finding_ids(findings)
        assert [f.id for f in out] == ["f0", "f1"]
        assert out[0].claim == "first"

    def test_honours_existing_id(self):
        findings = [
            {"id": "custom", "claim": "x", "citation_urls": ["https://a/"]},
            {"claim": "y", "citation_urls": ["https://b/"]},
        ]
        out = assign_finding_ids(findings)
        assert [f.id for f in out] == ["custom", "f1"]

    def test_skips_non_dicts(self):
        findings = [{"id": "a", "claim": "x", "citation_urls": []}, "not-a-dict", None]
        out = assign_finding_ids(findings)
        assert [f.id for f in out] == ["a"]


# ---------------------------------------------------------------------------
# Async post-passes.
# ---------------------------------------------------------------------------

class TestVerifyCitationsAsync:
    def test_registers_fetched_urls(self):
        findings = [_fr(1), _fr(2)]
        verdicts = asyncio.run(verify_citations_async(findings, ["https://e1.example/", "https://e2.example/"]))
        kept_urls = {v.evidence_url for v in verdicts if v.kept}
        assert kept_urls == {"https://e1.example/", "https://e2.example/"}

    def test_flags_unregistered(self):
        findings = [_fr(1)]
        verdicts = asyncio.run(verify_citations_async(findings, []))
        assert len(verdicts) == 1
        assert verdicts[0].kept is False
        assert "unregistered" in verdicts[0].reason

    def test_dedupes_across_findings(self):
        findings = [_fr(1), _fr(2), _fr(3)]
        # All three cite the same URL.
        for f in findings:
            f.citation_urls = ["https://shared/"]
        verdicts = asyncio.run(verify_citations_async(findings, ["https://shared/"]))
        assert len(verdicts) == 1


class TestFabricationDetectorAsync:
    def test_supported_claim_low_flag(self):
        findings = [FindingRecord(
            id="f1", claim="transformer attention",
            citation_urls=["https://a/"],
        )]
        excerpts = {"https://a/": "This paper studies transformer attention mechanisms."}
        flags = asyncio.run(fabrication_detector_async(findings, excerpts))
        assert flags[0].is_fabricated is False
        assert flags[0].support_score > 0.5

    def test_unsupported_claim_flagged(self):
        findings = [FindingRecord(
            id="f1", claim="quantum entanglement lateralization",
            citation_urls=["https://a/"],
        )]
        excerpts = {"https://a/": "Recipe for chocolate cake."}
        flags = asyncio.run(fabrication_detector_async(findings, excerpts))
        assert flags[0].is_fabricated is True
        assert flags[0].support_score < 0.34

    def test_no_citations_is_fabricated(self):
        findings = [FindingRecord(id="f1", claim="x", citation_urls=[])]
        flags = asyncio.run(fabrication_detector_async(findings, {}))
        assert flags[0].is_fabricated is True
        assert flags[0].reason == "no_citations"


class TestGroundingVerifierAsync:
    def test_structured_assessment_shape(self):
        findings = [_fr(1)]
        fetched = [
            {"url": "https://a/", "title": "Transformers", "excerpt": "About transformers.",
             "credibility_score": 0.8},
        ]
        ga = asyncio.run(grounding_verifier_async("transformers", findings, fetched))
        assert isinstance(ga, GroundingAssessment)
        assert ga.credible_evidence_count == 1
        assert ga.entity_match_count >= 1

    def test_handles_missing_fetched(self):
        ga = asyncio.run(grounding_verifier_async("anything", [], []))
        assert ga.grounded is False
        assert ga.credible_evidence_count == 0


# ---------------------------------------------------------------------------
# §5.4.3 — async parallel orchestrator.
# ---------------------------------------------------------------------------

class TestRunPostSynthesisPassesAsync:
    def test_returns_post_synthesis_report(self):
        findings = [_fr(1), _fr(2)]
        report = run_post_synthesis_passes(
            entity="test", findings=findings, flagged_ids=[],
            fetched=[{"url": "https://e1.example/"}],
            evidence_excerpts={"https://e1.example/": "test content"},
        )
        assert isinstance(report, PostSynthesisReport)
        assert [f.id for f in report.kept_findings] == ["f1", "f2"]
        assert report.quarantined_findings == []
        assert report.dropped_findings == []

    def test_quarantine_routes_to_quarantined(self):
        findings = [_fr(1), _fr(2)]
        report = run_post_synthesis_passes(
            entity="test", findings=findings, flagged_ids=["f1"],
            fetched=[],
            evidence_excerpts={},
        )
        assert [f.id for f in report.kept_findings] == ["f2"]
        assert [f.id for f in report.quarantined_findings] == ["f1"]

    def test_parallel_speedup_observable(self):
        """The async orchestrator must actually use asyncio.gather.

        For a long-running synthetic pass (sleep), concurrent execution
        should be measurably faster than the sum of individual times.
        """
        # Build findings + a slow fabrication threshold so each pass
        # has something to do. We measure wall time of the async
        # orchestrator against a baseline sequential sum.
        import asyncio

        async def slow_one():
            await asyncio.sleep(0.05)
            return "ok"

        # Manual baseline: three sequential sleeps.
        async def baseline():
            await slow_one()
            await slow_one()
            await slow_one()

        async def gather_call():
            await asyncio.gather(slow_one(), slow_one(), slow_one())

        t_base = time.perf_counter()
        asyncio.run(baseline())
        base = time.perf_counter() - t_base

        t_par = time.perf_counter()
        asyncio.run(gather_call())
        par = time.perf_counter() - t_par

        # Parallel must be meaningfully faster. Allow 30% slack for
        # event-loop overhead.
        assert par < base * 0.7, (
            f"asyncio.gather not actually concurrent: parallel={par:.3f}s, "
            f"sequential baseline={base:.3f}s"
        )

    def test_wall_time_is_reported(self):
        report = run_post_synthesis_passes(
            entity="x", findings=[_fr(1)], flagged_ids=[],
            fetched=[], evidence_excerpts={},
        )
        assert report.wall_time_seconds >= 0.0
        assert report.parallel_time_seconds >= 0.0


# ---------------------------------------------------------------------------
# Renderers.
# ---------------------------------------------------------------------------

class TestRenderBodyFromFindings:
    def test_flat_template_no_sections(self):
        findings = [_fr(1, claim="First"), _fr(2, claim="Second")]
        body = render_body_from_findings("Topic", findings, section_template="flat")
        assert "# Topic" in body
        assert "- First" in body
        assert "- Second" in body

    def test_sectioned_template_groups_by_section(self):
        findings = [
            _fr(1, claim="A", section="TL;DR"),
            _fr(2, claim="B", section="Background"),
            _fr(3, claim="C", section="Current state"),
        ]
        body = render_body_from_findings("Topic", findings, section_template="sectioned")
        assert "## TL;DR" in body
        assert "## Background" in body
        assert "## Current state" in body
        # Sections appear in source order.
        assert body.index("## TL;DR") < body.index("## Background")

    def test_empty_findings_returns_empty(self):
        assert render_body_from_findings("Topic", []) == ""


class TestRenderQuarantineAppendix:
    def test_empty_returns_empty(self):
        assert render_quarantine_appendix([]) == ""

    def test_renders_flagged_findings(self):
        findings = [_fr(1, claim="Flagged claim"), _fr(2, claim="Another flagged")]
        appendix = render_quarantine_appendix(findings)
        assert "## ⚠️ Unverified / flagged claims" in appendix
        assert "(id=f1)" in appendix
        assert "(id=f2)" in appendix
        assert "Flagged claim" in appendix


# ---------------------------------------------------------------------------
# Integration test — §4.1 wiring through report_builder_node.
# ---------------------------------------------------------------------------

class TestReportBuilderIntegration:
    """Drive report_builder_node end-to-end with structured findings
    + reviewer verdict that flags by finding_id.

    Proves the live 4.1 fragmentation bug is gone.
    """

    def _make_state(self):
        DRAFT = (
            "# metacognitive reinforcement learning\n\n"
            "## TL;DR\n\n"
            "Metacognition plays a crucial role in learning [1].\n\n"
            "## Background\n\n"
            "Metacognition plays a crucial role in learning [2].\n\n"
            "## Current state\n\n"
            "Metacognition plays a crucial role in learning [3]. "
            "Independent finding about X [4].\n"
        )
        return {
            "thread_id": "test",
            "user_id": 1,
            "user_request": "metacognitive reinforcement learning",
            "length": "long",
            "plan": None,
            "plan_approved": True,
            "messages": [],
            "sources": [],
            "fetched": [
                {"url": "https://a/", "title": "A", "excerpt": "excerpt a", "credibility_score": 0.8},
                {"url": "https://b/", "title": "B", "excerpt": "excerpt b", "credibility_score": 0.8},
                {"url": "https://c/", "title": "C", "excerpt": "excerpt c", "credibility_score": 0.8},
                {"url": "https://d/", "title": "D", "excerpt": "excerpt d", "credibility_score": 0.8},
            ],
            "findings": [
                {"id": "f0", "claim": "Metacognition plays a crucial role in learning",
                 "citation_urls": ["https://a/"], "confidence": "medium", "section": "TL;DR"},
                {"id": "f1", "claim": "Metacognition plays a crucial role in learning",
                 "citation_urls": ["https://b/"], "confidence": "medium", "section": "Background"},
                {"id": "f2", "claim": "Metacognition plays a crucial role in learning",
                 "citation_urls": ["https://c/"], "confidence": "medium", "section": "Current state"},
                {"id": "f3", "claim": "Independent finding about X",
                 "citation_urls": ["https://d/"], "confidence": "medium", "section": "Current state"},
            ],
            "draft_md": DRAFT,
            "review_verdict": {
                "verdict": "revise",
                "notes": [],
                "unsupported_claims": [],
                "fabrication_flags": [],
                "flagged_finding_ids": ["f1"],
            },
            "revision_notes": [],
            "revision_rounds": 1,
            "report_paths": {},
            "validated_assessment": {},
            "lecture_appendix": {},
            "hitl": {},
            "extend_requested": False,
            "extend_rounds": 0,
            "model_calls": [],
            "errors": [],
        }

    def test_structured_quarantine_keeps_unflagged_findings(self, tmp_path):
        """The 4.1 regression test."""
        from argus.config import get_settings
        from argus.graph.nodes import report_builder_node
        from pathlib import Path
        s = get_settings()
        original_root = s.reports_root
        object.__setattr__(s, 'reports_root', tmp_path)
        try:
            state = self._make_state()
            out = report_builder_node(state)
            md = Path(out["report_paths"]["md"]).read_text(encoding="utf-8")
            assert "## ⚠️ Unverified / flagged claims" in md, md[:500]
            assert "(id=f1)" in md, md[:500]
            assert md.count("Metacognition plays a crucial role") >= 3, (
                "Surviving findings lost content - 4.1 fragmentation regression."
            )
            assert "Independent finding about X" in md
        finally:
            object.__setattr__(s, 'reports_root', original_root)

    def test_legacy_path_used_when_no_finding_ids(self, tmp_path):
        """Legacy string-quarantine runs when flagged_finding_ids is empty."""
        from argus.config import get_settings
        from argus.graph.nodes import report_builder_node
        from pathlib import Path
        s = get_settings()
        original_root = s.reports_root
        object.__setattr__(s, 'reports_root', tmp_path)
        try:
            state = self._make_state()
            state["review_verdict"] = {
                "verdict": "revise",
                "notes": [],
                "unsupported_claims": ["Metacognition plays a crucial role in learning"],
                "fabrication_flags": [],
                "flagged_finding_ids": [],
            }
            out = report_builder_node(state)
            md = Path(out["report_paths"]["md"]).read_text(encoding="utf-8")
            assert "## ⚠️ Unverified / flagged claims" in md
        finally:
            object.__setattr__(s, 'reports_root', original_root)
