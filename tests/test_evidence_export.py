"""v4 WP-C — evidence.json (spec §2.6), written by report_builder next to
report.md and validated by its own pydantic model (schema 1). It never
carries a markdown_path, a secret or an env value; its ``degradations``
are exactly the ``skip:`` / ``degrade:`` / ``legacy:`` lines the bench
counts from ``errors``.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from argus.graph.evidence_export import Evidence, write_evidence
from bench import metrics
from helpers import make_run_config

FETCHED = [
    {"url": "https://docs.example.com/guide", "title": "Guide", "markdown_path": "A:/cache/1.md",
     "canonical_url": "https://docs.example.com/guide", "domain": "docs.example.com", "provider": "exa",
     "fetcher": "exa_text", "published": "2026-09-01", "chars": 12345, "credibility_score": 0.81,
     "authority_prior": 3, "pinned": False},
    {"url": "file:///V/notes/clip.txt", "title": "My clip", "markdown_path": "A:/cache/2.md", "pinned": True,
     "provider": "appended-asset"},
]
NOTES = [{"source_id": 1, "source_url": FETCHED[0]["url"], "title": "Guide", "sub_qs": [0, 1], "relevance": 5,
          "stance": "supports", "source_type": "official_doc", "authority": 3, "cached": False,
          "claims": [{"text": "It pauses", "quote": "pauses the run", "confidence": "high", "quote_verified": True}]}]
ERRORS = ["[r1] skip: research.digest https://x.example (role research.digest: every engine failed after 3 "
          "attempts (last: DriverError: 429))",
          "analysis: dropped text before the first heading",
          "legacy: run_config defaults applied (pre-v4 checkpoint)",
          "[r2] degrade: compose.analyst → omit_analysis (empty reply)",
          "[r2] pdf: RuntimeError: boom"]
FINDINGS = [{"id": "f0", "claim": "It pauses", "citation_urls": [FETCHED[0]["url"]], "confidence": "high",
             "section": "Mechanism", "source_ids": [1], "sentence": "It pauses the run [1].", "status": "kept"},
            {"id": "f1", "claim": "It was built in 1999", "citation_urls": [], "confidence": "low",
             "section": "Analysis", "source_ids": [9], "sentence": "It was built in 1999 [9].", "status": "cut"}]


def _state(**over) -> dict:
    state = {
        "thread_id": "tg:42:abcd1234", "user_request": "How does it pause?", "length": "medium",
        "run_config": make_run_config("medium"),
        "brief": {"sub_questions": [{"q": "How?", "kind": "web"}, {"q": "Why?", "kind": "web"}], "summary": "s",
                  "archetype": "explainer"},
        "fetched": FETCHED, "evidence": NOTES, "coverage": {"0": 1, "1": 1},
        "gap_history": [{"round": 1, "wave": 1, "sub_qs": [1], "new_strong": 1, "source": "coverage",
                         "new_strong_by_subq": {"1": 1}}],
        "gap_waves": 1, "compose_runs": 2, "revision_rounds": 0, "revise_rounds": 1, "extend_rounds": 0,
        "queries": [{"query": "langgraph interrupt", "provider": "exa", "sub_qs": [0]}],
        "search_spend": {"exa_searches": 3, "exa_usd": 0.021, "recency": {"window": "m", "start_date": "2026-09-20"}},
        "analysis": {"md": "## x", "archetype": "explainer", "stance_map": {"0": {"supports": 1}},
                     "has_disagreement_section": False, "has_scenarios": False,
                     "takeaways": [{"text": "It pauses [1]", "source_ids": [1], "confidence": "medium",
                                    "reasons": ["1 independent source"]}]},
        "panel_verdict": {"verdict": "pass", "judged": 2, "total": 2, "flagged_finding_ids": [],
                          "skipped_judges": []},
        "findings": FINDINGS, "errors": ERRORS}
    state.update(over)
    return state


def _write(tmp_path, state=None) -> dict:
    path = tmp_path / "evidence.json"
    write_evidence(path, state or _state(), findings=FINDINGS, errors=ERRORS, degraded=True,
                   tldr={"source": "sections", "items": ["It pauses the run [1]."]},
                   engines={"calls": 3, "by_role": {}}, generated_at="2026-10-20T10:00:00+04:00")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_document_follows_section_2_6_and_validates(tmp_path):
    doc = _write(tmp_path)
    Evidence.model_validate(doc)
    assert list(doc) == ["schema", "run", "run_config", "engines", "brief", "coverage", "gap_history",
                         "compose_runs", "revision_rounds", "gap_waves", "revise_rounds", "extend_rounds",
                         "degraded", "sources", "notes", "findings", "analysis", "calculations", "tldr", "panel",
                         "search", "degradations", "errors"]          # calculations: graph/calc.py (after §2.6)
    assert doc["schema"] == 1
    assert doc["run"] == {"run_id": "abcd1234", "thread_id": "tg:42:abcd1234", "topic": "How does it pause?",
                          "generated_at": "2026-10-20T10:00:00+04:00", "length": "medium", "depth": "standard",
                          "recency": {"window": "m", "start_date": "2026-09-20"}, "archetype": "explainer",
                          "origin": "telegram"}
    assert "roles" not in doc["run_config"]["engines"] and doc["run_config"]["engines"]["profile"] == "fake"
    assert doc["engines"] == {"calls": 3, "by_role": {}}
    assert (doc["compose_runs"], doc["revision_rounds"], doc["gap_waves"], doc["revise_rounds"],
            doc["extend_rounds"], doc["degraded"]) == (2, 0, 1, 1, 0, True)
    assert doc["sources"][0] == {
        "id": 1, "url": "https://docs.example.com/guide", "canonical_url": "https://docs.example.com/guide",
        "domain": "docs.example.com", "registrable_domain": "example.com", "title": "Guide", "provider": "exa",
        "fetcher": "exa_text", "published": "2026-09-01", "source_type": "official_doc", "authority": 3,
        "credibility_score": 0.81, "chars": 12345, "pinned": False}
    assert doc["sources"][1]["id"] == 2 and doc["sources"][1]["pinned"] is True
    assert doc["sources"][1]["registrable_domain"] == "" and doc["sources"][1]["source_type"] == "other"
    assert doc["notes"] == [{"source_id": 1, "sub_qs": [0, 1], "relevance": 5, "stance": "supports",
                             "claims": [{"text": "It pauses", "quote": "pauses the run", "confidence": "high",
                                         "quote_verified": True}]}]
    assert doc["findings"][1] == {"id": "f1", "claim": "It was built in 1999", "sentence": "It was built in 1999 [9].",
                                  "source_ids": [9], "section": "Analysis", "status": "cut"}
    assert doc["analysis"] == {"takeaways": [{"text": "It pauses [1]", "source_ids": [1], "confidence": "medium",
                                              "reasons": ["1 independent source"]}],
                               "stance_map": {"0": {"supports": 1}}, "has_disagreement_section": False,
                               "has_scenarios": False}
    assert doc["tldr"] == {"source": "sections", "items": ["It pauses the run [1]."]}
    assert (doc["panel"]["judged"], doc["panel"]["total"], doc["panel"]["next"]) == (2, 2, "report_builder")
    assert doc["search"] == {"queries": _state()["queries"], "spend": _state()["search_spend"]}
    assert doc["errors"] == ERRORS


def test_degradations_are_the_lines_the_bench_counts(tmp_path):
    doc = _write(tmp_path)
    assert doc["degradations"] == [{"round": 1, "text": ERRORS[0].removeprefix("[r1] ")},
                                   {"round": 2, "text": ERRORS[2]},
                                   {"round": 2, "text": ERRORS[3].removeprefix("[r2] ")}], (
        "the round lives in round (§2.6), the text without its tag")
    run = {"state": {"errors": ERRORS, "model_calls": [{"role": "compose.section"}]}, "evidence": doc}
    assert metrics.degradations_count(run) == metrics.degradations_count({**run, "evidence": None}) == 3
    assert metrics._findings(run) == doc["findings"]


def test_no_markdown_path_secret_or_env_value(monkeypatch, tmp_path):
    monkeypatch.setenv("FREELLMAPI_API_KEY", "sk-very-secret-value")
    text = json.dumps(_write(tmp_path))
    assert "markdown_path" not in text and "A:/cache" not in text
    assert "sk-very-secret-value" not in text


def test_no_analysis_is_null(tmp_path):
    assert _write(tmp_path, _state(analysis=None))["analysis"] is None


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema=2),
    lambda d: d["findings"][0].update(status="maybe"),
    lambda d: d.update(markdown_path="x"),
    lambda d: d["sources"][0].update(markdown_path="x"),
    lambda d: d.update(tldr={"source": "llm", "items": []}),
])
def test_the_model_refuses_a_document_off_the_schema(tmp_path, mutate):
    doc = _write(tmp_path)
    mutate(doc)
    with pytest.raises(ValidationError):
        Evidence.model_validate(doc)