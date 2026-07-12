"""Robust synthesizer JSON parsing (2026-07-12).

Bug: a Medium 'LangChain vs LangGraph' run fetched 20 sources but shipped
an EMPTY report ('synthesis_no_evidence'). Cause: the weak served model
(llama-4-scout via the proxy's 'auto' route) wrote a valid report whose
'draft_md' value contained LITERAL newlines — invalid JSON. The direct
json.loads failed, and _parse_json_obj's balanced-brace fallback latched
onto the first INNER finding object ({claim, citation_urls, confidence}),
which has no 'findings'/'draft_md' key, so the synthesizer saw empty
findings and emitted a failure block — throwing away a real report.

Contract:
1. _parse_json_obj with require_any= must return the OUTER object even
   when the raw JSON has literal newlines/control chars in string values.
2. It must never return an inner object lacking every required key.
3. The synthesizer can salvage findings from a badly-broken response.
"""
from __future__ import annotations

import pytest

from argus.graph.nodes import _parse_json_obj, _salvage_findings


_NEWLINE_BROKEN = '''```json
{
  "findings": [
    {"claim": "LangChain is a framework for LLM apps", "citation_urls": ["https://a"], "confidence": "high"},
    {"claim": "LangGraph adds stateful graphs", "citation_urls": ["https://b"], "confidence": "high"}
  ],
  "draft_md": "# Comparison

## TL;DR
LangChain vs LangGraph — different tools.

## Key Findings
1. LangChain is a framework [1]
2. LangGraph adds state [2]"
}
```'''


def test_parser_recovers_outer_object_despite_literal_newlines():
    d = _parse_json_obj(_NEWLINE_BROKEN,
                        require_any=("findings", "draft_md", "no_evidence"))
    assert "findings" in d, "must recover the OUTER object, not an inner finding"
    assert len(d["findings"]) == 2
    assert "draft_md" in d and "LangChain vs LangGraph" in d["draft_md"]


def test_parser_require_any_rejects_inner_finding_object():
    # A bare finding object must NOT satisfy a synthesizer parse.
    only_finding = '{"claim": "x", "citation_urls": ["https://a"], "confidence": "high"}'
    with pytest.raises(ValueError):
        _parse_json_obj(only_finding,
                        require_any=("findings", "draft_md", "no_evidence"))


def test_parser_without_require_any_is_unchanged():
    # Backwards-compat: existing callers (intake/planner) pass no require_any.
    d = _parse_json_obj('{"mode": "deep", "cleaned": "x"}')
    assert d["mode"] == "deep"


def test_parser_handles_clean_json():
    d = _parse_json_obj('{"findings": [], "draft_md": "ok", "no_evidence": true}',
                        require_any=("findings", "draft_md", "no_evidence"))
    assert d["no_evidence"] is True


# ---------------------------------------------------------------------------
# findings salvage — last-resort extraction from a broken response
# ---------------------------------------------------------------------------


def test_salvage_findings_extracts_finding_objects():
    broken = '''here is the report {"findings": [
       {"claim": "First claim about X", "citation_urls": ["https://a", "https://b"], "confidence": "high"},
       {"claim": "Second claim about Y", "citation_urls": ["https://c"], "confidence": "medium"}
    ], "draft_md": "totally broken newlines
    everywhere'''
    findings = _salvage_findings(broken)
    assert len(findings) == 2
    assert findings[0]["claim"] == "First claim about X"
    assert findings[0]["citation_urls"] == ["https://a", "https://b"]
    assert findings[1]["confidence"] == "medium"


def test_salvage_findings_empty_when_none():
    assert _salvage_findings("no json at all here") == []
    assert _salvage_findings('{"draft_md": "no findings key"}') == []
