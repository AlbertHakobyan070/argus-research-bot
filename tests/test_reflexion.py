"""Panel reflexion loop test (v3): forced flagged finding → panel returns
revise → compose gets the judges' notes → next pass cites properly →
panel passes.

Drives the REAL v3 graph end-to-end with scripted LLMs (no network):
intake → brief → scout ══gate══ research → outline → compose → panel
→ (revise → compose → panel) → report_builder ══gate══ deliver.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from argus.config import get_settings
from argus.graph import build_graph
from argus.graph.nodes import _parse_json_obj
from argus.llm import CallRecord


def test_parse_json_obj_rejects_empty():
    """An LLM returning literally `{}` must not satisfy the parser —
    we want it to fall through to the schema-fallback path."""
    with pytest.raises(ValueError):
        _parse_json_obj("{}")
    with pytest.raises(ValueError):
        _parse_json_obj("some words then ```json {} ```")
    out = _parse_json_obj('{"sub_questions": ["q"], "planned_sources": []}')
    assert "sub_questions" in out


BRIEF_JSON = """{
  "sub_questions": [
    {"q": "What is the direct evidence that X is true?", "kind": "web"},
    {"q": "What mechanisms could explain X?", "kind": "paper"},
    {"q": "What counter-evidence exists against X?", "kind": "web"}
  ],
  "must_have_keywords": ["evidence", "mechanism"],
  "summary": "Investigate X.",
  "success_criteria": ["cites direct evidence"]
}"""


class _FakeResp:
    def __init__(self, content: str):
        self.content = content
        self.response_metadata = {"model_name": "fake"}
        self.usage_metadata = {"input_tokens": 1, "output_tokens": 1}


def _fake_record(tier, req, resp):
    return CallRecord(tier=tier, requested_model=req,
                      served_model="fake", served_provider="fake")


def test_panel_reflexion_revise_then_pass(monkeypatch, tmp_path):
    """First compose produces a finding the grounding judge flags →
    verdict revise → recompose with the judges' notes → judge passes.
    Asserts the loop is wired, notes flow, and the report lands on
    disk with the citation intact."""
    from argus.graph import brief as brief_mod
    from argus.graph import graph as graph_mod
    from argus.graph import research as research_mod
    from argus.graph import scout as scout_mod

    md_file = tmp_path / "x.md"
    md_file.write_text("# Source A\n\nX is true because reasons. " * 20,
                       encoding="utf-8")

    # ---- intake: replace outright (bound in graph.py at build time) --
    def _fake_intake(state):
        return {"mode": "deep", "user_request": state.get("user_request"),
                "model_calls": [], "messages": []}
    monkeypatch.setattr(graph_mod, "intake_node", _fake_intake)

    # ---- ONE scripted LLM for every stage -----------------------------
    # brief_mod.llm, compose_mod.llm and panel_mod.llm are the SAME
    # module object (argus.llm), so a single content-dispatching fake
    # serves all stages.
    compose_calls = {"write": 0}
    judge_calls = {"grounding": 0}

    def _fake_invoke(chat, msgs, **kw):
        system = msgs[0].content
        if "research scoper" in system:
            return _FakeResp(BRIEF_JSON)
        if "Extract the factual findings" in system:
            return _FakeResp(
                '{"findings": [{"claim": "X is true", "source_ids": [1], '
                '"confidence": "high"}]}')
        if "grounding auditor" in system:
            judge_calls["grounding"] += 1
            if judge_calls["grounding"] == 1:
                return _FakeResp(
                    '{"verdict": "revise", "flagged_finding_ids": ["f0"], '
                    '"unsupported_claims": ["X is true"], '
                    '"notes": ["quote the supporting evidence directly"]}')
            return _FakeResp('{"verdict": "pass", '
                             '"flagged_finding_ids": [], '
                             '"unsupported_claims": [], "notes": []}')
        if "COVERAGE" in system:
            return _FakeResp('{"verdict": "pass", "weak_sections": [], '
                             '"missing": [], "notes": []}')
        if "fabrication risk" in system:
            return _FakeResp('{"verdict": "pass", "suspicious": [], '
                             '"notes": []}')
        # flat report writer (short mode)
        compose_calls["write"] += 1
        if compose_calls["write"] == 1:
            return _FakeResp("## TL;DR\n\nX is true [1].\n\n"
                             "## Key Findings\n\n1. X is true [1].")
        return _FakeResp("## TL;DR\n\nX is true, per the direct quote "
                         "[1].\n\n## Key Findings\n\n1. X is true, with "
                         "the supporting quote attached [1].")

    monkeypatch.setattr(brief_mod.llm, "chat_for_tier",
                        lambda tier, **kw: object())
    monkeypatch.setattr(brief_mod.llm, "invoke_with_retry", _fake_invoke)
    monkeypatch.setattr(brief_mod.llm, "record_from_response", _fake_record)
    monkeypatch.setattr(brief_mod.llm, "resolve_tier",
                        lambda t, force=False: "fake")
    monkeypatch.setattr(brief_mod.llm, "pick_strong_and_judge",
                        lambda force=False: ("fake-strong", "fake-judge"))

    # ---- scout: deterministic queries + one wave hit -----------------
    monkeypatch.setattr(scout_mod, "plan_queries",
                        lambda brief: ([{"query": "evidence for X",
                                         "provider": "ddgs",
                                         "sub_qs": [0]}], [], []))
    monkeypatch.setattr(
        scout_mod, "run_query_wave",
        lambda queries: ([{"url": "https://example.com/a",
                           "title": "Source A", "snippet": "evidence",
                           "kind": "blog", "provider": "ddgs",
                           "sub_qs": [0, 1, 2], "text": "", "published": "",
                           "source": "ddgs", "summary": "evidence"}], []))

    # ---- research: fake fetch + digest, no follow-up wave ------------
    def _fake_fetch(src):
        return ({"url": src["url"], "title": src.get("title", ""),
                 "markdown_path": str(md_file), "section": "blog",
                 "excerpt": "X is true because reasons.",
                 "relevance_score": 0.0, "credibility_score": 0.9,
                 "credibility_flag": None, "sub_qs": src.get("sub_qs") or [],
                 "provider": "ddgs"}, [])
    monkeypatch.setattr(research_mod, "fetch_one_source", _fake_fetch)

    def _fake_digest(item, source_id, brief, user_request):
        note = {"source_id": source_id, "source_url": item["url"],
                "title": item["title"], "sub_qs": item.get("sub_qs") or [],
                "relevance": 5, "stance": "supports",
                "claims": [{"text": "X is true", "quote": "X is true",
                            "confidence": "high"}]}
        return note, None, ""
    monkeypatch.setattr(research_mod, "digest_one", _fake_digest)
    monkeypatch.setattr(research_mod, "score_fetched",
                        lambda items, user_request: items)
    monkeypatch.setattr(research_mod, "followup_queries",
                        lambda brief, gaps, prior: ([], []))

    # ---- reports land in tmp -----------------------------------------
    monkeypatch.setenv("ARGUS_REPORTS_ROOT", str(tmp_path))
    from argus import config as cfg_mod
    cfg_mod._cached = None
    assert get_settings().reports_root == tmp_path

    g = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "test:panel-reflexion"}}
    g.invoke({
        "thread_id": "test:panel-reflexion", "user_id": 0,
        "user_request": "is X true?",
        "messages": [], "plan": None, "sources": [], "fetched": [],
        "findings": [], "draft_md": "", "revision_notes": [],
        "revision_rounds": 0, "model_calls": [],
        "hitl": {"pending": False},
    }, config=cfg)

    # Gate 1: paused after scout (grounded plan gate).
    snap = g.get_state(cfg)
    assert tuple(snap.next) == ("research",), snap.next
    assert snap.values.get("plan"), "plan must be rendered for the gate"
    assert snap.values.get("sources"), "scout sources must be present"

    # Resume through research → compose → panel loop → report gate.
    g.invoke(Command(resume=True), config=cfg)
    snap = g.get_state(cfg)
    assert tuple(snap.next) == ("deliver",), snap.next

    # The panel loop ran: revise then pass.
    assert judge_calls["grounding"] == 2, judge_calls
    assert compose_calls["write"] == 2, compose_calls
    final = snap.values
    assert (final.get("review_verdict") or {}).get("verdict") == "pass"
    notes = " | ".join(final.get("revision_notes") or [])
    assert "quote the supporting evidence directly" in notes, notes

    # Finish: deliver.
    g.invoke(Command(resume=True), config=cfg)
    snap = g.get_state(cfg)
    assert not snap.next
    final = snap.values

    # Report on disk with the citation intact.
    paths = final.get("report_paths") or {}
    assert paths.get("md") and Path(paths["md"]).exists()
    md = Path(paths["md"]).read_text(encoding="utf-8")
    assert "https://example.com/a" in md
    # Evidence flowed: findings cite the fetched source.
    assert any("https://example.com/a" in (f.get("citation_urls") or [])
               for f in final.get("findings") or [])
