"""Panel reflexion loop test (v4): forced flagged finding → the panel wants a
rewrite (the flag ratio) → compose gets the judge's notes → the next pass
cites properly → the panel passes.

Drives the REAL graph end-to-end with scripted LLMs (no network):
intake → brief → scout ══gate══ research → outline → compose → analyst →
panel → (compose → analyst → panel) → report_builder ══gate══ deliver.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from argus.config import get_settings
from argus.graph import build_graph
from argus.graph.nodes import _parse_json_obj

from helpers import fake_llm, make_run_config


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


def test_panel_reflexion_revise_then_pass(monkeypatch, tmp_path):
    """First compose produces a finding the grounding judge flags →
    verdict revise → recompose with the judges' notes → judge passes.
    Asserts the loop is wired, notes flow, and the report lands on
    disk with the citation intact."""
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

    # ---- ONE scripted LLM for every stage (llm.complete on a fake driver)
    compose_calls = {"write": 0, "analyst": 0}
    judge_calls = {"grounding": 0}

    def _reply(system, human):
        if "research scoper" in system:
            return BRIEF_JSON
        if "Extract the factual findings" in system:
            return ('{"findings": [{"claim": "X is true", "sentence": "X is true [1].", '
                    '"source_ids": [1], "confidence": "high"}]}')
        if "grounding auditor" in system:
            judge_calls["grounding"] += 1
            if judge_calls["grounding"] == 1:
                ids = [line.split("]")[0][1:] for line in human.splitlines() if line.startswith("[f")]
                return json.dumps({"flagged": [{"id": i, "reason": "no quote"} for i in ids],
                                   "notes": ["quote the supporting evidence directly"]})
            return '{"flagged": [], "notes": []}'
        if "ANALYSIS part" in system:
            compose_calls["analyst"] += 1
            return "### Key takeaways\n\n- X is true [1]\n"
        # flat report writer (short mode)
        compose_calls["write"] += 1
        if compose_calls["write"] == 1:
            return "X is true, as the source states it plainly [1]."      # >= 40 chars: a body, not an empty write
        assert "quote the supporting evidence directly" in human, "the judge's notes reach the rewrite"
        return "X is true, per the direct quote from the source [1]."

    seen = fake_llm(monkeypatch, _reply)

    # ---- scout: deterministic queries + one wave hit -----------------
    monkeypatch.setattr(scout_mod, "plan_queries",
                        lambda brief, **kw: ([{"query": "evidence for X",
                                         "provider": "ddgs",
                                         "sub_qs": [0]}], [], []))
    monkeypatch.setattr(
        scout_mod, "run_wave",
        lambda queries, **kw: ([{"url": "https://example.com/a",
                           "title": "Source A", "snippet": "evidence",
                           "kind": "blog", "provider": "ddgs",
                           "sub_qs": [0, 1, 2], "text": "", "published": "",
                           "source": "ddgs", "summary": "evidence"}], []))

    # ---- research: fake fetch + digest, no follow-up wave ------------
    def _fake_fetch(src, **kw):
        return ({"url": src["url"], "title": src.get("title", ""),
                 "markdown_path": str(md_file), "section": "blog",
                 "excerpt": "X is true because reasons.",
                 "relevance_score": 0.0, "credibility_score": 0.9,
                 "credibility_flag": None, "sub_qs": src.get("sub_qs") or [],
                 "provider": "ddgs"}, [])
    monkeypatch.setattr(research_mod, "fetch_one_source", _fake_fetch)

    def _fake_digest(item, source_id, brief, user_request, **kw):
        note = {"source_id": source_id, "source_url": item["url"],
                "title": item["title"], "sub_qs": item.get("sub_qs") or [],
                "relevance": 5, "stance": "supports",
                "claims": [{"text": "X is true", "quote": "X is true",
                            "confidence": "high"}]}
        return note, [], []
    monkeypatch.setattr(research_mod, "digest_one", _fake_digest)
    monkeypatch.setattr(research_mod, "score_fetched",
                        lambda items, user_request: items)
    monkeypatch.setattr(research_mod, "followup_queries",
                        lambda brief, gaps, prior, **kw: ([], [], []))

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
        "run_config": make_run_config("short", **{"research.rerank": False,
                                                   "panel.max_compose_revisions": 1}),
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

    # The panel loop ran: a rewrite, then a pass.
    assert judge_calls["grounding"] == 2, judge_calls
    assert compose_calls == {"write": 2, "analyst": 2}, compose_calls
    final = snap.values
    assert (final.get("review_verdict") or {}).get("verdict") == "pass"
    assert (final["compose_runs"], final["revision_rounds"]) == (2, 1)
    rewrite = [m[1] for m in seen if m[0].startswith("You write a complete short")][1]
    assert "quote the supporting evidence directly" in rewrite

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
