"""Reflexion loop test: forced uncited claim → reviewer returns revise →
synthesizer gets revision notes → next pass cites the claim properly.

This proves the reflexion path (reviewer → synthesize) is wired without
needing a live LLM call. We monkey-patch the LLM chat adapters to
scripted responses.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from argus.config import get_settings
from argus.graph import build_graph
from argus.graph.nodes import (
    _parse_json_obj,  # used by tests
)
from argus.graph.state import ArgusState


def _seeded_state(thread_id: str) -> dict:
    return {
        "thread_id": thread_id, "user_id": 0,
        "user_request": "X",
        "messages": [], "plan": {
            "sub_questions": ["Q1"],
            "planned_sources": [{
                "kind": "paper", "query": "Q1",
                "target_url": "https://example.com/a",
                "rationale": "primary",
            }],
            "must_have_keywords": ["q"],
            "summary": "test",
        },
        "sources": [{
            "url": "https://example.com/a", "kind": "paper",
            "title": "Source A", "summary": "primary",
        }],
        "fetched": [{
            "url": "https://example.com/a", "title": "Source A",
            "markdown_path": str(Path(tempfile.gettempdir()) / "x.md"),
            "section": "paper",
            "excerpt": "primary evidence for Q1",
            "relevance_score": 1.0,
        }],
        "findings": [], "draft_md": "",
        "revision_notes": [], "revision_rounds": 0,
        "model_calls": [], "hitl": {"pending": False},
    }


def test_parse_json_obj_rejects_empty(monkeypatch):
    """An LLM returning literally `{}` must not satisfy the parser —
    we want it to fall through to the schema-fallback path."""
    with pytest.raises(ValueError):
        _parse_json_obj("{}")
    with pytest.raises(ValueError):
        _parse_json_obj("some words then ```json {} ```")
    # A real object is accepted.
    out = _parse_json_obj('{"sub_questions": ["q"], "planned_sources": []}')
    assert "sub_questions" in out


def test_reflexion_revise_then_pass(monkeypatch, tmp_path):
    """Drive the deep graph: first call returns an uncited claim
    (reviewer says "revise"), second pass returns a cited claim
    (reviewer says "pass"). We assert the revision_notes carry the
    reviewer's feedback across the loop and the verdict flips."""
    from argus.graph import nodes as nodes_mod
    from argus import llm as llm_mod
    from langchain_core.messages import AIMessage

    # Make the markdown_path exist so filter_node keeps it.
    (tmp_path / "x.md").write_text("# Source A\n\nevidence\n",
                                    encoding="utf-8")
    # Patch the file into the seeded fetched item.
    state = _seeded_state("test:reflexion")
    state["fetched"][0]["markdown_path"] = str(tmp_path / "x.md")

    # ---- Scripted LLM responses ----
    # synthesizer round 1: returns findings + draft WITHOUT citation.
    # synthesizer round 2: same content, but adds a citation URL.
    synth_calls = {"n": 0}

    def fake_synth_chat(tier, *, temperature=0.0, max_tokens=None,
                        model_override=None):
        class _C:
            def invoke(self, msgs):
                synth_calls["n"] += 1
                if synth_calls["n"] == 1:
                    return AIMessage(content=(
                        '{"findings": ['
                        '{"claim": "X is true", "citation_urls": [], '
                        '"confidence": "high"}], '
                        '"draft_md": "# X\\n\\nX is true.\\n\\n## Sources\\n"}'
                    ))
                return AIMessage(content=(
                    '{"findings": ['
                    '{"claim": "X is true", '
                    '"citation_urls": ["https://example.com/a"], '
                    '"confidence": "high"}], '
                    '"draft_md": "# X\\n\\nX is true [1].\\n\\n'
                    '## Sources\\n1. https://example.com/a\\n"}'
                ))
        return _C()

    # reviewer round 1: revise (uncited).  round 2: pass.
    rev_calls = {"n": 0}

    def fake_rev_chat(tier, *, temperature=0.0, max_tokens=None,
                      model_override=None):
        class _C:
            def invoke(self, msgs):
                rev_calls["n"] += 1
                if rev_calls["n"] == 1:
                    return AIMessage(content=(
                        '{"verdict": "revise", '
                        '"notes": ["cite a fetched source"], '
                        '"unsupported_claims": ["X is true"], '
                        '"fabrication_flags": []}'
                    ))
                return AIMessage(content=(
                    '{"verdict": "pass", "notes": [], '
                    '"unsupported_claims": [], "fabrication_flags": []}'
                ))
        return _C()

    monkeypatch.setattr(nodes_mod.llm, "chat_for_tier", fake_synth_chat)
    monkeypatch.setattr(nodes_mod.llm, "record_from_response",
                         lambda tier, req, resp: llm_mod.CallRecord(
                             tier=tier, requested_model=req,
                             served_model="fake", served_provider="fake"))
    monkeypatch.setattr(nodes_mod.llm, "resolve_tier",
                         lambda tier, force=False: "fake")
    # The reviewer node uses chat_for_tier("judge", ...) — we need the
    # SAME fake to serve as the judge. Override the reviewer specifically.
    from argus.graph import nodes as nm
    orig_reviewer = nm.reviewer_node
    rev_calls_local = {"n": 0}

    def _reviewer_with_fake(state):
        # Inline the reviewer's logic with the fake chat.
        from langchain_core.messages import SystemMessage, HumanMessage
        rev_calls_local["n"] += 1
        chat = fake_rev_chat("judge")
        resp = chat.invoke([
            SystemMessage(content="judge"),
            HumanMessage(content=(
                f"findings={state.get('findings')}\n"
                f"draft={state.get('draft_md')}\n"
                f"urls={[f['url'] for f in state.get('fetched') or []]}"
            )),
        ])
        import json as _json
        from argus.graph.state import ReviewVerdict
        data = _parse_json_obj(resp.content)
        v = ReviewVerdict.model_validate({
            "verdict": data.get("verdict", "revise"),
            "notes": data.get("notes", []),
            "unsupported_claims": data.get("unsupported_claims", []),
            "fabrication_flags": data.get("fabrication_flags", []),
        })
        notes = list(state.get("revision_notes") or []) + v.notes + [
            f"Unsupported claim: {c}" for c in v.unsupported_claims
        ]
        rounds = int(state.get("revision_rounds") or 0) + (
            1 if v.verdict == "revise" else 0
        )
        return {
            "review_verdict": v.model_dump(),
            "revision_notes": notes,
            "revision_rounds": rounds,
            "model_calls": [{
                "tier": "judge", "requested_model": "fake",
                "served_model": "fake", "served_provider": "fake",
            }],
            "messages": [{"role": "assistant",
                          "content": f"verdict: {v.verdict}"}],
        }

    monkeypatch.setattr(nm, "reviewer_node", _reviewer_with_fake)
    # Also override the route_after_review in the graph module to use the
    # patched function lookup.
    from argus.graph import graph as graph_mod
    graph_mod.reviewer_node = _reviewer_with_fake

    # Stub the intake node too so the test doesn't hit FreeLLMAPI.
    from argus.graph import nodes as nm2
    from argus.graph import graph as graph_mod2
    def _fake_intake(state):
        return {"mode": "deep", "user_request": state.get("user_request"),
                "model_calls": [], "messages": []}
    monkeypatch.setattr(nm2, "intake_node", _fake_intake)
    graph_mod2.intake_node = _fake_intake

    monkeypatch.setenv("ARGUS_REPORTS_ROOT", str(tmp_path))
    # Invalidate cached settings so the report_builder picks it up.
    from argus import config as cfg_mod
    cfg_mod._cached = None
    assert get_settings().reports_root == tmp_path

    g = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "test:reflexion"}}
    out = g.invoke(state, config=cfg)
    # First pass: pause before deliver (deliver interrupt).
    snap = g.get_state(cfg)
    assert snap.next, "expected interrupt before deliver"
    # Resume.
    g.invoke(Command(resume=True), config=cfg)
    snap = g.get_state(cfg)
    while snap.next:
        g.invoke(Command(resume=True), config=cfg)
        snap = g.get_state(cfg)
    final = snap.values
    # The reviewer must have been called at least twice (1 revise, 1 pass).
    assert rev_calls_local["n"] >= 2, (
        f"expected >= 2 reviewer calls, got {rev_calls_local['n']}"
    )
    # Final verdict is "pass".
    assert (final.get("review_verdict") or {}).get("verdict") == "pass"
    # Revision notes carry the reviewer's feedback across the loop.
    notes = " | ".join(final.get("revision_notes") or [])
    assert "cite a fetched source" in notes, notes
    assert "Unsupported claim" in notes, notes
    # Report on disk.
    paths = final.get("report_paths") or {}
    assert paths.get("md") and Path(paths["md"]).exists()
    md = Path(paths["md"]).read_text(encoding="utf-8")
    # The cited URL is in the final draft.
    assert "https://example.com/a" in md
