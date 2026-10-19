"""v4 WP-C — the panel's policy: ``panel_verdict.next``, the R14 gap signal with
the zero-yield stop, the diversity rerun, the v4 graph topology and the
compose budget invariants over fake loops (spec §2.4, §2.8, §3 WP-C).

``next`` is coverage first: an eligible gap (a sub-question under
``coverage_min_strong`` or owned by no section, then the coverage judge's
``missing_sub_qs``; minus those whose last gap wave added no strong note)
goes back to ``research`` while ``gap_waves < gap_waves_max``; else a
wanted rewrite goes to ``compose`` while ``revision_rounds <
max_compose_revisions``; else ``report_builder``.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from argus import config as config_mod
from argus import llm
from argus.engines import registry
from argus.engines.llm_openai import DriverError, DriverResponse
from argus.engines.schema import EngineSpec, ProviderSpec
from argus.graph import graph as graph_mod
from argus.graph import nodes as nodes_mod
from argus.graph.panel import (COVERAGE_SYSTEM, GROUNDING_SYSTEM, decide_next, gap_signal, panel_node,
                               route_after_panel)
from argus.graph.run_config import RunConfig
from helpers import fake_engines_snapshot, fake_llm, make_run_config

_FINDING = re.compile(r"^\[(f\d+)\] \(section: (.+?) · cites", re.M)


def _rc(length="medium", **overrides) -> RunConfig:
    return RunConfig.model_validate(make_run_config(length, **overrides))


# ---------------------------------------------------------------------------
# The decision table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gaps,rewrite,gap_waves,rounds,overrides,expected", [
    ([1], False, 0, 0, {}, "research"),
    ([1], True, 0, 0, {}, "research"),                     # coverage first
    ([1], True, 2, 0, {}, "compose"),                      # the gap reserve is spent (gap_waves_max 2)
    ([1], False, 2, 0, {}, "report_builder"),
    ([1], False, 0, 0, {"panel.coverage_to_research": False}, "report_builder"),
    ([], True, 0, 0, {}, "compose"),
    ([], True, 0, 1, {}, "report_builder"),                # max_compose_revisions 1 reached
    ([], False, 0, 0, {}, "report_builder"),
])
def test_the_next_table(gaps, rewrite, gap_waves, rounds, overrides, expected):
    rc = _rc(**overrides)
    assert decide_next(gaps=gaps, wants_rewrite=rewrite, gap_waves=gap_waves, revision_rounds=rounds,
                       rc=rc) == expected


def test_the_quick_preset_never_loops():
    rc = _rc("short")                                      # quick: gap_waves_max 0, max_compose_revisions 0
    assert decide_next(gaps=[0], wants_rewrite=True, gap_waves=0, revision_rounds=0, rc=rc) == "report_builder"


# ---------------------------------------------------------------------------
# The R14 gap signal and the zero-yield stop
# ---------------------------------------------------------------------------

BRIEF4 = {"sub_questions": [{"q": f"q{k}?", "kind": "web"} for k in range(4)], "summary": "s"}
OWNED_012 = {"sections": [{"title": "A", "sub_qs": [0, 1]}, {"title": "B", "sub_qs": [2]}]}


def test_coverage_gaps_come_first_then_the_judges_missing_sub_questions():
    state = {"brief": BRIEF4, "coverage": {"0": 3, "1": 1, "2": 3, "3": 3}, "outline": OWNED_012}
    # 1 is under-covered, 3 is owned by no section; the judge adds 2 and 0
    assert gap_signal(state, _rc(), [2, 0, 9]) == [1, 3, 2, 0]


def test_a_flat_report_has_no_allocation_starved_sub_question():
    state = {"brief": BRIEF4, "coverage": {"0": 3, "1": 3, "2": 3, "3": 3}, "outline": {"sections": []}}
    assert gap_signal(state, _rc(), []) == []


@pytest.mark.parametrize("history,expected", [
    ([{"round": 1, "wave": 1, "sub_qs": [1, 3], "new_strong": 1, "new_strong_by_subq": {"1": 0, "3": 1}}], [3]),
    ([{"round": 1, "wave": 1, "sub_qs": [1], "new_strong": 0}], [3]),           # no per-sub-question counts
    ([{"round": 1, "wave": 1, "sub_qs": [1], "new_strong": 0, "new_strong_by_subq": {"1": 0}},
      {"round": 1, "wave": 2, "sub_qs": [1], "new_strong": 2, "new_strong_by_subq": {"1": 2}}], [1, 3]),
])
def test_the_zero_yield_stop_reads_the_last_gap_wave_of_each_sub_question(history, expected):
    state = {"brief": BRIEF4, "coverage": {"0": 3, "1": 1, "2": 3, "3": 1}, "outline": {"sections": []},
             "gap_history": history}
    assert gap_signal(state, _rc(), []) == expected


# ---------------------------------------------------------------------------
# panel_node: the route it writes
# ---------------------------------------------------------------------------

FETCHED = [{"url": f"https://s{i}.example/p", "title": f"S{i}"} for i in range(1, 5)]
BRIEF2 = {"sub_questions": [{"q": "What is A?", "kind": "web"}, {"q": "What is B?", "kind": "web"}],
          "summary": "s"}


def _note(i, subq):
    return {"source_id": i, "source_url": FETCHED[i - 1]["url"], "title": f"S{i}", "sub_qs": [subq],
            "relevance": 4, "stance": "supports",
            "claims": [{"text": f"fact {i}", "quote": f"quote {i}", "confidence": "high", "quote_verified": True}]}


def _finding(k, section, sid):
    return {"id": f"f{k}", "claim": f"claim {k}", "citation_urls": [FETCHED[sid - 1]["url"]], "section": section,
            "source_ids": [sid], "sentence": f"Sentence {k} [{sid}].", "status": "kept"}


def _panel_state(**over) -> dict:
    state = {"thread_id": "tg:1:abcd1234", "user_request": "topic", "length": "medium",
             "run_config": make_run_config("medium"), "brief": BRIEF2, "fetched": FETCHED,
             "evidence": [_note(1, 0), _note(2, 0), _note(3, 1), _note(4, 1)],
             "coverage": {"0": 2, "1": 2},
             "outline": {"sections": [{"title": "A", "sub_qs": [0], "note_ids": [1, 2]},
                                      {"title": "B", "sub_qs": [1], "note_ids": [3, 4]}],
                         "changed_sections": []},
             "sections": [{"title": "A", "md": "Sentence 0 [1]. Sentence 1 [2].", "note_ids": [1, 2]},
                          {"title": "B", "md": "Sentence 2 [3]. Sentence 3 [4].", "note_ids": [3, 4]}],
             "findings": [_finding(0, "A", 1), _finding(1, "A", 2), _finding(2, "B", 3), _finding(3, "B", 4)],
             "model_calls": [], "errors": []}
    state.update(over)
    return state


def _judges(flag=(), weak=(), missing=()):
    def reply(system, human):
        if system == GROUNDING_SYSTEM:
            ids = [m[0] for m in _FINDING.findall(human)]
            return json.dumps({"flagged": [{"id": i, "reason": "no support"} for i in ids if i in flag],
                               "notes": ["cite the quote"]})
        assert system == COVERAGE_SYSTEM
        return json.dumps({"verdict": "revise" if weak or missing else "pass", "weak_sections": list(weak),
                           "missing_sub_qs": list(missing), "notes": []})
    return reply


def test_an_under_covered_sub_question_goes_back_to_research(monkeypatch):
    fake_llm(monkeypatch, _judges())
    state = _panel_state(coverage={"0": 2, "1": 1})
    out = panel_node(state)
    pv = out["panel_verdict"]
    assert (pv["next"], pv["coverage_gaps"], out["coverage_gaps"]) == ("research", [1], [1])
    assert route_after_panel({**state, **out}) == "research"
    assert out.get("revision_rounds", 0) == 0


def test_the_zero_yield_stop_skips_research(monkeypatch):
    fake_llm(monkeypatch, _judges())
    history = [{"round": 1, "wave": 1, "sub_qs": [1], "new_strong": 0, "source": "coverage",
                "new_strong_by_subq": {"1": 0}}]
    out = panel_node(_panel_state(coverage={"0": 2, "1": 1}, gap_history=history, gap_waves=1))
    assert out["panel_verdict"]["next"] == "report_builder" and out["coverage_gaps"] == []


def test_the_judges_missing_sub_question_is_a_gap(monkeypatch):
    fake_llm(monkeypatch, _judges(missing=[0]))
    out = panel_node(_panel_state())
    assert out["panel_verdict"]["coverage_gaps"] == [0] and out["panel_verdict"]["next"] == "research"


@pytest.mark.parametrize("flag,expected", [(["f0"], "report_builder"), (["f0", "f2"], "compose")])
def test_a_rewrite_needs_the_flag_ratio(monkeypatch, flag, expected):
    fake_llm(monkeypatch, _judges(flag=flag))
    out = panel_node(_panel_state(run_config=make_run_config("medium", **{"panel.rewrite_flag_ratio": 0.5})))
    pv = out["panel_verdict"]
    assert pv["next"] == expected and pv["wants_rewrite"] is (expected == "compose")
    assert pv["flagged_finding_ids"] == flag
    if expected == "compose":
        assert pv["verdict"] == "revise" and pv["revise_sections"] == ["A", "B"]
        assert out["revision_rounds"] == 1
        assert any('"claim 2" is not supported by its sources' in n for n in pv["notes"])
    else:
        assert pv["verdict"] == "pass" and out.get("revision_rounds", 0) == 0


def test_a_weak_section_wants_a_rewrite_of_that_section(monkeypatch):
    fake_llm(monkeypatch, _judges(weak=["B", "Nope"]))
    pv = panel_node(_panel_state())["panel_verdict"]
    assert (pv["next"], pv["revise_sections"], pv["wants_rewrite"]) == ("compose", ["B"], True)


def test_the_panel_no_longer_grows_revision_notes(monkeypatch):
    fake_llm(monkeypatch, _judges(flag=["f0", "f1", "f2"]))
    out = panel_node(_panel_state(revision_notes=["the user's note"]))
    assert "revision_notes" not in out
    assert out["review_verdict"]["verdict"] == "revise", "the v2 review_verdict is still written"


def test_no_findings_passes_straight_to_the_report(monkeypatch):
    fake_llm(monkeypatch, lambda system, human: pytest.fail("no judge without findings"))
    out = panel_node(_panel_state(findings=[]))
    assert out["panel_verdict"]["next"] == "report_builder" and out["panel_verdict"]["total"] == 0


def test_route_after_panel_returns_next():
    assert route_after_panel({"panel_verdict": {"next": "compose"}}) == "compose"
    assert route_after_panel({"panel_verdict": None}) == "report_builder"


# ---------------------------------------------------------------------------
# The diversity rerun
# ---------------------------------------------------------------------------

WRITER = {"role": "compose.section", "tier": "strong", "profile": "fake", "provider": "fake",
          "requested_model": "fake-model", "served_model": "fake-model", "routed_via": "", "ok": True}


def _served_by_engine(monkeypatch, reply):
    """A fake driver that serves each engine's own model id."""
    seen: list[str] = []

    class _Driver:
        def complete(self, engine, provider, messages, **kw):
            seen.append(engine.model)
            return DriverResponse(text=reply(engine.model, messages[0].content, messages[1].content),
                                  served_model=engine.model, routed_via="", finish_reason="stop", usage={},
                                  headers={}, raw=None)

    monkeypatch.setattr(registry, "provider", lambda name: ProviderSpec(name=name, base_url="http://fake/v1"))
    monkeypatch.setattr(llm, "_driver", _Driver())
    return seen


def _rc_with_judge_fallback(model: str | None, **overrides) -> dict:
    snap = fake_engines_snapshot()
    binding = snap.roles["panel.grounding"]
    snap.roles["panel.grounding"] = binding.model_copy(
        update={"fallbacks": [EngineSpec(provider="other", model=model)] if model else []})
    return RunConfig.from_preset(length="medium", engines=snap, origin="telegram",
                                 overrides={"panel.coverage": False, **overrides}).model_dump()


def _flag_on(model_flags: dict[str, list[str]]):
    def reply(model, system, human):
        ids = [m[0] for m in _FINDING.findall(human)]
        return json.dumps({"flagged": [{"id": i, "reason": "x"} for i in ids if i in model_flags.get(model, [])],
                           "notes": []})
    return reply


def test_a_judge_of_the_writers_family_is_rerun_on_another_family(monkeypatch):
    seen = _served_by_engine(monkeypatch, _flag_on({"fake-model": ["f0"], "other-model": ["f3"]}))
    out = panel_node(_panel_state(run_config=_rc_with_judge_fallback("other-model"), model_calls=[WRITER]))
    assert seen == ["fake-model", "other-model"]
    pv = out["panel_verdict"]
    assert pv["judge_not_diverse"] is False and pv["flagged_finding_ids"] == ["f3"], "the rerun's verdict counts"
    assert (pv["judged"], pv["total"]) == (4, 4)
    rerun = [c for c in out["model_calls"] if c["retry_reason"] == "diversity_rerun"]
    assert [(c["role"], c["served_model"]) for c in rerun] == [("panel.grounding", "other-model")]


@pytest.mark.parametrize("fallback,overrides", [("fake-model-2", {}), (None, {}),
                                                 ("other-model", {"panel.diversity_rerun": False})])
def test_no_diverse_rerun_means_judge_not_diverse(monkeypatch, fallback, overrides):
    seen = _served_by_engine(monkeypatch, _flag_on({}))
    out = panel_node(_panel_state(run_config=_rc_with_judge_fallback(fallback, **overrides), model_calls=[WRITER]))
    assert seen == ["fake-model"] and out["panel_verdict"]["judge_not_diverse"] is True


def test_a_failed_rerun_keeps_the_first_verdict(monkeypatch):
    def reply(model, system, human):
        if model == "other-model":
            raise DriverError("provider other: bad request")
        return _flag_on({"fake-model": ["f1"]})(model, system, human)
    _served_by_engine(monkeypatch, reply)
    out = panel_node(_panel_state(run_config=_rc_with_judge_fallback("other-model"), model_calls=[WRITER]))
    pv = out["panel_verdict"]
    assert pv["flagged_finding_ids"] == ["f1"] and pv["judged"] == 4 and pv["judge_not_diverse"] is True
    assert not any(e.startswith("[r1] skip: panel.grounding") for e in out["errors"]), "the batch was judged"


def test_a_judge_of_another_family_needs_no_rerun(monkeypatch):
    seen = _served_by_engine(monkeypatch, _flag_on({}))
    writer = {**WRITER, "served_model": "llama-3"}
    out = panel_node(_panel_state(run_config=_rc_with_judge_fallback("other-model"), model_calls=[writer]))
    assert seen == ["fake-model"] and out["panel_verdict"]["judge_not_diverse"] is False


# ---------------------------------------------------------------------------
# The v4 graph
# ---------------------------------------------------------------------------

def test_the_v4_topology():
    drawable = graph_mod.build_graph(checkpointer=MemorySaver()).get_graph()
    edges = {(e.source, e.target) for e in drawable.edges}
    assert "analyst" in drawable.nodes
    assert {("outline", "compose"), ("compose", "analyst"), ("analyst", "panel")} <= edges
    assert {("panel", "research"), ("panel", "compose"), ("panel", "report_builder")} <= edges
    assert ("compose", "panel") not in edges
    assert {("revise_prep", "compose"), ("extend_prep", "research")} <= edges


_WORDS = ["the runtime pauses graphs", "checkpoints persist state", "interrupts raise an error",
          "resume uses a command object", "subgraphs restart from the top", "the scheduler injects values"]


def test_compose_budget_invariants_over_gap_waves_a_rewrite_a_user_revise_and_an_extend(monkeypatch, tmp_path):
    from argus.graph import compose as compose_mod
    from argus.graph.research import _round, compute_coverage
    from argus.graph.state import ResearchBrief
    from bench import metrics

    brief = ResearchBrief.model_validate(BRIEF2)

    def research(state):
        evidence, fetched, extra = list(state.get("evidence") or []), list(state.get("fetched") or []), {}

        def add(k):
            i = len(evidence) + 1
            evidence.append(_note_any(i, k))
            fetched.append({"url": f"https://s{i}.example/p", "title": f"S{i}"})

        if state.get("coverage_gaps"):                    # a gap wave: the panel's coverage signal
            gaps, waves = list(state["coverage_gaps"]), int(state.get("gap_waves") or 0)
            for k in gaps:
                add(k)
            extra = {"coverage_gaps": [], "gap_waves": waves + 1, "gap_history": [
                *(state.get("gap_history") or []),
                {"round": _round(state), "wave": waves + 1, "sub_qs": gaps, "new_strong": len(gaps),
                 "source": "coverage", "new_strong_by_subq": {str(k): 1 for k in gaps}}]}
        elif not state.get("research_rounds"):            # the first pass leaves B under-covered
            for k in (0, 0, 1):
                add(k)
        else:                                             # a user Extend
            add(0)
        return {**extra, "evidence": evidence, "fetched": fetched, "coverage": compute_coverage(brief, evidence),
                "research_rounds": int(state.get("research_rounds") or 0) + 1}

    def outline(state):
        evidence = state.get("evidence") or []
        sections = [{"title": t, "focus": "", "sub_qs": [k],
                     "note_ids": [n["source_id"] for n in evidence if k in n["sub_qs"]]}
                    for k, t in enumerate(("Part A", "Part B"))]
        before = {s["title"]: s["note_ids"] for s in (state.get("outline") or {}).get("sections") or []}
        return {"outline": {"sections": sections,
                            "changed_sections": [s["title"] for s in sections if before.get(s["title"]) != s["note_ids"]]}}

    judge_calls: list[int] = []

    def reply(system, human):
        if system.startswith(compose_mod.SECTION_SYSTEM):
            ids = re.findall(r"^\[(\d+)\] ", human, re.M)
            return " ".join(f"Evidence {i}: {_WORDS[(int(i) - 1) % len(_WORDS)]} [{i}]." for i in ids)
        if system == compose_mod.EXTRACT_SYSTEM:
            md = human.split("\n\n", 1)[1].rsplit("\n\nReturn the findings JSON.", 1)[0]
            sentences = [s for s in re.split(r"(?<=\.)\s+", md) if s.strip()]
            return json.dumps({"findings": [{"claim": re.sub(r"\s*\[\d+\]", "", s), "sentence": s,
                                             "source_ids": [int(i) for i in re.findall(r"\[(\d+)\]", s)],
                                             "confidence": "high"} for s in sentences]})
        if system == GROUNDING_SYSTEM:
            judge_calls.append(1)
            flag = [fid for fid, sec in _FINDING.findall(human) if sec == "Part B" and len(judge_calls) == 2]
            return json.dumps({"flagged": [{"id": i, "reason": "no support"} for i in flag], "notes": []})
        assert system == COVERAGE_SYSTEM
        return json.dumps({"verdict": "pass", "weak_sections": [], "missing_sub_qs": [], "notes": []})

    fake_llm(monkeypatch, reply)
    for name, fn in {"intake_node": lambda s: {"mode": "deep"},
                     "brief_node": lambda s: {"brief": dict(BRIEF2), "plan": {"summary": "s", "sub_questions": []}},
                     "scout_node": lambda s: {"sources": []},
                     "research_node": research, "outline_node": outline}.items():
        monkeypatch.setattr(graph_mod, name, fn)
    monkeypatch.setattr(config_mod, "get_settings", lambda: SimpleNamespace(reports_root=tmp_path))
    monkeypatch.setattr(nodes_mod, "markdown_to_pdf", lambda *a, **k: None)

    def assert_budget(v):
        rc = v["run_config"]
        rnd = 1 + int(v.get("revise_rounds") or 0) + int(v.get("extend_rounds") or 0)
        gaps_now = sum(1 for g in v.get("gap_history") or [] if g["round"] == rnd)
        assert v["compose_runs"] == 1 + gaps_now + v["revision_rounds"]
        assert v["compose_runs"] <= 1 + rc["research"]["gap_waves_max"] + rc["panel"]["max_compose_revisions"]
        assert metrics.compose_budget_ok({"state": v}) is True

    g = graph_mod.build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t:budget"}}
    g.invoke({"thread_id": "t:budget", "user_id": 1, "user_request": "A and B", "length": "medium",
              "run_config": make_run_config("medium", **{"compose.analyst": False}),
              "messages": [], "sources": [], "fetched": [], "findings": [], "draft_md": "",
              "revision_notes": [], "revision_rounds": 0, "model_calls": [], "hitl": {"pending": False}}, config=cfg)
    g.invoke(Command(resume=True), config=cfg)              # plan gate → … → the report gate
    snap = g.get_state(cfg)
    v = snap.values
    assert snap.next == ("deliver",)
    assert [h["round"] for h in v["gap_history"]] == [1] and v["gap_waves"] == 1, "panel → research was taken"
    assert (v["compose_runs"], v["revision_rounds"]) == (3, 1)
    assert_budget(v)

    g.update_state(cfg, {"revision_requested": True, "revision_notes": ["make it shorter"]})
    g.invoke(Command(resume=True), config=cfg)              # a user revise: round 2
    v = g.get_state(cfg).values
    assert (v["revise_rounds"], v["compose_runs"], v["revision_rounds"]) == (1, 1, 0)
    assert_budget(v)

    g.update_state(cfg, {"extend_requested": True})
    g.invoke(Command(resume=True), config=cfg)              # an extend: round 3
    snap = g.get_state(cfg)
    v = snap.values
    assert snap.next == ("deliver",)
    assert (v["extend_rounds"], v["compose_runs"], v["revision_rounds"]) == (1, 1, 0)
    assert_budget(v)
    assert len(judge_calls) == 5


def _note_any(i, k):
    return {"source_id": i, "source_url": f"https://s{i}.example/p", "title": f"S{i}", "sub_qs": [k],
            "relevance": 4, "stance": "supports",
            "claims": [{"text": f"fact {i}", "quote": f"quote {i}", "confidence": "high", "quote_verified": True}]}