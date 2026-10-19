"""Depth rebalance guards (2026-07-12).

Locks in the fix for "reports too shallow and too filtered": legit
reference/curated/primary domains must clear the credibility floor
(they used to score ~0.11 and get cut), content farms must still be
filtered (fabrication guard intact), grounding must not over-fire, and
the default 'short' mode must not be skeletal.
"""
from __future__ import annotations

import pytest

from argus.graph.credibility import CREDIBILITY_FLOOR, DomainTrust, credibility_score
from argus.graph.grounding import check_grounding
from argus.graph.state import FetchedItem
from argus.graph.synthesis_modes import get_mode


def _item(url: str, title: str = "retrieval augmented generation") -> FetchedItem:
    return FetchedItem(url=url, title=title, excerpt="RAG small models",
                       markdown_path="x.md")


# ---------------------------------------------------------------------------
# legit sources clear the floor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://arxiv.org/abs/2410.12345",
    "https://github.com/org/rag",
    "https://huggingface.co/blog/rag",
    "https://paperswithcode.com/task/rag",
    "https://stackoverflow.com/questions/123/rag",
])
def test_legit_domains_clear_the_floor(url):
    s = credibility_score(_item(url), user_request="retrieval augmented generation")
    assert s >= CREDIBILITY_FLOOR, (
        f"{url} scored {s:.2f} < floor {CREDIBILITY_FLOOR} — legit sources "
        "must not be filtered as noise (this was the over-filtering bug)")


def test_wikipedia_is_no_longer_scored_as_noise():
    # The exact regression: Wikipedia scored 0.11 pre-rebalance.
    s = credibility_score(
        _item("https://en.wikipedia.org/wiki/Retrieval-augmented_generation"),
        user_request="retrieval augmented generation small models")
    assert s >= 0.5, f"Wikipedia should be a solidly-credible source, got {s:.2f}"


# ---------------------------------------------------------------------------
# fabrication / content-farm guard still intact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://thetechbriefs.com/best-rag-models-2025",
    "https://glm45.org/glm-5.2-specs",
    "https://ml-briefs.com/top-10-rag",
    "https://seo-content.ai/rag",
])
def test_content_farms_still_dropped(url):
    s = credibility_score(_item(url), user_request="rag")
    assert s < CREDIBILITY_FLOOR, (
        f"content farm {url} scored {s:.2f} >= floor — the anti-fabrication "
        "guard must stay intact after the depth rebalance")


def test_tier_ordering_preserved():
    e = {d["domain"]: d["score"] for d in DomainTrust.entries()}
    # primary > trusted > (neutral default) > low still holds.
    assert DomainTrust.score_for("https://arxiv.org/abs/1") > \
        DomainTrust.score_for("https://github.com/a/b")
    assert DomainTrust.score_for("https://github.com/a/b") > \
        DomainTrust.score_for("https://glm45.org/x")


# ---------------------------------------------------------------------------
# grounding no longer over-fires on a normally-covered topic
# ---------------------------------------------------------------------------


def _grounding_state(tmp_path, entity, docs, *, authority=1):
    fetched, evidence = [], []
    for i, (url, text) in enumerate(docs):
        md = tmp_path / f"g{i}.md"
        md.write_text(text, encoding="utf-8")
        fetched.append({"url": url, "title": text[:40], "markdown_path": str(md), "excerpt": text[:600]})
        evidence.append({"source_id": i + 1, "source_url": url, "relevance": 4, "authority": authority,
                         "claims": []})
    return {"brief": {"entities": [entity]}, "fetched": fetched, "evidence": evidence}


def test_grounding_holds_on_well_covered_topic(tmp_path):
    """Three relevant sources naming the topic = grounded (no strict-
    conservatism warning) — the over-conservative gate made shallow reports."""
    text = "Retrieval augmented generation grounds small models in fetched evidence."
    res = check_grounding(_grounding_state(tmp_path, "retrieval augmented generation", [
        ("https://en.wikipedia.org/wiki/Retrieval-augmented_generation", text),
        ("https://arxiv.org/abs/2410.00001", text),
        ("https://github.com/org/rag", text)]))
    assert res.n_credible >= 3, f"expected >=3 credible, got {res.n_credible}"
    assert res.grounded, "a topic with 3 relevant sources naming it must be grounded"


def test_grounding_still_flags_thin_topic(tmp_path):
    """A lone content farm about a nonexistent entity stays ungrounded."""
    res = check_grounding(_grounding_state(tmp_path, "GLM 5.2", [
        ("https://glm45.org/glm-5.2", "GLM 5.2 specs and benchmark scores")], authority=0))
    assert not res.grounded, "a lone content farm must not ground a rare entity"


# ---------------------------------------------------------------------------
# default depth is no longer skeletal
# ---------------------------------------------------------------------------


def test_short_mode_is_not_skeletal():
    m = get_mode("short")
    assert m.target_findings >= 6, f"short bumped to >=6 findings, got {m.target_findings}"
    assert m.max_tokens >= 1800
    assert m.target_chars_max >= 1400


def test_depth_ordering_across_modes_preserved():
    modes = [get_mode(k) for k in ("tldr", "short", "medium", "long", "lecture")]
    findings = [m.target_findings for m in modes]
    # non-decreasing depth as modes get longer (tldr=0 is fine)
    assert findings == sorted(findings), findings
    assert get_mode("lecture").target_findings > get_mode("short").target_findings


# ---------------------------------------------------------------------------
# v4 research depth: rerank, chunked digest, digest fields, quote checks,
# the research.digest ALL_SKIPPED_RAISES rule, first-pass caps
# ---------------------------------------------------------------------------

import json  # noqa: E402

from argus import llm  # noqa: E402
from argus.engines import registry  # noqa: E402
from argus.engines.llm_openai import DriverResponse  # noqa: E402
from argus.engines.schema import ProviderSpec  # noqa: E402
from argus.graph import research as research_mod  # noqa: E402
from argus.graph.run_config import RunConfig  # noqa: E402
from argus.graph.state import ResearchBrief, SubQuestion  # noqa: E402

from helpers import make_run_config  # noqa: E402

BRIEF = ResearchBrief(sub_questions=[SubQuestion(q="How do LangGraph interrupts work?"),
                                     SubQuestion(q="How does a run resume?")],
                      must_have_keywords=["langgraph", "interrupt"], summary="s")


def _resp(text: str) -> DriverResponse:
    return DriverResponse(text=text, served_model="m", routed_via="", finish_reason="stop",
                          usage={"prompt_tokens": 1, "completion_tokens": 1}, headers={}, raw=None)


@pytest.fixture
def facade(monkeypatch):
    """The real llm.complete on a fake driver; ``by_system[prompt](human) -> text``."""
    by_system: dict = {}
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(registry, "provider", lambda name: ProviderSpec(name=name, base_url="http://fake/v1"))

    class _Driver:
        def complete(self, engine, provider, messages, **kw):
            system, human = messages[0].content, messages[1].content
            seen.append((system, human))
            return _resp(by_system[system](human))

    monkeypatch.setattr(llm, "_driver", _Driver())
    return by_system, seen


def _cands(n=6):
    return [{"url": f"https://c{i}.example/p", "title": f"Candidate {i} about langgraph interrupts",
             "snippet": "interrupt resume", "sub_qs": [i % 2]} for i in range(n)]


def test_rerank_scores_order_the_wave_and_drop_off_topic(facade):
    by_system, _ = facade
    by_system[research_mod.RERANK_SYSTEM] = lambda human: json.dumps({"scores": [
        {"id": 1, "score": 1}, {"id": 2, "score": 5}, {"id": 3, "score": 0}, {"id": 4, "score": 4},
        {"id": 5, "score": 3}, {"id": 6, "score": 2}]})
    rc = RunConfig.model_validate(make_run_config("medium"))
    out = research_mod.rerank(_cands(), BRIEF, "q", rc=rc, wave=1, rnd=1, cap=4)
    assert [c["url"] for c in out.picked] == ["https://c1.example/p", "https://c3.example/p",
                                               "https://c4.example/p", "https://c5.example/p"]
    assert "https://c2.example/p" not in [c["url"] for c in out.picked], "score 0 = off-topic"
    assert (out.errors, out.degraded, [c["role"] for c in out.calls]) == ([], False, ["research.rerank"])
    assert out.event == {"wave": 1, "candidates": 6, "kept": 4}


def test_rerank_degrade_falls_back_to_keyword_triage_and_is_recorded(facade):
    by_system, _ = facade
    by_system[research_mod.RERANK_SYSTEM] = lambda human: "I would rank them by relevance, roughly..."
    rc = RunConfig.model_validate(make_run_config("medium"))
    out = research_mod.rerank(_cands(), BRIEF, "q", rc=rc, wave=1, rnd=1, cap=4)
    assert out.degraded is True
    assert len(out.picked) == 4, "keyword triage still picks the wave"
    assert len(out.calls) == 2 and not any(c["ok"] for c in out.calls), "every attempt is kept"
    (line,) = out.errors
    assert line.startswith("[r1] degrade: research.rerank → keyword_triage (role research.rerank: "
                           "every engine failed after 2 attempts (last: unparseable: ValueError: ")


# research.per_subq_target — the per-wave, per-sub-question selection quota (WP-B review)

def _dominated():
    """Sub-question 0 dominates the scores: six candidates at 5; sub-question 1
    three at 2; one untagged candidate at 4."""
    cands = ([{"url": f"https://a{i}.example/p", "title": "a", "snippet": "s", "sub_qs": [0]} for i in range(6)]
             + [{"url": f"https://b{i}.example/p", "title": "b", "snippet": "s", "sub_qs": [1]} for i in range(3)]
             + [{"url": "https://u.example/p", "title": "u", "snippet": "s", "sub_qs": []}])
    scores = [5] * 6 + [2] * 3 + [4]
    return cands, json.dumps({"scores": [{"id": i + 1, "score": s} for i, s in enumerate(scores)]})


def _quota_rc(per_subq: int) -> RunConfig:
    return RunConfig.model_validate(make_run_config("medium", **{"research.per_subq_target": per_subq}))


def test_rerank_quota_holds_when_one_sub_question_dominates_the_scores(facade):
    by_system, _ = facade
    cands, reply = _dominated()
    by_system[research_mod.RERANK_SYSTEM] = lambda human: reply
    out = research_mod.rerank(cands, BRIEF, "q", rc=_quota_rc(2), wave=1, rnd=1, cap=4)
    assert [c["url"] for c in out.picked] == ["https://a0.example/p", "https://a1.example/p",
                                               "https://b0.example/p", "https://b1.example/p"], (
        "score order alone would take a0..a3")
    assert out.event["kept"] == 4


def test_rerank_quota_leftovers_fill_by_rank(facade):
    by_system, _ = facade
    cands, reply = _dominated()
    by_system[research_mod.RERANK_SYSTEM] = lambda human: reply
    out = research_mod.rerank(cands, BRIEF, "q", rc=_quota_rc(2), wave=1, rnd=1, cap=9)
    assert [c["url"] for c in out.picked] == [
        "https://a0.example/p", "https://a1.example/p", "https://b0.example/p", "https://b1.example/p",   # quota
        "https://a2.example/p", "https://a3.example/p", "https://a4.example/p", "https://a5.example/p",   # leftovers
        "https://u.example/p"], "leftovers by rank: an untagged source only here, b2 (score 2) left out"


def test_the_quota_holds_on_the_degrade_path(facade):
    by_system, _ = facade
    by_system[research_mod.RERANK_SYSTEM] = lambda human: "I would rank them by relevance, roughly..."
    hot = [{"url": f"https://hot{i}.example/p", "title": "langgraph interrupt resume", "snippet": "langgraph interrupt",
            "sub_qs": [0]} for i in range(6)]
    cold = [{"url": f"https://cold{i}.example/p", "title": "unrelated", "snippet": "nothing", "sub_qs": [1]}
            for i in range(3)]
    out = research_mod.rerank(hot + cold, BRIEF, "q", rc=_quota_rc(2), wave=1, rnd=1, cap=4)
    assert out.degraded is True
    assert [c["sub_qs"][0] for c in out.picked].count(1) == 2, "keyword_triage applies the same quota"
    assert len(out.picked) == 4


def _long_doc(tmp_path):
    head = "LangGraph interrupt() pauses the run at a node. " * 160            # ~7 700 chars
    tail = "Resuming passes Command(resume=value) back into the node. " * 400   # ~23 000 chars
    md = tmp_path / "long.md"
    md.write_text(head + tail, encoding="utf-8")
    return md


def test_chunked_digest_merges_the_chunks(facade, tmp_path):
    by_system, seen = facade

    def digest(human):
        first = "Part 1 of 2" in human
        return json.dumps({
            "relevance": 3 if first else 5, "stance": "background" if first else "supports",
            "source_type": "official_doc",
            "claims": [{"text": "interrupt() pauses the run", "quote": "interrupt() pauses the run at a node",
                        "confidence": "high"}] if first else
                      [{"text": "Command(resume=...) resumes", "quote": "Resuming passes Command(resume=value)",
                        "confidence": "high"},
                       {"text": "interrupt() pauses the run", "quote": "pauses the run at a node",
                        "confidence": "medium"}]})

    by_system[research_mod.DIGEST_SYSTEM] = digest
    rc = RunConfig.model_validate(make_run_config("medium"))           # digest_chunks_long_doc = 2
    item = {"url": "https://docs.example/interrupts", "title": "Interrupts", "sub_qs": [0, 1],
            "markdown_path": str(_long_doc(tmp_path)), "authority_prior": 1}
    note, calls, errors = research_mod.digest_one(item, 7, BRIEF, "q", rc=rc, rnd=1)
    assert errors == [] and len(calls) == 2 and len(seen) == 2
    assert [c["item"] for c in calls] == ["https://docs.example/interrupts#chunk1",
                                          "https://docs.example/interrupts#chunk2"]
    assert (note["source_id"], note["relevance"], note["stance"]) == (7, 5, "supports"), (
        "relevance is the best chunk's; stance comes with it")
    assert [c["text"] for c in note["claims"]] == ["interrupt() pauses the run", "Command(resume=...) resumes"], (
        "claims merged, duplicates dropped")
    assert (note["source_type"], note["authority"]) == ("official_doc", 2)


def test_digest_verifies_quotes_against_the_document(facade, tmp_path):
    by_system, _ = facade
    by_system[research_mod.DIGEST_SYSTEM] = lambda human: json.dumps({
        "relevance": 4, "stance": "supports", "source_type": "blog",
        "claims": [{"text": "a", "quote": "“pauses THE run at a node”", "confidence": "high"},
                   {"text": "b", "quote": "LangGraph was released by the Vatican in 1802", "confidence": "low"},
                   {"text": "c", "quote": "", "confidence": "medium"}]})
    md = tmp_path / "short.md"
    md.write_text("LangGraph interrupt() pauses the run at a node. " * 20, encoding="utf-8")
    rc = RunConfig.model_validate(make_run_config("medium"))
    note, calls, errors = research_mod.digest_one(
        {"url": "https://b.example/p", "title": "B", "sub_qs": [0], "markdown_path": str(md)}, 1, BRIEF, "q",
        rc=rc, rnd=1)
    assert [c["quote_verified"] for c in note["claims"]] == [True, False, None]
    assert [c["item"] for c in calls] == ["https://b.example/p"], "a short document is one call"
    assert (note["source_type"], note["authority"]) == ("blog", 1)


def test_digest_skip_is_a_skip_line_with_the_attempts(monkeypatch, tmp_path):
    attempt = llm.CallRecord(role="research.digest", tier="cheap", profile="fake", provider="fake",
                             requested_model="m", served_model="", routed_via="", ok=False,
                             error="RateLimitError: 429")

    def complete(role, messages, **kw):
        raise llm.EngineSkipped("role research.digest: every engine failed after 1 attempt "
                                "(last: RateLimitError: 429)", role=role, attempts=[attempt])

    monkeypatch.setattr(research_mod.llm, "complete", complete)
    md = tmp_path / "d.md"
    md.write_text("Some document text about interrupts. " * 20, encoding="utf-8")
    rc = RunConfig.model_validate(make_run_config("medium"))
    note, calls, errors = research_mod.digest_one(
        {"url": "https://d.example/p", "title": "D", "sub_qs": [0], "markdown_path": str(md)}, 1, BRIEF, "q",
        rc=rc, rnd=2)
    assert note is None and calls == [attempt.model_dump()]
    assert errors == ["[r2] skip: research.digest https://d.example/p (role research.digest: every engine "
                      "failed after 1 attempt (last: RateLimitError: 429))"]


def _node_state(tmp_path, *, n_sources=3, evidence=None, length="medium", **overrides):
    return {"user_request": "langgraph interrupts", "brief": BRIEF.model_dump(), "length": length,
            "run_config": make_run_config(length, **{"research.rerank": False, **overrides}),
            "sources": [{"url": f"https://s{i}.example/p", "title": "langgraph interrupt", "snippet": "x",
                         "sub_qs": [i % 2]} for i in range(n_sources)],
            "fetched": [], "evidence": list(evidence or []), "queries": []}


def _fake_fetch(monkeypatch, tmp_path):
    from argus.graph.state import FetchedItem

    def fetch(src, **kw):
        md = tmp_path / (src["url"].split("//")[1].split(".")[0] + ".md")
        md.write_text("LangGraph interrupts pause runs. " * 20, encoding="utf-8")
        return FetchedItem(url=src["url"], title=src["title"], markdown_path=str(md),
                           sub_qs=src["sub_qs"]).model_dump(), []

    monkeypatch.setattr(research_mod, "fetch_one_source", fetch)
    monkeypatch.setattr(research_mod, "run_query_wave", lambda queries, **kw: ([], []))


def test_every_digest_skipped_with_no_evidence_raises(monkeypatch, tmp_path):
    _fake_fetch(monkeypatch, tmp_path)

    def skipped(item, source_id, brief, user_request, **kw):
        return None, [], [f"[r1] skip: research.digest {item['url']} (boom)"]

    monkeypatch.setattr(research_mod, "digest_one", skipped)
    monkeypatch.setattr(research_mod, "followup_queries", lambda *a, **kw: ([], [], []))
    with pytest.raises(llm.EngineError, match="digested 0 evidence notes") as e:
        research_mod.research_node(_node_state(tmp_path))
    assert e.value.role == "research.digest"


def test_skipped_digests_never_kill_a_run_that_holds_evidence(monkeypatch, tmp_path):
    _fake_fetch(monkeypatch, tmp_path)
    monkeypatch.setattr(research_mod, "digest_one", lambda item, *a, **kw: (
        None, [], [f"[r1] skip: research.digest {item['url']} (boom)"]))
    monkeypatch.setattr(research_mod, "followup_queries", lambda *a, **kw: ([], [], []))
    prior = [{"source_id": 1, "source_url": "https://old.example", "sub_qs": [0], "relevance": 4, "claims": []}]
    out = research_mod.research_node(_node_state(tmp_path, evidence=prior))
    assert out["evidence"] == prior
    assert sum(1 for e in out["errors"] if e.startswith("[r1] skip: research.digest")) == 3


def test_first_pass_respects_the_wave_and_total_caps(monkeypatch, tmp_path):
    _fake_fetch(monkeypatch, tmp_path)
    monkeypatch.setattr(research_mod, "digest_one", lambda item, source_id, *a, **kw: (
        {"source_id": source_id, "source_url": item["url"], "sub_qs": item["sub_qs"], "relevance": 1,
         "claims": []}, [], []))
    monkeypatch.setattr(research_mod, "followup_queries", lambda *a, **kw: ([], [], []))
    out = research_mod.research_node(_node_state(tmp_path, n_sources=20, length="short"))   # quick: 1 x 10, cap 12
    assert len(out["fetched"]) == 10
    assert out["search_spend"]["fetch_first"] == 10
    assert out["research_rounds"] == 1
