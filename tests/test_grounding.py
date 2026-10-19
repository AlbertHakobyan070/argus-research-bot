"""Tests for src/argus/graph/grounding.py (v4 spec §3 WP-B).

"Mentions the entity" means the FULL entity phrase from ``brief.entities``
(normalised, case-insensitive) occurs in the fetched text — never a single
query token: baseline b5, a nonexistent entity, passed v3's gate because any
one of six generic tokens counted. A source counts only when its digest note
is relevant (relevance ≥ 3) and not of authority 0.

Run with:
    PYTHONPATH='' ./venv/Scripts/python.exe -m pytest tests/test_grounding.py -q
"""
from __future__ import annotations

from pathlib import Path

from argus.graph.grounding import (
    DEFAULT_GROUNDING_THRESHOLD,
    GroundingResult,
    check_grounding,
    grounding_banner,
    grounding_warning_prompt,
)


def _state(tmp_path: Path, entities: list[str], docs: list[str], *, relevance=4, authority=1) -> dict:
    fetched, evidence = [], []
    for i, text in enumerate(docs):
        md = tmp_path / f"doc{i}.md"
        md.write_text(text, encoding="utf-8")
        url = f"https://site{i}.example/p"
        fetched.append({"url": url, "title": f"Doc {i}", "markdown_path": str(md), "excerpt": text[:600]})
        evidence.append({"source_id": i + 1, "source_url": url, "relevance": relevance,
                         "authority": authority, "claims": []})
    return {"user_request": "anything", "brief": {"entities": entities}, "fetched": fetched,
            "evidence": evidence}


def test_entity_phrase_not_tokens(tmp_path):
    """A document matching only single tokens of the entity is not a mention."""
    tokens_only = ("Quantum computing is advancing. Llama models are popular. "
                   "Version 9 of the toolkit shipped this week.")
    state = _state(tmp_path, ["Quantum Llama 9"], [tokens_only] * 3)
    r = check_grounding(state)
    assert (r.n_credible, r.mention_count, r.grounded) == (3, 0, False)
    assert r.entity_tokens == ["quantum llama 9"]

    named = "The Quantum-Llama 9 release notes describe the new model in detail."
    r = check_grounding(_state(tmp_path, ["Quantum Llama 9"], [named, named, tokens_only]))
    assert (r.mention_count, r.grounded) == (2, True), "the phrase, normalised and case-insensitive"


def test_the_phrase_is_searched_in_the_full_fetched_text(tmp_path):
    body = "Background. " * 400 + "Late in the text, GLM 5.2 is finally named."
    r = check_grounding(_state(tmp_path, ["GLM 5.2"], [body, body]))
    assert r.mention_count == 2 and r.grounded, "not only the title and the 600-char excerpt"


def test_a_longer_version_does_not_match_a_shorter_entity(tmp_path):
    r = check_grounding(_state(tmp_path, ["GLM 5"], ["GLM 5.2 is out.", "GLM 5.2 again."]))
    assert r.mention_count == 0 and not r.grounded
    r = check_grounding(_state(tmp_path, ["GLM 5.2"], ["glm-5.2 is out.", "GLM 5.2."]))
    assert r.mention_count == 2 and r.grounded


def test_only_relevant_authoritative_sources_count(tmp_path):
    named = "LangGraph interrupts pause a graph."
    weak = check_grounding(_state(tmp_path, ["LangGraph"], [named] * 3, relevance=2))
    assert (weak.n_credible, weak.grounded) == (0, False), "digest relevance < 3 does not count"
    farm = check_grounding(_state(tmp_path, ["LangGraph"], [named] * 3, authority=0))
    assert (farm.n_credible, farm.grounded) == (0, False), "authority-0 sources do not count"


def test_every_entity_needs_the_threshold(tmp_path):
    docs = ["Mamba and the Transformer compared.", "Transformer only.", "Transformer again."]
    r = check_grounding(_state(tmp_path, ["Mamba", "Transformer"], docs))
    assert (r.mention_count, r.grounded) == (1, False), "the least-supported entity decides"
    r = check_grounding(_state(tmp_path, ["Mamba", "Transformer"], docs[:1] * 2))
    assert (r.mention_count, r.grounded) == (2, True)


def test_no_entities_means_nothing_to_check(tmp_path):
    r = check_grounding(_state(tmp_path, [], ["anything"]))
    assert r.grounded is True and r.entity_tokens == []
    assert check_grounding({"user_request": "GLM 5.2", "fetched": []}).grounded is True, (
        "the request's tokens are never the entity")


def test_entities_but_nothing_fetched_is_not_grounded():
    r = check_grounding({"brief": {"entities": ["GLM 5.2"]}, "fetched": [], "evidence": []})
    assert (r.grounded, r.n_credible, r.mention_count) == (False, 0, 0)


def test_threshold_can_be_lowered(tmp_path):
    state = _state(tmp_path, ["GLM 5.2"], ["GLM 5.2 overview"])
    assert check_grounding(state).grounded is False
    assert check_grounding(state, threshold=1).grounded is True


def test_sample_titles_are_collected(tmp_path):
    r = check_grounding(_state(tmp_path, ["Transformer"], ["Transformer"] * 5))
    assert r.sample_evidence_titles == ["Doc 0", "Doc 1", "Doc 2"]


# --- Warning prompt + banner -----------------------------------------

def test_warning_prompt_empty_when_grounded():
    r = GroundingResult(grounded=True, entity_tokens=["transformer"], mention_count=5, n_credible=5, threshold=3)
    assert grounding_warning_prompt(r, "transformer survey") == ""


def test_warning_prompt_non_empty_when_not_grounded():
    r = GroundingResult(grounded=False, entity_tokens=["glm 5.2"], mention_count=1, n_credible=2, threshold=3)
    out = grounding_warning_prompt(r, "GLM 5.2")
    assert "GROUNDING WARNING" in out
    assert "GLM 5.2" in out
    assert "strictly conservative" in out.lower()
    assert "insufficient direct evidence" in out.lower()
    assert "1" in out and "2" in out


def test_banner_empty_when_grounded():
    r = GroundingResult(grounded=True, entity_tokens=["x"], mention_count=5, n_credible=5, threshold=3)
    assert grounding_banner(r) == ""


def test_banner_non_empty_with_counts_when_not_grounded():
    r = GroundingResult(grounded=False, entity_tokens=["unicorns"], mention_count=0, n_credible=2, threshold=3)
    out = grounding_banner(r)
    assert "Limited direct evidence" in out
    assert "0" in out and "2" in out


def test_default_threshold_constant_matches_documented():
    assert DEFAULT_GROUNDING_THRESHOLD == 2
