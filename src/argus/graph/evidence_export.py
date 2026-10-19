"""``evidence.json`` (v4 spec §2.6): the run's evidence, written by
report_builder next to ``report.md`` and validated by ``Evidence`` (schema
1) before it is written. Read by the console (WP-E2) and the bench (WP-G).

Strict models: an unknown key anywhere is refused, so no ``markdown_path``,
secret or env value can ride along. ``degradations`` are exactly the
``skip:`` / ``degrade:`` / ``legacy:`` lines of ``errors`` that the bench
counts (``bench.metrics.degradations_count``), each with its round (an
untagged line takes the current round).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..engines.urls import domain, registrable_domain
from . import run_config
from .research import _round
from .state import PanelVerdict, ResearchBrief

DEGRADATION_LINE = re.compile(r"^(?:\[r(\d+)\]\s*)?(?:skip|degrade|legacy):")   # = bench.metrics._DEGRADATION_LINE
Confidence = Literal["high", "medium", "low"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Run(_Strict):
    run_id: str
    thread_id: str
    topic: str
    generated_at: str
    length: str
    depth: str
    recency: dict | None
    archetype: str
    origin: str


class Source(_Strict):
    id: int
    url: str
    canonical_url: str
    domain: str
    registrable_domain: str
    title: str
    provider: str
    fetcher: str
    published: str
    source_type: str
    authority: int
    credibility_score: float | None
    chars: int
    pinned: bool


class Claim(_Strict):
    text: str
    quote: str
    confidence: Confidence
    quote_verified: bool | None


class Note(_Strict):
    source_id: int
    sub_qs: list[int]
    relevance: int
    stance: str
    claims: list[Claim]


class Finding(_Strict):
    id: str
    claim: str
    sentence: str
    source_ids: list[int]
    section: str
    status: Literal["kept", "cut", "unlocated"]


class Takeaway(_Strict):
    text: str
    source_ids: list[int]
    confidence: Confidence
    reasons: list[str]


class Analysis(_Strict):
    takeaways: list[Takeaway]
    stance_map: dict
    has_disagreement_section: bool
    has_scenarios: bool


class Tldr(_Strict):
    source: Literal["takeaways", "sections", "omitted"]
    items: list[str]


class Search(_Strict):
    queries: list[dict]
    spend: dict


class Degradation(_Strict):
    round: int
    text: str


class Evidence(_Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal[1] = Field(1, alias="schema")
    run: Run
    run_config: dict                 # RunConfig.model_dump() without engines.roles
    engines: dict                    # registry.summarize_calls()
    brief: dict
    coverage: dict[str, int]
    gap_history: list[dict]
    compose_runs: int
    revision_rounds: int
    gap_waves: int
    revise_rounds: int
    extend_rounds: int
    degraded: bool
    sources: list[Source]
    notes: list[Note]
    findings: list[Finding]
    analysis: Analysis | None
    tldr: Tldr
    panel: PanelVerdict
    search: Search
    degradations: list[Degradation]
    errors: list[str]


def degradations(errors: list[str], rnd: int) -> list[dict]:
    """The ``skip:`` / ``degrade:`` / ``legacy:`` lines as ``{round, text}``."""
    out = []
    for line in errors:
        m = DEGRADATION_LINE.match(line)
        if m:
            out.append({"round": int(m.group(1) or rnd), "text": line})
    return out


def write_evidence(path: Path, state, *, findings: list[dict], errors: list[str], degraded: bool, tldr: dict,
                   engines: dict, generated_at: str) -> dict:
    """Build, validate and write ``evidence.json``; returns the document."""
    rc = run_config.require(state)
    brief = ResearchBrief.model_validate(state.get("brief") or {})
    thread_id = state.get("thread_id") or ""
    spend = state.get("search_spend") or {}
    notes = state.get("evidence") or []
    by_id = {int(n["source_id"]): n for n in notes}
    sources = []
    for i, item in enumerate(state.get("fetched") or [], start=1):
        url, note = item.get("url") or "", by_id.get(i) or {}
        sources.append({
            "id": i, "url": url, "canonical_url": item.get("canonical_url") or "",
            "domain": item.get("domain") or domain(url), "registrable_domain": registrable_domain(url),
            "title": item.get("title") or "", "provider": item.get("provider") or "",
            "fetcher": item.get("fetcher") or "", "published": item.get("published") or "",
            "source_type": note.get("source_type") or "other",
            "authority": int(note.get("authority", item.get("authority_prior", 1))),
            "credibility_score": item.get("credibility_score"), "chars": int(item.get("chars") or 0),
            "pinned": bool(item.get("pinned"))})
    rc_dump = rc.model_dump(mode="json")
    rc_dump["engines"].pop("roles", None)
    analysis = state.get("analysis")
    doc = Evidence.model_validate({
        "schema": 1,
        "run": {"run_id": thread_id.rsplit(":", 1)[-1], "thread_id": thread_id,
                "topic": state.get("user_request") or "", "generated_at": generated_at, "length": rc.length,
                "depth": rc.depth, "recency": spend.get("recency"), "archetype": brief.archetype,
                "origin": rc.origin},
        "run_config": rc_dump, "engines": engines, "brief": brief.model_dump(),
        "coverage": {str(k): int(v) for k, v in (state.get("coverage") or {}).items()},
        "gap_history": list(state.get("gap_history") or []),
        **{k: int(state.get(k) or 0) for k in ("compose_runs", "revision_rounds", "gap_waves", "revise_rounds",
                                               "extend_rounds")},
        "degraded": degraded, "sources": sources,
        "notes": [{"source_id": int(n["source_id"]), "sub_qs": list(n.get("sub_qs") or []),
                   "relevance": int(n.get("relevance") or 0), "stance": n.get("stance") or "background",
                   "claims": [{"text": c.get("text") or "", "quote": c.get("quote") or "",
                               "confidence": c.get("confidence") or "medium",
                               "quote_verified": c.get("quote_verified")} for c in n.get("claims") or []]}
                  for n in notes],
        "findings": [{"id": f.get("id") or "", "claim": f.get("claim") or "", "sentence": f.get("sentence") or "",
                      "source_ids": list(f.get("source_ids") or []), "section": f.get("section") or "",
                      "status": f.get("status") or "kept"} for f in findings],
        "analysis": None if not analysis else {
            "takeaways": [{k: t[k] for k in ("text", "source_ids", "confidence", "reasons")}
                          for t in analysis.get("takeaways") or []],
            "stance_map": analysis.get("stance_map") or {},
            "has_disagreement_section": bool(analysis.get("has_disagreement_section")),
            "has_scenarios": bool(analysis.get("has_scenarios"))},
        "tldr": tldr,
        "panel": PanelVerdict.model_validate(state.get("panel_verdict") or {}).model_dump(),
        "search": {"queries": list(state.get("queries") or []), "spend": dict(spend)},
        "degradations": degradations(errors, _round(state)),
        "errors": list(errors),
    })
    out = doc.model_dump(mode="json", by_alias=True)
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


__all__ = ["Evidence", "write_evidence", "degradations", "DEGRADATION_LINE"]
