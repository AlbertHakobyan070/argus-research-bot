"""v4 scout node — the discovery wave behind the grounded plan gate.

For every brief sub-question, plan 1-2 targeted queries (one cheap
batched LLM call, ``scout.querygen``; degrade → the sub-question text
queries) and run them as one search wave (``engines/search.run_wave``,
first-pass budget) across Exa / DDGS / arXiv / GitHub. Search-API only —
no page fetching happens before the user approves the plan.

Output feeds the plan gate: ``sources`` (real, live URLs tagged with the
sub-questions they serve, deduped by canonical URL) + an updated ``plan``
whose search intents reflect the actual executed queries, and the run's
``search_spend`` (with the recency window decided here, once).
"""
from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from .. import llm
from ..engines.ledger import SpendLedger
from ..engines.search import resolve_recency
from ..engines.urls import canonical_url
from . import run_config
from .brief import brief_to_plan
from .jsonx import parse_json_obj
from .search_providers import exa_enabled, run_query_wave
from .state import ResearchBrief

logger = logging.getLogger("argus.scout")

SCOUT_SOURCE_CAP = 40

QUERYGEN_SYSTEM = """You write web/academic search queries for a research
brief. For each numbered sub-question, produce 1-2 short queries (3-9
words each) a search engine can answer. Queries must be CONCRETE — name
the entities, drop filler words. Vary the angle between the two queries.

Return ONLY JSON:
{"queries": [{"sub_q": <sub-question index, 0-based>, "q": "...",
              "provider": "web|arxiv|github"}, ...]}

provider: "arxiv" for academic-paper questions, "github" for
code/implementation questions, otherwise "web". Max 2 queries per
sub-question.
"""


def _provider_for_kind(kind: str) -> str:
    return {"paper": "arxiv", "repo": "github"}.get(kind, "web")


def _web_provider() -> str:
    """Web queries prefer Exa when a key is configured (budget-capped by
    the run's SpendLedger in run_wave), else DDGS."""
    return "exa" if exa_enabled() else "ddgs"


def _subquestion_query(i: int, brief: ResearchBrief) -> dict:
    sq = brief.sub_questions[i]
    prov = _provider_for_kind(sq.kind)
    return {"query": " ".join(re.sub(r"[?\"']", "", sq.q).split()[:9]),
            "provider": _web_provider() if prov == "web" else prov, "sub_qs": [i]}


def fallback_queries(brief: ResearchBrief) -> list[dict]:
    """Deterministic per-sub-question queries (the ``subquestion_text_queries``
    degradation of ``scout.querygen``)."""
    queries: list[dict] = []
    kw = " ".join(brief.must_have_keywords[:3])
    for i, sq in enumerate(brief.sub_questions):
        text = re.sub(r"[?\"']", "", sq.q).strip()
        words = text.split()
        q = " ".join(words[:9])
        prov = _provider_for_kind(sq.kind)
        queries.append({"query": q, "provider":
                        _web_provider() if prov == "web" else prov,
                        "sub_qs": [i]})
        if kw and sq.kind in ("mixed", "web"):
            queries.append({"query": f"{kw} {' '.join(words[:5])}"[:90],
                            "provider": _web_provider(), "sub_qs": [i]})
    return queries


def plan_queries(brief: ResearchBrief, *, rc: run_config.RunConfig, rnd: int,
                 runtime: Runtime | None = None) -> tuple[list[dict], list[dict], list[str]]:
    """``scout.querygen``: one batched call → per-sub-question queries
    ``{query, provider, sub_qs}``; a sub-question the reply skipped gets its
    deterministic query. Returns ``(queries, model_calls, errors)``; degrade
    → ``fallback_queries``, the ``degrade:`` line with every attempt, and a
    ``degrade`` event."""
    n_subs = len(brief.sub_questions)
    numbered = "\n".join(f"{i}. [{sq.kind}] {sq.q}"
                         for i, sq in enumerate(brief.sub_questions))
    academic = ("\nThe topic is academic: prefer arxiv for paper-like sub-questions.\n"
                if brief.is_academic_topic else "")
    human = (f"Research summary: {brief.summary}\n"
             f"Keywords: {', '.join(brief.must_have_keywords)}\n{academic}\n"
             f"Sub-questions:\n{numbered}\n\nReturn the queries JSON.")

    def validate(text: str) -> list[dict]:
        out: list[dict] = []
        per_subq: dict[int, int] = {}
        for item in parse_json_obj(text, require_any=("queries",)).get("queries") or []:
            if not isinstance(item, dict):
                continue
            q = str(item.get("q") or "").strip()
            try:
                idx = int(item.get("sub_q", -1))
            except (TypeError, ValueError):
                idx = -1
            if not q or not (0 <= idx < n_subs) or per_subq.get(idx, 0) >= 2:
                continue
            prov = str(item.get("provider") or "web")
            if prov not in ("web", "arxiv", "github"):
                prov = "web"
            out.append({"query": q[:120],
                        "provider": _web_provider() if prov == "web" else prov,
                        "sub_qs": [idx]})
            per_subq[idx] = per_subq.get(idx, 0) + 1
        if not out:
            raise ValueError("querygen returned no usable queries")
        return out

    try:
        res = llm.complete("scout.querygen", [SystemMessage(content=QUERYGEN_SYSTEM),
                                              HumanMessage(content=human[:6000])],
                           engines=rc.engines, temperature=0.2, max_tokens=800, validate=validate)
    except llm.EngineDegraded as e:
        logger.warning("scout.querygen degraded (%s); sub-question text queries", e)
        _emit(runtime, "degrade", {"role": e.role, "degradation": e.degradation, "reason": str(e)})
        return (fallback_queries(brief), [a.model_dump() for a in e.attempts],
                [f"[r{rnd}] degrade: scout.querygen → {e.degradation} ({e})"])
    queries = list(res.parsed)
    covered = {i for q in queries for i in q["sub_qs"]}
    queries += [_subquestion_query(i, brief) for i in range(n_subs) if i not in covered]
    return queries, [r.model_dump() for r in res.records], []


def _emit(runtime: Runtime | None, kind: str, data: dict) -> None:
    if runtime is not None:
        runtime.stream_writer({"kind": kind, "node": "scout", "data": data})


def scout_node(state, runtime: Runtime | None = None) -> dict:
    """Run the discovery wave. Pauses at the plan gate right after."""
    rc = run_config.require(state)
    brief = ResearchBrief.model_validate(state.get("brief") or {})
    rnd = 1 + int(state.get("revise_rounds") or 0) + int(state.get("extend_rounds") or 0)
    _emit(runtime, "stage", {"status": "start", "summary": "discovery wave"})
    queries, calls, errors = plan_queries(brief, rc=rc, rnd=rnd, runtime=runtime)

    # Exa category hint for paper-flavoured web queries.
    for q in queries:
        if q["provider"] == "exa":
            idx = (q.get("sub_qs") or [None])[0]
            if idx is not None and idx < len(brief.sub_questions) and (
                    brief.sub_questions[idx].kind == "paper"
                    or (brief.is_academic_topic and brief.sub_questions[idx].kind == "mixed")):
                q["category"] = "research paper"

    ledger = SpendLedger.for_run(rc, state.get("search_spend"))
    recency, note = resolve_recency(rc, brief)
    if note:
        errors.append(note)
    ledger.set_recency(recency)
    hits, wave_errors = run_query_wave(queries, cfg=rc.search, ledger=ledger, recency=recency, mode="first")
    errors.extend(wave_errors)
    _emit(runtime, "search", {"wave": 1, "mode": "first", "queries": len(queries), "hits": len(hits),
                              "errors": len(wave_errors)})

    # Pre-seeded sources (demos / deterministic tests / appended) first.
    # This also catches a plan-edit re-run: bot._replan_after_edit does
    # not clear state["sources"], so the PREVIOUS wave's hits keep their
    # slots (and their sub_qs tags, which index the old brief). When that
    # wave already filled SCOUT_SOURCE_CAP, no new-wave hit survives the
    # cap below.
    seen: set[str] = set()
    final: list[dict] = []
    for s in [*(state.get("sources") or []), *hits]:
        key = canonical_url(s.get("url") or "") or str(s.get("local_path") or "")
        if key and key not in seen:
            final.append(dict(s))
            seen.add(key)
    final = final[:SCOUT_SOURCE_CAP]

    plan = brief_to_plan(brief, queries)
    message = (f"🔎 Scout: {len(final)} candidate source(s) from {len(queries)} live "
               "queries — awaiting your approval.")
    _emit(runtime, "stage", {"status": "end", "summary": message})
    return {
        "sources": final,
        "queries": queries,
        "plan": plan,
        "errors": errors,
        "model_calls": calls,
        "search_spend": ledger.snapshot(),
        "messages": [{"role": "assistant", "content": message}],
    }


__all__ = ["scout_node", "plan_queries", "fallback_queries",
           "SCOUT_SOURCE_CAP", "QUERYGEN_SYSTEM"]
