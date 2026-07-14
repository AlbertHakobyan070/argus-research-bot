"""v3 scout node — the discovery wave behind the grounded plan gate.

For every brief sub-question, plan 1-2 targeted queries (one cheap
batched LLM call, deterministic fallback) and run them in parallel
across providers (Exa / DDGS / arXiv / GitHub). Search-API only — no
page fetching happens before the user approves the plan.

Output feeds the plan gate: ``sources`` (real, live URLs tagged with the
sub-questions they serve) + an updated ``plan`` whose search intents
reflect the actual executed queries.
"""
from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from .. import llm
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
    """Web queries prefer Exa when a key is configured (budget-capped in
    run_query_wave), else DDGS."""
    return "exa" if exa_enabled() else "ddgs"


def fallback_queries(brief: ResearchBrief) -> list[dict]:
    """Deterministic per-sub-question queries when the LLM call fails."""
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


def plan_queries(brief: ResearchBrief) -> tuple[list[dict], list[dict], list[str]]:
    """One batched cheap-LLM call → per-sub-question queries.

    Returns (queries, model_calls, errors); falls back deterministically.
    """
    numbered = "\n".join(f"{i}. [{sq.kind}] {sq.q}"
                         for i, sq in enumerate(brief.sub_questions))
    human = (f"Research summary: {brief.summary}\n"
             f"Keywords: {', '.join(brief.must_have_keywords)}\n\n"
             f"Sub-questions:\n{numbered}\n\nReturn the queries JSON.")
    try:
        chat = llm.chat_for_tier("cheap", temperature=0.2, max_tokens=800)
        resp = llm.invoke_with_retry(chat, [
            SystemMessage(content=QUERYGEN_SYSTEM),
            HumanMessage(content=human[:6000]),
        ])
        rec = llm.record_from_response("cheap", llm.resolve_tier("cheap"),
                                       resp)
        data = parse_json_obj(resp.content, require_any=("queries",))
        out: list[dict] = []
        per_subq: dict[int, int] = {}
        n_subs = len(brief.sub_questions)
        for item in data.get("queries") or []:
            if not isinstance(item, dict):
                continue
            q = str(item.get("q") or "").strip()
            try:
                idx = int(item.get("sub_q", -1))
            except (TypeError, ValueError):
                idx = -1
            if not q or not (0 <= idx < n_subs):
                continue
            if per_subq.get(idx, 0) >= 2:
                continue
            prov = str(item.get("provider") or "web")
            if prov not in ("web", "arxiv", "github"):
                prov = "web"
            out.append({"query": q[:120],
                        "provider": _web_provider() if prov == "web" else prov,
                        "sub_qs": [idx]})
            per_subq[idx] = per_subq.get(idx, 0) + 1
        # Any sub-question the LLM skipped gets a deterministic query.
        for i, sq in enumerate(brief.sub_questions):
            if i not in per_subq:
                prov = _provider_for_kind(sq.kind)
                out.append({
                    "query": " ".join(re.sub(r"[?\"']", "", sq.q).split()[:9]),
                    "provider": _web_provider() if prov == "web" else prov,
                    "sub_qs": [i]})
        if not out:
            raise ValueError("querygen returned no usable queries")
        return out, [rec.model_dump()], []
    except Exception as e:
        logger.warning("query generation failed (%s); deterministic "
                       "fallback", e)
        return fallback_queries(brief), [], [f"scout: querygen failed ({e})"]


def scout_node(state) -> dict:
    """Run the discovery wave. Pauses at the plan gate right after."""
    brief = ResearchBrief.model_validate(state.get("brief") or {})
    queries, calls, errors = plan_queries(brief)

    # Exa category hint for paper-flavoured web queries.
    for q in queries:
        if q["provider"] == "exa":
            idx = (q.get("sub_qs") or [None])[0]
            if idx is not None and idx < len(brief.sub_questions) \
                    and brief.sub_questions[idx].kind == "paper":
                q["category"] = "research paper"

    hits, wave_errors = run_query_wave(queries)
    errors.extend(wave_errors)

    # Pre-seeded sources (demos / deterministic tests / appended) first.
    seen: set[str] = set()
    final: list[dict] = []
    for s in state.get("sources") or []:
        url = s.get("url", "")
        if url and url not in seen:
            final.append(dict(s))
            seen.add(url)
    for h in hits:
        if h["url"] not in seen:
            final.append(h)
            seen.add(h["url"])
    final = final[:SCOUT_SOURCE_CAP]

    plan = brief_to_plan(brief, queries)
    out = {
        "sources": final,
        "queries": queries,
        "plan": plan,
        "errors": errors,
        "model_calls": calls,
        "messages": [{"role": "assistant",
                      "content": (f"🔎 Scout: {len(final)} candidate "
                                  f"source(s) from {len(queries)} live "
                                  "queries — awaiting your approval.")}],
    }
    return out


__all__ = ["scout_node", "plan_queries", "fallback_queries",
           "SCOUT_SOURCE_CAP", "QUERYGEN_SYSTEM"]
