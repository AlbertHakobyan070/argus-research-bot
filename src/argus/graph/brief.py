"""v3 brief node — scoping (replaces v2 planner + planner_reflect).

Turns the user request into a ResearchBrief: sub-questions with
source-kind hints, keywords, success criteria. NO URLs — the scout
searches live, so the LLM never gets a chance to invent links (the v2
hallucinated-URL bug class is structurally impossible here).

A v2-compatible ``plan`` dict is emitted alongside the brief so the
Telegram plan-gate renderer (``_format_plan``) works unchanged.

Quality gate: deterministic checks (sub-question count, no verbatim
echo of the request, keyword coverage) with ONE bounded re-draft —
the v2 planner_reflect behaviour, folded into the node.
"""
from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from .. import llm
from .jsonx import parse_json_obj
from .state import ResearchBrief, SubQuestion

logger = logging.getLogger("argus.brief")

BRIEF_SYSTEM = """You are Argus's research scoper.

Given a research topic/question, produce a research brief that a team of
search agents will execute against live web/arXiv/GitHub search.

Rules:
- 4-7 sub_questions. Each must probe a DIFFERENT aspect (background,
  current state, comparisons, limitations, applications, evidence...).
  Never echo the topic back as a sub-question.
- Each sub_question gets a kind hint for search routing:
  "paper" (academic/arXiv), "repo" (code/GitHub), "web" (news/docs/blogs),
  "mixed" (unsure).
- must_have_keywords: 3-8 distinct signal words/phrases for relevance
  filtering.
- success_criteria: 2-4 statements describing what a COMPLETE answer
  must contain.
- Do NOT include URLs anywhere. Live search finds the sources.

Return ONLY JSON:
{
  "sub_questions": [{"q": "...", "kind": "paper|repo|web|mixed"}, ...],
  "must_have_keywords": ["...", "..."],
  "summary": "1-2 sentence brief summary",
  "success_criteria": ["...", "..."]
}
"""

MIN_SUB_QUESTIONS = 3
VERBATIM_OVERLAP = 0.9


def _overlap_ratio(text: str, reference: str) -> float:
    t_words = {w for w in re.findall(r"\w+", (text or "").lower()) if len(w) > 2}
    r_words = {w for w in re.findall(r"\w+", (reference or "").lower()) if len(w) > 2}
    if not t_words:
        return 0.0
    return len(t_words & r_words) / len(t_words)


def check_brief_quality(brief: ResearchBrief, user_request: str) -> list[str]:
    """Deterministic quality checks. Returns list of failure descriptions."""
    failures: list[str] = []
    if len(brief.sub_questions) < MIN_SUB_QUESTIONS:
        failures.append(
            f"only {len(brief.sub_questions)} sub_questions "
            f"(need >= {MIN_SUB_QUESTIONS})")
    for i, sq in enumerate(brief.sub_questions):
        if _overlap_ratio(sq.q, user_request) >= VERBATIM_OVERLAP:
            failures.append(
                f"sub_question #{i} is a verbatim echo of the request")
            break
    if not brief.must_have_keywords:
        failures.append("no must_have_keywords")
    else:
        req_words = {w for w in re.findall(r"\w+", user_request.lower())
                     if len(w) > 2}
        hay = " ".join(brief.must_have_keywords).lower()
        if req_words and not any(w in hay for w in req_words):
            failures.append("keywords share no word with the request")
    return failures


def _parse_brief(text: str) -> ResearchBrief:
    data = parse_json_obj(text, require_any=("sub_questions",))
    sub_qs = []
    for sq in data.get("sub_questions") or []:
        if isinstance(sq, str):
            sq = {"q": sq}
        if not isinstance(sq, dict) or not (sq.get("q") or "").strip():
            continue
        kind = sq.get("kind", "mixed")
        if kind not in ("paper", "repo", "web", "mixed"):
            kind = "mixed"
        sub_qs.append(SubQuestion(q=sq["q"].strip(), kind=kind))
    return ResearchBrief(
        sub_questions=sub_qs,
        must_have_keywords=[str(k) for k in
                            (data.get("must_have_keywords") or []) if k][:8],
        summary=str(data.get("summary") or "")[:600],
        success_criteria=[str(c) for c in
                          (data.get("success_criteria") or []) if c][:4],
    )


def _fallback_brief(user_request: str) -> ResearchBrief:
    """Single-search fallback when the LLM output is unusable twice."""
    words = [w for w in re.findall(r"\w{3,}", user_request.lower())][:6]
    return ResearchBrief(
        sub_questions=[
            SubQuestion(q=f"What is known about {user_request}?", kind="web"),
            SubQuestion(q=f"What academic work covers {user_request}?",
                        kind="paper"),
            SubQuestion(q=f"What open-source projects relate to "
                          f"{user_request}?", kind="repo"),
        ],
        must_have_keywords=words,
        summary=("⚠️ Brief output was unparseable — fell back to a generic "
                 "3-angle brief. Live search still runs; Edit or Cancel if "
                 "the topic needs a richer scope."),
    )


def brief_to_plan(brief: ResearchBrief, queries: list[dict] | None = None) -> dict:
    """v2-compatible plan dict for the Telegram plan-gate renderer.

    ``_format_plan`` reads sub_questions (strings), planned_sources
    (kind + query), must_have_keywords, summary.
    """
    planned_sources = []
    for q in (queries or []):
        planned_sources.append({
            "kind": {"arxiv": "paper", "github": "repo"}.get(
                q.get("provider", ""), "search_result"),
            "query": q.get("query", ""),
            "target_url": None,
            "rationale": f"live {q.get('provider','web')} search",
        })
    if not planned_sources:
        for sq in brief.sub_questions:
            planned_sources.append({
                "kind": {"paper": "paper", "repo": "repo"}.get(
                    sq.kind, "search_result"),
                "query": sq.q,
                "target_url": None,
                "rationale": "live search",
            })
    return {
        "sub_questions": [sq.q for sq in brief.sub_questions],
        "planned_sources": planned_sources,
        "must_have_keywords": list(brief.must_have_keywords),
        "summary": brief.summary,
    }


def brief_node(state) -> dict:
    """Draft (or revise, when the user replied at the plan gate) the brief."""
    user_request = state.get("user_request") or ""
    feedback = (state.get("plan_feedback") or "").strip()
    prev = state.get("brief") if feedback else None

    def _invoke(extra: str = "") -> tuple[ResearchBrief, dict]:
        if feedback:
            human = (
                f"Topic / question: {user_request}\n\n"
                f"PREVIOUS BRIEF (JSON):\n{json.dumps(prev or {}, indent=2)}\n\n"
                f"USER FEEDBACK — revise the brief accordingly:\n{feedback}\n"
                f"{extra}\n\nReturn the revised brief as JSON."
            )
        else:
            human = (f"Topic / question: {user_request}\n{extra}\n"
                     "Draft the research brief as JSON.")
        chat = llm.chat_for_tier("strong", temperature=0.2, max_tokens=900)
        resp = llm.invoke_with_retry(chat, [
            SystemMessage(content=BRIEF_SYSTEM),
            HumanMessage(content=human[:8000]),
        ])
        rec = llm.record_from_response(
            "strong", llm.resolve_tier("strong"), resp)
        return _parse_brief(resp.content), rec.model_dump()

    calls: list[dict] = []
    errors: list[str] = []
    try:
        brief, rec = _invoke()
        calls.append(rec)
    except Exception as e:
        logger.warning("brief draft failed (%s); using fallback", e)
        brief = _fallback_brief(user_request)
        errors.append(f"brief: unparseable output ({e})")

    failures = check_brief_quality(brief, user_request)
    if failures:
        # ONE bounded re-draft with the failure diagnoses in hand.
        try:
            brief2, rec2 = _invoke(
                extra=("\nYour previous brief was rejected: "
                       + "; ".join(failures)
                       + ". Fix these gaps."))
            calls.append(rec2)
            if not check_brief_quality(brief2, user_request):
                brief = brief2
                failures = []
            elif len(brief2.sub_questions) > len(brief.sub_questions):
                brief = brief2  # still weak but strictly better
        except Exception as e:
            logger.warning("brief re-draft failed (%s); keeping first", e)
    if failures:
        errors.append(f"brief: weak after re-draft: {failures}")

    plan = brief_to_plan(brief)
    msg = (f"📋 Brief drafted: {len(brief.sub_questions)} sub-questions. "
           "Scouting live sources…")
    return {
        "brief": brief.model_dump(),
        "plan": plan,
        "plan_feedback": "",   # consumed
        "errors": errors,
        "model_calls": calls,
        "messages": [{"role": "assistant", "content": msg}],
        "hitl": {"pending": True, "kind": "plan_approval",
                 "ctx": {"plan": plan}},
    }


__all__ = ["brief_node", "brief_to_plan", "check_brief_quality",
           "BRIEF_SYSTEM"]
