"""Brief node — scoping (replaces v2 planner + planner_reflect).

Turns the user request into a ResearchBrief: sub-questions with
source-kind hints, keywords, success criteria, and (v4, spec §3 WP-I) the
report archetype, time sensitivity with a recency hint, the academic flag
and the entity phrases the grounding check looks for. NO URLs — the scout
searches live, so the LLM never gets a chance to invent links (the v2
hallucinated-URL bug class is structurally impossible here). The links the
USER typed are read first (``anchors``): the brief drafts from an excerpt of
each, and each one read becomes a pinned source.

A v2-compatible ``plan`` dict is emitted alongside the brief so the
Telegram plan-gate renderer (``_format_plan``) works unchanged.

``brief.draft`` runs through ``llm.complete``; the parser is its validate
hook, so an unparseable reply is nudged, retried and recorded, and a role
whose engines are exhausted raises (spec §2.1: ``brief.draft`` must raise —
there is no generic fallback brief). Quality gate: deterministic checks
(sub-question count, no verbatim echo of the request, keyword coverage)
with ONE bounded re-draft — the v2 planner_reflect behaviour, folded into
the node.
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from .. import llm
from ..engines.ledger import SpendLedger
from . import run_config
from .anchors import brief_context, plan_view, read_anchors
from .archetypes import DEFAULT_ARCHETYPE, TEMPLATES
from .grounding import normalise
from .jsonx import parse_json_obj
from .state import ResearchBrief, SubQuestion

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
- archetype: the report shape that fits the request best, one of
  "explainer" (how something works / what it is), "definition" (what a
  term means), "comparison" (X vs Y, differences, which is better),
  "forecast" (outlook, prediction, future prices or trends), "sota_review"
  (state of the art: approaches and results), "howto" (steps to do or
  build something), "news_update" (what happened recently), "decision"
  (which option to choose for a goal).
- time_sensitive: true when the answer depends on recent events or data
  (prices, news, releases, "latest", "this month"), else false.
- recency_hint: when time_sensitive, how recent the sources must be —
  "d" (a day), "w" (a week), "m" (a month) or "y" (a year); else null.
- is_academic_topic: true when peer-reviewed research is the main
  evidence (science, medicine, ML research); false for products, markets,
  news and how-tos.
- entities: up to 6 proper names the request is about — products,
  models, organisations, people, projects, places — copied as written in
  the request. Never a date, period, price, metric or generic noun:
  "Bitcoin price outlook for Q4" → ["Bitcoin"];
  "LangGraph vs CrewAI" → ["LangGraph", "CrewAI"];
  a general topic that names nothing specific ("causes of inflation") → [].
- When the request comes with CONTEXT FROM THE USER'S LINKS, those pages
  are what the request is about or refers to: write sub-questions about
  that specific project, product or page (what it is and does, who it is
  for, how it is built, how it compares), name it in entities, and use
  its own terms in the keywords — never only the generic market around it.
- Do NOT include URLs anywhere. Live search finds the sources.

Return ONLY JSON:
{
  "sub_questions": [{"q": "...", "kind": "paper|repo|web|mixed"}, ...],
  "must_have_keywords": ["...", "..."],
  "summary": "1-2 sentence brief summary",
  "success_criteria": ["...", "..."],
  "archetype": "<one of the archetypes above>",
  "time_sensitive": <true|false>,
  "recency_hint": <"d"|"w"|"m"|"y"|null>,
  "is_academic_topic": <true|false>,
  "entities": ["<proper name>", ...]
}
"""

_MAX_ENTITIES = 6
# A normalised token that names a period or a number, never a thing (spec §3 WP-I).
_PERIOD_TOKEN = re.compile(r"q[1-4]|h[12]|fy\d{2,4}|\d+(?:\.\d+)*|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?"
                           r"|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?"
                           r"|nov(?:ember)?|dec(?:ember)?")

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


def _flag(value) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _entities(raw, asked: str) -> list[str]:
    """The entity phrases, lower-cased and whitespace-collapsed, kept only
    when ``asked`` (the request, any plan feedback and the previous brief's
    entities) names them — compared the way the grounding check normalises
    them — so an invented name or abbreviation never becomes something the
    documents must contain; a phrase made only of period/number tokens
    ("Q4", "FY26", "March 2025") is no entity; one per normalised form, at
    most ``_MAX_ENTITIES``."""
    hay = f" {normalise(asked)} "
    out: list[str] = []
    keys: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, str):
            continue
        phrase = " ".join(item.lower().split())
        key = normalise(phrase)
        if (key and key not in keys and f" {key} " in hay
                and not all(_PERIOD_TOKEN.fullmatch(t) for t in key.split())):
            out.append(phrase)
            keys.add(key)
    return out[:_MAX_ENTITIES]


def _parse_brief(text: str, asked: str = "") -> ResearchBrief:
    """The ``brief.draft`` reply as a ResearchBrief — the validate hook: a
    reply without a ``sub_questions`` list raises (and is retried). The v4
    fields take their deterministic defaults when missing or malformed: an
    unknown archetype → ``DEFAULT_ARCHETYPE``; a recency hint is passed on as
    given (``resolve_recency`` ignores an unusable one with a note)."""
    data = parse_json_obj(text, require_any=("sub_questions",))
    raw_subs = data.get("sub_questions")
    if not isinstance(raw_subs, list):
        raise ValueError(f"'sub_questions' is {type(raw_subs).__name__}, not a list")
    sub_qs = []
    for sq in raw_subs:
        if isinstance(sq, str):
            sq = {"q": sq}
        if not isinstance(sq, dict) or not (sq.get("q") or "").strip():
            continue
        kind = sq.get("kind", "mixed")
        if kind not in ("paper", "repo", "web", "mixed"):
            kind = "mixed"
        sub_qs.append(SubQuestion(q=sq["q"].strip(), kind=kind))
    archetype = str(data.get("archetype") or "").strip().lower()
    hint = data.get("recency_hint")
    hint = hint.strip().lower() if isinstance(hint, str) else ""
    return ResearchBrief(
        sub_questions=sub_qs,
        must_have_keywords=[str(k) for k in
                            (data.get("must_have_keywords") or []) if k][:8],
        summary=str(data.get("summary") or "")[:600],
        success_criteria=[str(c) for c in
                          (data.get("success_criteria") or []) if c][:4],
        archetype=archetype if archetype in TEMPLATES else DEFAULT_ARCHETYPE,
        time_sensitive=_flag(data.get("time_sensitive")),
        recency_hint=hint if hint not in ("", "null", "none") else None,
        is_academic_topic=_flag(data.get("is_academic_topic")),
        entities=_entities(data.get("entities"), asked),
    )


def brief_to_plan(brief: ResearchBrief, queries: list[dict] | None = None,
                  anchors: list[dict] | None = None) -> dict:
    """v2-compatible plan dict for the Telegram plan-gate renderer.

    ``_format_plan`` reads sub_questions (strings), planned_sources
    (kind + query), must_have_keywords, summary, and ``anchors`` (the user's
    links: what was read, or why not).
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
        "anchors": plan_view(anchors or []),
    }


def brief_node(state) -> dict:
    """Draft (or revise, when the user replied at the plan gate) the brief.
    ``brief.draft`` exhausted → ``llm.EngineError`` propagates (the role must
    raise); every attempt that returned is in ``model_calls``."""
    rc = run_config.require(state)
    user_request = state.get("user_request") or ""
    feedback = (state.get("plan_feedback") or "").strip()
    prev = state.get("brief") if feedback else None
    # what the entity phrases must come from: an entity of an earlier plan edit survives the next one
    asked = "\n".join([user_request, feedback, *((prev or {}).get("entities") or [])])
    calls: list[dict] = []
    # the user's own links, read before drafting (a re-draft keeps those already read) — from the user's
    # own words too: intake's refined request may drop a link ("…from the provided GitHub URL", run e02ea8b8)
    raw = next((m.get("content") or "" for m in state.get("messages") or [] if m.get("role") == "user"), "")
    ledger = SpendLedger.for_run(rc, state.get("search_spend"))
    anchors, anchor_sources, errors = read_anchors(f"{raw}\n{user_request}\n{feedback}", rc=rc, ledger=ledger,
                                                   known=state.get("anchors"))
    context = brief_context(anchors)
    context = f"\n{context}\n" if context else ""

    def _draft(extra: str = "") -> ResearchBrief:
        if feedback:
            human = (
                f"Topic / question: {user_request}\n{context}\n"
                f"PREVIOUS BRIEF (JSON):\n{json.dumps(prev or {}, indent=2)}\n\n"
                f"USER FEEDBACK — revise the brief accordingly:\n{feedback}\n"
                f"{extra}\n\nReturn the revised brief as JSON."
            )
        else:
            human = (f"Topic / question: {user_request}\n{context}{extra}\n"
                     "Draft the research brief as JSON.")
        res = llm.complete("brief.draft", [SystemMessage(content=BRIEF_SYSTEM),
                                           HumanMessage(content=human[:14000])],
                           engines=rc.engines, temperature=0.2, max_tokens=1200,
                           validate=lambda text: _parse_brief(text, asked))
        calls.extend(r.model_dump() for r in res.records)
        return res.parsed

    brief = _draft()
    failures = check_brief_quality(brief, user_request)
    if failures:
        # ONE bounded re-draft with the failure diagnoses in hand.
        brief2 = _draft(extra=("\nYour previous brief was rejected: "
                               + "; ".join(failures)
                               + ". Fix these gaps."))
        if not check_brief_quality(brief2, user_request):
            brief = brief2
            failures = []
        elif len(brief2.sub_questions) > len(brief.sub_questions):
            brief = brief2  # still weak but strictly better
    if failures:
        errors.append(f"brief: weak after re-draft: {failures}")

    plan = brief_to_plan(brief, anchors=anchors)
    read = sum(1 for a in anchors if a["ok"])
    msg = (f"📋 Brief drafted: {len(brief.sub_questions)} sub-questions"
           + (f", {read} of your {len(anchors)} link(s) read" if anchors else "")
           + ". Scouting live sources…")
    out_sources = {"sources": [*anchor_sources, *(state.get("sources") or [])]} if anchor_sources else {}
    return {
        **out_sources,
        "anchors": anchors,
        "search_spend": ledger.snapshot(),
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
