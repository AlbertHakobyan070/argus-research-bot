"""Argus LangGraph node implementations.

Each node is a function (state) -> partial state update. We deliberately
keep the implementations small and obvious rather than reaching for
prebuilt agents — the contract says "explicit node/subgraph, no black-box
prebuilt agent".
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .. import llm
from ..tools import (
    HarvestResult, harvest_sources, snatch_url, crawl_url,
    normalize_to_markdown, markdown_to_pdf,
)
from .state import (
    ArgusState, Finding, FetchedItem, PlannedSource,
    ResearchPlan, ReviewVerdict, ValidatedAssessment,
)
from .synthesis_modes import get_mode, is_valid as is_valid_length
from .credibility import score_fetched, CREDIBILITY_FLOOR
from .researcher_subgraph import run_researcher_subgraph

logger = logging.getLogger("argus.nodes")


# ---------------------------------------------------------------------------
# intake
# ---------------------------------------------------------------------------

INTAKE_SYSTEM = """You are the intake classifier for Argus, a research bot.

Given the user's raw request, decide:
- mode: "quick" for short factual / definitional questions; "deep" for
  anything that wants a report with sources.
- refined_query: a clean version of the user's request to feed the planner.

Return ONLY valid JSON matching the schema:
{"mode": "quick" | "deep", "refined_query": "..."}
"""


def intake_node(state: ArgusState) -> dict:
    """Classify request as quick vs deep and clean it up."""
    raw = state.get("user_request") or ""
    chat = llm.chat_for_tier("cheap", temperature=0.0, max_tokens=200)
    resp = chat.invoke([
        SystemMessage(content=INTAKE_SYSTEM),
        HumanMessage(content=raw[:4000]),
    ])
    rec = llm.record_from_response("cheap", "cheap", resp)
    try:
        parsed = _parse_json_obj(resp.content)
    except Exception:
        parsed = {"mode": "deep" if len(raw) > 80 else "quick",
                  "refined_query": raw.strip()}
    mode = parsed.get("mode", "deep")
    if mode not in ("quick", "deep"):
        mode = "deep"
    return {
        "mode": mode,
        "user_request": parsed.get("refined_query", raw).strip() or raw,
        "model_calls": [rec.model_dump()],
        "messages": [{"role": "user", "content": raw, "ts": _ts()}],
    }


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are Argus's research planner.

Produce a structured research plan that maximises the chance of finding
*primary-source* evidence: arXiv preprints, HF models, official docs,
named engineers' blogs, GitHub repos, Hacker News threads. SEO content
farms are banned as evidence — never include them in `planned_sources`.

# URL INTEGRITY RULES (strictly enforced)

1. **Do NOT invent URLs.** If you do not have a specific URL in mind, leave
   `target_url` empty / null and use `kind: "search_result"` instead. The
   researcher will run the query and find real primary sources.
2. **Do NOT guess arXiv IDs.** Real arXiv IDs look like `arxiv.org/abs/2402.12345`.
   If you do not know the exact ID, emit `kind: "search_result"` with a query
   such as `"transformer interpretability survey 2024"`. Never invent numeric IDs.
3. **Verify URLs by reasoning:** if you cite a github.com URL, the repo must
   plausibly exist (e.g. `github.com/openai/gpt-4`, `github.com/NVIDIA-AI-Blueprints/aiq`).
   If unsure, use `kind: "search_result"`.
4. **Prefer primary-source URLs you are confident about** (official docs at
   `docs.nvidia.com`, `python.langchain.com`, `huggingface.co/docs`; well-known
   GitHub orgs). When unsure, fall back to `search_result` — that is the safe
   default and the researcher will resolve it.

Return JSON only, matching:
{
  "sub_questions": ["...", "..."],
  "planned_sources": [
    {"kind": "paper|repo|news|blog|official_doc|search_result",
     "query": "search string or paper title",
     "target_url": "https://... (only when you know the exact URL)",
     "rationale": "why this source is primary"}
  ],
  "must_have_keywords": ["...", "..."],
  "summary": "1-2 sentence plan summary"
}

Aim for 4-7 sub_questions and 6-12 planned_sources. Mark each source's
kind correctly. Prefer primary kinds (paper/repo/official_doc) when you
know the right venue; otherwise emit `kind: "search_result"` with a query.
"""


def planner_node(state: ArgusState) -> dict:
    """LLM drafts a research plan. Triggers HITL interrupt."""
    chat = llm.chat_for_tier("strong", temperature=0.2, max_tokens=1200)
    resp = chat.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=(
            f"Topic / question: {state['user_request']}\n\n"
            "Draft the research plan."
        )),
    ])
    rec = llm.record_from_response("strong", llm.resolve_tier("strong"), resp)
    planner_errors: list[str] = []
    try:
        data = _parse_json_obj(resp.content)
        plan = ResearchPlan.model_validate(data)
    except Exception as e:
        logger.warning("Planner returned non-JSON (%s); falling back.", e)
        logger.debug("raw planner output: %r", resp.content[:1000])
        # NEVER put the raw LLM output in the summary — _format_plan
        # renders it in the Telegram preview (raw ```json blob observed
        # live, 2026-07-10). Neutral notice + loud errors entry instead.
        plan = ResearchPlan(
            sub_questions=[state["user_request"]],
            planned_sources=[PlannedSource(
                kind="search_result",
                query=state["user_request"],
                rationale="Fallback: a single web search.",
            )],
            summary=("⚠️ Planner output was unparseable — falling back to "
                     "a single-search plan. Live search still runs; Edit "
                     "or Cancel if the topic needs a richer plan."),
        )
        planner_errors.append(f"planner: unparseable output ({e})")
    return {
            "plan": plan.model_dump(),
            "errors": planner_errors,
            "model_calls": [rec.model_dump()],
            "messages": [{"role": "assistant", "content": (
                f"📋 Drafted plan with {len(plan.planned_sources)} sources. "
                "Awaiting your approval."), "ts": _ts()}],
            # Set HITL pending; the bot layer will surface this and the
            # LangGraph interrupt will pause the run.
            "hitl": {"pending": True, "kind": "plan_approval",
                     "ctx": {"plan": plan.model_dump()}},
        }


# ---------------------------------------------------------------------------
# planner_reflect (handoff action #7 — HANDOFF-RESEARCH-2026-07-08.md §6)
# ---------------------------------------------------------------------------
#
# Sits between planner_node and researcher_node. Inspects the freshly
# drafted plan against five quality signals and re-invokes planner_node
# ONCE if any signal fails. Bounded at MAX_ATTEMPTS = 2 so a chronically
# bad planner cannot loop indefinitely; on the second weak plan we
# proceed with a warning so the rest of the pipeline can still run.
#
# Rationale: T6-class orphan-topic bugs (planner produces an empty /
# single-source / lazy plan, researcher ends up with zero candidates,
# report says "no evidence") were being papered over by the researcher's
# fallback path. Catching them at planning time means the fallback only
# fires for genuinely niche topics that no reasonable plan can cover.
#
# Quality signals:
#   (a) planned_sources count  >= 3 ideal; < 2 triggers re-plan
#   (b) source_type diversity   >= 2 distinct kinds
#   (c) sub_questions count     >= 3
#   (d) no sub_question is too similar to user_request verbatim
#   (e) at least 1 planned_source mentions a word from user_request
# ---------------------------------------------------------------------------

MAX_PLAN_ATTEMPTS = 2
REFLECT_MIN_SOURCES = 3           # >= ideal (a)
REFLECT_MIN_SOURCE_KINDS = 2      # diversity (b)
REFLECT_MIN_SUB_QUESTIONS = 3     # (c)
REFLECT_VERBATIM_OVERLAP = 0.9    # >= of sub_question chars that must
                                  # match user_request before we treat
                                  # the sub_question as a verbatim
                                  # echo of the request (lazy plan)


def _normalise(s: str) -> str:
    return (s or "").strip().lower()


def _verbatim_overlap_ratio(text: str, reference: str) -> float:
    """Return the fraction of words in ``text`` that also appear in
    ``reference`` (set intersection). Used to detect sub_questions that
    are just the user_request echoed back, which is a lazy-plan signal.
    """
    t = _normalise(text)
    r = _normalise(reference)
    if not t or not r:
        return 0.0
    t_words = {w for w in re.findall(r"\w+", t) if len(w) > 2}
    r_words = {w for w in re.findall(r"\w+", r) if len(w) > 2}
    if not t_words:
        return 0.0
    return len(t_words & r_words) / len(t_words)


def _reflect_plan_quality(plan: dict | None,
                          user_request: str) -> list[tuple[str, bool, str]]:
    """Run the five plan-quality checks.

    Returns a list of ``(check_name, passed, detail)`` tuples so callers
    can both log the failing reasons and decide what to do.
    """
    plan = plan or {}
    sources = plan.get("planned_sources") or []
    sub_qs = plan.get("sub_questions") or []
    kinds = {s.get("kind", "") for s in sources if s.get("kind")}

    req_norm = _normalise(user_request)
    req_words = {w for w in re.findall(r"\w+", req_norm) if len(w) > 2}

    # (a) planned_sources count
    src_ok = len(sources) >= REFLECT_MIN_SOURCES
    src_detail = f"{len(sources)} planned_sources (need >= {REFLECT_MIN_SOURCES})"

    # (b) source_type diversity
    div_ok = len(kinds) >= REFLECT_MIN_SOURCE_KINDS
    div_detail = f"{len(kinds)} distinct kinds {sorted(kinds)} (need >= {REFLECT_MIN_SOURCE_KINDS})"

    # (c) sub_questions count
    sq_ok = len(sub_qs) >= REFLECT_MIN_SUB_QUESTIONS
    sq_detail = f"{len(sub_qs)} sub_questions (need >= {REFLECT_MIN_SUB_QUESTIONS})"

    # (d) any sub_question is too similar to user_request verbatim
    dupe_idx = -1
    dupe_ratio = 0.0
    for i, sq in enumerate(sub_qs):
        r = _verbatim_overlap_ratio(sq, user_request)
        if r > dupe_ratio:
            dupe_ratio = r
            dupe_idx = i
    dup_ok = dupe_ratio < REFLECT_VERBATIM_OVERLAP
    dup_detail = (f"max verbatim overlap {dupe_ratio:.2f} "
                  f"(threshold {REFLECT_VERBATIM_OVERLAP}) "
                  f"on sub_question #{dupe_idx}")

    # (e) at least one planned_source mentions a word from user_request
    hit_idx = -1
    if req_words:
        for i, s in enumerate(sources):
            haystack = _normalise(
                (s.get("query") or "") + " " + (s.get("rationale") or "")
            )
            if any(w in haystack for w in req_words):
                hit_idx = i
                break
    kw_ok = hit_idx >= 0
    kw_detail = (f"keyword hit on planned_source #{hit_idx}"
                 if kw_ok else
                 f"no planned_source mentions a word from user_request "
                 f"({sorted(req_words)[:5]})")

    return [
        ("sources_count",       src_ok, src_detail),
        ("source_diversity",    div_ok, div_detail),
        ("sub_questions_count", sq_ok,  sq_detail),
        ("sub_question_duplicate", dup_ok, dup_detail),
        ("keyword_coverage",    kw_ok,  kw_detail),
    ]


def planner_reflect_node(state: ArgusState) -> dict:
    """Reflect on the freshly-drafted plan; re-plan ONCE if weak.

    Reads ``state["plan"]`` and ``state["user_request"]``, runs
    :func:`_reflect_plan_quality`, and:
      • if every check passes: returns ``plan_attempts=1`` (or whatever
        it was) with no re-plan. The pipeline proceeds to researcher.
      • if any check fails AND ``plan_attempts < MAX_PLAN_ATTEMPTS``:
        appends reflection notes to ``state["plan"]["reflection_notes"]``,
        calls ``planner_node(state)`` to draft a replacement, merges the
        returned ``plan`` / ``model_calls`` / ``messages`` into state,
        and increments ``plan_attempts``.
      • if any check fails AND we have already hit
        ``MAX_PLAN_ATTEMPTS``: returns the (still-weak) plan with a
        warning appended to ``messages`` so downstream nodes know the
        plan was weak but the user isn't stuck.

    The reflect node never raises on a malformed plan — an empty dict
    simply fails every check, which is the correct diagnostic.
    """
    user_request = state.get("user_request") or ""
    plan = state.get("plan")
    plan_attempts = int(state.get("plan_attempts") or 0)

    # Bump the counter for *this* planner invocation. The reflect node
    # runs once per planner call, so a freshly-set state arrives with
    # plan_attempts == 1 (the initial planner_node ran). If we're being
    # called as a re-plan (state already has reflection_notes from a
    # previous reflect pass), increment further.
    if plan_attempts < 1:
        plan_attempts = 1
    checks = _reflect_plan_quality(plan, user_request)
    failed = [(name, detail) for (name, ok, detail) in checks if not ok]

    if not failed:
            return {
                "plan": plan,
                "plan_attempts": plan_attempts,
                "messages": [{"role": "assistant",
                              "content": (f"✅ plan passed reflect "
                                          f"({len(checks)} checks ok)."),
                              "ts": _ts()}],
            }

    # Plan is weak. Decide: re-plan or warn-and-proceed.
    if plan_attempts >= MAX_PLAN_ATTEMPTS:
        # Bounded: we've already tried twice. Proceed with warning so
        # the user can still get a (likely-thin) report rather than
        # hanging at the HITL plan-approval gate forever.
        return {
            "plan_attempts": plan_attempts,
            "messages": [{"role": "assistant",
                          "content": (f"⚠️ plan still weak after "
                                      f"{plan_attempts} attempts — "
                                      f"proceeding. failed checks: "
                                      f"{[n for n, _ in failed]}"),
                          "ts": _ts()}],
            "errors": [f"planner_reflect: plan weak after "
                       f"{plan_attempts} attempts: "
                       f"{[n for n, _ in failed]}"],
        }

    # Re-plan: append reflection notes to the current plan dict and
    # call planner_node again so the LLM can produce a fresh plan with
    # the weakness diagnoses in hand.
    plan_dict = dict(plan or {})
    existing_notes = list(plan_dict.get("reflection_notes") or [])
    existing_notes.append({
        "attempt": plan_attempts,
        "failed_checks": [name for name, _ in failed],
        "details": {name: detail for name, detail in failed},
    })
    plan_dict["reflection_notes"] = existing_notes
    # Build a state for the planner that includes the reflection notes
    # in the user_request context — the planner's prompt reads
    # ``state["user_request"]`` and we don't want to overwrite the
    # original request, so we stash the notes on a side-channel key.
    planner_state = dict(state)
    planner_state["plan"] = plan_dict
    # Reflection guidance for the planner LLM: tell it why the last
    # plan was rejected so it can target the gaps.
    reflection_brief = (
        "Your previous plan was rejected by planner_reflect. "
        "Re-plan addressing these gaps:\n"
        + "\n".join(f"- {name}: {detail}" for name, detail in failed)
        + "\n\nUser request: " + user_request
    )
    planner_state["user_request"] = reflection_brief

    replanned = planner_node(planner_state)

    # Merge: keep the original user_request on the returned state so
    # downstream nodes (researcher, etc.) see the right thing, not the
    # reflection-brief.
    replanned["user_request"] = user_request
    replanned["plan_attempts"] = plan_attempts + 1
    replanned.setdefault("messages", []).append({
        "role": "assistant",
        "content": (f"🔁 re-planning (attempt {plan_attempts + 1}/"
                    f"{MAX_PLAN_ATTEMPTS}) — failed checks: "
                    f"{[n for n, _ in failed]}"),
        "ts": _ts(),
    })
    return replanned


# ---------------------------------------------------------------------------
# researcher
# ---------------------------------------------------------------------------

def researcher_node(state: ArgusState) -> dict:
    """Gather candidate sources via the 3-way researcher subgraph.

    Delegates to :func:`run_researcher_subgraph`, which fans out to three
    sub-researchers in parallel — arXiv (papers), GitHub (repos), and web
    search (blogs / news / official docs via DDGS) — and merges their
    results by URL. This replaces the legacy single-node
    harvest + arXiv + planner-URL path.

    Two properties matter for the bugs this fixes (2026-07-09):

    * **No hallucinated URLs.** The subgraph searches fresh for each source
      kind instead of trusting the planner's ``target_url`` values, so the
      planner's invented URLs never enter the candidate set. (The old node
      copied ``planned_source.target_url`` straight into ``sources`` — that
      is how fabricated links like ``https://nvidia.com/ai-blueprint``
      reached the fetcher.)
    * **Failure isolation + observability.** A failure in any one sub is
      captured as an error string and surfaced in ``state["errors"]`` (via
      the ``operator.add`` reducer) without poisoning the other subs.

    The arXiv T6 orphan-topic fallback (raw ``user_request`` query when the
    structured plan yields nothing) is preserved inside ``arxiv_sub``.
    Pre-seeded ``state["sources"]`` (demo/static corpus) are honoured — the
    subgraph's merge step lists them first. See ``researcher_subgraph.py``.
    """
    return run_researcher_subgraph(state)


def _matches_plan(text: str, plan: ResearchPlan) -> bool:
    text = text.lower()
    keywords = [k.lower() for k in plan.must_have_keywords]
    if not keywords:
        return True
    hits = sum(1 for k in keywords if k in text)
    return hits >= max(1, len(keywords) // 3)


def _arxiv_search_raw(query: str) -> list[dict]:
    """T6 fallback: arXiv search driven by a free-form query string.

    Used when researcher_node ends up with zero candidate sources after
    the standard harvest + plan-driven arxiv pass (orphan / niche
    topic). Same Atom-XML parsing as _arxiv_search but bypasses the
    ResearchPlan object.
    """
    import httpx
    q = (query or "").strip()[:200]
    if not q:
        return []
    params = {
        "search_query": f"all:{q}",
        "start": 0, "max_results": 8,
        "sortBy": "relevance", "sortOrder": "descending",
    }
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as c:
            r = c.get("https://export.arxiv.org/api/query", params=params)
            r.raise_for_status()
    except Exception:
        return []
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.DOTALL)
    out: list[dict] = []
    for ent in entries:
        title_m = re.search(r"<title>(.*?)</title>", ent, re.DOTALL)
        link_m = re.search(r"<id>(.*?)</id>", ent)
        sum_m = re.search(r"<summary>(.*?)</summary>", ent, re.DOTALL)
        title = (title_m.group(1).strip() if title_m else "")
        link = (link_m.group(1).strip() if link_m else "")
        summary = (sum_m.group(1).strip()[:400] if sum_m else "")
        if title and link and link.startswith("http"):
            out.append({
                "kind": "paper", "title": title, "url": link,
                "summary": summary, "source": "arxiv-raw",
            })
    return out[:8]


def _arxiv_search(plan: ResearchPlan) -> list[dict]:
    """Use the arxiv skill (httpx, no key) for a real arXiv search."""
    import httpx
    query = " ".join(plan.must_have_keywords[:5]) or plan.sub_questions[0]
    url = "https://export.arxiv.org/api/query"  # arXiv moved to HTTPS
    params = {
        "search_query": f"all:{query}",
        "start": 0, "max_results": 6,
        "sortBy": "relevance", "sortOrder": "descending",
    }
    with httpx.Client(timeout=15.0, follow_redirects=True) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
    # Minimal Atom parse: extract <entry> blocks.
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.DOTALL)
    out: list[dict] = []
    for ent in entries:
        title_m = re.search(r"<title>(.*?)</title>", ent, re.DOTALL)
        link_m = re.search(r"<id>(.*?)</id>", ent)
        sum_m = re.search(r"<summary>(.*?)</summary>", ent, re.DOTALL)
        title = (title_m.group(1).strip() if title_m else "")
        link = (link_m.group(1).strip() if link_m else "")
        summary = (sum_m.group(1).strip()[:400] if sum_m else "")
        if title and link:
            out.append({
                "kind": "paper", "title": title, "url": link,
                "summary": summary, "source": "arxiv",
            })
    return out


# ---------------------------------------------------------------------------
# fetcher
# ---------------------------------------------------------------------------

def fetcher_node(state: ArgusState) -> dict:
    """Fetch each source URL into markdown via snatch/crawl/normalize.

    Per-URL tool failures append to ``state["errors"]`` (via the
    ``Annotated[list[str], operator.add]`` reducer) instead of being
    silently logged-and-forgotten. If *every* URL failed we also append
    a summary error so the report can surface "all fetches failed" rather
    than the synthesizer running against an empty evidence list and
    producing a vacuous "no evidence" report. See Argus T2 (Pattern E).
    """
    fetched: list[dict] = []
    errors: list[str] = []
    sources = state.get("sources") or []
    for src in sources:
        url = src.get("url")
        # T6 fix: a source without a URL is not fetchable. Record and
        # skip without crashing the whole node. Old code raised
        # KeyError on src["url"], caught by the broad except, logged
        # only -> the user got a silent "no evidence" report.
        if not url or not isinstance(url, str) or not url.startswith("http"):
            msg = f"fetcher skipping source with no/empty/non-http URL: {src!r}"
            logger.info(msg)
            errors.append(msg)
            continue
        kind = src.get("kind", "search_result")
        try:
            # Pick tool based on kind.
            if kind == "official_doc" or "github.com" in url or "arxiv.org" in url:
                # use normalize (article_convert) — works for both
                res = normalize_to_markdown(url)
                if res.ok and res.markdown_path:
                    fetched.append(FetchedItem(
                        url=url, title=res.title or src.get("title", ""),
                        markdown_path=res.markdown_path,
                        section=kind,
                        excerpt=(res.markdown_text or "")[:600],
                    ).model_dump())
                    continue
            if kind == "paper" and "arxiv.org/abs/" in url:
                # convert abs to pdf url, then snatch as paper
                pdf = url.replace("/abs/", "/pdf/") + ".pdf"
                res = snatch_url(pdf, kind="papers")
                # T5 fix: also require markdown_path on disk, matching
                # every other branch in this function. Otherwise a
                # silently-empty snatch result (rc=0 but no file)
                # would record a fake "fetched" entry with no evidence.
                if res.ok and res.markdown_path and Path(res.markdown_path).exists():
                    fetched.append(FetchedItem(
                        url=url, title=res.title or src.get("title", ""),
                        markdown_path=res.markdown_path,
                        section=kind,
                        excerpt="",
                    ).model_dump())
                    continue
            # default: try snatch first
            res = snatch_url(url, kind="auto")
            if res.ok and res.markdown_path:
                fetched.append(FetchedItem(
                    url=url, title=res.title or src.get("title", ""),
                    markdown_path=res.markdown_path,
                    section=kind,
                    excerpt=_read_excerpt(res.markdown_path),
                ).model_dump())
                continue
            # fallback: crawl
            res = crawl_url(url, deep=False, max_pages=4)
            if res.ok and res.markdown_path:
                fetched.append(FetchedItem(
                    url=url, title=src.get("title", ""),
                    markdown_path=res.markdown_path,
                    section=kind,
                    excerpt=_read_excerpt(res.markdown_path),
                ).model_dump())
                continue
            # Phase 2 last resort: bot-walled / Cloudflare sites (the plain
            # requests GET inside snatch/crawl returns 403 or a JS challenge).
            # Retry once with Scrapling's stealth fetcher (camoufox). Kept
            # last because it spins up a headless browser and is slow.
            res = snatch_url(url, kind="auto", stealth=True)
            if res.ok and res.markdown_path:
                fetched.append(FetchedItem(
                    url=url, title=res.title or src.get("title", ""),
                    markdown_path=res.markdown_path,
                    section=kind,
                    excerpt=_read_excerpt(res.markdown_path),
                ).model_dump())
                continue
            # All fetch strategies (normalize/snatch/crawl/stealth) failed.
            msg = f"fetch {url} failed: all tools returned not-ok (incl. stealth)"
            errors.append(msg)
            logger.warning(msg)
        except Exception as e:
            msg = f"fetch {url} failed: {e!r}"
            logger.warning(msg)
            errors.append(msg)
    if sources and not fetched:
        # Every URL failed. Surface this as a single summary error so the
        # synthesizer's "no evidence" report can be replaced with an
        # actionable diagnostic.
        errors.append(
            f"fetcher_node: all {len(sources)} source(s) failed to fetch"
        )
    out: dict[str, Any] = {
        "fetched": fetched,
        "messages": [{"role": "assistant",
                      "content": f"📥 {len(fetched)} items fetched."}],
    }
    if errors:
        out["errors"] = errors
    return out


def _read_excerpt(md_path: str | None, limit: int = 600) -> str:
    if not md_path:
        return ""
    try:
        text = Path(md_path).read_text(encoding="utf-8", errors="replace")
        return text[:limit]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# normalizer (alias for any not-yet-normalized items; minimal because the
# fetcher already normalizes via the underlying scripts).
# ---------------------------------------------------------------------------

def normalizer_node(state: ArgusState) -> dict:
    """Re-confirm every fetched item has a markdown_path on disk.

    Most work happens in fetcher; this node only patches missing ones.
    """
    patched = []
    for f in state.get("fetched") or []:
        item = FetchedItem.model_validate(f)
        if not item.markdown_path or not Path(item.markdown_path).exists():
            # try once more via normalize
            res = normalize_to_markdown(item.url)
            if res.ok and res.markdown_path:
                item.markdown_path = res.markdown_path
                item.excerpt = (res.markdown_text or "")[:600]
                item.title = item.title or res.title
        patched.append(item.model_dump())
    return {"fetched": patched}


# ---------------------------------------------------------------------------
# credibility — tag fetched items with a 0..1 trust score.
# Items below the 0.4 floor get `credibility_flag = "low_credibility"` but
# are NOT dropped; the downstream filter_node still does its own keep_topN.
# ---------------------------------------------------------------------------

def credibility_node(state: ArgusState) -> dict:
    """Score every fetched item on (domain trust + URL pattern + title
    relevance). Pure local compute — no LLM call.

    Returns the updated ``fetched`` list with ``credibility_score`` and
    ``credibility_flag`` fields populated. If ``fetched`` is empty we
    pass through cleanly (no errors, no model_calls).
    """
    user_request = state.get("user_request") or ""
    fetched = [FetchedItem.model_validate(f) for f in state.get("fetched") or []]
    if not fetched:
        return {
            "fetched": [],
            "messages": [{"role": "assistant",
                          "content": "🛡️ 0 items to score (nothing fetched)."}],
        }
    scored = score_fetched(fetched, user_request=user_request)
    flagged = sum(1 for s in scored if (s.credibility_score or 0.0) < CREDIBILITY_FLOOR)
    return {
        "fetched": [s.model_dump() for s in scored],
        "messages": [{"role": "assistant",
                      "content": (f"🛡️ {len(scored)} items scored, "
                                  f"{flagged} flagged as low-credibility "
                                  f"(below {CREDIBILITY_FLOOR:.1f}).")}],
    }


# ---------------------------------------------------------------------------
# filter / rank
# ---------------------------------------------------------------------------

_FILTER_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "into", "that", "this", "what",
    "how", "why", "are", "was", "were", "has", "have", "its", "their",
})


def _keyword_words(keywords: list[str]) -> list[str]:
    """Tokenise the plan's must_have_keywords into distinct signal words.

    The old filter matched each *whole* keyword phrase (e.g.
    ``"nvidia ai-q blueprint"``) as a literal substring, so a source whose
    text said "AI-Q NVIDIA Blueprint" (different word order) scored zero and
    got dropped — the empty-report bug on the AI-Q query (2026-07-08). We
    split phrases into words so partial, out-of-order overlap still counts.
    Words shorter than 3 chars are dropped (they match too much as
    substrings, e.g. "ai" in "domain").
    """
    words: list[str] = []
    seen: set[str] = set()
    for k in keywords:
        for w in re.split(r"[^a-z0-9]+", (k or "").lower()):
            if len(w) >= 3 and w not in _FILTER_STOPWORDS and w not in seen:
                seen.add(w)
                words.append(w)
    return words


def filter_node(state: ArgusState) -> dict:
    """Rank fetched sources and keep the top-N, with a credibility floor.

    P2 fix (2026-07-09): previously ``filter_node`` ranked by
    ``(relevance, credibility)`` and capped at 14 but **never** dropped
    items below :data:`credibility.CREDIBILITY_FLOOR`. A neutral-scored
    content farm with high keyword overlap survived and got cited as
    evidence (thetechbriefs in the GLM 5.2 report). Now we enforce the
    floor: drop items below the floor. If that empties the set, fall
    back to the pre-floor top-N so the report is never empty (matches
    the Phase-1 "don't empty the report" principle).
    """
    from .credibility import CREDIBILITY_FLOOR

    fetched = [FetchedItem.model_validate(f) for f in state.get("fetched") or []]
    plan = ResearchPlan.model_validate(state.get("plan") or {})
    kw_words = _keyword_words(plan.must_have_keywords) or \
        _keyword_words([state.get("user_request", "")])
    kept: list[FetchedItem] = []
    for f in fetched:
        if not f.markdown_path:
            continue  # no evidence body on disk → nothing to synthesize from
        haystack = (f.title + " " + f.excerpt).lower()
        if kw_words:
            hits = sum(1 for w in kw_words if w in haystack)
            f.relevance_score = hits / len(kw_words)
        # else: leave relevance_score as-is (e.g. pre-seeded test corpus)
        kept.append(f)
    kept.sort(key=lambda x: (x.relevance_score, x.credibility_score or 0.0),
              reverse=True)
    full_ranked = kept[:14]

    # P2 — enforce credibility floor with safety net. If dropping
    # below-floor items empties the report, fall back to full_ranked
    # (sorted list) so we never deliver a 0-sourced report for a
    # difficult query.
    above_floor = [
        f for f in full_ranked
        if (f.credibility_score or 0.0) >= CREDIBILITY_FLOOR
    ]
    if above_floor:
        kept = above_floor
    else:
        # Safety net. Same shape as the Phase-1 "don't empty the
        # report" guard.
        kept = full_ranked

    out: dict[str, Any] = {"fetched": [f.model_dump() for f in kept]}
    if fetched and not kept:
        out["errors"] = [
            f"filter_node: dropped all {len(fetched)} fetched item(s) — "
            "none had a markdown body on disk"
        ]
    return out


# ---------------------------------------------------------------------------
# synthesizer
# ---------------------------------------------------------------------------

# T7 — per-mode synthesizer prompts. Each template has a target
# length + structure the LLM is asked to match, plus a JSON-return
# contract (with an optional validated_assessment block for long /
# lecture modes).
#
# We keep the original SYNTH_SYSTEM for the short / flat path so the
# behaviour change vs T6 is zero for users on default "short"; the
# other modes get bespoke prompts that explicitly ask for the
# extended structure.

SYNTH_SYSTEM = """You are Argus's senior research synthesizer.

You will be given a topic, the user's question, and a set of fetched
evidence items (each with a URL and a markdown excerpt). Produce a
grounded report:

1. Every factual claim MUST cite at least one evidence URL using
   the literal format [n] where n is the index in the evidence list.
2. Only use information that appears in the evidence. If the evidence
   is insufficient for a claim, omit it (do NOT fabricate).
3. You may ONLY cite [n] slots where n is a real index in the numbered
   evidence list you are given. Never invent an [n] ref or [n] entry
   in the Sources block — empty/dangling citation slots are dropped
   before the report ships, and any claim that depends on a missing
   ref is removed with it.
3. Structure: ## TL;DR, ## Key Findings (each finding = claim + citations),
   ## Sources (numbered), ## Open Questions.
4. Keep prose tight; prefer numbered findings over paragraphs.

Return a JSON object:
{
  "findings": [
    {"claim": "...", "citation_urls": ["https://...", "..."],
     "confidence": "high|medium|low"},
    ...
  ],
  "draft_md": "# Title\\n\\n## TL;DR\\n...\\n## Sources\\n..."
}

Make sure every URL in citation_urls appears in the Sources section.
"""


SYNTH_SYSTEM_MEDIUM = """You are Argus's senior research synthesizer producing a
MEDIUM-length report (target 3000-6500 chars, ~8 cited findings,
sub-headings for organization).

You will be given a topic, the user's question, and a set of fetched
evidence items. Produce a grounded report:

1. Every factual claim MUST cite at least one evidence URL using
   the literal format [n] where n is the index in the evidence list.
2. Only use information that appears in the evidence. If the evidence
   is insufficient for a claim, omit it (do NOT fabricate).
3. You may ONLY cite [n] slots where n is a real index in the numbered
   evidence list you are given. Never invent an [n] ref or [n] entry
   in the Sources block — empty/dangling citation slots are dropped
   before the report ships, and any claim that depends on a missing
   ref is removed with it.
3. Structure: # Title, ## TL;DR (1 short paragraph), then ## Key Findings
   (each finding = claim + citations + inline confidence), then
   ## Background (1-2 short paragraphs of context, cited), then
   ## Current state (2-3 short paragraphs / bullet groups, cited),
   then ## Open questions (1-2 bullets), then ## Sources (numbered).
4. Aim for ~8 findings. Each finding 1-2 sentences.
5. Sub-headings (### level) welcome inside ## Current state.

Return JSON only:
{
  "findings": [
    {"claim": "...", "citation_urls": ["https://...", "..."],
     "confidence": "high|medium|low"},
    ...
  ],
  "draft_md": "the full markdown report, ready to render"
}
"""


SYNTH_SYSTEM_LONG = """You are Argus's senior research synthesizer producing a
LONG report (target 10000-16000 chars, ~12 cited findings, sub-headings,
validated assessment).

You will be given a topic, the user's question, and a set of fetched
evidence items. Produce a deeply-grounded, well-structured report:

1. Every factual claim MUST cite at least one evidence URL using
   the literal format [n] where n is the index in the evidence list.
2. Only use information that appears in the evidence. If the evidence
   is insufficient for a claim, omit it (do NOT fabricate).
3. You may ONLY cite [n] slots where n is a real index in the numbered
   evidence list you are given. Never invent an [n] ref or [n] entry
   in the Sources block — empty/dangling citation slots are dropped
   before the report ships, and any claim that depends on a missing
   ref is removed with it.
3. Structure: # Title, ## TL;DR (1 paragraph), ## Background (1-2
   paragraphs of context, cited), ## Current state of the field
   (3-5 sub-sections, deeply cited, mix of paragraphs and bulleted
   findings), ## Open problems (1-2 paragraphs, more speculative
   but still cited where evidence exists), ## Sources (numbered),
   ## Quality assessment (per-section confidence + open challenges,
   see schema below).
4. Aim for ~12 findings. Findings are NOT a flat list — each finding
   belongs to a sub-section of ## Current state and references back
   to that sub-section in its "section" field.
5. Validated assessment: rate your confidence per ## section
   (high/medium/low), and surface "open challenges" for sections
   where evidence is thin or conflicting.

Return JSON only:
{
  "findings": [
    {"claim": "...", "citation_urls": ["https://...", "..."],
     "confidence": "high|medium|low",
     "section": "Current state > <sub-section name>"},
    ...
  ],
  "draft_md": "the full markdown report, ready to render",
  "validated_assessment": {
    "sections": [
      {"name": "Background",
       "confidence": "high|medium|low",
       "open_challenges": ["...","..."],
       "claim_count": 0},
      ...
    ],
    "overall_relevancy": "high|medium|low",
    "knowledge_density": {
      "claims_per_section": {"Background": 0, ...},
      "citations_per_claim": 0.0,
      "conflict_markers": 0
    }
  }
}


CRITICAL (added 2026-07-09 - empty-Long-mode bug fix): if the evidence
does not support even ONE grounded finding, do NOT return an empty
`findings[]` field silently. Instead, return exactly:

  {"findings": [], "no_evidence": true,
   "no_evidence_reason": "<one short sentence naming what is missing>"}

The report builder uses the `no_evidence=true` flag to surface a
classified "synthesis_no_evidence" marker instead of a misleading
1.4 KB skeleton. An empty `findings[]` without `no_evidence=true` is
treated as a synthesis bug and produces a loud failure block.
"""


SYNTH_SYSTEM_LECTURE = """You are Argus's senior research synthesizer producing a
LECTURE-style deep-dive (target 14000-24000 chars, ~15 cited findings,
Parts I-IV + References + Appendix).

You will be given a topic, the user's question, and a set of fetched
evidence items. Produce a richly-structured, lecture-format report:

1. Every factual claim MUST cite at least one evidence URL using
   the literal format [n] where n is the index in the evidence list.
2. Only use information that appears in the evidence. If the evidence
   is insufficient for a claim, omit it (do NOT fabricate).
3. You may ONLY cite [n] slots where n is a real index in the numbered
   evidence list you are given. Never invent an [n] ref or [n] entry
   in the Sources block — empty/dangling citation slots are dropped
   before the report ships, and any claim that depends on a missing
   ref is removed with it.
3. STRUCTURE (mandatory, in this exact order):
   - # <Topic>: A Research Report  (title page heading)
   - ## Executive TL;DR (1 paragraph)
   - ## Part I — Background (1-2 sub-sections, deeply cited)
   - ## Part II — Current state of the field (3-5 sub-sections,
     each a mini-essay with cited findings + inline code/diagram
     references where helpful)
   - ## Part III — Open problems (1-2 sub-sections, more speculative
     but still cited where evidence exists; surface strongest
     opposing claims)
   - ## Part IV — Practice and exercises (where applicable: 2-4
     concrete actions a practitioner can take this week)
   - ## References (numbered, with author/year/venue when extractable
     from the URL or evidence excerpt; otherwise URL only)
   - ## Quality assessment + Appendix (per-section confidence +
     tool calls log + density metrics)
4. Aim for ~15 findings spread across Parts I-III. Each finding
   1-3 sentences. Use sub-headings (###) freely.
5. Validated assessment: rate your confidence per ## section /
   Part (high/medium/low), surface open challenges, and compute
   knowledge density (claims_per_section, citations_per_claim,
   conflict_markers).

Return JSON only:
{
  "findings": [
    {"claim": "...", "citation_urls": ["https://...", "..."],
     "confidence": "high|medium|low",
     "section": "Part II > <sub-section name>"},
    ...
  ],
  "draft_md": "the full lecture-format markdown, ready to render",
  "validated_assessment": {
    "sections": [
      {"name": "Part I — Background",
       "confidence": "high|medium|low",
       "open_challenges": ["...","..."],
       "claim_count": 0},
      ...
    ],
    "overall_relevancy": "high|medium|low",
    "knowledge_density": {
      "claims_per_section": {"Part I — Background": 0, ...},
      "citations_per_claim": 0.0,
      "conflict_markers": 0
    }
  },
  "lecture_appendix": {
    "methodology": "1-2 sentence note on what was searched and how.",
    "tool_calls": ["arxiv: query '...'", "harvest: ...", ...],
    "density_metrics": {
      "total_chars": 0,
      "total_findings": 0,
      "total_citations": 0,
      "unique_sources": 0,
      "avg_citations_per_finding": 0.0
    }
  }
}


CRITICAL (added 2026-07-09 - empty-Long-mode bug fix): if the evidence
does not support even ONE grounded finding, do NOT return an empty
`findings[]` field silently. Instead, return exactly:

  {"findings": [], "no_evidence": true,
   "no_evidence_reason": "<one short sentence naming what is missing>"}

The report builder uses the `no_evidence=true` flag to surface a
classified "synthesis_no_evidence" marker instead of a misleading
1.4 KB skeleton. An empty `findings[]` without `no_evidence=true` is
treated as a synthesis bug and produces a loud failure block.
"""


SYNTH_SYSTEM_TLDR = """You are Argus. Produce a single short paragraph
(80-400 chars) that answers the user's question using the evidence
provided. Every factual claim must cite one of the evidence URLs
inline using [n]. If the evidence is too thin to answer, say so.

Return JSON only:
{
  "findings": [],
  "draft_md": "<one short paragraph, possibly with [1] inline citations>"
}
"""


def _synth_prompt_for(length: str) -> str:
    """Return the right system prompt for the requested length mode.

    The branch order matters: lecture > long > medium > tldr > short.
    short keeps the legacy SYNTH_SYSTEM so a default-mode user gets the
    same report shape they got pre-T7.
    """
    if length == "lecture":
        return SYNTH_SYSTEM_LECTURE
    if length == "long":
        return SYNTH_SYSTEM_LONG
    if length == "medium":
        return SYNTH_SYSTEM_MEDIUM
    if length == "tldr":
        return SYNTH_SYSTEM_TLDR
    return SYNTH_SYSTEM


def _parse_validated_assessment(data: dict) -> dict:
    """Best-effort parse of an LLM-returned validated_assessment block.

    Returns an empty dict if the LLM did not include one. The schema is
    loose by design (free-form per-section structure), so we don't try
    to enforce the full ValidatedAssessment model here — we just want
    to ensure ``sections`` is a list and the density dict exists.
    """
    va = data.get("validated_assessment") or {}
    if not isinstance(va, dict):
        return {}
    sections = va.get("sections") or []
    if not isinstance(sections, list):
        sections = []
    density = va.get("knowledge_density") or {}
    if not isinstance(density, dict):
        density = {}
    return {
        "sections": sections,
        "overall_relevancy": va.get("overall_relevancy", "medium"),
        "knowledge_density": density,
        "reviewer_unsupported": va.get("reviewer_unsupported", []) or [],
        "reviewer_fabrication_flags": va.get("reviewer_fabrication_flags", []) or [],
    }


def _parse_lecture_appendix(data: dict) -> dict:
    """Parse an LLM-returned lecture_appendix block (methodology + density)."""
    ap = data.get("lecture_appendix") or {}
    if not isinstance(ap, dict):
        return {}
    return {
        "methodology": ap.get("methodology", ""),
        "tool_calls": ap.get("tool_calls", []) or [],
        "density_metrics": ap.get("density_metrics", {}) or {},
    }


def synthesizer_node(state: ArgusState) -> dict:
    """T7 — mode-aware synthesizer.

    Behaviour by ``state["length"]`` (default = "short"):

    - ``tldr``     — single short paragraph, no findings list.
    - ``short``    — legacy T6 behaviour: TL;DR + numbered findings +
                     Sources.
    - ``medium``   — TL;DR + Background + Current state + Open
                     questions + Sources; ~8 findings.
    - ``long``     — sectioned report with ## Background / ##
                     Current state (3-5 sub-sections) / ## Open
                     problems / ## Sources / ## Quality
                     assessment; ~12 findings; validated_assessment
                     block on state.
    - ``lecture``  — Parts I-IV + References + Appendix; ~15
                     findings; validated_assessment +
                     lecture_appendix on state.

    The LLM call's ``max_tokens`` and ``temperature`` are read from
    ``synthesis_modes.SYNTHESIS_MODES`` so this node does not own
    those knobs.
    """
    length = state.get("length") or "short"
    if not is_valid_length(length):
        length = "short"
    mode = get_mode(length)

    chat = llm.chat_for_tier(
        "strong", temperature=mode.temperature, max_tokens=mode.max_tokens,
    )
    fetched = [FetchedItem.model_validate(f) for f in state.get("fetched") or []]
    # Build evidence corpus (cap length).
    ev_lines: list[str] = []
    for i, f in enumerate(fetched, start=1):
        ev_lines.append(
            f"[{i}] {f.title or '(no title)'} -- {f.url}\n"
            f"    excerpt: {(f.excerpt or '')[:400]}"
        )
    evidence = "\n".join(ev_lines) or "(no evidence fetched)"
    revision_notes = state.get("revision_notes") or []
    revision_block = ""
    if revision_notes:
        revision_block = (
            "\n\nPRIOR REVIEWER NOTES (must address):\n"
            + "\n".join(f"- {n}" for n in revision_notes)
        )

    target_phrase = (
        f"Target length: {mode.label} ({mode.target_chars_min}-"
        f"{mode.target_chars_max} chars, ~{mode.target_findings} findings, "
        f"template={mode.template})."
    )

    # P5 — grounding gate. If the evidence does not actually mention the
    # queried entity, append a strict-conservatism warning to the
    # system prompt so the synthesizer doesn't bridge gaps with
    # fabrication. Pure function, no LLM call.
    from .grounding import (
        check_grounding as _check_grounding,
        grounding_warning_prompt as _grounding_warning_prompt,
    )
    _grounding = _check_grounding(state)
    _grounding_warn = _grounding_warning_prompt(
        _grounding, state.get("user_request") or "",
    )

    user = (
        f"Topic: {state['user_request']}\n"
        f"Mode: {length}\n"
        f"{target_phrase}\n\n"
        f"Evidence ({len(fetched)} items):\n{evidence}\n"
        f"{revision_block}\n\n"
        "Write the JSON response now."
    )
    resp = chat.invoke([
        SystemMessage(content=_synth_prompt_for(length) + _grounding_warn),
        HumanMessage(content=user[:14000]),
    ])
    rec = llm.record_from_response("strong", llm.resolve_tier("strong"), resp)
    findings: list[Finding] = []
    draft_md = ""
    validated: dict = {}
    lecture_appendix: dict = {}
    # T7-emptyLong-fix (2026-07-09): severity classification for the
    # synthesis outcome. "ok" = normal success. "no_evidence" = LLM
    # was honest, evidence base too thin to ground any finding
    # (acceptable - narrow scope on retry is the user's move).
    # "empty_fallback" = LLM violated the no_evidence contract by
    # returning empty findings[] WITHOUT the no_evidence flag (bug,
    # loud failure block). "synthesis_error" = parse / LLM call
    # failed (degraded gracefully - bot stays alive, user sees a
    # retryable error block).
    synthesis_outcome: str = "ok"
    synthesis_error: str = ""
    synthesis_no_evidence_reason: str = ""
    try:
        data = _parse_json_obj(resp.content)
        if data.get("no_evidence") is True:
            synthesis_outcome = "no_evidence"
            synthesis_no_evidence_reason = (
                data.get("no_evidence_reason")
                or "Evidence base too thin to support any grounded finding."
            )
            draft_md = _no_evidence_failure_md(
                state["user_request"], length=length,
                fetched_count=len(fetched),
                reason=synthesis_no_evidence_reason,
            )
            logger.info(
                "synthesizer: no_evidence (length=%s, fetched=%d, reason=%s)",
                length, len(fetched), synthesis_no_evidence_reason,
            )
        else:
            for i, fd in enumerate(data.get("findings") or []):
                try:
                    f = Finding.model_validate(fd)
                except Exception as e:
                    logger.warning("bad finding: %s", e)
                    continue
                # Assign a stable per-run id so post-synthesis passes
                # (quarantine, fabrication detector, grounding verifier)
                # can route findings by id instead of substring-matching
                # prose. The synthesizer is not required to produce an
                # id — we assign one here in source order.
                if not f.id:
                    f.id = f"f{i}"
                findings.append(f)
            if (not findings) and (not (data.get("draft_md") or "").strip()):
                synthesis_outcome = "empty_fallback"
                synthesis_no_evidence_reason = (
                    "Synthesizer returned empty findings[] and empty draft_md "
                    "without the required no_evidence=true flag."
                )
                draft_md = _no_evidence_failure_md(
                    state["user_request"], length=length,
                    fetched_count=len(fetched),
                    reason=synthesis_no_evidence_reason,
                )
                logger.warning(
                    "synthesizer: empty findings[] with no no_evidence flag "
                    "(length=%s, fetched=%d) - emitting loud failure block",
                    length, len(fetched),
                )
            else:
                draft_md = data.get("draft_md") or _draft_md_from_findings(
                    state["user_request"], findings, length=length)
            if mode.include_validated_assessment:
                validated = _parse_validated_assessment(data)
            if mode.include_appendix:
                lecture_appendix = _parse_lecture_appendix(data)
    except Exception as e:
        synthesis_outcome = "synthesis_error"
        synthesis_error = str(e)
        logger.warning("Synthesizer returned non-JSON / failed parse: %s", e)
        draft_md = (
            f"# {state['user_request']}\n\n"
            "⚠️ **Synthesis failed.**\n\n"
            "- **Class:** `synthesis_error`\n"
            f"- **Reason:** {synthesis_error[:500]}\n\n"
            "The Argus pipeline could not produce a structured JSON response "
            "on this attempt. Try `/research <topic>` again; if the failure "
            "persists, narrow the scope.\n"
        )

    out: dict[str, Any] = {
        "findings": [f.model_dump() for f in findings],
        "draft_md": draft_md,
        "synthesis_outcome": synthesis_outcome,
        "model_calls": [rec.model_dump()],
        "messages": [{"role": "assistant",
                      "content": (
                          f"🧠 Synthesized {len(findings)} findings ({mode.label})."
                          if synthesis_outcome == "ok" else
                          f"⚠️ Synthesis outcome: {synthesis_outcome} ({mode.label})."
                      )}],
    }
    if synthesis_error:
        out["synthesis_error"] = synthesis_error
    if synthesis_no_evidence_reason:
        out["synthesis_no_evidence_reason"] = synthesis_no_evidence_reason
    if validated:
        out["validated_assessment"] = validated
    if lecture_appendix:
        out["lecture_appendix"] = lecture_appendix
    return out


def _no_evidence_failure_md(topic: str, *, length: str,
                            fetched_count: int, reason: str) -> str:
    """Loud, classified failure block (T7-emptyLong-fix, 2026-07-09).

    Replaces the previous quiet 1.4 KB skeleton that carried the
    literal sentinels `Sectioned report (Long). 0 findings.` and
    `Fallback document -- open problems not enumerated.`. Used when
    the synthesizer was honest about a thin evidence base
    (`no_evidence=true`) or when it silently violated the contract
    (empty_fallback).

    The class, mode, fetched count, and reason are surfaced in the
    block so a kanban reviewer or the user can immediately diagnose
    why the report is short - instead of getting a markdown that
    looks like a successful run with empty sections.
    """
    from .synthesis_modes import get_mode
    mode = get_mode(length)
    return (
        f"# {topic}\n\n"
        "⚠️ **Synthesis could not produce findings.**\n\n"
        "- **Class:** `synthesis_no_evidence`\n"
        f"- **Mode:** {mode.label}\n"
        f"- **Fetched evidence:** {fetched_count} item(s)\n"
        f"- **Reason:** {reason or '(no reason given)'}\n\n"
        "This is a classified synthesis failure (not a successful report). "
        "The pipeline could not derive at least one grounded, cited "
        "finding from the fetched evidence.\n\n"
        "**Try:**\n"
        "1. `/research <narrower scope>` - fewer sub-topics, more concrete entity.\n"
        "2. Wait a minute and retry - transient OpenRouter 5xx and malformed\n"
        "   JSON are auto-recovered on the next run.\n"
        "3. If this keeps happening for the same topic, the entity may\n"
        "   genuinely have thin public coverage; consult a human-curated\n"
        "   source instead.\n"
    )


def _draft_md_from_findings(topic: str, findings: list[Finding],
                            *, length: str = "short") -> str:
    """Fallback markdown builder when the LLM failed to return one.

    Honors the length mode so a fallback document still respects the
    user's HITL choice (lecture fallback has Parts I-IV; tldr fallback
    is a single paragraph). The synthesizer normally returns its own
    draft_md; this only runs when JSON parsing failed.
    """
    from .synthesis_modes import get_mode
    mode = get_mode(length)
    urls: list[str] = []
    for f in findings:
        urls.extend(f.citation_urls)
    unique_urls = list(dict.fromkeys(urls))
    if length == "tldr":
        # Single paragraph: just join the findings as one bullet run.
        body = " ".join(f.claim for f in findings) or "(no findings)"
        return f"# {topic}\n\n{body}\n"
    if mode.template == "lecture":
        parts = [f"# {topic}: A Research Report", "",
                 f"_Mode: lecture • findings: {len(findings)} • "
                 f"sources: {len(unique_urls)}_", "", "## Executive TL;DR", "",
                 "Lecture-format fallback. The synthesizer did not return a "
                 "structured draft; the findings below are presented as a "
                 "Parts I-III walkthrough.", "", "## Part I -- Background", ""]
        for f in findings[: max(1, len(findings) // 4)]:
            cites = " ".join(f"[{_url_index(c, findings)}]" for c in f.citation_urls)
            parts.append(f"- **{f.claim}** {cites} _(confidence: {f.confidence})_")
        parts += ["", "## Part II -- Current state of the field", ""]
        mid_lo = max(1, len(findings) // 4)
        mid_hi = max(mid_lo + 1, 3 * len(findings) // 4)
        for f in findings[mid_lo:mid_hi]:
            cites = " ".join(f"[{_url_index(c, findings)}]" for c in f.citation_urls)
            parts.append(f"- **{f.claim}** {cites} _(confidence: {f.confidence})_")
        parts += ["", "## Part III -- Open problems", ""]
        for f in findings[mid_hi:]:
            cites = " ".join(f"[{_url_index(c, findings)}]" for c in f.citation_urls)
            parts.append(f"- **{f.claim}** {cites} _(confidence: {f.confidence})_")
        parts += ["", "## References", ""]
        for i, u in enumerate(unique_urls, 1):
            parts.append(f"{i}. {u}")
        parts += ["", "## Quality assessment", "",
                  "_Fallback document -- validated assessment unavailable._"]
        return "\n".join(parts)
    if mode.template == "sectioned":
        parts = [f"# {topic}", "", "## TL;DR", "",
                 f"Sectioned report ({mode.label}). {len(findings)} findings.",
                 "", "## Background", ""]
        bg_count = max(1, len(findings) // 4)
        for f in findings[:bg_count]:
            cites = " ".join(f"[{_url_index(c, findings)}]" for c in f.citation_urls)
            parts.append(f"- **{f.claim}** {cites} _(confidence: {f.confidence})_")
        parts += ["", "## Current state of the field", ""]
        for f in findings[bg_count:]:
            cites = " ".join(f"[{_url_index(c, findings)}]" for c in f.citation_urls)
            parts.append(f"- **{f.claim}** {cites} _(confidence: {f.confidence})_")
        parts += ["", "## Open problems", "",
                  "_Fallback document -- open problems not enumerated._",
                  "", "## Sources", ""]
        for i, u in enumerate(unique_urls, 1):
            parts.append(f"{i}. {u}")
        return "\n".join(parts)
    # default / flat
    parts = [f"# {topic}", "", "## TL;DR", "",
             "Report generated from the synthesized findings below.", "",
             "## Key Findings", ""]
    for i, f in enumerate(findings, 1):
        cites = " ".join(f"[{_url_index(c, findings)}]" for c in f.citation_urls)
        parts.append(f"{i}. **{f.claim}** {cites} _(confidence: {f.confidence})_")
    parts += ["", "## Sources", ""]
    for i, u in enumerate(unique_urls, 1):
        parts.append(f"{i}. {u}")
    return "\n".join(parts)


def _url_index(url: str, findings: list[Finding]) -> int:
    """Return a stable 1-based index for a URL across the findings list."""
    urls = []
    for f in findings:
        for c in f.citation_urls:
            if c not in urls:
                urls.append(c)
    return urls.index(url) + 1 if url in urls else 0


# ---------------------------------------------------------------------------
# reviewer
# ---------------------------------------------------------------------------

REVIEW_SYSTEM = """You are Argus's adversarial reviewer.

You will see:
- The user's original question
- The list of findings (each with claim + citation URLs + confidence)
- The full draft markdown

Your job:
- Every factual claim in the draft MUST trace to a citation URL.
- Flag any claim that sounds fabricated, has no citation, or whose
  citation URL is not present in the fetched evidence list.
- Flag any sentence that introduces a number, date, or specific name
  not supported by the evidence.
- Verdict MUST be "pass" or "revise".
- If you say "revise", list specific actionable notes the synthesizer
  must follow.

You are from a DIFFERENT model family than the synthesizer. Do not
defer to the synthesizer's confidence labels — verify them yourself.

Return JSON only:
{
  "verdict": "pass" | "revise",
  "notes": ["...", "..."],
  "unsupported_claims": ["the exact claim text...", "..."],
  "fabrication_flags": ["the exact suspicious text...", "..."],
  "flagged_finding_ids": ["f0", "f3", "..."]
}

CRITICAL: each finding in the input has an `id` field (e.g. "f0",
"f3"). When you flag a claim as unsupported or fabricated, list
the corresponding `id` in `flagged_finding_ids`. This is the
routing key the report builder uses to move the finding to the
## flagged-claims appendix - matching by id is exact, matching
by claim text fragments the prose when the same claim recurs
across sections.
"""


def reviewer_node(state: ArgusState) -> dict:
    chat = llm.chat_for_tier("judge", temperature=0.1, max_tokens=1500)
    findings = state.get("findings") or []
    fetched = state.get("fetched") or []
    fetched_urls = {f["url"] for f in fetched}
    user = (
        f"Topic: {state['user_request']}\n\n"
        f"Findings JSON:\n{json.dumps(findings[:20], indent=2)[:6000]}\n\n"
        f"Draft markdown:\n```\n{(state.get('draft_md') or '')[:6000]}\n```\n\n"
        f"Allowed citation URLs ({len(fetched_urls)}):\n"
        + "\n".join(f"- {u}" for u in list(fetched_urls)[:30])
        + "\n\nReturn JSON verdict now."
    )
    resp = chat.invoke([
        SystemMessage(content=REVIEW_SYSTEM),
        HumanMessage(content=user[:14000]),
    ])
    rec = llm.record_from_response("judge", llm.resolve_tier("judge"), resp)
    flagged_ids: list[str] = []
    try:
        data = _parse_json_obj(resp.content)
        verdict = ReviewVerdict.model_validate({
            "verdict": data.get("verdict", "revise"),
            "notes": data.get("notes", []),
            "unsupported_claims": data.get("unsupported_claims", []),
            "fabrication_flags": data.get("fabrication_flags", []),
        })
        flagged_ids = list(data.get("flagged_finding_ids") or [])
    except Exception as e:
        logger.warning("Reviewer non-JSON, defaulting to revise: %s", e)
        verdict = ReviewVerdict(
            verdict="revise",
            notes=["Reviewer did not return structured JSON; please tighten the draft."],
        )
    notes = list(state.get("revision_notes") or [])
    notes.extend(verdict.notes)
    notes.extend(f"Unsupported claim: {c}" for c in verdict.unsupported_claims)
    notes.extend(f"Fabrication flag: {f}" for f in verdict.fabrication_flags)
    rounds = int(state.get("revision_rounds") or 0) + (
        1 if verdict.verdict == "revise" else 0
    )
    return {
        "review_verdict": {**verdict.model_dump(), "flagged_finding_ids": flagged_ids},
        "revision_notes": notes,
        "revision_rounds": rounds,
        "model_calls": [rec.model_dump()],
        "messages": [{"role": "assistant",
                      "content": f"🔬 Reviewer verdict: {verdict.verdict}"}],
    }


def route_after_review(state: ArgusState) -> str:
    v = (state.get("review_verdict") or {}).get("verdict")
    rounds = int(state.get("revision_rounds") or 0)
    max_rounds = int(os.environ.get("ARGUS_MAX_REVISIONS", "3"))
    if v == "pass" or rounds >= max_rounds:
        return "report_builder"
    return "synthesizer"


# ---------------------------------------------------------------------------
# report_builder
# ---------------------------------------------------------------------------

def report_builder_node(state: ArgusState) -> dict:
    """T7 — mode-aware report builder.

    Responsibilities:
    1. Pick the markdown template based on ``state["length"]``
       (delegated to ``_draft_md_from_findings`` if the synthesizer
       failed and returned no draft_md).
    2. Prepend a T7 title-page block (mode, date, quality summary,
       per-section confidence) so the user can see what they're getting
       before scrolling.
    3. Append a lecture appendix (methodology + tool calls + density)
       when ``state["length"] == "lecture"``.
    4. Merge reviewer-flagged unsupported_claims / fabrication_flags
       into the validated_assessment so the title page reflects the
       adversarial pass.
    5. Write ``metadata.json`` sidecar with the same fields + the full
       validated_assessment + lecture_appendix blobs.
    6. Render markdown -> PDF via ReportLab (fast, no browser) with
       fallback to intel-stack Chromium for the styled HTML route.
    7. Trigger the report-preview HITL gate.
    """
    from ..config import get_settings
    from .synthesis_modes import get_mode
    from .report_builder_helpers import (
        build_sidecar_metadata, merge_reviewer_into_assessment,
        now_iso, render_lecture_appendix, render_title_block,
        sidecar_to_json,
    )
    s = get_settings()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    length = state.get("length") or "short"
    mode = get_mode(length)
    topic = re.sub(r"[^A-Za-z0-9_-]+", "_", state["user_request"])[:50] or "report"
    folder = s.reports_root / f"{stamp}_{topic}_{length}"
    folder.mkdir(parents=True, exist_ok=True)
    md_path = folder / "report.md"
    pdf_path = folder / "report.pdf"

    # Merge reviewer verdict into the validated_assessment so the
    # title-page summary is honest about adversarial findings.
    validated = state.get("validated_assessment") or {}
    validated = merge_reviewer_into_assessment(
        validated, state.get("review_verdict"),
    )

    # Compose the final markdown: title block + body + (optional) appendix.
    n_findings = len(state.get("findings") or [])
    n_sources = len(state.get("fetched") or [])
    revision_rounds = int(state.get("revision_rounds") or 0)

    # P4 - quarantine reviewer-flagged fabrication claims BEFORE we
    # compose the body so the title-block flags report honest numbers.
    #
    # Two paths:
    #   (a) Structured (default, 2026-07-09): route findings by
    #       ``finding_id`` through ``post_synthesis.quarantine_by_id``.
    #       This is the structural fix for the §4.1 fragmentation
    #       bug - when the same flagged claim appears in multiple
    #       sections, the structured path routes it exactly once
    #       instead of partially matching substrings and producing
    #       orphaned prose fragments.
    #   (b) Legacy string-quarantine fallback: when findings lack
    #       ids (legacy checkpoint replay) or the structured path
    #       raises, fall back to the substring-matching legacy
    #       ``quarantine_flagged_claims``. Removed when the legacy
    #       module is retired (post 5.4 of the argus-async handoff).
    review_verdict = state.get("review_verdict") or {}
    body = state.get("draft_md") or "# (empty draft)"
    used_structured_quarantine = False
    if review_verdict:
        # The reviewer may flag findings by id (new structured
        # contract) OR by claim text (legacy substring contract). We
        # support both - preferring id-based when available.
        flagged_ids = review_verdict.get("flagged_finding_ids") or []
        flagged_claim_text = (
            list(review_verdict.get("unsupported_claims") or [])
            + list(review_verdict.get("fabrication_flags") or [])
        )
        try:
            from ..post_synthesis import (
                assign_finding_ids,
                quarantine_by_id as _quarantine_by_id,
                render_quarantine_appendix as _render_quarantine_appendix,
            )
            findings_records = assign_finding_ids(state.get("findings") or [])
            if findings_records and flagged_ids:
                kept, q, d, unmatched = _quarantine_by_id(
                    findings_records, flagged_ids, mode="move",
                )
                n_quarantined = len(q)
                n_still_unmatched = len(unmatched)
                # Append the structured appendix. The body itself is
                # left intact (the synthesizer already wrote it from
                # the surviving findings); the appendix is the only
                # place where flagged claims appear.
                if q:
                    body = body + _render_quarantine_appendix(q)
                used_structured_quarantine = True
                logger.info(
                    "quarantine(structured): moved %d finding(s) to "
                    "appendix (%d unmatched); kept %d",
                    n_quarantined, n_still_unmatched, len(kept),
                )
        except Exception as _e:
            logger.warning(
                "structured quarantine skipped (%s); falling back to "
                "legacy string-quarantine", _e,
            )

        if not used_structured_quarantine and flagged_claim_text:
            try:
                from ..quarantine import quarantine_flagged_claims as                     _quarantine_flagged_claims
                _qres = _quarantine_flagged_claims(
                    body,
                    review_verdict.get("unsupported_claims") or [],
                    review_verdict.get("fabrication_flags") or [],
                    mode="move",
                )
                n_quarantined = len(_qres.quarantined)
                n_still_unmatched = len(_qres.still_unmatched)
                body = _qres.cleaned_text
                if n_quarantined:
                    logger.info(
                        "quarantine(legacy): moved %d flagged "
                        "sentence(s) to appendix (%d unmatched)",
                        n_quarantined, n_still_unmatched,
                    )
            except Exception as _e:
                logger.warning(
                    "quarantine pass skipped (%s); body unchanged", _e,
                )


    title_block = render_title_block(
        topic=state["user_request"],
        length=length,
        length_label=mode.label,
        stamp=now_iso(),
        n_findings=n_findings,
        n_sources=n_sources,
        revision_rounds=revision_rounds,
        validated_assessment=validated,
    )
    appendix = ""
    if mode.include_appendix:
        appendix = render_lecture_appendix(
            state.get("lecture_appendix") or {},
            density_metrics={},
            total_chars=len(body),
        )

    # P5 — grounding banner prepended ABOVE the body so the reader sees
    # the warning BEFORE the asserted content. Empty string when
    # grounded - the helper handles that.
    try:
        from .grounding import (
            check_grounding as _check_grounding,
            grounding_banner as _grounding_banner,
        )
        _grounding_banner = _grounding_banner(_check_grounding(state))
    except Exception as _e:
        logger.warning("grounding banner skipped (%s); report unchanged", _e)
        _grounding_banner = ""

    md_text = (
        title_block
        + _grounding_banner
        + body
        + "\n\n---\n\n_Generated by Argus • "
        + now_iso()
        + f" • mode: {mode.label}_\n"
        + appendix
    )

    # Citation integrity pass (lifted from NVIDIA AI-Q Blueprint).
    # Every URL in the final report must resolve to a URL that was
    # actually fetched by a researcher tool. Anything else is stripped,
    # along with its inline citation, and recorded in the audit trail.
    # This is the structural fix for the 2026-07-08 hallucinated-URL bug.
    try:
        from ..citations import (
            SourceRegistry as _CitationRegistry,
            verify_citations as _verify_citations,
            sanitize_report as _sanitize_report,
        )
        from ..sources_block import (
            sanitize_sources_block as _sanitize_sources_block,
        )
        _reg = _CitationRegistry()
        _reg.add_from_fetched(state.get("fetched") or [])
        _verified = _verify_citations(md_text, _reg)
        _sanitized = _sanitize_report(_verified.verified_report)
        md_text = _sanitized.cleaned_text
        # P3 — drop dangling [N] source slots left behind by
        # verify_citations (which only strips the URL, not the label).
        # Also renumbers body refs so nothing points at an empty slot.
        try:
            _src_block = _sanitize_sources_block(md_text)
            if _src_block.dropped_count:
                md_text = _src_block.cleaned_text
                logger.info(
                    "sources block: dropped %d dangling slot(s), "
                    "renumbered %d ref(s)",
                    _src_block.dropped_count,
                    len(_src_block.renumbered),
                )
        except Exception as _e:
            logger.warning(
                "sources-block post-pass skipped (%s); report unchanged",
                _e,
            )
        if _verified.removed_citations or _sanitized.removed:
            logger.info(
                "citation integrity: stripped %d unregistered URL(s), %d sanitized URL(s)",
                len(_verified.removed_citations),
                len(_sanitized.removed),
            )
    except Exception as _e:
        # Citation module failure must NOT block report delivery; degrade
        # gracefully and log. The user's bot should always produce a report.
        logger.warning("citation integrity pass skipped (%s); report unchanged", _e)

    md_path.write_text(md_text, encoding="utf-8")

    pdf_ok = False
    try:
        # markdown_to_pdf uses playwright via the intel-stack helper. That
        # helper imports playwright at runtime, so if it isn't in the
        # argus venv we delegate to the intel-stack venv python.
        try:
            markdown_to_pdf(md_text, str(pdf_path), title=topic)
            pdf_ok = pdf_path.exists()
        except (ModuleNotFoundError, ImportError) as e:
            # Fallback: render via intel-stack's python + the _common helper.
            logger.info("argus venv missing playwright; delegating PDF render")
            import importlib.util
            import subprocess
            helper_path = (Path(os.environ.get(
                "INTEL_STACK_DIR",
                r"A:\Hermes\Agents\intel-stack\scripts")) / "_common.py")
            helper_path.parent.mkdir(parents=True, exist_ok=True)
            # We need a small wrapper: write the md to disk, call _common.
            tmp_py = folder / "_render_pdf.py"
            tmp_py.write_text(
                "import sys, os\n"
                f"sys.path.insert(0, r'{helper_path.parent}')\n"
                "os.environ.pop('PYTHONPATH', None)\n"
                f"from _common import markdown_to_pdf\n"
                f"markdown_to_pdf(open(r'{md_path}','r',encoding='utf-8').read(),\n"
                f"                  r'{pdf_path}', title=r'{topic}')\n",
                encoding="utf-8",
            )
            intel_py = os.environ.get(
                "INTEL_PYTHON",
                r"A:\Hermes\Agents\intel-stack\venv\Scripts\python.exe",
            )
            res = subprocess.run(
                [intel_py, str(tmp_py)],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "PYTHONPATH": ""},
            )
            if res.returncode != 0:
                logger.warning("PDF fallback failed: %s", res.stderr[-500:])
            pdf_ok = pdf_path.exists()
    except Exception as e:
        logger.warning("PDF render failed: %s", e)

    meta = build_sidecar_metadata(
        topic=state["user_request"],
        thread_id=state.get("thread_id"),
        user_id=state.get("user_id"),
        length=length,
        length_label=mode.label,
        stamp=stamp,
        n_findings=n_findings,
        n_sources=n_sources,
        revision_rounds=revision_rounds,
        validated_assessment=validated,
        lecture_appendix=state.get("lecture_appendix"),
        model_calls=state.get("model_calls") or [],
    )
    (folder / "metadata.json").write_text(
        sidecar_to_json(meta), encoding="utf-8",
    )
    return {
        "report_paths": {
            "folder": str(folder),
            "md": str(md_path),
            "pdf": str(pdf_path) if pdf_ok else None,
        },
        "hitl": {"pending": True, "kind": "report_preview",
                 "ctx": {"paths": {
                     "folder": str(folder),
                     "md": str(md_path),
                     "pdf": str(pdf_path) if pdf_ok else None,
                 }}},
        "messages": [{"role": "assistant",
                      "content": f"📝 Report ready ({mode.label}). "
                                 "Awaiting your sign-off."}],
    }


# ---------------------------------------------------------------------------
# deliver + Phase 2 HITL "extend research" loop
# ---------------------------------------------------------------------------

MAX_EXTEND_ROUNDS = 2


def _will_extend(state: ArgusState) -> bool:
    """True when the user asked to extend and we're under the round cap."""
    return bool(state.get("extend_requested")) and \
        int(state.get("extend_rounds") or 0) < MAX_EXTEND_ROUNDS


def deliver_node(state: ArgusState) -> dict:
    """Records the paths; the Telegram bot layer does the actual send.

    If the user asked to extend research at the preview gate (and we're under
    the round cap), this is a no-op pass-through — ``route_after_deliver``
    sends the run back through ``extend_prep`` instead of finalising.
    """
    if _will_extend(state):
        return {"messages": [{"role": "assistant",
                              "content": "↪️ Extending research…"}]}
    paths = state.get("report_paths") or {}
    return {
        "hitl": {"pending": False},
        "messages": [{"role": "assistant", "content": (
            f"✅ Delivered. Folder: {paths.get('folder')}")}],
    }


def route_after_deliver(state: ArgusState) -> str:
    """Loop back to deepen research, or end the run."""
    return "extend" if _will_extend(state) else "end"


def extend_prep_node(state: ArgusState) -> dict:
    """Phase 2 — broaden the plan and gather MORE sources for another pass.

    Runs the researcher subgraph itself (rather than routing back to the
    ``researcher`` node, which is gated by ``interrupt_before`` and would
    re-trigger the plan-approval pause). New sources are merged with the
    existing ones by the subgraph's pre-seed step; the run then flows on to
    ``fetcher`` → … → ``report_builder`` for a fresh preview.

    To surface genuinely new material (not the same URLs), we fold salient
    words from the plan's sub_questions into ``must_have_keywords`` so the
    subgraph's web/GitHub searches broaden their coverage.
    """
    plan = ResearchPlan.model_validate(state.get("plan") or {})
    extra: list[str] = []
    for sq in plan.sub_questions:
        extra += [w for w in re.split(r"[^A-Za-z0-9]+", sq) if len(w) > 4]
    plan.must_have_keywords = list(
        dict.fromkeys(list(plan.must_have_keywords) + extra))[:8]

    seeded = dict(state)
    seeded["plan"] = plan.model_dump()
    research = run_researcher_subgraph(seeded)

    rnd = int(state.get("extend_rounds") or 0) + 1
    out: dict[str, Any] = {
        "plan": plan.model_dump(),
        "sources": research.get("sources", []),
        "extend_requested": False,
        "extend_rounds": rnd,
        # Fresh review budget for the re-synthesis of the widened evidence.
        "revision_rounds": 0,
        "messages": [{"role": "assistant",
                      "content": f"🔎 Extending research (round {rnd}) — "
                                 f"{len(research.get('sources') or [])} sources now."}],
    }
    if research.get("errors"):
        out["errors"] = research["errors"]
    return out


# ---------------------------------------------------------------------------
# quick_answer
# ---------------------------------------------------------------------------

def quick_answer_node(state: ArgusState) -> dict:
    """Cheap single-shot answer. Still routes through FreeLLMAPI, so it
    is grounded in the model's training (no fabrication guarantees)."""
    chat = llm.chat_for_tier("cheap", temperature=0.3, max_tokens=600)
    resp = chat.invoke([
        SystemMessage(content=(
            "You are Argus, a research assistant. Answer the user's question "
            "concisely and accurately. If you don't know, say so. No fake citations."
        )),
        HumanMessage(content=state["user_request"][:4000]),
    ])
    rec = llm.record_from_response("cheap", llm.resolve_tier("cheap"), resp)
    return {
        "quick_answer": resp.content,
        "model_calls": [rec.model_dump()],
        "messages": [{"role": "assistant", "content": resp.content}],
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_json_obj(text: str, *, min_keys: int = 1) -> dict:
    """Best-effort JSON parse; tolerates ```json fences and embedded JSON.

    Strategy:
      1. Strip outer fences and try direct parse.
      2. Balanced-brace match — find the outermost valid JSON object.
      3. Fall back to a ```json fenced block (the model may have
         embedded its real JSON inside a markdown code block).

    ``min_keys`` rejects trivially-empty objects like ``{}`` so callers
    can force the fallback path to fire when the LLM produced no useful
    structured output.
    """
    raw = text or ""

    def _accept(d: dict) -> dict | None:
        if not isinstance(d, dict):
            return None
        if len(d) < min_keys:
            return None
        return d

    # 1. Direct (after fence stripping).
    t = raw.strip()
    t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        d = _accept(json.loads(t))
        if d is not None:
            return d
    except Exception:
        pass
    # 2. Balanced braces — find the first '{' whose matching '}' yields
    #    valid JSON.
    s = t
    for i, ch in enumerate(s):
        if ch == "{":
            depth = 0
            for j in range(i, len(s)):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = s[i:j + 1]
                        try:
                            d = _accept(json.loads(candidate))
                            if d is not None:
                                return d
                        except Exception:
                            break
    # 3. Try json-fenced block (the model sometimes dumps JSON inside
    #    a code fence that itself is in a string field).
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL |
                  re.IGNORECASE)
    if m:
        try:
            d = _accept(json.loads(m.group(1)))
            if d is not None:
                return d
        except Exception:
            pass
    raise ValueError(
        f"no useful JSON object found in: {raw[:200]!r} "
        f"(required >= {min_keys} keys)"
    )


def _ts() -> str:
    return datetime.now().astimezone().isoformat()