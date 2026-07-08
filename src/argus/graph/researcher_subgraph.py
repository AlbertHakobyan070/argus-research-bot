"""Argus — researcher subgraph (handoff action #1, T8).

Architecture
------------
The legacy ``researcher_node`` was a single node that fanned-out
internally to harvest + arxiv + planner URLs, and (T6) fell back to a
raw arxiv query when the topic was orphan / niche. This worked for
medium-difficulty topics but the failure mode — one bad planner step
poisoning all sources — was a single point of failure.

The new design is a LangGraph **subgraph** with three sub-researchers
running in parallel, each scoped to one source kind:

```
                    ┌─ arxiv_sub  ──┐
researcher_supervisor ─┤─ github_sub ──┼─► merge_research → researcher_out
                    └─ web_sub    ──┘
```

Each sub-researcher receives the same ``ResearchPlan`` slice it's
responsible for, talks to its own tool set (arxiv API, GitHub search,
DDGS web search), and emits a ``SubResearchResult`` (URL list + the
sub-researcher's identifier + the sub-question it answered). The
supervisor node only assembles the plan-slice routing — it does NOT
make LLM calls itself (the cheap research cost model wants
specialization over generalization).

The merge_research node then:
- dedupes by URL,
- tags each source with ``sub_kind`` so downstream nodes can see where
  it came from,
- caps at 18 (legacy cap),
- records errors per sub (so a github_sub 0-results doesn't kill the
  arxiv + web outputs).

State and integration
---------------------
The subgraph accepts the parent ArgusState's ``plan`` field via the
``subgraph_input`` TypedDict. The wrapper function
``run_researcher_subgraph(state, ...)`` extracts the plan, runs the
subgraph, and returns a state diff suitable for merging back into the
parent graph (the same dict shape that legacy ``researcher_node``
returned: ``{"sources": [...], "errors": [...], "messages": [...]}``).

The parent graph's ``researcher_node`` is preserved as a thin shim
that calls ``run_researcher_subgraph`` so existing tests + interrupt
wiring keep working. The shim is the ONLY change to nodes.py outside
of this file.

Why a subgraph and not three separate nodes
-------------------------------------------
The handoff §2 explains the cost/quality trade-off. Three reasons
that matter specifically for argus:

1. **Failure isolation.** A network error in DDGS does not poison the
   arxiv + github results. The merge node sees ``errors=["ddgs: ..."]``
   but still ships arxiv + github.
2. **Parallelism.** The three subs run in one async gather, not
   serially in one node function. Wall-clock for the researcher phase
   drops from O(sum) to O(max).
3. **Composition.** Each sub is a pure function of (plan_slice, tool).
   Adding a "huggingface_sub" is one new node + one new entry in the
   supervisor's routing table — no surgery on the legacy
   ``researcher_node``.

Tests
-----
``tests/test_researcher_subgraph.py`` mocks all HTTP, verifies:
- arxiv_sub returns mocked arxiv entries;
- github_sub returns mocked github search entries;
- web_sub returns mocked DDGS entries;
- merge_research dedupes by URL across subs;
- merge_research caps at 18;
- supervisor routes planned_source.kind="paper" → arxiv_sub,
  "repo" → github_sub, "blog/news/official_doc/search_result" → web_sub;
- any sub raising an exception is logged and does not poison siblings.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any, Literal, TypedDict

import operator
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .state import ResearchPlan, PlannedSource

logger = logging.getLogger("argus.subgraph")


# ---------------------------------------------------------------------------
# Subgraph state — small TypedDict; only the fields the subs actually need.
# ---------------------------------------------------------------------------

SubKind = Literal["arxiv", "github", "web"]


class SubResearchResult(TypedDict, total=False):
    """One sub-researcher's output."""
    sub_kind: SubKind
    sources: list[dict]
    error: str  # empty if success


class ResearcherSubgraphState(TypedDict, total=False):
    # Inputs
    user_request: str
    plan: dict | None  # ResearchPlan.model_dump()
    pre_seeded_sources: list[dict]  # state["sources"] passed in
    # Outputs (append-only so each sub just returns its own slice)
    sub_results: Annotated[list[SubResearchResult], operator.add]
    final_sources: list[dict]
    errors: list[str]               # collected by merge_research from sub_results


# ---------------------------------------------------------------------------
# Routing: which planned_source.kind goes to which sub?
# ---------------------------------------------------------------------------

_KIND_TO_SUB: dict[str, SubKind] = {
    "paper": "arxiv",
    "repo": "github",
    "blog": "web",
    "news": "web",
    "official_doc": "web",
    "search_result": "web",
}


def _route_planned_source(ps: PlannedSource) -> SubKind:
    return _KIND_TO_SUB.get(ps.kind, "web")


# ---------------------------------------------------------------------------
# Sub-researchers — pure functions of (plan_slice, user_request) → sources.
# ---------------------------------------------------------------------------

async def arxiv_sub(state: ResearcherSubgraphState) -> dict:
    """Sub-researcher for arxiv papers.

    Uses ``_arxiv_search`` from nodes.py (re-imported lazily so this
    module stays import-clean for tests). Falls back to raw-query arxiv
    pass if the plan has no keywords (orphan-topic defense from T6).
    """
    from .nodes import _arxiv_search, _arxiv_search_raw  # local import
    plan_dict = state.get("plan") or {}
    try:
        plan = ResearchPlan.model_validate(plan_dict)
    except Exception:
        plan = ResearchPlan()
    sources: list[dict] = []
    error = ""
    items: list[dict] = []
    try:
        try:
            items = _arxiv_search(plan)
        except (IndexError, KeyError) as e:
            # Empty plan (no keywords, no sub_questions) → let the
            # user_request fallback path produce sources instead.
            logger.info("arxiv_sub: structured search skipped (%r)", e)
            items = []
        sources.extend(items)
        if not sources and state.get("user_request"):
            items = _arxiv_search_raw(state["user_request"])
            sources.extend(items)
    except Exception as e:
        error = f"arxiv_sub failed: {e!r}"
        logger.warning(error)
    return {"sub_results": [{"sub_kind": "arxiv",
                              "sources": sources,
                              "error": error}]}


async def github_sub(state: ResearcherSubgraphState) -> dict:
    """Sub-researcher for GitHub repos + READMEs.

    Hits GitHub's public search API (no key, 60 req/h unauthenticated).
    Falls back to a duckduckgo site-filter search if the rate limit
    hits — gives at least a URL hint for the fetcher to follow.
    """
    import httpx
    plan_dict = state.get("plan") or {}
    keywords = (plan_dict.get("must_have_keywords") or [])[:3]
    sub_q = (plan_dict.get("sub_questions") or [])
    if not keywords and sub_q:
        keywords = sub_q[0].split()[:3]
    query = " ".join(keywords) or state.get("user_request", "")
    sources: list[dict] = []
    error = ""
    if not query.strip():
        return {"sub_results": [{"sub_kind": "github",
                                  "sources": sources,
                                  "error": error}]}
    try:
        params = {"q": query, "sort": "stars",
                  "order": "desc", "per_page": 6}
        async with httpx.AsyncClient(timeout=12.0,
                                      follow_redirects=True) as c:
            r = await c.get("https://api.github.com/search/repositories",
                             params=params,
                             headers={"Accept": "application/vnd.github+json"})
            if r.status_code == 200:
                for it in r.json().get("items", [])[:6]:
                    sources.append({
                        "kind": "repo",
                        "title": it.get("full_name") or it.get("name", ""),
                        "url": it.get("html_url", ""),
                        "summary": (it.get("description") or "")[:400],
                        "source": "github-search",
                        "sub_kind": "github",
                    })
            elif r.status_code == 403:
                error = "github_sub: rate-limited (403)"
                logger.warning(error)
            else:
                error = f"github_sub: HTTP {r.status_code}"
    except Exception as e:
        error = f"github_sub failed: {e!r}"
        logger.warning(error)
    return {"sub_results": [{"sub_kind": "github",
                              "sources": sources,
                              "error": error}]}


async def web_sub(state: ResearcherSubgraphState) -> dict:
    """Sub-researcher for blogs, news, official docs.

    Uses ``ddgs_search`` if available (handoff action #4), else falls
    back to plain httpx + DDG HTML scrape (kept for hermetic tests).
    Either way, we never call LLM here — research cost stays bounded.
    """
    plan_dict = state.get("plan") or {}
    keywords = (plan_dict.get("must_have_keywords") or [])[:3]
    sub_q = (plan_dict.get("sub_questions") or [])
    if not keywords and sub_q:
        keywords = sub_q[0].split()[:3]
    query = " ".join(keywords) or state.get("user_request", "")
    sources: list[dict] = []
    error = ""
    if not query.strip():
        return {"sub_results": [{"sub_kind": "web",
                                  "sources": sources,
                                  "error": error}]}
    try:
        # Lazy import so tests can monkeypatch the function without
        # paying the duckduckgo-search import cost at module load.
        try:
            from ..tools import ddgs_search  # type: ignore
            results = ddgs_search(query, max_results=8)
            for r in results:
                sources.append({
                    "kind": r.get("kind", "blog"),
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "summary": r.get("snippet", "")[:400],
                    "source": r.get("source", "ddgs"),
                    "sub_kind": "web",
                })
        except ImportError:
            # Graceful degrade: ddgs not installed yet — return empty.
            error = "web_sub: ddgs_search not available (action #4 pending)"
            logger.info(error)
    except Exception as e:
        error = f"web_sub failed: {e!r}"
        logger.warning(error)
    return {"sub_results": [{"sub_kind": "web",
                              "sources": sources,
                              "error": error}]}


# ---------------------------------------------------------------------------
# Supervisor + merge
# ---------------------------------------------------------------------------

def supervisor_node(state: ResearcherSubgraphState) -> dict:
    """No-op — the parallel dispatch happens at the graph level.

    We keep this node as the explicit entrypoint so the routing intent
    is visible in graph diagrams and so a future LLM-driven
    plan-rebalancer has somewhere to live.
    """
    return {}


def merge_research(state: ResearcherSubgraphState) -> dict:
    """Dedup sub-results by URL, tag with sub_kind, cap at 18.

    Pre-seeded sources (from state["sources"]) come first so demos /
    deterministic tests retain priority, then union the three sub
    outputs in deterministic order (arxiv → github → web).
    """
    seen: set[str] = set()
    final: list[dict] = []
    errors: list[str] = []

    # 1. Pre-seeded sources (demos, manual_seed).
    for s in state.get("pre_seeded_sources") or []:
        url = s.get("url", "")
        if url and url not in seen:
            final.append(dict(s))
            seen.add(url)

    # 2. Sub results in deterministic order.
    for kind in ("arxiv", "github", "web"):
        for result in state.get("sub_results") or []:
            if result.get("sub_kind") != kind:
                continue
            if result.get("error"):
                errors.append(result["error"])
            for s in result.get("sources") or []:
                url = s.get("url", "")
                if url and url not in seen:
                    final.append(dict(s))
                    seen.add(url)

    # 3. Cap at 18 (legacy cap, matches researcher_node).
    final = final[:18]
    return {"final_sources": final, "errors": errors}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Build the subgraph
# ---------------------------------------------------------------------------

def build_researcher_subgraph(
    *, checkpointer=None,
) -> CompiledStateGraph:
    """Construct the 3-way parallel researcher subgraph.

    Returns a CompiledStateGraph that runs:
    supervisor → (arxiv_sub | github_sub | web_sub) → merge_research.
    """
    g = StateGraph(ResearcherSubgraphState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("arxiv_sub", arxiv_sub)
    g.add_node("github_sub", github_sub)
    g.add_node("web_sub", web_sub)
    g.add_node("merge_research", merge_research)

    g.add_edge(START, "supervisor")
    # Parallel fan-out: all three subs read supervisor's state and
    # write to sub_results independently.
    g.add_edge("supervisor", "arxiv_sub")
    g.add_edge("supervisor", "github_sub")
    g.add_edge("supervisor", "web_sub")
    # Fan-in: all three subs must complete before merge runs.
    g.add_edge("arxiv_sub", "merge_research")
    g.add_edge("github_sub", "merge_research")
    g.add_edge("web_sub", "merge_research")
    g.add_edge("merge_research", END)
    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Public API — the legacy ``researcher_node`` calls this.
# ---------------------------------------------------------------------------

def run_researcher_subgraph(
    state: dict[str, Any],
    *,
    subgraph: CompiledStateGraph | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Drive the subgraph from a parent ArgusState slice and return a diff.

    Returns the same shape legacy ``researcher_node`` returned:
    ``{"sources": [...], "errors": [...], "messages": [...]}``.
    """
    sg = subgraph or build_researcher_subgraph()
    sub_state: ResearcherSubgraphState = {
        "user_request": state.get("user_request", "") or "",
        "plan": state.get("plan") or {},
        "pre_seeded_sources": list(state.get("sources") or []),
        "sub_results": [],
        "final_sources": [],
    }
    cfg = config or {"configurable": {"thread_id":
                                         state.get("thread_id", "default")}}
    # Use ainvoke so LangGraph schedules the async sub-researchers in
    # parallel; the sync invoke path refuses async nodes.
    #
    # This function is called from the sync ``researcher_node``. Under the
    # bot's async ``graph.astream``, LangGraph runs sync nodes on a worker
    # thread with no running loop, so ``asyncio.run`` works directly. If we
    # are ever called ON a thread that already has a running loop,
    # ``asyncio.run`` raises RuntimeError — we then run the subgraph in a
    # fresh loop on a dedicated worker thread. (The old fallback scheduled
    # onto ``get_event_loop()`` and blocked on ``.result()``, which
    # deadlocks when that loop is the caller's own.)
    def _run_fresh_loop() -> dict:
        return asyncio.run(sg.ainvoke(sub_state, config=cfg))

    try:
        out = _run_fresh_loop()
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            out = ex.submit(_run_fresh_loop).result()
    final_sources = out.get("final_sources", []) or []
    errs = out.get("errors", []) or []
    msgs = [{"role": "assistant",
              "content": f"🔎 {len(final_sources)} candidate sources "
                         f"(3-way subgraph: arxiv|github|web)."}]
    diff: dict[str, Any] = {
            "sources": final_sources,
        "messages": msgs,
        "errors": list(errs),
    }

    return diff
# ---------------------------------------------------------------------------
# Async convenience (used by nodes.py if it's already inside an event loop)
# ---------------------------------------------------------------------------

async def arun_researcher_subgraph(
    state: dict[str, Any],
    *,
    subgraph: CompiledStateGraph | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Async wrapper — runs the synchronous subgraph in a thread."""
    return await asyncio.to_thread(run_researcher_subgraph, state,
                                    subgraph=subgraph, config=config)