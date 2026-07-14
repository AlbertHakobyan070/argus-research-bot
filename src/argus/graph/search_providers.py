"""v3 search providers — one interface over Exa / DDGS / arXiv / GitHub.

Every provider returns ``SearchHit`` dicts so the scout and research
nodes can treat sources uniformly. Providers never raise: failures come
back as an error string so one provider's outage can't poison a wave
(the v2 failure-isolation property, kept).

Exa (https://exa.ai) is optional: enabled iff ``EXA_API_KEY`` is set.
Its killer feature for Argus is search + full page text in ONE call —
hits that carry text skip the fetch stage entirely. The free plan is
~1000 requests/month, so calls per run are capped
(``ARGUS_EXA_MAX_CALLS``, default 6) and DDGS takes the overflow.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Literal

import httpx

logger = logging.getLogger("argus.search")

Provider = Literal["exa", "ddgs", "arxiv", "github"]

# A SearchHit is a plain dict (checkpoint-friendly) with keys:
#   url, title, snippet, kind, provider, sub_qs (list[int]), text (optional
#   full page text — Exa only), published (optional ISO date).

_EXA_ENDPOINT = "https://api.exa.ai/search"
_EXA_TIMEOUT = 20.0


def exa_enabled() -> bool:
    return bool(os.environ.get("EXA_API_KEY", "").strip())


def exa_max_calls() -> int:
    return int(os.environ.get("ARGUS_EXA_MAX_CALLS", "6"))


def exa_search(query: str, *, category: str | None = None,
               num_results: int = 8, want_text: bool = True,
               sub_qs: list[int] | None = None) -> tuple[list[dict], str]:
    """One Exa /search call. Returns (hits, error). Never raises.

    ``category`` maps research intents onto Exa's index slices
    (e.g. "research paper", "news", "github").
    """
    key = os.environ.get("EXA_API_KEY", "").strip()
    if not key:
        return [], "exa: no EXA_API_KEY"
    body: dict[str, Any] = {
        "query": query[:400],
        "type": "auto",
        "numResults": max(1, min(int(num_results), 20)),
    }
    if category:
        body["category"] = category
    if want_text:
        body["contents"] = {"text": {"maxCharacters": 8000}}
    try:
        with httpx.Client(timeout=_EXA_TIMEOUT) as c:
            r = c.post(_EXA_ENDPOINT, json=body,
                       headers={"x-api-key": key,
                                "Content-Type": "application/json"})
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        err = f"exa: {type(e).__name__}: {e}"
        logger.warning(err)
        return [], err
    hits: list[dict] = []
    for it in data.get("results", []) or []:
        url = (it.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        hits.append({
            "url": url,
            "title": (it.get("title") or "").strip(),
            "snippet": (it.get("summary") or (it.get("text") or "")[:300]),
            "kind": _kind_from_category(category),
            "provider": "exa",
            "sub_qs": list(sub_qs or []),
            "text": (it.get("text") or "") if want_text else "",
            "published": it.get("publishedDate") or "",
            # v2 pipeline compat (merge/dedupe + plan-gate rendering).
            "source": "exa",
            "summary": (it.get("summary") or "")[:400],
        })
    return hits, ""


def _kind_from_category(category: str | None) -> str:
    return {"research paper": "paper", "news": "news",
            "github": "repo"}.get(category or "", "search_result")


def ddgs_hits(query: str, *, max_results: int = 10,
              sub_qs: list[int] | None = None) -> tuple[list[dict], str]:
    """DDGS web search (free, keyless). Returns (hits, error)."""
    try:
        from ..tools import ddgs_search
        results = ddgs_search(query, max_results=max_results)
    except Exception as e:
        err = f"ddgs: {type(e).__name__}: {e}"
        logger.warning(err)
        return [], err
    hits = []
    for r in results:
        url = (r.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        hits.append({
            "url": url,
            "title": (r.get("title") or "").strip(),
            "snippet": (r.get("snippet") or "")[:400],
            "kind": r.get("kind", "blog"),
            "provider": "ddgs",
            "sub_qs": list(sub_qs or []),
            "text": "",
            "published": "",
            "source": r.get("source", "ddgs"),
            "summary": (r.get("snippet") or "")[:400],
        })
    return hits, ""


def arxiv_hits(query: str, *, max_results: int = 10,
               sub_qs: list[int] | None = None) -> tuple[list[dict], str]:
    """arXiv Atom API search (free, keyless). Returns (hits, error)."""
    q = (query or "").strip()[:200]
    if not q:
        return [], ""
    params = {
        "search_query": f"all:{q}",
        "start": 0, "max_results": max(1, min(int(max_results), 25)),
        "sortBy": "relevance", "sortOrder": "descending",
    }
    try:
        with httpx.Client(timeout=25.0, follow_redirects=True) as c:
            r = c.get("https://export.arxiv.org/api/query", params=params)
            r.raise_for_status()
            text = r.text
    except Exception as e:
        err = f"arxiv: {type(e).__name__}: {e}"
        logger.warning(err)
        return [], err
    hits: list[dict] = []
    for ent in re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL):
        title_m = re.search(r"<title>(.*?)</title>", ent, re.DOTALL)
        link_m = re.search(r"<id>(.*?)</id>", ent)
        sum_m = re.search(r"<summary>(.*?)</summary>", ent, re.DOTALL)
        title = (title_m.group(1).strip() if title_m else "")
        link = (link_m.group(1).strip() if link_m else "")
        summary = (sum_m.group(1).strip()[:400] if sum_m else "")
        if title and link and link.startswith("http"):
            hits.append({
                "url": link,
                "title": re.sub(r"\s+", " ", title),
                "snippet": summary,
                "kind": "paper",
                "provider": "arxiv",
                "sub_qs": list(sub_qs or []),
                "text": "",
                "published": "",
                "source": "arxiv",
                "summary": summary,
            })
    return hits, ""


def github_hits(query: str, *, max_results: int = 8,
                sub_qs: list[int] | None = None) -> tuple[list[dict], str]:
    """GitHub repo search (keyless, 60 req/h). Returns (hits, error)."""
    q = (query or "").strip()
    if not q:
        return [], ""
    try:
        with httpx.Client(timeout=12.0, follow_redirects=True) as c:
            r = c.get("https://api.github.com/search/repositories",
                      params={"q": q[:200], "sort": "stars",
                              "order": "desc",
                              "per_page": max(1, min(int(max_results), 15))},
                      headers={"Accept": "application/vnd.github+json"})
        if r.status_code == 403:
            return [], "github: rate-limited (403)"
        if r.status_code != 200:
            return [], f"github: HTTP {r.status_code}"
        items = r.json().get("items", [])
    except Exception as e:
        err = f"github: {type(e).__name__}: {e}"
        logger.warning(err)
        return [], err
    hits = []
    for it in items:
        url = it.get("html_url", "")
        if not url:
            continue
        hits.append({
            "url": url,
            "title": it.get("full_name") or it.get("name", ""),
            "snippet": (it.get("description") or "")[:400],
            "kind": "repo",
            "provider": "github",
            "sub_qs": list(sub_qs or []),
            "text": "",
            "published": "",
            "source": "github-search",
            "summary": (it.get("description") or "")[:400],
        })
    return hits, ""


# ---------------------------------------------------------------------------
# Query wave execution — parallel fan-out over provider calls.
# ---------------------------------------------------------------------------

# One planned query: {"query": str, "provider": Provider, "sub_qs": [int],
#                     "category": str|None (exa only)}


def run_query_wave(queries: list[dict], *,
                   exa_budget: int | None = None) -> tuple[list[dict], list[str]]:
    """Execute a list of planned queries in parallel; merge + dedupe.

    Sync entrypoint (nodes are sync; they run on LangGraph worker
    threads). Internally fans out with asyncio.to_thread, matching the
    proven v2 subgraph pattern. Exa queries beyond ``exa_budget`` are
    transparently downgraded to DDGS.
    """
    budget = exa_max_calls() if exa_budget is None else exa_budget
    planned: list[dict] = []
    exa_used = 0
    for q in queries:
        q = dict(q)
        if q.get("provider") == "exa":
            if not exa_enabled() or exa_used >= budget:
                q["provider"] = "ddgs"
            else:
                exa_used += 1
        planned.append(q)

    async def _one(q: dict) -> tuple[list[dict], str]:
        prov = q.get("provider", "ddgs")
        query = q.get("query", "")
        sub_qs = q.get("sub_qs") or []
        if prov == "exa":
            return await asyncio.to_thread(
                exa_search, query, category=q.get("category"),
                sub_qs=sub_qs)
        if prov == "arxiv":
            return await asyncio.to_thread(arxiv_hits, query, sub_qs=sub_qs)
        if prov == "github":
            return await asyncio.to_thread(github_hits, query, sub_qs=sub_qs)
        return await asyncio.to_thread(ddgs_hits, query, sub_qs=sub_qs)

    async def _all() -> list[tuple[list[dict], str]]:
        return await asyncio.gather(*(_one(q) for q in planned))

    def _fresh_loop() -> list[tuple[list[dict], str]]:
        return asyncio.run(_all())

    try:
        results = _fresh_loop()
    except RuntimeError:
        # Called on a thread that already has a running loop — run the
        # wave in a fresh loop on a dedicated worker thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            results = ex.submit(_fresh_loop).result()

    merged: list[dict] = []
    errors: list[str] = []
    by_url: dict[str, dict] = {}
    for hits, err in results:
        if err:
            errors.append(err)
        for h in hits:
            url = h["url"]
            if url in by_url:
                # Merge sub-question tags; prefer a hit that has text.
                prev = by_url[url]
                tags = list(dict.fromkeys(
                    list(prev.get("sub_qs") or []) + list(h.get("sub_qs") or [])))
                if h.get("text") and not prev.get("text"):
                    h["sub_qs"] = tags
                    by_url[url] = h
                    merged[merged.index(prev)] = h
                else:
                    prev["sub_qs"] = tags
                continue
            by_url[url] = h
            merged.append(h)
    return merged, errors


__all__ = [
    "exa_enabled", "exa_max_calls", "exa_search", "ddgs_hits",
    "arxiv_hits", "github_hits", "run_query_wave",
]
