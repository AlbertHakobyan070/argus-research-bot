"""Search providers — one interface over Exa / DDGS / arXiv / GitHub.

Every provider returns ``SearchHit`` dicts so the scout and research
nodes can treat sources uniformly. Providers never raise: failures come
back as an error string so one provider's outage can't poison a wave
(the v2 failure-isolation property, kept).

Exa (https://exa.ai) is optional: enabled iff ``EXA_API_KEY`` is set.
Its killer feature for Argus is search + full page text in ONE call —
hits that carry text skip the fetch stage entirely. How many Exa searches
a run may make is a per-run budget (``run_config.search.exa``, spent
through the ``SpendLedger``); ``engines/search.run_wave`` plans each wave
under it and is re-exported here as ``run_query_wave``.
"""
from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Literal

import httpx

from ..engines.search import run_wave as run_query_wave

if TYPE_CHECKING:
    from ..engines.ledger import SpendLedger

logger = logging.getLogger("argus.search")

Provider = Literal["exa", "ddgs", "arxiv", "github"]

# A SearchHit is a plain dict (checkpoint-friendly) with keys:
#   url, title, snippet, kind, provider, sub_qs (list[int]), text (optional
#   full page text — Exa only), published (ISO date, "" when unknown);
#   run_wave adds canonical_url.

_EXA_ENDPOINT = "https://api.exa.ai/search"
_EXA_TIMEOUT = 20.0


def exa_enabled() -> bool:
    return bool(os.environ.get("EXA_API_KEY", "").strip())


def exa_search(query: str, *, category: str | None = None,
               num_results: int = 8, want_text: bool = True,
               sub_qs: list[int] | None = None, text_max_chars: int = 12000,
               include_domains: list[str] | None = None,
               start_published_date: str | None = None, search_type: str = "auto",
               ledger: SpendLedger | None = None) -> tuple[list[dict], str]:
    """One Exa /search call. Returns (hits, error). Never raises.

    ``category`` maps research intents onto Exa's index slices
    (e.g. "research paper", "news", "github"). With a ``ledger``, the
    response's ``costDollars.total`` and the hits that came with page text
    (``exa_text_pages``) are counted there.
    """
    key = os.environ.get("EXA_API_KEY", "").strip()
    if not key:
        return [], "exa: no EXA_API_KEY"
    body: dict[str, Any] = {
        "query": query[:400],
        "type": search_type,
        "numResults": max(1, min(int(num_results), 20)),
    }
    if category:
        body["category"] = category
    if include_domains:
        body["includeDomains"] = list(include_domains)
    if start_published_date:
        body["startPublishedDate"] = start_published_date
    if want_text:
        body["contents"] = {"text": {"maxCharacters": int(text_max_chars)}}
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
    if ledger is not None:
        cost = (data.get("costDollars") or {}).get("total")
        if cost:
            ledger.add_usd(float(cost))
        pages = sum(1 for it in data.get("results") or [] if (it.get("text") or "").strip())
        if want_text and pages:
            ledger.count("exa_text_pages", pages)
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
            "published": (it.get("publishedDate") or "")[:10],     # ISO 8601 → the date
            # v2 pipeline compat (merge/dedupe + plan-gate rendering).
            "source": "exa",
            "summary": (it.get("summary") or "")[:400],
        })
    return hits, ""


def _kind_from_category(category: str | None) -> str:
    return {"research paper": "paper", "news": "news",
            "github": "repo"}.get(category or "", "search_result")


def ddgs_hits(query: str, *, max_results: int = 10, sub_qs: list[int] | None = None,
              timelimit: str | None = None) -> tuple[list[dict], str]:
    """DDGS web search (free, keyless). Returns (hits, error); ``timelimit``
    is DDGS's recency window (d/w/m/y)."""
    try:
        from ..tools import ddgs_search
        results = ddgs_search(query, max_results=max_results, timelimit=timelimit)
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


def arxiv_hits(query: str, *, max_results: int = 10, sub_qs: list[int] | None = None,
               sort_by: str = "relevance") -> tuple[list[dict], str]:
    """arXiv Atom API search (free, keyless). Returns (hits, error);
    ``sort_by="submittedDate"`` lists the newest first (recency)."""
    q = (query or "").strip()[:200]
    if not q:
        return [], ""
    params = {
        "search_query": f"all:{q}",
        "start": 0, "max_results": max(1, min(int(max_results), 25)),
        "sortBy": sort_by, "sortOrder": "descending",
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
        pub_m = re.search(r"<published>(\d{4}-\d{2}-\d{2})", ent)
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
                "published": pub_m.group(1) if pub_m else "",
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


__all__ = [
    "exa_enabled", "exa_search", "ddgs_hits",
    "arxiv_hits", "github_hits", "run_query_wave",
]
