"""One search wave (v4 spec §2.8 ``run_wave``): planned queries → provider
calls under the run's budgets → hits deduped by canonical URL.

Planning runs in query order on the caller's (node) thread:

- ``exa``    one ``exa_search_<mode>`` unit from the ledger per query (a gap
             wave at most 2); over budget → the web provider, one
             ``search:`` line per wave; no ``EXA_API_KEY`` → DDGS, silently
             (a configuration state, not a failure);
- ``arxiv``  up to ``max_queries_per_run`` per run (``arxiv_queries`` in the
             ledger); the rest go to the web provider with one ``search:``
             line; requests are serialised under one lock, at least
             ``min_interval_s`` apart (arXiv's API asks for 3 s);
- ``github`` by ``github_mode``: ``exa_domain`` → Exa restricted to
             github.com (repo search when Exa is not usable), ``repo_search``
             → the GitHub API, ``off`` → the web provider;
- anything else (``ddgs``) → DDGS.

Recency (only when the caller passes one — ``resolve_recency``): Exa
``startPublishedDate``, DDGS ``timelimit``, arXiv ``sortBy=submittedDate``;
undated Exa hits are dropped and counted (``dropped_undated``). Provider
failures come back as their own error strings; one provider's outage never
empties a wave, and a hit whose URL ``canonical_url`` rejects is dropped with
an errors line naming it.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import TYPE_CHECKING

from .urls import canonical_url

if TYPE_CHECKING:
    from ..graph.run_config import RunConfig, SearchConfig
    from ..graph.state import ResearchBrief
    from .ledger import SpendLedger

MODES = {"first": "first pass", "gap": "gap wave", "extend": "extend"}
RECENCY_DAYS = {"d": 1, "w": 7, "m": 30, "y": 365}
_GAP_EXA_PER_WAVE = 2
_WAVE_WORKERS = 6
_sleep = time.sleep
_arxiv_last = [0.0]            # monotonic time of the last arXiv request, process-wide (the API limits per client)
_ARXIV_LOCK = threading.Lock()


def resolve_recency(rc: RunConfig, brief: ResearchBrief, *, today: date | None = None
                    ) -> tuple[dict | None, str | None]:
    """``({"window", "start_date"} | None, note | None)``: an explicit
    ``run_config.recency`` wins ("any": no recency, so nothing is dropped as
    undated, even for a time-sensitive brief); else the brief's
    ``recency_hint``, only for a ``time_sensitive`` brief. An unusable hint is
    ignored with a note."""
    window = rc.recency
    if window == "any":
        return None, None
    if window is None and brief.time_sensitive and brief.recency_hint:
        hint = brief.recency_hint.strip().lower()
        if hint not in RECENCY_DAYS:
            return None, f"recency: ignoring the brief's hint '{brief.recency_hint}' (expected d/w/m/y)"
        window = hint
    if window is None:
        return None, None
    start = (today or date.today()) - timedelta(days=RECENCY_DAYS[window])
    return {"window": window, "start_date": start.isoformat()}, None


def _ran_on(moved: Counter) -> str:
    return ", ".join(f"{n} quer{'y' if n == 1 else 'ies'} ran on {target}" for target, n in moved.items())


def _plan(queries: list[dict], cfg: SearchConfig, ledger: SpendLedger, mode: str, exa_key: bool
          ) -> tuple[list[tuple[str, dict]], list[str]]:
    """[(provider, query)] in query order, plus the planning lines."""
    exa_usable = exa_key and "exa" in cfg.providers_enabled
    arxiv_left = (max(0, cfg.arxiv.max_queries_per_run - ledger.spent("arxiv_queries"))
                  if cfg.arxiv.enabled and "arxiv" in cfg.providers_enabled else 0)
    exa_in_wave = 0
    over_exa: Counter = Counter()
    over_arxiv: Counter = Counter()

    def exa() -> bool | None:
        """True: one Exa search taken; False: Exa not configured; None: over budget."""
        nonlocal exa_in_wave
        if not exa_usable:
            return False
        if mode == "gap" and exa_in_wave >= _GAP_EXA_PER_WAVE:
            return None
        if not ledger.take(f"exa_search_{mode}"):
            return None
        exa_in_wave += 1
        return True

    def web(over: Counter | None) -> str:
        got = exa()
        if got:
            return "exa"
        if got is None and over is not None:
            over["ddgs"] += 1
        return "ddgs"

    plan: list[tuple[str, dict]] = []
    for q in queries:
        prov = q.get("provider") or "ddgs"
        if prov == "arxiv":
            if arxiv_left > 0:
                arxiv_left -= 1
                ledger.count("arxiv_queries")
                plan.append(("arxiv", q))
                continue
            if cfg.arxiv.enabled and "arxiv" in cfg.providers_enabled:
                target = web(None)
                over_arxiv[target] += 1
            else:
                target = web(over_exa)
            plan.append((target, q))
        elif prov == "github" and "github" in cfg.providers_enabled and cfg.github_mode != "off":
            if cfg.github_mode == "exa_domain":
                got = exa()
                if got:
                    plan.append(("exa", {**q, "include_domains": ["github.com"]}))
                    continue
                if got is None:
                    over_exa["github"] += 1
            ledger.count("github_queries")
            plan.append(("github", q))
        elif prov in ("exa", "github"):
            plan.append((web(over_exa), q))
        else:
            plan.append(("ddgs", q))
    lines = []
    if over_exa:
        lines.append(f"search: exa budget ({MODES[mode]}) spent — {_ran_on(over_exa)}")
    if over_arxiv:
        lines.append(f"search: arxiv cap ({cfg.arxiv.max_queries_per_run} per run) reached — {_ran_on(over_arxiv)}")
    return plan, lines


def _arxiv(sp, cfg: SearchConfig, query: str, **kw) -> tuple[list[dict], str]:
    """One arXiv request at a time, ``min_interval_s`` apart."""
    with _ARXIV_LOCK:
        wait = cfg.arxiv.min_interval_s - (time.monotonic() - _arxiv_last[0])
        if wait > 0:
            _sleep(wait)
        try:
            return sp.arxiv_hits(query, **kw)
        finally:
            _arxiv_last[0] = time.monotonic()


def run_wave(queries: list[dict], *, cfg: SearchConfig, ledger: SpendLedger, recency: dict | None,
             mode: str) -> tuple[list[dict], list[str]]:
    """Run one wave of planned queries (``{query, provider, sub_qs[, category]}``)
    → ``(hits, errors)``; every hit carries ``canonical_url``."""
    from ..graph import search_providers as sp      # lazy: the argus.graph package imports this module

    if mode not in MODES:
        raise ValueError(f"run_wave: unknown mode {mode!r} (expected one of {sorted(MODES)})")
    plan, errors = _plan(queries, cfg, ledger, mode, sp.exa_enabled())

    def one(step: tuple[str, dict]) -> tuple[str, tuple[list[dict], str]]:
        prov, q = step
        query, sub_qs = q.get("query") or "", list(q.get("sub_qs") or [])
        if prov == "exa":
            paper = q.get("category") == "research paper"
            return prov, sp.exa_search(
                query, category=q.get("category"), num_results=cfg.exa.num_results, want_text=True,
                sub_qs=sub_qs, text_max_chars=cfg.exa.text_max_chars_paper if paper else cfg.exa.text_max_chars,
                include_domains=q.get("include_domains"), search_type=cfg.exa.search_type, ledger=ledger,
                start_published_date=f"{recency['start_date']}T00:00:00.000Z" if recency else None)
        if prov == "arxiv":
            return prov, _arxiv(sp, cfg, query, sub_qs=sub_qs,
                                sort_by="submittedDate" if recency else "relevance")
        if prov == "github":
            return prov, sp.github_hits(query, sub_qs=sub_qs)
        ledger.count("ddgs_queries")
        return prov, sp.ddgs_hits(query, max_results=cfg.ddgs_max_results, sub_qs=sub_qs,
                                  timelimit=recency["window"] if recency else None)

    results = []
    if plan:
        with ThreadPoolExecutor(max_workers=min(_WAVE_WORKERS, len(plan)),
                                thread_name_prefix="argus-search") as ex:
            results = list(ex.map(one, plan))

    merged: list[dict] = []
    by_key: dict[str, int] = {}
    dropped = 0
    for prov, (hits, err) in results:
        if err:
            errors.append(err)
        for h in hits:
            if recency and prov == "exa" and not h.get("published"):
                dropped += 1                    # an undated hit can't show it is recent
                continue
            try:
                key = canonical_url(h["url"])
            except ValueError as e:             # one malformed hit never fails the wave
                errors.append(f"search: {prov} hit dropped, malformed URL {h['url']!r} ({e})")
                continue
            h = {**h, "canonical_url": key}
            if key not in by_key:
                by_key[key] = len(merged)
                merged.append(h)
                continue
            prev = merged[by_key[key]]
            tags = list(dict.fromkeys([*(prev.get("sub_qs") or []), *(h.get("sub_qs") or [])]))
            keep = h if h.get("text") and not prev.get("text") else prev
            merged[by_key[key]] = {**keep, "sub_qs": tags}
    if dropped:
        ledger.count("dropped_undated", dropped)
    return merged, errors


__all__ = ["run_wave", "resolve_recency", "MODES", "RECENCY_DAYS"]
