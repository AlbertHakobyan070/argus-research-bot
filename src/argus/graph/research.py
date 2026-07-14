"""v3 research node — the deep phase (runs after plan approval).

This is the piece v2 never had: sources are not just fetched and ranked
by keyword overlap — every fetched document is READ by a cheap-tier LLM
(the *digest* pass) which extracts atomic claims with quotes, relevance
and stance, keyed to the brief sub-questions. Writers downstream work
from these EvidenceNotes, never from 400-char excerpts.

Wave loop (bounded):
  triage → parallel fetch → parallel digest → coverage check
     └─(gaps + budget left)→ follow-up queries → merge sources → next wave

Extend / append (`/continue`) re-enter here naturally: the node only
fetches sources that aren't already in ``fetched``, so appended local
files and extend-wave hits are digested incrementally.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .. import llm
from ..tools import crawl_url, normalize_to_markdown, snatch_url
from .credibility import CREDIBILITY_FLOOR, credibility_score, score_fetched
from .jsonx import parse_json_obj
from .search_providers import run_query_wave
from .state import EvidenceNote, FetchedItem, ResearchBrief

logger = logging.getLogger("argus.research")

# Budgets (env-overridable).
MAX_WAVES = int(os.environ.get("ARGUS_RESEARCH_WAVES", "2"))
FETCH_CAP_PER_WAVE = int(os.environ.get("ARGUS_FETCH_PER_WAVE", "16"))
TOTAL_FETCH_CAP = int(os.environ.get("ARGUS_TOTAL_FETCH_CAP", "28"))
PER_SUBQ_TARGET = 3          # sources we aim to fetch per sub-question
COVERAGE_MIN_STRONG = 2      # notes with relevance >= 3 per sub-question
_FETCH_CONCURRENCY = int(os.environ.get("ARGUS_FETCH_CONCURRENCY", "4"))
_DIGEST_CONCURRENCY = int(os.environ.get("ARGUS_DIGEST_CONCURRENCY", "4"))
DIGEST_HEAD_CHARS = 6000
DIGEST_MID_CHARS = 2000


# ---------------------------------------------------------------------------
# Triage — pick which candidate sources to fetch this wave.
# ---------------------------------------------------------------------------

def _snippet_relevance(src: dict, keywords: list[str]) -> float:
    hay = ((src.get("title") or "") + " "
           + (src.get("snippet") or src.get("summary") or "")).lower()
    words = [w for k in keywords
             for w in re.split(r"[^a-z0-9]+", (k or "").lower())
             if len(w) >= 3]
    words = list(dict.fromkeys(words))
    if not words:
        return 0.5
    return sum(1 for w in words if w in hay) / len(words)


def triage(sources: list[dict], fetched_urls: set[str],
           brief: ResearchBrief, *, cap: int) -> list[dict]:
    """Rank un-fetched candidates and pick this wave's fetch set.

    Score = credibility prior (domain/URL heuristics) + snippet keyword
    relevance. Balanced across sub-questions (round-robin by tag) so one
    hot sub-question can't starve the others. Local (``local_path``)
    sources are always taken — they are explicit user appends.
    """
    pending = [s for s in sources
               if (s.get("url") or "") not in fetched_urls
               and (s.get("url") or s.get("local_path"))]
    locals_ = [s for s in pending if s.get("local_path")]
    web = [s for s in pending if not s.get("local_path")]

    scored: list[tuple[float, dict]] = []
    for s in web:
        item = FetchedItem(url=s.get("url", ""), title=s.get("title", ""))
        cred = credibility_score(item, user_request=" ".join(
            brief.must_have_keywords) or "")
        rel = _snippet_relevance(s, brief.must_have_keywords)
        score = 0.55 * cred + 0.45 * rel
        if cred < CREDIBILITY_FLOOR:
            score -= 0.25   # soft demotion, not a hard drop
        scored.append((score, s))
    scored.sort(key=lambda t: t[0], reverse=True)

    # Round-robin across sub-question tags for balance.
    by_subq: dict[int, list[dict]] = {}
    untagged: list[dict] = []
    for score, s in scored:
        tags = s.get("sub_qs") or []
        if tags:
            by_subq.setdefault(tags[0], []).append(s)
        else:
            untagged.append(s)
    picked: list[dict] = list(locals_)
    idxs = sorted(by_subq)
    rank = 0
    while len(picked) < cap and (any(by_subq.values()) or untagged):
        progressed = False
        for i in idxs:
            bucket = by_subq.get(i) or []
            if rank < len(bucket) and len(picked) < cap:
                picked.append(bucket[rank])
                progressed = True
        rank += 1
        if not progressed:
            while untagged and len(picked) < cap:
                picked.append(untagged.pop(0))
            break
    return picked[:max(cap, len(locals_))]


# ---------------------------------------------------------------------------
# Fetch — one source → FetchedItem dict (moved from v2 nodes.py; the
# battle-tested tool-routing + local-file handling is kept verbatim,
# with one v3 addition: Exa full-text short-circuit).
# ---------------------------------------------------------------------------

def _read_excerpt(md_path: str | None, limit: int = 600) -> str:
    if not md_path:
        return ""
    try:
        return Path(md_path).read_text(
            encoding="utf-8", errors="replace")[:limit]
    except Exception:
        return ""


def _scratch_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "argus_local_md"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_one_source(src: dict) -> tuple[dict | None, list[str]]:
    """Fetch ONE source into a FetchedItem dict (or None) + error strings."""
    errors: list[str] = []
    url = src.get("url")
    sub_qs = list(src.get("sub_qs") or [])
    provider = src.get("provider") or src.get("source") or ""

    # Appended LOCAL sources (vault transcripts, saved reports) carry
    # ``local_path`` instead of a fetchable URL.
    local_path = src.get("local_path")
    if local_path:
        try:
            p = Path(local_path)
            if not p.exists():
                msg = f"research: appended local file missing: {local_path}"
                logger.warning(msg)
                return None, [msg]
            if p.suffix.lower() == ".md":
                md_path = p
            elif p.suffix.lower() == ".txt":
                md_path = _scratch_dir() / (p.stem + ".md")
                md_path.write_text(
                    p.read_text(encoding="utf-8", errors="replace"),
                    encoding="utf-8")
            else:
                res = normalize_to_markdown(str(p))
                if not (res.ok and res.markdown_path):
                    return None, [f"research: could not normalize local file "
                                  f"{p.name}: {res.error or 'not ok'}"]
                md_path = Path(res.markdown_path)
            return FetchedItem(
                url="file:///" + str(p).replace("\\", "/"),
                title=src.get("title") or p.stem,
                markdown_path=str(md_path),
                section=src.get("kind", "local"),
                excerpt=_read_excerpt(str(md_path)),
                sub_qs=sub_qs, provider="local",
            ).model_dump(), []
        except Exception as e:
            msg = f"research: local file {local_path} failed: {e!r}"
            logger.warning(msg)
            return None, [msg]

    if not url or not isinstance(url, str) or not url.startswith("http"):
        msg = f"research skipping source with no/empty/non-http URL: {src!r}"
        logger.info(msg)
        return None, [msg]

    # v3: Exa already returned the full page text with the search hit —
    # persist it and skip the network fetch entirely.
    exa_text = (src.get("text") or "").strip()
    if exa_text and len(exa_text) > 300:
        try:
            safe = re.sub(r"[^A-Za-z0-9]+", "_", url)[-60:]
            md_path = _scratch_dir() / f"exa_{safe}.md"
            md_path.write_text(exa_text, encoding="utf-8")
            return FetchedItem(
                url=url, title=src.get("title", ""),
                markdown_path=str(md_path), section=src.get("kind", ""),
                excerpt=exa_text[:600], sub_qs=sub_qs, provider="exa",
            ).model_dump(), []
        except Exception as e:
            logger.warning("exa text persist failed (%r); falling through "
                           "to network fetch", e)

    kind = src.get("kind", "search_result")
    try:
        if kind == "official_doc" or "github.com" in url or "arxiv.org" in url:
            res = normalize_to_markdown(url)
            if res.ok and res.markdown_path:
                return FetchedItem(
                    url=url, title=res.title or src.get("title", ""),
                    markdown_path=res.markdown_path, section=kind,
                    excerpt=(res.markdown_text or "")[:600],
                    sub_qs=sub_qs, provider=provider,
                ).model_dump(), []
        if kind == "paper" and "arxiv.org/abs/" in url:
            pdf = url.replace("/abs/", "/pdf/") + ".pdf"
            res = snatch_url(pdf, kind="papers")
            if res.ok and res.markdown_path and Path(res.markdown_path).exists():
                return FetchedItem(
                    url=url, title=res.title or src.get("title", ""),
                    markdown_path=res.markdown_path, section=kind,
                    excerpt=_read_excerpt(res.markdown_path),
                    sub_qs=sub_qs, provider=provider,
                ).model_dump(), []
        res = snatch_url(url, kind="auto")
        if res.ok and res.markdown_path:
            return FetchedItem(
                url=url, title=res.title or src.get("title", ""),
                markdown_path=res.markdown_path, section=kind,
                excerpt=_read_excerpt(res.markdown_path),
                sub_qs=sub_qs, provider=provider,
            ).model_dump(), []
        res = crawl_url(url, deep=False, max_pages=4)
        if res.ok and res.markdown_path:
            return FetchedItem(
                url=url, title=src.get("title", ""),
                markdown_path=res.markdown_path, section=kind,
                excerpt=_read_excerpt(res.markdown_path),
                sub_qs=sub_qs, provider=provider,
            ).model_dump(), []
        # Last resort: bot-walled / Cloudflare sites via stealth fetcher.
        res = snatch_url(url, kind="auto", stealth=True)
        if res.ok and res.markdown_path:
            return FetchedItem(
                url=url, title=res.title or src.get("title", ""),
                markdown_path=res.markdown_path, section=kind,
                excerpt=_read_excerpt(res.markdown_path),
                sub_qs=sub_qs, provider=provider,
            ).model_dump(), []
        msg = f"fetch {url} failed: all tools returned not-ok (incl. stealth)"
        logger.warning(msg)
        return None, [msg]
    except Exception as e:
        msg = f"fetch {url} failed: {e!r}"
        logger.warning(msg)
        return None, [msg]


def _parallel_fetch(picked: list[dict]) -> tuple[list[dict], list[str]]:
    fetched: list[dict] = []
    errors: list[str] = []
    if not picked:
        return fetched, errors
    workers = max(1, min(_FETCH_CONCURRENCY, len(picked)))
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="argus-fetch") as ex:
        for item, errs in ex.map(fetch_one_source, picked):
            if item is not None:
                fetched.append(item)
            errors.extend(errs)
    return fetched, errors


# ---------------------------------------------------------------------------
# Digest — a cheap LLM READS each fetched document and extracts evidence.
# ---------------------------------------------------------------------------

DIGEST_SYSTEM = """You are a research analyst reading ONE source document.

You get the research question, specific sub-questions this source was
gathered for, and the document text. Extract the evidence THIS document
actually contains:

- claims: up to 5 atomic factual claims RELEVANT to the question. Each
  claim needs a short supporting quote copied from the document (<= 200
  chars). Skip claims the document does not directly support.
- relevance: 0-5 — how useful this document is for the sub-questions
  (0 = off-topic, 5 = directly answers one or more).
- stance: does the document "supports" the mainstream framing of the
  question, is it "mixed", does it "contradicts" common claims, or is it
  general "background"?

Numbers, dates, names: only include them inside claims if the quote
contains them. Never invent.

Return ONLY JSON:
{"relevance": 0-5, "stance": "supports|mixed|contradicts|background",
 "claims": [{"text": "...", "quote": "...",
             "confidence": "high|medium|low"}]}
"""


def _doc_text_for_digest(md_path: str | None) -> str:
    if not md_path:
        return ""
    try:
        text = Path(md_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= DIGEST_HEAD_CHARS + DIGEST_MID_CHARS:
        return text
    mid_at = len(text) // 2
    return (text[:DIGEST_HEAD_CHARS]
            + "\n\n[... document truncated ...]\n\n"
            + text[mid_at:mid_at + DIGEST_MID_CHARS])


def digest_one(item: dict, source_id: int, brief: ResearchBrief,
               user_request: str) -> tuple[dict | None, dict | None, str]:
    """Digest ONE fetched item → (EvidenceNote dict, model_call, error)."""
    doc = _doc_text_for_digest(item.get("markdown_path"))
    if len(doc) < 200:
        return None, None, (f"digest: {item.get('url')} has <200 chars of "
                            "text; skipped")
    tags = list(item.get("sub_qs") or [])
    sub_q_lines = "\n".join(
        f"- ({i}) {brief.sub_questions[i].q}"
        for i in tags if 0 <= i < len(brief.sub_questions)) or "- (any aspect)"
    human = (
        f"Research question: {user_request}\n"
        f"Sub-questions this source was gathered for:\n{sub_q_lines}\n\n"
        f"Document title: {item.get('title') or '(untitled)'}\n"
        f"Document URL: {item.get('url')}\n\n"
        f"DOCUMENT TEXT:\n{doc}\n\nReturn the evidence JSON."
    )
    try:
        chat = llm.chat_for_tier("cheap", temperature=0.1, max_tokens=900)
        resp = llm.invoke_with_retry(chat, [
            SystemMessage(content=DIGEST_SYSTEM),
            HumanMessage(content=human[:12000]),
        ])
        rec = llm.record_from_response("cheap", llm.resolve_tier("cheap"),
                                       resp).model_dump()
        data = parse_json_obj(resp.content, require_any=("relevance",
                                                         "claims"))
    except Exception as e:
        return None, None, f"digest: {item.get('url')} failed ({e})"
    claims = []
    for c in (data.get("claims") or [])[:5]:
        if not isinstance(c, dict) or not (c.get("text") or "").strip():
            continue
        conf = c.get("confidence", "medium")
        if conf not in ("high", "medium", "low"):
            conf = "medium"
        claims.append({"text": str(c["text"])[:500],
                       "quote": str(c.get("quote") or "")[:250],
                       "confidence": conf})
    try:
        rel = max(0, min(5, int(data.get("relevance", 0))))
    except (TypeError, ValueError):
        rel = 0
    stance = data.get("stance", "background")
    if stance not in ("supports", "mixed", "contradicts", "background"):
        stance = "background"
    note = EvidenceNote(
        source_id=source_id,
        source_url=item.get("url", ""),
        title=item.get("title", ""),
        sub_qs=tags,
        relevance=rel,
        stance=stance,
        claims=claims,
    )
    return note.model_dump(), rec, ""


def _parallel_digest(fetched_new: list[dict], id_offset: int,
                     brief: ResearchBrief, user_request: str
                     ) -> tuple[list[dict], list[dict], list[str]]:
    notes: list[dict] = []
    calls: list[dict] = []
    errors: list[str] = []
    if not fetched_new:
        return notes, calls, errors
    workers = max(1, min(_DIGEST_CONCURRENCY, len(fetched_new)))
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="argus-digest") as ex:
        futs = [ex.submit(digest_one, item, id_offset + i + 1, brief,
                          user_request)
                for i, item in enumerate(fetched_new)]
        for f in futs:
            note, call, err = f.result()
            if note is not None:
                notes.append(note)
            if call is not None:
                calls.append(call)
            if err:
                errors.append(err)
    return notes, calls, errors


# ---------------------------------------------------------------------------
# Coverage + follow-up queries
# ---------------------------------------------------------------------------

def compute_coverage(brief: ResearchBrief, evidence: list[dict]) -> dict:
    """Per sub-question: count of strong notes (relevance >= 3)."""
    cov = {str(i): 0 for i in range(len(brief.sub_questions))}
    for n in evidence:
        if int(n.get("relevance") or 0) < 3:
            continue
        for i in n.get("sub_qs") or []:
            key = str(i)
            if key in cov:
                cov[key] += 1
    return cov


def _gap_subqs(coverage: dict) -> list[int]:
    return sorted(int(k) for k, v in coverage.items()
                  if v < COVERAGE_MIN_STRONG)


FOLLOWUP_SYSTEM = """Earlier searches under-covered some research
sub-questions. Write ONE better search query per listed sub-question —
different wording/angle than before, concrete, 3-9 words.

Return ONLY JSON:
{"queries": [{"sub_q": <index>, "q": "...",
              "provider": "web|arxiv|github"}]}
"""


def followup_queries(brief: ResearchBrief, gaps: list[int],
                     prior_queries: list[dict]
                     ) -> tuple[list[dict], list[dict]]:
    """One cheap LLM call for gap-targeted queries; deterministic fallback."""
    from .scout import _web_provider  # late import (no cycle at module load)
    gap_lines = "\n".join(f"- ({i}) {brief.sub_questions[i].q}"
                          for i in gaps if i < len(brief.sub_questions))
    prior = "\n".join(f"- {q.get('query')}" for q in prior_queries[-10:])
    human = (f"Under-covered sub-questions:\n{gap_lines}\n\n"
             f"Queries already tried:\n{prior}\n\nReturn the queries JSON.")
    try:
        chat = llm.chat_for_tier("cheap", temperature=0.4, max_tokens=500)
        resp = llm.invoke_with_retry(chat, [
            SystemMessage(content=FOLLOWUP_SYSTEM),
            HumanMessage(content=human[:5000]),
        ])
        rec = llm.record_from_response("cheap", llm.resolve_tier("cheap"),
                                       resp).model_dump()
        data = parse_json_obj(resp.content, require_any=("queries",))
        out = []
        for item in data.get("queries") or []:
            q = str(item.get("q") or "").strip() if isinstance(item, dict) else ""
            try:
                idx = int(item.get("sub_q", -1))
            except (TypeError, ValueError):
                idx = -1
            if not q or idx not in gaps:
                continue
            prov = item.get("provider", "web")
            if prov not in ("web", "arxiv", "github"):
                prov = "web"
            out.append({"query": q[:120],
                        "provider": _web_provider() if prov == "web" else prov,
                        "sub_qs": [idx]})
        if out:
            return out, [rec]
    except Exception as e:
        logger.warning("followup querygen failed (%s); deterministic "
                       "fallback", e)
    out = []
    for i in gaps:
        if i < len(brief.sub_questions):
            words = re.sub(r"[?\"']", "",
                           brief.sub_questions[i].q).split()
            kw = " ".join(brief.must_have_keywords[:2])
            out.append({"query": f"{kw} {' '.join(words[:6])}"[:90].strip(),
                        "provider": _web_provider(), "sub_qs": [i]})
    return out, []


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

def research_node(state) -> dict:
    """Deep research: waves of triage → fetch → digest → coverage."""
    brief = ResearchBrief.model_validate(state.get("brief") or {})
    user_request = state.get("user_request") or ""
    append_only = bool(state.get("append_only"))

    sources: list[dict] = list(state.get("sources") or [])
    fetched: list[dict] = list(state.get("fetched") or [])
    evidence: list[dict] = list(state.get("evidence") or [])
    queries: list[dict] = list(state.get("queries") or [])
    fetched_urls = {f.get("url", "") for f in fetched}
    # Local appends surface as file:/// in fetched but keep local_path in
    # sources — map both so re-runs don't re-fetch.
    for f in fetched:
        u = f.get("url", "")
        if u.startswith("file:///"):
            fetched_urls.add(u[len("file:///"):].replace("/", "\\"))

    calls: list[dict] = []
    errors: list[str] = []
    waves = 0
    max_waves = 1 if append_only else MAX_WAVES

    while True:
        room = TOTAL_FETCH_CAP - len(fetched)
        if room <= 0:
            break
        picked = triage(sources, fetched_urls, brief,
                        cap=min(FETCH_CAP_PER_WAVE, room))
        # Skip locals already ingested (matched via local_path).
        picked = [s for s in picked
                  if not (s.get("local_path")
                          and str(s["local_path"]) in fetched_urls
                          or (s.get("local_path") and
                              "file:///" + str(s["local_path"]).replace("\\", "/")
                              in {f.get("url") for f in fetched}))]
        if not picked and waves == 0 and not fetched:
            errors.append("research: no fetchable sources — scout wave "
                          "found nothing usable")
            break
        new_fetched, fetch_errs = _parallel_fetch(picked)
        errors.extend(fetch_errs)
        for item in new_fetched:
            fetched_urls.add(item.get("url", ""))
        # Score credibility on the batch (title now known post-fetch).
        if new_fetched:
            scored = score_fetched(
                [FetchedItem.model_validate(f) for f in new_fetched],
                user_request=user_request)
            new_fetched = [s.model_dump() for s in scored]
        notes, digest_calls, digest_errs = _parallel_digest(
            new_fetched, len(fetched), brief, user_request)
        calls.extend(digest_calls)
        errors.extend(digest_errs)
        fetched.extend(new_fetched)
        evidence.extend(notes)
        waves += 1

        coverage = compute_coverage(brief, evidence)
        gaps = _gap_subqs(coverage)
        if append_only or not gaps or waves >= max_waves \
                or len(fetched) >= TOTAL_FETCH_CAP:
            break
        fq, fq_calls = followup_queries(brief, gaps, queries)
        calls.extend(fq_calls)
        if not fq:
            break
        hits, wave_errs = run_query_wave(fq)
        errors.extend(wave_errs)
        queries.extend(fq)
        known = {s.get("url") for s in sources}
        for h in hits:
            if h["url"] not in known:
                sources.append(h)
                known.add(h["url"])

    coverage = compute_coverage(brief, evidence)
    if fetched and not evidence:
        errors.append(
            f"research: fetched {len(fetched)} source(s) but digested 0 "
            "evidence notes — reports would be ungrounded")
    if sources and not fetched:
        errors.append(
            f"research: all {len([s for s in sources if s.get('url')])} "
            "source(s) failed to fetch")

    strong = sum(1 for n in evidence if int(n.get("relevance") or 0) >= 3)
    out: dict[str, Any] = {
        "sources": sources,
        "fetched": fetched,
        "evidence": evidence,
        "coverage": coverage,
        "queries": queries,
        "append_only": False,   # consumed — clear for later passes
        "research_rounds": int(state.get("research_rounds") or 0) + waves,
        "model_calls": calls,
        "messages": [{"role": "assistant",
                      "content": (f"📚 Research: {len(fetched)} sources read, "
                                  f"{len(evidence)} digested "
                                  f"({strong} strong) over {waves} wave(s).")}],
    }
    if errors:
        out["errors"] = errors
    return out


__all__ = ["research_node", "triage", "fetch_one_source", "digest_one",
           "compute_coverage", "followup_queries", "DIGEST_SYSTEM",
           "MAX_WAVES", "TOTAL_FETCH_CAP", "COVERAGE_MIN_STRONG"]
