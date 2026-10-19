"""v4 research node — the deep phase (runs after plan approval).

Every fetched document is READ by a cheap LLM (the *digest*,
``research.digest``), which extracts atomic claims with quotes, relevance,
stance and the source type, keyed to the brief's sub-questions; each quote
is then checked against the document (verbatim verification). Writers
downstream work from these EvidenceNotes, never from raw excerpts.

One node run is one *pass*, chosen from the state:

- first   (no research yet) — up to ``research.waves`` waves over the
          scout's sources; after a wave that leaves coverage gaps, one
          follow-up querygen (``research.followup_querygen``) + search;
- gap     (``coverage_gaps``, set by the panel) — ONE wave restricted to
          those sub-questions: follow-up queries, ``gap_fetch_per_wave``
          fetches, at most 2 Exa searches; appends to ``gap_history``,
          ``gap_waves += 1``, clears ``coverage_gaps``; refused at
          ``gap_waves_max``;
- extend  (a user Extend; ``extend_prep`` only bumped its counters) — one
          fresh allowance (``fetch_per_wave`` fetches, ``max_searches //
          waves`` Exa searches), widened queries
          (``research.extend_querygen``), then waves as in the first pass;
- append  (``append_only``: /continue after /append) — exactly the
          appended sources, no search.

A wave is: pick (``research.rerank`` scores the candidates, or keyword
``triage``; at most ``per_subq_target`` per sub-question before the
leftover slots go by rank) → fetch (``engines/fetchers``) → digest →
coverage. Every budget goes through one ``SpendLedger`` (spec §2.4); pinned
``/append`` sources bypass every cap and win the dedupe. Between items the node reads
``runtime.drain_requested`` (spec §2.5): once it is set no new fetch
starts, what was fetched is still digested, the remaining waves are
skipped, and the partial update carries a ``cancelled:`` line.
"""
from __future__ import annotations

import bisect
import logging
import math
import re
import tempfile
from collections import Counter
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from .. import llm
from ..engines import fetchers
from ..engines.ledger import SpendLedger
from ..engines.urls import canonical_url, domain
from ..tools import normalize_to_markdown
from . import run_config
from .credibility import (CREDIBILITY_FLOOR, authority_prior, credibility_score, note_authority,
                          score_fetched)
from .jsonx import parse_json_obj
from .scout import _web_provider
from .search_providers import exa_enabled, run_query_wave
from .state import EvidenceNote, FetchedItem, ResearchBrief

logger = logging.getLogger("argus.research")

_APPENDED = ("appended-asset", "appended-url")      # bot/history.py _sources_from_refs
_RERANK_MAX = 40          # candidates one rerank call scores; keyword triage pre-selects them
_EXCERPT_CHARS = 600
_STRONG = 3               # a "strong" note: relevance >= 3
_MODE_LABEL = {"first": "first pass", "gap": "gap wave", "extend": "extend", "append": "appended sources"}
_DEGRADE_LINE = re.compile(r"degrade: (\S+) → (\S+) \((.*)\)$", re.DOTALL)
SOURCE_TYPES = ("official_doc", "paper", "repo", "news", "blog", "forum", "other")


def _round(state) -> int:
    """k of the ``[r<k>]`` line tag: 1 + revise_rounds + extend_rounds (spec §2.8)."""
    return 1 + int(state.get("revise_rounds") or 0) + int(state.get("extend_rounds") or 0)


def _emit(runtime: Runtime | None, kind: str, data: dict) -> None:
    """One custom stream event ``{kind, node, data}`` (spec §2.5), from the node thread."""
    if runtime is not None:
        runtime.stream_writer({"kind": kind, "node": "research", "data": data})


def _is_pinned(src: dict) -> bool:
    """A user-appended source (/append, local file): bypasses every cap."""
    return bool(src.get("pinned") or src.get("local_path") or src.get("source") in _APPENDED)


def _local_url(local_path: str) -> str:
    return "file:///" + str(Path(local_path)).replace("\\", "/")


def _source_key(src: dict) -> str:
    """The dedupe key: the canonical URL; ``file:///…`` for local appends.
    ``ValueError`` for a malformed URL (``_keep_valid`` drops those first)."""
    return _local_url(src["local_path"]) if src.get("local_path") else canonical_url(src.get("url") or "")


def _keep_valid(sources: list[dict], errors: list[str]) -> list[dict]:
    """``sources`` without those whose URL ``canonical_url`` rejects — one errors
    line each, so one bad hit never fails the node (spec §2.8)."""
    out = []
    for s in sources:
        try:
            _source_key(s)
        except ValueError as e:
            errors.append(f"research: dropped source with a malformed URL {s.get('url')!r} ({e})")
            continue
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Triage — keyword ranking of un-fetched candidates (the rerank's degradation).
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


def _quota_pick(ranked: list[dict], cap: int, per_subq: int) -> list[dict]:
    """A wave's ``cap`` picks from ``ranked`` (best first) under the
    ``research.per_subq_target`` quota: each sub-question — a source's first
    ``sub_qs`` tag, triage's bucket — first takes up to ``per_subq`` of its best,
    in rank order; the leftover slots then go by rank, and only they take
    untagged sources. Quota picks come first, then the leftovers."""
    quota, per = [], Counter()
    for i, s in enumerate(ranked):
        tags = s.get("sub_qs") or []
        if len(quota) < cap and tags and per[tags[0]] < per_subq:
            per[tags[0]] += 1
            quota.append(i)
    taken = set(quota)
    leftovers = [i for i in range(len(ranked)) if i not in taken][:max(0, cap - len(quota))]
    return [ranked[i] for i in quota + leftovers]


# ---------------------------------------------------------------------------
# Rerank — research.rerank scores the candidates before anything is fetched.
# ---------------------------------------------------------------------------

RERANK_SYSTEM = """You rank candidate sources for a research brief BEFORE any page is fetched.

You get the research question, its numbered sub-questions and numbered
candidates (domain, title, snippet, and the sub-questions each was found
for). Score EVERY candidate 0-5 for how likely its page holds specific,
first-hand evidence for the sub-questions:
5 = primary source, official documentation or a paper directly on point;
3 = useful secondary coverage; 1 = tangential; 0 = off-topic, spam or an
SEO page.

Return ONLY JSON: {"scores": [{"id": <candidate number>, "score": <0-5>}, ...]}
"""


@dataclass
class RerankResult:
    picked: list[dict]
    calls: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    degraded: bool = False
    event: dict = field(default_factory=dict)       # the "rerank" event's data


def _sub_question_lines(brief: ResearchBrief, idxs=None) -> str:
    n = len(brief.sub_questions)
    idxs = range(n) if idxs is None else [i for i in idxs if 0 <= i < n]
    return "\n".join(f"- ({i}) {brief.sub_questions[i].q}" for i in idxs) or "- (none)"


def rerank(candidates: list[dict], brief: ResearchBrief, user_request: str, *,
           rc: run_config.RunConfig, wave: int, rnd: int, cap: int) -> RerankResult:
    """``research.rerank``: one call scores every candidate; the wave takes
    ``cap`` of them by score under the ``per_subq_target`` quota
    (``_quota_pick``; ties keep candidate order, score 0 is dropped, a
    candidate the reply leaves out ranks after every scored one). Degrade →
    ``keyword_triage``: the same quota over ``triage``'s order, and the
    ``degrade:`` line says so."""
    n = len(candidates)
    lines = []
    for i, c in enumerate(candidates, 1):
        snippet = " ".join((c.get("snippet") or c.get("summary") or "").split())[:160]
        tags = ", ".join(str(t) for t in c.get("sub_qs") or []) or "any"
        lines.append(f"{i}. [{domain(c.get('url') or '') or 'unknown'}] {(c.get('title') or '')[:120]} — "
                     f"{snippet} (sub-questions: {tags})")
    human = (f"Research question: {user_request}\nSub-questions:\n{_sub_question_lines(brief)}\n\n"
             "Candidates:\n" + "\n".join(lines) + "\n\nReturn the scores JSON.")

    def validate(text: str) -> dict[int, int]:
        scores: dict[int, int] = {}
        for s in parse_json_obj(text, require_any=("scores",)).get("scores") or []:
            if not isinstance(s, dict):
                continue
            try:
                i, v = int(s.get("id")), int(s.get("score"))
            except (TypeError, ValueError):
                continue
            if 1 <= i <= n:
                scores[i] = max(0, min(5, v))
        if not scores:
            raise ValueError("no candidate scores in the reply")
        return scores

    per_subq = rc.research.per_subq_target
    try:
        res = llm.complete("research.rerank", [SystemMessage(content=RERANK_SYSTEM), HumanMessage(content=human)],
                           engines=rc.engines, temperature=0.0, max_tokens=200 + 16 * n, validate=validate)
    except llm.EngineDegraded as e:
        picked = _quota_pick(triage(candidates, set(), brief, cap=n), cap, per_subq)
        return RerankResult(picked, [a.model_dump() for a in e.attempts],
                            [f"[r{rnd}] degrade: research.rerank → {e.degradation} ({e})"], True,
                            {"wave": wave, "candidates": n, "kept": len(picked)})
    scores = res.parsed
    keep = sorted((i for i in range(1, n + 1) if scores.get(i, 0.5) > 0), key=lambda i: -scores.get(i, 0.5))
    picked = _quota_pick([candidates[i - 1] for i in keep], cap, per_subq)
    return RerankResult(picked, [r.model_dump() for r in res.records], [], False,
                        {"wave": wave, "candidates": n, "kept": len(picked)})


# ---------------------------------------------------------------------------
# Fetch — one source → FetchedItem dict (the chain lives in engines/fetchers).
# ---------------------------------------------------------------------------

def _scratch_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "argus_local_md"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch_local(src: dict, local_path: str, sub_qs: list[int]) -> tuple[dict | None, list[str]]:
    """Appended LOCAL sources (vault transcripts, saved reports) carry
    ``local_path`` instead of a fetchable URL."""
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
            md_path.write_text(p.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        else:
            res = normalize_to_markdown(str(p))
            if not (res.ok and res.markdown_path):
                return None, [f"research: could not normalize local file {p.name}: {res.error or 'not ok'}"]
            md_path = Path(res.markdown_path)
        text = md_path.read_text(encoding="utf-8", errors="replace")
        url = _local_url(local_path)
        return FetchedItem(
            url=url, title=src.get("title") or p.stem, markdown_path=str(md_path),
            section=src.get("kind", "local"), excerpt=text[:_EXCERPT_CHARS], sub_qs=sub_qs,
            provider="local", canonical_url=url, fetcher="local", chars=len(text),
            authority_prior=authority_prior(url), pinned=True,
        ).model_dump(), []
    except Exception as e:      # any failure on the user's own file is reported, never dropped silently
        msg = f"research: local file {local_path} failed: {e!r}"
        logger.warning(msg)
        return None, [msg]


def fetch_one_source(src: dict, *, cfg, ledger: SpendLedger | None,
                     exa_enabled: bool) -> tuple[dict | None, list[str]]:
    """Fetch ONE source into a FetchedItem dict (or None) + error strings."""
    sub_qs = list(src.get("sub_qs") or [])
    if src.get("local_path"):
        return _fetch_local(src, src["local_path"], sub_qs)
    url = src.get("url")
    if not url or not isinstance(url, str) or not url.startswith("http"):
        msg = f"research skipping source with no/empty/non-http URL: {src!r}"
        logger.info(msg)
        return None, [msg]
    res = fetchers.fetch(src, cfg=cfg, ledger=ledger, exa_enabled=exa_enabled, pinned=_is_pinned(src))
    if not res.ok:
        msg = (f"fetch {url} failed: {res.error} "
               f"(rungs tried: {', '.join(t['rung'] for t in res.rungs_tried) or 'none'})")
        logger.warning(msg)
        return None, [msg]
    kind = src.get("kind") or "search_result"
    excerpt = Path(res.markdown_path).read_text(encoding="utf-8")[:_EXCERPT_CHARS]
    return FetchedItem(
        url=url, title=res.title or src.get("title") or "", markdown_path=res.markdown_path,
        section=kind, excerpt=excerpt, sub_qs=sub_qs, provider=src.get("provider") or src.get("source") or "",
        canonical_url=canonical_url(url), domain=domain(url), published=src.get("published") or "",
        fetcher=res.fetcher, fetch_ms=res.ms, chars=res.chars,
        authority_prior=authority_prior(url, kind), pinned=_is_pinned(src), rungs_tried=res.rungs_tried,
    ).model_dump(), []


def _parallel_fetch(picked: list[dict], *, cfg, ledger: SpendLedger, exa_on: bool, kind: str | None = None,
                    stop: Callable[[], bool] | None = None,
                    on_fetch: Callable[[dict, dict | None], None] | None = None
                    ) -> tuple[list[dict], list[str]]:
    """Fetch ``picked`` → ``(fetched, errors)``, both in pick order.

    At most ``cfg.concurrency`` fetches are in flight; the next is submitted
    only when one finishes and only while ``stop()`` is false — the drain
    check, read on the node thread (pool threads never see the runtime).
    Each non-pinned source takes one ``kind`` unit from the ledger at
    submit; pinned sources never do (``kind=None``: no budget, for direct
    calls). ``on_fetch(src, item | None)`` runs on the node thread as each
    fetch completes."""
    if not picked:
        return [], []
    workers = max(1, min(cfg.concurrency, len(picked)))
    queue = list(enumerate(picked))
    results: dict[int, tuple[dict | None, list[str]]] = {}
    unfunded = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="argus-fetch") as ex:
        running: dict = {}
        while True:
            while queue and len(running) < workers and not (stop is not None and stop()):
                i, src = queue.pop(0)
                if kind is not None and not _is_pinned(src) and not ledger.take(kind):
                    unfunded += 1
                    continue
                running[ex.submit(fetch_one_source, src, cfg=cfg, ledger=ledger, exa_enabled=exa_on)] = i
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                i = running.pop(fut)
                results[i] = fut.result()
                if on_fetch is not None:
                    on_fetch(picked[i], results[i][0])
    order = sorted(results)
    errors = [e for i in order for e in results[i][1]]
    if unfunded:
        errors.append(f"research: {kind} budget spent — {unfunded} source(s) left unfetched")
    return [results[i][0] for i in order if results[i][0] is not None], errors


# ---------------------------------------------------------------------------
# Quote verification (spec §2.8, B4) — verbatim.
# ---------------------------------------------------------------------------

# quote marks and emphasis asterisks dropped, dashes unified, "_" a space (``_x_`` emphasis, snake_case alike)
_QUOTE_FOLD = str.maketrans({**dict.fromkeys("\"'`“”„‟‘’‚‛«»‹›*"), **dict.fromkeys("–—‐‑−", "-"), "_": " "})
# no run here crosses a "]" (link text: nor a "["), so the scans never overlap: linear on many unclosed "[" / "]("
_FOOTNOTE = re.compile(r"\[\d+\](?:\([^)\]]*\))?")        # a footnote marker, [3] (or linked: [3](#fn3))
_MD_LINK = re.compile(r"!?\[([^\[\]]*)\]\([^)\]]*\)")     # a markdown link or image → its text
_LIST_MARKER = re.compile(r"^[ \t]*\d+[.)][ \t]+", re.M)  # an ordered-list marker at a line start
_BLOCKQUOTE = re.compile(r"^[ \t]*(?:>[ \t]?)+", re.M)    # blockquote markers at a line start: ">", ">>", "> >"
# whole numbers (a sign only when no word character precedes it, . , : only between digits, a trailing %),
# the comparison symbols, words: "0.5%" ≠ "5%", "-5%" ≠ "5%", "≤" ≠ "≥"
_TOKEN = re.compile(r"(?<!\w)[-+]?\d+(?:[.,:]\d+)*%?|\d+(?:[.,:]\d+)*%?|[<>≤≥≠±=]|\w+")
_NEGATORS = frozenset("not never cannot cant dont doesnt isnt arent wasnt werent wont wouldnt shouldnt "
                      "couldnt without".split())          # the fold deletes apostrophes: don't → dont
_ELLIPSIS = re.compile(r"\.\.\.|…")
_SEGMENT_GAP = 3000          # chars from one ellipsis segment's end to the next one's first character
_MAX_SEGMENTS = 8            # a quote with more ellipsis segments never verifies (bounds the occurrence scan)


def _words(text: str) -> str:
    """``text`` normalised — ``<=`` ``>=`` ``!=`` as ``≤`` ``≥`` ``≠`` (first, so a
    ``>=`` opening a line is no blockquote marker), markdown links and images
    (their text), footnote markers, blockquote and ordered-list markers at line
    starts, quote marks, dashes, case — as its tokens (``_TOKEN``), one space before
    each and after the last. A ``>`` anywhere else stays a symbol token."""
    text = _MD_LINK.sub(r"\1", _FOOTNOTE.sub("", text.replace("<=", "≤").replace(">=", "≥").replace("!=", "≠")))
    text = _LIST_MARKER.sub("", _BLOCKQUOTE.sub("", text))    # blockquote markers first: "> 1. x" is a list item
    return " " + "".join(t + " " for t in _TOKEN.findall(text.translate(_QUOTE_FOLD).lower()))


class QuoteIndex:
    """One document's normalised tokens (``_words``), built once per document."""

    def __init__(self, text: str):
        self.words = _words(text or "")


def quote_verified(quote: str, idx: QuoteIndex) -> bool | None:
    """Is ``quote`` verbatim in the document (spec §2.8)? ``None``: no quote.

    The quote's tokens, normalised as the document's, must occur contiguously
    and in order, the first not right after a negator (an excerpt may not cut a
    negation off); an ellipsis (``...`` / ``…``) splits it into at most
    ``_MAX_SEGMENTS`` segments that must each occur so, in order, each one's first
    character at most ``_SEGMENT_GAP`` chars (of the normalised document) after the
    previous one's end. A substituted, added or dropped word, number or symbol fails."""
    segments = [w for w in (_words(part) for part in _ELLIPSIS.split(quote or "")) if w != " "]
    if not segments:
        return None
    if len(segments) > _MAX_SEGMENTS:
        return False
    ends: list[int] | None = None
    for seg in segments:
        starts, at = [], idx.words.find(seg)
        while at >= 0:
            starts.append(at)
            at = idx.words.find(seg, at + 1)
        if ends is None:                          # the first segment: not right after a negator token
            starts = [s for s in starts if idx.words[idx.words.rfind(" ", 0, s) + 1:s] not in _NEGATORS]
        else:                                     # after a previous end; its first char (s + 1) within the gap
            starts = [s for s in starts
                      if (i := bisect.bisect_right(ends, s)) and s + 1 - ends[i - 1] <= _SEGMENT_GAP]
        if not starts:
            return False
        ends = [s + len(seg) - 1 for s in starts]     # the trailing space, where the next segment may start
    return True


# ---------------------------------------------------------------------------
# Digest — a cheap LLM READS each fetched document and extracts evidence.
# ---------------------------------------------------------------------------

DIGEST_SYSTEM = """You are a research analyst reading ONE source document.

You get the research question, specific sub-questions this source was
gathered for, and the document text (or one part of a long document).
Extract the evidence THIS text actually contains:

- claims: up to 5 atomic factual claims RELEVANT to the question. Each
  claim needs a short supporting quote copied VERBATIM from the text
  (<= 200 chars). Skip claims the text does not directly support.
- relevance: 0-5 — how useful this document is for the sub-questions
  (0 = off-topic, 5 = directly answers one or more).
- stance: does the document "supports" the mainstream framing of the
  question, is it "mixed", does it "contradicts" common claims, or is it
  general "background"?
- source_type: what the document is — "official_doc" (the vendor's or
  project's own documentation), "paper" (academic paper or preprint),
  "repo" (a code repository or its README), "news", "blog", "forum"
  (Q&A or discussion thread) or "other".

Numbers, dates, names: only include them inside claims if the quote
contains them. Never invent.

Return ONLY JSON:
{"relevance": 0-5, "stance": "supports|mixed|contradicts|background",
 "source_type": "official_doc|paper|repo|news|blog|forum|other",
 "claims": [{"text": "...", "quote": "...",
             "confidence": "high|medium|low"}]}
"""


def _head_and_middle(text: str, head: int, mid: int) -> str:
    if len(text) <= head + mid:
        return text
    at = len(text) // 2
    return text[:head] + "\n\n[... document truncated ...]\n\n" + text[at:at + mid]


def _digest_chunks(text: str, research) -> list[str]:
    """What the digest reads: the whole text when it fits ``head + mid``
    chars; else ``digest_chunks_long_doc`` equal parts (no more than the
    length needs), each cut to its head + middle."""
    head, mid = research.digest_head_chars, research.digest_mid_chars
    if len(text) <= head + mid:
        return [text]
    k = min(research.digest_chunks_long_doc, math.ceil(len(text) / (head + mid)))
    size = math.ceil(len(text) / k)
    return [_head_and_middle(text[i * size:(i + 1) * size], head, mid) for i in range(k)]


def _word_cut(text: str, limit: int) -> str:
    """``text`` cut to at most ``limit`` chars at the last whitespace at or before
    ``limit`` (a single token longer than ``limit`` keeps the hard cut), so a quote
    shortened here never ends in a word fragment, which ``quote_verified`` rejects."""
    if len(text) <= limit:
        return text
    cut = next((i for i in range(limit, 0, -1) if text[i].isspace()), 0)
    return text[:cut].rstrip() or text[:limit]


def _digest_reply(text: str) -> dict:
    """The validate hook: the reply's JSON, normalised (raises on no JSON)."""
    data = parse_json_obj(text, require_any=("relevance", "claims"))
    claims = []
    for c in (data.get("claims") or [])[:5]:
        if not isinstance(c, dict) or not str(c.get("text") or "").strip():
            continue
        conf = c.get("confidence") if c.get("confidence") in ("high", "medium", "low") else "medium"
        claims.append({"text": str(c["text"])[:500], "quote": _word_cut(str(c.get("quote") or ""), 250),
                       "confidence": conf})
    try:
        relevance = max(0, min(5, int(data.get("relevance", 0))))
    except (TypeError, ValueError):
        relevance = 0
    stance = data.get("stance")
    source_type = data.get("source_type")
    return {"relevance": relevance, "claims": claims,
            "stance": stance if stance in ("supports", "mixed", "contradicts", "background") else "background",
            "source_type": source_type if source_type in SOURCE_TYPES else "other"}


def digest_one(item: dict, source_id: int, brief: ResearchBrief, user_request: str, *,
               rc: run_config.RunConfig, rnd: int) -> tuple[dict | None, list[dict], list[str]]:
    """Digest ONE fetched item → ``(EvidenceNote dict | None, model_calls, errors)``.

    A long document is read in ``digest_chunks_long_doc`` parts (call items
    ``<url>#chunk<i>``) and merged: relevance is the best part's and the
    stance and source type come with it; claims are deduplicated by text.
    A part whose engines are exhausted is a ``skip:`` line; the note is
    built from the parts that were read."""
    url = item.get("url") or ""
    path = item.get("markdown_path")
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace") if path else ""
    except OSError as e:
        return None, [], [f"digest: {url} unreadable ({type(e).__name__}: {e})"]
    text = re.sub(r"\n{3,}", "\n\n", raw).strip()
    if len(text) < 200:
        return None, [], [f"digest: {url} has <200 chars of text; skipped"]
    tags = list(item.get("sub_qs") or [])
    sub_q_lines = "\n".join(f"- ({i}) {brief.sub_questions[i].q}"
                            for i in tags if 0 <= i < len(brief.sub_questions)) or "- (any aspect)"
    chunks = _digest_chunks(text, rc.research)
    parts: list[dict] = []
    calls: list[dict] = []
    errors: list[str] = []
    for n, chunk in enumerate(chunks, 1):
        label = f"{url}#chunk{n}" if len(chunks) > 1 else url
        part = f"Part {n} of {len(chunks)} of the document (each part is read separately).\n" if len(chunks) > 1 else ""
        human = (f"Research question: {user_request}\n"
                 f"Sub-questions this source was gathered for:\n{sub_q_lines}\n\n"
                 f"Document title: {item.get('title') or '(untitled)'}\n"
                 f"Document URL: {url}\n{part}\n"
                 f"DOCUMENT TEXT:\n{chunk}\n\nReturn the evidence JSON.")
        try:
            res = llm.complete("research.digest", [SystemMessage(content=DIGEST_SYSTEM),
                                                   HumanMessage(content=human[:12000])],
                               engines=rc.engines, temperature=0.1, max_tokens=900, validate=_digest_reply,
                               item=label)
        except llm.EngineSkipped as e:
            calls += [a.model_dump() for a in e.attempts]
            errors.append(f"[r{rnd}] skip: research.digest {label} ({e})")
            continue
        calls += [r.model_dump() for r in res.records]
        parts.append(res.parsed)
    if not parts:
        return None, calls, errors
    best = max(parts, key=lambda p: p["relevance"])
    idx = QuoteIndex(text)
    claims, seen = [], set()
    for p in parts:
        for c in p["claims"]:
            key = " ".join(c["text"].lower().split())
            if key not in seen:
                seen.add(key)
                claims.append({**c, "quote_verified": quote_verified(c["quote"], idx)})
    note = EvidenceNote(
        source_id=source_id, source_url=url, title=item.get("title", ""), sub_qs=tags,
        relevance=best["relevance"], stance=best["stance"], claims=claims, source_type=best["source_type"],
        authority=note_authority(int(item.get("authority_prior", 1)), best["source_type"]),
    )
    return note.model_dump(), calls, errors


def _parallel_digest(items: list[dict], id_offset: int, brief: ResearchBrief, user_request: str, *,
                     rc: run_config.RunConfig, rnd: int,
                     on_digest: Callable[[int, dict | None], None] | None = None
                     ) -> tuple[list[dict], list[dict], list[str]]:
    """Digest ``items`` on ``digest_concurrency`` workers → ``(notes, calls,
    errors)`` in item order; source ids continue after ``id_offset``."""
    if not items:
        return [], [], []
    notes: list[dict] = []
    calls: list[dict] = []
    errors: list[str] = []
    workers = max(1, min(rc.research.digest_concurrency, len(items)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="argus-digest") as ex:
        futs = [ex.submit(digest_one, item, id_offset + i + 1, brief, user_request, rc=rc, rnd=rnd)
                for i, item in enumerate(items)]
        for i, f in enumerate(futs):
            note, item_calls, errs = f.result()
            if note is not None:
                notes.append(note)
            calls += item_calls
            errors += errs
            if on_digest is not None:
                on_digest(id_offset + i + 1, note)
    return notes, calls, errors


# ---------------------------------------------------------------------------
# Coverage + follow-up / extend queries
# ---------------------------------------------------------------------------

def compute_coverage(brief: ResearchBrief, evidence: list[dict]) -> dict:
    """Per sub-question: count of strong notes (relevance >= 3)."""
    cov = {str(i): 0 for i in range(len(brief.sub_questions))}
    for n in evidence:
        if int(n.get("relevance") or 0) < _STRONG:
            continue
        for i in n.get("sub_qs") or []:
            key = str(i)
            if key in cov:
                cov[key] += 1
    return cov


def _gap_subqs(coverage: dict, min_strong: int) -> list[int]:
    return sorted(int(k) for k, v in coverage.items() if v < min_strong)


FOLLOWUP_SYSTEM = """Earlier searches under-covered some research
sub-questions. Write ONE better search query per listed sub-question —
different wording/angle than before, concrete, 3-9 words.

Return ONLY JSON:
{"queries": [{"sub_q": <index>, "q": "...",
              "provider": "web|arxiv|github"}]}
"""

EXTEND_SYSTEM = """The user asked to EXTEND finished research: find sources
the earlier searches missed. For each listed sub-question write 1-2 search
queries from a NEW angle — adjacent aspects, primary sources, recent
developments, counter-evidence — never a rewording of a query already
tried; concrete, 3-9 words.

Return ONLY JSON:
{"queries": [{"sub_q": <index>, "q": "...",
              "provider": "web|arxiv|github"}]}
"""

_QUERYGEN = {"research.followup_querygen": (FOLLOWUP_SYSTEM, 1),
             "research.extend_querygen": (EXTEND_SYSTEM, 2)}


def followup_queries(brief: ResearchBrief, gaps: list[int], prior_queries: list[dict], *,
                     rc: run_config.RunConfig, role: str = "research.followup_querygen",
                     rnd: int = 1) -> tuple[list[dict], list[dict], list[str]]:
    """One querygen call → ``(queries, model_calls, errors)``:
    ``research.followup_querygen`` for coverage gaps (one query per
    sub-question), ``research.extend_querygen`` for a user Extend (up to
    two). No queries → no search. Degrade (``skip_followup_wave`` /
    ``skip_extend_wave``) → no queries and the ``degrade:`` line."""
    system, per_subq = _QUERYGEN[role]
    gaps = [i for i in gaps if 0 <= i < len(brief.sub_questions)]
    prior = "\n".join(f"- {q.get('query')}" for q in prior_queries[-12:]) or "- (none)"
    human = (f"Sub-questions:\n{_sub_question_lines(brief, gaps)}\n\n"
             f"Queries already tried:\n{prior}\n\nReturn the queries JSON.")

    def validate(text: str) -> list:
        items = parse_json_obj(text, require_any=("queries",)).get("queries")
        if not isinstance(items, list):
            raise ValueError(f"'queries' is {type(items).__name__}, not a list")
        return items

    try:
        res = llm.complete(role, [SystemMessage(content=system), HumanMessage(content=human[:5000])],
                           engines=rc.engines, temperature=0.4, max_tokens=500, validate=validate)
    except llm.EngineDegraded as e:
        return [], [a.model_dump() for a in e.attempts], [f"[r{rnd}] degrade: {role} → {e.degradation} ({e})"]
    out: list[dict] = []
    per: Counter = Counter()
    for item in res.parsed:
        if not isinstance(item, dict):
            continue
        q = " ".join(str(item.get("q") or "").split())
        try:
            idx = int(item.get("sub_q", -1))
        except (TypeError, ValueError):
            continue
        if not q or idx not in gaps or per[idx] >= per_subq:
            continue
        prov = item.get("provider") if item.get("provider") in ("arxiv", "github") else _web_provider()
        out.append({"query": q[:120], "provider": prov, "sub_qs": [idx]})
        per[idx] += 1
    return out, [r.model_dump() for r in res.records], []


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class _Pass:
    """One research_node run: the working lists, the ledger, the wave loop."""

    def __init__(self, state, rc: run_config.RunConfig, brief: ResearchBrief, runtime: Runtime | None):
        self.state, self.rc, self.brief, self.runtime = state, rc, brief, runtime
        self.user_request = state.get("user_request") or ""
        self.rnd = _round(state)
        self.ledger = SpendLedger.for_run(rc, state.get("search_spend"))
        self.recency = self.ledger.snapshot()["recency"]          # decided once, by the scout
        self.exa_on = "exa" in rc.search.providers_enabled and exa_enabled()
        self.calls: list[dict] = []
        self.errors: list[str] = []
        self.sources = _keep_valid(list(state.get("sources") or []), self.errors)
        self.fetched = list(state.get("fetched") or [])
        self.evidence = list(state.get("evidence") or [])
        self.queries = list(state.get("queries") or [])
        self.extra: dict[str, Any] = {}
        self.waves = 0
        self.done = {canonical_url(f.get("url") or "") for f in self.fetched}
        self.attempted: set[str] = set()          # submitted this run: a failure is not retried in a later wave
        self.last = (0, 0, 1)                     # (fetches done, picked, wave) for the cancelled: line
        self.drain_seen = self.drain_noted = False

    # -- events and the drain ----------------------------------------------
    def emit(self, kind: str, data: dict) -> None:
        _emit(self.runtime, kind, data)

    def draining(self) -> bool:
        if self.runtime is None or not self.runtime.drain_requested:
            return False
        if not self.drain_seen:
            self.drain_seen = True
            self.emit("stage", {"status": "draining", "summary": self.runtime.drain_reason or ""})
        return True

    def stop_drained(self) -> None:
        if not self.drain_noted:
            self.drain_noted = True
            k, n, w = self.last
            self.errors.append(f"cancelled: research stopped after {k}/{n} fetches in wave {w} "
                               f"({self.runtime.drain_reason})")

    def note(self, calls: list[dict], lines: list[str]) -> None:
        """A role's attempts and its error lines; a ``degrade:`` line is also an event."""
        self.calls += calls
        for line in lines:
            m = _DEGRADE_LINE.search(line)
            if m:
                self.emit("degrade", {"role": m[1], "degradation": m[2], "reason": m[3]})
        self.errors += lines

    # -- search, pick, one wave ----------------------------------------------
    def search(self, queries: list[dict], mode: str, wave: int) -> None:
        hits, errs = run_query_wave(queries, cfg=self.rc.search, ledger=self.ledger, recency=self.recency, mode=mode)
        self.errors += errs
        self.queries += queries
        index = {_source_key(s): i for i, s in enumerate(self.sources)}
        for h in _keep_valid(hits, self.errors):
            key = h.get("canonical_url") or canonical_url(h.get("url") or "")
            if key not in index:
                index[key] = len(self.sources)
                self.sources.append(h)
                continue
            prev = self.sources[index[key]]
            tags = list(dict.fromkeys([*(prev.get("sub_qs") or []), *(h.get("sub_qs") or [])]))
            keep = h if h.get("text") and not prev.get("text") and not _is_pinned(prev) else prev
            self.sources[index[key]] = {**keep, "sub_qs": tags}
        self.emit("search", {"wave": wave, "mode": mode, "queries": len(queries), "hits": len(hits),
                             "errors": len(errs)})

    def pick(self, *, kind: str, cap: int, wave: int, only: set[int] | None = None,
             pinned_only: bool = False) -> list[dict]:
        """Pinned pending sources (always; never reranked, quota'd or capped) + up
        to ``cap`` (and the ledger's ``kind`` remainder) of the others under the
        ``per_subq_target`` quota, restricted to ``only`` sub-questions. Pinned
        sources are visited first, so a pinned copy wins the dedupe."""
        pending, seen = [], set()
        for s in sorted(self.sources, key=lambda s: not _is_pinned(s)):
            key = _source_key(s)
            if key and key not in seen and key not in self.done and key not in self.attempted:
                seen.add(key)
                pending.append(s)
        pinned = [s for s in pending if _is_pinned(s)]
        web = [s for s in pending if not _is_pinned(s)
               and (only is None or set(s.get("sub_qs") or []) & only)]
        cap = min(cap, self.ledger.remaining(kind))
        if pinned_only or cap <= 0 or not web:
            return pinned
        if not self.rc.research.rerank:
            return pinned + _quota_pick(triage(web, set(), self.brief, cap=len(web)), cap,
                                        self.rc.research.per_subq_target)
        r = rerank(triage(web, set(), self.brief, cap=_RERANK_MAX), self.brief, self.user_request,
                   rc=self.rc, wave=wave, rnd=self.rnd, cap=cap)
        self.note(r.calls, r.errors)
        self.emit("rerank", r.event)
        return pinned + r.picked

    def wave(self, picked: list[dict], *, kind: str | None, wave: int) -> None:
        done = [0]

        def on_fetch(src: dict, item: dict | None) -> None:
            done[0] += 1
            self.emit("fetch", {"url": src.get("url") or (item or {}).get("url", ""), "ok": item is not None,
                                "fetcher": (item or {}).get("fetcher", ""), "ms": (item or {}).get("fetch_ms", 0),
                                "pinned": _is_pinned(src)})

        def on_digest(source_id: int, note: dict | None) -> None:
            self.emit("digest", {"source_id": source_id, "ok": note is not None,
                                 "relevance": note["relevance"] if note else None, "cached": False})

        self.attempted |= {_source_key(s) for s in picked}
        new, errs = _parallel_fetch(picked, cfg=self.rc.fetch, ledger=self.ledger, exa_on=self.exa_on,
                                    kind=kind, stop=self.draining, on_fetch=on_fetch)
        self.errors += errs
        self.last = (done[0], len(picked), wave)
        self.waves += 1
        if new:
            scored = score_fetched([FetchedItem.model_validate(f) for f in new], user_request=self.user_request)
            new = [s.model_dump() for s in scored]
        notes, calls, derrs = _parallel_digest(new, len(self.fetched), self.brief, self.user_request,
                                               rc=self.rc, rnd=self.rnd, on_digest=on_digest)
        self.calls += calls
        self.errors += derrs
        self.fetched += new
        self.done |= {canonical_url(f.get("url") or "") for f in new}
        self.evidence += notes

    def run_waves(self, *, kind: str, search_mode: str, max_waves: int, cap: int,
                  only: set[int] | None = None, opening: list[dict] | None = None) -> None:
        """Up to ``max_waves`` waves. ``opening``: the queries searched before
        wave 1; before every later wave, while coverage gaps and the pass's
        fetch budget remain, one follow-up querygen + search (none after a
        querygen gave nothing). The drain is read at the top of every wave,
        before every pick (so no rerank call runs after it) and before every
        fetch submit."""
        followups = True
        for w in range(1, max_waves + 1):
            if self.draining():
                return self.stop_drained()
            if w == 1 and opening:
                self.search(opening, search_mode, w)
            elif w > 1:
                gaps = _gap_subqs(compute_coverage(self.brief, self.evidence), self.rc.research.coverage_min_strong)
                if not gaps or self.ledger.remaining(kind) <= 0:
                    break                          # nothing to cover, or nothing left to fetch it with
                if followups:
                    fq, calls, errs = followup_queries(self.brief, gaps, self.queries, rc=self.rc,
                                                       role="research.followup_querygen", rnd=self.rnd)
                    self.note(calls, errs)
                    followups = bool(fq)
                    if fq:
                        self.search(fq, search_mode, w)
            if self.draining():
                return self.stop_drained()
            picked = self.pick(kind=kind, cap=cap, wave=w, only=only)
            if not picked:
                if w == 1 and kind == "fetch_first" and not self.fetched:
                    self.errors.append("research: no fetchable sources — scout wave found nothing usable")
                break
            self.wave(picked, kind=kind, wave=w)
            if self.draining():
                return self.stop_drained()

    # -- the four passes -------------------------------------------------------
    def first_pass(self) -> None:
        r = self.rc.research
        self.run_waves(kind="fetch_first", search_mode="first", max_waves=r.waves, cap=r.fetch_per_wave)

    def extend_pass(self) -> None:
        self.ledger.grant_extend()
        if self.draining():                       # before the querygen: a drain costs no LLM call
            return self.stop_drained()
        everything = list(range(len(self.brief.sub_questions)))
        queries, calls, errs = followup_queries(self.brief, everything, self.queries, rc=self.rc,
                                                role="research.extend_querygen", rnd=self.rnd)
        self.note(calls, errs)
        r = self.rc.research
        self.run_waves(kind="fetch_extend", search_mode="extend", max_waves=r.waves, cap=r.fetch_per_wave,
                       opening=queries)

    def gap_pass(self, coverage_gaps: list[int]) -> None:
        r = self.rc.research
        gap_waves = int(self.state.get("gap_waves") or 0)
        history = list(self.state.get("gap_history") or [])
        self.extra = {"coverage_gaps": [], "gap_waves": gap_waves, "gap_history": history}
        if gap_waves >= r.gap_waves_max:
            self.errors.append(f"research: gap wave refused — gap_waves_max ({r.gap_waves_max}) reached")
            return
        gaps = sorted({int(i) for i in coverage_gaps if 0 <= int(i) < len(self.brief.sub_questions)})
        if not gaps:
            return
        coverage = compute_coverage(self.brief, self.evidence)
        owned = {i for s in (self.state.get("outline") or {}).get("sections") or [] for i in s.get("sub_qs") or []}
        starved = any(coverage.get(str(i), 0) < r.coverage_min_strong or i not in owned for i in gaps)
        if self.draining():                       # before the querygen: a drain costs no LLM call
            return self.stop_drained()
        queries, calls, errs = followup_queries(self.brief, gaps, self.queries, rc=self.rc,
                                                role="research.followup_querygen", rnd=self.rnd)
        self.note(calls, errs)
        before = len(self.evidence)
        self.run_waves(kind="fetch_gap", search_mode="gap", max_waves=1, cap=r.gap_fetch_per_wave,
                       only=set(gaps), opening=queries)
        strong = [n for n in self.evidence[before:] if int(n.get("relevance") or 0) >= _STRONG]
        history.append({"round": self.rnd, "wave": gap_waves + 1, "sub_qs": gaps, "new_strong": len(strong),
                        "source": "coverage" if starved else "judge",
                        "new_strong_by_subq": {str(i): sum(1 for n in strong if i in (n.get("sub_qs") or []))
                                               for i in gaps}})
        self.extra["gap_waves"] = gap_waves + 1

    def append_pass(self) -> None:
        picked = self.pick(kind="fetch_first", cap=0, wave=1, pinned_only=True)
        if picked:
            self.wave(picked, kind=None, wave=1)
            if self.draining():
                self.stop_drained()

    def finish(self, mode: str) -> dict:
        coverage = compute_coverage(self.brief, self.evidence)
        if self.fetched and not self.evidence:   # ALL_SKIPPED_RAISES (spec §2.3): never ship a hollow report
            last = next((e for e in reversed(self.errors) if "research.digest" in e or e.startswith("digest:")), "")
            raise llm.EngineError(
                f"research: fetched {len(self.fetched)} source(s) but digested 0 evidence notes — reports would "
                "be ungrounded" + (f" (last: {last})" if last else ""), role="research.digest",
                attempts=[llm.CallRecord(**c) for c in self.calls if c.get("role") == "research.digest"])
        if self.sources and not self.fetched:
            self.errors.append(f"research: all {len([s for s in self.sources if s.get('url')])} "
                               "source(s) failed to fetch")
        strong = sum(1 for n in self.evidence if int(n.get("relevance") or 0) >= _STRONG)
        message = (f"📚 Research ({_MODE_LABEL[mode]}): {len(self.fetched)} sources read, "
                   f"{len(self.evidence)} digested ({strong} strong) over {self.waves} wave(s).")
        out: dict[str, Any] = {
            "sources": self.sources, "fetched": self.fetched, "evidence": self.evidence,
            "coverage": coverage, "queries": self.queries,
            "append_only": False,   # consumed — clear for later passes
            "research_rounds": int(self.state.get("research_rounds") or 0) + self.waves,
            "model_calls": self.calls, "search_spend": self.ledger.snapshot(),
            "messages": [{"role": "assistant", "content": message}], **self.extra,
        }
        if self.errors:
            out["errors"] = self.errors
        self.emit("stage", {"status": "end", "summary": message})
        return out


def research_node(state, runtime: Runtime | None = None) -> dict:
    """One research pass (see the module doc); LangGraph injects ``runtime``."""
    rc = run_config.require(state)
    brief = ResearchBrief.model_validate(state.get("brief") or {})
    p = _Pass(state, rc, brief, runtime)
    if state.get("append_only"):
        mode = "append"
    elif state.get("coverage_gaps"):
        mode = "gap"
    elif not int(state.get("research_rounds") or 0) and not int(state.get("extend_rounds") or 0):
        mode = "first"
    else:
        mode = "extend"
    p.emit("stage", {"status": "start", "summary": _MODE_LABEL[mode]})
    if mode == "append":
        p.append_pass()
    elif mode == "gap":
        p.gap_pass(list(state.get("coverage_gaps") or []))
    elif mode == "first":
        p.first_pass()
    else:
        p.extend_pass()
    return p.finish(mode)


__all__ = ["research_node", "triage", "rerank", "RerankResult", "fetch_one_source", "digest_one",
           "QuoteIndex", "quote_verified", "compute_coverage", "followup_queries",
           "DIGEST_SYSTEM", "RERANK_SYSTEM", "SOURCE_TYPES"]
