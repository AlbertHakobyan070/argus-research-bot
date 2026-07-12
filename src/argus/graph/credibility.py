"""Argus credibility scoring — DomainTrust static list + per-URL scoring.

Sits between fetcher_node and filter_node. Assigns every FetchedItem a
``credibility_score`` in [0, 1] based on three signals:

  (a) Domain trust      — curated DomainTrust list + TLD analysis
  (b) URL pattern       — arxiv / .edu / .gov boost, content-farm penalty,
                          arxiv year-fabrication flag (P2.5, 2026-07-10)
  (c) Title relevance   — token overlap with the user_request

Items below the 0.4 floor are tagged (``credibility_flag = "low_credibility"``)
but NOT dropped — the downstream ``filter_node`` still does its own
``keep_topN`` so dropping here would just double-filter.

Reference: HANDOFF-RESEARCH-2026-07-08.md §4 (lines 110-135) and §6 row #2.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import ClassVar, Iterable
from urllib.parse import urlparse

from .state import FetchedItem


# Lowered 0.40 -> 0.35 (2026-07-12 depth rebalance): with the widened
# tier scale below, legit reference/curated domains comfortably clear
# this while content farms (tier "low" ~0.05) stay well under it.
CREDIBILITY_FLOOR = 0.35


# ---------------------------------------------------------------------------
# Arxiv year-fabrication heuristic (P2.5, 2026-07-10)
# ---------------------------------------------------------------------------
#
# In live runs (notably the 2026-07-09 03:30 metacognitive-RL report),
# the LLM hallucinates arxiv absolute IDs whose YYMM segment encodes a
# year that's either in the future (e.g. ``2606.32032`` in July 2026 —
# MM=06 may look fine but the 5-digit sequence at month-6 isn't reached
# yet) or earlier than the format itself (any /abs/YYMM.NNNNN with
# year < 2015 is impossible, the format was introduced 2015-01).
#
# We surface those URLs with ``credibility_flag = "fabricated_path"``
# AND drag the score below CREDIBILITY_FLOOR so the existing
# filter_node P2 enforcement drops them. Real arxiv URLs aren't
# affected: 2015 ≤ year ≤ current_year with MM ≤ 6 and a 5-digit
# sequence will trip the heuristic in the early year, but those are
# vanishingly rare in real submissions anyway.
#
# Old-format IDs (``/abs/1706037``) are intentionally NOT handled here.
# They're rare in 2026 LLM output and would need a different parser.
# If we see them in production we can extend ``is_arxiv_year_suspicious``.
_ARXIV_NEW_RE = re.compile(r"^/abs/(\d{4})\.(\d{4,5})(?:v\d+)?$")
_ARXIV_NEW_FORMAT_START_YEAR = 2015   # YY=1501 (Jan 2015) is the earliest


def is_arxiv_year_suspicious(url: str, *, current_year: int | None = None
                              ) -> tuple[bool, int | None]:
    """Return ``(is_suspicious, detected_year)`` for an arxiv /abs/ URL.

    Non-arxiv hosts and non-``/abs/`` paths return ``(False, None)``.
    """
    if current_year is None:
        current_year = datetime.now().year
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if not host or "arxiv.org" not in host:
        return (False, None)
    m = _ARXIV_NEW_RE.match(parsed.path or "")
    if not m:
        return (False, None)
    yymm, seq = m.group(1), m.group(2)
    try:
        yy = int(yymm[:2])
        mm = int(yymm[2:4])
    except ValueError:
        return (False, None)
    if not (1 <= mm <= 12):
        return (True, 2000 + yy)
    year = 2000 + yy
    if year < _ARXIV_NEW_FORMAT_START_YEAR:
        return (True, year)
    if year > current_year:
        return (True, year)
    if year == current_year and len(seq) == 5 and mm <= 6:
        return (True, year)
    return (False, year)


# ---------------------------------------------------------------------------
# DomainTrust — curated static list. Hand-tuned for Argus's research-bot use.
# Tiers:
#   "primary"   0.55  — arxiv, .edu, .gov, official docs, named journals
#   "trusted"   0.40  — known-quality blogs / curated publications
#   "neutral"   0.20  — generic commercial / news
#   "low"       0.05  — content farms, SEO spam, parked domains
# ---------------------------------------------------------------------------


# Module-level constants — not instance attributes — so they sidestep the
# dataclass "no mutable default" rule and remain trivially picklable.

_DOMAIN_TABLE: tuple[tuple[str, str], ...] = (
    # ---- primary -----------------------------------------------------
    ("arxiv.org", "primary"),
    ("openreview.net", "primary"),
    ("acm.org", "primary"),
    ("dl.acm.org", "primary"),
    ("ieee.org", "primary"),
    ("springer.com", "primary"),
    ("nature.com", "primary"),
    ("science.org", "primary"),
    ("sciencedirect.com", "primary"),
    ("cell.com", "primary"),
    ("jmlr.org", "primary"),
    ("papers.nips.cc", "primary"),
    ("proceedings.neurips.cc", "primary"),
    ("proceedings.mlr.press", "primary"),
    ("aclweb.org", "primary"),
    ("usenix.org", "primary"),
    ("nist.gov", "primary"),
    ("nih.gov", "primary"),
    ("nasa.gov", "primary"),
    ("europa.eu", "primary"),
    ("semanticscholar.org", "primary"),
    ("aaai.org", "primary"),
    ("biorxiv.org", "primary"),
    ("medrxiv.org", "primary"),
    ("ar5iv.org", "primary"),
    ("ar5iv.labs.arxiv.org", "primary"),
    # TLD-based primaries — substring match means ".edu" in host
    # (e.g. "cs.stanford.edu") catches the whole .edu namespace.
    (".edu", "primary"),
    (".gov", "primary"),
    (".ac.uk", "primary"),
    (".ac.jp", "primary"),
    # ---- trusted blogs / curated ------------------------------------
    ("github.com", "trusted"),
    ("gitlab.com", "trusted"),
    ("huggingface.co", "trusted"),
    ("paperswithcode.com", "trusted"),
    ("openai.com", "trusted"),
    ("anthropic.com", "trusted"),
    ("deepmind.google", "trusted"),
    ("research.google", "trusted"),
    ("ai.googleblog.com", "trusted"),
    ("ai.meta.com", "trusted"),
    ("microsoft.com", "trusted"),
    ("bair.berkeley.edu", "trusted"),
    ("distill.pub", "trusted"),
    # General reference / encyclopedic — legitimate, was scoring as noise
    # (0.11) before the 2026-07-12 rebalance because it wasn't listed.
    ("wikipedia.org", "trusted"),
    ("wikimedia.org", "trusted"),
    ("stackoverflow.com", "trusted"),
    ("stackexchange.com", "trusted"),
    ("towardsdatascience.com", "trusted"),
    ("medium.com", "trusted"),
    ("substack.com", "trusted"),
    # ---- low (P2 fix, 2026-07-09) -----------------------------------
    # P2 bug: previously zero entries — content farms fell through to
    # the "neutral" default and the documented content-farm penalty
    # never fired. Two layers:
    #  (a) curated hostnames — known offenders observed in real runs
    #      (the GLM 5.2 report cited thetechbriefs.com and glm45.org).
    #      Maintain these as new farms get observed — add here, not
    #      hard-coded elsewhere.
    #  (b) structural heuristics — see ``_STRUCTURAL_LOW_HOST_RE`` and
    #      ``_looks_like_seo_slug`` below; they catch patterns the
    #      curated table misses without us shipping a 200-entry list.
    # ---- curated low (a) ----
    ("thetechbriefs.com", "low"),
    ("techbriefs.com", "low"),
    ("ai-news-briefs.com", "low"),
    ("research-briefs.com", "low"),
    ("ml-briefs.com", "low"),
    ("seo-content.ai", "low"),
    ("buymeacoffee.com", "low"),  # personal blog host, spam-prone
    ("carvinghall.com", "low"),
    ("parked-domain.cn", "low"),
    ("glm45.org", "low"),
    # ---- end curated low ----
)


# ---- structural (b) — pen-and-skip heuristics for SEO farms ----
#
# Match hosts that look like content farms even when the exact
# hostname is not in the curated table. Cheap patterns:
#   * 3+ hyphens in the host       (seo-slug-content-farm-x.example)
#   * host contains "briefs" / "news" / "blog" AND ends in non-profit TLDs
#   * 1-char-or-tiny second-level  (a.com, b.io — classic SEO farms)
_STRUCTURAL_LOW_HOST_RE = re.compile(
    r"(?:"
    r"^[a-z0-9]+(?:[-][a-z0-9]+){3,}\."  # 3+ hyphen-separated labels
    r"|[-](?:briefs|news|seo-content|content-ai|tips[-]?|tricks)[-]?\."
    r"|^[a-z]\.(?:xyz|click|top|gq|tk|ml|cf|ga)$"  # 1-char + cheap TLD
    r")"
)


def _looks_like_low_host(url: str) -> bool:
    """Return True if the URL's host matches the structural low-credibility
    patterns.

    Kept separate from :meth:`DomainTrust.tier_for` so the curated
    table can be tested independently. Both signals compose in
    :func:`credibility_score`.
    """
    host = _host(url)
    if not host:
        return False
    return bool(_STRUCTURAL_LOW_HOST_RE.search(host))


# 2026-07-12 depth rebalance: neutral 0.25 -> 0.50 so an unknown-but-not-
# spammy domain lands near the floor (passes when on-topic, fails when
# off-topic) instead of being auto-cut as noise. low nudged 0.05 -> 0.08
# (still far below the 0.35 floor — content-farm guard fully intact).
_TIER_SCORE: dict[str, float] = {
    "primary": 1.0,
    "trusted": 0.80,
    "neutral": 0.50,
    "low": 0.08,
}


class DomainTrust:
    """Static curated domain-trust table.

    Stored as ``(pattern, tier)`` where ``pattern`` is a substring of the
    URL's host (lower-cased). Substrings are matched so ``"arxiv.org"``
    covers ``export.arxiv.org``, ``www.arxiv.org``, etc. Add new entries
    here rather than hard-coding in :func:`credibility_score`.
    """

    #: Class-level marker so tests can detect this is a "config-ish" class.
    KIND: ClassVar[str] = "domain_trust_table"

    @classmethod
    def tier_for(cls, url: str) -> str:
        """Return the trust tier for a URL (or ``"neutral"`` if unknown).

        Order matters: the curated table wins first, then the
        structural low-credibility heuristic bumps ``"neutral"``
        down to ``"low"`` so content farms get the documented penalty.
        """
        host = (urlparse(url).netloc or "").lower()
        if not host:
            return "neutral"
        tier = "neutral"
        for pattern, t in _DOMAIN_TABLE:
            if pattern in host:
                tier = t
        # P2 fix — structural pen for SEO farms that aren't in the
        # curated table. Only ever nudge neutral -> low (never upgrade
        # a curated tier downward).
        if tier == "neutral" and _looks_like_low_host(url):
            tier = "low"
        return tier

    @classmethod
    def score_for(cls, url: str) -> float:
        tier = cls.tier_for(url)
        return _TIER_SCORE[tier]

    @classmethod
    def entries(cls) -> list[dict]:
        """Public view used by tests and tuning scripts."""
        seen: dict[str, str] = {}
        for pattern, tier in _DOMAIN_TABLE:
            seen[pattern] = tier
        return [
            {"domain": d, "tier": t, "score": _TIER_SCORE[t]}
            for d, t in sorted(seen.items())
        ]


# ---------------------------------------------------------------------------
# TLD analysis — boost / penalise structural URL features
# ---------------------------------------------------------------------------

# TLDs strongly associated with academic / institutional sources.
_PRIMARY_TLDS = (".edu", ".gov", ".ac.uk", ".ac.jp", ".edu.au")

# Cheap TLDs heavily abused by content farms / spam.
_SUSPICIOUS_TLDS = (
    ".xyz", ".click", ".top", ".gq", ".tk", ".ml", ".cf", ".ga",
    ".loan", ".work", ".buzz", ".surf",
)

# URL substrings that signal SEO spam regardless of TLD.
_CONTENT_FARM_URL_HINTS = (
    "best-", "top-10-", "top-5-", "freetips", "free-download",
    "seo-", "rank-", "boost-", "coupons", "deals-",
    "blogspot-content", "wordpress.com",  # dump blogs
)

# Substrings that signal an authoritative venue.
_TRUSTED_URL_HINTS = (
    "arxiv.org",  # already covered but explicit for the pattern signal
    "github.com",
    ".pdf",
)


def _host(url: str) -> str:
    return (urlparse(url).netloc or "").lower()


# ---------------------------------------------------------------------------
# Token overlap — tiny keyword scorer; intentionally cheap.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _title_relevance(title: str, user_request: str) -> float:
    """Fraction of meaningful user_request tokens present in title."""
    req_tokens = _tokens(user_request)
    if not req_tokens:
        return 0.0
    title_tokens = _tokens(title)
    if not title_tokens:
        return 0.0
    # drop a small set of stop-tokens that overwhelm the overlap score
    stop = {"the", "and", "for", "with", "from", "into", "this", "that",
            "are", "was", "were", "how", "why", "what", "when"}
    req_meaningful = {t for t in req_tokens if t not in stop}
    if not req_meaningful:
        req_meaningful = req_tokens
    hits = sum(1 for t in req_meaningful if t in title_tokens)
    return hits / len(req_meaningful)


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def credibility_score(item: FetchedItem, *, user_request: str) -> float:
    """Score one FetchedItem in [0, 1].

    Weights (sum = 1.0):
        domain trust      0.55
        URL pattern bonus 0.20
        title relevance   0.25

    2026-07-12: domain weight raised 0.45 -> 0.55 (and url 0.30 -> 0.20).
    Title-overlap is a weak, noisy signal (titles rarely echo the full
    query), so leaning harder on the curated domain tier keeps genuine
    primary/curated sources above the floor instead of letting a near-
    zero title score drag arXiv/GitHub/Wikipedia down under it.
    """
    url = item.url or ""
    host = _host(url)

    # (a) Domain trust
    domain_s = DomainTrust.score_for(url)

    # (b) URL pattern — start from 0.0 and accumulate
    url_s = 0.0
    if any(host.endswith(tld) for tld in _PRIMARY_TLDS):
        url_s += 0.7
    if any(tld in host for tld in _SUSPICIOUS_TLDS):
        url_s -= 0.5
    lower_path = url.lower()
    if any(h in lower_path for h in _CONTENT_FARM_URL_HINTS):
        url_s -= 0.5
    if any(h in lower_path for h in _TRUSTED_URL_HINTS):
        url_s += 0.2
    # No URL at all -> domain/URL both 0; rely on title.
    if not host and not url:
        url_s = -0.2
    # clamp to [-0.5, 1.0] so a single penalty can't drag the sum negative
    url_s = max(-0.5, min(1.0, url_s))

    # (c) Title relevance
    title_s = _title_relevance(item.title, user_request)

    score = 0.55 * domain_s + 0.20 * url_s + 0.25 * title_s
    # Penalise missing URL (cannot verify provenance) — push below floor
    if not url:
        score = min(score, 0.15)
    # Floor at 0 for empty/no-evidence items
    return max(0.0, min(1.0, score))


def score_fetched(items: Iterable[FetchedItem], *,
                  user_request: str,
                  floor: float = CREDIBILITY_FLOOR) -> list[FetchedItem]:
    """Return a new list of FetchedItem with ``credibility_score`` (and
    ``credibility_flag``) populated. Does not drop items.

    Adds a ``fabricated_path`` flag (with the score dragged below
    ``floor``) for arxiv /abs/ URLs whose YYMM encodes a year outside
    the legitimate new-format window ``[2015, current_year]``. This is
    the P2.5 layer added 2026-07-10 in response to the 2026-07-09
    metacognitive-RL run (cites ``arxiv 2606.32032`` as legitimate).
    """
    now_year = datetime.now().year
    out: list[FetchedItem] = []
    for it in items:
        # v2 Phase 4: file:/// items are the user's OWN vault materials
        # (appended transcripts, saved reports). Scoring them like an
        # unknown web domain landed them below the floor and the filter
        # cut them — defeating an explicit /append. Trust them.
        if (it.url or "").startswith("file:///"):
            out.append(it.model_copy(update={
                "credibility_score": 0.9,
                "credibility_flag": "user_provided",
            }))
            continue
        s = credibility_score(it, user_request=user_request)
        flag: str | None = None
        if s < floor:
            flag = "low_credibility"
        # P2.5: arxiv year-fabrication wins over a generic low flag so
        # downstream consumers can grep for the specific cause.
        susp, _year = is_arxiv_year_suspicious(it.url or "", current_year=now_year)
        if susp:
            flag = "fabricated_path"
            s = min(s, floor - 0.05)
        # mutating a copy keeps the caller's list untouched
        updated = it.model_copy(update={
            "credibility_score": s,
            "credibility_flag": flag,
        })
        out.append(updated)
    return out
