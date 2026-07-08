"""Argus credibility scoring — DomainTrust static list + per-URL scoring.

Sits between fetcher_node and filter_node. Assigns every FetchedItem a
``credibility_score`` in [0, 1] based on three signals:

  (a) Domain trust      — curated DomainTrust list + TLD analysis
  (b) URL pattern       — arxiv / .edu / .gov boost, content-farm penalty
  (c) Title relevance   — token overlap with the user_request

Items below the 0.4 floor are tagged (``credibility_flag = "low_credibility"``)
but NOT dropped — the downstream ``filter_node`` still does its own
``keep_topN`` so dropping here would just double-filter.

Reference: HANDOFF-RESEARCH-2026-07-08.md §4 (lines 110-135) and §6 row #2.
"""
from __future__ import annotations

import re
from typing import ClassVar, Iterable
from urllib.parse import urlparse

from .state import FetchedItem


CREDIBILITY_FLOOR = 0.4


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
    # TLD-based primaries — substring match means ".edu" in host
    # (e.g. "cs.stanford.edu") catches the whole .edu namespace.
    (".edu", "primary"),
    (".gov", "primary"),
    (".ac.uk", "primary"),
    (".ac.jp", "primary"),
    # ---- trusted blogs / curated ------------------------------------
    ("github.com", "trusted"),
    ("huggingface.co", "trusted"),
    ("openai.com", "trusted"),
    ("anthropic.com", "trusted"),
    ("deepmind.google", "trusted"),
    ("research.google", "trusted"),
    ("ai.meta.com", "trusted"),
    ("bair.berkeley.edu", "trusted"),
    ("distill.pub", "trusted"),
    ("towardsdatascience.com", "trusted"),
    ("medium.com", "trusted"),
    ("substack.com", "trusted"),
)

_TIER_SCORE: dict[str, float] = {
    "primary": 1.0,
    "trusted": 0.80,
    "neutral": 0.25,
    "low": 0.05,
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
        """Return the trust tier for a URL (or ``"neutral"`` if unknown)."""
        host = (urlparse(url).netloc or "").lower()
        if not host:
            return "neutral"
        tier = "neutral"
        for pattern, t in _DOMAIN_TABLE:
            if pattern in host:
                tier = t
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

    score = 0.45 * domain_s + 0.30 * url_s + 0.25 * title_s
    # Penalise missing URL (cannot verify provenance) — push below floor
    if not url:
        score = min(score, 0.15)
    # Floor at 0 for empty/no-evidence items
    return max(0.0, min(1.0, score))


def score_fetched(items: Iterable[FetchedItem], *,
                  user_request: str,
                  floor: float = CREDIBILITY_FLOOR) -> list[FetchedItem]:
    """Return a new list of FetchedItem with ``credibility_score`` (and
    ``credibility_flag``) populated. Does not drop items."""
    out: list[FetchedItem] = []
    for it in items:
        s = credibility_score(it, user_request=user_request)
        # mutating a copy keeps the caller's list untouched
        updated = it.model_copy(update={
            "credibility_score": s,
            "credibility_flag": "low_credibility" if s < floor else None,
        })
        out.append(updated)
    return out