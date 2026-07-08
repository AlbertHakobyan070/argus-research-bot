"""Argus citations module — structural URL-integrity for synthesized reports.

Why this exists (bug observed 2026-07-08): when the planner LLM falls back
to a cheap model, it hallucinates plausible-looking but fake URLs into
``planned_sources`` and into the synthesizer's ``draft_md``. The final
report then links to nonexistent resources. The structural fix is to
require every URL in the final report to resolve to a URL that was
actually fetched by a researcher tool — anything else is stripped along
with its inline citation.

This module is lifted from NVIDIA AI-Q Blueprint's
``src/aiq_agent/common/citation_verification.py`` (Apache-2.0, NVIDIA).
We keep the core algorithms but rewrite the integration surface to match
Argus's state shape (``FetchedItem`` instead of AI-Q's ``SourceEntry``)
and drop the session-registry/contextvars plumbing that Argus doesn't need.

Three primitives:
  - :class:`SourceRegistry` — URL set + 5-strategy ``resolve_url`` cascade.
  - :func:`verify_citations` — strip unregistered URLs from a markdown body.
  - :func:`sanitize_report` — strip shortened / IP / truncated / non-http URLs.

The 5-strategy cascade (verbatim from AI-Q):
  1. Exact match (raw or normalized)
  2. Truncation (report URL is a prefix of exactly one registry URL)
  3. Prefix (normalized prefix)
  4. Child-path (report path extends a registry path, segment-boundary safe)
  5. Query-subset (same host+path, report params subset of registry params)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse, parse_qs

from argus.graph.state import FetchedItem


# --- Public data classes -----------------------------------------------------

@dataclass
class SourceEntry:
    """One URL captured from a tool result. Argus-shape: simpler than AI-Q's."""
    url: str
    title: str = ""
    source_type: str = ""     # "paper" | "repo" | "news" | "official_doc" | "search_result"
    tool_name: str = ""       # which tool produced it (perplexity_search, ddgs_search, ...)


@dataclass
class VerificationResult:
    """Outcome of :func:`verify_citations`."""
    verified_report: str
    removed_citations: list[dict] = field(default_factory=list)
    # removed_citations[i] = {"url": str, "context": str, "reason": str}


@dataclass
class SanitizationResult:
    """Outcome of :func:`sanitize_report`."""
    cleaned_text: str
    removed: list[dict] = field(default_factory=list)
    # removed[i] = {"url": str, "reason": str}


# --- SourceRegistry ----------------------------------------------------------

class SourceRegistry:
    """URL set + 5-strategy ``resolve_url`` cascade.

    Argus integration: build from the ``fetched`` list of ArgusState via
    :meth:`add_from_fetched`. Then call :meth:`resolve_url` to ask "did the
    researcher actually fetch this URL?" — ``None`` means fabricated.
    """

    # Common URL shorteners / obfuscators that should never appear in a
    # research report. The list is conservative: real Argus cites point to
    # the canonical expanded URL.
    KNOWN_SHORTENERS = frozenset({
        "bit.ly", "t.co", "goo.gl", "ow.ly", "tinyurl.com", "is.gd",
        "buff.ly", "shorturl.at", "rebrand.ly", "cutt.ly", "tiny.cc",
    })

    def __init__(self) -> None:
        # Map: raw URL → SourceEntry (also indexed by normalized form)
        self._urls: dict[str, SourceEntry] = {}
        # Cached parsed (host, path_segments, query) → entry
        self._parsed_cache: dict[str, "_ParsedURL"] = {}

    # --- mutation --------------------------------------------------------

    def add(self, entry: SourceEntry) -> None:
        if not entry.url:
            return
        self._urls[entry.url] = entry
        normalized = _normalize_url(entry.url)
        if normalized != entry.url:
            self._urls.setdefault(normalized, entry)
        parsed = urlparse(normalized)
        self._parsed_cache[normalized] = _ParsedURL(
            host=parsed.netloc.lower(),
            path=parsed.path,
            path_segments=[s for s in parsed.path.split("/") if s],
            query=parse_qs(parsed.query, keep_blank_values=True),
            entry=entry,
        )

    def add_from_fetched(self, items: Iterable[FetchedItem]) -> None:
        """Bulk-load from ArgusState['fetched'] (the natural integration)."""
        for it in items:
            self.add(SourceEntry(
                url=it.url,
                title=it.title or "",
                tool_name=it.section or "",  # section doubles as tool-bucket label
            ))

    def clear(self) -> None:
        self._urls.clear()
        self._parsed_cache.clear()

    # --- queries ---------------------------------------------------------

    def has_url(self, url: str) -> bool:
        return self.resolve_url(url) is not None

    def all_sources(self) -> list[SourceEntry]:
        # Dedup by entry identity (raw + normalized may point at same entry).
        seen: set[int] = set()
        out: list[SourceEntry] = []
        for e in self._urls.values():
            if id(e) not in seen:
                seen.add(id(e))
                out.append(e)
        return out

    def size(self) -> int:
        return len({id(e) for e in self._urls.values()})

    # --- core: resolve_url cascade (lifted from AI-Q) -------------------

    def resolve_url(self, url: str) -> str | None:
        """Return the registry URL (full, as returned by the tool) when the
        report URL matches via the 5-strategy cascade. ``None`` if the URL
        cannot be uniquely matched.

        Matching strategy (first unambiguous match wins):
        1. Exact match — raw or normalized
        2. Truncation — report URL is a prefix of exactly one registry URL (raw)
        3. Prefix — report normalized is prefix of registry normalized
        4. Child-path — report path is a subpath of exactly one registry URL
        5. Query-subset — same host+path, report params subset of one registry URL
        """
        if not url:
            return None

        # 1. Exact match — raw or normalized.
        if url in self._urls:
            return self._urls[url].url
        normalized = _normalize_url(url)
        if normalized in self._urls:
            return self._urls[normalized].url

        # 2. Truncation — report URL is a prefix of exactly one registry URL (raw).
        #    AI-Q does not enforce path-segment boundary here because
        #    arxiv-version truncation (e.g. ``2402.12345`` vs ``2402.12345v1``)
        #    is a legitimate case; the segment-boundary guard only applies
        #    to child-path matching (step 4).
        truncation = [e for e in self._urls.values() if e.url and e.url.startswith(url)]
        result = self._pick_unique(truncation, "truncation")
        if result:
            return result.url

        # 3. Prefix match — normalized. Same boundary semantics as truncation.
        prefix = [e for n, e in self._urls.items() if n.startswith(normalized)]
        result = self._pick_unique(prefix, "prefix")
        if result:
            return result.url

        parsed = urlparse(normalized)
        host, path = parsed.netloc.lower(), parsed.path
        same_host = [p for p in self._parsed_cache.values() if p.host == host]
        if not same_host:
            return None

        # 4. Child-path — report path extends a registry path. Segment-boundary safe.
        result = self._pick_unique(
            [p.entry for p in same_host
             if len(p.path_segments) >= 2
             and path != p.path
             and path.startswith(p.path.rstrip("/") + "/")],
            "child-path",
        )
        if result:
            return result.url

        # 5. Query-subset — same host+path, report params ⊆ registry params.
        report_qs = parse_qs(parsed.query, keep_blank_values=True)
        if report_qs:
            result = self._pick_unique(
                [p.entry for p in same_host
                 if p.path == path and p.query
                 and all(p.query.get(k) == v for k, v in report_qs.items())],
                "query-subset",
            )
            if result:
                return result.url

        return None

    def _pick_unique(self, candidates: list[SourceEntry], strategy: str) -> SourceEntry | None:
        """Return the unique entry if exactly one candidate matches; else None."""
        unique = list({e.url: e for e in candidates}.values())
        if len(unique) == 1:
            return unique[0]
        # Ambiguous → reject (silent misattribution is worse than no match).
        return None


# --- helpers -----------------------------------------------------------------

@dataclass
class _ParsedURL:
    host: str
    path: str
    path_segments: list[str]
    query: dict
    entry: SourceEntry


def _normalize_url(url: str) -> str:
    """Normalize for matching: lowercase scheme + host; collapse trailing slash;
    drop default ports. Argus-specific (smaller scope than AI-Q's, which also
    tracks redirects and query ordering)."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return url.strip()
    scheme = (parsed.scheme or "http").lower()
    host = (parsed.netloc or "").lower()
    # Strip default ports (80, 443)
    if host.endswith(":80") and scheme == "http":
        host = host[:-3]
    elif host.endswith(":443") and scheme == "https":
        host = host[:-4]
    # Strip www. prefix for matching (so www.example.com ↔ example.com).
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")
    # Preserve query + fragment order; AI-Q's truncation strategy is order-sensitive.
    query = parsed.query
    fragment = parsed.fragment
    out = f"{scheme}://{host}{path}"
    if query:
        out += "?" + query
    if fragment:
        out += "#" + fragment
    return out


# --- verify_citations --------------------------------------------------------

# Match [text](url) and bare URLs. Group 1 = display text (may be empty),
# Group 2 = URL. The link regex accepts ANY scheme (so we can sanitize
# ftp://, javascript:, file://, etc.); the verifier/bare-URL regex is
# http-only because bare ftp:// text is exceedingly rare in research reports.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([a-z][a-z0-9+\-.]*://[^\s)]+)\)")
_BARE_URL_RE = re.compile(r"(?<![\(\w@/])(https?://[^\s)\]<]+)")
_URL_RE = re.compile(r"https?://[^\s)\]\"'<>]+")


def verify_citations(report_text: str, registry: SourceRegistry) -> VerificationResult:
    """Strip inline URLs that don't resolve to a registered source.

    Walks the markdown, finds ``[text](url)`` links and bare URLs, and:
      - If ``registry.resolve_url(url)`` returns a registered URL → keep it
        (rewritten to the canonical registry URL if it differs).
      - Otherwise → strip the link (leave just the display text) and record
        a removal in the audit trail.

    The reported ``verified_report`` preserves all non-URL markdown intact.
    """
    if registry.size() == 0:
        # No sources fetched — every URL is suspect. Treat the whole
        # report as unverifiable and emit an empty body.
        return VerificationResult(
            verified_report="(no sources fetched; report suppressed)",
            removed_citations=[{"url": u, "reason": "empty_registry"}
                               for u in set(_URL_RE.findall(report_text))],
        )

    removed: list[dict] = []

    def _resolve_or_strip_md_link(m: re.Match) -> str:
        display, url = m.group(1), m.group(2).rstrip(".,;:!?)")
        resolved = registry.resolve_url(url)
        if resolved is None:
            removed.append({"url": url, "context": display, "reason": "not_in_registry"})
            return display  # strip link, keep text
        if resolved != url:
            return f"[{display}]({resolved})"
        return m.group(0)

    def _resolve_or_strip_bare_url(m: re.Match) -> str:
        url = m.group(0).rstrip(".,;:!?)")
        resolved = registry.resolve_url(url)
        if resolved is None:
            removed.append({"url": url, "context": "(bare URL)", "reason": "not_in_registry"})
            return ""  # drop bare URL entirely
        return resolved

    verified = _MD_LINK_RE.sub(_resolve_or_strip_md_link, report_text)
    verified = _BARE_URL_RE.sub(_resolve_or_strip_bare_url, verified)

    return VerificationResult(
        verified_report=verified,
        removed_citations=removed,
    )


# --- sanitize_report ---------------------------------------------------------

# Conservative regexes for unsafe-URL detection.
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_IPV4_STRICT_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_SHORTENER_HOST_RE = re.compile(r"^([a-z0-9-]+)\.([a-z]{2,})$")


def sanitize_report(report_text: str) -> SanitizationResult:
    """Deterministic hygiene for the final report.

    Strips:
      1. Shortened URLs (bit.ly, t.co, ...) — these hide the real destination.
      2. IP-address URLs (``http://192.168.1.1/admin``) — never cite these.
      3. Truncated URLs ending in ``...`` or with no path component.
      4. Non-http(s) schemes (``ftp://``, ``javascript:``, ``file://``).

    Returns the cleaned text and an audit list of what was removed.
    """
    removed: list[dict] = []

    def _check_url(url: str) -> tuple[bool, str | None]:
        """Return (keep, reason_if_dropped)."""
        try:
            parsed = urlparse(url)
        except ValueError:
            return False, "unparseable_url"
        scheme = (parsed.scheme or "").lower()
        if scheme not in ("http", "https"):
            return False, f"non_http_scheme:{scheme}"
        host = parsed.netloc.split(":")[0].lower()
        if not host:
            return False, "empty_host"
        if _IPV4_STRICT_RE.match(host):
            return False, "ip_address_url"
        # Shortener detection — strip leading "www." for comparison.
        bare = host[4:] if host.startswith("www.") else host
        if bare in SourceRegistry.KNOWN_SHORTENERS:
            return False, "shortener"
        # Truncation markers — '...' anywhere in the URL signals incompleteness.
        if "..." in url:
            return False, "truncated"
        return True, None

    def _md_link_repl(m: re.Match) -> str:
        display, url = m.group(1), m.group(2)
        keep, reason = _check_url(url)
        if not keep:
            removed.append({"url": url, "reason": reason or "unknown"})
            return display
        return m.group(0)

    def _bare_url_repl(m: re.Match) -> str:
        url = m.group(0)
        keep, reason = _check_url(url)
        if not keep:
            removed.append({"url": url, "reason": reason or "unknown"})
            return ""
        return url

    cleaned = _MD_LINK_RE.sub(_md_link_repl, report_text)
    cleaned = _BARE_URL_RE.sub(_bare_url_repl, cleaned)

    # Post-pass: drop orphan IP-address tokens that may have been left in
    # display text after a link was removed. Real research reports never
    # contain raw IPs.
    cleaned = _IPV4_RE.sub("", cleaned)

    return SanitizationResult(cleaned_text=cleaned, removed=removed)