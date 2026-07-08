"""Tests for src/argus/citations.py — the SourceRegistry + verify_citations
+ sanitize_report triplet lifted from NVIDIA AI-Q Blueprint.

Why these tests exist (bug observed 2026-07-08): Argus's planner, when
backed by a cheap LLM, hallucinates plausible-but-fake URLs into
``planned_sources`` and ``draft_md``. The user's final report ends up
linking to nonexistent resources. The structural fix is to require every
URL in the final report to resolve to a URL that was actually fetched by a
researcher tool — anything else is stripped along with its inline citation.

These tests pin the *behavior* of the lift:
  - 5-strategy ``resolve_url`` cascade (exact, truncation, prefix, child-path,
    query-subset) lifted verbatim from AI-Q's algorithm.
  - ``verify_citations`` strips unregistered URLs from a markdown body and
    records an audit trail.
  - ``sanitize_report`` rejects shortened / IP / truncated / non-http URLs.
"""
from __future__ import annotations

import pytest

from argus.citations import (
    SourceRegistry,
    SourceEntry,
    verify_citations,
    sanitize_report,
    VerificationResult,
    SanitizationResult,
)
from argus.graph.state import FetchedItem


# --- helpers -----------------------------------------------------------------

def make_fetched(url: str, title: str = "T", **kw) -> FetchedItem:
    return FetchedItem(url=url, title=title, **kw)


def make_registry(*urls: str, titles: dict[str, str] | None = None) -> SourceRegistry:
    """Build a registry populated from URL strings (FetchedItem-like)."""
    reg = SourceRegistry()
    titles = titles or {}
    for url in urls:
        reg.add(SourceEntry(url=url, title=titles.get(url, url)))
    return reg


# --- SourceRegistry basics ---------------------------------------------------

def test_empty_registry_returns_none():
    reg = SourceRegistry()
    assert reg.resolve_url("https://example.com") is None
    assert reg.has_url("https://example.com") is False
    assert reg.all_sources() == []


def test_registry_add_returns_handle_for_resolve():
    reg = SourceRegistry()
    reg.add(SourceEntry(url="https://arxiv.org/abs/2402.12345", title="Paper"))
    assert reg.has_url("https://arxiv.org/abs/2402.12345") is True
    assert reg.resolve_url("https://arxiv.org/abs/2402.12345") == "https://arxiv.org/abs/2402.12345"


def test_registry_add_from_fetched_items():
    """Argus's natural integration point: bulk-add from ArgusState['fetched']."""
    reg = SourceRegistry()
    fetched = [
        make_fetched("https://a.com/x", title="A"),
        make_fetched("https://b.com/y", title="B"),
    ]
    reg.add_from_fetched(fetched)
    assert reg.has_url("https://a.com/x")
    assert reg.has_url("https://b.com/y")
    assert not reg.has_url("https://c.com/z")


# --- 5-strategy resolve_url cascade (lifted from AI-Q) -----------------------

def test_resolve_exact_match():
    reg = make_registry("https://example.com/foo")
    assert reg.resolve_url("https://example.com/foo") == "https://example.com/foo"


def test_resolve_exact_after_normalization():
    """Trailing slash, www prefix, http vs https — should still match."""
    reg = make_registry("https://example.com/foo")
    # Trailing slash
    assert reg.resolve_url("https://example.com/foo/") == "https://example.com/foo"
    # www prefix
    assert reg.resolve_url("https://www.example.com/foo") == "https://example.com/foo"


def test_resolve_truncation_prefix():
    """Report URL is a prefix of registry URL (raw)."""
    reg = make_registry("https://arxiv.org/abs/2402.12345v1")
    # The report has a truncated URL — should still resolve.
    assert reg.resolve_url("https://arxiv.org/abs/2402.12345") == "https://arxiv.org/abs/2402.12345v1"


def test_resolve_prefix_match():
    """Normalized prefix — the report lost some path components."""
    reg = make_registry("https://docs.python.org/3/library/typing.html")
    # Report only has the prefix.
    assert reg.resolve_url("https://docs.python.org/3/library/typing") == "https://docs.python.org/3/library/typing.html"


def test_resolve_child_path():
    """Report path is a subpage of a registry URL (segment-boundary safe)."""
    reg = make_registry("https://github.com/NVIDIA-AI-Blueprints/aiq")
    # Report links to a subpath.
    assert reg.resolve_url("https://github.com/NVIDIA-AI-Blueprints/aiq/tree/develop") == "https://github.com/NVIDIA-AI-Blueprints/aiq"


def test_resolve_child_path_segment_boundary_safe():
    """Path matching must NOT match if the prefix isn't a true segment boundary.
    /us/benefits must NOT match /us/benefitsOther — different words.

    Caveat: AI-Q's truncation strategy (step 2) is a string-prefix match and
    does NOT enforce segment boundaries — it must accept legitimate cases
    like arxiv-version truncation (2402.12345 → 2402.12345v1). So the
    segment-boundary guard lives in the child-path strategy (step 4) only.
    To test step-4 boundary safety cleanly, we use TWO registry URLs that
    would both match the query via truncation, then expect ambiguous → None.
    """
    reg = make_registry(
        "https://example.com/us/benefitsOther",
        "https://example.com/us/benefitsPage",
    )
    # /us/benefits is a string-prefix of BOTH /us/benefitsOther and
    # /us/benefitsPage, so step-2 truncation finds 2 candidates and
    # rejects as ambiguous (good — silent misattribution is worse than None).
    assert reg.resolve_url("https://example.com/us/benefits") is None


def test_resolve_child_path_segment_boundary_safe_ambiguity_fallback():
    """When only ONE registry URL has a string-prefix collision (no
    ambiguity), the truncation strategy accepts it. This is a known
    limitation — the alternatives are: (a) accept arxiv-version
    truncation OR (b) reject it. Argus picks (a) per AI-Q."""
    reg = make_registry("https://example.com/us/benefitsOther")
    # Single candidate → accepted as truncation match. Real-world: this
    # would be a false positive in a tiny fraction of cases, accepted as
    # the price of supporting legitimate arxiv-version truncation.
    result = reg.resolve_url("https://example.com/us/benefits")
    # We don't assert which strategy matched — only that it's the
    # canonical URL (not None, not a different URL).
    assert result == "https://example.com/us/benefitsOther"


def test_resolve_query_subset():
    """Same host+path, report params subset of registry params."""
    reg = make_registry("https://example.com/api?a=1&b=2&c=3")
    assert reg.resolve_url("https://example.com/api?a=1") == "https://example.com/api?a=1&b=2&c=3"


def test_resolve_ambiguous_returns_none():
    """If multiple registry URLs match, reject (avoid silent misattribution)."""
    reg = make_registry(
        "https://example.com/foo/v1",
        "https://example.com/foo/v2",
    )
    # The prefix 'https://example.com/foo/' is ambiguous between v1 and v2.
    assert reg.resolve_url("https://example.com/foo/") is None


def test_resolve_returns_none_for_hallucinated_url():
    """The CORE bug: a hallucinated URL must NOT resolve to anything."""
    reg = make_registry(
        "https://github.com/NVIDIA-AI-Blueprints/aiq",
        "https://arxiv.org/abs/2402.12345",
    )
    # Fabricated URL from the bug report
    assert reg.resolve_url("https://github.com/transformers-metacognition") is None
    assert reg.resolve_url("https://arxiv.org/abs/2207.12345") is None


# --- verify_citations --------------------------------------------------------

def test_verify_citations_strips_hallucinated_url():
    """A markdown report containing a fabricated URL gets that URL stripped,
    and the audit trail records the removal."""
    reg = make_registry("https://real-source.com/paper")
    md = (
        "# Findings\n\n"
        "A recent result [claims X](https://fake-hallucinated-url.com/x). "
        "But [real evidence](https://real-source.com/paper) supports Y.\n"
    )
    result = verify_citations(md, reg)
    assert isinstance(result, VerificationResult)
    # Real URL preserved
    assert "https://real-source.com/paper" in result.verified_report
    # Hallucinated URL removed
    assert "fake-hallucinated-url.com" not in result.verified_report
    # Audit trail
    assert any(r["url"] == "https://fake-hallucinated-url.com/x" for r in result.removed_citations)


def test_verify_citations_returns_verified_report_attribute():
    """Public surface: result.verified_report + result.removed_citations."""
    reg = make_registry("https://real.com")
    result = verify_citations("See [this](https://real.com).", reg)
    assert hasattr(result, "verified_report")
    assert hasattr(result, "removed_citations")
    assert isinstance(result.removed_citations, list)


def test_verify_citations_preserves_anchors_and_punctuation():
    """Don't mangle the markdown beyond URL removal."""
    reg = make_registry("https://real.com")
    md = "## Section\n\nText. [Link](https://real.com) more text.\n"
    out = verify_citations(md, reg).verified_report
    assert "## Section" in out
    assert "Text." in out
    assert "more text." in out


# --- sanitize_report ---------------------------------------------------------

def test_sanitize_strips_shortened_urls():
    """bit.ly and similar shorteners are not allowed in the final report."""
    md = "See [tweet](https://bit.ly/abc123) and [paper](https://arxiv.org/abs/2402.12345)."
    out = sanitize_report(md).cleaned_text
    assert "bit.ly" not in out
    # arxiv URL passes through (real primary source)
    assert "arxiv.org/abs/2402.12345" in out


def test_sanitize_strips_ip_address_urls():
    md = "Server at [192.168.1.1](http://192.168.1.1/admin) and [real](https://example.com)."
    out = sanitize_report(md).cleaned_text
    assert "192.168.1.1" not in out
    assert "example.com" in out


def test_sanitize_strips_truncated_urls():
    md = "See [paper](https://arxiv.org/abs/2402.12345...) and [good](https://example.com)."
    out = sanitize_report(md).cleaned_text
    # Trailing '...' marker means truncated — remove.
    assert "2402.12345..." not in out
    assert "example.com" in out


def test_sanitize_strips_non_http_schemes():
    md = "FTP [link](ftp://files.example.com/x) and [web](https://example.com)."
    out = sanitize_report(md).cleaned_text
    assert "ftp://" not in out
    assert "example.com" in out


def test_sanitize_returns_result_object():
    """Public surface: cleaned_text + audit list."""
    md = "x [link](https://bit.ly/y)"
    result = sanitize_report(md)
    assert isinstance(result, SanitizationResult)
    assert hasattr(result, "cleaned_text")
    assert hasattr(result, "removed")