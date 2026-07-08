"""Hermetic tests for bench/scorer.py.

We don't run the full graph here (that's the bench's job). These tests
verify the scoring logic against fixed inputs so future refactors
don't drift the metrics.

What we test:
  - _extract_urls dedupes and preserves order (markdown + bare).
  - _length_target returns sane values for each mode.
  - score_one computes citation_integrity, domain_coverage, density.
  - aggregate rolls up per-query stats correctly.
  - render_markdown produces a parseable table.
  - Main CLI: reads raw.jsonl + queries.jsonl, writes score.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import scorer


# --- _extract_urls -----------------------------------------------------------

def test_extract_urls_handles_markdown_links():
    md = "See [paper](https://arxiv.org/abs/2402.12345) and [code](https://github.com/x/y)."
    urls = scorer._extract_urls(md)
    assert urls == ["https://arxiv.org/abs/2402.12345", "https://github.com/x/y"]


def test_extract_urls_handles_bare_urls():
    md = "Visit https://example.com and https://nvidia.com for details."
    urls = scorer._extract_urls(md)
    assert "https://example.com" in urls
    assert "https://nvidia.com" in urls


def test_extract_urls_dedupes_preserving_order():
    md = "[a](https://x.com) and bare https://x.com and [b](https://x.com)."
    urls = scorer._extract_urls(md)
    # First occurrence wins.
    assert urls.count("https://x.com") == 1
    assert urls[0] == "https://x.com"


def test_extract_urls_strips_trailing_punctuation():
    md = "See (https://example.com) and [link](https://example.com/path)."
    urls = scorer._extract_urls(md)
    assert all(not u.endswith(",.;:!?") for u in urls)


def test_extract_urls_empty_input():
    assert scorer._extract_urls("") == []
    assert scorer._extract_urls("no urls here") == []


# --- _length_target ----------------------------------------------------------

@pytest.mark.parametrize("mode,expected", [
    ("quick", 800),
    ("short", 3000),
    ("medium", 8000),
    ("long", 16000),
    ("lecture", 32000),
    ("unknown", 5000),  # default
])
def test_length_target_per_mode(mode, expected):
    assert scorer._length_target(mode) == expected


# --- score_one ---------------------------------------------------------------

def _record(md: str = "", **kw) -> dict:
    """Build a minimal record for scoring."""
    base = {
        "id": "q001",
        "query": "anything",
        "status": "ok",
        "duration_s": 5.0,
        "n_sources": 3,
        "n_findings": 2,
        "report_md": md,
        "length": "short",
    }
    base.update(kw)
    return base


def test_score_one_zero_urls():
    """Report with no URLs: citation_integrity = 0.0 (not NaN)."""
    s = scorer.score_one(_record("No links here."), ["github.com"])
    assert s["url_count"] == 0
    assert s["citation_integrity"] == 0.0
    assert s["domain_coverage"] == 0.0


def test_score_one_all_primary():
    """Report with all-arxiv URLs scores 1.0 citation_integrity."""
    md = "[a](https://arxiv.org/abs/2402.12345) [b](https://arxiv.org/abs/2401.99999)"
    s = scorer.score_one(_record(md), ["arxiv.org"])
    assert s["url_count"] == 2
    assert s["primary_url_count"] == 2
    assert s["citation_integrity"] == 1.0
    assert s["domain_coverage"] == 1.0


def test_score_one_mixed_primary_secondary():
    md = "[a](https://arxiv.org/abs/2402.12345) [b](https://example-blog.com/x)"
    s = scorer.score_one(_record(md), [])
    assert s["url_count"] == 2
    assert s["primary_url_count"] == 1
    assert s["citation_integrity"] == 0.5
    # No expected_domains -> coverage is None
    assert s["domain_coverage"] is None


def test_score_one_domain_coverage_partial():
    md = "[a](https://github.com/x/y)"
    s = scorer.score_one(_record(md), ["github.com", "arxiv.org", "nvidia.com"])
    assert s["domain_coverage"] == pytest.approx(1/3, abs=1e-3)


def test_score_one_length_within_target():
    """3k char report in short mode (target 3k, ±50%) passes length_ok."""
    md = "x" * 3500
    s = scorer.score_one(_record(md, length="short"), [])
    assert s["length_chars"] == 3500
    assert s["length_target"] == 3000
    assert s["length_ok"] is True


def test_score_one_length_outside_target():
    """100 char report in long mode (target 16k) fails length_ok."""
    md = "x" * 100
    s = scorer.score_one(_record(md, length="long"), [])
    assert s["length_ok"] is False


def test_score_one_citation_density():
    """5 URLs in a roughly 5k-char report ≈ 1.0 URL per 1k chars.

    Tolerance ±10% because the markdown link markup itself contributes to
    the character count, so a report with exactly N URLs and exactly
    1000*N other characters will measure slightly higher than N per 1k.
    """
    md = " ".join(f"[a{i}](https://example.com/{i})" for i in range(5)) + " " + "x" * 4500
    s = scorer.score_one(_record(md, length="short"), [])
    assert 0.9 < s["citation_density"] < 1.2


# --- aggregate ---------------------------------------------------------------

def test_aggregate_all_ok():
    recs = [
        _record("x" * 3000, id="q1", status="ok", duration_s=10),
        _record("x" * 3000, id="q2", status="ok", duration_s=20),
    ]
    per = [scorer.score_one(r, []) for r in recs]
    agg = scorer.aggregate(per)
    assert agg["n_total"] == 2
    assert agg["n_ok"] == 2
    assert agg["pass_rate"] == 1.0
    assert agg["mean_duration_s"] == 15.0


def test_aggregate_partial_failures():
    recs = [
        _record(id="q1", status="ok"),
        _record(id="q2", status="timeout"),
        _record(id="q3", status="ok"),
    ]
    per = [scorer.score_one(r, []) for r in recs]
    agg = scorer.aggregate(per)
    assert agg["n_ok"] == 2
    assert agg["pass_rate"] == pytest.approx(0.666, rel=1e-2)


def test_aggregate_all_failed():
    recs = [_record(id="q1", status="error"), _record(id="q2", status="timeout")]
    per = [scorer.score_one(r, []) for r in recs]
    agg = scorer.aggregate(per)
    assert agg["n_ok"] == 0
    assert agg["pass_rate"] == 0.0


# --- render_markdown ---------------------------------------------------------

def test_render_markdown_contains_table_headers():
    md = scorer.render_markdown(
        [{"id": "q1", "status": "ok", "duration_s": 5.0,
          "url_count": 3, "primary_url_count": 2,
          "citation_integrity": 0.667, "domain_coverage": 0.5,
          "length_chars": 3000, "length_target": 3000}],
        {"n_ok": 1, "n_total": 1, "pass_rate": 1.0,
         "mean_citation_integrity": 0.667, "mean_citation_density": 1.0,
         "mean_domain_coverage": 0.5, "mean_duration_s": 5.0,
         "mean_n_sources": 3, "mean_url_count": 3},
        "short",
    )
    assert "| ID | Status |" in md
    assert "Mean citation integrity" in md
    assert "67%" in md  # 0.667 formatted as 67%


def test_render_markdown_handles_none_coverage():
    md = scorer.render_markdown(
        [{"id": "q1", "status": "ok", "duration_s": 5.0,
          "url_count": 0, "primary_url_count": 0,
          "citation_integrity": 0.0, "domain_coverage": None,
          "length_chars": 0, "length_target": 3000}],
        {"n_ok": 1, "n_total": 1, "pass_rate": 1.0,
         "mean_citation_integrity": 0.0, "mean_citation_density": 0.0,
         "mean_domain_coverage": None, "mean_duration_s": 5.0,
         "mean_n_sources": 0, "mean_url_count": 0},
        "short",
    )
    assert "—" in md  # em-dash for None coverage


# --- main CLI: end-to-end ----------------------------------------------------

def test_main_end_to_end(tmp_path: Path):
    """Drive the scorer CLI with synthetic raw.jsonl and verify outputs."""
    # Write a raw.jsonl with two records.
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        json.dumps(_record("[a](https://arxiv.org/abs/2402.12345)", id="q001")) + "\n"
        + json.dumps(_record("No links.", id="q002")) + "\n",
        encoding="utf-8",
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        '{"id":"q001","query":"x","expected_domains":["arxiv.org"]}\n'
        '{"id":"q002","query":"y","expected_domains":["github.com"]}\n',
        encoding="utf-8",
    )
    out = tmp_path / "score.json"
    md_out = tmp_path / "score.md"

    # Run main() with sys.argv patching.
    import sys
    argv_backup = sys.argv
    try:
        sys.argv = [
            "scorer.py",
            "--raw", str(raw),
            "--queries", str(queries),
            "--out", str(out),
            "--md", str(md_out),
            "--length", "short",
        ]
        rc = scorer.main()
    finally:
        sys.argv = argv_backup
    assert rc == 0

    score_obj = json.loads(out.read_text(encoding="utf-8"))
    assert score_obj["length"] == "short"
    assert score_obj["aggregate"]["n_total"] == 2
    assert score_obj["aggregate"]["n_ok"] == 2

    md_text = md_out.read_text(encoding="utf-8")
    assert "Argus Bench" in md_text
    assert "q001" in md_text
    assert "q002" in md_text