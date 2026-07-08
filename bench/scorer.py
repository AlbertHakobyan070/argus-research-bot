"""Argus Deep Research Bench — scorer.

Reads results/raw.jsonl from the runner and computes per-query scores:

  - citation_integrity   (0.0–1.0): fraction of inline URLs in the report
                          pointing at known primary-source domains
                          (arxiv/github/.edu/.org/official docs).
                          Proxy for the new SourceRegistry's effectiveness.
  - url_count            (int): total inline URLs (markdown + bare).
  - primary_url_count    (int): URLs pointing at known primary domains.
  - citation_density     (float): URLs per 1000 chars of report.
  - domain_coverage      (float): fraction of expected_domains (from
                          queries.jsonl) actually present in the report.
  - length_ok            (bool): report length is within ±50% of expected
                          for the chosen length mode.

Output:
  - results/score.json  (one record per query + an aggregate summary)
  - results/score.md    (human-readable table for commit messages)

Usage:
    PYTHONPATH="" ./venv/Scripts/python.exe -m bench.scorer \\
        --raw bench/results/raw.jsonl \\
        --queries bench/queries.jsonl \\
        --out bench/results/score.json \\
        --md bench/results/score.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path


# Match [text](url) and bare URLs.
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_BARE_URL_RE = re.compile(r"(?<![\(\w@/])(https?://[^\s)\]<]+)")


# Domains that count as "primary" for citation-integrity scoring.
# Includes academic (.edu, arxiv), standards bodies (.org for many
# foundations), official documentation, and recognized engineering
# blogs. Anything outside this set is treated as secondary/unverified.
PRIMARY_DOMAINS = (
    "arxiv.org", "github.com", ".edu",
    "python.langchain.com", "langchain-ai.github.io",
    "postgresql.org", "duckduckgo.com", "nvidia.com",
    "huggingface.co", "openai.com", "anthropic.com",
    "docs.nvidia.com", "pytorch.org", "tensorflow.org",
    "kubernetes.io", "w3.org", "ietf.org",
    "acm.org", "ieee.org", "usenix.org",
    "mozilla.org", "developer.mozilla.org",
)


def _length_target(length: str) -> int:
    """Target character count for each length mode (approximate)."""
    return {
        "quick": 800,
        "short": 3000,
        "medium": 8000,
        "long": 16000,
        "lecture": 32000,
    }.get(length, 5000)


def _extract_urls(text: str) -> list[str]:
    """Extract all URLs from markdown text, deduped, in order."""
    out: list[str] = []
    seen: set[str] = set()
    for pat in (_MD_LINK_RE, _BARE_URL_RE):
        for m in pat.finditer(text):
            url = m.group(2 if pat is _MD_LINK_RE else 0).rstrip(".,;:!?)")
            if url not in seen:
                seen.add(url)
                out.append(url)
    return out


def score_one(record: dict, expected_domains: list[str]) -> dict:
    """Score a single query's result."""
    md = record.get("report_md") or ""
    urls = _extract_urls(md)
    n_urls = len(urls)

    primary_hits = sum(
        1 for u in urls if any(d in u for d in PRIMARY_DOMAINS)
    )
    citation_integrity = (primary_hits / n_urls) if n_urls else 0.0

    if expected_domains:
        hits = sum(1 for d in expected_domains if any(d in u for u in urls))
        domain_coverage = hits / len(expected_domains)
    else:
        domain_coverage = None

    target = _length_target(record.get("length", "short"))
    length_ok = abs(len(md) - target) <= 0.5 * target

    citation_density = (n_urls * 1000 / len(md)) if md else 0.0

    return {
        "id": record["id"],
        "status": record.get("status"),
        "duration_s": record.get("duration_s"),
        "n_sources": record.get("n_sources", 0),
        "n_findings": record.get("n_findings", 0),
        "url_count": n_urls,
        "primary_url_count": primary_hits,
        "citation_integrity": round(citation_integrity, 3),
        "citation_density": round(citation_density, 2),
        "domain_coverage": (round(domain_coverage, 3)
                            if domain_coverage is not None else None),
        "length_chars": len(md),
        "length_target": target,
        "length_ok": length_ok,
    }


def aggregate(per_query: list[dict]) -> dict:
    """Compute aggregate stats across all queries."""
    ok = [r for r in per_query if r["status"] == "ok"]
    n_total = len(per_query)
    n_ok = len(ok)
    if not ok:
        return {"n_total": n_total, "n_ok": 0, "pass_rate": 0.0}

    def _mean(key: str):
        vals = [r[key] for r in ok if r.get(key) is not None]
        return round(statistics.mean(vals), 3) if vals else None

    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "pass_rate": round(n_ok / n_total, 3),
        "mean_citation_integrity": _mean("citation_integrity"),
        "mean_citation_density": _mean("citation_density"),
        "mean_domain_coverage": _mean("domain_coverage"),
        "mean_duration_s": _mean("duration_s"),
        "mean_n_sources": _mean("n_sources"),
        "mean_url_count": _mean("url_count"),
    }


def _cov_label(agg: dict) -> str:
    if agg.get("mean_domain_coverage") is None:
        return "—"
    return f"{agg['mean_domain_coverage']:.0%}"


def render_markdown(per_query: list[dict], agg: dict, length: str) -> str:
    """Human-readable Markdown summary."""
    lines = [
        f"# Argus Bench — {length} mode",
        "",
        f"**{agg['n_ok']}/{agg['n_total']}** queries completed "
        f"({agg['pass_rate']*100:.0f}% pass rate)",
        "",
        "## Per-query",
        "",
        "| ID | Status | Duration | URLs | Primary | Cite-Integ | Coverage | Length |",
        "|----|--------|----------|------|---------|------------|----------|--------|",
    ]
    for r in per_query:
        cov = (f"{r['domain_coverage']:.0%}"
               if r.get("domain_coverage") is not None else "—")
        lines.append(
            f"| `{r['id']}` | {r['status']} | {r['duration_s']:.1f}s "
            f"| {r['url_count']} | {r['primary_url_count']} "
            f"| {r['citation_integrity']:.0%} | {cov} "
            f"| {r['length_chars']}/{r['length_target']} |"
        )
    lines.extend([
        "",
        "## Aggregate",
        "",
        f"- **Mean citation integrity:** {agg['mean_citation_integrity']:.0%}",
        f"- **Mean citation density:** {agg['mean_citation_density']:.2f} URLs/1k chars",
        f"- **Mean domain coverage:** {_cov_label(agg)}",
        f"- **Mean duration:** {agg['mean_duration_s']:.1f}s",
        f"- **Mean sources fetched:** {agg['mean_n_sources']:.1f}",
        f"- **Mean URLs in report:** {agg['mean_url_count']:.1f}",
        "",
        "_citation_integrity_ = fraction of inline URLs pointing at primary sources "
        "(arxiv/github/.edu/.org/official docs). _coverage_ = fraction of "
        "`expected_domains` actually cited. _length_ = chars vs target for mode.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Argus Deep Research Bench scorer.")
    parser.add_argument("--raw", required=True, type=Path,
                        help="Path to raw.jsonl from runner.py")
    parser.add_argument("--queries", required=True, type=Path,
                        help="Path to queries.jsonl (for expected_domains)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Path to write score.json")
    parser.add_argument("--md", type=Path, default=None,
                        help="Optional path to write score.md")
    parser.add_argument("--length", default="short",
                        help="Length mode label (for the markdown header)")
    args = parser.parse_args()

    queries_by_id: dict[str, dict] = {}
    with args.queries.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries_by_id[q["id"]] = q

    records = []
    with args.raw.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    per_query = []
    for rec in records:
        qid = rec["id"]
        q = queries_by_id.get(qid, {})
        per_query.append(score_one(rec, q.get("expected_domains", [])))

    agg = aggregate(per_query)
    out_obj = {"length": args.length, "aggregate": agg, "per_query": per_query}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"wrote {args.out}")

    if args.md:
        args.md.write_text(render_markdown(per_query, agg, args.length),
                           encoding="utf-8")
        print(f"wrote {args.md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())