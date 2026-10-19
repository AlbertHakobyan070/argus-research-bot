"""Argus live check — one real research run, graph-direct, then the acceptance table.

Replaces tests/manual_e2e.py (v4 spec §3 WP-G). The run is the bench's own
(``bench.runner._run_one``): the real graph on a fresh MemorySaver, the live
engines (resolved first with ``registry.active_snapshot()``) and live search,
``run_config`` with ``origin="live_check"``, auto-resume through the plan and
report gates, ``durability="sync"``, ``control=RunControl()``. It writes the
bench result layout ``<out>/<qid>/{state.json, report.md, metadata.json,
evidence.json (once WP-C writes it), record.json}``, prints the acceptance
table and the run's ``bench.metrics``, and exits 1 when any check fails. A
check whose field the run does not have (``compose_runs`` before WP-C, say)
is "n/a", not a failure. The report folder goes under ``<out>/<qid>/reports``
(``ARGUS_REPORTS_ROOT``), never into the vault.

Usage (from the repo root; one live run against the proxy's quota):
    PYTHONPATH="" PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe scripts/live_check.py \\
        --query q010 --length medium [--depth standard] [--profile free-proxy] \\
        [--recency m] [--out DIR] [--expect-banner]

``--query`` takes a bench query id (``bench/queries.jsonl``) or the query
text; ``--expect-banner`` (nonexistent-entity queries, q013/q014) expects the
grounding banner and at most 3 findings. Then compare with the v3 baseline:
    PYTHONPATH="" ./venv/Scripts/python.exe -m bench.compare --a bench/results/v3-baseline --b <out>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from argus.graph.run_config import PRESETS  # noqa: E402
from argus.graph.state import VALID_LENGTHS  # noqa: E402
from bench import metrics, runner  # noqa: E402

NA = metrics.NA
QUERIES = ROOT / "bench" / "queries.jsonl"
BANNER = "Limited direct evidence"                       # graph/grounding.py grounding_banner()
FAILURE_TEXTS = ("synthesis_no_evidence", "report suppressed")
EVIDENCE_KEYS = ("schema", "run", "run_config", "engines", "brief", "coverage", "gap_history",
                 "compose_runs", "revision_rounds", "gap_waves", "revise_rounds", "extend_rounds",
                 "degraded", "sources", "notes", "findings", "analysis", "tldr", "panel", "search",
                 "degradations", "errors")                # spec §2.6, schema 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Argus live check: one graph-direct run + acceptance table.")
    p.add_argument("--query", default="q010", help="a bench query id or the query text (default q010)")
    p.add_argument("--length", default="medium", choices=VALID_LENGTHS)
    p.add_argument("--depth", default=None, choices=list(PRESETS), help="default: the length's own preset")
    p.add_argument("--profile", default=None, help="engines profile (default: the active one)")
    p.add_argument("--recency", default=None, choices=["d", "w", "m", "y"],
                   help="run_config.recency (default: the brief's time-sensitivity decides)")
    p.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "argus_live_check")
    p.add_argument("--expect-banner", action="store_true",
                   help="expect the grounding banner and <= 3 findings (nonexistent entity)")
    return p.parse_args(argv)


def resolve_query(arg: str, path: Path = QUERIES) -> dict:
    """The bench record for an id or its exact text, else an ad-hoc query."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        if arg in (record["id"], record["query"]):
            return record
    if re.fullmatch(r"q\d{3}", arg):
        raise SystemExit(f"unknown bench query id {arg!r} (not in {path})")
    slug = re.sub(r"[^a-z0-9]+", "_", arg.lower()).strip("_")[:40]
    if not slug:
        raise SystemExit("--query needs a bench id or some text")
    return {"id": slug, "query": arg}


# ---------------------------------------------------------------------------
# The checks: each takes the run (metrics.load_run) and its computed metrics
# and returns (PASS | FAIL | n/a, detail)
# ---------------------------------------------------------------------------

def _verdict(ok: bool, detail) -> tuple[str, str]:
    return ("PASS" if ok else "FAIL"), str(detail)


def _metric(m: dict, name: str, ok) -> tuple[str, str]:
    value = m[name]
    return (NA, "field not in this run") if value == NA else _verdict(ok(value), metrics.fmt(value))


def check_run_completed(run, m):
    rec = run["record"]
    if rec is None:
        return "FAIL", "no record.json"
    detail = f"{rec['status']} in {rec['duration_s']} s" + (f" ({rec['error']})" if rec.get("error") else "")
    return _verdict(rec["status"] == "ok", detail)


def check_fetched(run, m):
    n = len(run["state"].get("fetched") or [])
    return _verdict(n > 0, n)


def check_evidence(run, m):
    n = len(run["state"].get("evidence") or [])
    return _verdict(n > 0, n)


def check_cited_findings(run, m):
    findings = run["state"].get("findings") or []
    n = sum(1 for f in findings if f.get("citation_urls") or f.get("source_ids"))
    return _verdict(n > 0, f"{n} of {len(findings)}")


def check_few_findings(run, m):
    n = len(run["state"].get("findings") or [])
    return _verdict(n <= 3, n)


def check_report_size(run, m):
    md = run["report"]
    return ("FAIL", "no report.md") if md is None else _verdict(len(md) > 1000, f"{len(md)} chars")


def check_sources_block(run, m):
    return _metric(m, "sources_block_present", lambda v: v is True)


def check_cite_markers(run, m):
    return _metric(m, "cite_markers_unresolved", lambda v: v == 0)


def check_flagged_in_body(run, m):
    return _metric(m, "flagged_claims_in_body", lambda v: v == 0)


def check_judged(run, m):
    """judged == total. A v4 panel verdict whose total is 0 while findings were
    due for the grounding judge counted nothing (the v3 panel writes the
    PanelVerdict defaults): that fails, it does not pass as 0 == 0."""
    state = run["state"]
    pv = state.get("panel_verdict") or {}
    if "total" not in pv:
        return NA, "no judged/total in panel_verdict"
    judged, total = int(pv.get("judged") or 0), int(pv["total"])
    if total == 0:
        panel = (state.get("run_config") or {}).get("panel") or {}
        findings = state.get("findings") or []
        if findings and panel.get("enabled") and panel.get("grounding"):
            return "FAIL", f"judged 0 of total 0 with {len(findings)} findings: the panel recorded no judging"
        return NA, "nothing was due for judging"
    return _verdict(judged == total, f"{judged}/{total}")


def check_compose_budget(run, m):
    return _metric(m, "compose_budget_ok", lambda v: v is True)


def check_evidence_json(run, m):
    ev = run["evidence"]
    if ev is None:
        return NA, "no evidence.json (WP-C writes it)"
    if not isinstance(ev, dict):
        return "FAIL", "not a JSON object"
    problems = []
    if ev.get("schema") != 1:
        problems.append(f"schema {ev.get('schema')!r}")
    missing = [k for k in EVIDENCE_KEYS if k not in ev]
    if missing:
        problems.append("missing " + ", ".join(missing))
    if '"markdown_path":' in json.dumps(ev):
        problems.append("carries a markdown_path")
    return _verdict(not problems, "; ".join(problems) or "schema 1, every §2.6 key")


def check_metadata_engines(run, m):
    md = run["metadata"]
    if md is None:
        return "FAIL", "no metadata.json"
    return _verdict("engines" in md, "present" if "engines" in md else "missing")


def check_no_failure_text(run, m):
    md = run["report"]
    if md is None:
        return NA, "no report.md"
    hits = [t for t in FAILURE_TEXTS if t in md]
    return _verdict(not hits, ", ".join(hits) or "none")


def check_banner_absent(run, m):
    md = run["report"]
    return (NA, "no report.md") if md is None else _verdict(BANNER not in md, "present" if BANNER in md else "absent")


def check_banner_present(run, m):
    md = run["report"]
    return (NA, "no report.md") if md is None else _verdict(BANNER in md, "present" if BANNER in md else "absent")


def checks(expect_banner: bool) -> list[tuple[str, object]]:
    return [
        ("run completed", check_run_completed),
        ("fetched > 0", check_fetched),
        ("evidence > 0", check_evidence),
        ("findings <= 3", check_few_findings) if expect_banner
        else ("findings with citations > 0", check_cited_findings),
        ("report.md > 1000 chars", check_report_size),
        ("sources_block_present", check_sources_block),
        ("cite_markers_unresolved == 0", check_cite_markers),
        ("flagged_claims_in_body == 0", check_flagged_in_body),
        ("judged == total", check_judged),
        ("compose_budget_ok", check_compose_budget),
        ("evidence.json valid", check_evidence_json),
        ("metadata.json has engines", check_metadata_engines),
        ("no synthesis_no_evidence / report suppressed text", check_no_failure_text),
        ("grounding banner present", check_banner_present) if expect_banner
        else ("grounding banner absent", check_banner_absent),
    ]


def run_checks(run: dict, *, expect_banner: bool) -> list[tuple[str, str, str]]:
    m = metrics.compute(run)
    return [(name, *fn(run, m)) for name, fn in checks(expect_banner)]


def render(rows: list[tuple[str, str, str]]) -> str:
    lines = ["| check | result | detail |", "|---|---|---|"]
    lines += [f"| {name} | {status} | {detail.replace('|', '/')} |" for name, status, detail in rows]
    failed = [name for name, status, _ in rows if status == "FAIL"]
    na = [name for name, status, _ in rows if status == NA]
    notes = []
    if failed:
        notes.append("failed: " + ", ".join(failed))
    if na:
        notes.append("n/a: " + ", ".join(na))
    lines += ["", f"OVERALL: {'FAIL' if failed else 'PASS'}" + (f" ({'; '.join(notes)})" if notes else "")]
    return "\n".join(lines) + "\n"


def exit_code(rows: list[tuple[str, str, str]]) -> int:
    return 1 if any(status == "FAIL" for _, status, _ in rows) else 0


async def _main(args: argparse.Namespace) -> int:
    query = resolve_query(args.query)
    qdir = args.out / query["id"]
    os.environ["ARGUS_REPORTS_ROOT"] = str(qdir / "reports")   # before anything reads Settings
    engines = await runner._engines_snapshot(args.profile)
    record = await runner._run_one(
        query, length=args.length, depth=args.depth, engines=engines, timeout_s=None,
        state_dir=args.out, origin="live_check",
        overrides={"recency": args.recency} if args.recency else None)
    if not (qdir / "state.json").exists():
        print(f"run {record['status']}: {record.get('error')} - no checkpoint, nothing to check")
        return 1
    run = metrics.load_run(qdir)
    rows = run_checks(run, expect_banner=args.expect_banner)
    print(f"live check: {query['id']} ({args.length}, {args.depth or 'preset'} depth) -> {qdir}\n")
    print(render(rows))
    print(metrics.render_table({run["qid"]: metrics.compute(run)}))
    return exit_code(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=os.environ.get("ARGUS_BENCH_LOG", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
