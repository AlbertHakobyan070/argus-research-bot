"""Argus bench v2 — deterministic metrics (v4 spec §3 WP-G).

    PYTHONPATH="" ./venv/Scripts/python.exe -m bench.metrics --results bench/results/<label>

Reads every ``<results>/<qid>/state.json`` (the full final state that
``bench/runner.py`` and ``scripts/live_check.py`` dump) with the files beside
it, and prints one markdown table: a row per metric, a column per query and
the mean over the queries that have a number. No LLM is called.

Files per query: ``state.json``, ``report.md``, ``evidence.json`` (written by
WP-C; its findings, degradations and analysis are used when it is present),
``metadata.json``, ``record.json`` (the runner's record) and ``summary.json``
(the v3 baseline capture's record; only ``status`` and ``elapsed_s`` are
read). A run that failed before its first checkpoint leaves only
``record.json``: it is listed, with every metric but ``status`` and
``wall_clock_s`` "n/a". A metric that needs a field the run did not have is
``"n/a"`` — never 0 and never a crash: the v3 baseline
(``bench/results/v3-baseline``) has none of the v4 fields. A ratio with
nothing to divide by is ``"n/a"`` too.

One rule per metric for every run, so the baseline and a v4 run are measured
alike:

  status                record.json ``status`` (ok / timeout / error), else the v3 capture's
  unique_sources        distinct canonical URLs (spec §2.8 rules) in ``fetched``
  dupe_ratio            1 − unique_sources / fetched
  fetched_success_ratio fetched / (fetched + "fetch <url> failed" lines in ``errors``)
  fetchers_used         ``FetchedItem.fetcher`` counts ("(none)" when unrecorded)
  pinned_ingested_ratio appended (/append) or pinned sources whose URL is in ``fetched``
  evidence_strong_ratio notes with relevance ≥ 3 / notes
  coverage_min/mean     over ``coverage`` (strong notes per sub-question)
  gap_history, gap_waves  the state's; 0 / [] in a v4 run (``run_config`` set) that had no gap wave
  compose_runs          the state's (compose executions in the current user round)
  rewrite_rounds        panel rewrites this round: ``revision_rounds``, less one when the state
                        has no ``compose_runs`` and the final verdict is ``revise`` (the v3 panel
                        counts revise verdicts, and the last one, at the cap, ships as-is)
  compose_budget_ok     spec §2.4 per round, round = 1 + revise_rounds + extend_rounds:
                        compose_runs == 1 + gap waves of the round + revision_rounds,
                        compose_runs ≤ 1 + gap_waves_max + max_compose_revisions,
                        revision_rounds ≤ max_compose_revisions, gap_waves ≤ gap_waves_max
  quote_verified_ratio  claims with quote_verified True / claims checked (not None)
  claims_per_primary_source  claims per evidence note from a primary domain (scorer.PRIMARY_DOMAINS)
  cite_markers_unresolved  markers above the last ``## Sources`` heading (outside what the normaliser
                        never reads: code, URLs, link destinations, footnote definitions) that do
                        not resolve: every unnormalised form (``【n】`` ``［n］`` ``〔n〕`` ``【n†…】``
                        ``[n, m]`` ``[1-3]`` ``[^n]`` ``[Source n]``) plus each ``[n]`` without an entry
  sources_block_present a ``## Sources`` / ``## References`` heading with at least one entry
  flagged_claims_in_body  flagged findings (``panel_verdict.flagged_finding_ids``) not cut:
                        status ``kept`` / ``unlocated``, or no status (v3, which never cut by id);
                        "n/a" when the grounding judge failed and ``panel_verdict.total`` is
                        absent or 0 (nothing was judged)
  cross_section_duplicate_ratio  findings (not cut, not in the "Analysis" section) whose claim is
                        ≥ 0.85 similar to one in another section / those findings; needs two sections
  judged_ratio          panel_verdict judged / total
  calls_total, calls_by_role, reroute_rate, judge_writer_same_family
                        ``registry.summarize_calls`` (logical calls: ok + exhausted; v3 → legacy.<tier>)
  routed_via_captured_ratio  ok call records with a non-empty routed_via / ok records
  degradations_count    evidence.json degradations, else the facade's skip / degrade / legacy
                        lines in ``errors``; "n/a" without evidence.json and without a call
                        record of a v4 role (stored ``legacy.*`` records come from the
                        removed Phase 1–2 wrappers, which raised instead of writing such
                        a line, and the v3 nodes' fallbacks wrote none either)
  json_parse_failures   records that failed the validate hook ("unparseable:") + error lines
                        quoting the JSON parser ("no useful JSON object found", not an exhaustion)
  exa_spend             ``search_spend``: searches, text_pages, contents_pages, usd
  wall_clock_s          record.json duration_s, else the v3 capture's elapsed_s
  median_source_age_days  run date (first message ``ts``) − published date of each fetched
                        source (its own ``published``, else its search hit's)
  takeaways_present, takeaways_with_confidence_ratio  ``analysis.takeaways``
  disagreement_section_present_when_conflict  when a note ``contradicts``
  scenarios_present_for_forecast  when ``brief.archetype`` is ``forecast``
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from argus.engines import registry
from argus.engines.urls import canonical_url
from argus.graph.assemble import CITE_RE, prose_text

from .scorer import PRIMARY_DOMAINS

NA = "n/a"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from e


def load_run(qdir: Path) -> dict:
    """One query's results folder: its state and the files beside it (None when
    absent). A run that failed before its first checkpoint has only record.json:
    its state is ``{}``."""
    files = {name: _read_json(qdir / f"{name}.json")
             for name in ("state", "evidence", "metadata", "record", "summary")}
    if files["state"] is None:
        if files["record"] is None:
            raise FileNotFoundError(f"{qdir}: no state.json")
        files["state"] = {}
    report = qdir / "report.md"
    return {"qid": qdir.name, **files,
            "report": report.read_text(encoding="utf-8") if report.exists() else None}


def collect(results: Path) -> dict[str, dict]:
    """``{qid: metrics}`` for every ``<results>/<qid>/`` with a state.json or a
    record.json, in qid order."""
    qdirs = sorted({p.parent for name in ("state.json", "record.json") for p in results.glob(f"*/{name}")})
    if not qdirs:
        raise SystemExit(f"no <qid>/state.json or <qid>/record.json under {results}")
    return {q.name: compute(load_run(q)) for q in qdirs}


def _has(state: dict, key: str) -> bool:
    return state.get(key) is not None


def _ratio(num: int, den: int):
    return round(num / den, 3) if den else NA


def _v4_counter(state: dict, key: str, zero):
    """A v4 counter a node writes only when its event happens: absent from a v4
    run (``run_config`` set) means it never happened; absent from a v3 run, the
    field did not exist."""
    if _has(state, key):
        return state[key]
    return zero if _has(state, "run_config") else NA


def _findings(run: dict):
    ev = run["evidence"]
    return ev["findings"] if isinstance(ev, dict) and _has(ev, "findings") else run["state"].get("findings")


def _analysis(run: dict):
    ev = run["evidence"]
    return (ev.get("analysis") if isinstance(ev, dict) else None) or run["state"].get("analysis")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def unique_sources(run):
    fetched = run["state"].get("fetched")
    return NA if fetched is None else len({canonical_url(f.get("url") or "") for f in fetched})


def dupe_ratio(run):
    fetched = run["state"].get("fetched") or []
    unique = {canonical_url(f.get("url") or "") for f in fetched}
    return _ratio(len(fetched) - len(unique), len(fetched))


# v3 research.fetch_one_source: "fetch <url> failed: …"; WP-B may write "fetch: <url> failed …"
_FETCH_FAILED = re.compile(r"^(?:\[r\d+\]\s*)?fetch:?\s+\S+\s+failed")


def fetched_success_ratio(run):
    state = run["state"]
    fetched = state.get("fetched") or []
    failed = sum(1 for e in state.get("errors") or [] if _FETCH_FAILED.match(e))
    return _ratio(len(fetched), len(fetched) + failed)


def fetchers_used(run):
    items = [f for f in run["state"].get("fetched") or [] if "fetcher" in f]
    return dict(sorted(Counter(f["fetcher"] or "(none)" for f in items).items())) if items else NA


_APPENDED = ("appended-asset", "appended-url")      # bot/history.py _sources_from_refs


def pinned_ingested_ratio(run):
    state = run["state"]
    fetched = state.get("fetched") or []
    pinned = ({s.get("url") for s in state.get("sources") or []
               if s.get("pinned") or s.get("source") in _APPENDED}
              | {f.get("url") for f in fetched if f.get("pinned")})
    return _ratio(len(pinned & {f.get("url") for f in fetched}), len(pinned))


# ---------------------------------------------------------------------------
# Evidence and coverage
# ---------------------------------------------------------------------------

def evidence_strong_ratio(run):
    notes = run["state"].get("evidence") or []
    return _ratio(sum(1 for n in notes if int(n.get("relevance") or 0) >= 3), len(notes))


def coverage_min(run):
    cov = run["state"].get("coverage") or {}
    return min(cov.values()) if cov else NA


def coverage_mean(run):
    cov = run["state"].get("coverage") or {}
    return round(statistics.mean(cov.values()), 2) if cov else NA


def quote_verified_ratio(run):
    checked = [c["quote_verified"] for n in run["state"].get("evidence") or []
               for c in n.get("claims") or [] if c.get("quote_verified") is not None]
    return _ratio(sum(1 for v in checked if v is True), len(checked))


def claims_per_primary_source(run):
    notes = [n for n in run["state"].get("evidence") or []
             if any(d in (n.get("source_url") or "") for d in PRIMARY_DOMAINS)]
    return round(sum(len(n.get("claims") or []) for n in notes) / len(notes), 2) if notes else NA


# ---------------------------------------------------------------------------
# Compose bookkeeping (spec §2.4; the orchestrator's compose-budget ruling: §2.4's per-round invariant)
# ---------------------------------------------------------------------------

def gap_history(run):
    return _v4_counter(run["state"], "gap_history", [])


def gap_waves(run):
    return _v4_counter(run["state"], "gap_waves", 0)


def compose_runs(run):
    state = run["state"]
    return state["compose_runs"] if _has(state, "compose_runs") else NA


def rewrite_rounds(run):
    state = run["state"]
    if not _has(state, "revision_rounds"):
        return NA
    rounds = state["revision_rounds"]
    # the v3 panel (no compose_runs) counts revise verdicts; the last one, at the cap, ships as-is
    if not _has(state, "compose_runs") and (state.get("panel_verdict") or {}).get("verdict") == "revise":
        return rounds - 1
    return rounds


def compose_budget_ok(run):
    state = run["state"]
    if not all(_has(state, k) for k in ("compose_runs", "revision_rounds", "run_config")):
        return NA
    rc = state["run_config"]
    gap_max, rev_max = rc["research"]["gap_waves_max"], rc["panel"]["max_compose_revisions"]
    rnd = 1 + int(state.get("revise_rounds") or 0) + int(state.get("extend_rounds") or 0)
    gaps_this_round = sum(1 for g in state.get("gap_history") or [] if g.get("round") == rnd)
    runs, rev = state["compose_runs"], state["revision_rounds"]
    return (runs == 1 + gaps_this_round + rev and runs <= 1 + gap_max + rev_max
            and rev <= rev_max and int(state.get("gap_waves") or 0) <= gap_max)


# ---------------------------------------------------------------------------
# The report body: citation markers and the Sources block
# ---------------------------------------------------------------------------

_SOURCES_HEADING = re.compile(r"^[ \t]*##[ \t]+(?:Sources|References)[ \t]*$", re.M)
_BLOCK_END = re.compile(r"^[ \t]*(?:##[ \t]+\S|---[ \t]*$)", re.M)
_ENTRY = re.compile(r"^[ \t]*(?:\[(\d+)\]|(\d+)\.)[ \t]+\S", re.M)
_RAW_MARKERS = re.compile(                           # the forms normalize_cite_markers rewrites (§2.8)
    r"【\s*\d[^】]*】|［\s*\d+\s*］|〔\s*\d+\s*〕"
    r"|\[\s*\d+\s*(?:[,;]\s*\d+\s*)+\]|\[\s*\d+\s*[-–]\s*\d+\s*\]"
    r"|\[\^\d+\]|\[Source\s+\d+\]", re.I)


def _split_report(md: str) -> tuple[str, set[int] | None]:
    """(body, Sources entry ids). The body is everything above the last
    ``## Sources`` heading: v4 appendices sit above it, the v3 flagged appendix
    (whose labels are local to each claim) below. None: no Sources heading."""
    heads = list(_SOURCES_HEADING.finditer(md))
    if not heads:
        return md, None
    rest = md[heads[-1].end():]
    end = _BLOCK_END.search(rest)
    block = rest[:end.start()] if end else rest
    return md[:heads[-1].start()], {int(a or b) for a, b in _ENTRY.findall(block)}


def cite_markers_unresolved(run):
    md = run["report"]
    if md is None:
        return NA
    body, ids = _split_report(md)
    body = prose_text(body)             # as normalize_cite_markers reads it: no code, URL, link target, definition
    plain = sum(1 for n in CITE_RE.findall(body) if ids is None or int(n) not in ids)
    return len(_RAW_MARKERS.findall(body)) + plain


def sources_block_present(run):
    md = run["report"]
    return NA if md is None else bool(_split_report(md)[1])


# ---------------------------------------------------------------------------
# Findings: flags, duplicates, judging
# ---------------------------------------------------------------------------

_GROUNDING_FAILED = "panel: grounding judge failed"        # graph/panel.py's error line


def flagged_claims_in_body(run):
    state = run["state"]
    pv, findings = state.get("panel_verdict"), _findings(run)
    if pv is None or findings is None:
        return NA
    if not pv.get("total") and any(_GROUNDING_FAILED in e for e in state.get("errors") or []):
        return NA                       # the judge that flags ids never answered: nothing was measured
    flagged = set(pv.get("flagged_finding_ids") or [])
    return sum(1 for f in findings if f.get("id") in flagged and f.get("status") != "cut")


def _norm(text: str) -> str:
    text = CITE_RE.sub(" ", unicodedata.normalize("NFKC", text).casefold())
    return " ".join(re.findall(r"\w+", text))


def cross_section_duplicate_ratio(run):
    # The analyst restates evidence-section findings by design (WP-I): duplication is measured
    # between evidence sections only.
    findings = [f for f in _findings(run) or [] if f.get("status") != "cut" and f.get("section") != "Analysis"]
    if len({f.get("section") or "" for f in findings}) < 2:
        return NA
    texts = [_norm(f.get("claim") or "") for f in findings]
    dup: set[int] = set()
    # ponytail: pairwise difflib, O(n²) — fine for one report's findings (tens to a few hundred)
    for i, j in itertools.combinations(range(len(findings)), 2):
        if (findings[i].get("section") != findings[j].get("section")
                and SequenceMatcher(None, texts[i], texts[j], autojunk=False).ratio() >= 0.85):
            dup.update((i, j))
    return _ratio(len(dup), len(findings))


def judged_ratio(run):
    pv = run["state"].get("panel_verdict") or {}
    return _ratio(int(pv.get("judged") or 0), pv["total"]) if pv.get("total") else NA


# ---------------------------------------------------------------------------
# Engines and provenance
# ---------------------------------------------------------------------------

def _summary(run) -> dict | None:
    state = run["state"]
    if "model_calls" not in state:
        return None
    return registry.summarize_calls(state["model_calls"] or [], state.get("errors") or [])


def calls_total(run):
    s = _summary(run)
    return NA if s is None else s["calls"]


def calls_by_role(run):
    s = _summary(run)
    return NA if s is None else {role: row["calls"] for role, row in sorted(s["by_role"].items())}


def reroute_rate(run):
    s = _summary(run)
    return NA if s is None or s["reroute_rate"] is None else s["reroute_rate"]


def judge_writer_same_family(run):
    s = _summary(run)
    return NA if s is None or s["judge_writer_same_family"] is None else s["judge_writer_same_family"]


def routed_via_captured_ratio(run):
    calls = run["state"].get("model_calls") or []
    if not any("routed_via" in c for c in calls):
        return NA
    ok = [c for c in calls if c.get("ok", True)]
    return _ratio(sum(1 for c in ok if c.get("routed_via")), len(ok))


_DEGRADATION_LINE = re.compile(r"^(?:\[r\d+\]\s*)?(?:skip|degrade|legacy):")


def degradations_count(run):
    ev, state = run["evidence"], run["state"]
    if isinstance(ev, dict) and _has(ev, "degradations"):
        return len(ev["degradations"])
    v4_role = any(c.get("role") and not c["role"].startswith("legacy.") for c in state.get("model_calls") or [])
    if ev is None and not v4_role:
        return NA                       # nothing writes skip: / degrade: lines yet: unmeasured, not 0
    return sum(1 for e in state.get("errors") or [] if _DEGRADATION_LINE.match(e))


def json_parse_failures(run):
    state = run["state"]
    if "model_calls" not in state:
        return NA
    records = [a for c in state["model_calls"] or []
               for a in [*((c.get("extra") or {}).get("prior_attempts") or []), c]]
    failed = sum(1 for r in records if str(r.get("error") or "").startswith("unparseable:"))
    return failed + sum(1 for e in state.get("errors") or []      # graph/jsonx.py's own message
                        if "no useful JSON object found" in e and "every engine failed after" not in e)


def exa_spend(run):
    spend = run["state"].get("search_spend")
    if spend is None:
        return NA
    return {key: spend.get(f"exa_{key}", NA) for key in ("searches", "text_pages", "contents_pages", "usd")}


def _from_record(run, record_key: str, summary_key: str):
    """The runner's record.json value, else the v3 capture's summary.json one."""
    for doc, key in ((run["record"], record_key), (run["summary"], summary_key)):
        if doc is not None and doc.get(key) is not None:
            return doc[key]
    return NA


def status(run):
    return _from_record(run, "status", "status")


def wall_clock_s(run):
    return _from_record(run, "duration_s", "elapsed_s")


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def median_source_age_days(run):
    state = run["state"]
    ts = next((m["ts"] for m in state.get("messages") or [] if m.get("ts")), "")
    if not _ISO_DATE.match(ts):
        return NA
    run_date = date.fromisoformat(ts[:10])
    hit_published = {s.get("url"): s.get("published") or "" for s in state.get("sources") or []}
    ages = [(run_date - date.fromisoformat(p[:10])).days
            for f in state.get("fetched") or []
            for p in [f.get("published") or hit_published.get(f.get("url"), "")]
            if _ISO_DATE.match(p)]
    return statistics.median(ages) if ages else NA


# ---------------------------------------------------------------------------
# Insight proxies
# ---------------------------------------------------------------------------

def takeaways_present(run):
    analysis = _analysis(run)
    return NA if analysis is None else bool(analysis.get("takeaways"))


def takeaways_with_confidence_ratio(run):
    takeaways = (_analysis(run) or {}).get("takeaways") or []
    return _ratio(sum(1 for t in takeaways if t.get("confidence") in ("high", "medium", "low")),
                  len(takeaways))


def disagreement_section_present_when_conflict(run):
    analysis = _analysis(run)
    conflict = any(n.get("stance") == "contradicts" for n in run["state"].get("evidence") or [])
    return bool(analysis.get("has_disagreement_section")) if analysis is not None and conflict else NA


def scenarios_present_for_forecast(run):
    analysis = _analysis(run)
    forecast = (run["state"].get("brief") or {}).get("archetype") == "forecast"
    return bool(analysis.get("has_scenarios")) if analysis is not None and forecast else NA


# the run's status first, then spec §3 WP-G order
METRICS = {
    "status": status, "unique_sources": unique_sources, "dupe_ratio": dupe_ratio,
    "fetched_success_ratio": fetched_success_ratio, "fetchers_used": fetchers_used,
    "pinned_ingested_ratio": pinned_ingested_ratio, "evidence_strong_ratio": evidence_strong_ratio,
    "coverage_min": coverage_min, "coverage_mean": coverage_mean,
    "gap_history": gap_history, "gap_waves": gap_waves, "compose_runs": compose_runs,
    "rewrite_rounds": rewrite_rounds, "compose_budget_ok": compose_budget_ok,
    "quote_verified_ratio": quote_verified_ratio, "claims_per_primary_source": claims_per_primary_source,
    "cite_markers_unresolved": cite_markers_unresolved, "sources_block_present": sources_block_present,
    "flagged_claims_in_body": flagged_claims_in_body,
    "cross_section_duplicate_ratio": cross_section_duplicate_ratio, "judged_ratio": judged_ratio,
    "calls_total": calls_total, "calls_by_role": calls_by_role, "reroute_rate": reroute_rate,
    "routed_via_captured_ratio": routed_via_captured_ratio,
    "judge_writer_same_family": judge_writer_same_family, "degradations_count": degradations_count,
    "json_parse_failures": json_parse_failures, "exa_spend": exa_spend, "wall_clock_s": wall_clock_s,
    "median_source_age_days": median_source_age_days, "takeaways_present": takeaways_present,
    "takeaways_with_confidence_ratio": takeaways_with_confidence_ratio,
    "disagreement_section_present_when_conflict": disagreement_section_present_when_conflict,
    "scenarios_present_for_forecast": scenarios_present_for_forecast,
}


def compute(run: dict) -> dict:
    """Every metric of one query's run (``load_run`` shape)."""
    return {name: fn(run) for name, fn in METRICS.items()}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def fmt(value) -> str:
    """A table cell (ASCII apart from the data itself)."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, dict):
        return ", ".join(f"{k}: {fmt(v)}" for k, v in value.items()) or "none"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _mean(values: list):
    present = [v for v in values if v != NA]
    if not present:
        return NA
    if not all(isinstance(v, (int, float)) for v in present):      # bools count as 0/1
        return "-"
    return round(statistics.mean(float(v) for v in present), 3)


def _cell(value) -> str:
    return fmt(value).replace("|", "\\|")


def render_table(per_query: dict[str, dict]) -> str:
    qids = list(per_query)
    lines = ["| metric | " + " | ".join(qids) + " | mean |", "|---" * (len(qids) + 2) + "|"]
    for name in METRICS:
        values = [per_query[q][name] for q in qids]
        lines.append(f"| {name} | " + " | ".join(_cell(v) for v in [*values, _mean(values)]) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Argus bench v2: deterministic metrics per query.")
    parser.add_argument("--results", required=True, type=Path,
                        help="a results folder with one <qid>/state.json per query")
    args = parser.parse_args(argv)
    sys.stdout.write(render_table(collect(args.results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
