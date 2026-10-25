# Argus Deep Research Bench

Eval harness for measuring whether Argus graph changes actually move the
needle on report quality. Bench v2 (v4 spec §3 WP-G) measures with
**deterministic metrics** computed from each run's full final state; no LLM
judges a report.

## What it measures

`bench/metrics.py` computes, per query, metrics for depth (unique sources,
duplicates, fetch success, evidence strength, coverage, gap waves, quote
verification, claims per primary source, source age), trust (unresolved
citation markers, the Sources block, flagged claims still in the body,
cross-section duplicates, judged / total, the compose budget), engines and
cost (logical calls by role, reroute rate, `routed_via` capture, judge/writer
model family, degradations, JSON parse failures, Exa spend, wall clock) and
insight proxies (takeaways and their confidence labels, a disagreement section
when sources conflict, scenarios for forecasts). The module docstring defines
each one.

A metric that needs a field the run did not have is `n/a`, never 0: the v3
baseline in `results/v3-baseline/` predates every v4 field. The first row is
the run's `status` (`ok`, `timeout` or `error`, from its record), so a failed
run shows as one; a run that failed before its first checkpoint is listed
with only its status and wall clock.

`bench/scorer.py` keeps the original URL metrics (citation integrity, density,
domain coverage, length) over `raw.jsonl`, for continuity.

## Layout

```
bench/
├── README.md
├── queries.jsonl          # {id, query, expected_domains, difficulty}; ids are stable
├── runner.py              # drives the graph per query (auto-approves both gates)
├── metrics.py             # deterministic metrics per query → markdown table
├── compare.py             # metric deltas between two result folders
├── scorer.py              # v1 URL metrics over raw.jsonl
└── results/               # gitignored
    ├── v3-baseline/<qid>/ # the "before" of every v4 comparison
    └── <label>/
        ├── raw.jsonl      # runner output, one record per query
        ├── reports/       # the runs' report folders (unless ARGUS_REPORTS_ROOT is set)
        └── <qid>/         # state.json, report.md, metadata.json,
                           # evidence.json (once WP-C writes it), record.json
```

## Usage

From the repo root, in Git Bash:

```bash
# 1. Run the bench (live proxy + live search; minutes per query)
PYTHONPATH="" ./venv/Scripts/python.exe -m bench.runner \
    --queries bench/queries.jsonl --out bench/results/<label>/raw.jsonl \
    --length short [--depth standard] [--profile free-proxy] [--timeout 300]

# 2. Metrics per query, then the deltas against the v3 baseline
PYTHONPATH="" ./venv/Scripts/python.exe -m bench.metrics --results bench/results/<label>
PYTHONPATH="" ./venv/Scripts/python.exe -m bench.compare \
    --a bench/results/v3-baseline --b bench/results/<label>

# 3. The v1 URL scores (optional)
PYTHONPATH="" ./venv/Scripts/python.exe -m bench.scorer \
    --raw bench/results/<label>/raw.jsonl --queries bench/queries.jsonl \
    --out bench/results/<label>/score.json --md bench/results/<label>/score.md --length short
```

## The live check

`scripts/live_check.py` drives one query through the runner's own `_run_one`
(`origin="live_check"`), writes the same `<out>/<qid>/` layout (default out:
`argus_live_check` in the system temp folder) and prints an acceptance table:
it exits 1 when a check fails. `--query` takes a bench id or the query text.

```bash
PYTHONPATH="" PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe scripts/live_check.py \
    --query q010 --length medium [--depth standard] [--recency m] [--expect-banner]
```

`--expect-banner` is for the nonexistent-entity queries: it expects the
grounding banner and at most 3 findings.

## Queries

Hand-curated technical questions, plus forecasts and time-sensitive topics
(q011, q012), nonexistent entities (q013, q014; run the live check with
`--expect-banner`), a non-technical explainer (q015) and a comparison (q016).
Keep ids stable so runs stay comparable; add queries at the end.

## Constraints

- A fresh `MemorySaver` per query: no SQLite checkpoints, no thread-id
  collisions.
- The runner auto-resumes through the plan gate and the report gate; the goal
  is to measure the graph's output, not the HITL UX.
- The engines are resolved once per bench run from the committed
  `argus_engines.json` plus the local file (`--profile` picks a profile); the
  bot's `/engines` changes are not read. The free proxy reroutes every
  request, so runs are comparable only while it serves the same models.
- The runner sets `ARGUS_REPORTS_ROOT` to `<label>/reports/` unless it is
  already set (in the environment or `.env`), so the finished reports land
  beside the results, not in the vault; each query's `report.md`,
  `metadata.json` and `evidence.json` are copied into `<label>/<qid>/`. The
  live check always sets it to `<out>/<qid>/reports`.
- A rerun into the same `<label>/` first deletes each query's last layout
  files: a failed rerun leaves its own `record.json` (and `state.json` when it
  checkpointed), never the last run's report beside them.

## Limitations

- **No GAIA / HotpotQA integration.** The query set is hand-curated.
- **One sample per query.** The proxy picks a different model per call, so a
  single row is a sample, not a measurement of the mean.
- **Heuristic definitions.** "Primary source" is the scorer's domain list;
  cross-section duplicates are claims at ≥ 0.85 text similarity; the
  canonical URL follows the spec's rule list. Each definition is fixed, so
  before/after comparisons stay fair, but none is a human judgement.
- **No human eval.** Whether a report is *useful* still needs an Albert
  eyeball check. An LLM bench judge (RACE-lite / FACT-lite) is optional /
  later.
