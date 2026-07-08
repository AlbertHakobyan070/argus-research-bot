# Argus Deep Research Bench

Eval harness for measuring whether Argus graph improvements actually
move the needle on report quality.

## What it measures

For each query in `queries.jsonl`:

- **`citation_integrity`** — fraction of inline URLs in the report that
  point at known primary-source domains (arxiv.org, github.com, .edu,
  official documentation sites). Proxy for the `SourceRegistry`'s
  effectiveness: if the citation registry is working, fabricated URLs
  should be stripped before they reach the report.
- **`citation_density`** — URLs per 1000 chars of report.
- **`domain_coverage`** — fraction of `expected_domains` (per query)
  actually cited.
- **`length_ok`** — report length within ±50% of target for the mode.
- **Duration, sources fetched, findings count** — efficiency metrics.

## Layout

```
bench/
├── README.md           # this file
├── __init__.py
├── queries.jsonl       # {id, query, expected_domains, difficulty}
├── runner.py           # drives the Argus graph per query
├── scorer.py           # computes scores from raw.jsonl
└── results/
    ├── raw.jsonl       # runner output (one record per query)
    ├── score.json      # scorer output (machine-readable)
    └── score.md        # scorer output (human-readable)
```

## Usage

```bash
# 1. Run the bench (5–30 min depending on length mode and proxy speed)
cd "A:\Hermes\Agents\argus"
set PYTHONPATH=
venv\Scripts\python.exe -m bench.runner ^
    --queries bench\queries.jsonl ^
    --out bench\results\raw.jsonl ^
    --length short ^
    --timeout 300

# 2. Score the results
venv\Scripts\python.exe -m bench.scorer ^
    --raw bench\results\raw.jsonl ^
    --queries bench\queries.jsonl ^
    --out bench\results\score.json ^
    --md bench\results\score.md ^
    --length short

# 3. Read score.md
type bench\results\score.md
```

## Comparing branches (the actual reason this exists)

```bash
# Baseline: checkout yesterday's state
git checkout feat/argus-v1
venv\Scripts\python.exe -m bench.runner --out bench/results/baseline.jsonl ...
venv\Scripts\python.exe -m bench.scorer --raw bench/results/baseline.jsonl ...

# Head: today's fixes
git checkout feat/argus-fixes
venv\Scripts\python.exe -m bench.runner --out bench/results/head.jsonl ...
venv\Scripts\python.exe -m bench.scorer --raw bench/results/head.jsonl ...

# Diff
diff bench/results/baseline.md bench/results/head.md
```

If `mean_citation_integrity` goes UP after today's `SourceRegistry`
work, the citation lift actually moved the needle. If it stays flat,
we shipped complexity for nothing.

## Constraints

- The bench always uses a fresh `MemorySaver` per query (no SQLite
  checkpoints, no thread-id collisions).
- The runner auto-resumes through LangGraph interrupts (`researcher`
  and `deliver`) without user input — the goal is to measure graph
  output, not HITL UX.
- Default timeout per query is 300s. Increase with `--timeout` for
  `long` / `lecture` modes.
- The bench uses the same FreeLLMAPI as the bot, so results will
  drift if the proxy's tier resolution changes mid-run. Pin model
  via env var if reproducibility matters: `ARGUS_BENCH_MODEL=...`.

## Limitations (documented honestly)

- **No GAIA / HotpotQA integration.** The 10-query set in
  `queries.jsonl` is hand-curated. Wiring GAIA would add 2–3h and
  isn't worth it for the first bench iteration.
- **Primary-domain heuristic.** Citation integrity is computed by
  matching against a hardcoded list of "good" domains. A URL like
  `nytimes.com/2026/ai-breakthrough.html` would be scored as
  non-primary even if it points at a legitimate source. A more
  robust scorer would use a domain-authority signal (Majestic,
  Moz, etc.) — out of scope for v1.
- **No human eval.** The bench measures objective signals only.
  Whether a report is *useful* to a human reader still needs an
  Albert eyeball check.