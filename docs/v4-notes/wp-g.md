# WP-G — bench v2 (deterministic metrics) and the live check (Task 6)

Spec: `docs/argus-v4-design.md` §3 WP-G, with §2.4 (budget semantics and the per-round compose invariant), §2.6 (`evidence.json`, read when present), §2.8 (the cut/flag semantics and the citation-marker forms) and §5. The bench now measures every run from its full final state with deterministic metrics, and the "before" of every later comparison is the v3 baseline table at the end of this note. `scripts/live_check.py` replaces `tests/manual_e2e.py`. No LLM judges a report (the orchestrator's no-LLM-judge decision: RACE-lite / FACT-lite stay optional / later, with no role and no cache).

### What changed

- `bench/metrics.py` (new): `python -m bench.metrics --results DIR` reads every `DIR/<qid>/state.json` with the files beside it and prints one markdown table: a row per metric, a column per query and the mean over the queries that have a number. The module docstring defines every metric. The rules:
  - A metric that needs a field the run did not have is `"n/a"`, never 0 and never a crash. So is a ratio with nothing to divide by.
  - The first row is the run's `status`, so a failed run is visible in the metrics and in `compare`.
  - One definition per metric for every run, so the baseline and a v4 run are measured alike, and metrics are always recomputed from the raw state.
  - `evidence.json` (WP-C) supplies the findings (with their `status`), the degradations and the analysis when it is present; everything else comes from `state.json` and `report.md`.
  - The engine metrics reuse `registry.summarize_calls` (logical calls, v3 records counted as `legacy.<tier>`), "primary source" reuses `bench/scorer.py`'s `PRIMARY_DOMAINS`, and the `[n]` marker is `graph/assemble.CITE_RE`.
- `bench/compare.py` (new): `python -m bench.compare --a DIR --b DIR` prints one table, a row per metric and a column per query in both folders; each cell reads `a -> b`, with the difference when both are numbers. Queries in only one folder are named under it.
- `bench/runner.py`:
  - The result layout `<out dir>/<qid>/{state.json, report.md, metadata.json, evidence.json, record.json}`. `report.md`, `metadata.json` and (when `report_paths["evidence"]` is set) `evidence.json` are copied from the report folder; a missing file makes the record an `error`. `record.json` is always written. Each query's run first deletes the five layout files its last run left in `<qid>/`, so a failed rerun never sits beside the last run's report.
  - `_main_async` sets `ARGUS_REPORTS_ROOT` to `<out dir>/reports` unless it is already set (in the environment or `.env`), before anything reads `Settings`. A bench run no longer writes its report folders into the vault by default.
  - `_run_one(..., origin="bench", overrides=None)`, and `timeout_s=None` means no limit. The live check passes `origin="live_check"` and `{"recency": ...}`.
  - Every `astream` call passes `stream_mode=["updates", "custom"]`, `durability="sync"` and the run's one `control=RunControl()` (§1.1; `langgraph.runtime.RunControl`, confirmed on the installed langgraph). Nothing in the bench requests a drain.
  - An error record reads `"<Type>: <message>"`; `repr()` of an `OSError` drops the file name.
- `bench/queries.jsonl`: q011–q016 with the orchestrator's exact query texts, in the existing record shape. `expected_domains` is `[]` and `difficulty` is `""` for all six: no value could be justified without a run.
- `bench/README.md`: bench v2, POSIX commands, no counts. The `ARGUS_BENCH_MODEL` line is gone (nothing reads that variable).
- `scripts/live_check.py` (new): `--query` (a bench id or the query text; default `q010`), `--length` (default `medium`), `--depth`, `--profile`, `--recency`, `--out` (default `argus_live_check` under `tempfile.gettempdir()`), `--expect-banner`. It resolves the engines first (`runner._engines_snapshot`, which awaits `registry.active_snapshot()`), runs once through `runner._run_one`, writes the bench layout to `<out>/<qid>/`, prints the acceptance table and the run's metrics, and exits 1 when any check fails. The report folder goes under `<out>/<qid>/reports` (`ARGUS_REPORTS_ROOT`), never into the vault.
- `scripts/verify_live.sh`: step 3 runs the live check on q010 medium.
- `tests/manual_e2e.py` is deleted, and `.github/workflows/ci.yml` no longer ignores it.

### Decisions where the spec is silent, and the orchestrator's rulings

- **`compose_budget_ok`** follows the orchestrator's compose-budget ruling (§2.4's per-round invariant, not WP-G's stale formula): round = 1 + `revise_rounds` + `extend_rounds`; `compose_runs == 1 + <gap_history entries of that round> + revision_rounds`, `compose_runs ≤ 1 + gap_waves_max + max_compose_revisions`, `revision_rounds ≤ max_compose_revisions`, `gap_waves ≤ gap_waves_max`, the maxima from `run_config`. Without `compose_runs`, `revision_rounds` or `run_config` it is `"n/a"`. `revise_rounds`, `extend_rounds`, `gap_waves` and `gap_history` are read as 0 / `[]` when absent: the graph writes them only when a user round or a gap wave happens, so reading their absence as "n/a" would make the check "n/a" on every bench run.
- **`gap_waves` / `gap_history`** are 0 / `[]` in a v4 run (`run_config` set) that had no gap wave, and `"n/a"` in a v3 state.
- **`rewrite_rounds`** counts the panel rewrites that ran this round. With `compose_runs` in the state (v4), that is `revision_rounds`. Without it (the v3 panel, also after WP-D), `revision_rounds` counts revise verdicts, and the last one at the cap ships the report as-is: a final `revise` verdict means one rewrite fewer. q010's three revise verdicts are two rewrites, matching the capture's three compose runs.
- **`judged == total`** (live check): `"n/a"` without `total` in the panel verdict (the v3 shape). A v4 panel verdict with `total` 0 while findings were due for the grounding judge (`panel.enabled` and `panel.grounding`) **fails**: since WP-D, the v3 panel writes the `PanelVerdict` defaults `judged=0, total=0`, and 0 == 0 must not pass. `judged_ratio` is `"n/a"` when `total` is 0.
- **`flagged_claims_in_body`** counts the findings in `panel_verdict.flagged_finding_ids` whose `status` is not `cut`: `kept`, `unlocated` (§2.8) or no status at all. v3 has no status and never cut by id (report_builder left the body intact and only appended the appendix), so every v3 flag counts. The v3 text-only flags (the precision judge's `fabrication_flags`, through the legacy string quarantine WP-C removes) are outside the metric. The baseline's only one, q001's TL;DR, was cut. The metric is `"n/a"` when the grounding judge failed (`panel: grounding judge failed` in `errors`) and `panel_verdict.total` is absent or 0: the judge that flags ids never answered, so an empty flag list measures nothing. q001 is that case: its grounding judge's JSON did not parse, the v3 panel let it vote pass, and its unparsed output had flagged most findings, all still in the body. With a recorded `total` (v4), the flags still count by id.
- **`cite_markers_unresolved`** scans the report above the last `## Sources` heading, with code spans and fences removed. v4 appendices sit above Sources and carry real ids; the v3 flagged appendix sits below, with labels local to each claim. Every unnormalised form of §2.8 counts (`【n】`, `［n］`, `〔n〕`, `【n†…】`, `[n, m]`, `[1-3]`/`[1–3]`, `[^n]`, `[Source n]`), and so does each `[n]` without a Sources entry.
- **`canonical_url`** implements §2.8's rule list in `bench/metrics.py`, because WP-B's `engines/urls.py` does not exist yet (a `ponytail:` comment marks the swap; it is safe for comparisons because the baseline is recomputed too). Finding for WP-B: q001 fetched `github.com/NVIDIA-AI-Blueprints/aiq?tab=readme-ov-file` and `.../aiq/`, one page. §2.8's tracking list (`utm_*`, `ref`, `fbclid`, `gclid`) keeps them apart, so the baseline's q001 `dupe_ratio` is 0, where the capture's host+path rule counted one duplicate. If `tab` should count, the list needs it (orchestrator call).
- **`claims_per_primary_source`** uses the scorer's `PRIMARY_DOMAINS`, which v3 data supports. The digest's `source_type` / `authority` (WP-B) would be a different definition, and switching would change both sides of every comparison at once.
- **`median_source_age_days`**: the run date is the first message's `ts` (intake writes it in v3 and v4); a fetched source's date is its own `published` (WP-B), else its search hit's `published`. So the v3 baseline gets a value. Only `YYYY-MM-DD…` dates count.
- **`status`** and **`wall_clock_s`**: `record.json` `status` / `duration_s`, else the v3 capture's `summary.json` `status` / `elapsed_s`. A folder with only `record.json` is a run that failed before its first checkpoint: `load_run` reads its state as `{}`, so it is listed with its status and wall clock and every other metric `"n/a"`, instead of dropping out of the table.
- **`degradations_count`**: `evidence.json` `degradations` when present, else the facade's `skip:` / `degrade:` / `legacy:` lines in `errors`. Without `evidence.json`, it is `"n/a"` unless some call record has a role outside `legacy.*`. After WP-D every call goes through a `legacy.*` wrapper, which raises on exhaustion, and the v3 nodes' fallbacks are log-only or free text, so no such line can exist yet; v3 records have no `role` at all. Counting would report "nothing measured" as 0.
- **`json_parse_failures`**: attempt records that failed the `validate` hook (`unparseable:`), plus error lines quoting `graph/jsonx.py`'s "no useful JSON object found" that are not the facade's exhaustion message (which quotes its last record).
- **`fetched_success_ratio`** counts failures from the `errors` lines `fetch <url> failed` (v3) and `fetch: <url> failed` (allowed for WP-B). WP-B must keep one of these forms, or this regex needs updating.
- **`cross_section_duplicate_ratio`**: findings not cut whose claim is ≥ 0.85 similar (`difflib`, the threshold §2.8 uses for locating a sentence) to a finding in another section; `"n/a"` with fewer than two sections (flat reports).
- **Live check, `--expect-banner`** (q013/q014): "findings ≤ 3" replaces "findings with citations > 0", and "grounding banner present" replaces "grounding banner absent" (WP-C's q010 acceptance asks for "banners absent"). The banner is detected by graph/grounding.py's phrase "Limited direct evidence".
- **Live check, findings**: "findings with citations > 0" and "findings ≤ 3" count only the findings the report kept: status `kept`, or no status (v3). A `cut` or `unlocated` finding never counts (`flagged_claims_in_body == 0` covers the unlocated ones).
- **Live check, `evidence.json` valid**: until WP-C's pydantic model exists, the check is structural: a JSON object with `schema == 1`, every §2.6 top-level key, and no `markdown_path` key anywhere.
- **Live check, no timeout** (like `manual_e2e`); a run that leaves no checkpoint prints its record's error and exits 1. A "run completed" row (the record's status) heads the table.

### Behaviour changes for handoff.md

- `tests/manual_e2e.py` is gone. The graph-direct live check is `scripts/live_check.py`: it prints an acceptance table, exits 1 on any failed check, and leaves the bench layout in its out folder for `bench.metrics` / `bench.compare`. `scripts/verify_live.sh` step 3 runs it on q010 medium. Until WP-C that step fails by design (see below).
- `python -m bench.metrics --results DIR` and `python -m bench.compare --a DIR --b DIR` are new. The bench writes `<label>/<qid>/` folders next to `raw.jsonl`.
- `bench/queries.jsonl` gains the forecast, nonexistent-entity, non-technical and comparison queries q011–q016.
- The bench record's `error` reads `"<Type>: <message>"` instead of the exception's repr.
- A bench run writes its report folders under `bench/results/<label>/reports/` instead of the vault, unless `ARGUS_REPORTS_ROOT` is set. A rerun into the same results folder first deletes each query's last layout files.
- The root `README.md` still mentions `tests/manual_e2e.py`. It is WP-H's (docs pass), so it is left unchanged here, as is a docstring line in `tests/test_bot_markdown_fallback.py`.

### New env vars

None. The live check sets `ARGUS_REPORTS_ROOT` for its own run, the bench runner sets it to `<label>/reports` when it is unset, and both honour the bench's existing `ARGUS_BENCH_LOG`.

### Test files touched

- New: `tests/test_bench_metrics.py`:
  - every metric on a v3-shaped `state.json` (the baseline's keys) and on a v4 run;
  - the edge cases (canonical URLs, failed-fetch lines, appended and pinned sources, the compose budget per round including a user revise and an extend, every marker form, the Sources block, flags by status, `evidence.json` precedence, judging, degradations, wall clock, source age, the insight proxies);
  - `load_run` / `collect` / the CLI, `compare`, the runner's layout and `astream` arguments on a fake graph, and the queries;
  - from the task review: the `status` row, `rewrite_rounds` on both state shapes, a failed grounding judge, legacy-only call records, a failed rerun into the same folder with and without a checkpoint (`t6_stale.py`), the runner's reports root, and the review's hand-computed cases (`t6_handcheck.py`).
- New: `tests/test_live_check.py`: argument parsing, query resolution, every check on passing and failing runs (the findings checks on kept findings only), the check list under `--expect-banner`, the table and the exit status, and `_main` resolving the engines before its one run. An autouse fixture fails any test that leaves `ARGUS_REPORTS_ROOT` changed: the `_main` tests set it before deleting it, because deleting an unset variable records nothing for monkeypatch to restore.
- `tests/test_run_config_resume.py::test_bench_run_records_its_run_config_and_dumps_the_final_state` writes the report files its fake report builder names, which the runner now copies, and asserts that the run reached END (`snap.next == ()`) on the graph the runner built.
- Deleted: `tests/manual_e2e.py`. It was never collected by pytest (its name has no `test_` prefix), so the suite's count is unchanged.

### Verification evidence (2026-10-19)

Logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\task6\`. Every suite run used `FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1`; no live LLM call was made and no bot was touched.

- RED (`red.log`): with only the tests written, both files failed at collection (no `bench.metrics`, `bench.compare` or `scripts/live_check.py`). RED (`red2.log`), with the new modules but the runner and queries unchanged: the runner tests failed on `_run_one() got an unexpected keyword argument 'origin'` and on the old loop not reading `(mode, chunk)` events, and the query tests on the missing q011–q016.
- GREEN: the focused files pass, with WP-D's bench test on the real graph (stubbed nodes) proving the new event loop still sees both static interrupts. The full suite (`suite_green.log`) passes except the live-proxy tests (`tests/test_llm.py` and the `tests/test_graph.py` fixture), which fail on the closed port by design, as at the base commit (`suite_base.log`).
- `python -m bench.metrics --results bench/results/v3-baseline` runs end to end (`baseline_metrics.md`, the table below). Where they overlap, it agrees with the baseline capture's own analysis (`…\8368cbb2-…\scratchpad\v3-baseline\SUMMARY.md`): the unnormalised markers (q010 has them, q001 none), q010's flagged claims still in the body, q001's flags as unmeasured (the capture calls its 0 "not a clean grounding pass"), the rewrites (the capture's compose runs less one), the Sources blocks, the call counts, the wall clock and the strong-note shares. The one difference is q001's `?tab=` duplicate (above).
- `python -m bench.compare` on the baseline against itself renders the table (`compare_self.md`). `scripts/live_check.py` with the proxy on the closed port exits 1 at engine resolution with the explicit `EngineConfigError`, before any run.
- Fix round. RED (`fix_red.log`): the new tests failed on the missing `status` row, `rewrite_rounds` 3 for q010's shape, a failed grounding judge read as 0 flags, legacy-only records read as 1 degradation, run 1's report files beside run 2's record, the unset reports root, and the findings checks counting cut and unlocated findings. `fix_red_envleak.log`: the old `delenv(raising=False)` leaks `ARGUS_REPORTS_ROOT`, and the new fixture catches it. GREEN: the focused files (`fix_green_focused.log`) and the full suite (`fix_suite_green.log`, the same live-proxy failures as at the base). The review's own scripts on the fixed tree: `fix_reviewer_scripts.log`. The baseline table below is recomputed (`baseline_metrics_fix.md`).

### The v3 baseline ("before")

`python -m bench.metrics --results bench/results/v3-baseline` (q001 short and q010 medium, captured at `012a756`):

| metric | q001 | q010 | mean |
|---|---|---|---|
| status | ok | ok | - |
| unique_sources | 28 | 28 | 28 |
| dupe_ratio | 0 | 0 | 0 |
| fetched_success_ratio | 1 | 1 | 1 |
| fetchers_used | n/a | n/a | n/a |
| pinned_ingested_ratio | n/a | n/a | n/a |
| evidence_strong_ratio | 0.286 | 0.821 | 0.553 |
| coverage_min | 0 | 0 | 0 |
| coverage_mean | 1.67 | 4.33 | 3 |
| gap_history | n/a | n/a | n/a |
| gap_waves | n/a | n/a | n/a |
| compose_runs | n/a | n/a | n/a |
| rewrite_rounds | 2 | 2 | 2 |
| compose_budget_ok | n/a | n/a | n/a |
| quote_verified_ratio | n/a | n/a | n/a |
| claims_per_primary_source | 2.04 | 4.22 | 3.13 |
| cite_markers_unresolved | 0 | 21 | 10.5 |
| sources_block_present | yes | yes | 1 |
| flagged_claims_in_body | n/a | 4 | 4 |
| cross_section_duplicate_ratio | n/a | 0 | 0 |
| judged_ratio | n/a | n/a | n/a |
| calls_total | 46 | 68 | 57 |
| calls_by_role | legacy.cheap: 37, legacy.judge: 2, legacy.strong: 7 | legacy.cheap: 48, legacy.judge: 3, legacy.strong: 17 | - |
| reroute_rate | 1 | 1 | 1 |
| routed_via_captured_ratio | n/a | n/a | n/a |
| judge_writer_same_family | yes | yes | 1 |
| degradations_count | n/a | n/a | n/a |
| json_parse_failures | 1 | 1 | 1 |
| exa_spend | n/a | n/a | n/a |
| wall_clock_s | 375.7 | 258.6 | 317.15 |
| median_source_age_days | 555 | 409 | 482 |
| takeaways_present | n/a | n/a | n/a |
| takeaways_with_confidence_ratio | n/a | n/a | n/a |
| disagreement_section_present_when_conflict | n/a | n/a | n/a |
| scenarios_present_for_forecast | n/a | n/a | n/a |

### The post-WP-D reference run (the orchestrator's)

`live_check.py` on q010 medium against this tree is recorded as the post-WP-D reference and is not expected to pass. What the checks can show before WP-B/WP-C, from the code (an expectation, not a result):

- `cite_markers_unresolved == 0` fails whenever a writer emits `【n】`-style markers, because `normalize_cite_markers` is still the identity.
- `flagged_claims_in_body == 0` fails whenever the grounding judge flags an id, because the v3 report builder leaves flagged findings in the body. It is `n/a` when the grounding judge fails instead (the v3 panel writes `total` 0).
- `judged == total` fails: the v3 panel writes `judged=0, total=0` with findings present.
- `compose_budget_ok` is `n/a` (the v3 compose writes no `compose_runs`), and so is `evidence.json valid` (nothing writes it yet).
- In its metrics, `degradations_count` is `n/a` (`legacy.*` call records only), and `rewrite_rounds` follows the v3 rule (no `compose_runs`).

The orchestrator's run records the actual table.
