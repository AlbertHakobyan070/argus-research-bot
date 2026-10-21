# WP-H — hardening and docs (Task 14)

Spec: `docs/argus-v4-design.md` §3 WP-H, with §6 (migration), §7 (env vars,
removed knobs), §2.5 (cancel semantics), §2.7 (the console) and the engines
sections, which the docs now describe. The Task 14 pointers (items earlier
reviews parked here) are part of the scope, including two controller rulings:
delete `tests/test_e2e_research.py`, and the approved `scripts/run.sh` /
`run_mac.sh` pre-flight edit.

### What changed

Code (Step 1 and pointers 1–5, 10):

- **The Phase 1–2 wrappers are gone.** `llm.py` loses `chat_for_tier`,
  `invoke_with_retry`, `record_from_response`, `resolve_tier`,
  `pick_strong_and_judge` and their private types; `complete()` accepts
  `ROLES` only. `engines/roles.py` loses `LEGACY_ROLES`. `engines/registry.py`
  binds and resolves `ROLES` only, and `current_snapshot_cached()` is replaced
  by `cached_snapshot()`, which returns the cached active snapshot or `None`
  and never resolves or raises. Its one caller outside tests is the plan
  card's render (`bot/runs.py` `_payload`), which must not validate: with no
  snapshot cached it now shows the problem "not validated yet — Approve
  validates it" (it showed the old developer text "no engines snapshot
  resolved: every drive entry must …"). `/engines set` no longer mentions the
  `legacy.*` roles.
- **The `run_query_wave` alias is gone.** `graph/scout.py` and
  `graph/research.py` import `engines.search.run_wave`; `graph/search_providers.py`
  no longer imports or exports it (the lazy import in `run_wave` stays, now
  because the `argus.graph` package imports `engines.search`).
- **Old snapshots stay loadable (pointer 1).** `EnginesSnapshot` drops the
  retired `legacy.<tier>` keys before validation (a `mode="before"` validator
  in `engines/schema.py`), so a run config checkpointed during the v4 build,
  whose `engines.roles` still binds them, loads and resumes. Any other unknown
  role still raises. The library never held them: `/engines set` and
  `PUT /api/engines` always refused `legacy.*` roles in overrides.
  `summarize_calls` keeps counting stored records under `legacy.<tier>` (v3
  records, and wrapper records in runs from the build), so `/engines` and the
  bench still read those runs; `_WRITER_ROLES` / `_JUDGE_ROLES` keep the two
  names for the family check.
- **Dependencies (pointer 2).** Nothing in `src/`, `tests/`, `scripts/` or
  `bench/` imports `langchain_openai` or `tenacity`, so both leave
  `pyproject.toml` (the venv is untouched; `tenacity` still arrives through
  `langchain-core`). There is no lock file.
- **`tests/test_e2e_research.py` deleted (pointer 3, ruling)**, and its entry
  removed from the `--ignore` list in `.github/workflows/ci.yml`. The live
  end-to-end path is `scripts/verify_live.sh` → `scripts/live_check.py`.
- **`tests/test_orphan_reconcile.py`**: the unused `from argus import bot as
  bot_mod` is gone (pointer 4).
- **Run-script pre-flight (pointer 5, ruling).** After the `argus.bot` poller
  kill and its assertion, both scripts stop every cloudflared that is **this
  repo's binary** and assert none is left:
  - The targets come from the bot's own code, run in the venv:
    `_PROJECT_ROOT/tools/bin/cloudflared(.exe)` and `webapp.tunnel.binary()`
    (which reads `ARGUS_CLOUDFLARED`, `.env` included).
  - `run.sh` (Windows): `Win32_Process.ExecutablePath`, normalised with
    `GetFullPath` and compared case-insensitively against the resolved
    targets; `Stop-Process -Force`, then the count must be 0. An empty count
    (PowerShell or WMI unavailable) is fatal, like the poller check.
  - `run_mac.sh` (macOS/Linux): `pgrep -f` with each target as an anchored,
    regex-escaped ERE `^<path>( |$)` — the spawn's `argv[0]` is
    `str(tunnel.binary())` exactly — then TERM, KILL, and the count must be 0.
  - A cloudflared started from any other binary or path is never matched.
- **`Settings.engines_file` / `engines_local_file` removed (pointer 10).**
  Nothing read them; the registry's `default_paths()` is the one, authoritative
  reader of `ARGUS_ENGINES_FILE` / `ARGUS_ENGINES_LOCAL_FILE` (relative paths
  from the repo root). `config._repo_path_env`, their only user, went with them.
- **`scripts/verify_live.sh`** checks the proxy at the configured
  `FREELLMAPI_BASE_URL` (through `get_settings()`, so `.env` applies) instead
  of a hardcoded port, and its header describes the v4 path. It never called a
  removed knob or the deleted test.
- **`scripts/demo_run.py` (v4 refresh)**: patches `run_wave`, prints the
  active profile and the engine each tier resolved to (and the snapshot's
  warnings) instead of `resolve_tier`, prints `judged/total` and the
  `evidence.json` path, and lists calls by role.

Stale prose (pointer 7): `citations.py` (no `perplexity_search`), the
`ResearchBrief` docstring (no `ResearchPlan`), the `state.py` comment that
named `bot.py`, the `bot/` package docstring (rewritten for the v4 modules
and hooks), test docstrings and assert messages that named `bot.py` /
`bot._…` / `manual_e2e.py`, and the one test docstring that said "intel-stack
Chromium". The comments the pointer named in `compose.py` and
`report_builder_helpers.py` were already fixed by WP-C. The two historical
`tonight/HANDOFF-2026-07-09-*.md` files get a one-line "Historical" note at
the top and are otherwise unchanged.

Docs (Step 2, pointers 6, 8, 9):

- `HELP_TEXT` (`bot/common.py`) lists `/engines transcriber`, the console
  (the **Argus** menu button), `/status` (with the console link), `/cancel`
  (stop after the current step) and `/continue` (resume, re-attach or
  extend); it stays under the existing length guard. `/ask` says "no search".
- `docs/help.md`: the plan-gate knobs (depth, profile, recency) and card
  lines, the preview's quality line, what the report guarantees (cut-not-flag,
  the deterministic TL;DR, the Analysis, both banners, degradation notes,
  `evidence.json`), `/ask` as a no-search answer, the console, cancel by
  drain and the busy rule, `/continue` on a stopped run, `/status`,
  `/append` always ingested, the transcription engines, and every `/engines`
  subcommand.
- `README.md`: rewritten for v4, structure only — the graph, the stages,
  engines, run configuration, the layout, tests, demo, operations, secrets.
  No Chromium fallback, `manual_e2e`, `fetch_live_models`, removed knobs or
  `bot.py` remain.
- `.env.example` folds in `.env.mac.example` (deleted): placeholders only,
  grouped by area, every variable the code reads, the removed knobs pointing
  at `REMOVED_ENV_KNOBS`. No default values are copied: a commented line keeps
  the code's default.
- `handoff.md`: a merged "What v4 changed" (the notes' behaviour changes,
  deduplicated), the setup steps updated (`cp .env.example .env`, startup
  without the proxy, the verify step without counts, cloudflared on a new
  machine), the pre-flight, and troubleshooting rows for the console and
  engines. The 2026-08-07 download-fix table stays as a dated history section.
- `docs/research-engine-v3.md`: the superseded note at the top.
- `docs/v4-notes/`: `README.md` (new) indexes the notes, points to the API
  routes, the resting-run route table, the security notes and the documented
  ceilings where they live, and merges the env-var lists; every note
  statement WP-H made untrue carries an inline *"WP-H: …"* marker. No fact,
  table or limit was removed from any note.

Not built (spec: optional / later): the digest cache, `compose.editor`, a
`TimeoutPolicy` on research, a between-item drain in compose / analyst /
panel, inline `[uncited]` markers, the LLM bench judge.

### Decisions where the spec is silent

- **`cached_snapshot()` instead of no accessor.** The plan card needs a sync,
  non-validating look at the active profile's state, so a renamed accessor with
  an honest contract (`None`, no raise) replaces the wrapper one.
- **Drop, not map, the retired roles.** No call names them any more, so a
  mapped binding would never be used; dropping keeps the snapshot's keys equal
  to `ROLES`.
- **The pre-flight resolves its targets through the bot's code** rather than
  re-implementing `ARGUS_CLOUDFLARED` and `.env` parsing in shell.
- **`ARGUS_CLOUDFLARED` is a target as ruled**, so pointing it at a shared
  system binary (e.g. Homebrew's) would let the pre-flight stop other
  cloudflared processes of that binary. `handoff.md` says to point it only at
  a binary Argus alone uses, or to copy it into `tools/bin/`.

### Known ceilings

- The pre-flight runs the venv's Python once to resolve the targets; a broken
  venv stops the script there (loudly, under `set -e`), as the bot would fail
  anyway.
- `run_mac.sh` matches the command line's start, so a cloudflared launched by
  hand with a different spelling of the same path (a symlink, `./tools/…`
  from another directory) is not matched. The bot always spawns
  `str(tunnel.binary())`, which is what is matched.
- `ExecutablePath` is empty for processes Windows does not let the user read;
  those are never matched (a cloudflared of this repo runs as the user).

### Behaviour changes for handoff.md

- `run.sh` / `run_mac.sh` also stop an orphaned cloudflared of this repo
  before starting the bot.
- The plan card of a run whose active profile has no cached snapshot (an
  untracked run viewed after a restart) shows "not validated yet — Approve
  validates it".
- `verify_live.sh` probes the configured proxy URL.
- `/help` and `/help full` describe the v4 commands.

### New env vars

None. `ARGUS_CLOUDFLARED` is now also read (through `webapp.tunnel.binary()`)
by the run-script pre-flight.

### Test files touched

- RED-first for pointer 1: `tests/test_engines_schema.py`
  (`test_a_stored_snapshot_naming_the_retired_legacy_roles_still_loads`) and
  `tests/test_run_config.py`
  (`test_require_loads_a_checkpointed_config_naming_the_retired_legacy_roles`).
- Minimal edits for the deletions: `tests/helpers.py` (`ROLES` only),
  `tests/test_engines_schema.py`, `tests/test_engines_registry.py`,
  `tests/test_engines_cmd.py`, `tests/test_llm_facade.py` (the wrapper test
  deleted; `quick.answer` as the non-JSON role), `tests/test_llm.py` (live),
  `tests/test_graph.py` (live), `tests/test_live_check.py`;
  `run_query_wave` → `run_wave` in `tests/test_extend.py`,
  `tests/test_continue_extend.py`, `tests/test_gap_mode.py`,
  `tests/test_reflexion.py`, `tests/test_research_depth.py`,
  `tests/test_search_providers.py` (the alias test deleted); the dead
  `search_providers` guards in `test_extend.py` / `test_continue_extend.py`
  repointed to `research.run_wave` or dropped.
- `tests/test_run_config.py`: the `Settings` knob tests check
  `registry.default_paths()` for the engines files, and that `Settings` has no
  `engines_file` / `engines_local_file`.
- `tests/test_bot.py`: `HELP_TEXT` and `docs/help.md` must name the v4
  commands and the console.
- Docstrings / messages only: `tests/test_bot_markdown_fallback.py`,
  `tests/test_transcript.py`, `tests/test_pdf_render.py`.
- Deleted: `tests/test_e2e_research.py`. Unused import removed:
  `tests/test_orphan_reconcile.py`.

### Verification evidence (2026-10-21)

Logs in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t14\`.
No bot, cloudflared or live service was started, stopped, signalled or called.

- `baseline.log`: the full suite at `2cc328d` with the closed proxy port —
  only the live-proxy tests (`test_llm.py`, `test_graph.py`) fail.
- `red.log`: with `LEGACY_ROLES` deleted and no loader fix, both new tests
  fail with `unknown role 'legacy.cheap'` (and `.strong`, `.judge`) — a stored
  snapshot could not load. `green_focused.log`: green after the validator.
- `focused1.log`: every touched test file, green.
- `full_suite.log`: the full suite with `FREELLMAPI_BASE_URL` on a closed
  port — only the live-proxy tests fail, no warnings summary.
- `greps.log`: the brief's `src/` grep is empty; the wider greps show only
  removed-knob listings (`REMOVED_ENV_KNOBS`, `.env.example`, `handoff.md`,
  the warning test), the spec, the plan, the notes' history, and tests of
  stored record shapes (`legacy.<tier>` records in the bench and
  `summarize_calls` tests).
- `preflight_dryrun.log`: the Windows pre-flight's resolution and filter,
  listing only — it matches exactly the running cloudflared of this repo's
  `tools/bin/`, the assertion's count reads it, and a target list naming only
  a non-existent binary matches nothing. `preflight_mac_dryrun.log`:
  `run_mac.sh`'s `cf_pids`, copied verbatim, against a fake `pgrep -f` over a
  fixture process table (paths with spaces, dots and parentheses; a sibling
  `cloudflared-old`; another user's Homebrew cloudflared; the path appearing
  as an argument of another program): only the repo binary's processes match.
  Both scripts pass `bash -n`. The target resolution is a plain assignment
  (no pipe), so a failing Python stops the script under `set -e` before any
  kill (checked with a failing import), and an empty list is fatal too.
- `demo_closed_port.log`: `scripts/demo_run.py` imports and stops at the
  engines resolution with the proxy's `EngineConfigError` on a closed port.
