# WP-D — RunConfig wiring, state additions, library migration (Task 5)

Spec: `docs/argus-v4-design.md` §3 WP-D, with §2.4 (the RunConfig rule, the presets and the state additions), the WP-D named edits of §5, and §6 (migration). Every run now carries its own `RunConfig` in its checkpoint. A new run sets `state["run_config"]`. Every resume entry point first resolves the engines snapshot and calls `ensure_in_checkpoint`, so a pre-v4 checkpoint gets the legacy defaults visibly. `graph/state.py` holds every §2.4 key and model field and is frozen from here on. No node reads the config yet; the nodes keep the v3 wrappers until their WPs migrate them.

### What changed

- `src/argus/graph/state.py`, the §2.4 additions (frozen after this task):
  - `ArgusState` keys: `run_config`, `search_spend`, `analysis`, `tldr_md`, `coverage_gaps`, `gap_history`, `gap_waves`, `compose_runs`, `cancel_note`. All are last-value channels. `coverage_gaps` must be (a gap wave clears it), and the spec gives no reducer for the others.
  - `report_paths` documents the `evidence` key, `sections[*]` carry `note_ids` and `outline` carries `changed_sections` (plain dicts: comments only).
  - Model fields, each with the §2.4 default so a pre-v4 checkpoint's dicts still load:
    - `FetchedItem`: `canonical_url`, `domain`, `published`, `fetcher`, `fetch_ms`, `chars`, `authority_prior`, `pinned`;
    - `EvidenceClaim`: `quote_verified`;
    - `EvidenceNote`: `source_type`, `authority`, `cached`;
    - `ResearchBrief`: `archetype`, `time_sensitive`, `recency_hint`, `is_academic_topic`, `entities`;
    - `OutlineSection`: `note_ids`;
    - `Finding`: `source_ids`, `sentence`, `status`;
    - `PanelVerdict`: `next`, `coverage_gaps`, `wants_rewrite`, `judge_not_diverse`, `uncited_specifics`, `skipped_judges`, `judged`, `total`.
  - The key matters. LangGraph silently drops a key `ArgusState` doesn't declare, from the input, from node returns and from `aupdate_state`. Before this change, `ensure_in_checkpoint` on the real graph wrote the `errors` line and lost the config.
- `src/argus/library.py`:
  - `_migrate(db)`, run by `open()` after the schema script. It checks `PRAGMA table_info(runs)` and adds each missing column: `config_json TEXT`, `depth TEXT`, `archetype TEXT`, `origin TEXT DEFAULT 'telegram'`, `exa_usd REAL`. It is guarded and additive only, so a second open is a no-op. Fresh and old databases go through the same path. Pre-v4 rows keep NULL in the new columns, and `origin` is `'telegram'` (the only pre-v4 writer).
  - `create_run(..., config_json, depth, archetype=None, origin)`, as §3 lists it. `config_json`, `depth` and `origin` are required. `config_json` is the RunConfig as JSON text (`rc.model_dump_json()`), and anything but a JSON object raises `ValueError`.
  - `set_run_config(run_id, config)` stores a (re)built config dict, with the `length` and `depth` columns it implies. Approve with another length now updates the row (the v3 stale-length bug). It leaves `origin` alone, so the row keeps where the run was started. It returns False for an unknown run.
  - `get_run_config(run_id)`: the stored dict; None for an unknown run or a pre-v4 row, as `get_run` returns None for a missing run.
  - `update_run(run_id, /, **cols)`: sets `runs` columns by name, plus `updated_at`. The identity columns, `created_at` and unknown names raise `ValueError`, and so does an unknown status. It returns False for an unknown run.
  - `get_json(key, default=None)` and `set_json(key, value)` on the settings table. A stored value that isn't JSON raises.
  - `exa_usd_month_to_date()`: the sum of `exa_usd` over the runs created this calendar month (UTC). A run without a recorded spend counts 0.
- `src/argus/config.py`:
  - `max_revision_rounds` is gone. Nothing read it; the panel reads `ARGUS_MAX_REVISIONS` itself until WP-C.
  - New `Settings` fields, read from the environment by `load()`: `webapp_enabled`, `webapp_port`, `engines_file`, `engines_local_file`, `cancel_drain_timeout_s` and `shutdown_drain_s`. Their dataclass defaults are the only defaults; `load()` falls back to them. `ARGUS_WEBAPP_ENABLED` takes true/false/1/0/yes/no/on/off and raises on anything else. The engines paths resolve as the registry resolves them (relative to the repository). *(WP-H: `engines_file` / `engines_local_file` removed — nothing read them; the registry's `default_paths()` is the one reader of `ARGUS_ENGINES_FILE` / `ARGUS_ENGINES_LOCAL_FILE`.)*
- `src/argus/graph/run_config.py`: the module docstring only. The contract stays as the Task 1 fix round left it.
- `src/argus/bot/research.py` (named edits):
  - `research_cmd` and `ask_cmd` build `RunConfig.from_preset(length=…, engines=<the resolved snapshot>, origin="telegram")`, put it in `state_in["run_config"]` and pass it to `create_run`.
  - `_ensure_run_config(info)` resolves the snapshot and calls `ensure_in_checkpoint`. It replaces WP-A's `await registry.active_snapshot()` line at the start of `_replan_after_edit`, `_revise_after_feedback`, `_resume_after_plan` and `_resume_after_report`.
  - **Approve** (`_approve_run_config`) runs before anything else changes. It resolves the snapshot, ensures the checkpoint's config, and writes `rc.rebuild(length=<chosen>, engines=<snapshot>)` with that length to the checkpoint and then to the library row. `rebuild`, never a fresh `from_preset`, so console overrides survive a length, depth or profile change. Depth follows the length (`depth=None`): the plan gate has no depth control yet (WP-E1 adds one).
  - **Extend** resolves and ensures before it sets `extend_requested` and the `running` status.
  - `_resume_after_plan` no longer writes `length`. Approve writes it together with the rebuilt config, and the report-gate paths and `/continue` never change it. This also removes a swallowed `aupdate_state` failure ("defaulting length").
- `src/argus/bot/history.py` (named edit): `continue_cmd` refuses a mid-pipeline run first, and only then calls `ensure_in_checkpoint`. That covers the plan-gate and report-gate re-attach and the extend fork, and nothing writes to a checkpoint a drive may still own.
- `bench/runner.py` (named edit):
  - `--depth` (a preset; default: the length's own) and `--profile`.
  - `--length` takes the real lengths. The old `quick` choice was never a valid length, and `from_preset` refuses it.
  - Each query starts with `run_config` (`origin="bench"`) and streams with `durability="sync"`.
  - The full final state (the checkpoint's values, also after a timeout or an error) goes to `<out dir>/<qid>/state.json`, the shape of the v3 baseline's `state.json` (`json.dumps(values, indent=2, ensure_ascii=False, default=str)`). The record gains `state_path`, `depth` and `profile`.
  - The engines are resolved once per bench run, from the committed file plus the local file. The bot's `/engines` changes live in its library and are not read. `--profile` switches in-process through the registry's own `set_active_profile` (the full validation `/engines use` does). The switch persists into a throwaway in-memory library, which is closed afterwards, even when the switch fails.
- `scripts/demo_run.py` (named edit): it resolves the snapshot before the routing printout and starts the run with `run_config` (`origin="demo"`). It had raised "no engines snapshot resolved" since WP-A.
- `tests/manual_e2e.py` (named edit): the run starts with `run_config` (length `short`, so the quick preset; `origin="live_check"`, the value meant for the graph-direct live check that replaces this script in WP-G). A new acceptance line checks `run_config.preset_origin == 'quick'` on the final state and counts toward `OVERALL`.

### Contract decisions where the spec is silent

- **Ordering on resume** (a WP-A review finding): every WP-D resume edit resolves the engines and ensures the config before it touches the run. Approve changes `plan_approved`, the keyboard and the status only after the snapshot, the ensure, the rebuild and the writes succeed. Extend sets `extend_requested` and `running` only after the snapshot and the ensure. A failure (the proxy down after a restart, a checkpoint without `length`, overrides that don't fit the new depth's gap reserve) leaves the checkpoint, the library status, the in-flight entry and the keyboard as they were. The error propagates: an `EngineConfigError` reaches the chat through WP-A's error handler, and anything else is logged.
- **Depth at Approve follows the length.** A run started short (quick) and approved at long becomes deep, with its overrides re-applied. With no depth control at the gate, this matches what `/research /length long` does at the start.
- **`set_run_config` does not rewrite `origin`.** The row's `origin` is where the run started. A legacy run's config says `origin="legacy"`, and its row keeps `'telegram'`.
- **`manual_e2e` uses `origin="live_check"`.** The `RunConfig.origin` values have no `manual_e2e`; that script is the graph-direct live check WP-G replaces with `scripts/live_check.py`.
- **`metadata.json["run_config"]` is not written here.** §3 and §5 give `graph/report_builder_helpers.py` (and `metadata.json`'s `run_config`) to WP-C, and WP-D has no permitted edit there. The run's config is in the checkpoint, the bench `state.json` and the library row; `manual_e2e` checks it on the final state.
- **Month-to-date Exa spend** counts runs by `created_at`, in UTC. A run that crosses the 1st counts in its start month.
- **The plan-edit and revise replies** are consumed (`_pending_reply.pop` in `handle_pending_reply`) before `_replan_after_edit` / `_revise_after_feedback` resolve. Only the in-memory reply is lost on a failure; the run stays at its gate with its keyboard, and the user taps Edit or Revise again. Left as is (not a WP-D edit; WP-E1 rewrites the driver).

### Behaviour changes for handoff.md

- Every new run (`/research`, `/ask`, the bench, the demo, `manual_e2e`) records its RunConfig: in the checkpoint, and for the bot in the library row (`config_json`, `depth`, `origin`).
- Resuming a pre-v4 run through any button or `/continue` gives it the preset of its own length (`legacy=True`, `origin="legacy"`) and adds the line "legacy: run_config defaults applied (pre-v4 checkpoint)" to its `errors`, once.
- Approve updates the library row's length and depth. `/runs` and `/status` used to keep the length the run was started with.
- Approve or Extend with the engines unresolvable (for example the proxy down after a restart) now leaves the run at its gate, buttons intact, with the engines error in the chat. Before, the run was marked running with its keyboard gone, and stranded.
- A checkpoint without `length` (none in the current checkpoint DB) can't be resumed: `ensure_in_checkpoint` raises, and the bot logs it.
- The first bot start on this code migrates the library: the new `runs` columns are added, and existing rows are untouched.
- `/engines` shows the month-to-date Exa spend from the library (it said `n/a` until now). It stays $0.00 until WP-C writes `runs.exa_usd`.
- The bench takes `--depth` and `--profile`, refuses `--length quick` (use `--depth quick`), and writes a `state.json` per query. `bench/runner.py` and `scripts/demo_run.py` run again.
- `Settings.max_revision_rounds` is removed. `ARGUS_MAX_REVISIONS` is still read by the panel until WP-C.

### New env vars

Read into `Settings` (§7). Their readers come with WP-E1 (the drain timeouts) and WP-E2 (the console):

- `ARGUS_WEBAPP_ENABLED` (default `true`) and `ARGUS_WEBAPP_PORT` (default `8787`);
- `ARGUS_CANCEL_DRAIN_TIMEOUT_S` (default `300`): how long a cancel waits for the drain before `task.cancel()`;
- `ARGUS_SHUTDOWN_DRAIN_S` (default `120`): the bound on draining every live run at shutdown;
- `ARGUS_ENGINES_FILE` and `ARGUS_ENGINES_LOCAL_FILE`, which WP-A introduced, are now also `Settings` fields. The registry still resolves them itself. *(WP-H: no longer `Settings` fields.)*

### Test files touched

- New: `tests/test_run_config_resume.py`:
  - the Review Focus 1 test `test_pre_v4_checkpoint_survives_every_resume_path`, across approve, plan edit, report send/extend/revise, and `/continue` at the plan gate, at the report gate and as the extend fork;
  - `require()` and the resume entries refusing a checkpoint whose length disagrees with its config;
  - approve rebuilding with console overrides kept (`research.waves` survives a quick → deep change) and writing the length and config to the checkpoint and the library;
  - the ordering tests: engines refusal and missing length at approve, extend, send and the `/continue` fork, and a rebuild that can't fit the gap reserve, each leaving the run unchanged;
  - `/research`, `/ask` and the bench starting with their config, the bench's `state.json`, and the bench's throwaway library being closed;
  - the §2.4 keys surviving a checkpoint, the model-field defaults, and the new `Literal` fields refusing unknown values.
- Extended:
  - `tests/test_library.py`: the migration on a frozen copy of the pre-v4 schema (columns added, rows intact, `origin` default, idempotent), a fresh DB, `create_run` storing and validating `config_json`, `set_run_config`, `get_run_config`, `update_run`, the JSON settings and `exa_usd_month_to_date`.
  - `tests/test_run_config.py`: `ensure_in_checkpoint` on a graph over the real `ArgusState`, and the new `Settings` knobs (defaults, environment, boolean parsing).
- Minimal fixes for breakage caused by this change:
  - `tests/test_continue_extend.py`, `tests/test_delete_browser.py`, `tests/test_engines_cmd.py`, `tests/test_hitl_edit_revise.py`, `tests/test_media_bot.py`, `tests/test_plan_hitl_gating.py`, `tests/test_run_scoping.py`: no `max_revision_rounds` in their `Settings`.
  - `tests/test_orphan_reconcile.py`, `tests/test_delete_browser.py`, `tests/test_engines_cmd.py`: `create_run` gets `config_json`, `depth` and `origin`.
  - The graph doubles get a `created_at`, so `ensure_in_checkpoint` can read them: `_FakeGraph` in `tests/test_continue_extend.py`, `_Graph` in `tests/test_engines_cmd.py` (with a v4 `run_config`), the revise snapshot in `tests/test_hitl_edit_revise.py` (with a v4 `run_config`), and the seeded graph in `tests/test_plan_hitl_gating.py` (with a v4 `run_config` and an awaitable `aupdate_state`). That file also gains the `active_snapshot` fixture its sibling files already had.
  - `tests/test_engines_cmd.py`: the `/engines` Exa line now reads the library, so the test records a spend and expects it, instead of `n/a`.
- Named edit: `tests/manual_e2e.py` (above).

### Verification evidence (2026-10-19)

Scripts and logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\task5\`. Every suite run used `FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1`, so nothing could reach the proxy and no live LLM call was made.

- `probe_gates.py`: on the real graph, an input `run_config` is dropped before the state change, and an update with no `as_node` leaves `next` unchanged at the plan gate, at the report gate and after the end.
- RED (`red.log`), with only the tests written: the resume paths end with "state['run_config'] missing"; approve under an engines failure leaves the status `running` (the stranding bug); the model fields, `Settings` knobs and library helpers are missing; and `run_config` is lost on the `ArgusState` graph.
- GREEN: the focused files pass. The full suite (`suite_green.log`) passes except the live-proxy tests (`tests/test_llm.py` and the `tests/test_graph.py` fixture), which fail on the closed port by design, as at the start of the task (`suite_base.log`).
- A bench smoke run with `--profile` hung at exit: the throwaway library was never closed. It is fixed, with a regression test. Each mode now exits promptly with its error on the closed port (`--profile no-such`, `--profile local-ollama`, the default profile), and `--length quick` is refused. `demo_run.py` and `manual_e2e.py` now stop at the engines resolution with the proxy's refusal, not "no engines snapshot resolved". Their live runs are the orchestrator's.
- Migration (`migrate_copy.py`, `migration.log`), on a copy of the real `argus_library.sqlite` made with SQLite's online backup API from a read-only (`mode=ro`) connection, in the scratch folder and deleted afterwards:
  - the row count of every table is the same before and after the migration and after a second open;
  - every row's pre-existing columns are unchanged;
  - the v4 columns were added, and `PRAGMA integrity_check` is `ok` throughout;
  - the second open changed nothing.
- Review Focus 1 on real data (`migration.log`): `ensure_in_checkpoint` ran on a backup copy of the real `argus_checkpoints.sqlite`, made the same way and deleted afterwards. Every per-run thread got the legacy config for its own length, with one `errors` line, `require()` held, and the second call wrote nothing; their `next` was unchanged. The one per-chat thread from before per-run threads (`tg:<chat>`, no library row, so no entry point can reach it) had its `next` moved by the update with no `as_node` (from finished to `brief`); see the task report.
- Static scan (WP-A's stdlib `ast` scanner, `static_scan.log`): no new findings. The unused imports it lists in edited older files were already there at the base commit.

### Final review fix (2026-10-21): a failed `Library.open()` closes its connection (M1)

`Library.open()` bound the aiosqlite connection to the library only after the schema script and the migration, so one that raised (a corrupt or read-only DB) left the connection and its worker thread open. A failure there now closes the connection and re-raises; the library stays unopened. Test: `tests/test_library.py`, a migration that raises leaves its connection closed and `_conn()` refusing. Logs: `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\final-fix\`.
