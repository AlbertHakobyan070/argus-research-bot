# WP-E1 — Bot UX: run driver (drain cancel, one live drive), progress card, plan knobs, quality line (Task 10)

Spec: `docs/argus-v4-design.md` §3 WP-E1 and §2.5 (the lifecycle invariants and the `since()`-only bus), with Review Focus 5 (cancel while a node is busy) and 8 (two approvals race), and the pointers earlier tasks parked for WP-E1. Built on `81aa109`, which carries the ruling that adds `"any"` to RunConfig's `recency`.

### What changed

- `src/argus/bot/runs.py` (new): the run driver. Handlers and the console call it and relay `Outcome(code, text, run_id)`; `text` is empty when the driver's own messages already say it.
  - **Registry**: `common._inflight` holds a `RunHandle` per tracked run: the task, a fresh `RunControl` per drive, a per-run `asyncio.Lock`, the stage, the drive's start, the progress card, the gate message, the plan-gate choices and the cached profile checks. A run is tracked while it drives or rests at a gate. Every action takes the run's lock and checks the registry first. A second actor gets `already_running`; Cancel is the exception, since it is always allowed.
  - **Busy**: a start, approve, re-plan, Extend, Revise or `/continue` resume while another run drives returns `busy`. The text names that run's id, its topic (cut) and its stage, and says to try again when it reaches a gate. Runs resting at a gate do not count. The claim is made before the first await, so concurrent starts cannot both pass.
  - **Drives** are plain `asyncio.create_task` with `stream_mode=["tasks", "updates", "custom"]`, `durability="sync"` and `control=RunControl()`. Each drive publishes:
    - every node `custom` event as it arrives;
    - a `stage start` per task;
    - a `stage end` per update, unless the node published its own `stage end`.

    The drive ends at a gate (the plan card or the report preview, which become `gate_msg`), at the end, on `GraphDrained`, or on an error. Every drive end writes `runs.exa_usd`, calls `bus.finish` and forgets the run unless it rests at a gate.
  - **Cancel**: `cancel_run` calls `request_drain("user cancel")` and publishes `stage draining`; the card then shows "⏹ cancelling — finishing the current step". `GraphDrained` at the next boundary does the following:
    - writes `cancel_note = "cancelled after node <node> (<reason>)"`;
    - sets status `cancelled`;
    - publishes `stage cancelled`;
    - edits the card to "⏹ Cancelled after <node> — /continue <id> resumes it".

    A deadline task calls `task.cancel()` after `ARGUS_CANCEL_DRAIN_TIMEOUT_S`, with a WARNING ("the drain did not finish within … s"). The drive's end cancels the deadline. A run resting at a gate is cancelled at once: status `cancelled`, and the gate message edited without its keyboard.
  - **Shutdown**: `drain_all` (a POST_STOP hook) does three things:
    - requests a drain (`"shutdown"`) on every live drive;
    - waits up to `ARGUS_SHUTDOWN_DRAIN_S`;
    - cancels what still runs, with a WARNING ("still running … s after the shutdown drain"), and awaits it.
  - **Actions**: `start_run`, `approve_plan` (rebuilds the RunConfig with the knobs through `rc.rebuild` into the checkpoint, then the library row), `edit_plan`, `await_reply` (arms the Edit / Revise reply), `report_action` (send · extend · revise · cancel), `set_plan_choice`, `continue_run`. Every resume resolves before it mutates: the engines snapshot, `ensure_in_checkpoint` (the legacy injection), and this graph's nodes. A refusal returns `refused` with "⚠️ Could not <action> run `<id>`: <error>" and leaves the run as it was.
  - **Views**: `plan_payload` (the plan card's data, also the console's), `snapshot` (`in_flight`, `awaiting` = the gate only while no drive is live, `next`, `stage`, `last_seq`, the checkpoint values), `console_url`, `tracked`.
- `src/argus/bot/progress.py` (new):
  - `RunEvent` (frozen);
  - `RunEventBus`:
    - `publish` numbers events per run;
    - `since(after, limit)` reads them;
    - `last_seq`;
    - `finish` keeps the buffer an hour, then drops it; seq keeps rising, and a later publish revives the buffer;
    - a ring buffer per run;
  - `describe(event)`: one line per event kind;
  - `TelegramCard`: one message per drive. It polls `since()` every 3 s and edits the message only when the text changed. It shows the stage, the counts (fetched / digested / sections / judge batches, with failures), the last event, the cancelling line, and a final line from `stop()`. A Telegram failure is logged and retried on the next poll.
- `src/argus/bot/research.py`:
  - The handlers (`/research`, `/ask`, the `len` / `dep` / `prof` / `rec` / `plan` / `report` buttons, the Edit / Revise replies) only call the driver and relay its outcome.
  - The drive code (`_resume_after_plan`, `_replan_after_edit`, `_revise_after_feedback`, `_resume_after_report`, `_finalize_run`, `_ensure_run_config`, `_approve_run_config`) moved into `runs.py`.
  - `_format_plan(payload)`:
    - the header carries the depth ("(auto)" while it follows the length);
    - an "Archetype · Recency" line;
    - an "Engines · Exa" line, with the profile's problem behind "⚠" and the Exa estimate as "≈ $…", "n/a" (no prices) or "off";
    - a rebuild error.
  - `_plan_keyboard(payload)`:
    - the lengths, then the depths (✅ on the current);
    - `⚙ <profile>` and `🕒 <recency>` (each tap cycles);
    - Approve / Edit / Cancel;
    - no console button.
  - `_send_report_preview(bot, chat_id, run_id, values)` returns `(message, text, parse mode)`. Its head holds `_quality_lines`: "🧪 Quality: judged <j>/<n> · <k> cut", the report's `> ⚠ …` banners without the `> `, and this round's degradation notes (`render_degradation_notes`, the first few, then a count).
- `src/argus/bot/history.py`:
  - `/status`: each tracked run's stage, elapsed time, last event and cancelling state, or the gate it waits at, then the recent runs, then "Console: <url> (ok|down)" or "Console: not running";
  - `/cancel` calls `cancel_run` per run picked;
  - `/continue` passes the pending appended sources and an `on_report` callback (`_mark_appended`) to `continue_run`.
- `src/argus/bot/common.py`:
  - `HELP_TEXT` lists `/engines` (pointer 11);
  - `_inflight` holds `RunHandle`s;
  - `_chat_entries` is gone (`runs.tracked`).
- `src/argus/bot/app.py`:
  - `POST_INIT_HOOKS.append(runs.init)` and `POST_STOP_HOOKS.append(runs.drain_all)`;
  - `_run_hooks`, used by `_post_stop` and `_post_shutdown`, runs every hook, logs a failing one and re-raises the first failure;
  - `_post_shutdown` closes the library and the saver in a `finally` (pointer 1).

Approved edits outside WP-E1's files (the `recency` ruling):

- `src/argus/graph/run_config.py`: `recency: Literal["d", "w", "m", "y", "any"] | None`.
- `src/argus/engines/search.py`: `resolve_recency` returns `(None, None)` for `"any"`. That means no recency filter even for a time-sensitive brief, and nothing is dropped as undated.

### Decisions where the spec is silent

- **The busy rule covers `/ask` too**: it spends the same free-tier quota.
- **A gate wins over a drain only in LangGraph.** A cancel during the plan drive reaches the plan gate first, because `interrupt_after=["scout"]` fires before the drain boundary. The driver still treats it as a cancel: `cancel_note` names scout, status is `cancelled`, no plan card appears, and `/continue` re-attaches the plan. The same applies at the report gate.
- **A forced `task.cancel()`** (after the drain bound or at shutdown) sets status `cancelled` and writes no `cancel_note`. The node's thread may still be running, and the checkpoint is the one from before it. `/continue` re-runs that node. That node's own Exa spend is not in `runs.exa_usd`.
- **Actions need a tracked run.** After a restart, a button on an old message answers that the run is not in flight and that `/continue <id>` re-attaches it. `continue_run` registers the run and re-sends its plan card or preview. `snapshot()` still reports an untracked run's gate as `awaiting`, by the spec formula.
- **`/continue` on a stopped run** resumes from its checkpoint (`astream(None)`, status `running`). A stopped run is one whose `next` is not a gate: cancelled, or failed. Before, `/continue` answered "mid-pipeline — wait"; the registry now tells a live drive from a stopped one. A quick `/ask` run is refused.
- **Extend and Revise are refused at their round caps** (`MAX_EXTEND_ROUNDS`, `MAX_REVISE_ROUNDS`). Past the cap the graph would end the run silently.
- **Older-topology checkpoints**: a checkpoint whose `branch:to:*` channels name nodes this graph lacks is refused on every resume (approve, re-plan, Extend, Revise, `/continue` resume or fork). These are v2 threads. Re-attaching to such a checkpoint and Send still work (pointer 5).
- **The profile knob** cycles the engines profiles. Each non-active profile is resolved once per run when the cycle first reaches it, and its snapshot or its problem is cached. Approve uses the cached snapshot. The active profile is resolved at Approve itself (`active_snapshot()`). A run approved on another profile carries that profile's snapshot in `rc.engines`, and the global active profile is unchanged.
- **The recency knob** is the `recency` override (`None` = the brief decides). The scout decides recency once, and research reuses `search_spend["recency"]`. So an Approve that changes recency also rewrites `search_spend["recency"]` (`resolve_recency` on the rebuilt config) and keeps the rest of the scout's spend.
- **Depth "auto"**: a run whose depth equals its length's default depth follows the length until a depth is tapped. After that the tapped depth stays through length changes.
- **The driver owns the gate messages**: the plan card, the preview, and their edits on approve, edit, Extend, Revise, Send, Cancel and knob changes. An action from the console edits the Telegram message the same way ("✅ approved from console" and the like). Taps on an older plan card act on the run, and the current card is the one edited.
- **Stage events**: the driver publishes `stage start` per task. A node that publishes its own start (scout, research) therefore shows two starts; the second carries the node's summary.
- **The spend write comes after the gate presentation**, so a failed write is the last chat line. The preview has already been sent and the status is set.
- **A node event's `node` field** falls back to the running stage when a node omits it.

### Behaviour changes for handoff.md

- `/research` and `/ask` start with a live progress card (stage, counts, last event), not a series of progress messages. The card's last edit says how the drive ended. `/ask`'s answer arrives as a chat message below its card.
- While one run executes, a second `/research`, `/ask`, Approve, Edit, Extend, Revise or `/continue` resume is refused as "⏳ busy — run `<id>` (`<topic>`) is at `<stage>`; try again when it reaches a gate". A second tap on a running run answers "already running".
- `/cancel` (and a Cancel button) stops a running drive at the end of its current step, with the checkpoint saved. `/continue <id>` resumes it. A run resting at a gate is cancelled at once. Before, `/cancel` only forgot the run while its graph kept going.
- `/status` shows each run's stage, elapsed time and last event, and the console URL and its health.
- The plan card has a depth row, a `⚙ <profile>` button and a `🕒 <recency>` button. The header names the depth, and two lines name the archetype, the recency window, the engines profile (with its problem) and the Exa estimate. The estimate shows only when Exa is enabled (the key, and `exa` in the run's providers). Recency `any` switches the filter off.
- The report preview opens with the quality line, the "⚠ Not fact-checked" / "⚠ Degraded run" banners and the degradation notes.
- Send, Extend, Revise, Approve and Cancel edit their gate message (the keyboard goes). A failed resume (engines unresolvable, a broken checkpoint, a cap) is refused in the chat, and the run stays at its gate. Before, it reached only the log.
- A failed drive, including a failed re-plan, reports "⚠️ Research failed: <error>", sets status `error` and leaves the run resumable with `/continue`. A re-plan failure used to say "⚠️ Re-plan failed" and left the run between nodes.
- Shutdown (`post_stop`) drains live drives for at most `ARGUS_SHUTDOWN_DRAIN_S`, then cancels them. A drained run is `cancelled` and resumable.
- `/help` lists `/engines`.

### New env vars

None new. `ARGUS_CANCEL_DRAIN_TIMEOUT_S` (default 300) and `ARGUS_SHUTDOWN_DRAIN_S` (default 120) were added to `Settings` by WP-D and are read by the driver from this WP on.

### Test files touched

- New: `tests/test_runs_driver.py`, which runs the real graph with stubbed nodes on a MemorySaver, a real Library and a mock bot. It covers:
  - cancel while a node is busy, with a node that ignores `runtime` and with the research pattern, plus `/continue` to completion;
  - the fake-clock `task.cancel()` fallback;
  - the deadline cancelled by a drive that ends in time;
  - a cancel during the plan drive;
  - busy;
  - the two-approvals race;
  - the shutdown drain, both the bounded and the in-time case;
  - plain tasks;
  - failures reported;
  - `exa_usd` on every drive end, a failed spend write, and the spend write after the preview (moved from `test_cut_not_flag.py`);
  - a refused approve;
  - the older topology;
  - the Extend cap;
  - Send;
  - the depth, recency and profile knobs and a bad knob;
  - the Exa estimate, also with Exa left out of the providers;
  - the console approve sync;
  - the preview quality line;
  - `/status` and `/cancel`.
- New: `tests/test_progress_bus.py`. It covers bus ordering and limits, the ring buffer, `finish` expiry and revival, frozen events, `describe` for every kind, and the card (render, cancelling, own-drive events only, stop, a failed send retried).
- `tests/helpers.py`: `settle_runs()` awaits the drives the driver still holds.
- `tests/test_bot.py`:
  - `_format_plan` / `_plan_keyboard` take the payload;
  - new tests cover the `/engines` help line, the keyboard rows, the Exa and recency lines, the hook runner (pointer 1) and the hook registration.
- `tests/test_plan_hitl_gating.py`: rewritten on the driver, with the same contracts. Length taps re-render with the keyboard and never resume, Approve resumes once (also double-tapped), Edit keeps the keyboard, and notices go to unknown and stale buttons.
- `tests/test_hitl_edit_revise.py`: the bot-wiring tests drive the real (stubbed) graph through the driver.
- `tests/test_continue_extend.py`: the `/continue` tests drive through `continue_run`; the graph double streams and merges updates.
- `tests/test_run_scoping.py`: the graph double takes the driver's stream kwargs and reports `next`; a second `/research` waits for the first drive to reach its gate.
- `tests/test_engines_cmd.py`: the drive entries map to the driver (approve, re-plan, Send, Extend, Revise and the three commands). A refusal now reaches the chat, not the error handler.
- `tests/test_run_config_resume.py`:
  - every resume path runs through the driver; an attached run is a `RunHandle` at its gate;
  - refusals are chat messages that change nothing (no raise);
  - the brief-exhausted failure is the drive's "⚠️ Research failed".
- `tests/test_bot_markdown_fallback.py`: the HTML-wiring check reads `_send_report_preview`.
- `tests/test_cut_not_flag.py`: the run-driver test moved to `test_runs_driver.py`.

### Verification evidence (2026-10-20)

Scripts and logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t10\`. Every suite run used `FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1`, no test started the bot, and no test called a live service.

- RED (`red.log`): with only the new tests written, both files failed at collection (`ModuleNotFoundError: No module named 'argus.bot.progress'`; `ImportError: cannot import name 'runs' from 'argus.bot'`).
- GREEN: the focused files pass (`green.log`). The full suite (`suite-final.log`) passes except the live-proxy tests (`tests/test_llm.py` and the `tests/test_graph.py` fixture). Those fail on the closed port by design, as in the baseline (`baseline-suite.log`).
- The cancel-during-the-plan-drive test fails with the fix disabled: the run stops at `awaiting_plan`.
- `probe_tasks_stream.py`: the `tasks` stream's start and result chunks, a drain after the busy node (`next` names the following node), and the resume to the end. `probe_cancel.py`: `task.cancel()` ends a drive at once while its node's thread is still blocked.
- `probe_v2_extend.py` / `probe_v2_revise.py`, run on copies of the real checkpoint DB: a v2-topology report-gate thread resumes to the same gate without running a node. That is why Extend and Revise refuse such threads.
- The live checks (cancel while busy, two approvals, `/continue` after a restart) are the orchestrator's.
