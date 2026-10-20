# WP-E1 — Bot UX: run driver (drain cancel, one live drive), progress card, plan knobs, quality line (Task 10)

Spec: `docs/argus-v4-design.md` §3 WP-E1 and §2.5 (the lifecycle invariants and the `since()`-only bus), with Review Focus 5 (cancel while a node is busy) and 8 (two approvals race), and the pointers earlier tasks parked for WP-E1. Built on `81aa109`, which carries the ruling that adds `"any"` to RunConfig's `recency`.

### What changed

- `src/argus/bot/runs.py` (new): the run driver. Handlers and the console call it and relay `Outcome(code, text, run_id)`; `text` is empty when the driver's own messages already say it.
  - **Registry**: `common._inflight` holds a `RunHandle` per tracked run: the task, a fresh `RunControl` per drive, a per-run `asyncio.Lock`, the stage, the drive's start, the progress card, the gate message, the plan-gate choices and the cached profile checks. A run is tracked while it drives or rests at a gate. Every action takes the run's lock and checks the registry first. A second actor gets `already_running`; Cancel is the exception, since it is always allowed.
  - **Attach on action** (§2.5 ruling, fix round 1): a restart empties the registry, so an action on a run that is not tracked attaches it first: the library row, the checkpoint's gate, `ensure_in_checkpoint`, a registered handle with that gate and the default plan choices. Then the action proceeds. Every action routes through this (`_hold`): approve, Edit / `await_reply`, the knobs, cancel, Send, Extend, Revise, and the same calls from the console. A tap passes its message id, so the tapped card becomes the gate message; its text is rendered from the payload at the first edit. An attach that cannot resolve (the engines snapshot, a checkpoint without a length) is that action's refusal. Only a run with no library row, or one not resting at a gate, answers "not in flight … /continue".
  - **Busy**: a start, approve, re-plan, Extend, Revise or `/continue` resume while another run drives returns `busy`. The text names that run's id, its topic (cut) and its stage, and says to try again when it reaches a gate. Runs resting at a gate do not count. The claim is made before the first await, so concurrent starts cannot both pass. An exception between the claim and the launch resets the claim (a `finally`) and reaches the chat as the action's refusal. The Edit and Revise buttons check busy (and Revise its round cap) before they arm the reply.
  - **Drives** are plain `asyncio.create_task` with `stream_mode=["tasks", "updates", "custom"]`, `durability="sync"` and `control=RunControl()`. Each drive publishes every node `custom` event as it arrives, plus each task's stage transitions once:
    - the driver's `stage start` per task (a node's own `start`, which follows it, is dropped);
    - the driver's `stage end` per update, unless the node published its own `stage end` first;
    - one `stage draining` (the driver's; research's own one is dropped).

    The drive ends at a gate (the plan card or the report preview, which become `gate_msg`), at the end, on `GraphDrained`, or on an error. Every drive end writes `runs.exa_usd` (and `runs.archetype` once the brief exists), calls `bus.finish` and forgets the run unless it rests at a gate. Launching a drive clears the chat's pending Edit / Revise reply for that run.
  - **Cancel**: `cancel_run` calls `request_drain("user cancel")` and publishes `stage draining`; the card then shows "⏹ cancelling — finishing the current step". `GraphDrained` at the next boundary does the following:
    - writes `cancel_note = "cancelled after node <node> (<reason>)"`;
    - sets status `cancelled`;
    - publishes `stage cancelled`;
    - edits the card to "⏹ Cancelled after <node> — /continue <id> resumes it".

    A deadline task calls `task.cancel()` after `ARGUS_CANCEL_DRAIN_TIMEOUT_S`, with a WARNING ("the drain did not finish within … s"). The deadline is bound to the drive it was armed for: the end of that drive cancels it, and a later drive's launch cancels any left over. A cancel accepted after the stream returned (while the gate is presented, during the report-gate callback or the spend write) finds no boundary to drain at: once the presentation is done, the run gets the gate cancel. A run resting at a gate is cancelled at once: status `cancelled`, and the gate message edited without its keyboard. `/cancel <id>` finds an untracked run through the library, as `/continue` does.
  - **Shutdown**: `drain_all` (a POST_STOP hook) does four things:
    - refuses new drives from then on: a claim or a launch answers "⏹ Argus is shutting down — try again after the restart";
    - requests a drain (`"shutdown"`) on every live drive;
    - waits up to `ARGUS_SHUTDOWN_DRAIN_S`;
    - cancels what still runs, with a WARNING ("still running … s after the shutdown drain"), and awaits it.
  - **Actions**: `start_run` (with `engines_profile`, §2.7), `approve_plan` (rebuilds the RunConfig with the knobs through `rc.rebuild` into the checkpoint, then the library row), `edit_plan`, `await_reply` (arms the Edit / Revise reply), `report_action` (send · extend · revise · cancel), `set_plan_choice`, `continue_run`. Every resume resolves before it mutates: the engines snapshot, `ensure_in_checkpoint` (the legacy injection), and this graph's nodes. The console's knobs on `approve_plan` (`length`, `depth`, `engines_profile`) reach the choices only after everything resolved. A refusal returns `refused` with "⚠️ Could not <action> run `<id>`: <error>" and leaves the run as it was.
  - **Send ends the run**: `_deliver` sends the files, then `aupdate_state(…, {}, as_node="deliver")` moves the checkpoint past the report gate to END. A sent run is finished (`next == ()`), never awaiting.
  - **Pinned appends are always ingested** (§2.4): `/continue` on a finished run forks the append-only pass with exactly its pending `/append` sources, and first moves a `done` run still resting at `("deliver",)` (sent by an older bot) past its gate. A report-gate Extend runs its normal pass with the pending appends merged in as pinned sources. The next report gate marks each one ingested or failed, with the fork's bookkeeping (`history.appends_marker`). Any other resume or re-attach (approve, re-plan, Revise, Send, `/continue` re-attach or resume) leaves them pending and says so in the chat: their count, and the path that ingests them.
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
  - `edit_text`, the edit both the card and the gate messages use: Telegram's "message is not modified" counts as success, a `RetryAfter` waits its `retry_after` and tries once more, and a message that no longer exists is reported so the caller logs it once and stops editing it.
- `src/argus/bot/research.py`:
  - The handlers (`/research`, `/ask`, the `len` / `dep` / `prof` / `rec` / `plan` / `report` buttons, the Edit / Revise replies) only call the driver and relay its outcome.
  - The drive code (`_resume_after_plan`, `_replan_after_edit`, `_revise_after_feedback`, `_resume_after_report`, `_finalize_run`, `_ensure_run_config`, `_approve_run_config`) moved into `runs.py`.
  - `_format_plan(payload)`:
    - the header carries the depth ("(auto)" while it follows the length);
    - an "Archetype · Recency" line;
    - an "Engines · Exa" line, with the profile's problem behind "⚠" and the Exa estimate as "≈ $…", "n/a" (no prices) or "off";
    - a rebuild error.
  - `_plan_keyboard(payload)`:
    - the lengths, then the depths and Auto (✅ on the current; on Auto while the depth follows the length);
    - `⚙ <profile>` and `🕒 <recency>` (each tap cycles);
    - Approve / Edit / Cancel;
    - no console button.
  - `_send_report_preview(bot, chat_id, run_id, values)` returns `(message, text, parse mode)`; `_preview_bodies(values)` builds its HTML and plain bodies (also for a preview attached from a tap). Its head holds `_quality_lines`: "🧪 Quality: judged <j>/<n> · <k> cut", the `> ⚠ …` banners of the report's title block (the lines before its first `## ` heading) without the `> `, and this round's degradation notes (`render_degradation_notes`, the first few, then a count).
  - A button passes its message id to the driver (the gate message of a run it attaches). An Edit / Revise reply that is refused after it was consumed says so: "(your reply was not used — tap Edit/Revise and send it again)".
- `src/argus/bot/history.py`:
  - `/status`: each tracked run's stage, elapsed time, last event and cancelling state, or the gate it waits at, then the recent runs, then "Console: <url> (ok|down)", "Console: down — <webapp_error>" when `webapp_ok` is false, or "Console: not running". `webapp_public_url`, `webapp_ok` and `webapp_error` are the WP-E2 contract (§2.7): WP-E2 sets them.
  - `/cancel` calls `cancel_run` per run picked; a run id that is not tracked is resolved through `lib.resolve_run`;
  - `/continue` passes the pending appended sources and an `on_report` callback to `continue_run`; `pending_appends` and `appends_marker` (`_mark_appended`) are shared with the driver's report-gate Extend.
- `src/argus/bot/common.py`:
  - `HELP_TEXT` lists `/engines` (pointer 11);
  - `_inflight` holds `RunHandle`s;
  - `_chat_entries` is gone (`runs.tracked`).
- `src/argus/bot/app.py`:
  - `POST_INIT_HOOKS.append(runs.init)` and `POST_STOP_HOOKS.append(runs.drain_all)`;
  - `_run_hooks`, used by `_post_stop` and `_post_shutdown`, runs every hook, logs a failing one and re-raises the first failure;
  - `_post_stop` stores that first failure instead of raising it: PTB skips `shutdown()` and `post_shutdown` when `post_stop` raises, so the library and saver would never close (pointer 1, the POST_STOP half);
  - `_post_shutdown` closes the library and the saver in a `finally` (pointer 1), then raises the stored stop-hook failure, or its own first one;
  - `_reconcile_orphaned_runs` (approved edit): a run whose checkpoint `next` is not a gate gets status `cancelled` and "♻️ Run `<id>` (<topic>) was interrupted by a bot restart mid-step. /continue <id> resumes it from its last saved step."; the "can't be resumed … /research it again" error is left only for a run with no checkpoint.

Approved edits outside WP-E1's files (the `recency` ruling):

- `src/argus/graph/run_config.py`: `recency: Literal["d", "w", "m", "y", "any"] | None`.
- `src/argus/engines/search.py`: `resolve_recency` returns `(None, None)` for `"any"`. That means no recency filter even for a time-sensitive brief, and nothing is dropped as undated.

### Decisions where the spec is silent

- **The busy rule covers `/ask` too**: it spends the same free-tier quota.
- **A gate wins over a drain only in LangGraph.** A cancel during the plan drive reaches the plan gate first, because `interrupt_after=["scout"]` fires before the drain boundary. The driver still treats it as a cancel: `cancel_note` names scout, status is `cancelled`, no plan card appears, and `/continue` re-attaches the plan. The same applies at the report gate.
- **A forced `task.cancel()`** (after the drain bound or at shutdown) sets status `cancelled` and writes `cancel_note = "force-cancelled during node <node> (<reason>); the node was abandoned"`. The node's thread may still be running, and it never wrote: the checkpoint is the one from before it, and `next` names that node. The card says what `/continue` then does. A forced cancel inside research rests at `("research",)`, which is the plan gate: `/continue` re-attaches the plan, and Approve reruns research. Elsewhere `/continue` reruns the node. That node's own Exa spend is not in `runs.exa_usd`.
- **Untracked runs attach on action** (§2.5 ruling; this replaced "actions need a tracked run"). After a restart, a button on an old message attaches its run when the run rests at a gate, and the action proceeds on that message. `snapshot()["awaiting"]` and the action's outcome agree. A run with no library row, or one not resting at a gate, still answers that the run is not in flight and that `/continue <id>` re-attaches it. `continue_run` registers the run and re-sends its plan card or preview. `plan_payload` stays a view of tracked runs.
- **`/continue` on a stopped run** resumes from its checkpoint (`astream(None)`, status `running`). A stopped run is one whose `next` is not a gate: cancelled, or failed. A run that failed or was force-cancelled inside research is not one of them: its `next` is `("research",)`, the plan gate, so `/continue` re-attaches its plan. Before, `/continue` answered "mid-pipeline — wait"; the registry now tells a live drive from a stopped one. A quick `/ask` run is refused.
- **Extend and Revise are refused at their round caps** (`MAX_EXTEND_ROUNDS`, `MAX_REVISE_ROUNDS`). Past the cap the graph would end the run silently.
- **Older-topology checkpoints**: a checkpoint whose `branch:to:*` channels name nodes this graph lacks is refused on every resume (approve, re-plan, Extend, Revise, `/continue` resume or fork). These are v2 threads. Re-attaching to such a checkpoint and Send still work (pointer 5), and an untracked one attaches on action under the same rules. `/continue` on a sent (`done`) one is refused before anything moves: the Extend fork it would take cannot run on an older graph.
- **The unused-appends notice** reads the library's pending rows while the action resolves; the fork and the report-gate Extend ingest them and say nothing. After Send the line names `/continue <id>` only.
- **The reconcile's `cancelled`** applies to any checkpoint whose `next` is not a gate, including `()` (a run whose Send finished the checkpoint but not the status write); `/continue` then extends it.
- **The profile knob** cycles the engines profiles. Each non-active profile is resolved once per run when the cycle first reaches it, and its snapshot or its problem is cached. A card render never validates live: the active profile's state is the cached snapshot (`current_snapshot_cached()`), another profile's the cached result. Approve validates the chosen profile live: the active one through `active_snapshot()`, another through `resolve_snapshot`, its result cached again. A run approved on another profile carries that profile's snapshot in `rc.engines`, and the global active profile is unchanged. `start_run(engines_profile=…)` validates its profile the same way at the start; it is the run's from the plan drive on and the card's profile choice.
- **The recency knob** is the `recency` override (`None` = the brief decides). The scout decides recency once, and research reuses `search_spend["recency"]`. So an Approve that changes recency also rewrites `search_spend["recency"]` (`resolve_recency` on the rebuilt config) and keeps the rest of the scout's spend.
- **Depth "auto"**: a run whose depth equals its length's default depth follows the length until a depth is tapped. After that the tapped depth stays through length changes.
- **The driver owns the gate messages**: the plan card, the preview, and their edits on approve, edit, Extend, Revise, Send, Cancel and knob changes. An action from the console edits the Telegram message the same way ("✅ approved from console" and the like). Taps on an older plan card act on the run, and the current card is the one edited.
- **Stage events**: each task's transitions are published once. The driver's `stage start` comes first, so a node's own start (scout, research) is dropped with its summary; a node's own `stage end` comes before the update, so it wins over the driver's.
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
- Shutdown (`post_stop`) drains live drives for at most `ARGUS_SHUTDOWN_DRAIN_S`, then cancels them. A drained run is `cancelled` and resumable. From the start of the shutdown, a new drive is refused with "⏹ Argus is shutting down — try again after the restart". A failing stop hook no longer keeps the library and saver open: it is raised after they close.
- `/help` lists `/engines`.
- `/research` and `/ask` refuse the start when the library cannot record the run ("⚠️ Could not start the research: <error>"); before, a failed row write was only logged ("continuing without registry row") and the run went on.

Fix round 1 (Task 10 review):

- Buttons on messages sent before a restart keep working: the run attaches and the action proceeds on that message. `/cancel <id>` works for such a run too. Only a run with no library row, or one not resting at a gate, still gets "not in flight … /continue".
- Send finishes the run: the checkpoint moves past the report gate, so `/continue` extends a sent run. A run an older bot sent (status `done`, still at its report gate) is moved past the gate by `/continue`, then extended.
- `/append` sources are always ingested: `/continue` on a finished run ingests exactly them, and a report-gate Extend merges them into its normal pass. The next report gate marks each one ingested or failed. Approve, Edit, Revise, Send, a `/continue` re-attach or resume leave them pending and say how many and what ingests them.
- The restart reconcile marks a run interrupted mid-drive `cancelled` and DMs "♻️ Run `<id>` (<topic>) was interrupted by a bot restart mid-step. /continue <id> resumes it from its last saved step." Before, it was `error` with "/research it again", which now applies only to a run with no checkpoint.
- A cancel that lands while the plan card or the preview is being sent cancels the run at its gate once it is shown. Before, the run stayed at its gate after "Cancelling…", and that cancel's deadline could later cancel the next drive.
- A forced cancel writes its `cancel_note`, and its card says what `/continue` will do.
- `/status` shows why the console is down (`webapp_error`) when `webapp_ok` is false.
- The plan card's depth row has an Auto button. A knob tap that changes nothing keeps the card's formatting (it used to flip it to raw Markdown). A flood-limited edit waits Telegram's `retry_after`, and a deleted card is no longer edited.
- Edit and Revise answer busy (and Revise its round cap) when tapped, before they wait for a reply. A reply that is refused anyway says it was not used.
- Approving a plan clears a pending Edit reply, so the next pasted link reaches the media menu.
- The preview's "⚠" banner line reads only the report's title block; a quoted "⚠" in the body is not a banner.
- A failure to mark the appended sources at the report gate reaches the chat.
- `runs.archetype` is written at every drive end once the brief exists.

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

Fix round 1:

- `tests/test_runs_driver.py`:
  - the fixture also sets the handlers' settings and ACL, caches the active snapshot as the real `active_snapshot()` does, and restores `runs._app` and `runs._shutting_down` after each test;
  - existing tests: the two forced-cancel tests assert the note, `next` and the card's `/continue` line; the spend-write order now carries the archetype (R-b); the older-topology run is untracked at its report gate (R-a), and its sent (`done`) case is a test of its own;
  - new tests: attach on action for approve, an Edit reply, a knob, cancel, Send, Extend and Revise, with the notice kept for a run that is not at a gate, and `/cancel <id>` on an untracked run; a cancel during the plan presentation; an old deadline and a later drive; the archetype; busy for `/ask`, a `/continue` resume, Edit, Extend and Revise; a cancel racing an approve's preparation; Send ending the run; `/append` + `/continue` on a sent run; the older bot's sent run advanced then forked; a report-gate Extend with appends; the unused-appends notice on a re-attach, on Approve and on Send; a failed report-gate callback; `start_run(engines_profile=…)`; `/status` at a gate and with the console down; the stage sequence of a drive and a single drain event; a delivered run's buffer expiry; the gate edit's "not modified", `RetryAfter` and deleted message; a raising gate edit; a refused console approve; the shutdown refusal, also for a drive prepared across it; Edit / Revise busy and cap checks; a refused reply; a launch clearing the reply; renders that never validate live; the `dep:` / `prof:` / `rec:` buttons through `on_callback`.
- `tests/test_progress_bus.py`: `since()`'s default limit; the card's "not modified", `RetryAfter` and deleted message.
- `tests/test_bot.py`: the keyboard-rows test's depth row ends with `dep:auto`; the Auto check; title-block banners; the HTML-escaped quality line; the `_post_stop` → `_post_shutdown` path; the hook-runner docstring no longer claims a raising stop hook still closes the saver.
- `tests/test_orphan_reconcile.py`: the mid-pipeline orphan is `cancelled` with the new DM (it was the `error` case); the `error` text is a no-checkpoint case.
- `tests/test_run_config_resume.py`: Send's `next` is `()`; every action path also runs untracked (attach on action); `_FAIL_ENTRIES` gains the re-plan and Revise replies and `/continue` at the plan and report gates.

### Verification evidence (2026-10-20)

Scripts and logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t10\`. Every suite run used `FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1`, no test started the bot, and no test called a live service.

- RED (`red.log`): with only the new tests written, both files failed at collection (`ModuleNotFoundError: No module named 'argus.bot.progress'`; `ImportError: cannot import name 'runs' from 'argus.bot'`).
- GREEN: the focused files pass (`green.log`). The full suite (`suite-final.log`) passes except the live-proxy tests (`tests/test_llm.py` and the `tests/test_graph.py` fixture). Those fail on the closed port by design, as in the baseline (`baseline-suite.log`).
- The cancel-during-the-plan-drive test fails with the fix disabled: the run stops at `awaiting_plan`.
- `probe_tasks_stream.py`: the `tasks` stream's start and result chunks, a drain after the busy node (`next` names the following node), and the resume to the end. `probe_cancel.py`: `task.cancel()` ends a drive at once while its node's thread is still blocked.
- `probe_v2_extend.py` / `probe_v2_revise.py`, run on copies of the real checkpoint DB: a v2-topology report-gate thread resumes to the same gate without running a node. That is why Extend and Revise refuse such threads.
- The live checks (cancel while busy, two approvals, `/continue` after a restart) are the orchestrator's.

Fix round 1: scripts and logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t10-fix1\`, with the same environment and the same rules.

- RED (`red.log`): the new and updated tests, written first, fail against the unchanged code. `red-final-tests-vs-head.log` runs the final tests against an archived copy of HEAD's `src`.
- GREEN: the focused files pass (`focused-final.log`). The full suite (`suite-final.log`) passes except the same live-proxy tests, and no warnings summary appears.
- `probe_send_to_end.py`: after Send's `aupdate_state(…, as_node="deliver")`, the thread rests at END (`next == ()`), and the `/continue` fork from there goes to `extend_prep`. `probe_cancel_note_sqlite.py`: on the real `AsyncSqliteSaver`, a `cancel_note` written after `GraphDrained` persists.
- Not checked live: this round's rules forbid starting the bot or calling the bridge.
