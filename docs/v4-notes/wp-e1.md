# WP-E1 — Bot UX: run driver (drain cancel, one live drive), progress card, plan knobs, quality line (Task 10)

Spec: `docs/argus-v4-design.md` §3 WP-E1 and §2.5 (the lifecycle invariants and the `since()`-only bus), with Review Focus 5 (cancel while a node is busy) and 8 (two approvals race), and the pointers earlier tasks parked for WP-E1. Built on `81aa109`, which carries the ruling that adds `"any"` to RunConfig's `recency`.

### What changed

- `src/argus/bot/runs.py` (new): the run driver. Handlers and the console call it and relay `Outcome(code, text, run_id)`; `text` is empty when the driver's own messages already say it.
  - **Registry**: `common._inflight` holds a `RunHandle` per tracked run: the task, a fresh `RunControl` per drive, a per-run `asyncio.Lock`, the stage, the drive's start, the progress card, the gate message, the plan-gate choices and the cached profile checks. A run is tracked while it drives or rests at a gate. Every action takes the run's lock and checks the registry first. A second actor gets `already_running`; Cancel is the exception, since it is always allowed.
  - **Attach on action** (§2.5 ruling, fix round 1): a restart empties the registry, so an action on a run that is not tracked attaches it first: the library row, the checkpoint's gate, `ensure_in_checkpoint`, a registered handle with that gate and the default plan choices. Then the action proceeds. Every action routes through this (`_hold`): approve, Edit / `await_reply`, the knobs, cancel, Send, Extend, Revise, and the same calls from the console. A tap passes its message id, so the tapped card becomes the gate message when its button belongs to that gate (the knobs and `plan:*` to the plan gate, `report:*` to the report gate; fix round 2); its text is rendered from the payload at the first edit. A tapped card of the other gate gets that button's refusal and is never adopted or edited. An attach that cannot resolve (the engines snapshot, a checkpoint without a length) is that action's refusal. An actor that waited on a handle another actor then forgot (that actor's attach failed) looks the run up and attaches it again, once, and gets its own outcome. Only a run with no library row, or one not resting at a gate, answers "not in flight … /continue".
  - **Busy**: a start, approve, re-plan, Extend, Revise or `/continue` resume while another run drives returns `busy`. The text names that run's id, its topic (cut) and its stage, and says to try again when it reaches a gate. Runs resting at a gate do not count. The claim is made before the first await, so concurrent starts cannot both pass. An exception between the claim and the launch resets the claim (a `finally`) and reaches the chat as the action's refusal. The Edit and Revise buttons check busy (and Revise its round cap) before they arm the reply.
  - **Drives** are plain `asyncio.create_task` with `stream_mode=["tasks", "updates", "custom"]`, `durability="sync"` and `control=RunControl()`. Each drive publishes every node `custom` event as it arrives, plus each task's stage transitions once:
    - the driver's `stage start` per task (a node's own `start`, which follows it, is dropped);
    - the driver's `stage end` per update, unless the node published its own `stage end` first;
    - one `stage draining` (the driver's; research's own one is dropped).

    The drive ends at a gate (the plan card or the report preview, which become `gate_msg`), at the end, on `GraphDrained`, or on an error. Every drive end writes `runs.exa_usd` (and `runs.archetype` once the brief exists), calls `bus.finish` and forgets the run unless it rests at a gate. Launching a drive clears the chat's pending Edit / Revise reply for that run.
  - **Cancel**: `cancel_run` calls `request_drain("user cancel")` and publishes `stage draining`; the card then shows "⏹ cancelling — finishing the current step". `GraphDrained` at the next boundary does the following:
    - writes `cancel_note = "cancelled after node <node> (<reason>)"`, or "cancelled before it started (<reason>)" when no node of the drive finished. The note never moves `next` (fix rounds 3 and 4). It is written as the last node the drive finished, else as the route its drive launched with, else as LangGraph infers it. That route is the launching action's own (the re-plan's `intake`, the `/continue` fork's `deliver`); a plain `/continue` resume takes it from where it rests (the route table under "Resting-run writes keep `next`"). Inference, the last node that ran, is left for Approve, Extend, Revise and a resume from a state a real node produced;
    - sets status `cancelled`;
    - publishes `stage cancelled`;
    - edits the card to "⏹ Cancelled after <node> — <hint>." or "⏹ Cancelled before it started — <hint>.". The hint follows the checkpoint's `next` (fix round 3): at a gate, "/continue <id> re-attaches the plan card — repeat the action there" (or "the report card"), and elsewhere "/continue <id> resumes it from <node>". A quick `/ask`, which `/continue` refuses, gets no hint.

    A deadline task calls `task.cancel()` after `ARGUS_CANCEL_DRAIN_TIMEOUT_S`, with a WARNING ("the drain did not finish within … s"). The deadline is bound to the drive it was armed for: the end of that drive cancels it, and a later drive's launch cancels any left over. A cancel accepted after the stream returned (while the gate is presented, during the report-gate callback or the spend write) finds no boundary to drain at: once the presentation is done, the run gets the gate cancel. A run resting at a gate is cancelled at once: status `cancelled`, and the gate message edited without its keyboard. A tapped Cancel on a card of the other gate is refused ("Run `<id>` is not at its plan gate." / "… report preview."), as that card's other buttons are. `/cancel <id>` finds an untracked run through the library, as `/continue` does.
  - **Shutdown** (fix round 2): `drain_all` (a POST_STOP hook) does the following, all within `ARGUS_SHUTDOWN_DRAIN_S`:
    - refuses new claims from then on: an action claimed after the start answers "⏹ Argus is shutting down — try again after the restart" before it writes anything;
    - waits for the actions claimed before the start that still prepare. Each resume (approve, re-plan, Extend, Revise, `/continue`) launches its drive, and the launch requests the drain (`"shutdown"`) at once, so its first boundary drains it. A start (`/research`, `/ask`, the console's) is refused at its launch instead, with the same text, and its library row is marked `cancelled`: a fresh thread drained before its first node keeps no input (its values hold no `run_config` or `length`), so it could not be resumed, and its cancel note would be an ambiguous update;
    - requests a drain (`"shutdown"`) on every other live drive and waits for them;
    - cancels what still runs, with a WARNING ("still running … s after the shutdown drain"), and awaits it.

    No action ends with its writes (status, request flags, gate edits) and no drive. There are two exceptions. A refused start's row is marked `cancelled`. A preparation that outlasts the bound is logged ("still preparing … s after the shutdown began"), its launch is refused, and its writes stay (a `ponytail:` ceiling in `_launch`).
  - **Actions**: `start_run` (with `engines_profile`, §2.7), `approve_plan` (rebuilds the RunConfig with the knobs through `rc.rebuild` into the checkpoint, then the library row), `edit_plan`, `await_reply` (arms the Edit / Revise reply), `report_action` (send · extend · revise · cancel), `set_plan_choice`, `continue_run`. Every resume resolves before it mutates: the engines snapshot, `ensure_in_checkpoint` (the legacy injection, written as the route of where the run rests, so it keeps `next`; fix round 4), and this graph's nodes. The console's knobs on `approve_plan` (`length`, `depth`, `engines_profile`) reach the choices only after everything resolved. A refusal returns `refused` with "⚠️ Could not <action> run `<id>`: <error>" and leaves the run as it was.
  - **Send ends the run**: `_deliver` sends the files, then `aupdate_state(…, {"extend_requested": False, "revision_requested": False}, as_node="deliver")` moves the checkpoint past the report gate to END. A sent run is finished (`next == ()`), never awaiting. Every report-gate action writes both request flags (fix round 2): Extend sets `extend_requested` and clears `revision_requested`, Revise the reverse, and Send clears both.
  - **Pinned appends are always ingested** (§2.4): `/continue` on a finished run forks the append-only pass with exactly its pending `/append` sources, and first moves a `done` run still resting at `("deliver",)` (sent by an older bot) past its gate. A report-gate Extend runs its normal pass with the pending appends merged in as pinned sources. The next report gate marks each one ingested or failed, with the fork's bookkeeping (`history.appends_marker`). Both merges are idempotent (fix round 3) and keep the pin (fix round 4): `history.merged_sources` never adds an appended source the run's `sources` already carry as a pinned copy (research's own `_is_pinned`), matched as research dedupes them (the canonical URL). An unpinned copy, a URL the scout found, does not count: the pinned copy is merged, wins research's dedupe and is fetched, so an append of a URL the scout found but research never fetched is ingested. So an Extend or a fork drained or refused after its merge leaves its appends pending without a duplicate, and the next ingesting pass marks them. A `/continue` resume of such a pass ingests the appends its checkpoint already carries as pinned copies (`history.carried_appends`) and marks them at its report gate. A resume that runs no research leaves an append that was never merged pending, and says so. Any other resume or re-attach (approve, re-plan, Revise, Send, `/continue` re-attach or resume) leaves the rest pending and says so in the chat: their count, and the path that ingests them.
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
  - A button passes its message id to the driver (the gate message of a run it attaches, when the button belongs to that gate); both Cancel buttons call `cancel_run` with their card's gate. An Edit / Revise reply that is refused after it was consumed says so: "(your reply was not used — tap Edit/Revise and send it again)".
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
  - `_reconcile_orphaned_runs` (approved edit): a run whose checkpoint `next` is not a gate gets status `cancelled` and "♻️ Run `<id>` (<topic>) was interrupted by a bot restart mid-step. /continue <id> resumes it from its last saved step."; the "can't be resumed … /research it again" error is left only for a run with no checkpoint. A quick `/ask` run is checked first (fix round 2): status `cancelled` and "♻️ /ask `<id>` (<topic>) was interrupted by a bot restart — ask again with /ask." The reconcile reads every thread through the deep graph, and a quick thread at its own `deliver` step reads as `("deliver",)` there, so a check in the mid-drive branch alone would leave it at the report-gate branch.
  - The comments on `on_callback` and `_post_init` describe attach on action (approved comment edits, fix round 2).

Approved edits outside WP-E1's files (the `recency` ruling):

- `src/argus/graph/run_config.py`: `recency: Literal["d", "w", "m", "y", "any"] | None`.
- `src/argus/engines/search.py`: `resolve_recency` returns `(None, None)` for `"any"`. That means no recency filter even for a time-sensitive brief, and nothing is dropped as undated.
- `src/argus/graph/run_config.py` (fix round 4): `ensure_in_checkpoint(graph, cfg, *, engines, as_node=None)` passes `as_node` to its `aupdate_state` when given. Its other callers and tests are unchanged.

### Decisions where the spec is silent

- **The busy rule covers `/ask` too**: it spends the same free-tier quota.
- **A gate wins over a drain only in LangGraph.** A cancel during the plan drive reaches the plan gate first, because `interrupt_after=["scout"]` fires before the drain boundary. The driver still treats it as a cancel: `cancel_note` names scout, status is `cancelled`, no plan card appears, and `/continue` re-attaches the plan. The same applies at the report gate.
- **A forced `task.cancel()`** (after the drain bound or at shutdown) sets status `cancelled` and writes `cancel_note = "force-cancelled during node <node> (<reason>); the node was abandoned"`. The node's thread may still be running, and it never wrote: the checkpoint is the one from before it, and `next` names that node. The card says what `/continue` then does. A forced cancel inside a first research pass rests at `("research",)`, which is then the plan gate: `/continue` re-attaches the plan, and Approve reruns research. Inside an Extend, gap or append pass it rests there too, but that is no gate (live-check fixes, below): `/continue` reruns research. Elsewhere `/continue` reruns the node. That node's own Exa spend is not in `runs.exa_usd`. The note write and the `next` read are guarded (fix round 2). A failure is logged with its traceback and said in the chat ("⚠️ Run <id>: could not record the cancel note: <error>"). The run still ends `cancelled`, its card says "/continue <id> resumes it", and the `CancelledError` is re-raised. A quick `/ask`'s card names no `/continue`. The note is written as the drained one is (fix round 3), so a re-plan or fork force-cancelled inside its first node keeps its `next`: `("brief",)` and `("extend_prep",)`, not its old gate. So does a `/continue` resume of either (fix round 4).
- **Untracked runs attach on action** (§2.5 ruling; this replaced "actions need a tracked run"). After a restart, a button on an old message attaches its run when the run rests at a gate, and the action proceeds on that message. `snapshot()["awaiting"]` and the action's outcome agree. A run with no library row, or one not resting at a gate, still answers that the run is not in flight and that `/continue <id>` re-attaches it. A quick `/ask` answers "(run <id> was a quick /ask — nothing to cancel or continue)" instead (fix round 3). `continue_run` registers the run through `_hold` (from its row, without attaching it) and re-sends its plan card or preview; it gets the same single retry behind another actor's forget (fix round 3). `plan_payload` stays a view of tracked runs. The tapped card is adopted only for its own gate. The retry after another actor's forget is any forget, not only a failed attach: the second look-up attaches the run when it still rests at a gate (after a gate cancel too, which leaves the run at its gate), else it answers not in flight.
- **`/continue` on a stopped run** resumes from its checkpoint (`astream(None)`, status `running`). A stopped run is one whose `next` is not a gate: cancelled, or failed. A run that failed or was force-cancelled inside a first research pass is not one of them: its `next` is `("research",)`, the plan gate, so `/continue` re-attaches its plan. One stopped inside a later pass (an Extend, a gap wave, an append pass) is: `("research",)` is no gate for it (live-check fixes, below). Before, `/continue` answered "mid-pipeline — wait"; the registry now tells a live drive from a stopped one. A quick `/ask` run is refused.
- **Extend and Revise are refused at their round caps** (`MAX_EXTEND_ROUNDS`, `MAX_REVISE_ROUNDS`). Past the cap the graph would end the run silently.
- **Older-topology checkpoints**: a checkpoint whose `branch:to:*` channels name nodes this graph lacks is refused on every resume (approve, re-plan, Extend, Revise, `/continue` resume or fork). These are v2 threads. Re-attaching to such a checkpoint and Send still work (pointer 5), and an untracked one attaches on action under the same rules. `/continue` on a sent (`done`) one is refused before anything moves: the Extend fork it would take cannot run on an older graph.
- **The unused-appends notice** reads the library's pending rows while the action resolves; the fork and the report-gate Extend ingest them and say nothing. After Send the line names `/continue <id>` only.
- **The reconcile's `cancelled`** applies to any checkpoint whose `next` is not a gate, including `()` (a run whose Send finished the checkpoint but not the status write); `/continue` then extends it.
- **The profile knob** cycles the engines profiles. Each non-active profile is resolved once per run when the cycle first reaches it, and its snapshot or its problem is cached. A card render never validates live: the active profile's state is the cached snapshot (`current_snapshot_cached()`), another profile's the cached result. Approve validates the chosen profile live: the active one through `active_snapshot()`, another through `resolve_snapshot`, its result cached again. A run approved on another profile carries that profile's snapshot in `rc.engines`, and the global active profile is unchanged. `start_run(engines_profile=…)` validates its profile the same way at the start; it is the run's from the plan drive on and the card's profile choice. The default choices carry the run's own profile from `rc.engines` (fix round 2), so a re-attached run (a tap after a restart, a `/continue`) keeps it, and Approve keeps it in `rc.engines`. Its card shows no problem for that profile until Approve validates it live: a render never validates. A re-attached run whose profile is gone from the engines file is the exception (fix round 3). A static check (`registry.bindings`) caches the registry's own problem text for it, so the card marks it. The profile knob cycles from the first profile instead of failing on `list.index`, and Approve refuses with that text until a profile that exists is chosen.
- **The recency knob** is the `recency` override (`None` = the brief decides). The scout decides recency once, and research reuses `search_spend["recency"]`. So an Approve that changes recency also rewrites `search_spend["recency"]` (`resolve_recency` on the rebuilt config) and keeps the rest of the scout's spend.
- **Depth "auto"**: a run whose depth equals its length's default depth follows the length until a depth is tapped. After that the tapped depth stays through length changes.
- **The driver owns the gate messages**: the plan card, the preview, and their edits on approve, edit, Extend, Revise, Send, Cancel and knob changes. An action from the console edits the Telegram message the same way ("✅ approved from console" and the like). Taps on an older plan card act on the run, and the current card is the one edited.
- **Stage events**: each task's transitions are published once. The driver's `stage start` comes first, so a node's own start (scout, research) is dropped with its summary; a node's own `stage end` comes before the update, so it wins over the driver's.
- **The spend write comes after the gate presentation**, so a failed write is the last chat line. The preview has already been sent and the status is set.
- **A node event's `node` field** falls back to the running stage when a node omits it.
- **Every report-gate action writes both request flags** (fix round 2). A drive drained at its first boundary runs no node, so an Extend or Revise drained at shutdown leaves its flag at `("deliver",)`. `route_after_deliver` would then send a later Send to `extend_prep` or `revise_prep`, and turn a later Revise into an Extend round (Extend wins). So each action states its whole request: Send clears both flags, Extend sets its own and clears Revise's, Revise the reverse.
- **A drive launched during the shutdown drains at once**: the launch requests the drain itself. `drain_all`'s wait would otherwise leave a fast drive time to run on before the drain lands. Its first boundary then stops it before any node runs, so the card says "Cancelled before it started" (fix round 3). Only resumes launch then. The note keeps their `next`: it is written as the route its drive launched with, the action's own or, for a plain `/continue` resume, the one where it rests (fix rounds 3 and 4), and by inference only where neither applies. Round 2 said the note write simply found its `as_node`, but inference re-runs the edges of the last node that ran. It moved a re-plan or a fork back to its old gate (fixed in round 3), and also a plain `/continue` resume of either (fixed in round 4). So `/continue` resumes them where they stopped, or re-attaches the gate they stopped at. A start is refused at its launch: on a fresh thread the note write raises "Ambiguous update, specify as_node", and the thread keeps no input to resume from.
- **Resting-run writes keep `next`** (fix round 4). A state write without `as_node` runs as the last node that ran: LangGraph infers it from `versions_seen`, which an `as_node` write never updates, and recomputes `next` from that node's edges. So every driver write on a resting run passes an `as_node` that reproduces the checkpoint's `next`. The order is: the last node this drive finished, then the launching action's own route, then the route of the resting state from the table below (`runs._ROUTES`). Inference is left for every other resting state: only a real node leads there (a gate, or the next node of a stopped drive), so the last node that ran is the right one. The table holds each resting state that the driver's `as_node` writes (`runs.py`; `history.py` makes none) can leave, with the one node that leads there. Since each is the only node with an edge to its state, the same route also fits that state when a real node produced it.

  | Resting `next` | Left by | Written as |
  |---|---|---|
  | `("brief",)` | Edit: `edit_plan` writes the feedback `as_node="intake"` | `intake` |
  | `("extend_prep",)` | the `/continue` fork, `as_node="deliver"` with `extend_requested`; an older bot's sent run advanced with an Extend flag set | `deliver` |
  | `("revise_prep",)` | an older bot's sent run advanced with a Revise flag set | `deliver` |
  | `()` | Send, `as_node="deliver"` with both flags cleared; an older bot's sent run advanced with no flag set | `deliver` |

  Two writes take the table's route: the legacy injection on every resume path (`ensure_in_checkpoint(…, as_node=…)`, through `_resumable`), and the cancel note of a plain `/continue` resume, which launches with the route of the state it rests at. Approve, Extend and Revise write at their gates by inference. Before, a re-plan drained or failed in brief, or a fork drained, rested at `("brief",)` / `("extend_prep",)`. Their resume, stopped again before that node finished, wrote its note by inference. The re-plan went back to its plan gate with the edit unread, and the fork back to its report gate with `append_only` left for the next Extend. A pre-v4 checkpoint resting at `("brief",)` got its injection by inference too, so its `/continue` ran research without an Approve.

### Behaviour changes for handoff.md

- `/research` and `/ask` start with a live progress card (stage, counts, last event), not a series of progress messages. The card's last edit says how the drive ended. `/ask`'s answer arrives as a chat message below its card.
- While one run executes, a second `/research`, `/ask`, Approve, Edit, Extend, Revise or `/continue` resume is refused as "⏳ busy — run `<id>` (`<topic>`) is at `<stage>`; try again when it reaches a gate". A second tap on a running run answers "already running".
- `/cancel` (and a Cancel button) stops a running drive at the end of its current step, with the checkpoint saved. `/continue <id>` resumes it. A run resting at a gate is cancelled at once. Before, `/cancel` only forgot the run while its graph kept going.
- `/status` shows each run's stage, elapsed time and last event, and the console URL and its health.
- The plan card has a depth row, a `⚙ <profile>` button and a `🕒 <recency>` button. The header names the depth, and two lines name the archetype, the recency window, the engines profile (with its problem) and the Exa estimate. The estimate shows only when Exa is enabled (the key, and `exa` in the run's providers). Recency `any` switches the filter off.
- The report preview opens with the quality line, the "⚠ Not fact-checked" / "⚠ Degraded run" banners and the degradation notes.
- Send, Extend, Revise, Approve and Cancel edit their gate message (the keyboard goes). A failed resume (engines unresolvable, a broken checkpoint, a cap) is refused in the chat, and the run stays at its gate. Before, it reached only the log.
- A failed drive, including a failed re-plan, reports "⚠️ Research failed: <error>", sets status `error` and leaves the run resumable with `/continue`. A re-plan failure used to say "⚠️ Re-plan failed" and left the run between nodes.
- Shutdown (`post_stop`) drains live drives for at most `ARGUS_SHUTDOWN_DRAIN_S`, then cancels them. A drained run is `cancelled` and resumable. From the start of the shutdown, a new action that would drive is refused with "⏹ Argus is shutting down — try again after the restart", before it changes anything. One accepted just before still starts its drive, which is drained at once; a `/research` or `/ask` accepted just before is refused instead. A failing stop hook no longer keeps the library and saver open: it is raised after they close.
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

Fix round 2 (Task 10 re-review):

- An Approve, Extend or Revise accepted just before a shutdown no longer leaves its writes without a drive: its drive starts and is drained, so the run ends `cancelled` and resumable. After the restart, Send ends the run, and Extend and Revise do what they say, whatever flag a drained drive left.
- A forced cancel whose `cancel_note` cannot be written still ends the run `cancelled` and stops its card, and the chat says the note could not be recorded.
- A cancelled `/ask` is never told to `/continue`. After a restart, an interrupted `/ask` is `cancelled` with "♻️ /ask `<id>` (<topic>) was interrupted by a bot restart — ask again with /ask."
- Two actions at once on a run attached after a restart, whose first attach fails, both get the refusal. The tapped card keeps its content.
- An old card of the other gate (say, a plan card while the run waits at its report preview) gets "Run `<id>` is not at its plan gate." for every button, Cancel included. It is never edited later.
- A run started on another engines profile keeps it after a restart (a tap or `/continue`) and on Approve.

Fix round 3 (Task 10 re-review 2):

- A cancelled or shutdown-drained re-plan, `/continue` fork or other drive stays where it stopped. Before, a re-plan or fork drained before its first node finished fell back to its old gate: the edit was lost, or a sent run reopened with `append_only` left behind. `/continue` then redrafts the plan with the edit, or runs the fork's own append-only pass.
- A drained run's card says what `/continue` does from where it stopped: it resumes from the next node, or it re-attaches the plan or report card, where the action is repeated. A drive drained before any node ran says "Cancelled before it started".
- A run whose engines profile was removed from the engines file shows that problem on its card after a restart. Its profile button moves to the first profile, and Approve refuses until a profile that exists is chosen.
- `/cancel <id>` on a finished `/ask` says it was a quick `/ask`, with nothing to cancel or continue.
- A `/continue` that waited behind a failed attach gets its own answer, not "not in flight".
- Appended sources are never merged twice: an Extend or fork that was drained or refused leaves them pending, and the next pass that ingests them marks them.

Fix round 4:

- A re-plan or a `/continue` fork that stopped before its first node (cancelled, shut down, or a re-plan that failed in brief) now keeps its place through every later stop. Before, its `/continue` resume, stopped again before that node finished, sent the re-plan back to its plan card with the edit unread, and the fork back to its report card with its append-only pass left for the next Extend. `/continue` then redrafts the plan with the edit, or runs the fork's own append pass.
- A pre-v4 run left mid-re-plan (resting at `brief`) gets its legacy RunConfig without moving: `/continue` redrafts its plan and stops at the plan card. Before, the injection moved it past the plan gate, and `/continue` ran research without an Approve.
- An `/append` of a URL that the run's scout found but research never read is now ingested. It used to be marked failed ("not fetched") on every retry. A `/continue` resume that runs no research leaves such an append pending and names it, where it used to mark it failed.

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

Fix round 2:

- `tests/test_runs_driver.py`:
  - the fixture also restores `runs._launch_closed`;
  - `test_a_drive_prepared_across_the_shutdown_start_never_launches` is replaced by the C4 rule's tests: an Extend and an Approve prepared across the shutdown start launch and are drained (then Send ends the Extend's run after a restart); a start prepared across it is refused before its launch, its row `cancelled`; and an action claimed after the start is refused with its status, checkpoint and card unchanged;
  - new tests: a failing forced-cancel note write; a drained and a force-cancelled `/ask` with no `/continue` line; a tap that waited on a failed attach; an old plan card tapped at the report gate (Approve and Cancel), then a console Cancel; a re-attached run's engines profile through a tap and through `/continue`.
- `tests/test_orphan_reconcile.py`: `_seed` takes the run's mode; a quick orphan mid-drive and at its `deliver` step.

Fix round 3:

- `tests/test_runs_driver.py`:
  - the Approve-at-shutdown test's card line is M-1's "Cancelled before it started — … re-attaches the plan card — repeat the action there.";
  - new tests:
    - the cancel note keeps `next` for each launch kind (Approve, re-plan, Extend, Revise, a `/continue` resume, the `/continue` fork), drained at its launch and force-cancelled inside its first node (a `_Hold` on that node);
    - a drained re-plan's `/continue` redrafts with the edit;
    - a drained fork resumes its own append pass and marks the appends;
    - the drained card's two hint wordings;
    - a re-attached run whose profile left the engines file;
    - `/cancel <id>` on a finished `/ask`;
    - `/continue` behind a failed attach;
    - an Extend with appends drained, then one that completes.

Fix round 4:

- `tests/test_runs_driver.py`, new tests:
  - a resume from a route-written rest keeps `next`: a re-plan drained or failed in brief, or a fork drained; its `/continue` resume drained at launch or force-cancelled inside its first node (a `_Hold`); every cancel note keeps `next`, the card's hint names that node, and the next `/continue` redrafts the plan with the edit or runs the fork's own append pass;
  - an `/append` of a URL the scout found but research never fetched: its pinned copy is merged, research's own append pass (`_Pass.append_pass`, its fetch stubbed) picks it, and it is marked ingested;
  - a resume that runs no research leaves a never-merged append pending and names it.
- `tests/test_run_config_resume.py`: a pre-v4 checkpoint resting at `("brief",)` (a v3 re-plan's intake write) gets the injection with `next` unchanged; `/continue` redrafts with the edit and stops at the plan gate with nothing fetched.

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

Fix round 2: scripts and logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t10-fix2\`, with the same environment and the same rules. The re-reviewer's probes are in `…\scratchpad\t10-rereview1\`.

- RED (`red.log`): the new tests, written first, fail against the unchanged code, except the after-the-start refusal, which fills a gap. The gate-mismatch test was tightened to one chat line per tap and re-run against the unchanged code (the same log).
- GREEN: the focused files pass (`green.log`). The full suite (`suite-final.log`) passes except the same live-proxy tests, and no warnings summary appears.
- `probe_fix2.py` (hermetic): an Extend drained at its launch rests at `("deliver",)` with `extend_requested` set, and the empty END write then goes to `extend_prep` while the flag-clearing one ends at `()`. An Approve drained at its launch rests at `("research",)`. A quick thread at its `deliver` step reads as `("deliver",)` through the deep graph. A fresh thread drained at its first tick rests at `("__start__",)` with no `run_config` or `length` in its values: the note write raises "Ambiguous update, specify as_node", writing it `as_node="__start__"` drops the input, and `/continue` refuses it (`probe_fix2_start.log`, `probe_fix2_fresh.log`). That is why a start never launches into the shutdown.
- Not checked live, for the same reason.

Fix round 3: scripts and logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t10-fix3\`, with the same environment and the same rules. The re-reviewer's probes are in `…\scratchpad\t10-rereview2\`.

- RED (`red.log`): the new tests, written first, fail against the unchanged code: the cancel note for a re-plan and a fork (drained and force-cancelled), and each of the other items. The cancel-note cases for Approve, Extend, Revise and a plain resume already held there (LangGraph's inference fits them); they fill a gap. The `/continue` retry test was made deterministic (an event, not a sleep) and re-run against the unchanged code (the same log).
- GREEN: the focused files pass (`green1.log`). The full suite (`suite-final.log`) passes except the same live-proxy tests, and no warnings summary appears.
- Not checked live, for the same reason.

Fix round 4: scripts and logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t10-fix4\`, with the same environment and the same rules. The re-reviewer's probes are in `…\scratchpad\t10-rereview3\`. Copies of them were re-run there on the fixed code (`probe_*.after.log`).

- RED (`red.log`): the new tests, written first, fail against the unchanged code. In every second-stop case the card names the old gate, because the note moved `next` back. The scout-found append gets no pinned copy, the never-merged append is not left pending, and the legacy injection moves `("brief",)` to `("research",)`.
- GREEN: the new tests pass (`green-new.log`), and so do the driver-backed test files (`green-focused.log`). The full suite (`suite-final.log`) passes except the same live-proxy tests, and no warnings summary appears.
- The probes, re-run on the fixed code: every cancel note keeps `next`. The scout-found append is merged pinned, picked and ingested, and the never-merged append stays pending with its notice. The pre-v4 checkpoint stops at its plan gate with no research run.
- Not checked live, for the same reason.

### Live-check fixes (Task 12 live check, 2026-10-21)

The orchestrator's console live check of the real bot (run `34098f65`) found two driver defects. Both are fixed in one commit with the WP-B fetch bound (wp-b.md).

- **The plan card always fits and parses** (`bot/research.py` `_format_plan`, `bot/common.py`). The live card (7 long sub-questions, 14 intents, 40 sources) failed twice. Its Markdown did not parse ("can't find end of the entity", byte offset 2821), because an escaped `_` inside an italic entity ends the entity: Telegram's legacy parser honours no escape inside one. The plain retry was then "Message is too long", and the run sat at its plan gate with no buttons.
  - The card is measured in UTF-16 code units, as Telegram counts them, on the source text, which is never shorter than the parsed text. It must fit `TG_LIMIT` (4096) less room for the note a gate edit appends (`_NOTE_ROOM`).
  - A card that is too long is shortened step by step until it fits (`_PLAN_STEPS`). First the sub-question texts are capped, then the intent texts and the summary; this happens twice, the second time shorter. Then the lists are cut from their end: sources first, then intents, sub-questions last. Each cut list ends "…and N more". The header, the knobs line and the buttons always survive, and a header fact (a profile problem, a rebuild error) is capped at `_HEAD_CHARS`.
  - Model and scraped text sits outside entities, escaped (`_md_escape`). The summary and the intent queries are no longer italic. Text inside an entity loses that entity's end character (`_md_inside`: the intent kind in code, a source title in its link), and a source URL's parentheses are percent-encoded, because a URL ends at its first `)`.
  - `_md_escape` escapes only `_ * ` [`, the characters the legacy parser unescapes. An escaped `\` or `]` used to show its backslash.
  - `_safe_send` never fails silently. Its plain retry is cut to the limit (`_tg_cut`, marked "…(cut)"). If that also fails, it sends `fallback`, or "⚠️ This message could not be shown (<error>).", with the same `reply_markup`. `_present_plan` passes a short card naming the run, so the buttons reach the chat. The gate message keeps the full card's text, which a knob tap re-renders.
  - Other model-text senders: the report preview's HTML already parses, and its excerpt shrank to fit. A head too long on its own (long degradation notes) is now cut before it is escaped, both bodies fit the same budget, and the last-resort "report preview unavailable" message carries the report keyboard. The `/ask` answer (`_say`, Markdown) is cut to the limit on its plain retry rather than lost.
- **A stopped Extend at `research` is not the plan gate** (`bot/runs.py` `_gate`, `graph/research.py` `pass_mode`). After its force-cancel, the Extend rested at `("research",)`. `_GATES` read that as the plan gate, so the console showed "Plan waiting for you" with Approve and Edit (Task 10's parked Minor). `("research",)` is now the plan gate only while the run's research pass would be the first: `research.pass_mode(values) == "first"`, which is the node's own test (no `research_rounds`, no `extend_rounds`, no `coverage_gaps`, not `append_only`). `research_node` uses the same function.
  - Every gate read goes through `_gate(next, values)`: `_attach` (attach on action), `_at_rest`, the drained and forced-cancel card hints, `continue_run`, `snapshot()["awaiting"]` and `plan_payload`.
  - An Extend, gap wave or append pass resting at `("research",)`, drained after `extend_prep` or force-cancelled inside research, is a stopped run. Its card says "/continue <id> resumes it from research" (drained) or "… reruns research" (forced). `/continue` resumes that pass, and `extend_prep` does not run again. The console shows no plan, an old plan card's button answers "not in flight", and the library row stays `cancelled`.
  - The writes keep `next`. A resume of such a rest launches with no `_ROUTES` entry, so its writes take LangGraph's inference: the last node that ran, which is `extend_prep` or `panel`, each with its own edge to research.
  - A first pass stopped inside research is still the plan gate, as Task 10 decided: its research never wrote, so `pass_mode` is `first`.

Not changed (outside this fix's files): `bot/app.py`'s restart reconcile still marks any run resting at `("research",)` `awaiting_plan`, including an Extend interrupted by a restart mid-research. The driver's `_gate` then answers "not at a gate" for it, so the row and the snapshot disagree. The fix is one line: use `runs._gate(nxt, values)` there. The Markdown messages in `bot/media.py` (`/find`, `/yt`) put `_md_escape`d text inside italic entities, so the same parse failure applies to them. They still go out, through `_safe_send`'s plain retry, but without formatting.

Tests touched:

- `tests/test_bot.py`:
  - `_tg_legacy_markdown`, a strict port of Telegram's legacy parser (tdlib `parse_markdown`), with its own check on the live failure;
  - the long, entity-laden card fits and parses, keeps its header, knobs and buttons, and survives a gate note;
  - a short card is not cut;
  - `_safe_send`'s fallback keeps the keyboard;
  - the preview fits with long notes, and its last resort carries the report keyboard.
- `tests/test_runs_driver.py`:
  - a drained and a force-cancelled Extend at `("research",)` are no plan gate in `snapshot()`/`plan_payload()` and are never attached by Approve;
  - their card hints;
  - `/continue` resumes their research pass to the report gate;
  - a fresh run at its first plan gate is one, tracked and untracked;
  - a refused plan card becomes the short card with its buttons.

Verification: `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\live12-fix\`. `red.log` holds the new tests against the unchanged code. `green-focused.log` holds the focused files. `suite-final.log` is the full suite.
