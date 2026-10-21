# WP-E2 — Mini App console (Task 11: backend)

Spec: `docs/argus-v4-design.md` §2.7 (Mini App API and auth) and the backend half of §3 WP-E2, with Review Focus 4 (hostile markdown in the report HTML). Built on the run driver of WP-E1 (`bot/runs.py`, see `wp-e1.md`) and Task 11's pointers. The frontend under `webapp/static/` is Task 12's.

### What changed

- `src/argus/webapp/` (new package):
  - `auth.py`:
    - `validate_init_data(init_data, bot_token, allowed_user_id, max_age_s)` returns an `AuthUser(id, first_name)` or raises `AuthError(code)`. The codes are `missing_field`, `hash_mismatch`, `expired` and `wrong_user` (403; the others are 401).
    - The check follows §2.7 exactly: `parse_qsl` keeps blank values; it requires `hash` and `auth_date`; the secret is `HMAC("WebAppData", token)`; the data-check string is the sorted `k=v` lines without `hash`; the hashes are compared with `compare_digest` on bytes, so a non-ASCII hash is a mismatch, not a crash.
    - An `AuthError` carries the names of the fields present, never their values.
    - `sign_init_data(fields, bot_token)` is for the dev URL script only. No route calls it.
  - `render.py`: `render_report(md)` runs python-`markdown` (`tables`, `fenced_code`, `sane_lists`), then a stdlib `HTMLParser` allowlist sanitiser:
    - only the tags in `ALLOWED_TAGS` survive;
    - only `a[href,title]` and `ol[start]` keep attributes;
    - `href` must start with `http(s):`, `mailto:` or `#` once control characters and whitespace are removed, so `javascript:`, `vbscript:` and `data:` links lose their href and keep their text;
    - `<script>` and `<style>` are dropped together with their content; every other tag that is not allowed is dropped and its text kept;
    - all text is re-escaped, and tags left open are closed, so the output stays well-formed;
    - links get `rel="noopener noreferrer"`.

    Each `[n]` in a text node outside `code`/`pre` becomes `<span class="cite" data-n="n">[n]</span>` (reusing `graph.assemble.CITE_RE`). Attribute values are never touched.
  - `api.py`: the §2.7 routes, a thin layer over the driver.
    - Every action calls `runs.start_run` / `approve_plan` / `edit_plan` / `report_action` / `cancel_run` with `actor="webapp"`. `recency` goes into `overrides["recency"]`. The run's `chat_id` is the allowed user's.
    - Outcomes become HTTP: `busy`, `already_running` and `refused` → 409; `not_found` → 404; each with `{"error": {"code", "message"}}`, where the message is the driver's text. Success is 200 `{outcome, message, run_id}`, a superset of `{run_id}` that keeps the driver's notices (unused appends, for example). An unknown outcome code is a 500, never a success.
    - `GET /api/runs` reads the library rows plus `runs.tracked()`, with no `aget_state` per row. `in_flight` means a live handle; `awaiting` is the library's gate status while none runs.
    - `GET /api/runs/{id}` takes `runs.snapshot()` for `in_flight` / `awaiting` / `next` / `last_seq`, and its values for the counts, coverage, run config, panel verdict, report paths, `compose_runs` and `gap_waves`. The engines summary is `registry.summarize_calls`, and `plan` is `runs.plan_payload()`, the Telegram card's data.
    - `GET …/events?after=` reads `runs.bus.since` / `last_seq`.
    - `GET …/report?format=html` and `GET …/evidence` serve only the run's registered `report_paths["md"]` / `["evidence"]`, resolved (symlinks and `..` included) and contained in `reports_root`. A path outside is a 404 and is logged.
    - `POST …/send {what: [md|pdf|evidence]}` checks every requested file the same way before it sends any, then the bot sends them to the run's chat. Nothing about the run changes, and no driver call is made. A Telegram failure is a 502 naming how many files were sent.
    - `GET/PUT /api/engines`:
      - each provider's `api_key_env` becomes `api_key: {env, present}` (`{null, null}` for a keyless provider); the active snapshot comes with its warnings, `validation {ok, errors}`, and the month-to-date Exa spend;
      - PUT accepts only `active_profile`, `overrides: {profile: {tiers?, roles?}}` and `transcriber`. Any other key, at the top or inside an override, is refused with a 400 before anything changes ("not editable here … provider definitions and pricing are file-only");
      - the changes go through `registry.set_overrides` / `set_transcriber` / `set_active_profile`, which validate before they persist. An `EngineConfigError` is a 400 with its text (code `engine_config`), and a pydantic error is a 400 `bad_request`.
    - `GET /api/engines/providers/{name}/models` (`registry.live_models` in a thread) and `GET /api/config/presets` (`PRESETS`, each length's default depth, the Exa prices or null).
    - Middlewares:
      - `errors` turns every failure under `/api` into the error JSON: aiohttp's own 404/405, and a 500 that is logged with its traceback;
      - `auth` requires `Authorization: tma <initData>` on every `/api*` path. A refusal is logged as `webapp: <method> '<path>' refused: <code> (fields present: <names>)`: the path with `%r`, and only names matching `[A-Za-z0-9_]{1,32}`, at most 10, then "+N other" (fix round 1). The source IP is never trusted.
  - `server.py`:
    - `build_app(deps, static_dir=…)` serves `GET /` (`static/index.html`, a 404 until Task 12 adds it), `/static/*` (registered only when the directory exists; `show_index=False`, `follow_symlinks=False`) and the API. It sets the CSP, `X-Content-Type-Options: nosniff` and `Referrer-Policy: no-referrer` on every response (`on_response_prepare`). `CSP` is §2.7's string verbatim: `script-src 'self' https://telegram.org`, with no `'unsafe-inline'` anywhere.
    - `start(app)` (POST_INIT, after `runs.init`) and `stop(app)` (POST_SHUTDOWN, idempotent) are described under the behaviour changes below. The access log is off (`access_log=None`).
  - `tunnel.py`:
    - `find_url` takes `https://[a-z0-9-]+\.trycloudflare\.com` from a stderr line, except `https://api.trycloudflare.com`, which cloudflared quotes in its own failure lines ("failed to request quick Tunnel: Post …"). This exception is not in the spec's regex: without it, a failed request would be taken for the tunnel URL.
    - `binary()` is `ARGUS_CLOUDFLARED`, else `tools/bin/cloudflared.exe` (`nt`) or `tools/bin/cloudflared`. `command(binary, port)` is §2.7's argv, and `spawn` uses `start_new_session=True` on POSIX.
    - `Supervisor`:
      - the stderr pipe is tailed: every line is appended to `logs/cloudflared.log`, the first URL is taken within `URL_TIMEOUT_S` (60 s), and the pipe keeps draining;
      - when a process exits, prints no URL in time (it is then terminated) or cannot start, `on_down("<why>; restarting in <s> s")` is called, then the backoff: from 1 s, doubling, capped at 5 min, and reset after a process ran `STABLE_S` (60 s);
      - `stop()` cancels the loop and terminates the process (kill after 5 s); it is idempotent.
- `scripts/webapp_dev_url.py [--port]`: signs `user` (the allowed user), `auth_date` and `hash` with the token from .env, and prints `http://127.0.0.1:<port>/#tgWebAppData=…&tgWebAppVersion=9.0&tgWebAppPlatform=weba&tgWebAppThemeParams=…`. It never prints the token. It exits 1 when nothing answers on that port (an `httpx` GET with `trust_env=False`; any HTTP answer counts as up).
- `src/argus/bot/app.py`: `POST_INIT_HOOKS.append(webapp_server.start)` and `POST_SHUTDOWN_HOOKS.append(webapp_server.stop)`.
- `.gitignore`: `logs/`.

### Decisions where the spec is silent

- **The console's health while the tunnel starts**: `webapp_ok` is False with `webapp_error` "tunnel starting — waiting for the cloudflared URL" (no DM). `/status` then says "Console: down — tunnel starting …" until the URL arrives.
- **`ARGUS_WEBAPP_TUNNEL=none`** sets `webapp_public_url` to `http://127.0.0.1:<port>`, so `/status` shows where the console listens. No menu button is registered: Telegram needs HTTPS.
- **`ARGUS_WEBAPP_ENABLED=false`** starts nothing and sets no `bot_data` key, so `/status` says "Console: not running".
- **Failure DMs are de-duplicated**: one per run of failures (a crash-looping cloudflared does not DM every backoff), and reset by the next good URL. The first good URL after a "⚠️ Console down" DM sends "🌐 Console back up: <url>" (fix round 1; it replaced "Console tunnel restarted", and it also covers a failure before the first URL).
- **A menu-button failure** keeps the URL (the console is reachable) and sets `webapp_ok` False with "menu button not registered: …", plus a DM.
- **A missing cloudflared binary** (checked with `is_file()` before the supervisor starts) is a failure ("cloudflared not found at <path> (set ARGUS_CLOUDFLARED)"), not a retry loop. The local server keeps serving.
- **An unknown `ARGUS_WEBAPP_TUNNEL`** starts nothing and is reported.
- **`snapshot.plan` is `runs.plan_payload()`**, for a tracked run and (fix round 1) for an untracked one resting at its plan gate after a restart; see the fix-round section.
- **`PUT /api/engines` makes one change per request** (fix round 1): one part (overrides, the transcriber or the active profile), and overrides for one profile. Anything more is a 400 before anything changes, so a request is applied whole or not at all.
- **`GET /api/me`'s `active_profile`** is null when no engines config is loaded; `/api/engines` then says why (400 `engine_config`).
- **The webapp imports nothing from `argus.bot` at module level** (`_send_md_doc`, `_safe_send` and `runs` are imported inside the functions), because `bot.app` imports `webapp.server` at module level.

### Behaviour changes for handoff.md

- The bot serves the Mini App console on `127.0.0.1:${ARGUS_WEBAPP_PORT:-8787}` and logs `webapp: listening on 127.0.0.1:<port>`. With the default quick tunnel it starts `tools/bin/cloudflared(.exe)` and logs `tunnel: https://<x>.trycloudflare.com`. It sets the chat's `Argus` menu button to that URL and logs `menu button registered for chat <id>`. cloudflared's stderr goes to `logs/cloudflared.log`.
- The console is optional. If the server cannot start (port taken), cloudflared is missing, the tunnel dies or the menu button fails, the problem is logged, DM'd ("⚠️ Console down — …") and shown in `/status`; media and research in chat are unaffected. A killed cloudflared is restarted with backoff, and the new URL is re-registered and DM'd ("🌐 Console back up: <url>").
- `/status` prints the console URL and its health (WP-E1 reads the keys this WP sets).
- At shutdown the console keeps serving while the drives drain (`post_stop`), then closes in `post_shutdown`. A console action in that window gets the driver's shutdown refusal (409 `refused`).

### New env vars

Read by this WP (names from spec §7; `ARGUS_WEBAPP_ENABLED` and `ARGUS_WEBAPP_PORT` were already in `Settings`):

- `ARGUS_WEBAPP_TUNNEL`: `quick` (default) or `none`.
- `ARGUS_CLOUDFLARED`: the binary's path.
- `ARGUS_WEBAPP_PUBLIC_URL`: used instead of a tunnel when set.
- `ARGUS_WEBAPP_INITDATA_MAX_AGE_S`: default 3600.

### Test files touched

- New: `tests/webapp_fakes.py`. It holds:
  - `FakeRuns`, the driver's call surface, which records calls and answers with configurable `Outcome`s over a real `RunEventBus`;
  - `FakeLibrary`, `FakeBot` and `FakeProcess` (a cloudflared stand-in on a real `StreamReader`);
  - `seed_run`, `init_data` / `auth` / `dev_fragment` and cloudflared stderr fixtures;
  - `make_app(fake, reports_root=…)`, the real console app on the fakes, for Task 12's local previews.
- New: `tests/test_webapp_auth.py`. It covers:
  - valid, blank values kept, tampered, another token, a non-ASCII hash, expired (and a longer max age), wrong user, and each missing field;
  - the fields on the error;
  - through the server, every refusal's status and code, with logs that carry the field names and none of the values, initData or token;
  - the dev URL script: the signed fragment validates, the token is never printed, and a server that is down exits non-zero.
- New: `tests/test_webapp_api.py`. It covers:
  - `/api/me`;
  - the runs list's formula and its limits;
  - the run view's shape and the plan payload;
  - 404s;
  - events;
  - start (all the knobs, and a topic alone), bad bodies and bad JSON;
  - every outcome code as HTTP on every action route;
  - the plan, report and cancel calls as `webapp`, bad actions, and an unknown outcome as a 500;
  - report HTML, evidence, containment (outside the root, a `..` traversal, unregistered and missing files);
  - Send to chat (files only, all checked first, bad bodies);
  - the engines routes on the real registry: keys shown as `{env, present}` and never their values, a failing active profile, provider-definition changes refused, per-profile overrides persisted and reloaded, a profile and transcriber switch, `EngineConfigError` text, models, presets;
  - auth on every API route;
  - the index and static files.
- New: `tests/test_webapp_render.py`. It covers `test_hostile_markdown_is_sanitised` (Review Focus 4), the allowlist over every output tag and attribute, citations in text nodes only (not in code, a fenced block or an href), well-formed output, `test_csp_has_no_inline_script`, and the security headers on every response.
- New: `tests/test_tunnel.py`. It covers:
  - URL parsing from fixture stderr, cloudflared's API host not taken for the URL, the binary and the command;
  - the supervisor on fake processes: the URL, the backoff sequence and its cap, the reset after a stable run, the URL timeout, a spawn failure, an idempotent stop, and the stderr log;
  - `server.start` / `stop` on a fake bot: local only, a port already taken (logged, DM'd, `webapp_ok` False, never raised), a bad tunnel mode, disabled, a public URL with its menu button, cloudflared missing while local still serves, a quick tunnel with its URL, menu button, kill, restart and DMs, and a menu-button failure (Telegram or other);
  - the webapp importing before the bot in a fresh interpreter;
  - the hook registration and the `logs/` ignore.

### Verification evidence (2026-10-21)

Logs and the probe are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t11\`. Every run used `FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1`. No test started the bot or cloudflared, and none called a live service.

- RED (`red.log`): with only the tests written, all four files failed at collection (`ModuleNotFoundError: No module named 'argus.webapp'`). A second RED was found while implementing: `import argus.webapp.server` before `argus.bot` raised a circular-import `AttributeError`. The suite's import order hid it; `test_the_webapp_imports_before_the_bot` now guards it.
- GREEN: the four focused files pass. The full suite (`suite-final.log`, re-run after the last edits; `suite.log` is the first run) passes except the live-proxy tests in `tests/test_llm.py` and the `tests/test_graph.py` fixture, which fail on the closed port by design. No warnings summary appears.
- `probe_real_driver.py` / `.log` (hermetic) runs the console on the real `bot/runs.py`, a real `Library` and the real deep graph on a `MemorySaver`, with the tunnel off. It showed 401 without auth, the library-based list, the snapshot view, the driver's `not_found` as 404 for an untracked run with no checkpoint, the CSP on `/`, and an idempotent stop.
- Not checked live: the real cloudflared, the menu button inside Telegram and the frontend are the orchestrator's checks after Task 12.

### Fix round 1 (Task 11 review)

What changed:

- `webapp/tunnel.py`: the supervisor never dies silently.
  - `_tail` reads stderr to EOF whatever happens to the log. A log that cannot be opened or written is logged once (per supervisor) and the reading goes on. A line over the `StreamReader` limit is dropped (logged once), and the reading goes on.
  - `_run`: any exception in an iteration other than `CancelledError` (an `on_url` that raises, say) is logged with its traceback, reported as `on_down("supervisor error: <e>; restarting in <s> s")`, and followed by the normal backoff. The iteration's process is terminated first, so no cloudflared outlives its supervision. An `on_down` that raises is logged.
  - `stop()` waits for the cancelled loop without re-raising; a loop that had ended with an error is logged.
- `webapp/api.py`:
  - No log injection from unauthenticated requests: the path is logged with `%r` (auth refusals, a 500, the containment and send warnings), and the initData field names pass `_field_names` (only `[A-Za-z0-9_]{1,32}`, at most 10, then "+N other").
  - `PUT /api/engines` is atomic per request (above).
  - The path checks (`resolve()`, the containment, `is_file()`; `_checked_path`) and `render.render_report` run through `asyncio.to_thread`.
- `webapp/render.py`: an unclosed `<script>`/`<style>` drops everything after it, as before, and the output now ends with the visible marker `<p>[content removed by the sanitiser]</p>`. The dropped text is never emitted.
- `webapp/server.py`: the "back up" DM (above). The URL counter is gone: the DM follows from a "Console down" DM having gone out.
- Approved edit outside WP-E2's files (controller ruling): `src/argus/bot/runs.py` `plan_payload(run_id)` also answers for a run that is not tracked. It reads the library row and the checkpoint through `RunHandle.of(row, row["chat_id"])`, which is not registered. When `next == ("research",)` and the state has a `run_config`, it returns `_payload` with the choices `_attached_choices` gives an attach (the active profile's name from `registry.current_config()`, so nothing validates live). It writes nothing and registers nothing. A quick run, a legacy checkpoint (no `run_config`) or a run not at its plan gate gives None, as before.

Tests:

- `tests/test_tunnel.py`: an unopenable log (the pipe still drains, `on_down` runs, the supervisor restarts, one warning); an `on_url` that raises (reported, its process terminated, restarted; `stop()` does not re-raise); an over-long stderr line; `stop()` after the loop died; the "back up" DM after a failure before the first URL.
- `tests/test_webapp_auth.py`: a `%0A` field name and a `%0A` path each give exactly one log line, with no forged line; the field-name cap.
- `tests/test_webapp_api.py`:
  - one change per request (two parts, overrides for two profiles, nothing). `test_put_engines_switches_profile_and_transcriber` now makes its two switches as two PUTs, as item 4 requires;
  - the fakes bound to the driver (a wrong kwarg, a keyword-only arg passed positionally);
  - render and path checks in threads;
  - on the real driver (the `env` fixture of `test_runs_driver.py`), an untracked plan-gate run's `GET /api/runs/{id}` returns the tracked card's payload with no checkpoint write, no registry entry and no gate edit, then a console Approve attaches and runs it to its report gate; `plan_payload` is None for an untracked run off its plan gate and for an unknown run.
- `tests/test_webapp_render.py`: an unclosed `<style>` and `<script>` end with the marker and emit none of the dropped text; no marker when everything closed.
- `tests/webapp_fakes.py`: `FakeRuns` binds every call with `inspect.signature(getattr(runs, name)).bind(...)`.

Evidence (`A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t11-fix1\`, with the same environment and the same rules):

- RED (`red.log`): the new tests fail against 9182fd9. Two held already and fill a gap: the off-gate `plan_payload` test and the no-marker render test.
- GREEN: the focused files pass (`green.log`). The full suite (`suite-final.log`) passes except the same live-proxy tests, and no warnings summary appears.
- The reviewer's probes (`…\scratchpad\t11-review\`), re-run on the fixed code (`probe_*.after.log`). With an unopenable log the supervisor stays alive and reports the URL, a raising `on_url` is logged and the supervision goes on, and `stop()` returns cleanly. Every refusal is one log line with a `%r` path, and the injected field name is counted as "+1 other". The unclosed `<style>` renders as the marker.

## Task 12: frontend

Spec: the frontend half of §3 WP-E2, under §2.7's CSP. Built on the backend above and served from `webapp/static/` by `server.build_app` (`GET /` is `static/index.html`; everything else is under `/static/*`).

### What changed

- `scripts/sync_arsenal.py [--arsenal DIR] [--pair ledger] [--out DIR]` copies the arsenal kit into `src/argus/webapp/static/vendor/arsenal/`, mirroring the kit's relative tree:
  - `FIXED_FILES`, listed in the script: `tokens/{tokens,base,fonts}.css`; `kits/web-ui/ui.css` and each `components/*.css` it imports; `ui.classic.js`; `icons/icons.js`; `motion/motion.css`; `templates/app-shell.html` and `templates/console.html`;
  - the chosen pairing's `type/pair-<pair>.css`, the `type/faces/*.css` it `@import`s, the `vendor/fonts/web/**` font files those faces and `tokens/fonts.css` reference, and each font folder's `LICENSE.txt` (the fonts are SIL OFL);
  - every relative `@import`/`url()` of a copied stylesheet must resolve to a copied file, or it raises before writing anything. A listed file that is missing raises too;
  - the old snapshot is removed first. Then `MANIFEST.json` `{source_root, copied_at, pair, files: [{path, sha256}]}` and a `.gitattributes` (`* -text`) are written. The `.gitattributes` keeps git's `core.autocrlf` from rewriting line endings, so the sums hold on every checkout (the kit's files are CRLF on this PC, and a Linux CI checkout would otherwise change them);
  - without `--arsenal`, the source is the existing manifest's `source_root`, so a re-sync needs no path. With neither, it exits with a message naming `--arsenal`.
- The snapshot (`static/vendor/arsenal/`, pairing `ledger`) is committed with its manifest.
- `static/index.html` follows the kit's `app-shell.html` structure, with the hrefs pointing at the snapshot: the rail with the lane-track nav, the sticky top bar with breadcrumbs, `page-head` and `sec` sections. The template's demo content, command palette and theme picker/toggle are not used (the spec says no theme picker).
  - It loads `tokens.css`, `base.css`, `pair-ledger.css`, `ui.css` and `motion.css`, then `static/app.css`. Then the Telegram SDK from `https://telegram.org/js/telegram-web-app.js`, the kit's `icons.js` and `ui.classic.js`, and `app.js` as an ES module.
  - None of the console's own files has an inline script, an `on*=` handler or a `style=` attribute; the CSP allows none of them.
  - The favicon is `data:,`, an empty icon. Without it, the browser's request for `/favicon.ico` returns a 404, which shows as a console error.
- `static/app.css` is the console's layout on top of the kit. It uses token variables only, with no colour or font of its own.
- `static/auth.js` reads the launch parameters once:
  - initData comes from `Telegram.WebApp.initData` when the SDK loaded, else from the `#tgWebAppData=` fragment the dev URL provides;
  - a route may lead the fragment (`#/run/<id>&tgWebAppData=…`), so a signed link can open any view;
  - the parameters are then dropped from the address bar (`history.replaceState`; the route stays);
  - the console's own code keeps initData in memory only. It never puts it into storage, a log or a URL the server sees. (The Telegram SDK keeps its own launch parameters in sessionStorage; fix round 1.)
- `static/api.js` calls `fetch` with `Authorization: tma <initData>`. A non-2xx answer throws `ApiError(status, code, message)` from the server's `{"error": {code, message}}`.
- `static/app.js` is the hash router (`#/runs`, `#/run/<id>`, `#/new`, `#/engines`):
  - it calls `GET /api/me` first; the rail shows the user and the bot version;
  - each view module exports `render(el, params, ctx)` and may return a cleanup function (for polling, the MainButton or the popover);
  - Telegram: `ready()` and `expand()`. The theme is the kit's `dark` when `colorScheme` is `dark` and the default `ledger` otherwise, applied again on `themeChanged`. This holds without the SDK too: there is no OS fallback (fix round 1). The `BackButton` shows on every view except the runs list;
  - every `Telegram.WebApp` call is guarded, for an absent SDK and for a client version below the method's.
- `static/views/common.js` holds the shared helpers:
  - escaping: all text goes through `esc`;
  - status labels as the kit's `.status`, a shape plus a label;
  - the problem panel, and `act()`: the action's button is disabled while it runs, and a failure is shown;
  - `MainButton`/`BackButton`, and `openLink`, which goes through Telegram when it is there;
  - a confirm dialog on the kit's `.dlg` (no `window.confirm`);
  - the Exa estimate formula from `RunConfig.exa_cost_estimate_usd`: searches × search price + pages × page price, with any dotted overrides of the two caps applied.
- `static/views/runs.js` is the runs list (`GET /api/runs`). Each row shows the topic, id, length · depth, the update time and the status: running at a stage, plan or report waiting for you, done, error or cancelled.
- `static/views/run.js` shows one run:
  - progress: the snapshot's counts as KPIs and the events as a timeline, newest first, with only the newest few dozen shown.
    - While a drive runs, it polls `GET …/events?after=<last seq>` every 2 s and reads on in pages until the server's `last_seq` (the bus answers in bounded pages).
    - It re-fetches the snapshot when any event in a poll batch ends a step: a `stage` event that is not a start (end, error, cancelled, draining) or a `gate` event. This holds even when the next stage's start follows in the same batch (fix round 1). If the batch's last stage or gate event is not a start, it keeps re-fetching on each tick until the next stage starts or the drive is over, because the last stage end can arrive before the driver releases the run.
    - After an action, it polls for a few more ticks so the new drive shows up.
    - A polling failure, including a network one, stops the polling and shows the problem. The top bar's refresh starts it again. Only one poll chain runs at a time, and nothing is painted after the view's cleanup (fix round 1).
  - the plan gate (`snapshot.plan`):
    - it shows the summary, sub-questions, search intents, the found sources (or the "Live search found no sources" warning) and `plan.error`;
    - the knobs are the kit's console rows: length; depth (Auto shows the length's default depth); the engines profile with its `profile_problem`; recency, read-only, because the approve route takes no recency (the chat card's button changes it); and the Exa estimate. The estimate is the server's for the drafted plan, recomputed for other choices, and "off" when Exa is off for the run;
    - Approve (also the Telegram MainButton, which shows its progress and ignores a second tap while the action runs) sends `{action: "approve", length, depth, engines_profile}`. Edit needs feedback, and Cancel asks for confirmation in a dialog;
    - when `snapshot.plan` is null (a legacy checkpoint), it shows "Plan not available" and keeps Approve (sending no knobs), Edit and Cancel.
  - the report:
    - the fact-check line comes from `panel_verdict`. `judged < total` shows a "Not fact-checked" warning and is never shown as passed;
    - **Send to chat** has a checkbox for each file the run has and calls `POST …/send`;
    - at the report gate there are Deliver, Extend, Revise (needs feedback) and Cancel;
    - the reader puts the server-sanitised HTML into `article.prose`. At 390 px long URLs wrap and tables scroll inside themselves.
  - tap-to-verify:
    - every `.cite[data-n]` gets `role="button"` and a `tabindex`, so Enter and Space open it too;
    - a tap opens the kit's popover under the citation (`UI.place`). It shows the `evidence.json` source (title, domain, date, type, "pinned by you"), the first few quoted claims tagged "quote verified in the page" or "quote not verified", and a link that opens through Telegram;
    - `evidence.json` is fetched only when `report_paths` lists it. Without one, the popover says the run has no evidence file.
  - Cancel run, in the page head, asks for confirmation in a dialog. The text says the run finishes its current step for a live drive (the drain), or leaves its gate otherwise.
- `static/views/new.js` is the new-research form on the console layout. The fields:
  - topic (required), length, depth (Auto or a preset) and recency;
  - the engines profile, from `GET /api/engines`. If that returns an `engine_config` refusal, the refusal shows in the field and no profile is sent;
  - dotted overrides, one `key=value` per line. Values are parsed as JSON where they parse. A malformed line shows an error under the field, and nothing is sent;
  - the Exa upper bound for those settings: "n/a" without prices, and marked "when Exa is enabled", because the browser cannot know whether it is.

  Start (also the MainButton) calls `POST /api/runs` and opens the run's view.
- `static/views/engines.js` is the engines console, laid out as the kit's `console.html` sections (`set-sec` rows). It makes one change per PUT, as the backend requires:
  - active profile: the validation result (`validation.errors` as a warning, else "Validated · resolved …"), the snapshot's warnings, and each profile with "Make active" (`{active_profile}`). A profile that does not validate is refused with the registry's text, and nothing changes;
  - overrides for a chosen profile:
    - each tier has a provider and a model, with Override and Clear;
    - one role at a time has a primary, an ordered fallback list (add/remove) and `on_exhausted`, with Override role and Clear role override;
    - each PUT sends that profile's whole override object, `{overrides: {<profile>: {tiers, roles}}}`: the stored overrides plus the change;
    - model fields are free text with a `<datalist>` filled from the live model list (`GET …/providers/<name>/models`, once per provider per page). A failed listing is shown under the row, and an id can still be typed;
    - when the provider is unchanged, the saved spec starts from the original (`{...original}`), so per-model fields such as `temperature`, `max_tokens` and `extra_body` survive, and an empty model keeps its `model_from_preferences`. With another provider the spec is `{provider, model}`, and an empty model makes the row say "choose a model" (fix round 1).
  - the transcriber: `asr_engine`, sent as `{transcriber: {…current, asr_engine}}`; the engine's settings are shown read-only;
  - the month-to-date Exa spend and the configured prices;
  - the provider definitions, read-only: the base URL, the key env (set, missing or keyless), the live list or known models, and notes. Key values never reach the page.
- Problem views:
  - a 401/403 from any call replaces the view and stops everything. It shows "Session expired — reopen from the menu button" (`expired`) or "The console could not sign you in" (the other codes), with the server's message;
  - 409 `busy` / `already_running` / `refused`, 400 (`engine_config`, `bad_request`), 404, 502 `send_failed` and 500 each show an explicit panel next to the action: a kicker, a title, the server's message, the code and the status. The inputs are kept;
  - a view that cannot load shows the panel in its place;
  - no answer at all (a failed `fetch`, the tunnel gone), an answer that cannot be read, or an error page that is not the console's JSON (cloudflared's 502) becomes `ApiError(0, "network")`. It shows the "Connection — The console cannot reach the bot" panel or view, which says the tunnel may have restarted and to reopen the console from the menu button (fix round 1);
  - an error that is not an API answer is rethrown (a console error), never hidden.

### Decisions where the spec is silent

- **The engines console is a view in the one page, not a second HTML file.** The hash router has `#/engines`, so `views/engines.js` rebuilds `console.html`'s structure inside `index.html`. One page means one pairing: `ledger`, the house default. (`console.html` itself links `pair-civic`.)
- **The kit templates are in the snapshot as unchanged reference copies**, because the spec lists them. The console does not load them. Their own inline `<script>` and `onclick=` stay, since they are the kit's files and the manifest sums cover them. So the brief's grep finds those two lines under `static/vendor/`; under the console's CSP they could not run even if opened. Outside `vendor/` the grep is empty, and `test_the_console_has_no_inline_script_and_its_references_exist` guards that.
- **No `'unsafe-inline'` for styles either.** The kit's JS positions popovers and tooltips through the CSSOM (`el.style.left = …`), which `style-src 'self'` allows. Nothing uses `style=` attributes or injected `<style>`. The CSP is unchanged.
- **Recency at the plan gate is read-only.** `POST …/plan` takes `length`, `depth` and `engines_profile`; recency changes from the chat card.
- **The runs list does not poll**; the spec has only the run view poll. The top bar's refresh reloads any view.
- **A 4xx answer is a console line in Chromium.** Chromium logs every failed `fetch` response as "Failed to load resource". The kit's checker counts that line as an error. So the views whose job is to show a refusal (the expired-session view, and the 409/400 panels after an action) produce exactly that one line: the network log of the expected answer, never a script error. Controller ruling (fix round 1): on the expired-session and no-initData views, this single line for the expected 401 is accepted as the only console line.

### Behaviour changes for handoff.md

- The `Argus` menu button opens a working console. It shows the runs, a run's live progress, the plan gate, and the report with tap-to-verify citations and Send to chat, and it has a new-run form and the engines settings. It follows Telegram's light/dark theme. A session older than `ARGUS_WEBAPP_INITDATA_MAX_AGE_S` shows "Session expired — reopen from the menu button".
- To update the kit snapshot, run `./venv/Scripts/python.exe scripts/sync_arsenal.py`, then commit the folder. The source is the manifest's `source_root`; pass `--arsenal DIR` on another machine.

### New env vars

None.

### Test files touched

New: `tests/test_sync_arsenal.py`. It covers:

- on a fixture arsenal tree:
  - the fixed list plus the pairing's faces, fonts and licences are mirrored with a sha256 manifest, and nothing unlisted is copied (an unused font is not);
  - another pairing copies only its own faces;
  - a re-sync removes stale files;
  - the default source is the manifest's;
  - no source and no manifest is an error naming `--arsenal`;
  - a missing listed file, an unknown pairing and a reference outside the snapshot each raise;
- the committed snapshot matches its manifest, file for file and sum for sum;
- the console's own HTML has no inline script, no `on*=` handler and no `style=` attribute; every `/static/` reference in it exists; and every relative ES-module import resolves.

### Verification evidence (2026-10-21)

Logs, scripts and screenshots are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t12\`. No test or preview started the bot or cloudflared, or called a live service.

- **The preview** is `preview.py`. It runs `tests/webapp_fakes.make_app` on a `FakeRuns`/`FakeLibrary` with seeded runs:
  - a live drive, with a task publishing events so the polling has something to show;
  - a plan gate, a plan gate with no sources and a profile problem, and a legacy plan gate (`plan` null);
  - a report gate whose report has hostile markdown, a table and a long URL, with `evidence.json` and a PDF;
  - a delivered run without `evidence.json`, an error run and a cancelled run.

  The engines are the real `argus_engines.json` with `registry._get_json` patched (no network; ollama's listing fails) and a dummy key in that process only. A second instance (`--busy`) answers `busy` to a start and `already_running` to an approve. Every URL is signed by `webapp_fakes.init_data`, plus one expired URL (auth_date two hours back) and one without initData.
- **RED** (`red.log`): `tests/test_sync_arsenal.py` failed at collection before the script existed.
- **GREEN**: `tests/test_sync_arsenal.py` passes. The full suite (`suite.log`, `PYTHONPATH="" FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1 ./venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`) fails only the live-proxy tests in `tests/test_llm.py` and the `tests/test_graph.py` fixture (the closed port, by design), and no warnings summary appears.
- **The kit's render checker**, `tools/render.py check <url> --width 390 --height 844 --wait 2500`, ran in light (`--scheme light`) and dark (`--scheme dark --theme dark`) for every view (`check_all.sh`, `render-check-final.log`; `render-check-rerun.log` re-checks the plan and report views after the last edit):

  ```
  runs · light OK · dark OK
  run-live (polling) · light OK · dark OK
  run-plan · light OK · dark OK
  run-plan-nosources (profile problem) · light OK · dark OK
  run-legacy (plan null) · light OK · dark OK
  run-report (tap-to-verify, hostile markdown) · light OK · dark OK
  run-noevidence · light OK · dark OK
  new · light OK · dark OK
  engines · light OK · dark OK
  expired · light FAIL · dark FAIL — the only line: console.error: Failed to load resource: the server responded with a status of 401 (Unauthorized)
  no-initdata · light FAIL · dark FAIL — the same single 401 line
  ```

  The two FAILs are the views that exist to show a 401: Chromium logs the expected 401 response of `GET /api/me` as that line, and the page then shows the session-expired / sign-in view. There is no script error and no CSP violation in any view. The CSP was the server's, and the Telegram SDK loaded from telegram.org. Its `postEvent` lines are `console.log`, not errors.
- **Scripted flows** (`flows.py` / `flows-final.log`: Playwright on Edge at 390 px, failing on any assertion; `flow-*.png`):
  - tap `[2]`: the source, both quotes and their verified/not-verified tags, in light and dark;
  - a run with no `evidence.json` says so in the popover;
  - Send to chat: "Sent report.md, report.pdf to the chat.";
  - the plan knobs: Exhaustive recomputes the estimate;
  - Approve shows its toast; Cancel goes through the dialog;
  - `busy` on Start shows the panel with the driver's text and keeps the topic typed;
  - `already_running` on Approve shows its panel;
  - a malformed override line is refused before sending;
  - live polling: the newest timeline row changes within 4.5 s;
  - engines: making `local-ollama` active shows the registry's refusal ("role intake.classify: choose a model for provider ollama …"); a judge-tier override from the live list saves and shows "console override"; ollama's failed model listing is shown under the row.

  The only console errors are Chromium's lines for the expected 409/400 answers. `scrollWidth` is 390 on every page, so nothing scrolls horizontally.
- I looked at the light, dark and 390 px screenshots of every view (`*.png`).
- `grep -rn "<script>[^<]\|onclick=" src/argus/webapp/static --include=*.html` finds only the kit's reference templates under `static/vendor/` (see the decisions). Outside `vendor/` it is empty.
- Not checked live: the console inside Telegram, behind the tunnel, and against the real driver. Those are the orchestrator's §4 checks.

### Task 12 fix round 1 (review of fc132ee)

What changed (the sections above are updated to match):

- `views/run.js`, polling:
  - the snapshot is re-fetched when **any** event in a poll batch ends a step: a stage event that is not a start, or a gate. The driver publishes `[end A, start B]` milliseconds apart, so before this fix the batch's last event (the start) won, and the KPIs and stage stayed as loaded for the whole drive;
  - the "until the next start" retry follows the batch's last stage or gate event;
  - `tick` and `refresh` return after each `await` once the view is gone, so nothing is painted after cleanup. The catch that silently returned after cleanup is gone, and every error goes to the problem panel;
  - one poll chain at a time: a `ticking` flag makes `schedule()` and a second `tick()` do nothing while a tick runs, and the tick reschedules itself when it ends.
- `api.js`: a `fetch` rejection, an answer that cannot be read, or an error answer that is not the console's JSON (cloudflared's 502 page) becomes `ApiError(0, "network", …)`. The message ends "the tunnel may have restarted — reopen the console from the menu button".
- `views/common.js`:
  - `PROBLEMS.network` is "Connection — The console cannot reach the bot";
  - the MainButton wrapper shows progress, disables the button and ignores a second tap while its action runs, then restores it.
- `app.js`:
  - a failure of the start-up `GET /api/me` that is not 401/403 (network included) shows the "This view could not load" problem view, which the top bar's refresh retries; before, it rethrew and left the skeleton;
  - the theme is `dark` when `colorScheme === "dark"`, else the default `ledger`, also without the SDK (no OS fallback).
- `views/engines.js`:
  - with the provider unchanged, a saved tier, role or fallback spec starts from the original (`{...original}`, with `model_from_preferences` cleared when a model is typed), so per-model fields survive. Fallback rows carry their original index, so a removal does not shift which original a row merges with;
  - a 401/403 while loading a model list goes to the sign-in view through `showProblem`.
- Notes:
  - **sessionStorage**: the Telegram SDK keeps its own launch parameters in sessionStorage; the console's own code stores nothing.
  - **Controller ruling**: the single Chromium "Failed to load resource" line for the expected 401 is accepted as the only console line on the expired-session and no-initData views.
  - **The network view behaves the same way**: Chromium logs the 502 (or the failed fetch) once, and the page shows the panel.

Evidence (`A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t12-fix1\` and `…\t12\`):

- The reviewer's probes (`t12-review\probe_stage_end.mjs`, `probe_network.mjs`) were re-run on copies of the fixed files (`t12-fix1\static\`), with one shared stub (`fake_el.mjs`, a DOM element deep enough for `paint`; the reviewer's stubs had no element to paint into):
  - `probe_stage_end.after.log`: requests `['/api/runs/r1/events?after=0', '/api/runs/r1']`, "snapshot re-fetched after the stage end: true" (it was `false`);
  - `probe_network.after.log`: "tick resolved", "problem panel shown: true", "next poll scheduled: false" (before: "tick rejected: TypeError", panel false).
- `probe_guards.log` (new): the network panel names the problem and the menu button; a tick that resolves after cleanup paints nothing and schedules nothing; a second tick entering while one is in flight schedules no second chain (one timer). The same probe on the reviewer's copies of fc132ee crashes at the first case, with an unhandled `TypeError: Failed to fetch` (`probe_guards.before.log`).
- `t12\flows_fix1.py` / `flows-fix1.log` (Edge at 390 px):
  - aborting `…/events` on a live run shows the Connection panel, and no further events request follows within 4.5 s (polling stopped);
  - with every `/api` answer an HTML 502 (`preview.py --gateway`), the page shows "This view could not load" with the Connection panel;
  - with the SDK blocked and the OS dark, `data-theme` is `ledger`.
- `render-check-fix1.log`, `tools/render.py check` at 390 px in light and dark:
  - run-live, run-plan, run-legacy and run-report: OK in both;
  - the network view (`--gateway`): its only console line is Chromium's "502 (Bad Gateway)" log of the expected answer, in both.
- `flows-fix1-regression.log`: every earlier flow passes again.
- Full suite (`t12-fix1\suite.log`, `PYTHONPATH="" FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1 ./venv/Scripts/python.exe -m pytest -q`): the only failures are the live-proxy tests in `tests/test_llm.py` and the `tests/test_graph.py` fixture, by design. No warnings summary.
