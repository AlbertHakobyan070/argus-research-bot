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
      - `auth` requires `Authorization: tma <initData>` on every `/api*` path. A refusal is logged as `webapp: <method> <path> refused: <code> (fields present: <names>)`. The source IP is never trusted.
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
- **Failure DMs are de-duplicated**: one per run of failures (a crash-looping cloudflared does not DM every backoff), and reset by the next good URL. A good URL after the first DMs "🌐 Console tunnel restarted: <url>".
- **A menu-button failure** keeps the URL (the console is reachable) and sets `webapp_ok` False with "menu button not registered: …", plus a DM.
- **A missing cloudflared binary** (checked with `is_file()` before the supervisor starts) is a failure ("cloudflared not found at <path> (set ARGUS_CLOUDFLARED)"), not a retry loop. The local server keeps serving.
- **An unknown `ARGUS_WEBAPP_TUNNEL`** starts nothing and is reported.
- **`snapshot.plan` is `runs.plan_payload()`**, which is a view of tracked runs. A run resting at its plan gate but untracked (after a restart) therefore shows `plan: null`, while `awaiting` is `"plan"`. Approve and the other actions still work (they attach the run), and after the attach the payload appears.
- **`PUT /api/engines` applies its parts in order** (overrides, transcriber, active profile). A failure in a later part leaves the earlier ones applied (a `ponytail:` note in `put_engines`).
- **`GET /api/me`'s `active_profile`** is null when no engines config is loaded; `/api/engines` then says why (400 `engine_config`).
- **The webapp imports nothing from `argus.bot` at module level** (`_send_md_doc`, `_safe_send` and `runs` are imported inside the functions), because `bot.app` imports `webapp.server` at module level.

### Behaviour changes for handoff.md

- The bot serves the Mini App console on `127.0.0.1:${ARGUS_WEBAPP_PORT:-8787}` and logs `webapp: listening on 127.0.0.1:<port>`. With the default quick tunnel it starts `tools/bin/cloudflared(.exe)` and logs `tunnel: https://<x>.trycloudflare.com`. It sets the chat's `Argus` menu button to that URL and logs `menu button registered for chat <id>`. cloudflared's stderr goes to `logs/cloudflared.log`.
- The console is optional. If the server cannot start (port taken), cloudflared is missing, the tunnel dies or the menu button fails, the problem is logged, DM'd ("⚠️ Console down — …") and shown in `/status`; media and research in chat are unaffected. A killed cloudflared is restarted with backoff, and the new URL is DM'd and re-registered.
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
