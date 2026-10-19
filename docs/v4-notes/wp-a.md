# WP-A — Engines registry and LLM facade (Task 4)

Spec: `docs/argus-v4-design.md` §3 WP-A, with §2.2 (engines config, validation and activation) and §2.3 (the facade). Every LLM call now names a role and runs on a binding from an `EnginesSnapshot`: a primary, explicit fallbacks and a failure policy. Every attempt is a `CallRecord` with the requested model, the served model and `routed_via`. The Phase 1 nodes are unchanged: the v3 wrappers route through `complete()` with role `legacy.<tier>`.

### What changed

- `src/argus/engines/registry.py` (new), the async registry:
  - `load_config(paths=None, library=None)` merges the layers, lowest first: the defaults file (`ARGUS_ENGINES_FILE`, default `argus_engines.json`), the optional local file (`ARGUS_ENGINES_LOCAL_FILE`, default `argus_engines.local.json`), and the library settings keys `engines.active_profile`, `engines.overrides` and `engines.transcriber` (JSON, through `Library.get_setting`/`set_setting`). Dicts merge key by key. An engine spec (a dict with `provider`), a list or a scalar replaces. The file read runs in a worker thread.
    - Files are read as `utf-8-sig`, so a UTF-8 file with a BOM (Notepad) loads.
    - A file that can't be decoded (a Windows ANSI file with non-ASCII text, a UTF-16 file from PowerShell 5.1 `>`) raises `EngineConfigError` naming the file: "… is not UTF-8 (…); save it as UTF-8".
    - A file that can't be stat'ed or read raises "… cannot be read (<error>)", with no encoding advice. The stat is inside the same guard, so a `PermissionError` from `is_file()` is covered.
    - At startup either is one more logged and DM'd problem, never a crash.
  - Startup checks, with no I/O:
    - `validate_schema(cfg)`: the pydantic validators, names that match their keys, and every `model_from_preferences` pointing at a preference list its provider has.
    - `static_problems(cfg)`: covers the active profile's primaries only. It reports a missing `api_key_env` (one line per provider) and a fixed id outside a `known_models` provider's list.
  - Resolution:
    - `resolve_snapshot(cfg, profile)`: a fixed `model` resolves as itself. `model_from_preferences` takes the first preferred id in the live `/models` list (fetched in a worker thread). A spec with neither gets "choose a model for provider X (/engines models X, or the console)".
    - An unresolved or keyless primary raises `EngineConfigError` naming the role and the provider.
    - An unresolved or keyless fallback is dropped, with a line in `snapshot.warnings`.
    - `validate_active(cfg)` is `validate_schema` plus `resolve_snapshot` of the active profile.
  - The cache:
    - `active_snapshot()` returns the cached snapshot, or loads, validates and caches it. Only a success is cached.
    - `current_snapshot_cached()` is the sync accessor for the wrappers. It raises when no drive entry resolved first.
  - Changes, which validate before they persist:
    - `set_active_profile(name)`.
    - `set_overrides(profile, overrides)`: a full check for the active profile, the schema check for any other.
    - `set_transcriber(t)`.
  - `base_url(prov)` (env override first) and `api_key(prov)`. A keyless provider gets `"none"`, never `None`. A missing key raises.
  - `live_models(name)`: the live list, cached for `MODELS_TTL_S`, or `known_models` when the provider has no endpoint. An error names the URL and never the key: a key quoted by the underlying error is replaced by `***`, and that error is not chained.
  - `family(model_id)`: the prefix table of the original WP-A text.
  - `summarize_calls(model_calls, errors=None, *, exa_usd=None)`:
    - `by_role`, keyed by role. v3 records, which have no role, count as `legacy.<tier>`.
    - Each row counts **logical calls** in `calls`: `ok` successful calls plus `exhausted` calls. An exhausted call is each "role <id>: every engine failed after <n> attempts" that `errors` quotes, wherever in the line.
    - **Attempt records** are counted apart and are never labelled calls:
      - `attempts` and `failed_attempts` cover the records in `model_calls`, including the `prior_attempts` the wrappers attach;
      - `exhausted_attempts` is the sum of those `<n>`.
    - The row also has `skipped` and `degraded` (from the `skip:` and `degrade:` lines), `served_models`, `fallbacks`, `retries` and `avg_latency_ms`.
    - Top level: `calls` (logical), `exhausted`, `attempts`, `reroute_rate`, `judge_families`, `writer_families`, `judge_writer_same_family` (`None` when unknown), `warnings`, `degradations`, and `exa_usd` when it is given.
    - **During Phases 1–2, the per-attempt records of an exhausted legacy call are absent from `model_calls`.** The v3 call sites are outside WP-A: they catch the `EngineError` and keep only its text, so an exhausted call is visible only where its call site quotes the facade's message in `errors`.
      - The digest, brief draft, scout query generation (`scout: querygen failed (…)`), outline, compose and panel call sites quote it.
      - Two only log it: the brief re-draft (`graph/brief.py`, "brief re-draft failed … keeping first") and the follow-up query generation (`graph/research.py`, "followup querygen failed … deterministic fallback").
      - Intake and the quick answer don't catch it, so an exhaustion there fails the run.
      - The node migrations append `EngineError.attempts` to `model_calls` later.
- `src/argus/engines/llm_openai.py` (new), the driver. It calls the openai SDK directly:
  - One client per (provider, model), built with `max_retries=0`, `api_key=registry.api_key(...)`, the provider's `timeout_s` and the provider's `headers`.
  - `with_raw_response`, so response headers are read.
  - The token cap is sent under the provider's `token_param`.
  - `temperature` is omitted when the engine's `supports_temperature` says so (falling back to the provider's flag).
  - `response_format={"type": "json_object"}` when `json_mode` is set.
  - `extra_body` is the provider's, merged with the engine's.
  - `routed_via` comes from the header `x-routed-via`, else the body's `_routed_via`, else `x_groq`.
  - An unusable response raises `DriverError` and is never an empty success. That covers an `error` object in the body (FreeLLMAPI and upstreams send these with HTTP 200), no choices, a body that isn't JSON, and a body the SDK can't parse as a chat completion. `DriverError.code` carries the error object's numeric code.
- `src/argus/llm.py`, rewritten as the facade:
  - `complete(role, messages, *, engines, temperature, max_tokens, json_mode, expect_json, validate, item) -> LLMResult`, plus `NOT_VALIDATED`, `CallRecord`, `LLMResult` and `strip_reasoning`.
  - The exceptions `EngineError`, `EngineSkipped` and `EngineDegraded` (which carries `degradation`). `EngineConfigError` is re-exported from the registry.
  - Each engine is tried in turn: the primary, then the fallbacks.
    - Transient errors (connection, 5xx, 429, and a `DriverError` whose code is 429 or 5xx) get up to three tries, with a 1 s then 2 s backoff, or the `Retry-After` value up to `retry_after_cap_s`.
    - A timeout is retried at most `timeout_retry_max` times.
    - Any other error fails that engine, a `DriverError` without such a code included.
    - Every failed attempt is a `CallRecord` with `ok=False` and the provider's message. The provider's key is replaced by `***` in the record and in the log line.
    - Tagged reasoning blocks are then stripped.
    - A JSON role cut off at its limit (`finish_reason == "length"`) gets one retry at twice the budget, capped at 4096. With no budget, the retry uses 4096.
    - A reply the `validate` hook rejects gets one retry with "Reply with only the JSON object."
  - When every engine has failed, the binding's `on_exhausted` applies: `raise` raises `EngineError`, `skip_item` raises `EngineSkipped`, and `degrade` raises `EngineDegraded`. The message is "role <id>: every engine failed after <n> attempts (last: …)", so the role id and the attempt count can be parsed back out of `errors`.
  - A per-provider `rpm_limit` sliding window applies across threads.
  - The Phase 1–2 wrappers `chat_for_tier`, `invoke_with_retry`, `record_from_response`, `resolve_tier` and `pick_strong_and_judge` keep their call sites. They route through `complete("legacy.<tier>", …, engines=registry.current_snapshot_cached())`. `record_from_response` returns the successful attempt, with the earlier attempts in `extra["prior_attempts"]`.
- `argus_engines.json` (new):
  - The providers: `freellmapi`, `groq`, `openrouter`, `gemini`, `openai`, `anthropic`, `ollama` and `lmstudio`.
    - Key names only.
    - `freellmapi` and `openai` use `max_completion_tokens`.
    - `freellmapi` has `supports_json_mode: false`, as in v3.
    - `freellmapi`'s `model_preferences` are v3's `PREFERRED` lists, read from `abf1f77:src/argus/llm.py`, with `"auto"` last.
    - `anthropic` has `known_models` (Appendix B) and no `/models` endpoint.
  - The §2.2 profiles, with `active_profile: "free-proxy"`.
    - Every degradable and skippable role is declared with its shipped policy.
    - `brief.draft`, `compose.flat` and `quick.answer` stay `raise`.
    - `free-proxy` reproduces v3: each tier is `freellmapi` with `model_from_preferences` = that tier.
  - `pricing.exa`, from Appendix B, with its date.
- `src/argus/bot/engines_cmd.py` (new):
  - `/engines` shows the profile, every binding with its fallbacks and policy, the validation state, the snapshot warnings, the last run's served tally and reroute rate, the judge-diversity line with the command that fixes a ⚠, the month-to-date Exa spend (`n/a` until the library records it), and the profile list.
  - `/engines use <profile>`, `set <role> <provider>/<model>`, `tier <tier> <provider>/<model>`, `fallback <role> <provider>/<model>`, `reset [profile]` and `models <provider>`.
  - `set` and `fallback` take `ROLES` ids only. The `legacy.*` roles follow `tier`.
  - An error handler sends any `EngineConfigError` raised by a drive entry to the chat as its text. It then stops the generic handler from logging it as unhandled.
- `scripts/engines_probe.py` (new):
  - Usage: `[--provider NAME | --all] [--model ID]`.
  - For each provider whose key is set, it lists the models and then makes single direct calls through the driver: `plain` at 20 tokens, `json_mode`, `temperature=0.2`, the `cap16` check, and a 200-token JSON call that must return content. It also reports whether `routed_via` was captured.
  - It prints a Markdown table and suggested flag changes. It never prints keys: every row's text passes through one redaction step, so a key echoed in an error comes out as `***`.
  - Every driver failure (`openai.APIError`, `DriverError`, `EngineConfigError`) gives that check a FAIL row. The run continues and the table is printed.
  - It exits 1 when a call failed or a check did not hold. `--all` skips keyless local servers; pass them by name. When nothing can be probed (no key set), it says so and exits 1, with no empty table.
- Shared-file edits:
  - `src/argus/bot/app.py`:
    - `HANDLER_MODULES.append(engines_cmd)`.
    - A `_post_init` block: `load_config(library=lib)`, `validate_schema` and `static_problems`, with no network. Each problem is logged at WARNING, and one DM to `TELEGRAM_ALLOWED_USER_ID` lists them all. An `EngineConfigError` while loading counts as a problem, so startup continues.
    - A WARNING for each removed env knob that is still set.
  - `src/argus/bot/research.py` and `src/argus/bot/history.py`: one `await registry.active_snapshot()` at the start of each drive entry, after the existing guards and before any graph call. The entries are `research_cmd`, `ask_cmd`, `_replan_after_edit`, `_revise_after_feedback`, `_resume_after_plan`, `_resume_after_report` and `continue_cmd`.
  - `src/argus/graph/report_builder_helpers.py`: `build_sidecar_metadata(..., errors=None)` adds `metadata["engines"] = summarize_calls(model_calls, errors)`.
  - `src/argus/graph/nodes.py` (WP-C's file): one line in the fix round, approved by the orchestrator. The `report_builder_node` call to `build_sidecar_metadata` passes `errors=state.get("errors") or []`, so the summary sees the exhausted, skip and degrade lines.
  - `src/argus/config.py` (WP-D's file, two edits the WP-A text requires):
    - `Settings.request_timeout_seconds` (`ARGUS_TIMEOUT`) is deleted with the wrappers, its last reader.
    - The `FREELLMAPI_API_KEY` missing/malformed `RuntimeError` is removed, because the acceptance needs the bot to start without the key. The registry reports the missing key instead. The `freellmapi_*` fields stay, because `manual_e2e`, `demo_run` and `test_config_tools` read them.
  - `tests/manual_e2e.py`: after `build_graph`, it resolves the snapshot as the bot does.

### Contract decisions where the spec is silent

- **A policy-only binding follows the tier.** A profile's role binding whose primary equals that profile's own spec for the role's tier only declares the failure policy. Its engine follows the effective tier, overrides and `/engines tier` included. So `/engines tier judge …` moves `panel.grounding` and `legacy.judge` together. Any other primary is pinned. The caveat: editing a shipped profile's `tiers` in the local file pins the roles that repeated the old spec. Change tiers with an `overrides` block or `/engines tier` instead.
- **`EngineConfigError` is defined in the registry** and re-exported as `llm.EngineConfigError`, because `llm` imports the registry.
- **Refusal text reaches the chat through an error handler** that `engines_cmd.register` adds, rather than an edit to `app._on_error`. PTB runs error handlers in insertion order, and the handler raises `ApplicationHandlerStop` after replying.
- **Only a successful resolution is cached.** A proxy that comes back is picked up at the next drive entry, without a restart. A `set_*` call that lands during a resolution wins (single-user bot; no lock).
- **`/models` listings** are cached per (provider, URL) for `MODELS_TTL_S`. A failed listing is not cached. Within one `resolve_snapshot`, one failure serves every role of that provider.
- **Key checks follow the spec's B6 rule:** only a primary's key is required. A keyless provider (`ollama`, `lmstudio`) never needs one.
- **The OpenRouter `HTTP-Referer` and `X-Title` values** are the project's repository URL and `Argus`. The spec names only the headers.
- **`served_provider`** is the first segment of `routed_via`. v3 recorded it as `""` on every call.
- **`CallRecord.extra["headers"]`** keeps `retry-after`, `x-fallback-attempts` and `x-ratelimit-*` from the response.
- **v3's judge re-pick is not reproduced.** v3 re-picked the judge when it shared a family with the strong model. v4 binds the judge from the profile and reports the diversity instead: the `/engines` line and the `summarize_calls` warning.
- **`legacy.*` model override:** `invoke_with_retry` raises `ValueError` when a call site's `model_override` differs from the bound model. In practice the call sites pass the model `resolve_tier` returned.
- **Probe model choice:** `--model`, else a fixed tier model of any profile for that provider (cheapest tier first), else the first preference its `/models` list serves.

### Behaviour changes for handoff.md

- The bot starts with the proxy down or `FREELLMAPI_API_KEY` unset. Engines problems are logged and sent once as a DM at startup. Until they are fixed, `/research`, `/ask`, the plan and report buttons (approve, edit, revise, Send) and `/continue` reply with the `EngineConfigError` text, while media, `/find` and `/runs` keep working.
- `Settings` no longer checks the `freellmapi-` prefix of the key. A malformed key now shows up as an error at a run's first call, not at startup.
- **Fewer retries under proxy pressure.** v3's `ChatOpenAI` left the openai client's default of two SDK retries (`openai.DEFAULT_MAX_RETRIES`, checked in the venv) under its own three attempts. The spec sets `max_retries=0`, so each engine now gets three requests, with a 1 s then 2 s backoff, unless the proxy sends `Retry-After`. During a quota burst a call gives up sooner than in v3; see the live suite below.
- `ARGUS_TIMEOUT` and `ARGUS_LLM_RETRIES` are no longer read. A startup WARNING names each one still set, and `PERPLEXITY_API_KEY` (§7) too. The timeout is `ProviderSpec.timeout_s`; the retry policy is the facade's.
- `model_calls` entries carry the v4 fields: role, profile, provider, requested and served model, `routed_via`, attempt, fallback rank, retry reason, finish reason and token counts. `metadata.json` gains `engines` (the summary).
- The `/models` fetch is not retried; v3 retried it. A failed listing fails the resolution, with the URL in the text.
- `/engines` is new. `HELP_TEXT` and `docs/help.md` don't list it yet; they belong to other WPs.
- Until their WPs add `await registry.active_snapshot()`, `scripts/demo_run.py` (WP-D) and `bench/runner.py` (WP-G) raise "no engines snapshot resolved".
- A response with an error object in its body is a failed attempt, as in v3, never an empty success. So is a response with no choices or one that isn't a chat completion. An error object with code 429 or 5xx is retried like a 429 or 5xx status.
- An exhausted call's error text reads "role <id>: every engine failed after <n> attempts (last: …)". `metadata.json["engines"]` counts such calls in `calls` and `exhausted`, and keeps attempt counts apart (`attempts`, `failed_attempts`, `exhausted_attempts`). The earlier per-row `errors` field is gone.
- `argus_engines.json` and the local file must be UTF-8; a BOM is fine. Anything else is a startup problem (logged and DM'd) that names the file, and runs are refused until it is fixed.

### New env vars

- `ARGUS_ENGINES_FILE`: the defaults file, relative to the repository unless absolute. Default `argus_engines.json`.
- `ARGUS_ENGINES_LOCAL_FILE`: the optional, gitignored local file. Default `argus_engines.local.json`.
- The provider key names declared in `argus_engines.json`: `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`, besides the existing `FREELLMAPI_API_KEY` and `FREELLMAPI_BASE_URL`. A key is needed only when the active profile binds that provider as a primary.
- Removed: `ARGUS_TIMEOUT` and `ARGUS_LLM_RETRIES`, each with a startup WARNING.

### Test files touched

- New:
  - `tests/test_engines_registry.py`: resolution, the key check for primaries only, the follow rule, startup checks with no I/O, merge order and per-profile scoping, persistence and refusal, the `/models` TTL, the shipped file, `local-ollama` "choose a model", `family` and `summarize_calls`.
  - `tests/test_llm_facade.py`: the fallback walk; policy → exception class; `ALL_SKIPPED_RAISES` at a fake per-item site; `validate` → nudge → fallback → `on_exhausted`, including the Review Focus 2 test `test_untagged_reasoning_on_json_role_retries_then_falls_back` and its degrade variant; `strip_reasoning`; truncation; `NOT_VALIDATED`; and the driver on an httpx `MockTransport` (`max_retries == 0`, never `api_key=None`, `token_param`, the temperature omission, merged `extra_body`, `routed_via`). Also the legacy wrappers.
  - `tests/test_engines_cmd.py`: the `/engines` handlers on fake Updates, including `tier` fixing `legacy.*`; registration through `HANDLER_MODULES`; real `_post_init` runs on temporary DBs with a network that raises (a broken file, static problems and the removed-knob WARNINGs, a clean config); the error handler; every drive entry refusing before it drives; and a resume after a restart resolving before the wrappers run.
- Rewritten:
  - `tests/test_llm.py`: the live smoke, through the registry and `complete`.
  - `tests/test_llm_retry.py`: the facade's retry policy, `Retry-After` honoured and capped, and the rpm limiter on a fake clock.
- Edited:
  - `tests/test_reflexion.py`, `tests/test_brief.py`: the v4 `CallRecord`.
  - `tests/test_hitl_edit_revise.py`, `tests/test_run_scoping.py`, `tests/test_continue_extend.py`: a fake `active_snapshot`, and no `request_timeout_seconds`.
  - `tests/test_plan_hitl_gating.py`, `tests/test_delete_browser.py`, `tests/test_media_bot.py`: no `request_timeout_seconds`.
  - `tests/test_graph.py`: an autouse fixture that resolves the shipped config against the live proxy.
  - `tests/manual_e2e.py`: resolves before driving.
- Fix round 1: new cases in the three new files.
  - `test_llm_facade.py`: error bodies, no choices, non-JSON and non-completion bodies through the real driver and SDK; key redaction; exhaustion by policy; the truncation retry without a budget; the `ALL_SKIPPED_RAISES` case reworked to check the facade's `EngineSkipped`; and the risk-9 scenario through `report_builder_node`.
  - `test_engines_registry.py`: the decoding cases, a `/models` error that quotes the key, the new `summarize_calls` fields, and the probe's `--all` with no key.
  - `test_engines_cmd.py`: startup with a cp1252 or UTF-16 local file.
- Fix round 2:
  - `test_engines_registry.py`: the probe over five unusable answers (a 200 error body, one that echoes the key, empty choices, HTML that echoes the key, a 401 that echoes the key), and a file whose stat raises `PermissionError`.
  - `test_engines_cmd.py`: startup with that `PermissionError`.

### Verification evidence (2026-10-19)

Scripts and logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t4\`.

- **RED**, with only the tests written:
  - `red.log`: collection fails with `ImportError: cannot import name 'registry' from 'argus.engines'` in each file that imports it.
  - `red_b.log`, the edited existing tests: `CallRecord.__init__() got an unexpected keyword argument 'role'` in `test_reflexion`/`test_brief`, and `Settings.__init__() missing 1 required positional argument: 'request_timeout_seconds'` wherever a test builds `Settings`.
- **GREEN, focused** (`green1.log`): the new and edited test files with `FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1`, so nothing can reach the proxy: `139 passed`.
- **Mutation check** (`mutate.py`, `mutate.log`): each mutation is applied, its test is run, and the file is restored. Every one was caught:
  - SDK `max_retries=2`;
  - no second validate failure;
  - no tier-following;
  - no `active_snapshot()` in `_resume_after_plan`;
  - one extra timeout retry;
  - an uncapped `Retry-After`;
  - an api key of `None`;
  - the startup handler catching the wrong exception.
  - The last one first hung pytest at exit, because `_post_init` raised before the test closed the saver. The test helper now closes it in a `finally`.
- **Full suite, proxy URL on a closed port** (`suite_closed_port.log`; `suite_closed_port_final.log` after the last test edits): only the live tests fail. They are `tests/test_llm.py` and the `tests/test_graph.py` fixture, with `EngineConfigError … cannot list models at http://127.0.0.1:9/v1/models`. Everything else passed, so no other test reaches the network.
- **Full suite, live proxy** (`suite_live.log`): `1 failed, 635 passed, 1 skipped`.
  - The failure is `tests/test_graph.py::test_deep_graph_pauses_for_plan_approval` ("digest must produce evidence notes"). FreeLLMAPI answered 429 "All models exhausted" (OpenRouter `free-models-per-day`) and 502 (an upstream Mistral 403).
  - The re-run of only the live tests (`suite_live_rerun.log`) gave `1 failed, 3 passed`: the same test, this time at "the report must cite the fetched source" (the writer got 429 "All models exhausted" on every try).
  - Quota, per the brief; see the retry change under Behaviour changes. The quick graph test and both `test_llm.py` smokes passed live, with `routed_via` populated.
- **Probe dry runs** (`probe_dryrun.py`, `probe_dryrun.log`): an httpx `MockTransport` and a dummy key, in four modes (good, cap ignored, empty at budget, field refused). Each mode gave the expected status, suggestion and exit code. A key echoed in an error was redacted. `probe_closed_port.log` shows every call failing cleanly with exit 1.
- **Static scan** (`static_scan.py`, `static_scan.log`): there is no linter in the venv, so a stdlib `ast` scan of the WP-A modules and tests checked for unused imports and undefined names: clean. The unused imports it lists in edited older tests were already unused at `abf1f77`, and are left as they were.
- **Probe, the one live call** (`probe_live.log`): `PYTHONPATH="" PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe scripts/engines_probe.py --provider freellmapi`. In the model column, the `models` and `routed_via` rows show the requested model; the call rows show the served one.

| provider | model | check | status | finish | tokens (completion/reasoning) | ms | detail |
|---|---|---|---|---|---|---|---|
| freellmapi | llama-3.1-8b-instant | models | ok |  |  |  | 96 listed |
| freellmapi | openai/gpt-oss-safeguard-20b | plain | ok | stop | 45/29 | 5407 | 'The capital of France is Paris.' |
| freellmapi | openai/gpt-oss-safeguard-20b | json_mode | ok | stop | 67/49 | 297 | parses as a JSON object |
| freellmapi | openai/gpt-oss-safeguard-20b | temperature | ok | stop | 57/41 | 250 | temperature=0.2 accepted |
| freellmapi | codestral-latest | cap16 | FAIL | stop | 709/0 | 5670 | max_completion_tokens=16: ignored (709 tokens) |
| freellmapi | codestral-latest | json200 | ok | stop | 17/0 | 438 | not JSON; '```json\n{"city": "Paris", "country": "France"}\n```' |
| freellmapi | llama-3.1-8b-instant | routed_via | ok |  |  |  | groq/openai/gpt-oss-safeguard-20b, mistral/codestral-latest |

Suggested flag changes:
- freellmapi: response_format accepted on llama-3.1-8b-instant: "supports_json_mode": true is possible (config says false)
- freellmapi: the max_completion_tokens cap was ignored: try "token_param": "max_tokens"

(exit 1, from the `cap16` FAIL.)

What the probe shows:
- **`routed_via` is captured.** The proxy rerouted the requested `llama-3.1-8b-instant` to `openai/gpt-oss-safeguard-20b` (via groq) and then to `codestral-latest` (via mistral). Both appear in `routed_via`, and the `CallRecord` keeps the served model apart from the requested one.
- **The proxy doesn't enforce `max_completion_tokens` on rerouted calls.** `cap16` got 709 tokens from `codestral-latest`. `plain` got 45 tokens (29 of them reasoning) against a cap of 20. The suggestion to try `max_tokens` is untested, because only one live run is allowed. `argus_engines.json` keeps `max_completion_tokens`, which is what v3 sent and what the WP-A text requires. Token caps on `freellmapi` are therefore advisory, and budgets that depend on them need the orchestrator's attention.
- **The small-budget JSON call returned content** (no thinking-budget symptom). It was fenced in a ```` ```json ```` block, because `freellmapi` sends prompt-only JSON (`supports_json_mode: false`).
- **The `response_format` suggestion** names the requested model, but the call was served by `openai/gpt-oss-safeguard-20b`. `supports_json_mode` stays `false`, as in v3, because the proxy routes to backends whose support varies.

Not run here, per the brief's single live call (orchestrator checks):
- `tests/manual_e2e.py`.
- The bot started with `FREELLMAPI_BASE_URL` on a closed port, then `/runs` and `/research x`.
- The bot started with `FREELLMAPI_API_KEY` unset, checking for the DM.

The unit tests drive the same `_post_init`, drive entries and error handler.

### Fix round 1 — verification evidence (2026-10-19)

The fix round made no live calls. Logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t4fix\`. The review's repro scripts are in `scratchpad\t4-review\`.

- **RED** (`red_fix.log`): the new and changed test cases of the three new test files, run against `097c99d` (extracted with `git archive`, so `src/` is the pre-fix code) → `18 failed, 58 passed`. The failures:
  - an error body gave `''` instead of the fallback's text, and the call did not raise;
  - `[None] == [None, 4096]` for the truncation retry;
  - no "after N attempts" in the exhaustion text;
  - `KeyError: 'exhausted'`;
  - the BOM file rejected as invalid JSON;
  - the `/models` error printed `Bearer sk-proxy-secret`;
  - the probe's `--all` exited 0;
  - startup raised `UnicodeDecodeError` for the cp1252 and UTF-16 local files.
- **GREEN, focused** (`green_fix.log`): `test_llm_facade.py`, `test_engines_registry.py`, `test_engines_cmd.py` and `test_llm_retry.py` with `FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1` → `84 passed`.
- **Full suite, proxy URL on a closed port** (`suite_fix.log`): `2 failed, 646 passed, 1 skipped, 1 warning, 2 errors`. Only the live tests failed, as by design with no proxy: `test_llm.py` ×2, and the `test_graph.py` fixture ×2.
- **Review repros after the fix** (`review_repros_after.log`):
  - `errbody_check.py`: the legacy call now raises `EngineError: role legacy.cheap: every engine failed after 3 attempts (last: DriverError: provider fake: error in the response body: All models exhausted)`.
  - `startup_check.py`: cp1252 and UTF-16 give `EngineConfigError (caught by _post_init)`, and the BOM file loads.
  - `risk9_check.py`: the digests' error lines now carry "after 3 attempts". Its last line still reads the removed per-row `errors` field, so it stops with `KeyError`. `test_exhausted_legacy_digests_count_as_calls_in_the_report_metadata` checks the new fields instead.
- **Static scan:** clean apart from the known `__file__` false positive. **Model-id check:** none outside the spec list.

### Fix round 2 — verification evidence (2026-10-19)

The fix round made no live calls. Logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t4fix2\`. The re-review's repro is `scratchpad\t4-rereview\probe_crash.py`.

- **RED** (`red_fix2.log`): the round-2 test files run against `7d9bd04` (extracted with `git archive`) → `7 failed, 42 passed`.
  - Each probe case escaped with `argus.engines.llm_openai.DriverError`, one of them carrying "bad key sk-leaky-123" in the traceback.
  - The stat cases escaped with `PermissionError: [Errno 13] Access is denied`.
  - The decode message lacked "is not UTF-8".
  - The 401 case passed already: SDK status errors were caught and redacted before.
- **GREEN, focused** (`green_fix2.log`): `test_engines_registry.py`, `test_engines_cmd.py`, `test_llm_facade.py` and `test_llm_retry.py` → `91 passed`.
- **Full suite, proxy URL on a closed port** (`suite_fix2.log`): `2 failed, 653 passed, 1 skipped, 1 warning, 2 errors`. Only the live tests failed, as by design with no proxy: `test_llm.py` ×2 and the `test_graph.py` fixture ×2.
- **`probe_crash.py` after the fix** (`probe_crash_after.log`): all four shapes give "probe survived", each with a FAIL row first. The echoed key prints as `bad key ***`.
