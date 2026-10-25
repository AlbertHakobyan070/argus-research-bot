# v4 notes — index and cross-cutting reference

One note per work package of the v4 rebuild (`docs/argus-v4-design.md` §3).
Each note is the record of what its task built: what changed, the decisions
where the spec is silent and the orchestrator's rulings, behaviour changes,
env vars, test files touched and verification evidence. A later task that
made a statement untrue marks it inline (*"WP-H: …"*), so every note still
describes the code. The operator summary of all behaviour changes is in
`handoff.md` ("What v4 changed"); the user guide is `docs/help.md`.

| Note | Task | Covers |
|---|---|---|
| `wp-0.md` | 1–3 | contracts (roles, engines schema, RunConfig, assembly), test helpers; dead code and hidden fallbacks removed; `bot.py` split into the `bot/` package |
| `wp-a.md` | 4 | engines registry, the openai-SDK driver, the `llm.complete` facade, `argus_engines.json`, `/engines`, `engines_probe.py` |
| `wp-d.md` | 5 | RunConfig wiring and the legacy-checkpoint injection, state additions, library migration, new `Settings` knobs |
| `wp-g.md` | 6 | bench v2 (deterministic metrics, compare), `scripts/live_check.py`, the v3 baseline tables |
| `wp-b.md` | 7 | URLs, the spend ledger, the fetch chain and its SSRF guard, search waves, rerank, chunked digest, verbatim quote checks, gap and Extend passes, recency, the between-item drain |
| `wp-i.md` | 8 | archetypes, the outline (fresh and extend), the analyst, confidence labels |
| `wp-c.md` | 9 | compose assembly, batched judging, the panel policy, cut-not-flag and the twin rule, the report builder, `evidence.json` |
| `wp-e1.md` | 10 | the run driver (one live drive, cancel by drain, attach on action, shutdown), the progress card, plan knobs, the quality line |
| `wp-e2.md` | 11–12 | the Mini App console: backend (auth, API, render, tunnel) and frontend |
| `wp-f.md` | 13 | transcriber engines |
| `wp-h.md` | 14 | hardening and docs: the wrappers removed, old snapshots kept loadable, the run-script pre-flight, these docs |

## Where the reference material lives

- **API routes** of the console: `wp-e2.md` ("`api.py`: the §2.7 routes"),
  against spec §2.7.
- **The resting-run route table** (which `as_node` keeps `next` for each
  resting state): `wp-e1.md`, "Resting-run writes keep `next`".
- **Security notes:**
  - the fetch chain's SSRF guard, the pinned connection, proxies, the decoded
    byte cap and own-endpoint checks: `wp-b.md` (the fetch chain, fix rounds
    1–4, "Known limits after the fix rounds");
  - console auth (`initData` HMAC, the single allowed user, refusals logged by
    field names only), the HTML sanitiser and the CSP: `wp-e2.md`;
  - keys never logged or recorded (`***` redaction in records, logs and the
    probe): `wp-a.md`;
  - the run-script pre-flight that stops only this repo's cloudflared:
    `wp-h.md`.
- **Documented ceilings and known limits:**
  - `wp-b.md`: "Known limits after the fix rounds" (proxies, raw deflate,
    links containing `]`, degenerate documents),
    the live-check "Known ceilings" (abandoned fetch threads, a late Exa page),
    and the `ponytail:` second-level-domain list;
  - `wp-c.md`: "The twin rule: live evidence and known limits";
  - `wp-e1.md`: the `_launch` ceiling at shutdown, and what a forced
    `task.cancel()` cannot stop;
  - `wp-h.md`: the pre-flight's ceilings.

## Environment variables (merged)

New or changed in v4 (names; defaults live in `src/argus/config.py`,
`argus_engines.json` or spec §7, never here):

| Variable | Read by | Introduced in |
|---|---|---|
| `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | the registry, when the active profile binds that provider as a primary; `GROQ_API_KEY` also the shipped `openai_audio` transcriber | `wp-a.md`, `wp-f.md` |
| `ARGUS_ENGINES_FILE`, `ARGUS_ENGINES_LOCAL_FILE` | the registry only (`default_paths()`) | `wp-a.md` |
| `ARGUS_WEBAPP_ENABLED`, `ARGUS_WEBAPP_PORT` | `Settings` → the console | `wp-d.md`, `wp-e2.md` |
| `ARGUS_WEBAPP_TUNNEL`, `ARGUS_CLOUDFLARED`, `ARGUS_WEBAPP_PUBLIC_URL`, `ARGUS_WEBAPP_INITDATA_MAX_AGE_S` | the console (`webapp/`); `ARGUS_CLOUDFLARED` also the run-script pre-flight; `ARGUS_WEBAPP_PUBLIC_URL` also the SSRF guard | `wp-e2.md`, `wp-h.md` |
| `ARGUS_CANCEL_DRAIN_TIMEOUT_S`, `ARGUS_SHUTDOWN_DRAIN_S` | `Settings` → the run driver | `wp-d.md`, `wp-e1.md` |
| `FREELLMAPI_BASE_URL` | the `freellmapi` provider (`base_url_env`), the SSRF guard, `verify_live.sh` | existing |
| `ARGUS_REPORTS_ROOT`, `ARGUS_BENCH_LOG` | set or read by the bench runner and the live check | `wp-g.md` |
| `ARGUS_LOG_HTTP` | the bot's logging setup (`_configure_logging()` in `bot/app.py`) | `wp-h.md` |

Removed, each with one startup WARNING if still set (the list is
`REMOVED_ENV_KNOBS` in `src/argus/engines/registry.py`): `ARGUS_TIMEOUT`,
`ARGUS_LLM_RETRIES`, `PERPLEXITY_API_KEY` (`wp-a.md`, `wp-0.md`);
`ARGUS_RESEARCH_WAVES`, `ARGUS_FETCH_PER_WAVE`, `ARGUS_TOTAL_FETCH_CAP`,
`ARGUS_EXA_MAX_CALLS`, `ARGUS_FETCH_CONCURRENCY`, `ARGUS_DIGEST_CONCURRENCY`
(`wp-b.md`); `ARGUS_MAX_REVISIONS`, `ARGUS_COMPOSE_CONCURRENCY` (`wp-c.md`);
`ARGUS_WHISPER_MODEL` (`wp-f.md`).

## Superseded by WP-H

- The Phase 1–2 wrappers (`chat_for_tier`, `invoke_with_retry`,
  `record_from_response`, `resolve_tier`, `pick_strong_and_judge`),
  `LEGACY_ROLES` and the `legacy.*` bindings, `current_snapshot_cached()`
  (now `cached_snapshot()`, which returns `None` instead of raising) and the
  `run_query_wave` alias are gone. Stored records keep their `legacy.<tier>`
  role names (v3 records are counted under them), and a stored snapshot that
  binds those roles loads with them dropped.
- `Settings.engines_file` / `engines_local_file` are gone (no reader).
- `tests/test_e2e_research.py` is gone; `.env.mac.example` is folded into
  `.env.example`; `langchain-openai` and `tenacity` left `pyproject.toml`.
