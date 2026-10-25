# How it works

## Architecture

```
   Telegram chat (single user)            Mini App console (menu button)
              │                                        │  aiohttp on 127.0.0.1,
              ▼                                        ▼  cloudflared quick tunnel
   python-telegram-bot (async) ──► run driver (bot/runs.py) ◄── webapp/api.py
                                     one live drive per run · cancel by drain
                                     · event bus → progress card / console
                                          │
                                          ▼
   intake → brief → scout ══╣ PLAN GATE ╠══ research → outline → compose → analyst → panel
   panel → research         (a coverage gap: a targeted, bounded gap wave, then outline in extend mode)
   panel → compose          (a targeted, bounded rewrite of the weak sections)
   panel → report_builder ══╣ REPORT GATE ╠══ deliver → end
                                                     | Extend → extend_prep → research
                                                     | Revise → revise_prep → compose
   quick path (/ask):  intake → quick_answer → deliver
```

The two gates are LangGraph interrupts: after `scout` (plan approval over the
real sources live search found) and before `deliver` (report preview). The
run driver resumes them; it treats a gate as one only while no drive holds
the run.

### What each stage does

| Stage | Job |
|---|---|
| `intake` | classify quick vs deep, refine the request |
| `brief` | sub-questions, success criteria, archetype, time sensitivity, entities — no URLs, so hallucinated links are structurally impossible |
| `scout` | one live discovery wave per sub-question across the search providers; decides the recency window |
| **PLAN GATE** | Approve / Edit / Cancel over the real found sources, with length, depth, engines-profile and recency knobs |
| `research` | waves of rerank → fetch (an SSRF-guarded native fetcher, Exa text, optional intel-stack rungs) → digest (an LLM reads each document into claims with verbatim-checked quotes) under one spend ledger; gap waves and Extend passes are targeted and budgeted |
| `outline` | sections with allocated evidence notes, shaped by the archetype; an Extend keeps the titles |
| `compose` | parallel section writers over their own notes (plain markdown), then finding extraction per section |
| `analyst` | takeaways with confidence labels, disagreements and scenarios, from the findings |
| `panel` | batched grounding judge (every finding judged against its cited quotes), coverage review, deterministic precision scan; routes to a gap wave, a targeted rewrite or the report |
| `report_builder` | cuts rejected claims, compiles the TL;DR from kept findings, banners and degradation notes, Sources, PDF, `metadata.json`, `evidence.json` |
| **REPORT GATE** | Send / Extend / Revise / Cancel |
| `deliver` | ends the run, or routes to Extend / Revise |

Design documents, on GitHub:
[the v4 spec](https://github.com/AlbertHakobyan070/argus-research-bot/blob/main/docs/argus-v4-design.md),
[the v4 notes](https://github.com/AlbertHakobyan070/argus-research-bot/tree/main/docs/v4-notes)
(what each work package built and why) and
[the v3 engine](https://github.com/AlbertHakobyan070/argus-research-bot/blob/main/docs/research-engine-v3.md)
that v4 builds on.

## Engines

- `engines/roles.py` — the pipeline roles and their failure-policy classes.
- `engines/schema.py` / `argus_engines.json` — providers (base URL, key env
  var name, capabilities), profiles (tier engines + per-role bindings with
  fallbacks and `on_exhausted` = `raise` / `skip_item` / `degrade`), the
  transcriber block, dated prices.
- `engines/registry.py` — merges the defaults file, the optional
  `argus_engines.local.json` (gitignored) and the library's `/engines`
  choices; validates the schema at startup (no network) and the active
  profile at run start; resolves it into the snapshot every run carries.
- `llm.py` — the one facade, `complete(role, …)`: retries, fallbacks, the
  declared policy, and a `CallRecord` per attempt (requested vs served model,
  routing) in `state["model_calls"]` and `metadata.json["engines"]`.
- A declared fallback is never silent: it is a `degrade:` / `skip:` line in
  the report's degradation notes and, past a threshold, a "⚠ Degraded run"
  banner.

## Run configuration

Each run carries a `RunConfig` (`graph/run_config.py`): length, depth preset
(`quick` / `standard` / `deep` / `exhaustive`), recency, the research /
search / fetch / compose / panel knobs, and the engines snapshot. The
presets are the table in `PRESETS`; the console accepts dotted overrides.
Pre-v4 checkpoints resume with their length's legacy preset, recorded in the
run's errors and notes.

## Project layout

```
argus/
├── README.md, handoff.md       (front page; operations and migration notes)
├── mkdocs.yml                  (this site)
├── pyproject.toml
├── argus_engines.json          (engine providers, profiles, transcriber, prices)
├── .env.example                (NEVER commit .env)
├── docs/
│   ├── index.md, how-it-works.md, stylesheets/   (site pages)
│   ├── help.md                 (the /help full command guide)
│   ├── argus-v4-design.md      (v4 build spec)
│   ├── v4-notes/               (one note per work package)
│   └── research-engine-v3.md   (the v3 engine design)
├── src/argus/
│   ├── config.py               (Settings, .env loader)
│   ├── llm.py                  (the LLM facade)
│   ├── engines/                (roles, schema, registry, driver, ledger,
│   │                            URLs, fetchers, search, transcribers)
│   ├── graph/                  (state, run_config, the nodes, assembly,
│   │                            evidence export, precision, quarantine)
│   ├── bot/                    (the Telegram package; run driver, progress)
│   ├── webapp/                 (Mini App: server, auth, API, render, tunnel, static/)
│   ├── library.py              (run + asset registry, aiosqlite)
│   ├── tools.py                (intel-stack wrappers, search, PDF and YouTube helpers)
│   ├── media.py, transcribe.py (media downloads, transcription into the vault)
│   └── quarantine.py, citations.py, sources_block.py, cache_cleanup.py
├── bench/                      (runner, deterministic metrics, compare, queries)
├── scripts/
│   ├── run.sh / run_mac.sh     (pre-flight, then the bot)
│   ├── run_tests.sh, run_demo.sh, demo_run.py
│   ├── live_check.py, verify_live.sh   (live end-to-end checks)
│   ├── engines_probe.py        (probe a provider's request shapes)
│   ├── webapp_dev_url.py, sync_arsenal.py  (console dev URL, kit snapshot)
│   └── reconcile_vault.py
├── tests/                      (hermetic suite + live-proxy tests)
├── tools/bin/                  (gitignored: cloudflared)
├── argus_checkpoints.sqlite    (LangGraph checkpoints, one thread per run)
└── argus_library.sqlite        (run + asset registry)
```

## Tests

```bash
./scripts/run_tests.sh -q       # the whole suite; the live-proxy tests need the proxy
# hermetic only, as CI runs it: see the --ignore list in .github/workflows/ci.yml
```

`tests/test_llm.py` and `tests/test_graph.py` are the live-proxy tests
(the shipped engines config resolved against the live proxy; a
network-deterministic graph drive where the LLM reads a controlled source).
Shared test helpers live in `tests/helpers.py`.

**Verification rule:** green tests ≠ a working bot. For research changes,
drive the real graph end-to-end: `scripts/verify_live.sh` (proxy check, the
live tests, then `scripts/live_check.py`, which prints an acceptance table and
exits non-zero on any failed check). The proxy re-routes models, so also
check `served_model` in a run's `metadata.json`. `python -m bench.metrics` and
`python -m bench.compare` measure runs deterministically.

## Demo

```bash
./scripts/run_demo.sh
```

Drives the **full graph** on the active engines profile. The search wave and
page fetches are stubbed to a small controlled corpus so the run is
network-deterministic, but the LLM does the real work — it digests the corpus
into evidence notes, composes and analyses a sectioned report, and the panel
judges it. Output lands in `demo_output/`: the report folder (`report.md`,
`report.pdf`, `metadata.json`, `evidence.json`) and `demo_transcript.json`.
The demo prints, for every LLM call, the role, the **requested** model and
the **served** one.

## Operational notes

- See `handoff.md` in the repository for running the bot, the console and
  tunnel, migration, and troubleshooting.
- **Rate limits**: the free proxy throttles per IP and per key, and a run makes
  many per-source calls. Only one run researches at a time; depth presets and
  the per-run Exa budget bound the spend, and `/engines` shows this month's Exa
  cost.
- **Cancel** drains: the run finishes its current step, keeps its checkpoint,
  and `/continue` resumes it.
- **Source provenance**: every fetched item keeps its `markdown_path` on disk;
  `metadata.json` and `evidence.json` list the models, sources, quotes and
  verdicts.
- **Trust, not a guarantee**: the judge only sees the notes' quotes; a claim
  it cannot locate stays in the body and is listed as such, and an unjudged
  finding marks the report "⚠ Not fact-checked".

## Secret handling

- API keys (`FREELLMAPI_API_KEY`, the paid providers' keys, `EXA_API_KEY`) and
  `TELEGRAM_BOT_TOKEN` are read from the environment / `.env` only.
  `argus_engines.json` holds env var **names**, never values. Never logged,
  never hardcoded, never committed (`.env` is gitignored).
- `TELEGRAM_ALLOWED_USER_ID` gates the bot (and the console's signed
  `initData`) to a single Telegram user; unset, every user is refused.

## Where outputs live

Demo runs are written to `demo_output/` inside the project. Real Telegram
reports default to the vault research-history folder; override the
location with `ARGUS_REPORTS_ROOT` (and the vault root with
`ARGUS_VAULT_ROOT`) in `.env`. The run + asset registry and the LangGraph
checkpoints stay in the project dir (`argus_library.sqlite` /
`argus_checkpoints.sqlite`), never in a synced vault. The console's tunnel log
is `logs/cloudflared.log`.
