# Argus

Multi-agent Telegram research bot. Its brain is your local **FreeLLMAPI**
proxy (OpenAI-compatible, free-tier models behind one Bearer token).
The v3 research engine: brief → scout (live multi-query discovery over
Exa/DDGS/arXiv/GitHub) → deep research (every fetched source is READ and
digested into cited evidence notes, with gap-driven follow-up waves) →
outline → parallel section writers → tripartite review panel
(grounding/coverage/precision judges) → md + PDF report → Telegram,
with two human-in-the-loop gates and bounded revision loops.

Optional: set `EXA_API_KEY` in `.env` to enable Exa neural search
(search + full page text in one call; auto-falls back to DDGS without
it, and beyond `ARGUS_EXA_MAX_CALLS` per run).

## Quick start

```bash
# 1) secrets
cp .env.example .env
# Edit .env: paste your FREELLMAPI_API_KEY and TELEGRAM_BOT_TOKEN
#   (plus TELEGRAM_ALLOWED_USER_ID = your numeric Telegram id).

# 2) install (from the repo root)
uv venv --python 3.12 venv
./venv/Scripts/python.exe -m pip install -e .   # Windows venv layout

# 3) start the bot
./scripts/run.sh
```

Then in Telegram: `/research <topic>` (deep) or `/ask <question>` (quick).
Every run gets its own SQLite checkpoint thread (`tg:<chat>:<run8>`), is
registered in the Argus library DB (`argus_library.sqlite`), and reports
are persisted to the DS-vault research-history folder (override with
`ARGUS_REPORTS_ROOT` / `ARGUS_VAULT_ROOT`).

## ⚠ PYTHONPATH gotcha (read this or nothing will start)

The parent Hermes venv exports a `PYTHONPATH` that prepends its own
`site-packages` (different `pydantic_core` ABI). The argus venv must run
with a **clean** `PYTHONPATH`, otherwise pydantic_core fails to import
and the whole bot explodes on startup.

`scripts/run.sh`, `scripts/run_tests.sh`, and `scripts/run_demo.sh` all
start with `export PYTHONPATH=""`. Use them. If you bypass the scripts,
do `set PYTHONPATH=` (cmd) or `unset PYTHONPATH` (bash) first.

## Architecture (v3 research engine)

```
              ┌────────────── Telegram chat (single user) ──────────────┐
              │  /research <topic>   /ask <q>   /status   /cancel        │
              └──────────────────────────┬───────────────────────────────┘
                                         ▼
                       ┌────────────────────────────────┐
                       │  python-telegram-bot (async)    │
                       │  long-polling, HITL keyboards   │
                       └──────────┬─────────────────────┘
                                  ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                    LangGraph engine                          │
        │  AsyncSqliteSaver (one per process; thread_id=tg:<chat>:<run8>) │
        └──────────────────────────────────────────────────────────────┘

   intake ─► brief ─► scout ══╣ PLAN GATE ╠══► research ─► outline ─►
   compose ─► panel ─┬─(revise flagged sections)─► compose
                     └─(pass / budget)─► report_builder ══╣ REPORT GATE ╠══►
   deliver ─┬─ end
            ├─ extend ─► extend_prep ─► research   (more sources, new wave)
            └─ revise ─► revise_prep ─► compose    (same evidence + notes)

   Quick path (/ask):  intake ─► quick_answer ─► deliver

   ┌── search providers ──┐   ┌── page fetch ──┐   ┌── FreeLLMAPI ──┐
   │ Exa / DDGS / arXiv / │   │ intel-stack:   │   │  /v1/chat      │
   │ GitHub  (parallel    │   │ snatch / crawl │   │  /v1/models    │
   │ query waves)         │   │ / convert      │   │ 3-tier routing │
   └──────────────────────┘   └────────────────┘   └────────────────┘
```

The two `╣ GATE ╠` bars are LangGraph interrupts: the run pauses after
`scout` (plan approval, with the real live-search sources shown) and
before `deliver` (report preview). The Telegram layer drives each resume
with `Command(resume=…)`.

### What each stage does

| Stage | Job | LLM tier |
|---|---|---|
| `intake` | classify quick vs deep, refine the request | cheap |
| `brief` | draft sub-questions + success criteria (no URLs — hallucinated links are structurally impossible); deterministic quality checks + one bounded re-draft | strong |
| `scout` | one live discovery wave: 1–2 targeted queries **per sub-question** across Exa/DDGS/arXiv/GitHub, in parallel; search-API only, no page fetch | cheap (query-gen) |
| **PLAN GATE** | Telegram keyboard — Approve / Edit / Cancel — over the **real found sources** | — |
| `research` | the deep phase: waves of triage → parallel fetch → **digest** (a cheap LLM *reads each fetched document* into evidence notes: claims + quotes + relevance + stance, keyed to sub-questions) → coverage check → gap-targeted follow-up queries | cheap ×N |
| `outline` | plan the report sections against the gathered evidence | strong |
| `compose` | parallel per-section writers, each seeing only its own evidence notes; output is **plain markdown, never JSON**; findings extracted per section with citation ids | strong ×sections |
| `panel` | three judges (grounding / coverage / precision), family-diverse where the proxy allows; deterministic merge → section-targeted revision | judge + strong + cheap |
| **revise loop** | `panel` → `compose`, rewriting only the flagged sections, capped at `ARGUS_MAX_REVISIONS` (default 3) | — |
| `report_builder` | assemble md, citation verification + quarantine, render PDF (ReportLab) | — |
| **REPORT GATE** | Telegram keyboard — Send / Extend / Revise / Cancel | — |
| `deliver` | persist metadata, hand off to Telegram send; routes end / extend / revise | — |

### Evidence-first design (why v3 replaced the linear pipeline)

The old pipeline (`intake→planner→researcher→fetch→norm→rank→synth→rev`)
never let the writer read the evidence: the synthesizer saw only a
400-char excerpt per source while the full fetched markdown sat unread
on disk. v3's **digest** step reads each document with a cheap-tier LLM
into structured `EvidenceNote`s (claims with supporting quotes, a 0–5
relevance score, and a stance), keyed to the sub-questions the source
was gathered for. The section writers compose from those notes with
`[n]` source ids — so depth comes from evidence the model actually read,
and long-form output is plain markdown rather than the markdown-inside-
JSON that used to break on weaker proxy-routed models. See
[`docs/research-engine-v3.md`](docs/research-engine-v3.md) for the full
design and its sources.

### State

`ArgusState` (TypedDict, `src/argus/graph/state.py`) carries the run.
The load-bearing channels:

- `thread_id`, `user_id`, `user_request`, `mode`, `length`
- `messages`, `model_calls`, `errors` — append-only (reducers)
- `brief` — `ResearchBrief` (sub-questions + success criteria)
- `plan` — a v2-compatible dict the Telegram plan-gate renderer reads
- `queries` — the executed search queries (scout + follow-up waves)
- `sources` — candidate hits from the search waves
- `fetched` — `FetchedItem` list (markdown_path + excerpt on disk)
- `evidence` — `EvidenceNote` list (the digested, cited claims)
- `coverage` — per-sub-question strength count
- `outline`, `sections` — the section plan and the composed markdown
- `findings` — `Finding` list (`claim + citation_urls + confidence`)
- `draft_md`, `panel_verdict` / `review_verdict`, `revision_notes`
- `hitl` — pending gate descriptor; `report_paths`, `quick_answer`

### Search providers

`src/argus/graph/search_providers.py` gives every provider one uniform
`SearchHit` shape and runs a wave of queries in parallel, isolating
failures (one provider's outage never poisons the others):

- **Exa** — optional, enabled iff `EXA_API_KEY` is set. Returns search
  hits **with full page text in one call**, so those hits skip the fetch
  stage entirely. The free plan is ~1000 requests/month, so Exa calls
  per run are capped (`ARGUS_EXA_MAX_CALLS`, default 6) and DDGS takes
  the overflow.
- **DDGS** (keyless web), **arXiv** (Atom API), **GitHub** (repo search).

### Model-tier routing

`src/argus/llm.py` maps three tiers — `cheap` / `strong` / `judge` —
onto whatever the proxy currently serves:

- `fetch_live_models()` — calls `GET /v1/models` once, caches the ID list.
- `resolve_tier(tier)` — picks the first present preferred id per tier;
  falls back to `"auto"` (the proxy's own router) if none remain.
- `pick_strong_and_judge()` — keeps the judge in a different model family
  than the strong writer (avoids same-family judge bias).
- `chat_for_tier(tier)` / `record_from_response(...)` — build the
  `ChatOpenAI` client and capture the actually-served model + provider
  into a `CallRecord`.

Every LLM call is logged to `state["model_calls"]` (requested vs served),
so a run's `metadata.json` shows exactly which models answered.

## Architecture decisions

- **State checkpointer**: one long-lived `AsyncSqliteSaver` opened in PTB
  `post_init`, shared by every run. Threads are per-run
  (`thread_id = tg:<chat>:<run8>`), so runs are individually resumable
  and survive bot restarts; the run registry lives in
  `argus_library.sqlite` (`src/argus/library.py`).
- **Grounded plan gate**: the graph pauses *after* `scout`
  (`interrupt_after=["scout"]`), so the plan preview shows the real
  sources live search found — the brief never contains URLs, so there is
  nothing for the model to hallucinate. A second interrupt sits before
  `deliver` (report preview).
- **Bounded waves + revision**: `research` runs at most
  `ARGUS_RESEARCH_WAVES` search/digest waves (default 2) up to
  `ARGUS_TOTAL_FETCH_CAP` sources; the `panel → compose` revision loop is
  capped at `ARGUS_MAX_REVISIONS` (default 3). Every LLM output is parsed
  through `graph/jsonx.py`, which repairs and salvages weak-model JSON.
- **Failure isolation over silent swallowing**: search providers, the
  fetch path, and the digest step append to `state["errors"]` (via the
  `Annotated[list[str], operator.add]` reducer) instead of dropping
  exceptions. When every source fails to fetch, `research` appends an
  explicit summary error so the writer never runs on empty evidence.
- **PDF rendering**: ReportLab Platypus in the argus venv (no browser,
  deterministic, fast); the intel-stack Chromium path is a last-resort
  fallback (headless Chromium IO is flaky on Windows under memory
  pressure).
- **Tool interpreter routing**: `tools._run_script` defaults to
  `INTEL_PYTHON_BIN` (the intel-stack venv), which has
  `feedparser` / `crawl4ai` / `markitdown` / `yt_dlp`; the argus venv
  does not. Override with `python_bin=...` for a tool that needs a
  different interpreter.
- **No prebuilt agent**: every node is a small, explicit function — the
  contract is "explicit node/subgraph, no black box".

## Project layout

```
argus/
├── README.md               (this file)
├── pyproject.toml
├── .env.example            (NEVER commit .env)
├── docs/
│   ├── help.md             (the /help full command guide)
│   └── research-engine-v3.md (v3 engine design + sources)
├── src/argus/
│   ├── config.py           (Settings, .env loader, env validation)
│   ├── llm.py              (FreeLLMAPI tier routing, CallRecord)
│   ├── tools.py            (intel-stack subprocess wrappers + PDF render)
│   ├── bot.py              (python-telegram-bot async commands + HITL)
│   ├── library.py          (run + asset registry, aiosqlite)
│   └── graph/
│       ├── state.py            (TypedDict + Pydantic models)
│       ├── jsonx.py            (robust weak-model JSON parsers)
│       ├── search_providers.py (Exa / DDGS / arXiv / GitHub waves)
│       ├── brief.py            (scoping node)
│       ├── scout.py            (live discovery wave → plan gate)
│       ├── research.py         (triage → fetch → digest → coverage)
│       ├── compose.py          (outline + parallel section writers)
│       ├── panel.py            (tripartite review panel)
│       ├── nodes.py            (intake / report_builder / deliver / preps)
│       ├── credibility.py      (domain-trust scoring)
│       ├── grounding.py        (entity-mention grounding check)
│       └── graph.py            (build_graph + quick_answer_graph)
├── scripts/
│   ├── run.sh              (clears PYTHONPATH, starts the bot)
│   ├── run_tests.sh        (clears PYTHONPATH, runs pytest)
│   ├── run_demo.sh         (clears PYTHONPATH, runs demo_run.py)
│   ├── demo_run.py         (self-contained v3 end-to-end demo)
│   └── verify_live.sh      (live-proxy verification pass)
├── tests/                  (hermetic suite + live-proxy tests)
├── argus_checkpoints.sqlite (LangGraph checkpoints, one thread per run)
└── argus_library.sqlite     (run + asset registry; see src/argus/library.py)
```

## Tests

```bash
./scripts/run_tests.sh          # hermetic subset (what CI runs)
# Full suite incl. live-proxy tests (needs FreeLLMAPI up):
PYTHONPATH="" ./venv/Scripts/python.exe -m pytest tests/ --ignore=tests/manual_e2e.py -q
```

The hermetic subset (`.github/workflows/ci.yml`) runs everything that
doesn't need the live proxy. Notable files: `test_brief.py` (scoping +
no-URL guarantee), `test_search_providers.py` (wave merge/dedupe, Exa
budget, error isolation), `test_reflexion.py` (the full v3 graph with a
scripted panel revise→pass loop), `test_grounded_gate.py` (the plan gate
pauses after scout with real sources), `test_continue_extend.py`
(`/append` + `/continue` mechanics). `test_llm.py` and `test_graph.py`
are the live-proxy tests (tier resolution; a network-deterministic v3
drive where the LLM reads a controlled source and must still produce a
grounded, cited report).

**Verification rule:** green tests ≠ a working bot. For research changes,
drive the real graph end-to-end — `scripts/verify_live.sh`, or
`tests/manual_e2e.py` for a graph-direct deep run whose report you read
on disk. The proxy re-routes models, so also confirm behaviour against a
weaker served model (check `served_model` in a run's `metadata.json`).

## Demo

```bash
./scripts/run_demo.sh
```

Drives the **full v3 graph** against the live FreeLLMAPI proxy. The
search wave and page fetches are stubbed to a small controlled corpus so
the run is network-deterministic, but the LLM does the real work — it
digests the corpus into evidence notes, composes a sectioned report, and
the panel reviews it. Output lands in `demo_output/`:

- `<stamp>_<topic>_<length>/report.md` — markdown with cited findings.
- `<stamp>_<topic>_<length>/report.pdf` — same content, ReportLab.
- `<stamp>_<topic>_<length>/metadata.json` — `model_calls` telemetry.
- `demo_transcript.json` — the run summary (findings, evidence, verdict).

The transcript prints, for every LLM call, the **requested** tier vs the
**served** model — FreeLLMAPI's own router picks the actual model, so a
`strong` request may be served by whatever the proxy currently fronts.

## Operational notes

- **Memory budget**: the bot is a few hundred MB, but a research run
  fans out parallel fetches and per-source digest calls. On a tight box,
  don't co-run it with Docker + both Hermes gateways + heavy crawls.
- **Rate limits**: FreeLLMAPI throttles per IP and per key. A v3 run
  makes many cheap-tier calls (digests parallelize), so lean on the
  proxy's own throttling — Argus adds none of its own. Cap search cost
  with `ARGUS_EXA_MAX_CALLS` / `ARGUS_TOTAL_FETCH_CAP`.
- **Concurrency**: fetches and digests run in bounded thread pools
  (`ARGUS_FETCH_CONCURRENCY` / `ARGUS_DIGEST_CONCURRENCY`); the
  intel-stack `crawl.py` (playwright) is the per-source bottleneck.
- **Source provenance**: every fetched item keeps its `markdown_path` on
  disk, and each report folder's `metadata.json` lists the models and
  sources the run actually used.
- **Grounding, not a fabrication guarantee**: the panel enforces that
  claims trace to evidence, and the report_builder verifies every
  citation against the fetched sources. When the panel can't pass, the
  report still ships with its open questions surfaced in the body.

## Secret handling

- `FREELLMAPI_API_KEY` — read from `FREELLMAPI_API_KEY` env var only.
  Never logged, never hardcoded, never committed (`.env` is gitignored).
- `TELEGRAM_BOT_TOKEN` — same.
- `TELEGRAM_ALLOWED_USER_ID` — gates the bot to a single Telegram user.
  Without this set, the bot refuses to start.

## Where outputs live

Demo runs are written to `demo_output/` inside the project. Real Telegram
reports default to the vault research-history folder; override the
location with `ARGUS_REPORTS_ROOT` (and the vault root with
`ARGUS_VAULT_ROOT`) in `.env`. The run + asset registry and the LangGraph
checkpoints stay in the project dir (`argus_library.sqlite` /
`argus_checkpoints.sqlite`), never in a synced vault.
