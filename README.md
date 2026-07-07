# Argus

Multi-agent Telegram research bot. Its brain is your local **FreeLLMAPI**
proxy (OpenAI-compatible, ~99 free-tier models behind one Bearer token).
Argus plans → fetches primary sources → normalizes → ranks → synthesises
→ adversarially reviews → renders an md + PDF report → delivers it to
Telegram, with two human-in-the-loop gates and a bounded reflexion loop.

## Quick start

```bash
# 1) secrets
cp .env.example .env
# Edit .env: paste your FREELLMAPI_API_KEY and TELEGRAM_BOT_TOKEN
#   (plus TELEGRAM_ALLOWED_USER_ID = your numeric Telegram id).

# 2) install
cd A:\Hermes\Agents\argus
uv venv --python 3.12 venv
.\\venv\\Scripts\\python.exe -m pip install -e .

# 3) start the bot
./scripts/run.sh
```

Then in Telegram: `/research <topic>` (deep) or `/ask <question>` (quick).
Per-chat memory is checkpointed in SQLite; reports are persisted to
`A:\Hermes\Downloads\reports\<stamp>_<topic>\`.

## ⚠ PYTHONPATH gotcha (read this or nothing will start)

The parent Hermes venv exports a `PYTHONPATH` that prepends its own
`site-packages` (different `pydantic_core` ABI). The argus venv must run
with a **clean** `PYTHONPATH`, otherwise pydantic_core fails to import
and the whole bot explodes on startup.

`scripts/run.sh`, `scripts/run_tests.sh`, and `scripts/run_demo.sh` all
start with `export PYTHONPATH=""`. Use them. If you bypass the scripts,
do `set PYTHONPATH=` (cmd) or `unset PYTHONPATH` (bash) first.

## Architecture

```
              ┌────────────── Telegram chat (single user) ──────────────┐
              │  /research <topic>   /ask <q>   /status   /cancel       │
              └──────────────────────────┬──────────────────────────────┘
                                         ▼
                       ┌──────────────────────────────┐
                       │  python-telegram-bot v22 (async)│
                       │  long-polling, HITL keyboards  │
                       └──────────┬────────────────────┘
                                  ▼
                ┌─────────────────────────────────────┐
                │       LangGraph supervisor          │
                │  SqliteSaver (thread_id = chat_id)  │
                └─┬─────┬─────┬─────┬─────┬─────┬────┬─┘
                  ▼     ▼     ▼     ▼     ▼     ▼    ▼
               intake planner res  fetch norm  rank synth rev─► report ─► deliver
                                 │                ▲    │       builder   ▲
                                 │                │    │                │
                                 │  ┌─────────────┘    │                │
                                 │  │                  │                │
                                 ▼  ▼                  ▼                ▼
                         ┌─────────────────┐   ┌──────────────┐  Telegram send
                         │ intel-stack:    │   │  FreeLLMAPI  │  (md + PDF)
                         │ harvest / snatch│   │  /v1/chat    │
                         │ crawl / convert │   │  /v1/models  │
                         └─────────────────┘   └──────────────┘
                                                       ▲
                              3-tier routing:          │
                              cheap  → llama-3.1-8b    │
                              strong → qwen3-coder     │
                              judge  → gpt-oss-120b    │
                              (fallback to "auto")     │
```

### State

`ArgusState` (TypedDict, `src/argus/graph/state.py`) carries:

- `thread_id`, `user_id`, `user_request`, `mode`
- `messages` — append-only chat history (reducer)
- `plan` — `ResearchPlan.model_dump()`
- `sources` — candidate URLs from researcher
- `fetched` — `FetchedItem` list with markdown_path + excerpt
- `findings` — `Finding` list, each with `claim + citation_urls + confidence`
- `draft_md`, `review_verdict`, `revision_notes`, `revision_rounds`
- `model_calls` — append-only LLM call log (operator.add reducer)
- `hitl` — pending gate descriptor
- `report_paths`, `quick_answer`

### Nodes (one per task, no black-box prebuilt agent)

| Node | Job | LLM tier |
|---|---|---|
| `intake` | classify quick vs deep, refine the request | cheap |
| `planner` | draft sub-questions + planned_sources (primary only) | strong |
| **HITL** | Telegram inline keyboard — Approve / Edit / Cancel | — |
| `researcher` | harvest + arXiv + planner target_urls | — |
| `fetcher` | snatch / crawl / normalize per source kind | — |
| `normalizer` | re-confirm markdown_path on disk | — |
| `filter` | score by plan keywords, drop low-signal | — |
| `synthesizer` | LCEL with_structured_output(Pydantic) → cited findings | strong |
| `reviewer` | adversarial fresh-context check, different family | judge |
| **Reflexion** | `revise` → back to synthesizer with notes (≤3 rounds) | — |
| `report_builder` | assemble md, render PDF (ReportLab) | — |
| **HITL** | Telegram inline keyboard — Send / Revise / Cancel | — |
| `deliver` | persist metadata, hand off to Telegram send | — |

### Model-tier routing (on top of the proxy's own fallback)

`src/argus/llm.py`:

- `fetch_live_models()` — calls `GET /v1/models` once, caches the ID list.
- `resolve_tier(tier)` — picks the first present PREFERRED id per tier;
  falls back to `"auto"` if every preferred model disappeared.
- `pick_strong_and_judge()` — guarantees the judge is from a different
  model family than the synthesizer (avoids same-family judge bias).
- `chat_for_tier(tier)` — returns a `ChatOpenAI` pointed at FreeLLMAPI.
- `record_from_response(tier, requested, response)` — captures the
  actual served model + provider into a `CallRecord` for telemetry.

Every LLM call is logged to `state["model_calls"]`; the demo transcript
prints the requested vs served model for every call.

## Architecture decisions

- **State checkpointer**: `SqliteSaver` keyed on `thread_id = tg:<chat_id>`.
  Each user's in-flight run survives a bot restart.
- **HITL via LangGraph `interrupt_before`**: graph pauses before
  `researcher` (plan approval) and before `deliver` (report preview).
  The Telegram layer drives the resume with `Command(resume=...)`.
- **Reflexion loop**: conditional edge from `reviewer` back to
  `synthesizer` on `verdict == "revise"`, capped at
  `ARGUS_MAX_REVISIONS` (default 3). `revision_notes` accumulate so
  the next pass sees the full feedback trail.
- **PDF rendering**: ReportLab Platypus in the argus venv (no browser
  needed, deterministic, fast). The intel-stack Chromium path is a
  last-resort fallback because headless Chromium IO is flaky on
  Windows when the pagefile is tight.
- **No prebuilt agent**: every node is a small function with an
  obvious shape — the contract is "explicit node/subgraph, no black box".

## Project layout

```
argus/
├── README.md               (this file)
├── pyproject.toml
├── .env.example            (NEVER commit .env)
├── src/argus/
│   ├── config.py           (Settings, .env loader, env validation)
│   ├── llm.py              (FreeLLMAPI tier routing, CallRecord)
│   ├── tools.py            (intel-stack subprocess wrappers + PDF render)
│   ├── bot.py              (python-telegram-bot v22 async commands)
│   └── graph/
│       ├── state.py        (TypedDict with append-only reducers)
│       ├── nodes.py        (10 explicit node functions)
│       └── graph.py        (build_graph + quick_answer_graph)
├── scripts/
│   ├── run.sh              (clears PYTHONPATH, starts the bot)
│   ├── run_tests.sh        (clears PYTHONPATH, runs pytest)
│   ├── run_demo.sh         (clears PYTHONPATH, runs demo_run.py)
│   └── demo_run.py         (acceptance test: full deep loop end-to-end)
├── tests/
│   ├── test_config_tools.py
│   ├── test_llm.py         (live FreeLLMAPI contract tests)
│   ├── test_graph.py       (in-memory deep + quick graph)
│   ├── test_bot.py         (formatting + keyboard wiring)
│   └── test_reflexion.py   (reviewer→revise→synth→pass loop, scripted)
├── demo_output/            (created at runtime; reports + transcript)
└── .langgraph_checkpoints/ (SqliteSaver directory)
```

## Tests

```bash
./scripts/run_tests.sh
# 18 passed in ~140s — most of the time is the live LLM tests
# against FreeLLMAPI (intake, synthesizer, reviewer real calls).
```

Coverage:
- `test_llm.py` — live: `/v1/models` resolves, each tier resolves,
  strong+judge come from different families, chat smoke.
- `test_graph.py` — in-memory: deep graph pauses at the right
  interrupt, quick graph runs end-to-end.
- `test_bot.py` — keyboard labels, plan formatter, token-required raise.
- `test_reflexion.py` — scripted reflexion loop: uncited claim →
  revise → cited claim → pass.
- `test_config_tools.py` — radar.md parser unit, settings loader,
  unreachable-host handling.

## Demo

```bash
./scripts/run_demo.sh
```

Runs the full deep loop against the live FreeLLMAPI proxy with a static
3-item corpus (HELM, simple-evals, Open LLM Leaderboard). Output:

- `demo_output/<stamp>_<topic>/report.md` — markdown with cited findings.
- `demo_output/<stamp>_<topic>/report.pdf` — same content, ReportLab.
- `demo_output/<stamp>_<topic>/metadata.json` — model_calls telemetry.
- `demo_output/demo_transcript.json` — full run summary.

A real run produces something like:

```
findings: 5
fetched:  9
rounds:   0
verdict:  pass
md:       A:\Hermes\Agents\argus\demo_output\20260707_195348_...\report.md
pdf:      A:\Hermes\Agents\argus\demo_output\20260707_195348_...\report.pdf
calls:    4
  - cheap  -> req=cheap  served=llama-3.1-8b-instant
  - strong -> req=qwen/qwen3-coder:free served=meta-llama/llama-4-scout-17b-16e-instruct
  - strong -> req=qwen/qwen3-coder:free served=meta-llama/llama-4-scout-17b-16e-instruct
  - judge  -> req=openai/gpt-oss-120b:free served=meta-llama/llama-4-scout-17b-16e-instruct
```

Note: the requested model is what argus asked for; the **served** model
is what FreeLLMAPI's own router actually used (its built-in fallback
chose Llama 4 Scout for every strong/judge call here — that's the
proxy's primary-purpose fallback in action).

## Operational notes

- **Memory budget**: 16 GB box. The bot is a few hundred MB; don't co-run
  the bot with Docker + both Hermes gateways + the local RAG endpoints
  + heavy crawls. See the troubleshooting-cookbook "commit RAM" class.
- **Rate limits**: FreeLLMAPI's default is 120 req/min per IP. The
  FreeLLMAPI dashboard also tracks per-key caps. Argus does not add
  its own rate limiting — lean on the proxy's own throttling.
- **Crawl concurrency**: the fetcher iterates sources serially. With
  3-10 sources per run, total crawl time is bounded. The intel-stack
  `crawl.py` itself is the bottleneck (playwright-based).
- **Source provenance**: every fetched item carries its `markdown_path`
  on disk. The metadata.json in each report folder lists every URL the
  synthesizer/reviewer actually saw.
- **No fabrication guarantee**: the reviewer enforces a citation per
  claim. If it cannot pass, the report is still delivered (with the
  revision notes appended) — the user sees the open questions in the
  report body.

## Secret handling

- `FREELLMAPI_API_KEY` — read from `FREELLMAPI_API_KEY` env var only.
  Never logged, never hardcoded, never committed (`.env` is gitignored).
- `TELEGRAM_BOT_TOKEN` — same.
- `TELEGRAM_ALLOWED_USER_ID` — gates the bot to a single Telegram user.
  Without this set, the bot refuses to start.

## Where the demo outputs live

The demo runs are written to `argus/demo_output/` (inside the project,
not `A:\Hermes\Downloads\reports\`). To change where real Telegram
reports land, set `ARGUS_REPORTS_ROOT` in `.env`.
