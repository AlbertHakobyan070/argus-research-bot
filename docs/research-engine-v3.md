# Argus research engine v3 — design

> Replaces the v2 linear pipeline
> (`intake → planner → planner_reflect → researcher → fetcher → normalizer →
> credibility → filter → synthesizer → reviewer`) with a supervisor-style
> deep-research engine. 2026-07-15.

## Why v2 underdelivered

1. **The synthesizer never read the evidence.** It saw `excerpt[:400]` per
   source (~8 KB total for a whole report) while the full fetched markdown sat
   unread on disk. That was the depth ceiling — no prompt tuning could fix it.
2. **One search wave, one query per provider.** `" ".join(keywords[:3])` fed
   once to arXiv/GitHub/DDGS. No per-sub-question queries, no follow-up
   searches driven by gaps.
3. **Keyword-overlap ranking** on title+excerpt — no content-aware relevance.
4. **Monolithic JSON synthesis** with the whole report embedded in a JSON
   string (`draft_md`) — the exact weak-model fragility behind the
   2026-07-12 empty-report bug.
5. **Single reviewer**, one perspective, binary pass/revise, JSON-fragile.

## Sources for the design

- **LangChain `open_deep_research`** — three-phase scope/research/write;
  supervisor delegates research units that run in parallel and each ends
  with a *compression* step; explicit iteration budgets.
- **Anthropic multi-agent research system** — orchestrator/worker, effort
  scaling rules, parallel tool calls, dedicated citation pass at the end.
- **GPT-Researcher** — per-question source curation; per-source summaries
  *keyed to the question being asked*; 20+ sources for objectivity.
- **STORM** — outline first, then per-section writing from per-section
  references (never one giant generation).
- **NVIDIA AI-Q blueprint** — citation registry / verification (already in
  Argus v2, kept), report-rewriter as a separate loop (mirrors our
  revise gate).
- **Forward Future loop library** — claim-ledger pattern (evidence polarity +
  confidence per claim), multi-LLM convergence review (different model
  families must independently pass the work).
- **Albert's RAG proposal** — planner/researcher/writer/critic split with a
  tripartite judgment mechanism → realized as the 3-judge panel.

## Constraints that shaped it

- **The proxy reroutes to weak models.** Every structured output must be
  SMALL (no markdown-in-JSON). Long-form output is plain markdown. All JSON
  passes through the repairing parser (`jsonx.py`).
- **Bot contract is sacred**: two HITL gates (`interrupt_after` scout,
  `interrupt_before` deliver), per-run threads, AsyncSqliteSaver resume,
  `/continue`-extend-append-revise, run library, `plan`/`sources`/`fetched`/
  `findings`/`draft_md`/`report_paths` state keys.
- **Nodes stay sync** (LangGraph threads them under the bot's `astream`);
  internal parallelism via ThreadPoolExecutor / `asyncio.run` fan-out —
  the pattern already proven in v2.
- **Free tier**: cheap-tier calls may be numerous (they parallelize);
  strong-tier calls are budgeted per section; hard wave/iteration caps.

## The graph

```
intake ─► brief ─► scout ══╣ PLAN GATE (interrupt_after=["scout"]) ╠═►
research ─► outline ─► compose ─► panel ─┬─(revise: flagged sections)─► compose
                                         └─(pass / budget)─► report_builder
report_builder ══╣ REPORT GATE (interrupt_before=["deliver"]) ╠═► deliver
     deliver ─┬─ end
              ├─ extend ─► extend_prep ─► research   (more sources, new wave)
              └─ revise ─► revise_prep ─► compose    (same evidence + user notes)
```

Quick path (`/ask`) unchanged: `intake → quick_answer → deliver`.

### Node responsibilities

| Node | Tier | Output |
|---|---|---|
| `intake` | cheap | mode + refined query (unchanged from v2) |
| `brief` | strong | ResearchBrief: sub-questions (with kind hints), keywords, summary, success criteria. Deterministic quality checks + one bounded re-draft (absorbs v2 `planner_reflect`). Emits a v2-compatible `plan` dict so the Telegram plan gate renders unchanged. |
| `scout` | none | Discovery wave: multi-query search per sub-question across providers (Exa / DDGS / arXiv / GitHub), search-API only, no page fetching. Dedup + tag sources with sub-question ids. Feeds the grounded plan gate. |
| `research` | cheap ×N | The deep phase. Waves (≤2 + extend): triage (credibility prior + snippet relevance) → parallel fetch (Exa text short-circuit, else intel-stack snatch/crawl/stealth) → **digest**: per-source cheap-LLM read of the *actual fetched markdown* (chunk-capped) producing EvidenceNotes {claims+quote+confidence+relevance+stance} keyed to sub-questions → coverage check per sub-question → targeted follow-up queries if gaps and budget. |
| `outline` | strong | Small JSON: sections (title + which sub-questions + share of budget), driven by length mode. |
| `compose` | strong ×sections | Per-section writers in parallel. Input: only that section's EvidenceNotes (with [S#] source ids). Output: **plain markdown** (never JSON). Deterministic assembly + per-section findings extraction (small JSON per section, cheap tier) feeding the citation/quarantine machinery. |
| `panel` | judge+strong+cheap | Three parallel judges, different families where possible: grounding (claims↔evidence), coverage (brief satisfied?), precision (numbers/dates/names fabrication). Small JSON verdicts, deterministic merge → targeted section revision list. |
| `report_builder` | none | v2 machinery kept: title block, grounding banner, citation verification (registry), quarantine, sources-block sanitize, PDF, metadata.json. |
| `deliver`, `extend_prep`, `revise_prep` | none | Same contract as v2; extend rejoins at `research`, revise at `compose`. |

### State (additions; v2 keys preserved)

`brief`, `queries` (per-sub-question search queries), `evidence`
(EvidenceNote dicts), `coverage` (per-sub-question strength),
`outline`, `sections` (per-section markdown), `panel_verdict`
(merged; also written to `review_verdict` for bot compat),
`research_rounds`.

### Search providers

`search_providers.py` — uniform `SearchHit` interface over:
- **exa** (`EXA_API_KEY` present → enabled): neural search + full text in one
  call (hits with text skip the fetch stage entirely). Category mapping:
  papers → `research paper`, repos → github domain filter. Free plan ≈ 1000
  req/mo → budget ≤ ~8 Exa calls per run, DDGS takes the overflow.
- **ddgs** (v2 `ddgs_search`, kept), **arxiv** (Atom API, kept),
  **github** (repo search API, kept).

### Bot contact points changed

- `snap.next == ("fetcher",)` → `("research",)` (plan gate) — `/continue` +
  orphan reconcile.
- Progress-message node names.
- Everything else (keyboards, resume, callbacks, library) untouched.

## Budgets (medium run, ~5 sub-questions)

intake 1 + brief 1-2 + query-gen 1 + digests ~12-20 (cheap, parallel) +
follow-up 0-1 + outline 1 + sections 4-6 (strong, parallel) + findings 4-6
(cheap, parallel) + judges 3 ≈ **28-40 calls**, wall-clock dominated by
fetch+digest wave (parallel). v2 was ~8 calls but read 400 chars/source.
