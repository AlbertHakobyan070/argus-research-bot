# Argus — Handoff 2026-07-08 (AI-Q follow-ups + bug fixes)

**Date:** 2026-07-08 evening (~18:00 local, +04:00)
**Author:** coding-app (current me, post-AIQ-survey)
**Target reader:** cold-me, fresh session tomorrow morning

---

## 0. TL;DR — what happened today

Albert dropped the `understanding-aiq.html` report on NVIDIA AI-Q Blueprint
mid-session. I:

1. **Dispatched a subagent** to survey AI-Q's web-research layer at
   github.com/NVIDIA-AI-Blueprints/aiq (develop branch). Survey came back
   in 113s: the gold was `SourceRegistry + verify_citations +
   sanitize_report` in `src/aiq_agent/common/citation_verification.py`.
2. **Albert steered:** "implement these in Argus." Plan locked = A+B
   (immediate bug fixes + structural citation lift).
3. **Shipped Phase A (15 min)** — prompt hardening + Telegram HTML mode
   fix. Two bugs killed at the surface.
4. **Shipped Phase B (3 hours)** — citation registry lift + report-
   builder wiring + ddgs_search timelimit. Hallucinated URLs can no
   longer reach the final report.

Net: **143/143 tests passing** (was 96 yesterday, +47 today), 4 atomic
commits on `feat/argus-fixes`. **Bot not yet retested live.**

---

## 1. Branch state — EXACT

```
$ git branch --show-current
feat/argus-fixes

$ git log --oneline main..HEAD
e8b5135  feat(argus): ddgs_search timelimit (lifted from AI-Q news-search)
5484b13  feat(argus): wire citation-integrity pass into report_builder
e527cab  feat(argus): lift SourceRegistry + verify_citations + sanitize_report
a93af1b  fix(argus): two bot bugs surfaced during live AI-Q query (2026-07-08)
```

`main` is at the old state — **do NOT merge `feat/argus-fixes` into
`main` blindly**. The branch is meant to land via PR after Albert's
live retest passes.

`feat/argus-v1` (the earlier branch from yesterday) is **untouched**.
Don't cross-contaminate.

---

## 2. What is fixed (proven by tests, NOT yet proven by live bot)

### Phase A — bug fixes
- **Bug #1: Hallucinated URLs in planner output.** Tightened
  `PLANNER_SYSTEM` in `src/argus/graph/nodes.py:81` with URL-integrity
  rules. The planner is now explicitly told to emit `kind: search_result`
  when it doesn't know an exact URL, not invent one. Verified by 9
  tests in `tests/test_planner_no_hallucinated_urls.py`.
- **Bug #2: "Can't parse entities" on report preview.** Added
  `_html_escape_for_tg()` helper in `src/argus/bot.py:172`. The
  report-preview send now uses `ParseMode.HTML` with a try/except
  plain-text fallback. 9 tests in `tests/test_bot_markdown_fallback.py`.

### Phase B — structural citation integrity
- **New module `src/argus/citations.py`** (14 KB, ~400 LOC). Lifted
  from AI-Q but rewritten Argus-shaped (no session-registry/contextvars
  plumbing, no tool-name-specific URL extractors). Public API:
  - `SourceRegistry.add_from_fetched(items)` — bulk-load from
    ArgusState['fetched'].
  - `SourceRegistry.resolve_url(url)` — 5-strategy cascade (exact →
    truncation → prefix → child-path → query-subset).
  - `verify_citations(md, registry) -> VerificationResult` — strips
    unregistered URLs, audit trail.
  - `sanitize_report(md) -> SanitizationResult` — drops shorteners,
    IPs, truncated URLs, non-http schemes.
- **Wired into `report_builder_node`** in `src/argus/graph/nodes.py`
  between `md_text` composition and `md_path.write_text`. Wrapped in
  try/except so a citation-module failure can't block report delivery.
- **21 RED-first tests** in `tests/test_citations.py`.
- **ddgs_search timelimit** — `d/w/m/y/None` lifted from AI-Q's
  `duckduckgo_news_search`. 8 RED-first tests.

### Known limitation (documented in test, not a bug)
The truncation/prefix strategies (steps 2 & 3 of the cascade) do NOT
enforce path-segment boundaries — this is necessary to accept
arxiv-version truncation (`2402.12345` → `2402.12345v1`). The trade-off:
single-candidate non-arxiv cases like `/us/benefits` matching
`/us/benefitsOther` are accepted (rare false positive). Multi-candidate
ambiguity still correctly returns `None`. Documented in
`tests/test_citations.py::test_resolve_child_path_segment_boundary_safe_ambiguity_fallback`.

---

## 3. What is NOT proven yet (next session must verify)

### Live bot retest on the AI-Q query
The bug Albert hit was: planner hallucinated URLs in `planned_sources`
for "AI-Q" query. After today's fixes:
- Prompt fix reduces rate but cannot eliminate (8B models under
  pressure will still hallucinate).
- Citation registry structurally prevents fake URLs from reaching
  the report — they get stripped at `report_builder_node`.

**Cold-me: ask Albert to retest the AI-Q query (or any hallucination-
prone query).** What to look for:
- Bot log should show: `⚠️ citation integrity: stripped N
  unregistered URL(s), M sanitized URL(s)` if the planner emits fakes.
- The final report on disk should NOT contain `github.com/transformers-
  metacognition` or similar fakes.
- The report preview send should NOT crash with "Can't parse entities"
  even if the excerpt has `_`, `*`, `[`, or `&`.

If retest passes: merge `feat/argus-fixes` → `feat/argus-v1` (or
wherever Albert wants it). If retest fails: investigate, don't merge.

### `get_verified_sources` LangChain tool (B4, deferred)
The planner currently does NOT see the registry during planning —
only the report_builder sees it at write time. This means the planner
still wastes tokens generating URLs that will be stripped.

**Cold-me: optionally add `get_verified_sources` as a LangChain @tool**
(see `tests/` for the pattern; `make_langchain_tools()` in
`src/argus/tools.py:958` is where to register it). This was deferred
because the structural fix already prevents fake URLs from reaching the
report — the planner tool is a *nice-to-have* for token efficiency, not
a correctness requirement.

If you add it: build the registry from `state["sources"]` (researcher
output, populated before planner runs), expose as a tool the planner
can call to ask "is URL X real?". Tests in the same style as
`test_citations.py`.

---

## 4. Next moves (the user-facing roadmap)

Albert's original next-task list at end of last handoff (yesterday):
> #5 Deep Research Bench / GAIA harness (~3h)
> #6 WeasyPrint designed PDF (~4h, risky on 16GB box)

Today's session consumed the time that would have gone to #5 or #6.
Roadmap after live retest passes:

1. **Live retest of bot on AI-Q query** (15 min, Albert-driven).
2. **Merge `feat/argus-fixes` → `feat/argus-v1`** (5 min, no code).
3. **#5 Deep Research Bench harness** (3h, subagent-driven). This is
   the eval infrastructure that lets us MEASURE whether today's
   citation registry actually improves report quality vs. yesterday's
   version. Without #5, we have no way to know if all this work
   mattered. Highest ROI for the next session.
4. **(Optional) #4 `get_verified_sources` planner tool** (1h) — would
   let the planner avoid emitting URLs that will be stripped, saving
   tokens. Not a correctness fix; a token-efficiency fix.
5. **#6 WeasyPrint designed PDF** — defer. The ReportLab path Albert
   already has is adequate; WeasyPrint on a 16GB Windows box with
   flaky pagefile is risky and the gain over ReportLab is small
   (CSS-rendered callouts vs. Platypus ParagraphStyle).
6. **Sectioned report generation** — the AI-Q survey flagged this
   as the highest-value single steal from LangChain's
   `open_deep_research`. T7.4 ("lecture format") would be the
   natural home. ~2h, can wait for a focused session.

---

## 5. Open kanban (carryover from yesterday, still relevant)

- `t_92d06264` — T7 design/UX; body has 9-subtask breakdown. Today's
  citations work touches T7 because it changes the report_builder.
- `t_52c6aec5` — T4 live review; still `blocked`; awaiting dispatch
  + Albert's review.
- `t_7f2b625c` — T5; body says DONE; dispatcher hasn't flipped state.

**Cold-me: do not assume these are unblocked.** Check the kanban
state with `hermes kanban list` before resuming work on any of them.

---

## 6. Environmental gotchas learned this session (READ THESE)

### patch tool over-indentation bug — third occurrence today
The `patch` tool in multi-line edits of `nodes.py` and `bot.py` and
`tools.py` has an over-indentation bug that has now bitten me THREE
times in one session. Symptoms: `IndentationError: unexpected indent`
or `unindent does not match any outer indentation level`. Cause: the
patch tool adds 4 extra spaces to lines after a `(`. Already in
profile memory from earlier Argus work — **the workaround is to use
a Python script (write the fix to disk, run it) instead of `patch`**
when touching these files. Cold-me: DO NOT use `patch` on
`argus/graph/nodes.py`, `argus/bot.py`, or `argus/tools.py`. Use
the script approach (cache has working examples: `fix_bot2.py`,
`fix_nodes.py`, `fix_ddgs.py` under
`A:\HermesHome\hermes\profiles\coding-app\cache\`).

### `ddgs_search` timelimit kwargs — empty string semantics
The duckduckgo_search library accepts `timelimit='d'|'w'|'m'|'y'|None`.
Argus treats `''` as `None` so callers can pass env-var values without
filtering empty strings. Documented in `ddgs_search()` docstring.

### Citation registry is fail-soft
The citation-integrity pass in `report_builder_node` is wrapped in
`try/except`. A failure logs a warning and proceeds with the unmodified
`md_text`. This means: if a future change to `citations.py` has a bug,
the bot will still deliver reports — they just won't have the
integrity guarantee. **Cold-me: do NOT remove the try/except wrap.**

---

## 7. Files I created or modified today (so you can `git show` them)

```
src/argus/bot.py                                +65 / -7
src/argus/graph/nodes.py                        +48 / -2   (PLANNER_SYSTEM + report_builder)
src/argus/citations.py                          +400 / 0   (NEW — citation registry)
src/argus/tools.py                              +52 / -4   (timelimit)
tests/test_bot_markdown_fallback.py             +94 / 0    (NEW)
tests/test_citations.py                         +320 / 0   (NEW)
tests/test_planner_no_hallucinated_urls.py      +85 / 0    (NEW)
tests/test_web_search.py                        +87 / -1   (timelimit tests)
```

Total: ~1100 LOC added (mostly tests + the new citations module).

---

## 8. How to resume

**If Albert says "resume":**

1. Read this file + `tonight/WAKEUP-README.md`.
2. Confirm branch is still `feat/argus-fixes`:
   `cd /a/Hermes/Agents/argus && git branch --show-current`.
3. Run the test suite to confirm green state:
   `export PYTHONPATH="" && ./venv/Scripts/python.exe -m pytest tests/ --ignore=tests/manual_e2e.py --ignore=tests/test_e2e_research.py --ignore=tests/test_bot.py -q`
4. Ask Albert if he wants to **retest the bot live first** (recommended)
   or jump to task #5 (Deep Research Bench harness).
5. If retest: start the bot
   (`cd /d "A:\Hermes\Agents\argus" && set PYTHONPATH= && venv\Scripts\python.exe -m argus.bot`),
   have Albert send the AI-Q query from Telegram, watch the log for
   "citation integrity" messages, then read the on-disk report to
   verify no hallucinated URLs.
6. If task #5: see §9 below for the bench harness plan.

**If Albert says "go" without specifying:** default to live retest
first (5 min), then task #5.

---

## 9. Task #5 prep notes (so cold-me has the runway)

**Goal:** Build an eval harness that runs Argus against a fixed query
suite (e.g. GAIA, or a curated 10-query set), measures
citation-accuracy (proxy: does the report's URLs resolve via the new
registry?), and produces a JSON report.

**Approximate architecture:**
```
bench/
  queries.jsonl       # {id, query, expected_sources: [url, ...]}
  runner.py           # invoke argus graph with each query, capture report
  scorer.py           # resolve URLs in report via registry, compare
  report.py           # emit JSON + Markdown summary
```

**Estimated time:** 3h.
**Subagent-friendly:** yes — the bench itself is a separate tool, not
modifying Argus. Can be built in parallel with other Argus work.

**Critical risk:** the bench needs a fixed LLM tier (or pinned model
in FreeLLMAPI) — otherwise scores will be noisy across runs because
the model changes underneath. Set `ARGUS_BENCH_MODEL=qwen3-coder:free`
or similar in env. (Or whatever the `strong` tier currently resolves to.)

**Stretch goal:** add a `--compare` mode that diffs two branches
(`feat/argus-v1` vs `feat/argus-fixes`) on the same query set to
prove today's work actually moved the needle.

---

## 10. Pointer dump (so cold-me doesn't waste time re-finding things)

- Argus repo: `A:\Hermes\Agents\argus`
- AI-Q clone: `/tmp/aiq-survey/aiq` (depth 1, develop branch — discardable)
- Citation registry source of truth: AI-Q's
  `src/aiq_agent/common/citation_verification.py`
- AI-Q survey report: `tonight/HANDOFF-RESEARCH-2026-07-08.md`
  (yesterday's X1 sweep — has the broader competitive picture)
- Today's AI-Q follow-up survey: in
  `A:\HermesHome\hermes\profiles\coding-app\cache\delegation\subagent-summary-0-20260708_175500_554567.txt`
- Repair scripts I wrote (proof of patch-tool workaround):
  `A:\HermesHome\hermes\profiles\coding-app\cache\fix_bot2.py`,
  `fix_nodes.py`, `fix_ddgs.py`
- Personal-rag vault notes on AI-Q: not yet ingested — flagged in
  the AI-Q report §8 as a candidate for rag-ops ingest.
- FreeLLMAPI: localhost:3001 — healthy throughout the session.
- Telegram bot: `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_USER_ID`
  in `.env`. Tested working at session start (Albert successfully
  ran the bot and saw the bug).
- Patch-tool gotcha cache: `A:\HermesHome\hermes\profiles\coding-app\cache\`
  (see §6 — these scripts are reference for the next time you need
  to edit those files).

---

## 11. Don'ts (from this session, distilled)

- Don't use `patch` on `argus/graph/nodes.py`, `argus/bot.py`, or
  `argus/tools.py` for multi-line edits — it'll over-indent.
- Don't merge `feat/argus-fixes` without a live retest first.
- Don't merge `feat/argus-fixes` into `feat/argus-v1` directly
  without reviewing the diff for test conflicts (yesterday's
  `feat/argus-v1` may have drifted).
- Don't remove the try/except wrap around the citation-integrity
  pass in `report_builder_node` — it's there so the bot always
  produces a report even if the registry crashes.
- Don't disable the segment-boundary guard in step 4 (child-path)
  — it prevents `/us/benefits` from matching `/us/benefitsPage`
  when both are in the registry.
- Don't trust the planner not to hallucinate even after the prompt
  fix. The citation registry is the safety net.

---

## 12. Open question for Albert (cold-me: ask, don't assume)

> "Are we OK retiring the `feat/argus-v1` branch in favor of
> `feat/argus-fixes` going forward, or do you want to keep both?
> Today's 4 commits could land on either, but the diff between them
> may have grown."

If Albert says retire: rename `feat/argus-fixes` → `feat/argus-v1` and
fast-forward. If Albert says keep both: PR both branches to main and
let him review.

---

## 13. Single-sentence state

**Branch `feat/argus-fixes` at `e8b5135`, 143/143 tests green, 4 atomic
commits ready for live retest + merge. Cold-me: do the retest first.**

— coding-app, 2026-07-08 18:00