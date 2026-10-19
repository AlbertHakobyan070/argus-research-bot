# WP-B — research depth: search, fetch, triage, digest, gap mode, recency (Task 7)

Spec: `docs/argus-v4-design.md` §3 WP-B, with §2.8 (URLs, the ledger, the fetch chain, the search wave, anchored quote verification), §2.4 (the budgets), §2.5 (the between-item drain and the event kinds), §2.3 (the facade's skip/degrade rules and `ALL_SKIPPED_RAISES`) and §5. Research no longer depends on the intel-stack: a native fetch rung reads HTML, PDFs, arXiv and GitHub on its own, behind an SSRF guard and a byte cap. Every budget goes through one `SpendLedger`. Candidates are deduplicated by canonical URL and reranked by an LLM before they are fetched. Long documents are digested in parts, and every quote is checked against its document. The panel's coverage gaps get bounded gap waves, and a user Extend gets its own allowance. The node stops between items when a drain is requested, and emits the custom events. The five research roles (`scout.querygen`, `research.rerank`, `research.digest`, `research.followup_querygen`, `research.extend_querygen`) run on `llm.complete`.

### What changed

- `src/argus/engines/urls.py` (new): `canonical_url`, `domain`, `registrable_domain`, with the §2.8 rules.
  - Lowercase host, no `www.`, no default port, no trailing slash, no fragment. Tracking parameters are dropped (`utm_*`, `ref`, `fbclid`, `gclid`), and `http` becomes `https`.
  - `export.arxiv.org` becomes `arxiv.org`. arXiv `abs|pdf|html` pages (any version), ar5iv pages and Hugging Face paper pages become `https://arxiv.org/abs/<id>`.
  - `tab` is dropped on github.com only (the orchestrator's ruling on the WP-G finding).
  - Non-http URLs (`file:///` appends) are returned as given.
  - `registrable_domain` keeps the last two labels, or three under a small hand-kept second-level list (`co.uk`, `com.au`, `ac.uk`, …, plus `github.io` / `readthedocs.io`, where every subdomain is a different project). It is marked `ponytail:`: the list is not the Public Suffix List.
- `src/argus/engines/ledger.py` (new): `SpendLedger`, the one lock-guarded budget object.
  - `take(kind, n)` is an atomic check-and-count for `fetch_first|gap|extend`, `exa_search_first|gap|extend` and `exa_contents_pages`.
  - `count(kind, n)` bumps `exa_text_pages`, `arxiv_queries`, `ddgs_queries`, `github_queries` and `dropped_undated`.
  - `add_usd`, `remaining`, `spent`, `grant_extend()` and `set_recency`.
  - `for_run(rc, search_spend)` builds the §2.4 caps from the run config:
    - first pass: `first_pass_fetch_cap()` fetches and `max_searches − exa_search_reserve()` Exa searches;
    - gap reserve: `gap_reserve()` fetches and `exa_search_reserve()` searches;
    - each granted Extend: `fetch_per_wave` fetches and `max_searches // waves` searches;
    - `max_contents_pages`.
  - `snapshot()` is what the nodes write to `search_spend`: the §2.4 keys plus every per-kind counter, so a ledger rebuilt from it in the next node continues where the last one stopped.
- `src/argus/engines/fetchers.py` (new): `fetch(source, *, cfg, ledger, exa_enabled) -> FetchResult(ok, markdown_path, title, chars, fetcher, ms, error, rungs_tried)`. The rungs run in `FetchConfig.chain` order:
  - `exa_text`: the hit's inline Exa text (more than 300 chars), with no request.
  - `native` (httpx), by URL class:
    - arXiv: ar5iv's HTML through trafilatura, else `arxiv.org/pdf/<id>` through pypdf;
    - a GitHub repo: its raw README (`HEAD/README.md`) and nothing else;
    - a PDF (`.pdf` path, PDF content type or `%PDF` bytes): pypdf, stopping once the on-disk cap is reached;
    - anything else: `trafilatura.extract(html, include_links=False, favor_recall=True)` on the raw bytes, with the title from trafilatura's metadata.
  - `exa_contents`: `POST https://api.exa.ai/contents` with `{"urls": [url], "text": {"maxCharacters": max_chars_on_disk}}`. It runs only with Exa enabled, takes one `exa_contents_pages` unit first, and adds `costDollars.total` to the ledger.
  - `intel_snatch`, `intel_crawl`, `intel_stealth`: the intel-stack wrappers. They are in the chain only when both `INTEL_*` paths exist (checked at import).
  - **SSRF guard**: before every network rung, and on every redirect hop of `native` (at most 5), the host's addresses must be global. Anything that is not (`ipaddress.is_global`, IPv4-mapped IPv6 unwrapped: loopback, RFC 1918, link-local, CGNAT, reserved) is refused, and so are the proxy's (`FREELLMAPI_BASE_URL`) and the console's (`ARGUS_WEBAPP_PUBLIC_URL`) own host and port. The error reads `refused: private address (…)` and stops the whole chain.
  - **Byte cap**: a declared `Content-Length` over `max_bytes`, or a streamed body passing it, fails the rung (`too large (…)`).
  - Text on disk is LF-only and capped at `max_chars_on_disk`; `chars` is its length. A rung that yields under 200 chars of text fails. A rung that raises becomes `<rung>: <Type>: <message>`, and the chain moves on. A source no rung can fetch comes back `ok=False` with every rung's error and `rungs_tried`.
- `src/argus/engines/search.py` (new): `run_wave(queries, *, cfg, ledger, recency, mode) -> (hits, errors)` and `resolve_recency(rc, brief, *, today=None)`. Planning runs in query order on the node thread:
  - **Exa**: one `exa_search_<mode>` unit per query; a gap wave takes at most 2. Over budget, the query runs on DDGS, with one line per wave: `search: exa budget (first pass|gap wave|extend) spent — N queries ran on ddgs`. With no `EXA_API_KEY` it runs on DDGS silently, since that is a configuration state, not a failure.
  - **arXiv**: at most `max_queries_per_run` per run. Requests are serialised under one process-wide lock, at least `min_interval_s` apart. The overflow runs on the web provider, with one line: `search: arxiv cap (2 per run) reached — 2 queries ran on exa`.
  - **GitHub**, by `github_mode`:
    - `exa_domain`: Exa with `includeDomains: ["github.com"]`, or the repo search when Exa is not usable;
    - `repo_search`: the GitHub API, counted in `github_queries`;
    - `off`: the web provider.
  - **Recency**, only when a window is passed: Exa `startPublishedDate` (`<start>T00:00:00.000Z`), DDGS `timelimit`, arXiv `sortBy=submittedDate`. Undated Exa hits are dropped and counted (`dropped_undated`).
  - Hits are deduplicated by canonical URL, every hit gets `canonical_url`, sub-question tags are merged, and the hit with page text wins. Provider failures keep their own error strings.
- `src/argus/graph/search_providers.py`:
  - The provider functions stay. `exa_search` gains `text_max_chars`, `include_domains`, `start_published_date`, `search_type` and `ledger`; with a ledger it counts `costDollars.total` and the hits that came with text (`exa_text_pages`). Its `published` is the date part.
  - `ddgs_hits` gains `timelimit`. `arxiv_hits` gains `sort_by`, and parses `<published>` into the hit's date.
  - `run_query_wave` is now `engines.search.run_wave` (a re-export). `exa_max_calls` and the per-wave budget are gone.
- `src/argus/graph/research.py` (rewritten). One node run is one pass, chosen from the state:
  - **append** (`append_only`): exactly the pinned sources, with no search and no budget.
  - **gap** (`coverage_gaps`): refused at `gap_waves_max`, with a `research: gap wave refused — gap_waves_max (N) reached` line. Otherwise it is one wave:
    - one `research.followup_querygen` call and a `mode="gap"` search;
    - up to `gap_fetch_per_wave` fetches, only of sources tagged with a gap sub-question;
    - one `gap_history` entry `{round, wave, sub_qs, new_strong, source, new_strong_by_subq}`, `gap_waves += 1`, and `coverage_gaps` cleared.
    - `source` is `"coverage"` when a gap sub-question is under `coverage_min_strong` or no outline section owns it, else `"judge"`.
  - **first** (no research and no Extend yet): up to `research.waves` waves of up to `fetch_per_wave` fetches, all within the first-pass cap. Before each later wave, while the coverage has gaps, there is one follow-up querygen and a `mode="first"` search.
  - **extend** (every other entry, which is a user Extend: `extend_prep` only bumped its counters):
    - `grant_extend()`;
    - one `research.extend_querygen` call over every sub-question (up to 2 queries each, `item=""`) and a `mode="extend"` search;
    - then waves as in the first pass, spending `fetch_extend`.
  - A **wave** picks the pinned pending sources (always, never counted) and up to the cap of the others:
    - `research.rerank` scores up to 40 candidates (pre-selected by keyword `triage`). The wave takes the top scores; ties keep candidate order, a score of 0 is dropped, and a candidate the reply omits ranks after every scored one.
    - Degrade → `keyword_triage`: `triage` picks the wave and a `degrade:` line says so.
    - `research.rerank: false` uses `triage` directly.
    - The wave then fetches (`fetch_one_source` → `fetchers.fetch`), scores credibility and digests.
  - **Fetch pool**: at most `fetch.concurrency` fetches are in flight. The next is submitted only when one finishes, and only while no drain is requested. Each non-pinned source takes one unit of the pass's fetch kind at submit. Results come back in pick order.
  - Sources are deduplicated by canonical key (`file:///…` for local appends) against `fetched`, and against everything already submitted in this node run, so a failed fetch is not retried by a later wave of the same pass.
  - An unfetchable source is the line `fetch <url> failed: <rung errors> (rungs tried: <rungs>)`. The v3 `research: all N source(s) failed to fetch` line stays.
  - **Digest** (`research.digest`, on `digest_concurrency` workers):
    - a document longer than `digest_head_chars + digest_mid_chars` is read in `min(digest_chunks_long_doc, ⌈len / (head + mid)⌉)` equal parts, each cut to its head and middle; the call items are `<url>#chunk<i>`, and the prompt says `Part i of k`;
    - the merge takes the best part's relevance, with its stance and `source_type`, and keeps claims deduplicated by text;
    - `authority = note_authority(authority_prior, source_type)`, and every claim gets `quote_verified`;
    - an exhausted part is `[r<k>] skip: research.digest <item> (<facade message>)` with its attempts kept in `model_calls`, and the note is built from the parts that were read.
  - **ALL_SKIPPED_RAISES**: when the node would end with `fetched` non-empty and `evidence` empty, it raises `llm.EngineError(role="research.digest")` carrying the digest attempts. That replaces v3's "digested 0 evidence notes" errors line.
  - **Quote verification** (`QuoteIndex`, `quote_verified`):
    - quotes and documents are normalised (case, whitespace, quote marks, dashes); an exact substring is `True`;
    - otherwise the quote's word 4-grams vote for alignments in the document's 4-gram index, and the top 3 are compared on windows of `len(quote) + 40` chars;
    - no quote is `None`.
    - The similarity is computed over the window's matched span; see the decisions.
  - **Drain** (`runtime: Runtime | None = None`, injected by LangGraph by name). `runtime.drain_requested` is read on the node thread, inside the fetch pool before each submit and between waves and searches. Once it is set:
    - no new fetch starts, and the in-flight fetches finish and are digested;
    - the remaining waves and searches are skipped;
    - the node returns its normal partial update plus `cancelled: research stopped after <k>/<n> fetches in wave <w> (<reason>)`.
  - **Events** (`runtime.stream_writer({"kind", "node", "data"})`, from the node thread only):
    - `stage` with status `start`, `draining` or `end`;
    - `search` `{wave, mode, queries, hits, errors}` (counts);
    - `rerank` `{wave, candidates, kept}`;
    - `fetch` `{url, ok, fetcher, ms, pinned}`;
    - `digest` `{source_id, ok, relevance, cached}`;
    - `degrade` `{role, degradation, reason}`.
  - The `FetchedItem` v4 fields are filled: `canonical_url`, `domain`, `published` (from the hit), `fetcher`, `fetch_ms`, `chars`, `authority_prior`, `pinned`. Local appends get `fetcher="local"`, `pinned=True` and authority 2.
- `src/argus/graph/scout.py`: `plan_queries(brief, *, rc, rnd, runtime=None)` runs on `scout.querygen`:
  - a degrade gives `fallback_queries` plus `[r<k>] degrade: scout.querygen → subquestion_text_queries (…)`, every attempt kept, and a `degrade` event;
  - a sub-question the reply skipped still gets its deterministic query;
  - an academic brief (`is_academic_topic`) gets a prompt line, and its `mixed` sub-questions' Exa queries get the `research paper` category, as `paper` ones already did;
  - `scout_node(state, runtime=None)` reads `run_config.require(state)`, decides the recency window once (`resolve_recency`, recorded in `search_spend["recency"]`, with its note in `errors`), runs one `mode="first"` wave, deduplicates the seeded sources and the hits by canonical URL (seeded first, `SCOUT_SOURCE_CAP`), writes `search_spend`, and emits `stage` / `search` events.
- `src/argus/graph/credibility.py`:
  - `is_official_doc(url, kind="")`: a `docs.` or `developer(s).` host, `*.readthedocs.io|org`, a `/docs` or `/documentation` first path segment, or a hit labelled `official_doc`;
  - an official-doc prior in `credibility_score`, never on the low tier;
  - `authority_prior(url, kind="")` on the 0–3 scale: 0 for the low tier, 1 unknown, 2 trusted and the user's own `file:///` material, 3 primary or official docs. There is no arXiv year heuristic here, because a fetched page is real;
  - `note_authority(prior, source_type)`: the digest's `official_doc` label lifts an unknown source to 2, never a 0.
- `src/argus/graph/grounding.py` (rewritten). A source counts when its evidence note has relevance ≥ 3 and authority ≥ 1. Every entity of `brief.entities` must be named in at least the threshold's number of them:
  - the full phrase, normalised (lowercase, non-alphanumerics to spaces, a dot kept only between digits so `GLM 5` ≠ `GLM 5.2`);
  - searched in the fetched document's whole text (cached by path, mtime and size), else in title + excerpt when the file is gone;
  - `mention_count` is the least-supported entity's.
  - No entities means nothing to check, so the result is grounded. `extract_entity_tokens` and the request-token rule are gone.
- `src/argus/tools.py` (`ddgs_search` only): it raises when the package is missing and lets DDGS exceptions through. The search wave turns that into the errors line `ddgs: <Type>: <message>`, where v3 returned a silent `[]`.

### Decisions where the spec is silent, and the orchestrator's rulings

- **Extend boundary (ruling, option B).** `extend_prep_node` keeps only its counters and flags, like `revise_prep`. `research_node` detects the Extend pass (not `append_only`, no `coverage_gaps`, and `research_rounds` or `extend_rounds` non-zero) and owns the allowance, the querygen and the wave. No allowance is handed between nodes.
- **`search_spend` keys (ruling 1)**: the snapshot carries the per-kind ledger counters (`fetch_first`, `exa_search_gap`, …) beside the §2.4 keys, so the next node's ledger rebuilds exactly. `exa_searches` is their sum.
  - `exa_contents_pages` counts the pages requested from `/contents`, since the budget is taken before the request; a request that fails still counts.
  - `exa_usd` sums what Exa reported in `costDollars`, so it is what was billed.
- **`gap_history` (ruling 2)**: one entry per gap wave; `new_strong` counts the wave's new strong notes, and `new_strong_by_subq` (an addition) gives them per gap sub-question for the panel's zero-yield rule.
- **Events (ruling 3)**: the payload is `{kind, node, data}` with the §2.5 kinds, and `node` is `research` or `scout`. `search` reports counts.
- **SSRF (ruling 4)**: the guard runs on every redirect hop and before every network rung. It refuses every non-global address, a superset of §2.8's list. `ponytail:` marks two limits:
  - resolve-then-connect leaves a DNS-rebinding window (TOCTOU);
  - the intel-stack rungs are separate processes, so their own redirects are not guarded (only the source URL is).
- **Line formats (ruling 5)**: `skip:` / `degrade:` lines carry `[r<k>]` with `k = 1 + revise_rounds + extend_rounds`. The fetch, gap-refusal and `cancelled:` lines are exact and untagged. The `digest: <url> has <200 chars of text; skipped` line is not a facade skip, so it is untagged and not counted as one.
- **Grounding (ruling 6)**: full `brief.entities` phrases only. Nothing writes `entities` until WP-I's brief does, so until then every run is grounded and no banner shows.
- **Over-budget queries (ruling 7)** run on the web provider, with one `search:` line per wave. A DDGS failure is an errors line.
- **Quote verification similarity, a deviation from the literal formula.** `difflib.SequenceMatcher(quote, window).ratio()` against a window of `len(quote) + 40` chars is at most 2L / (2L + 40). That is below `quote_verify_fuzzy` (0.9) for every quote shorter than 180 chars, so no short quote with any LLM edit could ever verify. The ratio is therefore taken over the window's matched span, from the first to the last matching block. The anchors, the top-3 alignments, the windows and the threshold are as specified, and the ≤ 50 ms bound holds (`tests/test_quote_verify.py`).
- **Follow-ups that yield nothing.** After a follow-up querygen gives no queries (empty, or `skip_followup_wave` / `skip_extend_wave`), the pass makes no more follow-up calls. Its remaining waves still fetch pending candidates while gaps and budget remain: the degradation skips the search, not the budget. The gap pass does the same with its candidates, and the Extend pass fetches the leftovers.
- **Rerank candidates**: keyword `triage` pre-selects at most 40 candidates, which bounds the prompt. Pinned sources are never reranked.
- **Recency is decided once**, by the scout, and later passes reuse `search_spend["recency"]`. A run without one (a pre-v4 checkpoint) has no recency. An explicit `run_config.recency` wins. Otherwise the brief's `recency_hint` applies only to a `time_sensitive` brief, and an unusable hint is ignored with a note.
- **`research.digest` with `ALL_SKIPPED_RAISES`** raises at the node's end, whatever the pass, so a gap wave whose few digests fail never kills a run that already holds evidence.
- **`per_subq_target`** still has no reader, as in v3 (`PER_SUBQ_TARGET` was never used). The balance across sub-questions is triage's round-robin.

### Exa billing (Step 5, the one live search + one `/contents` call)

On 2026-10-19 a `/search` with `type: "auto"`, `numResults: 3` and `contents.text.maxCharacters: 2000` returned 3 results, all with text:

- `costDollars = {"search": {"neural": 0.007}, "total": 0.007}`, with no `contents` component.
- No result carried a `publishedDate`, and `resolvedSearchType` was empty.

A `/contents` call for one URL with `text.maxCharacters: 2000` returned:

- `costDollars = {"contents": {"text": 0.001}, "total": 0.001}`;
- `statuses = [{"id": <url>, "source": "cached", "status": "success"}]`.

Finding: **inline text pages were not billed**; the search price covered them. `/contents` pages are billed at the page price. This was measured with 3 results; the default 8 was not measured. Consequences:

- `RunConfig.exa_cost_estimate_usd` still counts `max_searches × num_results` inline pages at the page price, so it is a ceiling. Its docstring now says so. The formula is unchanged: it is the Task 1 contract, and `tests/test_run_config.py` asserts it. Dropping the inline-page term is the orchestrator's call.
- `exa_text_pages` stays a count of pages that came with text, not of billed pages.
- `exa_usd` is the reported cost, so the per-run spend is exact either way.

### Behaviour changes for handoff.md

- Research works without the intel-stack. On a host without both `INTEL_*` paths, the chain is `exa_text → native → exa_contents`.
- Fetching refuses private and loopback targets, including the proxy and the console. A download stops at `fetch.max_bytes`.
- The depth presets drive research: the wave count, fetches per wave, the first-pass cap and the gap reserve. The Exa budget is per run, where v3 had a per-wave cap that every wave reset, and the per-run spend lands in `search_spend["exa_usd"]`. Each user Extend gets a fresh allowance.
- Appended (`/append`) sources are always ingested, outside every cap.
- Duplicate URL forms (`www.`, tracking parameters, arXiv versions and mirrors, `?tab=` on GitHub) are fetched once.
- Evidence notes carry `source_type` and `authority`, and claims carry `quote_verified`.
- A run whose fetched sources all fail to digest stops with an error instead of shipping an ungrounded report. Individual digest failures are `[r<k>] skip:` lines.
- `extend_prep` no longer searches. The Extend pass's search happens inside research, and `research.extend_querygen` degrading is a `[r<k>] degrade: … → skip_extend_wave` line for the preview to surface.
- The grounding banner now needs `brief.entities` (WP-I). Until then it does not show.
- Cancel: research stops between fetches and returns what it has, with a `cancelled:` line.
- A DDGS failure is now visible in `errors`.
- `ARGUS_RESEARCH_WAVES`, `ARGUS_FETCH_PER_WAVE`, `ARGUS_TOTAL_FETCH_CAP`, `ARGUS_EXA_MAX_CALLS`, `ARGUS_FETCH_CONCURRENCY` and `ARGUS_DIGEST_CONCURRENCY` are no longer read; each logs one startup warning if set (`registry.REMOVED_ENV_KNOBS`). `README.md`, `.env.example` and `.env.mac.example` still describe them. That is WP-H's docs pass, so they are left unchanged here.

### New env vars

None. Removed: the six knobs above. The SSRF guard reads `FREELLMAPI_BASE_URL` and `ARGUS_WEBAPP_PUBLIC_URL`, which already exist.

### Files outside WP-B touched (approved)

- `src/argus/engines/registry.py`: `summarize_calls` derives `exhausted_attempts` from the error-line text only for `legacy.*` roles. A real role's call site keeps every attempt record, so re-counting from the text would count them twice. The six knobs are added to `REMOVED_ENV_KNOBS`.
- `src/argus/graph/nodes.py`: the wave deleted from `extend_prep_node` (ruling, option B). Its docstring and the non-append message follow, and the now-unused `ResearchBrief` import is gone. The `append_only` branch is unchanged.
- `bench/metrics.py`: imports `engines.urls.canonical_url` in place of its own copy, and drops that copy's constants and imports. The failed-fetch regex is unchanged, because research keeps the `fetch <url> failed` form.
- `scripts/demo_run.py`: `def fake_fetch(src, **kw):`, nothing else.
- `src/argus/graph/run_config.py`: the `exa_cost_estimate_usd` docstring only (the Step 5 wording; see Exa billing).

### Test files touched

- New: `tests/test_urls.py` (the canonicalisation table, the v3 duplicates, `domain`, `registrable_domain`), `tests/test_ledger.py` (no overshoot and no lost updates under a 4-thread pool, all-or-nothing `take(n)`, the first-pass / gap / Extend split, the quick preset, the snapshot round trip, unknown kinds), `tests/test_quote_verify.py` (exact / fuzzy / miss, normalisation, the threshold, the ≤ 50 ms bound on a 60 000-char document), `tests/test_fetchers.py` (Review Focus 3 and 7, every rung on fixtures, chain availability, the v4 `FetchedItem` fields), `tests/test_recency.py` (the provider mapping against fake clients, `dropped_undated`, the resolve rules), `tests/test_gap_mode.py` (the gap wave and `gap_history`, the refusal, the Review Focus 5 drain, the events, the Extend pass and its degrade, the append-only pass).
- Rewritten or extended (WP-B's):
  - `tests/test_search_providers.py`: the wave budgets, arXiv, `github_mode`, the DDGS errors line, dedupe, the alias, the scout's querygen degrade and spend;
  - `tests/test_grounding.py`: entity phrases;
  - `tests/test_research_depth.py`: the grounding guards on phrases; rerank and its degrade; the chunked digest; quote verification in the digest; the digest skip line; `ALL_SKIPPED_RAISES`; the first-pass caps;
  - `tests/test_credibility.py`: the official-doc prior, the authority scale and `note_authority`;
  - `tests/test_extend.py`: `extend_prep` only resets counters, and the append-only flag stays for research;
  - `tests/test_tools_routing_and_errors.py`: the fetch-error propagation on the v4 chain, with the exact failed-fetch line.
- Other WPs' tests, minimal edits for this change:
  - `tests/test_continue_extend.py`: the plain-continue assertion moves to where the wave now runs (it checks that research sees an Extend pass); `_parallel_fetch` keyword arguments; the digest stub's return shape; `run_config` in the append-only node test.
  - `tests/test_run_config_resume.py`: the `followup_queries` / `run_query_wave` patches for `extend_prep` removed.
  - `tests/test_reflexion.py`: seam signatures (`**kw`), the 3-tuple returns, `run_config`.
  - `tests/test_graph.py`: `fake_fetch(src, **kw)` and `run_config` in the deep test's input.
  - `tests/test_tools_truth.py`: `_parallel_fetch` keyword arguments; the stealth test drives the intel rungs through `engines/fetchers`.
  - `tests/test_web_search.py`: `test_ddgs_failure_raises`.
  - `tests/test_engines_registry.py`: a real role's `exhausted_attempts` is 0.
  - `tests/test_llm_facade.py`: the risk-9 test runs through `research.digest` on the facade.

### Verification evidence (2026-10-19)

The logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t7\`. Every suite run used `FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1`. No LLM call left the machine, and no bot was started, stopped or restarted.

- RED (`red.log`), with the tests written and no implementation:
  - the files importing the new modules failed at collection (`ImportError: cannot import name 'fetchers' / 'search' from 'argus.engines'`);
  - the rest failed on the missing v4 behaviour: gap mode, the drain and the events, entity-phrase grounding, rerank, the chunked digest and its quote checks, `ALL_SKIPPED_RAISES`, the first-pass caps, the credibility prior and the authority scale, and `extend_prep` still searching.
- GREEN: the focused files pass (`green-focused.log`), and so does every file this task touched.
- The full suite (`full-suite.log`) passes except the four live-proxy tests: `tests/test_llm.py` and the `tests/test_graph.py` fixture fail on the closed port by design, as in `baseline-suite.log` at the base commit.
- A scratch smoke ran the node with `research.rerank` on (the preset default), the real fetch chain on a mock network and a scripted `llm.complete` (`smoke_rerank_node.py`, not committed). It showed one rerank call, the fetches in rerank order, a 503 reported as its fetch line, `authority` 2 from the `official_doc` label, `quote_verified` on every claim, and the events in order.
- Exa (`exa-billing.log`, `exa_billing_probe.py`): the one search and the one `/contents` call above. Field names and values only; the key was never printed.
- The WP-B acceptance runs are the orchestrator's: one q010 medium/standard `live_check`, one `short` run with the `INTEL_*` paths unset, and `bench.compare` against the v3 baseline.
