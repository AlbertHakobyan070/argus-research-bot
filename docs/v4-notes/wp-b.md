# WP-B — research depth: search, fetch, triage, digest, gap mode, recency (Task 7)

Spec: `docs/argus-v4-design.md` §3 WP-B, with §2.8 (URLs, the ledger, the fetch chain, the search wave, verbatim quote verification), §2.4 (the budgets), §2.5 (the between-item drain and the event kinds), §2.3 (the facade's skip/degrade rules and `ALL_SKIPPED_RAISES`) and §5. Research no longer depends on the intel-stack: a native fetch rung reads HTML, PDFs, arXiv and GitHub on its own, behind an SSRF guard and a byte cap. Every budget goes through one `SpendLedger`. Candidates are deduplicated by canonical URL and reranked by an LLM before they are fetched, under a per-sub-question quota. Long documents are digested in parts, and every quote is checked against its document. The panel's coverage gaps get bounded gap waves, and a user Extend gets its own allowance. The node stops between items when a drain is requested, and emits the custom events. The research roles (`scout.querygen`, `research.rerank`, `research.digest`, `research.followup_querygen`, `research.extend_querygen`) run on `llm.complete`.

The first cut is commit `98ebc5b`. Its review found defects; fix round 1 (the section of that name below) records the rulings and what changed. The descriptions in "What changed" are the state after the fix round.

### What changed

- `src/argus/engines/urls.py` (new): `canonical_url`, `domain`, `registrable_domain`, with the §2.8 rules.
  - Lowercase host, no `www.`, no default port, no trailing slash, no fragment. Tracking parameters are dropped (`utm_*`, `ref`, `fbclid`, `gclid`), and `http` becomes `https`.
  - `export.arxiv.org` becomes `arxiv.org`. arXiv `abs|pdf|html` pages (any version), ar5iv pages and Hugging Face paper pages become `https://arxiv.org/abs/<id>`.
  - `tab` is dropped on github.com only (the orchestrator's ruling on the WP-G finding).
  - IPv6 hosts keep their brackets. A malformed http(s) URL (a bad port, an unclosed bracket) raises `ValueError`; every caller that canonicalises search hits drops that hit with an `errors` line naming it.
  - Non-http URLs (`file:///` appends) are returned as given.
  - `registrable_domain` keeps the last two labels, or three under a small hand-kept second-level list (`_SECOND_LEVEL`: `co.uk`, `com.au`, `ac.uk`, …, plus `github.io` / `readthedocs.io`, where every subdomain is a different project). It is marked `ponytail:`: the list is not the Public Suffix List.
- `src/argus/engines/ledger.py` (new): `SpendLedger`, the one lock-guarded budget object.
  - `take(kind, n)` is an atomic check-and-count for the capped kinds (`CAPPED`: `fetch_first|gap|extend`, `exa_search_first|gap|extend`, `exa_contents_pages`).
  - `count(kind, n)` bumps a counter with no cap check: the uncapped counters (`COUNTERS`: `exa_text_pages`, `arxiv_queries`, `ddgs_queries`, `github_queries`, `dropped_undated`), or a capped kind's spend that no cap may refuse (a pinned source's `exa_contents` page).
  - `add_usd`, `remaining`, `spent`, `grant_extend()` and `set_recency`.
  - `for_run(rc, search_spend)` builds the §2.4 caps from the run config:
    - first pass: `first_pass_fetch_cap()` fetches and `max_searches − exa_search_reserve()` Exa searches;
    - gap reserve: `gap_reserve()` fetches and `exa_search_reserve()` searches;
    - each granted Extend: `fetch_per_wave` fetches and `max_searches // waves` searches;
    - `max_contents_pages`.
  - `snapshot()` is what the nodes write to `search_spend`: the §2.4 keys plus every per-kind counter, so a ledger rebuilt from it in the next node continues where the last one stopped.
- `src/argus/engines/fetchers.py` (new): `fetch(source, *, cfg, ledger, exa_enabled, pinned=False) -> FetchResult(ok, markdown_path, title, chars, fetcher, ms, error, rungs_tried)`. The rungs run in `FetchConfig.chain` order:
  - `exa_text`: the hit's inline Exa text (longer than `_EXA_TEXT_MIN` chars), with no request.
  - `native` (httpx), by URL class:
    - arXiv: ar5iv's HTML through trafilatura, else `arxiv.org/pdf/<id>` through pypdf;
    - a GitHub repo: its raw README (`HEAD/README.md`) and nothing else;
    - a PDF (PDF content type, a `.pdf` path or `%PDF` bytes, but never a `text/html` response): pypdf, stopping once the on-disk cap is reached;
    - anything else: `trafilatura.extract(html, include_links=False, favor_recall=True)` on the raw bytes, with the title from trafilatura's metadata.
  - `exa_contents`: `POST https://api.exa.ai/contents` with `{"urls": [url], "text": {"maxCharacters": max_chars_on_disk}}`. It runs only with Exa enabled. An unpinned source takes one `exa_contents_pages` unit first; a pinned one counts it. `costDollars.total` is added to the ledger. A reply of any other shape (not a JSON object; `costDollars` missing or not an object; `total` missing or not a number; `results` missing or not a list; a result, its text or its title of the wrong type) is the rung's own `malformed reply: …` error; a valid `total` is recorded before the results are checked, since Exa billed it.
  - `intel_snatch`, `intel_crawl`, `intel_stealth`: the intel-stack wrappers. They are in the chain only when both `INTEL_*` paths exist (checked at import).
  - **The URL is read the WHATWG way first**: in an http(s) URL a backslash before the query is `/`, for every rung including `intel_*`, so `http://127.0.0.1\@evil.example/` is a request to 127.0.0.1.
  - **SSRF guard**: before every network rung, and on every redirect hop of `native` (at most `_MAX_REDIRECTS`), every address the host resolves to must be global (`ipaddress.is_global`: loopback, RFC 1918, link-local, CGNAT and reserved ranges are refused). An IPv6 address carrying an IPv4 one — IPv4-mapped, IPv4-compatible (`::a.b.c.d`) or NAT64 (`64:ff9b::/96`) — is judged by that IPv4 address. The proxy's (`FREELLMAPI_BASE_URL`) and the console's (`ARGUS_WEBAPP_PUBLIC_URL`) own endpoints are refused too, compared by host name and port. The name is compared IDNA-encoded, as the resolver encodes it: `。`, `．` and `｡` are label separators, as they are for the request, and fullwidth letters fold to ASCII (the request does not fold them: httpx refuses such a URL as `InvalidURL`). It is lowercased, and a trailing dot is stripped. Addresses are never compared: a public endpoint's address is shared (a CDN, a tunnel edge), and a private one is refused by the private-address rule anyway. The error reads `refused: private address (…)` and stops the whole chain.
  - **The checked address is the one connected to.** Each `native` hop is sent to `scheme://<checked address>[:port]/path?query` (IPv6 in brackets) with the original `Host` header and `extensions={"sni_hostname": host}`, so TLS still verifies the real name and no second DNS lookup can swap the target. When a host has several checked addresses, the next is tried when a connection is refused or times out (a blackholed IPv6 route), as a connect by name would. Keep-alive is off, because the pool keys connections by address. These pinned requests go on a client with an explicit transport, so httpx mounts no proxy on it.
  - **Through an environment proxy, by name.** When an HTTP(S) proxy from the environment applies to the URL (httpx's own `trust_env` selection, `NO_PROXY` included), the hop goes to that proxy by host name instead: the proxy resolves the name, so the pin cannot apply. The guard still runs first, on Argus's own resolution. A host `NO_PROXY` covers keeps the pin. The residual risk is under the known limits below.
  - **Byte cap on the decoded bytes**: requests send `Accept-Encoding: gzip, deflate`; the raw stream (`iter_raw()`) goes through a bounded `zlib.decompressobj` (`max_length`, the `unconsumed_tail` fed back), so the decoded body never exceeds `max_bytes` plus one chunk. Any other `Content-Encoding` (`br`, `zstd`, …) is refused before a byte is read, with an error naming it. A declared `Content-Length` over `max_bytes`, or a raw stream passing it, also fails the rung (`too large (…)`).
  - **Rung errors**: a rung's expected failures are its own error type — an HTTP status, an httpx transport or timeout error, a refusal, a size, an exception from trafilatura or pypdf — and the type of the underlying exception is recorded (`native: ConnectError: …`, `native: trafilatura: ValueError: …`, `native: pdf: PdfReadError: …`). Anything else propagates. `rungs_tried` is `[{"rung", "error"}]` in chain order, with `""` for the rung that succeeded, so a failed rung keeps its error when a later rung succeeds; the fetched item carries it (`FetchedItem.rungs_tried`). Nothing new is shown in the report.
  - Text on disk is LF-only and capped at `max_chars_on_disk`; `chars` is its length. A rung that yields under `_MIN_CHARS` chars of text fails. A source no rung can fetch comes back `ok=False` with every rung's error and `rungs_tried`.
- `src/argus/engines/search.py` (new): `run_wave(queries, *, cfg, ledger, recency, mode) -> (hits, errors)` and `resolve_recency(rc, brief, *, today=None)`. Planning runs in query order on the node thread:
  - **Exa**: one `exa_search_<mode>` unit per query; a gap wave takes at most `_GAP_EXA_PER_WAVE`. Over budget, the query runs on DDGS, with one line per wave: `search: exa budget (first pass|gap wave|extend) spent — <n> queries ran on ddgs`. With no `EXA_API_KEY` it runs on DDGS silently, since that is a configuration state, not a failure.
  - **arXiv**: at most `arxiv.max_queries_per_run` per run. Requests are serialised under one process-wide lock, at least `arxiv.min_interval_s` apart. The overflow runs on the web provider, with one line: `search: arxiv cap (<max_queries_per_run> per run) reached — <n> queries ran on <provider>`.
  - **GitHub**, by `github_mode`:
    - `exa_domain`: Exa with `includeDomains: ["github.com"]`, or the repo search when Exa is not usable;
    - `repo_search`: the GitHub API, counted in `github_queries`;
    - `off`: the web provider.
  - **Recency**, only when a window is passed: Exa `startPublishedDate` (`<start>T00:00:00.000Z`), DDGS `timelimit`, arXiv `sortBy=submittedDate`. Undated Exa hits are dropped and counted (`dropped_undated`).
  - Hits are deduplicated by canonical URL, every hit gets `canonical_url`, sub-question tags are merged, and the hit with page text wins. A hit whose URL `canonical_url` rejects is dropped with `search: <provider> hit dropped, malformed URL '<url>' (<reason>)`. Provider failures keep their own error strings.
- `src/argus/graph/search_providers.py`:
  - The provider functions stay. `exa_search` gains `text_max_chars`, `include_domains`, `start_published_date`, `search_type` and `ledger`; with a ledger it counts `costDollars.total` and the hits that came with text (`exa_text_pages`). Its `published` is the date part.
  - `ddgs_hits` gains `timelimit`. `arxiv_hits` gains `sort_by`, and parses `<published>` into the hit's date.
  - `run_query_wave` is now `engines.search.run_wave` (a re-export). `exa_max_calls` and the per-wave budget are gone.
- `src/argus/graph/research.py` (rewritten). One node run is one pass, chosen from the state:
  - **append** (`append_only`): exactly the pinned sources, with no search and no budget.
  - **gap** (`coverage_gaps`): refused at `gap_waves_max`, with a `research: gap wave refused — gap_waves_max (<n>) reached` line. Otherwise it is one wave:
    - one `research.followup_querygen` call and a `mode="gap"` search;
    - up to `gap_fetch_per_wave` fetches, only of sources tagged with a gap sub-question;
    - one `gap_history` entry `{round, wave, sub_qs, new_strong, source, new_strong_by_subq}`, `gap_waves += 1`, and `coverage_gaps` cleared.
    - `source` is `"coverage"` when a gap sub-question is under `coverage_min_strong` or no outline section owns it, else `"judge"`.
  - **first** (no research and no Extend yet): up to `research.waves` waves of up to `fetch_per_wave` fetches, all within the first-pass cap. Before each later wave, while the coverage has gaps and the pass's fetch budget is not spent, there is one follow-up querygen and a `mode="first"` search.
  - **extend** (every other entry, which is a user Extend: `extend_prep` only bumped its counters):
    - `grant_extend()`;
    - one `research.extend_querygen` call over every sub-question (up to the per-sub-question count in `_QUERYGEN`, `item=""`) and a `mode="extend"` search;
    - then waves as in the first pass, spending `fetch_extend`.
  - A **wave** picks the pinned pending sources (always; never reranked, never under the quota, never counted against a cap) and up to the cap of the others:
    - `research.rerank` scores up to `_RERANK_MAX` candidates (pre-selected by keyword `triage`). A score of 0 is dropped, ties keep candidate order, and a candidate the reply omits ranks after every scored one.
    - **The quota** (`research.per_subq_target`, `_quota_pick`): within the wave's cap a sub-question — a source's first `sub_qs` tag, the bucket `triage` uses — first takes up to `per_subq_target` of its best, in rank order (rerank score order, else triage order); the leftover slots then go by global rank, and only they take untagged sources. It applies to first-pass, Extend and gap waves, and to the degrade path.
    - Degrade → `keyword_triage`: the same quota over `triage`'s order, and a `degrade:` line says so.
    - `research.rerank: false` uses `triage` with the quota directly.
    - The wave then fetches (`fetch_one_source` → `fetchers.fetch`, with `pinned`), scores credibility and digests.
  - **Fetch pool**: at most `fetch.concurrency` fetches are in flight. The next is submitted only when one finishes, and only while no drain is requested. Each non-pinned source takes one unit of the pass's fetch kind at submit. Results come back in pick order.
  - Sources are deduplicated by canonical key (`file:///…` for local appends) against `fetched`, and against everything already submitted in this node run, so a failed fetch is not retried by a later wave of the same pass. Pinned sources are visited first, so when a pinned and an unpinned copy share a key the pinned copy wins. A source whose URL `canonical_url` rejects — from the state or from a search hit — is dropped with `research: dropped source with a malformed URL '<url>' (<reason>)`.
  - An unfetchable source is the line `fetch <url> failed: <rung errors> (rungs tried: <rungs>)`. The v3 `research: all N source(s) failed to fetch` line stays.
  - **Digest** (`research.digest`, on `digest_concurrency` workers):
    - a document longer than `digest_head_chars + digest_mid_chars` is read in `min(digest_chunks_long_doc, ⌈len / (head + mid)⌉)` equal parts, each cut to its head and middle; the call items are `<url>#chunk<i>`, and the prompt says `Part i of k`;
    - the merge takes the best part's relevance, with its stance and `source_type`, and keeps claims deduplicated by text;
    - `authority = note_authority(authority_prior, source_type)`, and every claim gets `quote_verified`;
    - an exhausted part is `[r<k>] skip: research.digest <item> (<facade message>)` with its attempts kept in `model_calls`, and the note is built from the parts that were read.
  - **ALL_SKIPPED_RAISES**: when the node would end with `fetched` non-empty and `evidence` empty, it raises `llm.EngineError(role="research.digest")` carrying the digest attempts. That replaces v3's "digested 0 evidence notes" errors line.
  - **Quote verification** (`QuoteIndex`, `quote_verified`), verbatim (spec §2.8 at `deaba46`):
    - the document and the quote are normalised alike: case, whitespace, quote marks, dashes, markdown markup (emphasis asterisks dropped, `_` read as a space, a link or image replaced by its text), footnote markers (`[n]`, linked or not), ordered-list markers at line starts. Both are then split into `\w+` tokens (`QuoteIndex` holds the document's, built once per document);
    - a quote verifies when its tokens occur contiguously and in order in the document's tokens;
    - an ellipsis (`...` / `…`) splits the quote into segments that must each occur so, in order, each starting within `_SEGMENT_GAP` characters of the previous one's end. The gap is measured on the normalised token text, a little shorter than the raw document;
    - no quote, or a quote with no word in it, is `None`.
    - Punctuation, spacing, quote marks and markup pass. A substituted, added or dropped word or number fails: a changed figure, a swapped entity, "at most" → "at least", a dropped "not".
    - The digest's reply (`_digest_reply`) shortens a quote over its length limit at the last whitespace at or before the limit; only a single token longer than the limit keeps the hard cut. A quote Argus shortened therefore never ends in a fragment.
  - **Drain** (`runtime: Runtime | None = None`, injected by LangGraph by name). `runtime.drain_requested` is read on the node thread: before an Extend or a gap pass's querygen (so a drain already requested costs no LLM call), at the top of every wave, before every pick (so no rerank call runs after a drain), and inside the fetch pool before each submit. Once it is set:
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
  - The `FetchedItem` v4 fields are filled: `canonical_url`, `domain`, `published` (from the hit), `fetcher`, `fetch_ms`, `chars`, `authority_prior`, `pinned`, and `rungs_tried`. Local appends get `fetcher="local"`, `pinned=True` and authority 2.
- `src/argus/graph/scout.py`: `plan_queries(brief, *, rc, rnd, runtime=None)` runs on `scout.querygen`:
  - a degrade gives `fallback_queries` plus `[r<k>] degrade: scout.querygen → subquestion_text_queries (…)`, every attempt kept, and a `degrade` event;
  - a sub-question the reply skipped still gets its deterministic query;
  - an academic brief (`is_academic_topic`) gets a prompt line, and its `mixed` sub-questions' Exa queries get the `research paper` category, as `paper` ones already did;
  - `scout_node(state, runtime=None)` reads `run_config.require(state)`, decides the recency window once (`resolve_recency`, recorded in `search_spend["recency"]`, with its note in `errors`), runs one `mode="first"` wave, deduplicates the seeded sources and the hits by canonical URL (seeded first and pinned seeded sources before the others, `SCOUT_SOURCE_CAP`; a malformed URL dropped with `scout: dropped source with a malformed URL …`), writes `search_spend`, and emits `stage` / `search` events.
- `src/argus/graph/credibility.py`:
  - `is_official_doc(url, kind="")`: a `docs.` or `developer(s).` host, `*.readthedocs.io|org`, a `/docs` or `/documentation` first path segment, or a hit labelled `official_doc`. The URL heuristics never match a user-content host (`_USER_CONTENT_HOSTS`: `docs.google.com`, `sites.google.com`, `notion.site`, `notion.so`, `substack.com`, `medium.com`, each with its subdomains);
  - an official-doc prior in `credibility_score`, never on the low tier;
  - `authority_prior(url, kind="")` on the 0–3 scale: 0 for the low tier, 1 unknown, 2 trusted, official docs by `is_official_doc` and the user's own `file:///` material, 3 the curated primary tier. There is no arXiv year heuristic here, because a fetched page is real;
  - `note_authority(prior, source_type)`: the digest's `official_doc` label lifts an unknown source to 2, never a 0.
- `src/argus/graph/grounding.py` (rewritten). A source counts when its evidence note has relevance of at least `_MIN_RELEVANCE` and authority ≥ 1. Every entity of `brief.entities` must be named in at least the threshold's number of them:
  - the full phrase, normalised (lowercase, non-alphanumerics to spaces, a dot kept only between digits so `GLM 5` ≠ `GLM 5.2`);
  - searched in the fetched document's whole text (cached by path, mtime and size), else in title + excerpt when the file is gone;
  - `mention_count` is the least-supported entity's.
  - No entities means nothing to check, so the result is grounded. `extract_entity_tokens` and the request-token rule are gone.
- `src/argus/bot/history.py` (`continue_cmd`, fix round 1): after the append pass, a pending `/append` ref is marked `ingested` only when research fetched its source, and `failed` otherwise. The user gets one message listing each failed ref with research's line for it, which names the URL, the rung errors and the rungs tried.
- `src/argus/tools.py` (`ddgs_search` only): it raises when the package is missing and lets DDGS exceptions through. The search wave turns that into the errors line `ddgs: <Type>: <message>`, where v3 returned a silent `[]`.

### Decisions where the spec is silent, and the orchestrator's rulings

- **Extend boundary (ruling, option B).** `extend_prep_node` keeps only its counters and flags, like `revise_prep`. `research_node` detects the Extend pass (not `append_only`, no `coverage_gaps`, and `research_rounds` or `extend_rounds` non-zero) and owns the allowance, the querygen and the wave. No allowance is handed between nodes.
- **`search_spend` keys (ruling 1)**: the snapshot carries the per-kind ledger counters (`fetch_first`, `exa_search_gap`, …) beside the §2.4 keys, so the next node's ledger rebuilds exactly. `exa_searches` is their sum.
  - `exa_contents_pages` counts the pages requested from `/contents`, since the budget is taken (or, for a pinned source, counted) before the request; a request that fails still counts.
  - `exa_usd` sums what Exa reported in `costDollars`, so it is what was billed.
- **`gap_history` (ruling 2)**: one entry per gap wave; `new_strong` counts the wave's new strong notes, and `new_strong_by_subq` (an addition) gives them per gap sub-question for the panel's zero-yield rule.
- **Events (ruling 3)**: the payload is `{kind, node, data}` with the §2.5 kinds, and `node` is `research` or `scout`. `search` reports counts.
- **SSRF (ruling 4)**: the guard runs on every redirect hop and before every network rung. It refuses every non-global address, a superset of §2.8's list. Fix round 1 closed the DNS-rebinding window (the connection goes to the checked address), except through an environment proxy (fix round 2; see the known limits). The intel-stack rungs are separate processes: their own redirects and JavaScript navigation are not guarded, only the source URL is — a residual risk outside §2.8, narrowed by the WHATWG reading of the URL (ruled).
- **Line formats (ruling 5)**: `skip:` / `degrade:` lines carry `[r<k>]` with `k = 1 + revise_rounds + extend_rounds`. The fetch, gap-refusal and `cancelled:` lines are exact and untagged. The `digest: <url> has <200 chars of text; skipped` line is not a facade skip, so it is untagged and not counted as one.
- **Grounding (ruling 6)**: full `brief.entities` phrases only. Nothing writes `entities` until WP-I's brief does, so until then every run is grounded and no banner shows.
- **Over-budget queries (ruling 7)** run on the web provider, with one `search:` line per wave. A DDGS failure is an errors line.
- **Quote verification (rulings, spec §2.8).** The rule changed three times:
  - The first cut matched fuzzily: a similarity over the window's matched span (`faa4cdf`), since a whole-window ratio caps below the threshold for every short quote.
  - Fix round 1 added the token rule (`607bac9`).
  - The re-review showed the token-set check still let a swap through when the replacement word sat nearby, and let a dropped "not" pass. The final ruling (`deaba46`) is the verbatim rule above, and `research.quote_verify_fuzzy` is removed.
  - The orchestrator replayed 649 real quotes from five runs on 2026-10-19. The verbatim rule verified no fewer quotes than the fuzzy match, because it bridges ellipses the fuzzy match could not:
    - Claude runs: 471 of 474, against 472. The one difference is a quote cut mid-word, which `_word_cut` now prevents.
    - free-proxy runs: 159 of 175, against 151.
  - The §2.8 performance bound holds (`tests/test_quote_verify.py`).
- **Follow-ups that yield nothing.** After a follow-up querygen gives no queries (empty, or `skip_followup_wave` / `skip_extend_wave`), the pass makes no more follow-up calls. Its remaining waves still fetch pending candidates while gaps and budget remain: the degradation skips the search, not the budget (ruled, no change). The gap pass does the same with its candidates, and the Extend pass fetches the leftovers.
- **Rerank candidates**: keyword `triage` pre-selects at most `_RERANK_MAX` candidates, which bounds the prompt. Pinned sources are never reranked.
- **Recency is decided once**, by the scout, and later passes reuse `search_spend["recency"]`. A run without one (a pre-v4 checkpoint) has no recency. An explicit `run_config.recency` wins. Otherwise the brief's `recency_hint` applies only to a `time_sensitive` brief, and an unusable hint is ignored with a note. Recency on live Exa was verified by the orchestrator (every hit dated).
- **`research.digest` with `ALL_SKIPPED_RAISES`** raises at the node's end, whatever the pass, so a gap wave whose few digests fail never kills a run that already holds evidence.

### Fix round 1 (WP-B review): the rulings and what changed

The review's probes found the defects below; the spec rulings are §2.4 (the `exa_cost_estimate_usd` comment, `faa4cdf`) and §2.8 (`canonical_url`, `SpendLedger`, `fetchers.fetch`, quote verification — `faa4cdf`, `607bac9`).

- **DNS rebinding (Critical).** `check_target` resolved the name and httpx resolved it again, so a rebinding resolver could swap a public answer for loopback. Ruling: connect to the checked address, with the original `Host` and `sni_hostname`, on every hop. Done as described under the fetch chain.
- **Byte cap on decoded bytes.** The cap was measured on what httpx had already decoded: one small brotli body decoded to hundreds of megabytes in one call. Ruling: gzip and deflate only, through a bounded decompressor; any other `Content-Encoding` refused before decoding.
- **IPv6 forms.** IPv4-compatible and NAT64 addresses passed as global. They are now judged by their embedded IPv4 address.
- **Backslash.** A backslash URL was parsed one way by the guard and another by a browser-based rung. The URL is now read the WHATWG way for every rung.
- **Own endpoints.** They were compared by the literal host only, so a trailing dot or the endpoint's own IP passed. They are now compared by normalised name and by resolved address, each with its port. Fix round 2 drops the address comparison, and fix round 3 compares the IDNA-encoded name (see below).
- **Rung errors.** A catch-all turned any exception, a bug included, into an error string. Ruling: expected failures are typed and recorded with the exception type; anything else propagates. `rungs_tried` keeps every failed rung's error, and the fetched item carries it (`FetchedItem.rungs_tried`, the one addition to the fetched item the orchestrator approved).
- **PDF detection.** A `text/html` response is HTML even at a `.pdf` path.
- **Quote verification.** The matched-span ratio alone verified altered figures, swapped entities and "at most" → "at least". Ruling: the token rule on top of the ratio. Markup normalisation keeps the legitimate fuzzy cases verifying. Fix round 3 superseded this with the verbatim rule.
- **Pinned sources.**
  - The pinned copy now wins the dedupe (`_Pass.pick`, and the scout's seeded sources). An append pass whose URL duplicated an unfetched scout source used to fetch nothing.
  - A pinned source never calls `take`: its `exa_contents` page is counted, so a spent page budget no longer refuses it. The paid-page counter is shared, so a pinned page does reduce what remains for unpinned pages — intended.
  - `continue_cmd` marked every pending ref `ingested` whatever happened. It now marks only the fetched ones, and lists the failed ones with their rungs.
- **`canonical_url`.** It raised on a malformed URL and every caller let the node fail. The ruling makes the `ValueError` the contract and the callers drop the hit with an errors line. IPv6 hosts used to lose their brackets; they keep them.
- **Follow-ups once the budget is spent.** A pass whose fetch budget was spent still ran the follow-up querygen and a search. The remaining budget is now checked first, from wave 2 on, in every pass.
- **Drain before rerank.** A drain requested during a pass's opening search was followed by a rerank call. The drain is now read before every pick.
- **Credibility.** URL heuristics no longer give the curated prior: official docs by heuristics get 2, and 3 stays for the curated primary tier. User-content hosts never match. The acceptance rule "official docs `authority ≥ 2`" still holds: a docs host gets 2 from its prior, and a project's GitHub Pages site gets 2 from the digest's `official_doc` label (`tests/test_credibility.py`). The stale comment that said the arXiv flag stops grounding from counting a source now says what the flag does in v4.
- **`exa_cost_estimate_usd` (orchestrator ruling).** It is now `max_searches · price_per_search_usd + max_contents_pages · price_per_page_usd`: inline search text is not billed (measured, below).
- **`research.per_subq_target`.** It was never read. It is now the per-wave, per-sub-question selection quota described under the wave.
- **Ruled, no change:** `skip_followup_wave` keeps fetching pending candidates; recency on live Exa was verified by the orchestrator; the intel rungs' own redirects and JavaScript navigation stay a residual risk.

### Fix round 2: the rulings and what changed

The rulings are in §2.8 (`fetchers.fetch`, `980f5cd`), from fix round 1's concerns.

- **Own endpoints by host, not address.** Comparing by resolved address refused every other site on a shared address and port (a CDN, a tunnel edge). Ruling: normalised host and port only. A private endpoint address is refused by the private-address rule anyway, so a loopback endpoint reached by its IP is still refused; a public endpoint's own IP is no longer an own endpoint.
- **Proxies.** httpcore's proxy tunnel ignores `sni_hostname`, so behind an environment proxy a pinned HTTPS request went as an IP URL (`CONNECT <address>`) and failed TLS verification. And the proxy selection saw the IP URL, not the name, so a `NO_PROXY` entry naming the host did not keep it off the proxy. Ruling: when httpx's `trust_env` selection picks a proxy for the URL, the hop goes to it by name; otherwise the pin, on a client that mounts no proxy. The selection is httpx's own: `Client._transport_for_url`, the call `send()` itself makes, since httpx has no public one.
- **Quote cut at a word boundary.** `_digest_reply` cut a quote at its limit mid-word, and the token rule rejected an edited quote that ended in the fragment. It now cuts at the last whitespace at or before the limit.
- **Accepted as fix round 1 did them, no change:** an appended ref that was not fetched is marked `failed` and named to the user with the rungs tried; raw-deflate bodies fail the `native` rung.

### Fix round 3: the rulings and what changed

The scoped re-review confirmed every earlier finding fixed. It found one Important defect and three Minors in the fix diff, and showed the token rule weaker than the spec claimed. The rulings are in §2.8 (quote verification) and §2.4 (`ResearchConfig`), at `deaba46`.

- **The /append retry re-queues the ref (N1, Important).** `continue_cmd` marks an appended ref it could not ingest `failed` and tells the user to `/append` it again, but `add_run_source` was `INSERT OR IGNORE`, so the retry left the row `failed` and queued nothing. `src/argus/library.py` (an orchestrator-approved edit to a WP-D file) now upserts:
  - a `failed` row is set back to `pending`;
  - `pending` and `ingested` rows are unchanged;
  - the `/append` reply's pending count includes the retried ref.
- **Next-address fallback (N2).** A connect timeout on the first checked address (a blackholed IPv6 route) ended the rung. Now the next address is tried on a timeout as well as on a refusal, as the old connect by name did.
- **Own endpoints by IDNA-encoded name (N3).** `proxy。example。net` (and the `．` and `｡` forms) reached the proxy: the resolver, the `Host` header and the SNI all read those dots as dots, but the guard compared the raw string. The guard now compares the IDNA-encoded name, the same encoding the resolver and the request use.
- **Malformed Exa replies (N4).** A `/contents` reply of the wrong shape raised `TypeError`, `AttributeError` or `KeyError` out of `fetch()` and failed the node. It is now the rung's `malformed reply: …` error, described under the fetch chain.
- **Verbatim quotes.** The token rule is replaced by the verbatim rule described under the digest. Deleted: the 4-gram anchor index, `_span_ratio`, the token-set check, and `research.quote_verify_fuzzy` (the field and its bounds row in `run_config.py`, an orchestrator-approved edit). A stored run_config that still carries the key loads, the key ignored (pydantic's default), and an override naming it is an unknown key. `_word_cut` stays.
- **Drain before an Extend or a gap querygen.** Both passes called their querygen before `run_waves` first read the drain. They now read it first, so a drain already requested costs no LLM call.
- **httpx bound.** The proxy selection reads httpx 0.28 internals, so `pyproject.toml` bounds the dependency to `httpx>=0.27,<0.29`, the range python-telegram-bot pins too. The installed version is in range, so nothing was reinstalled.
- **Accepted, no change:** the private httpx internals (the same client decides and routes); one extra client per GET; ClientHello-level proxy tests; a public endpoint's own IP is fetched (ruled).

Known limits after the fix rounds:

- **Proxies.** Through an environment HTTP(S) proxy, the proxy resolves the name itself. The guard checks Argus's own resolution first, but the proxy's answer can differ: a rebinding resolver can answer the proxy differently, split-horizon DNS can give the proxy an internal address, and a proxy inside a private network can reach hosts Argus's resolver sees as public. That residual SSRF risk applies only when such a proxy is configured; none is configured on this host.
- **Raw deflate.** A `deflate` body is decoded as the zlib format HTTP specifies; a server sending raw deflate fails that rung with a `zlib.error`.

### Exa billing (Step 5, the one live search + one `/contents` call)

On 2026-10-19 a `/search` with `type: "auto"`, `numResults: 3` and `contents.text.maxCharacters: 2000` returned 3 results, all with text:

- `costDollars = {"search": {"neural": 0.007}, "total": 0.007}`, with no `contents` component.
- No result carried a `publishedDate`, and `resolvedSearchType` was empty.

A `/contents` call for one URL with `text.maxCharacters: 2000` returned:

- `costDollars = {"contents": {"text": 0.001}, "total": 0.001}`;
- `statuses = [{"id": <url>, "source": "cached", "status": "success"}]`.

Finding: **inline text pages were not billed**; the search price covered them. `/contents` pages are billed at the page price. This was measured with 3 results, not the configured `num_results`. Consequences:

- `RunConfig.exa_cost_estimate_usd` counts searches and `/contents` pages only (the orchestrator's ruling above).
- `exa_text_pages` stays a count of pages that came with text, not of billed pages.
- `exa_usd` is the reported cost, so the per-run spend is exact either way.

### Behaviour changes for handoff.md

- Research works without the intel-stack. On a host without both `INTEL_*` paths, the chain is `exa_text → native → exa_contents`.
- Fetching refuses private and loopback targets, including the proxy and the console (by host name and port). It connects only to the address it checked, or by name through an HTTP(S) proxy from the environment when one applies, and it stops a download at `fetch.max_bytes` decoded bytes; a body in an encoding other than gzip or deflate is refused.
- The depth presets drive research: the wave count, fetches per wave, the per-sub-question quota, the first-pass cap and the gap reserve. The Exa budget is per run, where v3 had a per-wave cap that every wave reset, and the per-run spend lands in `search_spend["exa_usd"]`. Each user Extend gets a fresh allowance.
- Appended (`/append`) sources are always ingested, outside every cap; one that cannot be fetched is marked `failed` and named to the user with the rungs tried, and `/append`ing it again queues it again.
- Duplicate URL forms (`www.`, tracking parameters, arXiv versions and mirrors, `?tab=` on GitHub) are fetched once.
- Evidence notes carry `source_type` and `authority`, and claims carry `quote_verified`; a quote that is not verbatim (a substituted, added or dropped word or number) is not verified, and an ellipsis may bridge what the quote leaves out.
- A run whose fetched sources all fail to digest stops with an error instead of shipping an ungrounded report. Individual digest failures are `[r<k>] skip:` lines.
- `extend_prep` no longer searches. The Extend pass's search happens inside research, and `research.extend_querygen` degrading is a `[r<k>] degrade: … → skip_extend_wave` line for the preview to surface.
- The grounding banner now needs `brief.entities` (WP-I). Until then it does not show.
- Cancel: research stops between fetches and returns what it has, with a `cancelled:` line.
- A DDGS failure is now visible in `errors`.
- The removed research env knobs (`ARGUS_RESEARCH_WAVES`, `ARGUS_FETCH_PER_WAVE`, `ARGUS_TOTAL_FETCH_CAP`, `ARGUS_EXA_MAX_CALLS`, `ARGUS_FETCH_CONCURRENCY`, `ARGUS_DIGEST_CONCURRENCY`) are no longer read; each logs one startup warning if set (`registry.REMOVED_ENV_KNOBS`). `README.md`, `.env.example` and `.env.mac.example` still describe them. That is WP-H's docs pass, so they are left unchanged here.

### New env vars

None. Removed: the knobs above. The SSRF guard reads `FREELLMAPI_BASE_URL` and `ARGUS_WEBAPP_PUBLIC_URL`, which already exist.

### Files outside WP-B touched (approved)

- `src/argus/engines/registry.py`: `summarize_calls` derives `exhausted_attempts` from the error-line text only for `legacy.*` roles. A real role's call site keeps every attempt record, so re-counting from the text would count them twice. The removed knobs are added to `REMOVED_ENV_KNOBS`.
- `src/argus/graph/nodes.py`: the wave deleted from `extend_prep_node` (ruling, option B). Its docstring and the non-append message follow, and the now-unused `ResearchBrief` import is gone. The `append_only` branch is unchanged.
- `bench/metrics.py`: imports `engines.urls.canonical_url` in place of its own copy, and drops that copy's constants and imports. The failed-fetch regex is unchanged, because research keeps the `fetch <url> failed` form.
- `scripts/demo_run.py`: `def fake_fetch(src, **kw):`, nothing else.
- `src/argus/graph/run_config.py`: `exa_cost_estimate_usd`, its formula and docstring only (fix round 1, the orchestrator's ruling); `research.quote_verify_fuzzy`, the field and its bounds row, removed (fix round 3, orchestrator-approved).
- `src/argus/graph/state.py`: `FetchedItem.rungs_tried` (fix round 1, orchestrator-approved).
- `src/argus/bot/history.py`: `_sources_from_refs` carries each ref, `_mark_appended`, and `continue_cmd`'s ingested bookkeeping (fix round 1).
- `src/argus/library.py`: `add_run_source` queues a `failed` ref again (fix round 3, orchestrator-approved).
- `pyproject.toml`: `httpx>=0.27,<0.29` (fix round 3).

### Test files touched

- New: `tests/test_urls.py` (the canonicalisation table, IPv6 brackets, malformed URLs, the v3 duplicates, `domain`, `registrable_domain`), `tests/test_ledger.py` (no overshoot and no lost updates under a thread pool, all-or-nothing `take(n)`, the first-pass / gap / Extend split, the quick preset, the snapshot round trip, unknown kinds, `count` on a capped kind), `tests/test_quote_verify.py` (rewritten in fix round 3 for the verbatim rule: every altered case from both review rounds fails; exact quotes pass, as do markup and links in the source, quote-mark / dash / punctuation / whitespace differences, a footnote marker, ordered-list markers and ellipsis segments in order; segments out of order or more than the gap apart fail; the §2.8 performance bound; the digest's word-boundary cut), `tests/test_fetchers.py` (Review Focus 3 and 7, every rung on fixtures, chain availability, the v4 `FetchedItem` fields; fix round 1: the pinned connection and a loopback server behind a rebinding resolver, the decoded byte cap, the WHATWG URL for every rung, typed rung errors, HTML at a `.pdf` path, a pinned `exa_contents` page; fix round 2: own endpoints by host and port, the hop through an environment proxy by name — a loopback fake proxy records the `CONNECT` line and the ClientHello's server name — and `NO_PROXY` keeping the pin; fix round 3: the next address after a timeout, Unicode-dot and fullwidth spellings of own endpoints, every malformed Exa reply shape, a billed reply's cost kept), `tests/test_recency.py` (the provider mapping against fake clients, `dropped_undated`, the resolve rules), `tests/test_gap_mode.py` (the gap wave and `gap_history`, the refusal, the Review Focus 5 drain, the events, the Extend pass and its degrade, the append-only pass; fix round 1: the pinned copy winning the dedupe in an append and an Extend pass, malformed URLs, no follow-up on a spent budget, no rerank after a drain; fix round 3: no querygen call when a drain precedes an Extend or a gap pass).
- Rewritten or extended (WP-B's):
  - `tests/test_search_providers.py`: the wave budgets, arXiv, `github_mode`, the DDGS errors line, dedupe, a malformed hit, the alias, the scout's querygen degrade, spend, a malformed seeded source and the pinned copy;
  - `tests/test_grounding.py`: entity phrases;
  - `tests/test_research_depth.py`: the grounding guards on phrases; rerank and its degrade; the `per_subq_target` quota (a dominating sub-question, leftovers by rank, the degrade path); the chunked digest; quote verification in the digest; the digest skip line; `ALL_SKIPPED_RAISES`; the first-pass caps;
  - `tests/test_credibility.py`: the official-doc prior, the authority scale, user-content hosts, the acceptance rule, and `note_authority`;
  - `tests/test_extend.py`: `extend_prep` only resets counters, the append-only flag stays for research, and the real `extend_prep` and `research_node` on a `MemorySaver` graph through a plain `/continue`;
  - `tests/test_tools_routing_and_errors.py`: the fetch-error propagation on the v4 chain (an unexpected rung exception propagates), with the exact failed-fetch line; its `_http_client` patch takes the pinned client's keyword (fix round 2);
  - `tests/test_continue_extend.py`: only fetched appends are marked `ingested`, a failed one is marked `failed` and named with its rungs; `/append` again re-queues it, on the real in-memory `Library` (fix round 3);
  - `tests/test_library.py`: a `failed` source is queued again by the next append, `pending` and `ingested` rows unchanged (fix round 3).
- `tests/helpers.py`: `request_url` and `as_stream`, for the MockTransport fixtures: a request now goes to the checked address with the original `Host`, and the chain reads the raw stream, which httpx pre-reads for a Response built from bytes.
- Other WPs' tests, minimal edits for this change:
  - `tests/test_continue_extend.py`: the plain-continue assertion moves to where the wave now runs (it checks that research sees an Extend pass); `_parallel_fetch` keyword arguments; the digest stub's return shape; `run_config` in the append-only node test.
  - `tests/test_run_config_resume.py`: the `followup_queries` / `run_query_wave` patches for `extend_prep` removed.
  - `tests/test_run_config.py`: the estimate's expected values follow the ruled formula; the `quote_verify_fuzzy` bounds row removed, and a stored run_config still carrying that key loads (fix round 3).
  - `tests/test_reflexion.py`: seam signatures (`**kw`), the 3-tuple returns, `run_config`.
  - `tests/test_graph.py`: `fake_fetch(src, **kw)` and `run_config` in the deep test's input.
  - `tests/test_tools_truth.py`: `_parallel_fetch` keyword arguments; the stealth test drives the intel rungs through `engines/fetchers`.
  - `tests/test_web_search.py`: `test_ddgs_failure_raises`.
  - `tests/test_engines_registry.py`: a real role's `exhausted_attempts` is 0.
  - `tests/test_llm_facade.py`: the risk-9 test runs through `research.digest` on the facade.

### Verification evidence (2026-10-19)

The first cut's logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t7\`, fix round 1's in `…\scratchpad\t7fix\`, fix round 2's in `…\scratchpad\t7fix2\`, fix round 3's in `…\scratchpad\t7fix3\`. Every suite run used `FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1`. No LLM call left the machine, and no bot was started, stopped or restarted.

- RED (`t7\red.log`), with the tests written and no implementation:
  - the files importing the new modules failed at collection (`ImportError: cannot import name 'fetchers' / 'search' from 'argus.engines'`);
  - the rest failed on the missing v4 behaviour: gap mode, the drain and the events, entity-phrase grounding, rerank, the chunked digest and its quote checks, `ALL_SKIPPED_RAISES`, the first-pass caps, the credibility prior and the authority scale, and `extend_prep` still searching.
- Fix round 1, RED (`t7fix\red-old-code.log`): the fix round's test files run against `src/` at the first cut (`git archive`, `pytest -o pythonpath=…`); every test that demonstrates a review finding fails there.
- Fix round 2, RED (`t7fix2\red-da8762f.log`): its new tests run the same way against fix round 1's `src/`; each fails on the defect it names.
- Fix round 3, RED (`t7fix3\red-deaba46.log`): its new tests against fix round 2's `src/`, each failing on its defect. The new quote cases were also run through the old fuzzy rule (`t7fix3\old-rule.log`): it verified five altered quotes (the four the re-review named, and a dropped word) and rejected two legitimate ellipsis quotes. Switching off a normalisation step or the segment gap flips the case that guards it (`t7fix3\mutations.log`).
- The re-reviewer's probes (`t7-rereview\`) ran before and after fix round 3, under their own offline harness (`t7fix3\before.log`, `t7fix3\after.log`). The two quote probes ran adapted to the verbatim rule's API only (`t7fix3\adapt_probes.py`).
- GREEN: the focused files pass, and so does every file any round touched.
- The full suite (`t7\full-suite.log`, `t7fix\full-suite.log`, `t7fix2\full-suite.log`, `t7fix3\full-suite.log`) passes except the live-proxy tests (`tests/test_llm.py` and the `tests/test_graph.py` fixture), which fail on the closed port by design, as in `t7\baseline-suite.log` at the base commit.
- The reviewer's probes ran before and after the fix round (`t7fix\before-*.log`, `t7fix\after-*.log`). The MockTransport probes route by `str(request.url)` and hand httpx pre-read bodies, so after the fix they ran with only that harness adapted (`t7fix\adapt_probes.py`).
- A scratch smoke ran the node with `research.rerank` on (the preset default), the real fetch chain on a mock network and a scripted `llm.complete` (`t7\smoke_rerank_node.py`, not committed). It showed one rerank call, the fetches in rerank order, a 503 reported as its fetch line, `authority` 2 from the `official_doc` label, `quote_verified` on every claim, and the events in order.
- Exa (`t7\exa-billing.log`, `t7\exa_billing_probe.py`): the one search and the one `/contents` call above. Field names and values only; the key was never printed.
- The WP-B acceptance runs are the orchestrator's: one q010 medium/standard `live_check`, one `short` run with the `INTEL_*` paths unset, and `bench.compare` against the v3 baseline.
