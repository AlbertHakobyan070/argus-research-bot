# WP-C — Trust: compose assembly, batched judging, panel policy, cut-not-flag, report builder, evidence.json (Task 9)

Spec: `docs/argus-v4-design.md` §3 WP-C, with the contracts it names: §2.6 (`evidence.json`), §2.8 (assembly and `normalize_cite_markers`, judging R1, cutting B1, the TL;DR R2, the gap signal R14, the precision scan, the round tag), §2.3 (the per-role failure policies and the ALL_SKIPPED_RAISES rule) and §2.4 (the per-round counters). Built on `0c6cee6`, which carries the Task 9 rulings (the `exa_usd` writer, the skipped section writer, no inline markers).

### What changed

- `src/argus/graph/assemble.py`: `normalize_cite_markers` is real. `【n】`, `［n］`, `〔n〕`, `【n†…】` (and lists inside them), `[n, m]` / `[n; m]`, ranges `[1-3]` / `[1–3]` / `[1—3]`, `[^n]` and `[Source n]` / `[Sources n, m]` become `[n][m]…`. A range is expanded only when it is at most 20 wide (`[1-40]` and `[2019-2020]` stay as written); code spans and fences are left alone. The module docstring now gives the right reason for the flat lengths.
- `src/argus/graph/compose.py` (rewritten on `llm.complete`):
  - **Notes**: `render_notes(evidence, note_ids, *, max_chars, pinned)` renders the notes the outline allocated to a section plus the local and pinned ones (pinned first, then by relevance), over `outline.usable_notes` (relevance ≥ `research.writer_min_relevance`, a claim not failing quote verification); a claim with `quote_verified is False` is never shown. `note_ids=None` (the flat writer) means every usable note.
  - **What is written**: on a panel rewrite (`panel_verdict.next == "compose"`) the `revise_sections`; otherwise `outline.changed_sections` (all sections when the key is missing); a section with no previous version always. Every other section is reused byte-identical (md and `note_ids`), with its findings reset to `kept`; stale "Analysis" findings are dropped (the analyst rewrites them); all findings are re-ided `f0…` in outline order. `compose_runs` grows by one per run.
  - **Skips** (`compose.section`, `skip_item`), the R11 ruling: a skipped first write merges its notes into its nearest surviving neighbour (the previous section, else the next), which is written once more with the union of both note sets. If that re-write is skipped too, the neighbour keeps its version and `[r<k>] compose: section "<title>" dropped — notes <ids> not written (its merge into "<neighbour>" was skipped)` is recorded. A skipped re-write keeps the previous version. A reply under 40 characters is a skip (`[r<k>] skip: compose.section <title> (empty reply)`, its record marked failed). Every section skipped → `EngineError("compose: every section was skipped")`.
  - **Flat lengths**: one `compose.flat` call (`raise`); `FLAT_SYSTEM` no longer asks for `## TL;DR` / `## Key Findings` — the body goes under `assemble_draft`'s `## Findings`. An empty reply raises. A panel rewrite that does not target "Report" (it flagged only the analysis) reuses the body.
  - **Extraction**: `extract_findings(title, md, fetched, *, engines, rnd)` (public; the analyst uses it too) on `compose.findings_extract`: one call per written section, on the compose pool. Each finding carries the verbatim `sentence` (markers normalised), `source_ids` taken from the sentence's markers (the model's list only when the sentence has none), `status: kept`; a finding citing nothing is dropped, one citing only unknown ids is kept for report_builder's cut. A skipped call is `[r<k>] skip: compose.findings_extract <title> (…)`. Every attempted extraction skipped and no finding left → `EngineError` (ALL_SKIPPED_RAISES).
  - **Dedupe** (`compose.dedupe_pass`): a finding of a section written in this run whose claim nearly repeats (bench `_norm`, `difflib` ratio ≥ 0.85) a finding of an earlier section is cut from its section (its whole sentences, `quarantine.cut`) and dropped — unless its sentences also state a non-duplicate finding of that section, or cannot be located. The first occurrence wins; reused sections are never touched.
  - **Revision notes**: the user's revise notes newest first, then the panel's notes on a panel rewrite, at most 8.
  - `rc.compose.concurrency` replaces `ARGUS_COMPOSE_CONCURRENCY`; the LLM TL;DR writer (`_write_tldr`) is gone. `section` events: `{title, ok, chars, reused}`.
- `src/argus/graph/precision.py` (new): `scan(md)` — the uncited sentences (no `[n]`) with a digit or a currency sign, from the cuttable sentences (headings, code, tables and quotes hold none); `render_appendix(items)` → `## Uncited specifics`. No inline body markers.
- `src/argus/quarantine.py` (evolved; the v3 string quarantine `quarantine_flagged_claims` and its tests are removed with report_builder's legacy branch):
  - a block reader (paragraphs and list items hold the cuttable sentences; headings, code fences, tables, block quotes and rules never) and a sentence splitter (a terminator plus closing quotes/brackets and trailing `[n]` markers, followed by whitespace and a capital, a digit, a quote or markup; abbreviations and single-letter initials do not split);
  - `locate(text, md)`: normalised containment, else the best `difflib` ratio ≥ 0.85 over a window of 1–3 adjacent sentences (the smaller window on a tie) — the probe's condensed f6 is located over its two-sentence window;
  - `cut(md, indices)`: whole sentences only; a paragraph or item left empty is dropped, and so is a heading left with nothing under it; an untouched body comes back byte-identical;
  - `apply_cuts(findings, mds, flagged, n_sources)`: statuses recomputed from scratch — a flagged finding, or one whose ids all fall outside 1..n (reason `unknown source id (cites …)`), is `cut` when its `sentence` (its claim for a legacy finding) is located in its section or the analysis, else `unlocated`; rows deduplicated by span, unknown `[n]` markers stripped from the row text;
  - `render_cut_appendix` (`## ⚠ Flagged claims (removed from the body)`) and `render_unlocated_appendix` (`## ⚠ Flagged but not located (still in body)`).
- `src/argus/graph/panel.py` (rewritten):
  - **Grounding** (`panel.grounding`): findings grouped by section, split into chunks of ≤ `judge_batch_size` and packed in order; each prompt holds the batch's findings (`[id] (section: … · cites [n]…) sentence`) and exactly its cited notes — at most 3 claims per note, chosen by word overlap with the batch, quotes up to 250 chars, failed-quote claims left out. A prompt over 24 000 chars (system + human) is halved until it fits; a single finding still over it is skipped (`grounding batch <i>: input over the 24000-char budget`). Batches run on a pool of 3. The reply must carry a `flagged` list (`{"flagged": [{"id", "reason"}], "notes": [...]}`); a skipped, failed or unparseable batch is unjudged — `skipped_judges` gets `grounding batch <i>: <reason>` and `errors` the `[r<k>] skip: panel.grounding batch <i> (…)` line. `judged`, `total`, `judge_verdicts["grounding"]` (`pass` / `revise` / `unjudged` / `off`) and `judge_verdicts["flag_reasons"]` are recorded. `judge` events per batch.
  - **Diversity**: when a judge batch was served by a model of a writer's family (the ok `compose.section` / `compose.flat` / `compose.analyst` records), the batches are rerun once on the binding's first engine of another family (a snapshot copy with that engine as the only one; records tagged `retry_reason="diversity_rerun"`). The rerun's verdict replaces the first when it judged as much, and `judge_not_diverse` reports whether it is still same-family; no candidate, `diversity_rerun` off, or a failed rerun → the first verdict, `judge_not_diverse = True`, no skip line.
  - **Coverage** (`panel.coverage`): `{verdict, weak_sections, missing_sub_qs, notes}`; a skip is `[r<k>] skip: panel.coverage report (…)` and `coverage: <reason>` in `skipped_judges`.
  - **Precision**: `precision.scan` over the sections and the analysis → `uncited_specifics` (`panel.precision_mode == "deterministic"`).
  - **Decision** (`decide_next`, `gap_signal`): `next` is `research` while an R14 gap remains and `gap_waves < gap_waves_max` (and `coverage_to_research`); else `compose` when a rewrite is wanted — flagged / judged ≥ `rewrite_flag_ratio`, or a weak section — and `revision_rounds < max_compose_revisions`; else `report_builder`. `revise_sections` are the flagged findings' sections and the weak ones, in section order ("Analysis" excluded: the analyst re-runs after every compose). `revision_rounds` grows only on a rewrite; `coverage_gaps` (state) only when going to research. The panel no longer appends to `revision_notes`; its notes are `<section>: "<claim>" is not supported by its sources — <reason>`, then the judges' notes. The v2 `review_verdict` is still written. `route_after_panel` returns `next`; `ARGUS_MAX_REVISIONS` is no longer read.
- `src/argus/graph/nodes.py`:
  - `intake_node` on `intake.classify` (degrade → mode deep, the raw request, `[r<k>] degrade: intake.classify → default_classification (…)`, a `degrade` event); `quick_answer_node` on `quick.answer` (`raise`).
  - `revise_prep` and `extend_prep` reset `compose_runs` too; `revise_prep` marks every outline section changed.
  - `report_builder_node` (the legacy string branch and `post_synthesis` are gone): the cuts over the sections and the analysis; the TL;DR; `assemble_draft(…, appendices=[cut, unlocated, uncited], tldr_heading="Executive TL;DR" for lecture)`; the title block with three new lines; the grounding banner, then the "⚠ Not fact-checked" and "⚠ Degraded run" banners as `> ` lines; the citation-integrity passes (they raise now, never skipped); `## Degradation notes` after Sources; report.md, report.pdf, evidence.json, metadata.json. A PDF failure is `[r<k>] pdf: <Type>: <msg>`, a degradation note and a second write of report.md. It returns the findings with their status and `tldr_md`.
- `src/argus/graph/report_builder_helpers.py`: `info_lines` (`_Engines: <profile> · reroute <x %> · judge diversity ok|⚠ · judged <j>/<n> · warnings <w>_`; `_Recency: past month (sources published since …)_` when a window was applied; `_Research: <n> sources · coverage min <a> / mean <b> · gap waves <g> · compose runs <c>_`), `not_fact_checked`, `degraded_run`, `build_tldr`, `render_degradation_notes`; the sidecar gains `run_config` (without `engines.roles`), `evidence_path`, `degradations`, `degraded`, `exa_usd`, and takes `engines` (= `summarize_calls(…, exa_usd=…)`). The stale "intel-stack Chromium" docstring line is fixed.
- `src/argus/graph/evidence_export.py` (new): strict pydantic models (`extra="forbid"`, `schema` as an alias of `Literal[1]`) for every §2.6 block, and `write_evidence(path, state, *, findings, errors, degraded, tldr, engines, generated_at)`, which validates its own document before writing it.
- `src/argus/graph/graph.py`: `compose → analyst → panel`; `panel → research | compose | report_builder`.
- `src/argus/graph/synthesis_modes.py`: the docstring names `outline._MODE_SECTIONS` and the v4 flat body.
- `src/argus/post_synthesis.py` and `tests/test_post_synthesis.py`: deleted. Nothing imports the module; the only other mention is a docstring in the frozen `graph/state.py` (`Finding`).

Approved edits outside WP-C's files:

- `src/argus/graph/insight.py` (WP-I): the analysis findings come from `extract_findings` (a skipped extraction is the declared `skip:` line, with the round tag); an analysis with fixed headings and empty bodies takes `omit_analysis (no analysis content)`.
- `src/argus/engines/registry.py` (WP-A): `ARGUS_MAX_REVISIONS` and `ARGUS_COMPOSE_CONCURRENCY` in `REMOVED_ENV_KNOBS`.
- `src/argus/tools.py`: the base14 font fallback is deleted; a missing bundled font raises `TTFError` and nothing is cached (Q2).
- `src/argus/bot/research.py` (WP-E1): `_resume_after_plan` writes `runs.exa_usd` from `search_spend.exa_usd` at every report gate (Q1).
- `scripts/live_check.py` (WP-G): the v4 banner rows, "report.pdf exists", evidence.json validated by `Evidence`, `_kept` through `metrics._findings`.

### Decisions where the spec is silent

- **Dedupe is cross-section only** (as `cross_section_duplicate_ratio` measures it). A repeat inside one section is the writer's style, and its sentence often backs another finding.
- **The merge neighbour is the nearest surviving section** (previous first). The outline itself is not rewritten: the merged-away title stays in the outline with no section, so a later compose of the round treats it as new and tries it again.
- **The flat writer's notes are every usable note** (`note_ids=None`): flat lengths have no outline to allocate.
- **Extraction ids**: the sentence's markers win over the model's `source_ids` (the sentence is what the reader sees and what the cut locates).
- **The rewrite ratio's denominator is `judged`**, so a half-skipped judge cannot dilute a real flag rate.
- **A judge reply without a `flagged` list is unparseable** — never "nothing flagged" (the v3 failure mode).
- **The diversity rerun reruns every batch** and counts only when it judged at least as many findings; its events are not re-emitted.
- **report_builder recomputes the uncited specifics over the cut body** (the reader's text); the panel's `uncited_specifics` is the pre-cut metric.
- **TL;DR**: with an analysis, only its takeaways count (no fallback to section sentences — "nothing kept → omitted"), in the analyst's order; without one, the first kept finding of each section whose sentence is still located in the cut section. A run without sections (no usable evidence) has no TL;DR and no omission line.
- **`degraded`** = the degraded-run banner or any unjudged finding. Grounding switched off shows "⚠ Not fact-checked — the grounding judge is off" but is a configuration, not a degradation.
- **The degraded-run banner** counts over the whole run, per skippable role (`research.digest`, `compose.section`, `compose.findings_extract`, `panel.grounding`, `panel.coverage`): items = its ok records (diversity reruns excluded) plus its `skip:` lines; the reason is "unparseable", else the 429/5xx code, else the error type.
- **Degradation notes** show the bench's `skip:` / `degrade:` / `legacy:` lines plus `pdf:`, `tldr:`, `cancelled:` and `compose:` (a merge that lost notes) — the visible losses. `analysis:` lines (pointer 20) are informational: they stay in `errors` and evidence.json `errors`, not in the notes, `warnings` or `degradations`. evidence.json `degradations` and metadata.json `degradations` are exactly the bench set.
- **The citation-integrity passes run over the report up to Sources**, not over the degradation notes, the footer or the lecture appendix: a note naming a source that could not be digested keeps its URL.
- **The title block's `findings:` count is the kept findings.**
- **WP-0 decision 6 is superseded**: the flat writer no longer emits headings of its own, so the flat path is on `assemble_draft` (`## Findings`). (Its stated reason was also off for `tldr`: `TLDR_SYSTEM` always wrote a bare paragraph; only `FLAT_SYSTEM` — `short` and one-sub-question briefs — emitted `## TL;DR` / `## Key Findings`.)

### Behaviour changes for handoff.md

- Flagged and unknown-id sentences are removed from the report (whole sentences), listed in "## ⚠ Flagged claims (removed from the body)"; a flag that cannot be located stays and is listed under "## ⚠ Flagged but not located (still in body)". The v3 report kept them in the body under an appendix that said they were removed.
- The TL;DR is compiled from the kept takeaways (or the first kept finding per section), labelled; no LLM TL;DR. Lecture reports keep "Executive TL;DR".
- The analyst runs between compose and the panel, so medium and longer reports (and short, unless `compose.analyst` is off) get the "## Analysis" section.
- The panel can send the run back to research for a gap wave; it rewrites only for a real flag rate or a weak section, at most `panel.max_compose_revisions` times per round.
- New report surface: the engines / recency / research lines, the "⚠ Not fact-checked" and "⚠ Degraded run" banners, "## Uncited specifics", "## Degradation notes", Sources over the appendices, the flat body under "## Findings".
- `evidence.json` next to every report; metadata.json gains `run_config`, `evidence_path`, `degradations`, `degraded`, `exa_usd`; the library's `runs.exa_usd` is written at each report gate. A library failure on that write surfaces as "⚠️ resume failed: …" (it is not swallowed like the status write).
- `ARGUS_MAX_REVISIONS` and `ARGUS_COMPOSE_CONCURRENCY` are no longer read (one startup warning each if set). README still documents `ARGUS_MAX_REVISIONS` (WP-H's docs).
- A missing bundled font fails the PDF (report.md still ships, `report_paths["pdf"]` is None, a `pdf:` line and a degradation note) instead of rendering in Times.
- An intake engine failure degrades to a deep run on the raw request (visible `degrade:` line); a quick answer whose engines all fail raises.

### For WP-E1 (Task 10)

- The preview quality line should repeat the banners: they are the report's `> ⚠ …` lines, and evidence.json has `degraded`, `panel.judged` / `panel.total` / `panel.skipped_judges` and `tldr`.
- New events: compose `section` `{title, ok, chars, reused}`; panel `judge` `{name: "grounding", batch, verdict: "pass" | "<n> flagged" | "skipped", skipped}` and `{name: "coverage", verdict, skipped}`; intake `degrade`.

### New env vars

None. Removed: `ARGUS_MAX_REVISIONS`, `ARGUS_COMPOSE_CONCURRENCY` (registered in `registry.REMOVED_ENV_KNOBS`).

### Test files touched

- New: `tests/test_precision.py`, `tests/test_panel_policy.py`, `tests/test_judge_batches.py`, `tests/test_evidence_export.py`, `tests/test_cut_not_flag.py`, `tests/test_compose_reuse.py`, `tests/test_intake_quick.py`.
- WP-C's: `tests/test_assemble.py` (the marker table), `tests/test_quarantine.py` (the v4 splitter/locator/cutter; the string-quarantine tests removed with it), `tests/test_reflexion.py` (the v4 loop: flag ratio, analyst, `compose_runs`), `tests/test_grounded_gate.py` (stub the analyst), `tests/test_graph.py` (`run_config` in the live tests' state); `tests/test_post_synthesis.py` deleted.
- Approved: `tests/test_insight.py` (the real extractor's contract, the skipped Analysis extraction, list markers, the label-only analysis), `tests/test_live_check.py` (the new rows, a valid `Evidence` fixture), `tests/test_pdf_render.py` (`test_a_missing_font_dir_raises`, the tagged `pdf:` line, `run_config`).
- Minimal fixes for breakage caused by this change: `tests/test_continue_extend.py` and `tests/test_hitl_edit_revise.py` (WP-E1's) and `tests/test_run_config_resume.py` (WP-D's) stub the new `analyst_node` like every other node they stub; `tests/test_llm_facade.py` (WP-A's) gives its report_builder call a `run_config`.

### Verification evidence (2026-10-20)

Logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t9\` and next to it (`t9_baseline*.log`). Every run used `FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1`; no bot was started.

- Baseline (`0c6cee6`): only the four live-proxy tests (`tests/test_llm.py` and the `tests/test_graph.py` fixture) fail.
- RED (`red.log`, `red_summary.log`), tests written, no implementation: the new test files fail at collection on the missing modules and names; the marker table, the judge batches, the insight contract, the reflexion loop, the PDF and the graph-stub tests fail.
- GREEN (`green.log`): every touched test file passes.
- Full suite (`full2.log`): only the same four live-proxy tests fail.
- A q013-like fake run (`q013_fake.py`, `q013_fake.log`): see the task report.
