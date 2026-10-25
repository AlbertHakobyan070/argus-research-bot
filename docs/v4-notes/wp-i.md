# WP-I — Insight: archetypes, outline (fresh + extend), analyst, confidence labels (Task 8)

Spec: `docs/argus-v4-design.md` §3 WP-I, with the §2.8 node-level contracts it names (allocation, the analyst's input R12, the confidence label R13), §2.1 (the `brief.draft` / `outline.plan` / `compose.analyst` policies), §2.3 (the facade and its `validate` hook) and §2.5 (custom events). The brief now writes every v4 field, and the outline allocates evidence notes to sections. `insight.analyst_node` exists and is tested, but it is not in the graph yet: WP-C wires it (it owns `graph/graph.py`). The outline is live at once, through compose's re-export.

### What changed

- `src/argus/graph/archetypes.py` (new, plain data, nothing callable): `TEMPLATES[archetype] = {sectioned: [(title, focus)], flat: {body_emphasis, analyst_emphasis}, analyst_sections: (...)}` for `explainer`, `definition`, `comparison`, `forecast`, `sota_review`, `howto`, `news_update`, `decision`; `DEFAULT_ARCHETYPE`; the analysis heading names (`TAKEAWAYS`, `AGREEMENT`, `SCENARIOS`, `IMPLICATIONS`, `OPEN_QUESTIONS`). Only `forecast` and `decision` list the scenarios heading. The sectioned titles follow the first committed spec (`29203e2`, "as Phase C").
- `src/argus/graph/brief.py`:
  - The prompt asks for `archetype` (with a one-line description of each), `time_sensitive`, `recency_hint` (`d|w|m|y|null`), `is_academic_topic` and `entities`. Entities are defined as proper names only (products, models, organisations, people, projects, places), never a date, period, price, metric or generic noun, with the counter-examples "Bitcoin price outlook for Q4" → `["Bitcoin"]` and "LangGraph vs CrewAI" → `["LangGraph", "CrewAI"]`. The JSON example shows placeholders (`"<one of the archetypes above>"`, `<true|false>`), not values.
  - `brief.draft` runs through `llm.complete`, and `_parse_brief` is its `validate` hook. An unparseable reply is now an attempt: it is recorded, nudged and retried, then the next fallback. v3 dropped the successful call's record when parsing failed afterwards. A `sub_questions` value that is not a list is unparseable too.
  - Defaults when a field is missing or malformed: an unknown archetype → `explainer`; booleans only from `true` (or the string `"true"`); a recency hint is passed on lower-cased (`resolve_recency` ignores an unusable one with its note), `""`/`null`/`none` → `None`.
  - `entities`: each phrase lower-cased and whitespace-collapsed, kept only when the request, the plan feedback or the previous brief's entities name it (so an entity named in an earlier plan edit survives the next edit). The containment test normalises both sides the way `grounding.normalise` does. A phrase whose every normalised token is a period or a number (`q1`–`q4`, `h1`/`h2`, `fy` + 2–4 digits, a number, a month name or its 3-letter abbreviation) is dropped; numbers inside a name stay ("GPT 4", "GLM 5.2"). Deduplicated by normalised form, at most six.
  - `_fallback_brief` is gone. `brief.draft` must raise (spec §2.1), so exhausted engines raise `EngineError` on the first draft and on the re-draft. v3 kept the first brief when the re-draft failed (log only), and served a generic 3-angle brief when parsing failed.
- `src/argus/graph/outline.py` (new): `outline_node`, `default_outline`, `OUTLINE_SYSTEM`, `LECTURE_HINT` and `_MODE_SECTIONS` moved here from `compose.py`. Then:
  - **Usable notes** (`usable_notes`): a claim whose `quote_verified` is not `False`, and relevance ≥ `research.writer_min_relevance`; an appended local file or a pinned source at any relevance. Only these are indexed, allocated and counted.
  - **Fresh mode** (no outline sections yet): the prompt carries the brief, the evidence strength per sub-question, the archetype's sectioned shape as a hint ("at most N sections", the length's budget; a lecture gets `LECTURE_HINT` and the shape "inside the parts") and the note index (`[id] domain · relevance · sub-questions — first claim`, bounded). The reply is `sections[*].note_ids`, and `validate` needs at least one titled section. A repeated title (case and whitespace aside) merges into its first occurrence — sub-questions and notes unioned in order, the first focus kept — before the sections past the budget are dropped. `changed_sections` lists every section.
  - **Repair**, after either mode and after the degradation:
    - unknown ids and repeats are dropped, and a note keeps its first two sections;
    - sub-questions no section lists join the last section (the v3 rule);
    - every unallocated usable note goes to the section sharing most sub-questions with it, ties rotating over the tied sections (only a tie advances the rotation);
    - a section with fewer than `compose.section_min_notes` notes merges into the previous section (the next, for the first). Its sub-questions, notes and focus move and its title goes. A lone section stays.
  - **Extend mode** (an outline exists and usable notes arrived that no section holds — a gap wave or a user Extend). The prompt shows the existing sections (title, sub-questions, note ids, focus), the sub-questions with their strong-note count before the wave (STARVED below `research.coverage_min_strong`) and only the new notes. The reply `{"assign": [{note_id, section}], "new_section": {...} | null}` is read under these rules:
    - a repeated title of the existing outline (one written before titles were deduplicated) first merges into its first occurrence, which counts as changed when it gained notes;
    - assignments name a new note and an existing title (case and whitespace aside), else they are ignored;
    - the new section is accepted only with new-note ids, a starved sub-question and a title not already used;
    - a new section left under `section_min_notes` merges into the last existing section, so no existing title ever moves;
    - `changed_sections` lists the sections whose note set changed, the new one included.
    - No new usable note: no call, the outline unchanged, `changed_sections` empty.
  - **Degrade** (`outline.plan` → `one_section_per_subquestion_roundrobin_notes`): `default_outline` in fresh mode, the existing sections in extend mode, then the same repair. The line is `[r<k>] degrade: outline.plan → one_section_per_subquestion_roundrobin_notes (<reason>)`, and the node emits a `degrade` event.
  - Flat lengths (`tldr`, `short`) and one-sub-question briefs return `{"sections": [], "changed_sections": []}` without a call.
- `src/argus/graph/compose.py`: the outline block is removed, leaving one re-export line, `from .outline import OUTLINE_SYSTEM, default_outline, outline_node` (the names in compose's `__all__`; `graph.py` imports `outline_node` there).
- `src/argus/graph/insight.py` (new):
  - `stance_map(brief, evidence)`: per sub-question `str(k)`, the count of `supports` / `contradicts` / `mixed` / `background` notes and `contradicting_sources` (ids, ascending). Every note counts; an unknown stance counts as background.
  - `confidence_label(source_ids, sources, notes, takeaway_text, stance_map) -> (label, reasons)`, the R13 rule:
    - D = distinct registrable domains (`urls.registrable_domain` of the fetched URL) among the cited notes of authority ≥ 1;
    - V = a cited note of authority ≥ 1 holds a `quote_verified is True` claim sharing ≥ 2 non-stop words with the takeaway (citation markers removed; a small English stop list) — authority-0 notes count for neither D nor V (spec ruling `f9c255f`);
    - A = the highest cited authority;
    - high, medium and low as in §2.8;
    - a contradicting note on the cited notes' sub-questions caps high at medium and adds `contradicted by [n]…`;
    - reasons: `"<D> independent source(s)"`, `"quote verified"` / `"no verified quote"`, `"authority <A>/3"`, and the contradiction;
    - a cited id with no note (or no fetched item) is not a source; none left → `("low", ["no known source cited"])`.
  - `analyst_node(state, runtime=None)`:
    - **Skips** (`analysis: None`, no call): `tldr`, `compose.analyst` off, or no findings.
    - **Input** (R12): the per-section findings (`- claim [ids]`, by section, bounded), the stance-map lines, the coverage, and a source index (`[id] title — registrable domain · authority · published`) of the notes the findings cite plus the contradicting ones. Never the section bodies. The prompt also names the archetype, the success criteria and the headings to write, each with its guidance. The short variant adds the flat `analyst_emphasis` and the caps.
    - **Headings**: the archetype's `analyst_sections`. The short variant drops Implications, and drops agree/disagree unless a sub-question has a contradicting note.
    - **Post-processing**, all deterministic:
      - the headings are parsed at levels 2–4, in case, `&`, trailing-colon and leading-numbering (`1.`, `2)`) variants, a line of only bold text (`**Key takeaways**`, colon optional) is a heading when it names a fixed heading, `Takeaways` is the takeaways heading, and a repeated heading continues its first section;
      - everything before the first fixed heading — plain lines and other headings with their text — is dropped and recorded as `analysis: dropped text before the first heading`; other headings after it stay;
      - takeaway bullets without `[n]` move to "Open questions and limits" as `- (uncited) …` (the section is created when missing); a takeaways part with no cited bullet is recorded as `analysis: no cited takeaways`;
      - the short variant keeps at most 4 takeaways and 3 open questions (relocated ones included) and drops the headings it does not use; a trim is recorded as `analysis: short variant kept <k> of <n> takeaways, <q> of <m> open questions`;
      - each takeaway gets its label;
      - the markdown is rendered under `###` headings, with the label line "_Analysis — interpretation by Argus …_" first;
      - fixed headings without the takeaways heading are recorded as `analysis: takeaways heading missing`, and the text is kept.
    - **Findings**: extracted by compose's `_extract_findings("Analysis", …)` over the analysis without the labels, and skipped when it cites nothing. They are merged after the other findings and the whole list re-ided. Earlier "Analysis" findings are replaced, and are dropped on every path that omits the analysis.
    - **Output**: `analysis = {md, takeaways: [{text, source_ids, confidence, reasons}], stance_map, has_disagreement_section, has_scenarios, archetype}`; the interim `draft_md = assemble_draft(tldr=None, sections, analysis_md, fetched, flat=<no outline sections>)`. No TL;DR.
    - **Degrade** (`compose.analyst` → `omit_analysis`): `analysis: None`, the attempts in `model_calls`, `[r<k>] degrade: compose.analyst → omit_analysis (<reason>)`, and a `degrade` event. An empty reply takes the same degradation, with the reason `empty reply`, and so does a reply with none of the fixed headings, with `no analysis headings`.
    - **Events**: `analysis` `{takeaways, has_disagreement, has_scenarios}` on success.
- `bench/metrics.py` (orchestrator-approved edit): `cross_section_duplicate_ratio` leaves out findings whose section is `"Analysis"`. The analyst restates evidence-section findings by design; the metric measures duplication between evidence sections.

### Decisions where the spec is silent

- **Entities must be named in the request** (or the plan feedback, or the previous brief — spec ruling `f9c255f`). The grounding check requires every entity in at least the threshold's number of credible documents, so an invented abbreviation or expansion would fail a grounded run. The filter only removes phrases, so it errs towards no warning, never towards a false one. A phrase that names a thing plus a generic noun ("Bitcoin price") is the prompt's job (proper names only, with counter-examples); only period/number-only phrases are dropped deterministically.
- **What a note is, for allocation**: the notes the WP-C writer will read (relevance ≥ `writer_min_relevance` and a claim not failing verification; pinned and local always). Otherwise junk notes would count towards `section_min_notes`.
- **Ties rotate.** This is the "round-robin" of the degradation, and it is also the tie-break of the normal repair: one allocation function for both.
- **The v3 orphan rule stays.** A sub-question no section lists joins the last section. WP-B's gap-history `source` reads ownership by `sub_qs`, unchanged.
- **The "starved before the wave" coverage** is `compute_coverage` over the evidence without the new usable notes.
- **`definition` has a sectioned variant.** The first spec called it "flat only"; length wins and the WP-I test requires both variants.
- **Analysis headings are `###`**, because `assemble_draft` adds `## Analysis` above them.
- **Compaction keys on length `short`.** A medium run with one sub-question (a flat body) keeps the full analysis. The interim draft's `flat` follows the outline, as compose's flat path does.
- **Order of the deterministic steps**: drop the text before the first fixed heading, relocate uncited takeaways, then compact (so the open-question cap can drop a relocated one), then label, then extract.
- **R13 details**: the authority-0 exclusion applies to D and V (the review round's ruling, spec `f9c255f`); A reads every cited note. Local `file:///` sources share one registrable domain (`""`), so they count once.
- **An empty analyst reply** is recorded as the declared `omit_analysis` degradation, so it is counted and shown with the other degradations, not as a free-form error line.
- **The plan dict** does not carry the archetype or recency. The effective recency window is in `search_spend.recency` (decided by the scout), and WP-E1 can read the brief from the snapshot.

### Behaviour changes for handoff.md

- Every brief now carries an archetype, a time-sensitivity flag with a recency hint, the academic flag and the entities the request names. That switches on dormant WP-B behaviour:
  - the recency window for time-sensitive briefs, with `dropped_undated` counted (it was never time-sensitive before);
  - academic routing hints;
  - the entity-phrase grounding check, with its writer warning and report banner, whenever the request names something (it was always grounded before).
- When `brief.draft` is exhausted or its replies stay unparseable there is no generic fallback brief any more: `/research` sends "⚠️ Research failed: <the engine error>" to the chat, marks the run `error` and re-raises; a plan edit sends "⚠️ Re-plan failed: <the engine error>" (WP-E1's existing handler). Every brief attempt, failed parses included, is in `model_calls` under `brief.draft`.
- Medium, long and lecture outlines allocate evidence notes to sections (`note_ids`), and the archetype shapes the section titles. compose reads `note_ids` only from WP-C on, but the live section list already changes: merge-not-drop folds a starved section into its neighbour (its title goes, its sub-questions and focus move) and a repeated title merges into its first occurrence, so medium and longer runs can get fewer sections, other titles and other `sub_qs` than v3 wrote. After an Extend or a gap wave the outline keeps its section titles and lists `changed_sections`. An outline failure is now the `degrade: outline.plan → …` line, where v3 wrote "outline: LLM failed …; deterministic fallback".
- The analyst, the Analysis section and the takeaway labels appear once WP-C wires `analyst_node`.
- `cross_section_duplicate_ratio` no longer counts the Analysis section's findings.

### For WP-C (Task 9)

- Wire `insight.analyst_node` between compose and panel; it takes `runtime` for its events.
- `analyst_node` calls `compose._extract_findings(title, md, fetched)`, which returns `(findings, call | None, err)`. The extractor's rewrite (`compose.findings_extract`, `sentence`, `source_ids`) must keep that call working or update it. The analyst passes the analysis without its labels, so a finding's `sentence` is still contained in `analysis["md"]`: each label is appended after its bullet's text.
- `analysis["md"]` is the markdown for `## Analysis`, its label line included. `analysis["takeaways"]` (text with `[n]`, `confidence`, `reasons`) is what the deterministic TL;DR reads. A "contradicted by [n]" reason cites `[n]` in the analysis, so Sources lists it.
- `render_notes` should read `section["note_ids"]`. `TEMPLATES[archetype]["flat"]["body_emphasis"]` is for the flat writer and nothing reads it yet.
- `graph/synthesis_modes.py`'s docstring still names `compose._MODE_SECTIONS`, which now lives in `outline.py` (WP-C's file; not edited here).

### New env vars

None.

### Test files touched

- New: `tests/test_archetypes.py`, `tests/test_outline.py`, `tests/test_insight.py`.
- `tests/test_brief.py` (WP-I): the node tests moved from the legacy wrappers to the real facade on a fake driver. The fallback-brief test is replaced by the must-raise test, and the v4-field, entity and recorded-retry tests are new.
- `tests/helpers.py`: `fake_llm(monkeypatch, reply)`, the real `llm.complete` on a fake driver, shared by the files below.
- Minimal fixes for breakage caused by this change:
  - `tests/test_hitl_edit_revise.py` (WP-E1's): the plan-feedback brief test runs on `fake_llm` and passes a `run_config`;
  - `tests/test_reflexion.py` (WP-C's): `brief.draft` gets the same scripted reply through `fake_llm`.
- `tests/test_bench_metrics.py` (WP-G's): the test of the approved metric edit.
- Fix round 1: `tests/test_grounding.py` (WP-B's) for the approved grounding edit, `tests/test_run_config_resume.py` (WP-D's) for the approved `/research` edit; the WP-I files above.

### Fix round 1 (2026-10-20)

The task review of `aef6bd0` asked for fixes; the orchestrator's rulings are in the spec at `f9c255f` (R13 V, the grounding sentence, the WP-I entities and analyst-heading lines). The bullets above already describe the result; in short:

- `/research` tells the chat "⚠️ Research failed: <error>" before it re-raises (approved edit to WP-E1's `src/argus/bot/research.py`; the in-flight pop, the `error` status and the re-raise are unchanged).
- The analyst's heading contract: text before the first fixed heading dropped and recorded; no fixed heading → `omit_analysis (no analysis headings)`; numbered, bold-only and `Takeaways` headings recognised; `analysis: no cited takeaways` and the short-variant trim line recorded.
- R13: V ignores authority-0 notes.
- Outline: only a tie advances the tie rotation; a repeated title merges into its first occurrence (fresh reply and existing outline).
- Entities: the prompt defines proper names with counter-examples and placeholder JSON; period/number-only phrases dropped; the previous brief's entities count as named.
- Grounding (approved edit to WP-B's `src/argus/graph/grounding.py`): `normalise` drops a possessive `'s`/`’s`; a phrase also matches a run of whole tokens that spells it without its spaces ("CrewAI" ↔ "Crew AI"), one compiled pattern per entity per check. "GLM 5" still does not match "GLM 5.2".
- A test runs the analyst over the real `compose._extract_findings` on a fake driver (the call contract WP-C must keep or change on both sides).

Logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t8fix1\` (`red.log`, `green.log`, `full-suite.log`, the probe logs). The report is `.superpowers/sdd/2026-10-19-argus-v4/task-8-fix1-report.md`.

### Final review fix (2026-10-21): the degraded outline never repeats a title (I1)

`default_outline`, the `outline.plan` degrade path, titles each section by its first sub-question's first 70 characters, so two sub-questions that start alike gave two sections one title. It now ends with `_merge_repeated`, as the fresh reply and the extend path already did: the repeat merges into its first occurrence (sub-questions and notes unioned). Compose merges any repeat it still reads (wp-c.md). Test: `tests/test_outline.py`, a degraded outline of two sub-questions sharing their first 70 characters has unique titles, the merged section holding both sub-questions and their notes. Logs: `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\final-fix\`.

### Verification evidence (2026-10-19)

Logs are in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t8\`. Every run used `FREELLMAPI_BASE_URL=http://127.0.0.1:9/v1`, so no LLM call left the machine, and no bot was started.

- Baseline (`baseline-suite.log`, at `873682e`): only the four live-proxy tests (`tests/test_llm.py` and the `tests/test_graph.py` fixture) fail, on the closed port.
- RED (`red.log`), with the tests written and no implementation:
  - the files importing the new modules fail at collection (`cannot import name 'archetypes' / 'outline' / 'insight' from 'argus.graph'`);
  - the brief tests fail on the missing fields, the legacy call path and the fallback brief;
  - the metric test fails on the Analysis findings it still counts;
  - the plan-feedback test fails on the legacy path.
- GREEN (`green.log`): every touched test file passes.
- Full suite (`full-suite.log`, then `full-suite-final.log` after the self-review edits): only the same four live-proxy tests fail, as in the baseline.
- Live acceptance is the orchestrator's: one short no-regression run now. The analysis rows (labelled takeaways, the scenarios for the forecast, the Analysis section) follow in Task 9's runs, once the analyst is wired.
