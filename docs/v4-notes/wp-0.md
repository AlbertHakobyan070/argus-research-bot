# WP-0 — Prep

## WP-0a — contracts, dependencies, test fixtures (Task 1)

Spec: `docs/argus-v4-design.md` §3 WP-0 items 1, 2, 4, 6; contracts §2.1, §2.2, §2.4 and the `assemble` signatures of §2.8.

### What changed

- `pyproject.toml`: added `aiohttp>=3.14,<4`, `trafilatura>=2.3,<3`, `pypdf>=6,<7` to `dependencies`; nothing removed.
- `.gitignore`: `tools/bin/`, `argus_engines.local.json`, `bench/results/`, `docs/v4-notes/.scratch/`.
- New contract modules. They have no behaviour, and no node or bot code calls them yet:
  - `src/argus/engines/roles.py`: the §2.1 tables verbatim, plus a `Tier` alias (`Literal["cheap","strong","judge"]`) shared with the schema.
  - `src/argus/engines/schema.py`: the §2.2 models verbatim, with these validators:
    - `active_profile ∈ profiles`; `overrides` keys ⊆ `profiles`.
    - Every `EngineSpec.provider ∈ providers`: profile tiers, role primaries and fallbacks, override tiers and roles.
    - Profile and override `roles` keys ⊆ `ROLES`. An unknown role raises `ValueError`.
    - `on_exhausted` policy: `skip_item` only for skippable roles, `degrade` only for degradable roles, must-raise roles only `raise`.
    - `model_preferences` keys ⊆ tiers.
    - `ResolvedBinding`: the role is in `ROLES ∪ LEGACY_ROLES` and the policy fits it. `EnginesSnapshot`: each key equals its binding's role.
  - `src/argus/graph/run_config.py`: the §2.4 models verbatim, plus:
    - `PRESETS`, copied from the table.
    - `RunConfig.default_depth`, `from_preset` (dotted-key overrides) and `legacy`.
    - The budget helpers `gap_reserve`, `first_pass_fetch_cap` and `exa_search_reserve`.
    - `exa_cost_estimate_usd`, which reads the prices from the `ExaConfig` fields.
    - `require(state)` and `ensure_in_checkpoint(graph, cfg, *, engines)`.
  - `src/argus/graph/assemble.py`: `strip_fences`, `strip_duplicate_heading` (with its private `_normalize_heading_text`), `CITE_RE` and `build_sources_block`, moved verbatim from `compose.py`. Also new:
    - `assemble_draft(*, tldr, sections, analysis_md, fetched, flat)`, which renders TL;DR, then sections (flat: `## Findings` + body), then `## Analysis`, then `## Sources`.
    - `normalize_cite_markers`, which returns its input unchanged until WP-C.
- `src/argus/graph/compose.py`:
  - The moved definitions are deleted and imported from `assemble`. The four call sites now use the public names.
  - The unused `import re` is dropped, and `build_sources_block` leaves `__all__`, so there is no re-export.
  - `compose_node` still assembles inline; WP-C switches it to `assemble_draft`.
- `tests/conftest.py` (new): plain helper functions, not pytest fixtures. Import them with `from conftest import make_run_config, fake_engines_snapshot`.
  - `fake_engines_snapshot()` binds every `ROLES` id and the three `legacy.*` ids to provider `"fake"`, with the shipped §2.2 failure policy.
  - `make_run_config(length="short", depth=None, **overrides) -> dict` builds from the presets. Overrides use dotted keys, e.g. `**{"research.waves": 1}`.

### Dependencies (resolved by `uv pip install -e ".[dev,asr]"`, 2026-10-19)

- `aiohttp` 3.14.5, `trafilatura` 2.3.1, `pypdf` 6.21.1, plus their transitive dependencies.
- The install also satisfied extras that `pyproject.toml` already declared but the venv lacked: `python-telegram-bot[ext]` → APScheduler and tornado, with `cachetools` raised to the 7.x range that PTB pins.
- It also bumped `charset-normalizer`, `lxml` and `urllib3`.
- `uv pip check` reports every installed package compatible.
- `src/` uses neither `job_queue` nor webhooks.

### Contract decisions where the spec is silent or self-conflicting

1. **`RunConfig.legacy`: one name for both a field and a classmethod (§2.4).**
   - Pydantic treats a classmethod defined in the class body as the field's default. That breaks the model: instances get the bound method instead of `False`.
   - So the constructor is attached after the class is built (`RunConfig.legacy = classmethod(_legacy)`). `RunConfig.legacy(engines=…)` constructs, and `rc.legacy` reads the bool field.
   - A subclass would inherit the method as its field default, so do not subclass `RunConfig`.
2. **The length of `legacy()` is `"medium"`.** The signature takes no length, and medium is the length whose default depth is the standard preset. A resumed pre-v4 checkpoint keeps its own `state["length"]`, which stays authoritative.
3. **"Bounds" (item 6).**
   - Every numeric knob is ≥ 0. The three worker pools (`research.digest_concurrency`, `fetch.concurrency`, `compose.concurrency`) are ≥ 1.
   - `research.quote_verify_fuzzy` and `panel.rewrite_flag_ratio` are ≤ 1.
   - `total_fetch_cap ≥ gap_waves_max × gap_fetch_per_wave` and `exa.max_searches ≥ gap_waves_max × 2`, because the caps include the reserve.
   - Violations raise `ValueError`, naming the dotted key.
4. **Overrides.**
   - Keys are dotted leaf paths under `research`, `search`, `fetch`, `compose` and `panel`, or the top-level `recency` and `archetype_override`.
   - Dict knobs such as `search.source_mix_min` are set as a whole.
   - Any other key raises `ValueError`. That includes `length`, `depth`, `engines.*` and `legacy`, since the first two have their own parameters.
5. **`require`.** A missing key raises the spec message verbatim. A present but invalid value raises `ValueError("state['run_config'] invalid — …")`, chained to the pydantic error.
6. **`compose.analyst` "on unless length=tldr".** This applies to the quick preset only, as in the table. WP-I's `analyst_node` also no-ops for `tldr`.
7. **TL;DR heading.**
   - `assemble_draft` always renders `## TL;DR`. Today's sectioned compose uses `## Executive TL;DR` for `lecture`.
   - The frozen signature has no length, so that variant cannot be expressed. This matters only once WP-C moves `compose_node` onto `assemble_draft`.
8. **Legacy error string.**
   - `ensure_in_checkpoint` records `"legacy: run_config defaults applied (pre-v4 checkpoint)"`, as in spec §2.4.
   - The plan's Task 5 text words it `"run_config: legacy defaults applied (pre-v4 checkpoint)"`. The spec wins, and the `legacy:` prefix is what WP-C's degradation notes read.
   - `run_config` is not an `ArgusState` key until WP-D adds it, so the test drives `ensure_in_checkpoint` on a small `MemorySaver` graph that declares it.

### Behaviour changes for handoff.md

None. No node or bot behaviour changes in this step. The venv gains the three dependencies above.

### New env vars

None.

### Test files touched

- New: `tests/conftest.py`, `tests/test_engines_schema.py`, `tests/test_run_config.py`, `tests/test_assemble.py`.
- Modified: `tests/test_compose_headings.py`. Its one import line now points at `argus.graph.assemble`. The test bodies are unchanged.

### Verification evidence (2026-10-19)

- **Assembly capture.** `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\capture_assembly.py` drives the real `compose_node` sectioned path at 012a756 with only the writer, TL;DR and extraction LLM calls stubbed. It ran before anything moved. The expected strings in `tests/test_assemble.py` were then checked programmatically against the capture's output, with and without a TL;DR, and matched.
- **RED.** The focused run failed at collection with `ModuleNotFoundError: No module named 'argus.engines'`.
- **GREEN.** The focused run (the three new files plus `test_compose_headings.py`) passed. The full suite is green, with only the new tests added on top of the baseline.

## WP-0b — dead code and hidden fallbacks (Task 2)

Spec: `docs/argus-v4-design.md` §3 WP-0 item 5. One ruling overrides the spec text: `src/argus/quarantine.py`, `tests/test_quarantine.py` and the legacy quarantine branch of `report_builder_node` are **kept**. The independent review found that the module holds the only working sentence-cut logic. Task 8 evolves it into the structured cut and then removes the legacy branch. Everything else in item 5 is deleted as written.

### What changed

`src/argus/tools.py`:
- Deleted `harvest_sources`, `_parse_radar_md`, `HarvestResult`, `HarvestReport` (the wrapper for the intel-stack `harvest.py`) and its bullet in the module docstring.
- Deleted `perplexity_search`, `search_web`, `PERPLEXITY_SEARCH_URL` and `PERPLEXITY_MODEL`. `DEFAULT_WEB_TIMEOUT_S` and the `urllib.error` / `urllib.request` imports went with them, because only `perplexity_search` used them.
- `_normalize_web_hit` stays: `ddgs_search` uses it. `ddgs_search` and `_validate_timelimit` are untouched.
- `markdown_to_pdf` now calls `_render_pdf_reportlab` and nothing else, so a render failure propagates to the caller. The docstrings no longer mention a Chromium fallback.

`src/argus/post_synthesis.py`:
- Kept `FindingRecord`, `assign_finding_ids`, `quarantine_by_id` and `render_quarantine_appendix`.
- Deleted `verify_citations_async`, `fabrication_detector_async`, `grounding_verifier_async`, `run_post_synthesis_passes`, `run_post_synthesis_passes_async`, `render_body_from_findings` and the dataclasses `CitationVerdict`, `FabricationFlag`, `GroundingAssessment`, `PostSynthesisReport`.
- Also deleted `QuarantineDecision`. It is a dataclass that nothing builds or imports (no test either), and it is not on the keep list.
- The module docstring now describes only what is left.

`src/argus/graph/nodes.py` (`report_builder_node`, PDF part only):
- The in-process Chromium route and the `_render_pdf.py` subprocess under the intel-stack venv are gone.
- A render exception is logged, `report_paths["pdf"]` and its copy in `hitl.ctx.paths` become `None`, and the node returns `errors: ["pdf: <ExcType>: <message>"]`. A successful render returns no `errors` key, as before.
- The `os` and `Path` imports that only the fallback used are dropped. The legacy quarantine branch is untouched.

`src/argus/graph/state.py` and `src/argus/graph/__init__.py`: `ResearchPlan`, `PlannedSource` and their exports are removed. `brief_to_plan` builds a plain dict and never used them.

`cache_cleanup` stays. `.github/workflows/ci.yml` is unchanged: no test file was deleted, and the files in its `--ignore` list still exist.

### Deletion safety — grep results

Run on the tree before any edit, once per symbol: `git grep -n -w -e <symbol> -- src tests scripts bench` (the Perplexity family also case-insensitively, without `-w`). Output is condensed to `file: line numbers`.

```
harvest_sources
  src/argus/tools.py: 218
  tests/test_config_tools.py: 11,23,29
  tests/test_tools_routing_and_errors.py: 5,8,36,170,187
_parse_radar_md
  src/argus/tools.py: 255,267
  tests/test_config_tools.py: 37,57
HarvestResult
  src/argus/tools.py: 75,88,238,267,268,304,373
HarvestReport
  src/argus/tools.py: 85,221,256
  tests/test_tools_truth.py: 388,389
perplexity_search
  src/argus/citations.py: 51
  src/argus/tools.py: 1025,1045,1055,1082,1099,1207
  tests/test_web_search.py: 2
search_web
  src/argus/tools.py: 1011,1025,1185,1208
  tests/test_web_search.py: 1,6,12,97,100,134,135,136,160,199,224,270,277,281,297,352,365,367
_normalize_web_hit
  src/argus/tools.py: 1033,1094,1106,1176
DEFAULT_WEB_TIMEOUT_S
  src/argus/tools.py: 1030,1077
PERPLEXITY_SEARCH_URL
  src/argus/tools.py: 1028,1067
PERPLEXITY_MODEL
  src/argus/tools.py: 1029,1064
verify_citations_async
  src/argus/post_synthesis.py: 36,263,441,561
  tests/test_post_synthesis.py: 33,156,162,172
fabrication_detector_async
  src/argus/post_synthesis.py: 38,113,304,442,562
  tests/test_post_synthesis.py: 26,183,193,199
grounding_verifier_async
  src/argus/post_synthesis.py: 40,368,443,563
  tests/test_post_synthesis.py: 27,211,217
run_post_synthesis_passes_async
  src/argus/post_synthesis.py: 43,136,408,468,564
  tests/test_post_synthesis.py: 32
run_post_synthesis_passes
  src/argus/post_synthesis.py: 464,565
  tests/test_post_synthesis.py: 31,229,241,289
render_body_from_findings
  src/argus/post_synthesis.py: 49,475,566
  tests/test_post_synthesis.py: 29,304,315,323
CitationVerdict
  src/argus/post_synthesis.py: 100,145,266,270,280,288,291,555
  tests/test_post_synthesis.py: 20
FabricationFlag
  src/argus/post_synthesis.py: 108,146,309,340,343,353,556
  tests/test_post_synthesis.py: 21
GroundingAssessment
  src/argus/post_synthesis.py: 122,147,372,396,557
  tests/test_post_synthesis.py: 23,212
PostSynthesisReport
  src/argus/post_synthesis.py: 45,135,416,452,464,558
  tests/test_post_synthesis.py: 24,234
QuarantineDecision
  src/argus/post_synthesis.py: 86,554
ResearchPlan
  src/argus/graph/__init__.py: 4,20
  src/argus/graph/state.py: 42,79
  tests/test_tools_routing_and_errors.py: 33,46,48,59
PlannedSource
  src/argus/graph/__init__.py: 5,20
  src/argus/graph/state.py: 33,44
  tests/test_tools_routing_and_errors.py: 33,51
PERPLEXITY_ (prefix, any case)
  src/argus/citations.py: 51
  src/argus/tools.py: 1016,1025,1028,1029,1045,1046,1048,1052,1055,1064,1067,1082,1086,1095,1096,1099,1110,1198,1206,1207
  tests/test_web_search.py: 2,15,20,120,127,132,136,139,143,147,150,151,160,163,164,169,170,171,176,181,199,201,207,213,217,224
markdown_to_pdf (callers)
  src/argus/graph/nodes.py: 21,89,301,312,330,331
  src/argus/tools.py: 514,534,638
  tests/test_pdf_render.py: 31,85,132,146
```

Reading: every `src/` hit is the symbol's own definition, a self-reference inside code deleted in this step, or a comment. The only hit outside the defining module is the trailing comment at `src/argus/citations.py:51`. `scripts/` and `bench/` have no hits. `_normalize_web_hit` is not deleted, since `ddgs_search` still calls it. No live non-test caller exists, so nothing needed escalation.

### Behaviour changes for handoff.md

- **PDF: ReportLab only.** There is no in-process Chromium fallback and no intel-stack subprocess fallback, so `INTEL_PYTHON` and `INTEL_STACK_DIR` are no longer used for PDFs. If ReportLab raises, the run still continues: the report is delivered without a PDF, the bot shows `PDF: _failed_` (it already handled a missing PDF), and `errors` carries `pdf: <ExcType>: <message>`.
- `PERPLEXITY_API_KEY` is no longer read anywhere. The research engine never used it.
- The intel-stack `harvest.py` is no longer wrapped. It had no v3 caller.
- Not changed, but worth knowing: `_register_report_fonts` falls back to ReportLab's built-in fonts, with a logged warning, if a bundled font file is missing. That is a font fallback inside the one renderer, not a second renderer. It is outside item 5, and `test_markdown_to_pdf_survives_missing_font_dir` pins it.
- Prose that is now stale and was left alone because those files are outside this step's list: the PDF-rendering bullet in `README.md` (still mentions the Chromium last resort), the module docstring of `src/argus/graph/report_builder_helpers.py` ("via the intel-stack Chromium renderer"), the comment at `src/argus/citations.py:51` (names `perplexity_search`), and the `ResearchBrief` docstring in `src/argus/graph/state.py` ("replaces the v2 ResearchPlan").

### New env vars

None. `PERPLEXITY_API_KEY` stops being read.

### Test files touched

- `tests/test_pdf_render.py`: three tests added (the two required ones, plus a success-path check of the same node). The existing tests did not expect the Chromium path and are unchanged.
- `tests/test_web_search.py`: Perplexity and `search_web` cases removed; the docstring and imports are trimmed to match. `test_max_results_respected` now calls `ddgs_search` directly instead of going through `search_web`, because it was the only test that pinned `max_results` forwarding for `ddgs_search`.
- `tests/test_config_tools.py`: harvest cases removed.
- `tests/test_tools_routing_and_errors.py`: harvest case removed. The helpers `_plan_with_paper` and `_make_state` are removed too: nothing called them, and they needed the deleted `ResearchPlan` and `PlannedSource`. The docstring is updated. `test_run_script_explicit_override_still_works` still passes the literal script name `"harvest.py"` to `_run_script`; it tests the `python_bin` override, not harvest, so it stays.
- `tests/test_tools_truth.py`: the uncalled helper `_make_empty_harvest`, which held the `HarvestReport` import, is removed.
- `tests/test_post_synthesis.py`: async-pass and `render_body_from_findings` cases removed. The quarantine, id, appendix and report-builder integration tests stay, including the one that drives the legacy quarantine path.
- Kept as is, by the ruling: `tests/test_quarantine.py`.

### Removed tests

- `tests/test_web_search.py` (8): `test_search_web_returns_dicts_with_required_keys`, `test_search_web_empty_query_returns_empty_list`, `test_perplexity_without_api_key_returns_empty_and_warns`, `test_perplexity_with_api_key_parses_response`, `test_perplexity_http_failure_returns_empty`, `test_unknown_engine_returns_empty_and_warns`, `test_default_engine_is_ddgs`, `test_search_web_forwards_timelimit`.
- `tests/test_config_tools.py` (2): `test_harvest_dry` (it ran the real `harvest.py` subprocess), `test_parse_radar_md_extracts_items`.
- `tests/test_tools_routing_and_errors.py` (1): `test_harvest_sources_routes_through_intel_python`.
- `tests/test_post_synthesis.py` (15): `TestVerifyCitationsAsync` (`test_registers_fetched_urls`, `test_flags_unregistered`, `test_dedupes_across_findings`); `TestFabricationDetectorAsync` (`test_supported_claim_low_flag`, `test_unsupported_claim_flagged`, `test_no_citations_is_fabricated`); `TestGroundingVerifierAsync` (`test_structured_assessment_shape`, `test_handles_missing_fetched`); `TestRunPostSynthesisPassesAsync` (`test_returns_post_synthesis_report`, `test_quarantine_routes_to_quarantined`, `test_parallel_speedup_observable`, `test_wall_time_is_reported`); `TestRenderBodyFromFindings` (`test_flat_template_no_sections`, `test_sectioned_template_groups_by_section`, `test_empty_findings_returns_empty`).

Total: 26 removed, 3 added.

### Verification evidence (2026-10-19)

- **Baseline, unmodified tree.** The full suite gave 554 passed, 1 skipped, 1 failed. The failure was the live-proxy test `tests/test_graph.py::test_quick_graph_runs`, which got a FreeLLMAPI 429 ("All models exhausted"). It passed when run on its own straight afterwards, so the baseline is the 555 passed, 1 skipped of the plan.
- **RED.** `PYTHONPATH="" ./venv/Scripts/python.exe -m pytest -q tests/test_pdf_render.py -k "raises_without_chromium_fallback or records_pdf_failure"` gave `2 failed, 9 deselected`.
  - `test_markdown_to_pdf_raises_without_chromium_fallback`: `AssertionError: Chromium fallback was invoked`, with the log line `ReportLab render failed (reportlab exploded); trying Chromium.`
  - `test_report_builder_records_pdf_failure`: `AssertionError: []` on `assert len(errors) == 1 and errors[0].startswith("pdf: ")`, with only the log line `PDF render failed: reportlab exploded`.
  - The first test points `INTEL_STACK_DIR` at a temporary canary `_common.py` that raises. The real intel-stack helper exists on this machine, so without the canary the RED run would have launched a real browser.
- **GREEN.** The same command gave `2 passed, 9 deselected`.
- **Focused run** (`test_pdf_render`, `test_web_search`, `test_config_tools`, `test_tools_routing_and_errors`, `test_tools_truth`, `test_post_synthesis`, `test_quarantine`): all passed.
- **Static checks.** A stdlib `ast` scan found no import orphaned by the deletions and no undefined name in the edited modules (there is no linter in the venv). `import argus.tools, argus.post_synthesis, argus.graph, argus.graph.nodes` succeeds.
- **Full suite after the change:** 530 passed, 1 skipped, 2 failed. The two failures were `tests/test_graph.py::test_quick_graph_runs` and `::test_deep_graph_pauses_for_plan_approval`, again FreeLLMAPI 429 at the first LLM node: the baseline run had just spent the proxy's quota. Re-running `tests/test_graph.py` after the quota window gave 2 passed. That includes the deep graph running through `report_builder_node` and the real ReportLab render. Together: 532 passed, 1 skipped, which is the 555 of the baseline minus 26 removed plus 3 added.
