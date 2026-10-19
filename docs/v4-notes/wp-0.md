# WP-0 — Prep

## WP-0a — contracts, dependencies, test fixtures (Task 1 and its fix round)

Spec: `docs/argus-v4-design.md` §3 WP-0 Task 1; contracts §2.1, §2.2, §2.4 and the `assemble_draft` bullet of §2.8. The first commit built the contracts. The fix round, in its own commit, applied spec §2.0 (F1–F20) exactly, together with the Task 1 review findings. The §2 contracts are frozen after the fix round; changing them needs the orchestrator.

### What changed

- `pyproject.toml`:
  - First commit: `aiohttp>=3.14,<4`, `trafilatura>=2.3,<3` and `pypdf>=6,<7` added to `dependencies`; nothing removed.
  - Fix round: `langgraph>=1.2.8,<2` (was `>=1.2,<2`), for the APIs WP-A uses (`RunControl`, `GraphDrained`, `durability`, `runtime` injection), and `openai>=2.44,<3`, which the WP-A driver imports directly. The venv already had both inside these pins, so nothing was installed.
- `.gitignore`: `tools/bin/`, `argus_engines.local.json`, `bench/results/`, `docs/v4-notes/.scratch/`.
- Contract modules. They have no behaviour, and no node or bot code calls them yet:
  - `src/argus/engines/roles.py`: the §2.1 tables verbatim, including `ALL_SKIPPED_RAISES`, plus a `Tier` alias (`Literal["cheap","strong","judge"]`) shared with the schema.
  - `src/argus/engines/schema.py`: the §2.2 models verbatim, including `ExaPrices` and `PricingConfig`, with these validators:
    - `active_profile ∈ profiles`; `overrides` keys ⊆ `profiles`.
    - Every `EngineSpec.provider ∈ providers`: profile tiers, role primaries and fallbacks, override tiers and roles.
    - `EngineSpec` sets at most one of `model` and `model_from_preferences`. Setting neither is allowed: the model is chosen later.
    - `Profile.tiers` holds every tier. Its keys can only be tiers, so it holds exactly the tier set. Override tiers stay partial.
    - Profile and override `roles` keys ⊆ `ROLES`. An unknown role raises `ValueError`.
    - `on_exhausted` policy: `skip_item` only for skippable roles, `degrade` only for degradable roles, must-raise roles only `raise`.
    - `model_preferences` keys ⊆ tiers.
    - `ResolvedBinding`: the role is in `ROLES ∪ LEGACY_ROLES` and the policy fits it. `EnginesSnapshot`: each key equals its binding's role.
  - `src/argus/graph/run_config.py`: the §2.4 models verbatim, plus:
    - `PRESETS`, copied from the table.
    - `RunConfig.default_depth`, `from_preset` (dotted-key overrides, recorded in `overrides_applied`), `for_legacy_checkpoint` and `rebuild`.
    - The bounds validator, and the budget helpers `gap_reserve`, `first_pass_fetch_cap` and `exa_search_reserve`.
    - `exa_cost_estimate_usd(prices)`, which takes the prices from `EnginesConfig.pricing.exa` and returns `None` when there are none.
    - `require(state)` and `ensure_in_checkpoint(graph, cfg, *, engines)`.
  - `src/argus/graph/assemble.py`: `strip_fences`, `strip_duplicate_heading` (with its private `_normalize_heading_text`), `CITE_RE` and `build_sources_block`, moved verbatim from `compose.py`. Also new:
    - `assemble_draft(*, tldr, sections, analysis_md, appendices=None, fetched, flat, tldr_heading="TL;DR")`. It renders the TL;DR under `## <tldr_heading>`, then the sections (flat: `## Findings` + body), `## Analysis`, the appendices, and `## Sources`, which lists the ids cited anywhere above it, appendices included.
    - `normalize_cite_markers`, which returns its input unchanged until WP-C.
- `src/argus/graph/compose.py` (first commit):
  - The moved definitions are deleted and imported from `assemble`. Its call sites use the public names.
  - The unused `import re` is dropped, and `build_sources_block` leaves `__all__`, so there is no re-export.
  - `compose_node` still assembles inline; WP-C switches it to `assemble_draft`.
- `tests/helpers.py`: plain helper functions, not pytest fixtures. Import them with `from helpers import make_run_config, fake_engines_snapshot`, never from `conftest`. The first commit had them in `tests/conftest.py`; the fix round moved them and deleted that file.
  - `fake_engines_snapshot()` binds every `ROLES` id and the `legacy.*` ids to provider `"fake"`, with the shipped §2.2 failure policy.
  - `make_run_config(length="short", depth=None, **overrides) -> dict` builds from the presets. Overrides use dotted keys, e.g. `**{"research.waves": 1}`.

### Fix round (spec §2.0)

- Roles (F2–F5):
  - `compose.tldr` and `panel.precision` are gone from every table, because v4 has no LLM TL;DR and no LLM precision scan.
  - `research.extend_querygen` is new: cheap tier, JSON role, degrades to `skip_extend_wave`.
  - `research.rerank` degrades to `keyword_triage` instead of skipping.
  - `ALL_SKIPPED_RAISES` names the per-item roles whose node raises when every item was skipped (§2.3).
- Schema (F6–F10):
  - `ProviderSpec` gains `token_param`, `timeout_retry_max` and `retry_after_cap_s`.
  - `EngineSpec` gains `extra_body`, `supports_temperature` and the at-most-one-model validator.
  - `Profile` gains the every-tier validator.
  - `TranscriberConfig.faster_whisper` and `openai_audio` default to `{}`; their data moves to `argus_engines.json` with WP-F.
  - `ExaPrices`, `PricingConfig` and `EnginesConfig.pricing` are new.
- RunConfig (F11–F17):
  - The Exa price fields leave `ExaConfig`; prices now come from `EnginesConfig.pricing`.
  - `search.source_mix_min` and `archetype_override` are deleted, also as override roots.
  - New knobs: `fetch.max_bytes` and `panel.judge_batch_size`. `precision_mode` loses `"llm"`.
  - The §2.4 floors and ceilings are enforced.
  - New: `overrides_applied` and `rebuild`.
  - `for_legacy_checkpoint` replaces the first commit's `legacy` classmethod, which was attached after the class was built. The `legacy` bool field stays.
  - `require` checks the length.
- `assemble_draft` (F18): `appendices` and `tldr_heading`.
- Helpers (F19): `tests/conftest.py` became `tests/helpers.py`.
- Review findings in the same commit:
  - `ensure_in_checkpoint` refuses a thread that has no checkpoint (Important 1).
  - The flat-path warning, in `assemble.py` and in decision 6 below (Minor 1).
  - This note no longer hardcodes counts or resolved versions (Minor 4), and the stale remark about the plan's Task 5 wording is gone (Minor 5).
  - The first `common.py` sub-heading in WP-0c is reworded.

### Dependencies (resolved by `uv pip install -e ".[dev,asr]"` in the first commit, 2026-10-19)

- `aiohttp`, `trafilatura` and `pypdf`, plus their transitive dependencies, at the versions `uv` resolved inside the pins.
- The install also satisfied extras that `pyproject.toml` already declared but the venv lacked: `python-telegram-bot[ext]` → APScheduler and tornado, with `cachetools` raised into the range that PTB pins.
- It also bumped `charset-normalizer`, `lxml` and `urllib3`.
- `uv pip check` reports every installed package compatible.
- `src/` uses neither `job_queue` nor webhooks.

### Contract decisions where the spec is silent

1. **Bounds.**
   - The floors and ceilings written inline in §2.4 live in one table in `run_config.py`, enforced by the `RunConfig` bounds validator. So `from_preset` with an out-of-bounds override raises a pydantic `ValidationError` (a `ValueError`) that names the dotted key and the bound.
   - Every other numeric knob has the floor 0.
   - `total_fetch_cap ≥ gap_waves_max × gap_fetch_per_wave` and `exa.max_searches ≥ gap_waves_max × 2`, because the caps include the reserve.
2. **Overrides.**
   - Keys are dotted leaf paths under `research`, `search`, `fetch`, `compose` and `panel`, or the top-level `recency`.
   - Any other key raises `ValueError`. That includes `length` and `depth` (they have their own parameters), `engines.*`, `legacy`, `overrides_applied` and the deleted fields.
   - `from_preset` records the overrides as given, deep-copied, in `overrides_applied`.
   - `rebuild` calls `from_preset` with the new length, depth and engines and `overrides=overrides_applied`. It keeps `origin` and `legacy`, and gets a new `created_at`.
   - A recorded override that does not fit the new preset raises: for example, a `total_fetch_cap` below a deeper preset's gap reserve.
3. **`require`.**
   - A missing key raises the spec message verbatim.
   - A present but invalid value raises `ValueError("state['run_config'] invalid — …")`, chained to the pydantic error.
   - A `run_config` whose `length` differs from a set (non-`None`) `state["length"]` raises, naming both lengths.
4. **`ensure_in_checkpoint`.**
   - A thread with no checkpoint (`created_at is None`) raises `ValueError("no checkpoint for thread …")`. Before the fix round, a `thread_id` that never ran got a checkpoint holding the legacy config, with `next` at the graph's first node.
   - A present `run_config` is returned through `require`, and nothing is written.
   - Otherwise the checkpoint's own `length` picks the legacy preset. A checkpoint without `length` raises `ValueError` naming the thread. Neither refusal writes anything.
   - The recorded line is the spec's `"legacy: run_config defaults applied (pre-v4 checkpoint)"`. WP-C's degradation notes read its `legacy:` prefix.
   - `run_config` is not an `ArgusState` key until WP-D adds it, so the tests drive `ensure_in_checkpoint` on a small `MemorySaver` graph that declares it. LangGraph's `aupdate_state` silently drops an undeclared key, so WP-D must add the key in the same change that wires `ensure_in_checkpoint`.
5. **`compose.analyst` "on unless length=tldr".** This applies to the quick preset only, as in the table. WP-I's `analyst_node` also no-ops for `tldr`.
6. **`assemble_draft` and today's compose.**
   - With `tldr_heading` (lecture passes `"Executive TL;DR"`), it reproduces compose's sectioned assembly.
   - It does not reproduce compose's flat path, which serves `tldr` and `short` and any brief with one sub-question. `flat=True` adds `## Findings`, while the flat writer emits its own `## TL;DR` and `## Key Findings`.
   - So the flat path must not move onto `assemble_draft` before WP-C removes the flat writer's own headings. Otherwise those reports get `## Findings` above the model's `## TL;DR` and `## Key Findings`.
   - Each appendix is a markdown block with its own heading. Blank `tldr`, `analysis_md` and appendices are omitted.
7. **`exa_cost_estimate_usd` with Exa disabled.** It follows the spec formula, so it still reports a cost when `"exa"` is not in `search.providers_enabled`. Whether to show zero is a display question for WP-E1.

### Behaviour changes for handoff.md

None. No node or bot behaviour changes. The first commit added the dependencies above to the venv; the fix round installed nothing.

### New env vars

None.

### Test files touched

- First commit:
  - New: `tests/test_engines_schema.py`, `tests/test_run_config.py`, `tests/test_assemble.py`, and the helpers module (then `tests/conftest.py`).
  - Modified: `tests/test_compose_headings.py`. Its import of the moved helpers now points at `argus.graph.assemble`; the test bodies are unchanged.
- Fix round: `tests/conftest.py` renamed to `tests/helpers.py`, with its docstring updated. The new test files above are extended and import from `helpers`.

### Verification evidence (2026-10-19)

First commit:
- **Assembly capture.** `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\capture_assembly.py` drives the real `compose_node` sectioned path at 012a756 with only the writer, TL;DR and extraction LLM calls stubbed. It ran before anything moved. The expected strings in `tests/test_assemble.py` were then checked programmatically against the capture's output, with and without a TL;DR, and matched.
- **RED.** The focused run failed at collection with `ModuleNotFoundError: No module named 'argus.engines'`.
- **GREEN.** The focused run (the new test files plus `test_compose_headings.py`) passed. The full suite is green, with only the new tests added on top of the baseline.

Fix round (scripts and logs in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t1fix\`):
- **RED-1**, with only the tests changed (`red1.log`, `red1_assemble.log`):
  - The test files that import the helpers fail at collection on the new contract names (`ALL_SKIPPED_RAISES`, `ExaPrices`).
  - The new `assemble_draft` tests fail with `TypeError` on `tldr_heading` and `appendices`.
  - `import helpers` fails.
- **RED-2**, after adding only the names the tests import and moving the helpers (`red2.log`, `red2_tbline.log`):
  - Every new behavioural case fails on its own: removed roles still present, validators that do not raise, missing fields and methods, the old `exa_cost_estimate_usd` signature, bounds not enforced, the legacy config built from the standard preset instead of the checkpoint's length, and refusals not raised.
  - The only bounds cases that pass are the ratio ceilings (`quote_verify_fuzzy`, `rewrite_flag_ratio`) that the first commit already enforced.
- **GREEN.** The focused run (the test files above plus `test_compose_headings.py`) passed (`green_focused.log`). The full suite is green, including the live-proxy tests (`full_suite.log`).
- **Spec fidelity** (`spec_fidelity.py`, `spec_fidelity.log`):
  - It executes the §2.1 and §2.2 blocks and parses the §2.4 knob-class lines.
  - All of these match: the role tables; the §2.2 model fields (names, order, annotations, defaults); the §2.4 knob and `RunConfig` fields (names, order, defaults) and inline bounds; and the §2.4/§2.8 signatures.
  - Run against the pre-fix modules (extracted from HEAD with `git archive`), the same script fails in each of those areas (`spec_fidelity_at_HEAD.log`).
- **Important 1, before and after** (`probe_old_ensure.py`): on a thread that never ran, the pre-fix `ensure_in_checkpoint` wrote a checkpoint with the legacy config and `next` at the graph's first node; the fix round raises and writes nothing.
- **Static checks.**
  - The import smoke with `-W error` passes.
  - The installed `langgraph` and `openai` satisfy the new pins.
  - There are no unused imports in the touched files.
  - No `conftest` import or deleted name is left, outside the tests that assert its absence.

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

## WP-0c — split `bot.py` into the `bot/` package (Task 3)

Spec: `docs/argus-v4-design.md` §3 WP-0 item 3 and §1.2. A pure move: no behaviour change and no re-export shims. The orchestrator added two items during the task: a `POST_STOP_HOOKS` list run from a `post_stop` callback, and the rule that the shared mutable objects are reached only as attributes of their owning module and are never rebound.

### Symbol → module mapping

Written before anything moved, and covering every top-level statement of `bot.py`. A symbol is "placed" when item 3 names it or names the feature it belongs to. Anything item 3 does not place goes to `common.py` and is listed as such.

`common.py`:
- Shared helpers and constants, placed in `common.py`: `HELP_TEXT`, `_LENGTH_LABELS`, `_parse_length`, `_inflight`, `_pending_reply`, `_bot_data_get`, `_get_library`, `_set_run_status`, `_parse_indices`, `_allowed`, `_safe_send`, `_safe_edit_cb`, `_stream_progress`, `_html_escape_for_tg`, `_md_escape`, `_fmt_duration`, `_STATUS_GLYPHS`, `_fmt_bytes`, and the `_TG_*` constants (`_TG_UPLOAD_CAP_BYTES`, `_TG_CONNECT_TIMEOUT_S`, `_TG_READ_TIMEOUT_S`, `_TG_WRITE_TIMEOUT_S`, `_TG_POOL_TIMEOUT_S`, `_TG_MEDIA_WRITE_TIMEOUT_S`).
- Not placed by item 3, so here by the brief's rule:
  - `_HELP_MD_PATH`: the `/help full` guide path, beside `HELP_TEXT`. Used by `history.help_cmd`.
  - `_chat_entries`: a query over `_inflight`. Used by `history.status_cmd` and `history.cancel_cmd`.
  - `_send_md_doc`: a send helper beside `_safe_send`. Used by `research._resume_after_report`.

`research.py` (research/ask commands, plan/report keyboards and formatters, HITL callbacks and resume paths):
- `_format_plan`, `_plan_keyboard`, `_report_keyboard`, `research_cmd`, `ask_cmd`, `_replan_after_edit`, `_revise_after_feedback`, `_resume_after_plan`, `_send_report_preview`, `_resume_after_report`, `_finalize_run`.
- From `on_text`: the pending HITL reply branch, as `handle_pending_reply`.
- From `on_callback`: the run lookup with its not-in-flight notice, and the `len:*`, `plan:*` and `report:*` branches, as `handle_callback`.

`media.py` (video/find/fetch/quality/transcript(s), pools, `m:*`, `t:send`, the media branch of `on_text`):
- Pools: `_VIDEO_POOL_TTL_S`, `_video_pool`, `_pool_get`, `_pool_put`.
- Commands: `video_cmd`, `fetch_cmd`, `find_cmd`, `quality_cmd`, `transcript_cmd`, `transcripts_cmd`.
- Helpers used only by those commands and buttons: `_MEDIA_PROGRESS_INTERVAL_S` (the download progress throttle; its neighbour `_TG_UPLOAD_CAP_BYTES` goes to `common.py` under `_TG_*`), `_media_keyboard`, `_get_media_quality`, `_fetch_one_media`, `_transcript_for_item`.
- From `on_text`: the pasted-media-link branch, as `handle_text`.
- From `on_callback`: the `m:*` and `t:send` branches, as `handle_callback`.

`history.py`: `runs_cmd`, `status_cmd`, `cancel_cmd`, `append_cmd`, `continue_cmd`, `help_cmd`, `_sources_from_refs`. It imports `_format_plan`, `_plan_keyboard`, `_resume_after_plan` and `_send_report_preview` from `research`.

`delete.py` (browser + `del:*`): `_DELETE_PAGE_SIZE`, `_delete_sessions`, `_DEL_CATEGORY_LABELS`, `_dir_size`, `_safe_to_delete`, `_render_delete_page`, `_delete_load_items`, `delete_cmd`, `_perform_delete`, `_on_delete_callback`. From `on_callback`: the `del:*` dispatch line, as `handle_callback`.

`app.py`: `_on_error`, `_post_init`, `_reconcile_orphaned_runs`, `app_ctx`, `_post_shutdown`, `build_application`, `main`. The shared preambles of `on_text` (ACL, text, chat id) and `on_callback` (ACL, answer, the 3-part split, the stale-button notice) stay here as the two dispatchers.

`__init__.py`: the module docstring, and `main` imported from `app` (the only name the package exports). `__main__.py`: the `if __name__ == "__main__": main()` guard.

Every module that logs keeps `logger = logging.getLogger("argus.bot")`, the logger name `bot.py` used.

New names, as the interfaces later WPs build on:
- `app.py`: `HANDLER_MODULES`, `POST_INIT_HOOKS`, `POST_STOP_HOOKS`, `POST_SHUTDOWN_HOOKS`, and `_post_stop`, which runs `POST_STOP_HOOKS`.
- `register(app)` on `research`, `media`, `history` and `delete`.
- `research.handle_callback`, `research.handle_pending_reply`, `media.handle_callback`, `media.handle_text`, `delete.handle_callback`.

### What changed

- `src/argus/bot.py` is replaced by the package above. Each moved statement is its original source text, comments included. Only these location adaptations differ:
  - The shared mutable dicts are reached as attributes of their owning module. `research`, `history` and `delete` read and mutate `common._inflight` and `common._pending_reply`. Nothing outside `media` reads `_video_pool`. None of them is rebound outside its owner.
  - The three function-level `from .tools import …` in `media` become `from ..tools import …`.
  - `_HELP_MD_PATH` uses `parents[3]` instead of `parents[2]`, so it still resolves to the repo's `docs/help.md` from one directory deeper. `test_help_md_exists_and_covers_commands` pins it.
  - `json`, unused in `bot.py`, is not imported anywhere.
- New code, as listed above:
  - The module docstrings.
  - `HANDLER_MODULES`, and one `register(app)` per handler module.
  - The `POST_INIT_HOOKS`, `POST_STOP_HOOKS` and `POST_SHUTDOWN_HOOKS` lists, shipped empty, and `_post_stop`.
  - The two dispatchers, and the `handle_*` functions carved out of their branches.
- How the new code is wired:
  - `_post_init` ends by awaiting `POST_INIT_HOOKS`, and `_post_shutdown` starts by awaiting `POST_SHUTDOWN_HOOKS`.
  - `build_application` adds `.post_stop(_post_stop)` and replaces its block of `CommandHandler` lines with the `register` loop.
  - PTB calls `post_stop` only after `Application.stop()`, which awaits the tasks started with `app.create_task`.
  - `research.handle_pending_reply` returns `True` wherever the old branch returned, and `False` when no reply is pending. `on_text` then falls through to `media.handle_text`, as before.
- Command registration order:
  - `/delete` is now registered last, because `delete` is last in `HANDLER_MODULES`. `bot.py` registered it between `/continue` and `/status`.
  - Every handler is still in group 0, and each command handler matches only its own command, so no update reaches a different handler (see the evidence below).
- `scripts/run.sh`: the FATAL respawn hint names `src/argus/bot/` instead of `src/argus/bot.py`. Both launch scripts still run `python -m argus.bot`, which `__main__.py` serves. `run_mac.sh` is unchanged.
- Stale prose left alone, because it is outside this step's file list:
  - The `README.md` tree still shows `bot.py`.
  - Comments in `src/argus/graph/compose.py`, `src/argus/graph/state.py` and `src/argus/graph/report_builder_helpers.py` still say `bot.py`.
  - Test docstrings and assert messages still say `bot.py` or `bot._…`.
  - The package docstring, moved verbatim, still says the commands are registered in `build_application`.

### Behaviour changes for handoff.md

None.

### New env vars

None.

### Test files touched

How the tests were updated:
- Imports point at the module that defines each name.
- Patches target the module that looks the name up. A name the old test patched once on `bot.py` is now patched on every module that looks it up along the test's path.
- The shared dicts are reached as `bot_common._inflight`, `bot_common._pending_reply` and `bot_media._video_pool`. They are mutated in place and never rebound.
- The test bodies are otherwise unchanged.

Per file:
- Imports only: `tests/test_bot.py`, `tests/test_bot_markdown_fallback.py`, `tests/test_youtube.py`, and `tests/test_orphan_reconcile.py`, whose unused `from argus import bot as bot_mod` from the base commit stays.
- `tests/test_continue_extend.py`: `_allowed`, `get_settings` and `_resume_after_plan` patched on `history`.
- `tests/test_delete_browser.py`: `_allowed` patched on `app`, and `get_settings` on `app` and `delete`.
- `tests/test_e2e_research.py` (opt-in e2e): the in-flight registry reached through `bot_common`.
- `tests/test_hitl_edit_revise.py`: `_allowed` and `get_settings` patched on `app`, and `_replan_after_edit` and `_resume_after_plan` on `research`.
- `tests/test_media_bot.py`: `_allowed` and `get_settings` patched on `app` and `media`, and `reddit_search` and `download_media` on `media`.
- `tests/test_plan_hitl_gating.py`: `_allowed` and `get_settings` patched on `app`, and `_resume_after_plan` on `research`.
- `tests/test_run_scoping.py`: `_allowed` and `get_settings` patched on `research`.
- `tests/test_transcript.py`: `_parse_indices` taken from `common`; the pool helpers, `_video_pool` and the `time` module reference taken from `media`.

### Verification evidence (2026-10-19)

The previous implementer generated the package with a script that copies each moved block by its original line span. It stopped before the tests were finished. Its own equivalence check compared ASTs, which ignores comments, and compared the command handlers only as a sorted set. So this round re-verified everything against `bot.py` at `3e7a51f`, with its own scripts in `A:\DS_Vault\caches\tmp\claude\A--Hermes-Agents-argus\83d80941-f0f5-432e-a15e-8b7a5e0143b7\scratchpad\t3\`.

- **Verbatim move, by source text** (`verify_move.py`): PASS.
  - Each top-level statement of `bot.py` is compared as exact source text, with the comment block directly above it, against its new home.
  - It must appear in exactly one module. The one exception is `logger`, an identical line in each module that logs.
  - A difference passes only if one of the adaptations above, applied to the original text, reproduces it exactly.
  - `on_text` and `on_callback` are compared piece by piece with the dispatchers and the `handle_*` bodies.
  - Every non-blank line of `bot.py` is accounted for; the section-header comments moved with their code.
- **Shared state** (`verify_shared.py`, over the package and every test that touches the bot): PASS. There is no bare-name use outside the owner, no name import and no rebinding. Run before this round's test edits, it flagged the two tests that still imported `_inflight` by name.
- **Handler registration** (`compare_handlers.py`): PASS. It builds the `bot.py` application and the package's application with a dummy token, without polling or network.
  - The handlers are the same, all in group 0, with the same handler type at every position.
  - The error handler, `post_init`, `post_shutdown` and the request timeouts are the same. `post_stop` is now `_post_stop`, and all three hook lists are empty.
  - The only difference in order is the `/delete` command handler.
  - Synthetic updates were routed through both handler lists the way `Application.process_update` picks a handler:
    - every command, with and without arguments;
    - `@`-addressed, unknown and upper-case commands;
    - plain text and a pasted link;
    - every callback namespace, including stale and garbage data.

    Each update lands on the same callback in both applications, and none matches more than one handler.
- **Patch targets** (`bot_patchcheck.py`, a diagnostic pytest plugin): no divergences, and every bot test file passes under it.
  - It wraps every bot-module binding of the names the tests patch.
  - It flags any call to an unpatched binding while the same name is patched in another bot module. In the single-namespace original, that call would have hit the patch.
  - A deliberately wrong target was flagged.
- **Static names** (`check_names.py`, from the previous implementer's scratchpad): no undefined names in the package or the touched tests. The unused imports left in the tests are the same as at the base commit.
- **Import order**: the first imports of `argus.*` happen in the same order as before, except that `cache_cleanup` now loads before `config`. `cache_cleanup` imports only stdlib modules and reads no environment.
- **Import check**: `./venv/Scripts/python.exe -c "import argus.bot, argus.bot.app, argus.engines.schema, argus.graph.run_config, argus.graph.assemble"` exits 0.
- **Full suite** (`PYTHONPATH="" ./venv/Scripts/python.exe -m pytest -q`): green on the first run. It has the same count as after Task 2, and the only skip is the opt-in `tests/test_e2e_research.py`.
  - The live-proxy tests (the `test_llm.py` smoke test and `test_graph.py`) passed in that run.
  - The only warning is the pre-existing `SyntaxWarning` at `tests/test_tools_routing_and_errors.py:187`, in a file this step does not touch.
