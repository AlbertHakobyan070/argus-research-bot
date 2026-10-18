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
