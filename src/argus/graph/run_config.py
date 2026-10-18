"""Per-run configuration (v4 spec §2.4): knob models, depth presets, budget helpers.

Contract only — no node reads it yet (WP-D wires it). The rule for every
node: ``rc = run_config.require(state)`` (never ``state.get``), and every
resume entry point calls ``ensure_in_checkpoint`` first.
"""
from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ValidationError, model_validator

from ..engines.schema import EnginesSnapshot
from .state import Length

Depth = Literal["quick", "standard", "deep", "exhaustive"]


class ExaConfig(BaseModel):
    max_searches: int
    max_contents_pages: int
    search_type: Literal["auto", "fast", "instant"] = "auto"
    num_results: int = 8
    text_max_chars: int = 12000
    text_max_chars_paper: int = 20000
    price_per_search_usd: float = 0.007         # Exa list prices; display only, never hardcoded in code
    price_per_page_usd: float = 0.001
    prices_as_of: str = "2026-10-18"


class ArxivConfig(BaseModel):
    enabled: bool = True
    min_interval_s: float = 3.1
    max_queries_per_run: int = 2


class SearchConfig(BaseModel):
    providers_enabled: list[Literal["exa", "ddgs", "arxiv", "github"]] = ["exa", "ddgs", "arxiv", "github"]
    exa: ExaConfig
    arxiv: ArxivConfig
    github_mode: Literal["exa_domain", "repo_search", "off"] = "exa_domain"
    ddgs_max_results: int = 10
    source_mix_min: dict[Literal["paper", "repo", "web"], int] = {}


class FetchConfig(BaseModel):
    chain: list[Literal["exa_text", "native", "exa_contents", "intel_snatch", "intel_crawl", "intel_stealth"]] = [
        "exa_text", "native", "exa_contents", "intel_snatch", "intel_crawl", "intel_stealth"]
    max_chars_on_disk: int = 60000
    timeout_s: int = 30
    concurrency: int = 4


class ResearchConfig(BaseModel):
    waves: int
    fetch_per_wave: int
    total_fetch_cap: int
    gap_fetch_per_wave: int
    per_subq_target: int
    coverage_min_strong: int
    gap_waves_max: int
    digest_head_chars: int = 6000
    digest_mid_chars: int = 2000
    digest_chunks_long_doc: int = 1
    digest_concurrency: int = 4
    rerank: bool = True
    writer_min_relevance: int = 3
    quote_verify_fuzzy: float = 0.9


class ComposeConfig(BaseModel):
    concurrency: int = 3
    allocate_claims: bool = True
    dedupe_pass: bool = True
    section_min_notes: int = 2
    analyst: bool = True


class PanelConfig(BaseModel):
    enabled: bool = True
    grounding: bool = True
    coverage: bool = True
    precision_mode: Literal["deterministic", "llm", "off"] = "deterministic"
    max_compose_revisions: int = 1
    coverage_to_research: bool = True
    diversity_rerun: bool = True
    rewrite_flag_ratio: float = 0.30


# Shipped defaults per depth (spec §2.4 table; the bench tunes them).
# total_fetch_cap and exa.max_searches INCLUDE the gap reserve.
PRESETS: dict[Depth, dict] = {
    "quick": {
        "research": {"waves": 1, "fetch_per_wave": 10, "total_fetch_cap": 12,
                     "gap_fetch_per_wave": 0, "gap_waves_max": 0,
                     "per_subq_target": 2, "coverage_min_strong": 1,
                     "digest_chunks_long_doc": 1, "writer_min_relevance": 2},
        "search": {"exa": {"max_searches": 6, "max_contents_pages": 12}, "arxiv": {}},
        "fetch": {},
        "compose": {"analyst": True},           # off when length == "tldr" (from_preset)
        "panel": {"grounding": True, "coverage": False, "precision_mode": "deterministic",
                  "max_compose_revisions": 0},
    },
    "standard": {
        "research": {"waves": 3, "fetch_per_wave": 14, "total_fetch_cap": 48,
                     "gap_fetch_per_wave": 8, "gap_waves_max": 2,
                     "per_subq_target": 3, "coverage_min_strong": 2,
                     "digest_chunks_long_doc": 2, "writer_min_relevance": 3},
        "search": {"exa": {"max_searches": 16, "max_contents_pages": 40}, "arxiv": {}},
        "fetch": {},
        "compose": {"analyst": True},
        "panel": {"grounding": True, "coverage": True, "precision_mode": "deterministic",
                  "max_compose_revisions": 1},
    },
    "deep": {
        "research": {"waves": 4, "fetch_per_wave": 16, "total_fetch_cap": 78,
                     "gap_fetch_per_wave": 10, "gap_waves_max": 3,
                     "per_subq_target": 4, "coverage_min_strong": 3,
                     "digest_chunks_long_doc": 3, "writer_min_relevance": 3},
        "search": {"exa": {"max_searches": 26, "max_contents_pages": 64}, "arxiv": {}},
        "fetch": {},
        "compose": {"analyst": True},
        "panel": {"grounding": True, "coverage": True, "precision_mode": "deterministic",
                  "max_compose_revisions": 1},
    },
    "exhaustive": {
        "research": {"waves": 5, "fetch_per_wave": 20, "total_fetch_cap": 108,
                     "gap_fetch_per_wave": 12, "gap_waves_max": 3,
                     "per_subq_target": 5, "coverage_min_strong": 3,
                     "digest_chunks_long_doc": 4, "writer_min_relevance": 3},
        "search": {"exa": {"max_searches": 36, "max_contents_pages": 96}, "arxiv": {}},
        "fetch": {},
        "compose": {"analyst": True},
        "panel": {"grounding": True, "coverage": True, "precision_mode": "deterministic",
                  "max_compose_revisions": 2},
    },
}

_DEFAULT_DEPTH: dict[str, Depth] = {"tldr": "quick", "short": "quick", "medium": "standard",
                                    "long": "deep", "lecture": "deep"}
_KNOB_GROUPS = ("research", "search", "fetch", "compose", "panel")
_OVERRIDABLE = (*_KNOB_GROUPS, "recency", "archetype_override")    # roots a dotted override may address
_POOL_KNOBS = ("research.digest_concurrency", "fetch.concurrency", "compose.concurrency")
_RATIO_KNOBS = ("research.quote_verify_fuzzy", "panel.rewrite_flag_ratio")


def _leaves(data: dict, prefix: str = "") -> Iterator[tuple[str, object]]:
    """Dotted key → value for every leaf of a nested dict (an empty dict is a leaf)."""
    for key, value in data.items():
        if isinstance(value, dict) and value:
            yield from _leaves(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


class RunConfig(BaseModel):
    schema_version: int = 1
    length: Length
    depth: Depth
    preset_origin: str                                         # preset name the values came from
    recency: Literal["d", "w", "m", "y"] | None = None         # None → brief's time-sensitivity hint decides
    archetype_override: str | None = None
    research: ResearchConfig
    search: SearchConfig
    fetch: FetchConfig
    compose: ComposeConfig
    panel: PanelConfig
    engines: EnginesSnapshot
    created_at: str
    origin: Literal["telegram", "webapp", "bench", "live_check", "demo", "legacy"] = "telegram"
    legacy: bool = False

    @model_validator(mode="after")
    def _bounds(self) -> RunConfig:
        """No negative knob; worker pools >= 1; ratios <= 1; the gap reserve fits the caps that include it."""
        knobs = dict(_leaves(self.model_dump(include=set(_KNOB_GROUPS))))
        for key, value in knobs.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0:
                raise ValueError(f"{key} must be >= 0 (got {value})")
        for key in _POOL_KNOBS:
            if knobs[key] < 1:
                raise ValueError(f"{key} must be >= 1 (got {knobs[key]})")
        for key in _RATIO_KNOBS:
            if knobs[key] > 1:
                raise ValueError(f"{key} must be <= 1 (got {knobs[key]})")
        if self.first_pass_fetch_cap() < 0:
            raise ValueError(f"research.total_fetch_cap ({self.research.total_fetch_cap}) must include the gap "
                             f"reserve gap_waves_max × gap_fetch_per_wave ({self.gap_reserve()})")
        if self.search.exa.max_searches < self.exa_search_reserve():
            raise ValueError(f"search.exa.max_searches ({self.search.exa.max_searches}) must include the gap "
                             f"reserve gap_waves_max × 2 ({self.exa_search_reserve()})")
        return self

    @staticmethod
    def default_depth(length: str) -> Depth:
        """tldr, short → quick · medium → standard · long, lecture → deep."""
        if length not in _DEFAULT_DEPTH:
            raise ValueError(f"unknown length {length!r}")
        return _DEFAULT_DEPTH[length]

    @classmethod
    def from_preset(cls, *, length: str, depth: str | None = None, engines: EnginesSnapshot,
                    overrides: dict = {}, origin: str) -> RunConfig:
        """Preset for ``depth`` (default: ``default_depth(length)``), then dotted-key
        overrides such as ``{"research.waves": 2}``; unknown keys raise ValueError."""
        depth = depth or cls.default_depth(length)
        if depth not in PRESETS:
            raise ValueError(f"unknown depth {depth!r}")
        preset = copy.deepcopy(PRESETS[depth])
        if depth == "quick" and length == "tldr":   # table: compose.analyst "on unless length=tldr"
            preset["compose"]["analyst"] = False
        data = cls(length=length, depth=depth, preset_origin=depth, engines=engines, origin=origin,
                   created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   **preset).model_dump()
        knobs = dict(_leaves({k: data[k] for k in _OVERRIDABLE}))
        for key, value in overrides.items():
            if key not in knobs:
                raise ValueError(f"unknown run_config override {key!r} (dotted knob keys, e.g. 'research.waves')")
            *path, leaf = key.split(".")
            node = data
            for part in path:
                node = node[part]
            node[leaf] = value
        return cls.model_validate(data)

    def gap_reserve(self) -> int:
        return self.research.gap_waves_max * self.research.gap_fetch_per_wave

    def first_pass_fetch_cap(self) -> int:
        return self.research.total_fetch_cap - self.gap_reserve()

    def exa_search_reserve(self) -> int:
        return self.research.gap_waves_max * 2

    def exa_cost_estimate_usd(self) -> float:
        """Upper-bound Exa spend from the configured prices; display it as "≈"."""
        exa = self.search.exa
        return (exa.max_searches * exa.price_per_search_usd
                + (exa.max_searches * exa.num_results + exa.max_contents_pages) * exa.price_per_page_usd)


def _legacy(cls, *, engines: EnginesSnapshot) -> RunConfig:
    """Pre-v4 checkpoints: standard preset, legacy=True, origin="legacy".
    length="medium" is the length whose default depth is standard."""
    rc = cls.from_preset(length="medium", depth="standard", engines=engines, origin="legacy")
    return rc.model_copy(update={"legacy": True})


# Spec §2.4 names both a field and this constructor ``legacy``. Pydantic would
# take a classmethod defined in the class body as the field's default, so it is
# attached after the class is built; instances still read the bool field.
# A subclass would inherit the method as its default — do not subclass RunConfig.
RunConfig.legacy = classmethod(_legacy)


def require(state: Mapping) -> RunConfig:
    """The run's RunConfig; a missing or invalid ``run_config`` raises ValueError."""
    raw = state.get("run_config")
    if raw is None:
        raise ValueError("state['run_config'] missing — every entry point must set it")
    try:
        return RunConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"state['run_config'] invalid — {e}") from e


async def ensure_in_checkpoint(graph, cfg: dict, *, engines: EnginesSnapshot) -> RunConfig:
    """Resume paths: inject ``RunConfig.legacy()`` (visibly, via ``errors``) into a
    pre-v4 checkpoint that has no ``run_config``; return the thread's config."""
    snap = await graph.aget_state(cfg)
    if snap.values.get("run_config") is not None:
        return require(snap.values)
    rc = RunConfig.legacy(engines=engines)
    await graph.aupdate_state(cfg, {"run_config": rc.model_dump(),
                                    "errors": ["legacy: run_config defaults applied (pre-v4 checkpoint)"]})
    return rc
