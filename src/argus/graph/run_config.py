"""Per-run configuration (v4 spec §2.4): knob models, depth presets, budget helpers.

Every run carries one in its checkpoint (WP-D): new-run entry points set
``state["run_config"]``, every resume entry point calls
``ensure_in_checkpoint`` first. No node reads it yet; the rule for every
node is ``rc = run_config.require(state)`` (never ``state.get``).
"""
from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, model_validator

from ..engines.schema import EnginesSnapshot, ExaPrices
from .state import Length

Depth = Literal["quick", "standard", "deep", "exhaustive"]


class ExaConfig(BaseModel):
    max_searches: int
    max_contents_pages: int
    search_type: Literal["auto", "fast", "instant"] = "auto"
    num_results: int = 8
    text_max_chars: int = 12000
    text_max_chars_paper: int = 20000


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


class FetchConfig(BaseModel):
    chain: list[Literal["exa_text", "native", "exa_contents", "intel_snatch", "intel_crawl", "intel_stealth"]] = [
        "exa_text", "native", "exa_contents", "intel_snatch", "intel_crawl", "intel_stealth"]
    max_chars_on_disk: int = 60000
    max_bytes: int = 25_000_000                 # download cap; PDFs are often > 5 MB
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


class ComposeConfig(BaseModel):
    concurrency: int = 3
    allocate_claims: bool = True
    dedupe_pass: bool = True
    section_min_notes: int = 2
    analyst: bool = True
    calculations: Literal["auto", "off"] = "auto"   # graph/calc.py: worked numbers when the request asks for them


class PanelConfig(BaseModel):
    enabled: bool = True
    grounding: bool = True
    coverage: bool = True
    precision_mode: Literal["deterministic", "off"] = "deterministic"
    judge_batch_size: int = 10
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
_OVERRIDABLE = (*_KNOB_GROUPS, "recency")                          # roots a dotted override may address
# The bounds written inline in spec §2.4, as (floor, ceiling). Every other
# numeric knob only has the floor 0.
_BOUNDS: dict[str, tuple[float, float]] = {
    "search.exa.max_searches": (0, 60), "search.exa.max_contents_pages": (0, 200),
    "search.exa.num_results": (1, 20), "search.arxiv.max_queries_per_run": (0, 6),
    "fetch.concurrency": (1, 8),
    "research.waves": (1, 6), "research.fetch_per_wave": (1, 30), "research.total_fetch_cap": (0, 150),
    "research.gap_fetch_per_wave": (0, 20), "research.per_subq_target": (1, 8),
    "research.coverage_min_strong": (0, 5), "research.gap_waves_max": (0, 3),
    "research.digest_chunks_long_doc": (1, 6), "research.digest_concurrency": (1, 8),
    "research.writer_min_relevance": (0, 5),
    "compose.concurrency": (1, 6), "compose.section_min_notes": (1, 5),
    "panel.judge_batch_size": (4, 20), "panel.max_compose_revisions": (0, 3),
    "panel.rewrite_flag_ratio": (0, 1),
}


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
    recency: Literal["d", "w", "m", "y", "any"] | None = None  # None → brief's time-sensitivity hint decides; "any" → no recency filter, even for a time-sensitive brief
    research: ResearchConfig
    search: SearchConfig
    fetch: FetchConfig
    compose: ComposeConfig
    panel: PanelConfig
    engines: EnginesSnapshot
    overrides_applied: dict[str, Any] = {}                     # the dotted overrides from_preset applied (console / bench)
    created_at: str
    origin: Literal["telegram", "webapp", "bench", "live_check", "demo", "legacy"] = "telegram"
    legacy: bool = False

    @model_validator(mode="after")
    def _bounds(self) -> RunConfig:
        """Every numeric knob within its §2.4 bounds (floor 0 where the spec gives
        none); the gap reserve fits the caps that include it."""
        for key, value in _leaves(self.model_dump(include=set(_KNOB_GROUPS))):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            floor, ceiling = _BOUNDS.get(key, (0, None))
            if value < floor:
                raise ValueError(f"{key} must be >= {floor} (got {value})")
            if ceiling is not None and value > ceiling:
                raise ValueError(f"{key} must be <= {ceiling} (got {value})")
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
                    overrides: Mapping[str, Any] | None = None, origin: str) -> RunConfig:
        """Preset for ``depth`` (default: ``default_depth(length)``), then dotted-key
        overrides such as ``{"research.waves": 2}``, recorded in ``overrides_applied``;
        unknown keys raise ValueError."""
        depth = depth or cls.default_depth(length)
        if depth not in PRESETS:
            raise ValueError(f"unknown depth {depth!r}")
        overrides = copy.deepcopy(dict(overrides or {}))
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
        data["overrides_applied"] = overrides
        return cls.model_validate(data)

    @classmethod
    def for_legacy_checkpoint(cls, *, engines: EnginesSnapshot, length: str) -> RunConfig:
        """Pre-v4 checkpoints: the preset of ``default_depth(length)`` for the
        checkpoint's own length, marked ``legacy=True``, ``origin="legacy"``."""
        rc = cls.from_preset(length=length, engines=engines, origin="legacy")
        return rc.model_copy(update={"legacy": True})

    def rebuild(self, *, length: str, depth: str | None = None, engines: EnginesSnapshot) -> RunConfig:
        """Plan-gate changes: ``from_preset`` for the new length/depth/engines,
        re-applying ``overrides_applied`` so console overrides are never silently
        dropped; ``origin`` and ``legacy`` carry over."""
        rc = self.from_preset(length=length, depth=depth, engines=engines,
                              overrides=self.overrides_applied, origin=self.origin)
        return rc.model_copy(update={"legacy": self.legacy})

    def gap_reserve(self) -> int:
        return self.research.gap_waves_max * self.research.gap_fetch_per_wave

    def first_pass_fetch_cap(self) -> int:
        return self.research.total_fetch_cap - self.gap_reserve()

    def exa_search_reserve(self) -> int:
        return self.research.gap_waves_max * 2

    def exa_cost_estimate_usd(self, prices: ExaPrices | None) -> float | None:
        """Upper-bound Exa spend at ``prices`` (``EnginesConfig.pricing.exa``); display
        it as "≈". No prices configured → None, shown as "n/a". Searches at the
        search price plus ``/contents`` pages at the page price: inline search
        text is not billed (measured, WP-B — docs/v4-notes/wp-b.md)."""
        if prices is None:
            return None
        exa = self.search.exa
        return exa.max_searches * prices.price_per_search_usd + exa.max_contents_pages * prices.price_per_page_usd


def require(state: Mapping) -> RunConfig:
    """The run's RunConfig. A missing or invalid ``run_config``, or one whose
    ``length`` differs from a set ``state["length"]``, raises ValueError."""
    raw = state.get("run_config")
    if raw is None:
        raise ValueError("state['run_config'] missing — every entry point must set it")
    try:
        rc = RunConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"state['run_config'] invalid — {e}") from e
    length = state.get("length")
    if length is not None and length != rc.length:
        raise ValueError(f"state['run_config'] has length {rc.length!r} but state['length'] is {length!r}")
    return rc


async def ensure_in_checkpoint(graph, cfg: dict, *, engines: EnginesSnapshot,
                               as_node: str | None = None) -> RunConfig:
    """Resume paths: the thread's RunConfig. A pre-v4 checkpoint (no ``run_config``)
    gets ``for_legacy_checkpoint`` for its own length, recorded in ``errors`` and
    written ``as_node`` when given (the caller's node that keeps ``next``; None lets
    LangGraph infer it, the last node that ran). A thread with no checkpoint, or a
    checkpoint without ``length``, raises ValueError and nothing is written."""
    snap = await graph.aget_state(cfg)
    thread_id = cfg["configurable"]["thread_id"]
    if snap.created_at is None:
        raise ValueError(f"no checkpoint for thread {thread_id!r}")
    if snap.values.get("run_config") is not None:
        return require(snap.values)
    length = snap.values.get("length")
    if length is None:
        raise ValueError(f"checkpoint for thread {thread_id!r} has no length — cannot build its legacy run_config")
    rc = RunConfig.for_legacy_checkpoint(engines=engines, length=length)
    await graph.aupdate_state(cfg, {"run_config": rc.model_dump(),
                                    "errors": ["legacy: run_config defaults applied (pre-v4 checkpoint)"]},
                              as_node=as_node)
    return rc
