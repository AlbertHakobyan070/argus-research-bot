"""SpendLedger (v4 spec §2.4, §2.8): every research budget through one lock.

``take(kind, n)`` is an atomic check-and-count against the cap of a capped
kind; ``count(kind, n)`` bumps an uncapped counter; ``add_usd`` sums the Exa
cost. Pool threads only call these; the node thread writes ``snapshot()`` to
``state["search_spend"]`` and every node rebuilds its ledger from it with
``for_run`` — the snapshot carries the per-kind counters, so it round-trips.
Pinned sources (``/append``) never call ``take``.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..graph.run_config import RunConfig

CAPPED = ("fetch_first", "fetch_gap", "fetch_extend",
          "exa_search_first", "exa_search_gap", "exa_search_extend", "exa_contents_pages")
COUNTERS = ("exa_text_pages", "arxiv_queries", "ddgs_queries", "github_queries", "dropped_undated")
_EXTEND_KINDS = ("fetch_extend", "exa_search_extend")


class SpendLedger:
    def __init__(self, caps: Mapping[str, int], spent: Mapping[str, Any] | None = None, *,
                 extend_unit: Mapping[str, int] | None = None):
        """``caps`` per capped kind (missing → 0); ``spent`` a previous
        ``snapshot()``; ``extend_unit`` what one Extend allowance adds to the
        caps of ``fetch_extend`` / ``exa_search_extend``."""
        unknown = set(caps) - set(CAPPED)
        if unknown:
            raise ValueError(f"unknown capped kinds {sorted(unknown)}")
        spent = spent or {}
        self._lock = threading.Lock()
        self._caps = {k: int(caps.get(k, 0)) for k in CAPPED}
        self._spent = {k: int(spent.get(k) or 0) for k in (*CAPPED, *COUNTERS)}
        self._usd = float(spent.get("exa_usd") or 0.0)
        self._extend_unit = {k: int((extend_unit or {}).get(k, 0)) for k in _EXTEND_KINDS}
        self._allowances = int(spent.get("extend_allowances") or 0)
        self._recency = spent.get("recency")

    @classmethod
    def for_run(cls, rc: RunConfig, search_spend: Mapping[str, Any] | None) -> SpendLedger:
        """The run's caps (spec §2.4): the first pass gets what the gap reserve
        leaves; each granted Extend adds ``fetch_per_wave`` fetches and
        ``max_searches // waves`` Exa searches."""
        exa = rc.search.exa
        unit = {"fetch_extend": rc.research.fetch_per_wave,
                "exa_search_extend": exa.max_searches // rc.research.waves}
        allowances = int((search_spend or {}).get("extend_allowances") or 0)
        caps = {"fetch_first": rc.first_pass_fetch_cap(), "fetch_gap": rc.gap_reserve(),
                "exa_search_first": exa.max_searches - rc.exa_search_reserve(),
                "exa_search_gap": rc.exa_search_reserve(),
                "exa_contents_pages": exa.max_contents_pages,
                **{k: allowances * v for k, v in unit.items()}}
        return cls(caps, search_spend, extend_unit=unit)

    def take(self, kind: str, n: int = 1) -> bool:
        """Count ``n`` units of ``kind`` if they fit under its cap; all or nothing."""
        if kind not in CAPPED:
            raise ValueError(f"take(): {kind!r} is not a capped kind {CAPPED}")
        with self._lock:
            if self._spent[kind] + n > self._caps[kind]:
                return False
            self._spent[kind] += n
            return True

    def count(self, kind: str, n: int = 1) -> None:
        if kind not in COUNTERS:
            raise ValueError(f"count(): {kind!r} is not an uncapped counter {COUNTERS}")
        with self._lock:
            self._spent[kind] += n

    def add_usd(self, x: float) -> None:
        with self._lock:
            self._usd += float(x)

    def remaining(self, kind: str) -> int:
        if kind not in CAPPED:
            raise ValueError(f"remaining(): {kind!r} is not a capped kind {CAPPED}")
        with self._lock:
            return max(0, self._caps[kind] - self._spent[kind])

    def spent(self, kind: str) -> int:
        with self._lock:
            return self._spent[kind]

    def grant_extend(self) -> None:
        """One user Extend: a fresh allowance on top of every other budget."""
        with self._lock:
            self._allowances += 1
            for k, v in self._extend_unit.items():
                self._caps[k] += v

    def set_recency(self, recency: dict | None) -> None:
        with self._lock:
            self._recency = recency

    def snapshot(self) -> dict:
        """``state["search_spend"]``: the spec §2.4 keys plus the per-kind counters."""
        with self._lock:
            s = dict(self._spent)
            return {**s,
                    "exa_searches": s["exa_search_first"] + s["exa_search_gap"] + s["exa_search_extend"],
                    "exa_usd": round(self._usd, 6),
                    "extend_allowances": self._allowances,
                    "recency": self._recency}


__all__ = ["SpendLedger", "CAPPED", "COUNTERS"]
