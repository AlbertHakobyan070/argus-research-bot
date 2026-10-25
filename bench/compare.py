"""Argus bench v2 — compare two result sets (v4 spec §3 WP-G).

    PYTHONPATH="" ./venv/Scripts/python.exe -m bench.compare --a bench/results/v3-baseline --b <dir>

Computes ``bench.metrics`` for both folders and prints one markdown table: a
row per metric, a column per query present in both; each cell reads
``a -> b``, with the difference when both values are numbers. Queries in only
one folder are named under the table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import metrics


def _cell(a, b) -> str:
    cell = f"{metrics.fmt(a)} -> {metrics.fmt(b)}"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (a, b)):
        cell += f" ({round(b - a, 3):+g})"
    return cell.replace("|", "\\|")


def render(a: dict[str, dict], b: dict[str, dict], *, a_label: str, b_label: str) -> str:
    shared = [q for q in a if q in b]
    if not shared:
        raise SystemExit(f"no query in both {a_label} and {b_label}")
    lines = [f"a = {a_label}", f"b = {b_label}", "",
             "| metric | " + " | ".join(shared) + " |", "|---" * (len(shared) + 1) + "|"]
    for name in metrics.METRICS:
        lines.append(f"| {name} | " + " | ".join(_cell(a[q][name], b[q][name]) for q in shared) + " |")
    for side, mine, other in (("a", a, b), ("b", b, a)):
        only = [q for q in mine if q not in other]
        if only:
            lines.append(f"\nonly in {side}: {', '.join(only)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Argus bench v2: metric deltas between two result sets.")
    parser.add_argument("--a", required=True, type=Path, help="the 'before' results folder")
    parser.add_argument("--b", required=True, type=Path, help="the 'after' results folder")
    args = parser.parse_args(argv)
    sys.stdout.write(render(metrics.collect(args.a), metrics.collect(args.b),
                            a_label=str(args.a), b_label=str(args.b)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
