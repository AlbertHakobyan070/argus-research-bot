"""Argus Deep Research Bench — runner.

Drives the Argus graph against each query in queries.jsonl, captures
the final markdown report, and writes per-query JSONL records to
results/raw.jsonl, plus the bench v2 result layout per query (v4 spec
§3 WP-G), which ``bench.metrics`` and ``bench.compare`` read:

    <out dir>/<qid>/state.json      the full final state
    <out dir>/<qid>/report.md       copies of the run's report files, when
    <out dir>/<qid>/metadata.json   it reached report_builder
    <out dir>/<qid>/evidence.json   (once WP-C writes it)
    <out dir>/<qid>/record.json     the run's record (status, duration_s, …)

``scripts/live_check.py`` drives its one run through ``_run_one`` too.

Designed for comparison runs against the LIVE proxy + live search
(nothing is stubbed):
  - Engines: the committed argus_engines.json plus the local file (the
    bot's /engines changes live in its library and are not read),
    resolved once per bench run; ``--profile`` picks a profile. The
    proxy reroutes, so runs are comparable only while it serves the
    same models.
  - Every query starts with its ``run_config`` (v4 spec §2.4):
    ``--length`` and ``--depth`` (default: the length's preset),
    ``origin="bench"``.
  - Auto-resumes through LangGraph interrupts (plan approval + report
    preview) without user input — the goal is to measure the graph's
    output, not the HITL UX.
  - Times each run end-to-end.

Usage:
    PYTHONPATH="" ./venv/Scripts/python.exe -m bench.runner \\
        --queries bench/queries.jsonl \\
        --out bench/results/<label>/raw.jsonl \\
        --length short [--depth standard] [--profile free-proxy]

Notes:
    - The full final state of each query (the checkpoint's values, also
      after a timeout or an error) is written to
      ``<out dir>/<qid>/state.json``; the record's ``state_path`` names it.
    - Always uses a fresh checkpointer per query (MemorySaver), so
      thread_ids don't collide and previous state doesn't leak.
    - Pauses at the same two static interrupts the bot uses (plan gate
      after scout, report gate before deliver) and resumes each with
      ``Command(resume=True)``; LangGraph emits ``__interrupt__`` for
      static breakpoints in stream_mode="updates" (verified on 1.2.8).
      Every astream call passes ``stream_mode=["updates", "custom"]``,
      ``durability="sync"`` and the run's one ``control=RunControl()``
      (spec §1.1); nothing here requests a drain.
    - ``report_md`` is compose's ``draft_md`` (pre-report_builder: no
      title block, no citation sanitising). The finished report.md is
      written under Settings.reports_root: the bench sets
      ARGUS_REPORTS_ROOT to ``<out dir>/reports`` unless it is already set
      (in the environment or .env), so a bench run never writes into the
      vault by default.
    - A query's run first deletes the five layout files its last run left
      in ``<out dir>/<qid>/``: a failed rerun leaves its own record (and
      state), never the last run's report beside them.
    - Output is one JSON object per line; downstream scorer.py reads
      this directly.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import RunControl
from langgraph.types import Command

from argus.engines import registry
from argus.engines.schema import EnginesSnapshot
from argus.graph.graph import build_graph
from argus.graph.run_config import PRESETS, RunConfig
from argus.graph.state import VALID_LENGTHS
from argus.library import Library


logger = logging.getLogger("argus.bench.runner")


async def _engines_snapshot(profile: str | None) -> EnginesSnapshot:
    """The engines every query of this bench run uses (resolved once). A
    ``profile`` is switched to in-process, validated as ``/engines use``
    does; the switch persists only into a throwaway in-memory library."""
    if profile is not None:
        lib = Library(":memory:")
        await lib.open()
        try:                    # an open aiosqlite connection keeps the interpreter from exiting
            await registry.load_config(library=lib)
            await registry.set_active_profile(profile)
        finally:
            await lib.close()
    return await registry.active_snapshot()


LAYOUT = ("state.json", "report.md", "metadata.json", "evidence.json", "record.json")


def _copy_report_files(paths: dict, qdir: Path) -> None:
    """The run's report.md and metadata.json (report_builder writes both in its
    folder) and its evidence.json when there is one, next to its state.json."""
    if not paths.get("md"):
        return
    qdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths["md"], qdir / "report.md")
    shutil.copyfile(Path(paths["folder"]) / "metadata.json", qdir / "metadata.json")
    if paths.get("evidence"):
        shutil.copyfile(paths["evidence"], qdir / "evidence.json")


async def _run_one(query_obj: dict, *, length: str, depth: str | None,
                   engines: EnginesSnapshot, timeout_s: int | None, state_dir: Path,
                   origin: str = "bench", overrides: dict | None = None) -> dict:
    """Run a single query end-to-end. Auto-resume through interrupts.
    ``timeout_s`` None: no limit (the live check); ``overrides``: dotted
    RunConfig keys, e.g. ``{"recency": "m"}``."""
    qid = query_obj["id"]
    query = query_obj["query"]
    qdir = state_dir / qid
    for name in LAYOUT:                 # the last run's files must not outlive a failed rerun
        (qdir / name).unlink(missing_ok=True)
    thread_id = f"bench-{qid}-{uuid.uuid4().hex[:8]}"
    user_id = 0  # bench user
    rc = RunConfig.from_preset(length=length, depth=depth, engines=engines, overrides=overrides,
                               origin=origin)

    # Fresh checkpointer per query — no state leakage.
    graph = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": thread_id}}
    control = RunControl()

    state_in = {
        "thread_id": thread_id,
        "user_id": user_id,
        "user_request": query,
        "mode": "deep",
        "length": length,
        "run_config": rc.model_dump(),
        "messages": [],
        "plan": None,
        "sources": [],
        "fetched": [],
        "findings": [],
        "draft_md": "",
        "revision_notes": [],
        "revision_rounds": 0,
        "model_calls": [],
        "hitl": {"pending": False},
        "plan_approved": True,  # no effect in v3: the gate still fires; _drive resumes it
    }

    record = {
        "id": qid,
        "query": query,
        "expected_domains": query_obj.get("expected_domains", []),
        "difficulty": query_obj.get("difficulty", "?"),
        "thread_id": thread_id,
        "length": length,
        "depth": rc.depth,
        "profile": engines.profile,
        "status": "started",
        "duration_s": 0.0,
        "report_md": None,
        "report_path": None,
        "n_sources": 0,
        "n_findings": 0,
        "interrupted_at": None,
        "error": None,
        "state_path": None,
    }
    t0 = time.monotonic()
    try:
        # Drive intake → … → scout, then auto-resume through both gates.
        async def _drive(initial_input, max_steps: int = 60):
            """Stream the graph, auto-resume on interrupt. Returns final state."""
            steps = 0
            last_interrupt = None
            # Run via .astream() and feed resume commands on interrupt.
            config = cfg
            current_input = initial_input
            while steps < max_steps:
                steps += 1
                interrupted = False
                async for mode, chunk in graph.astream(current_input, config=config,
                                                       stream_mode=["updates", "custom"],
                                                       durability="sync", control=control):
                    if mode == "updates" and "__interrupt__" in chunk:
                        interrupted = True
                        last_interrupt = chunk["__interrupt__"]
                        break
                if not interrupted:
                    break
                # Resume with True (approve)
                current_input = Command(resume=True)
            return last_interrupt

        await asyncio.wait_for(
            _drive(state_in),
            timeout=timeout_s,
        )

        # Pull the final state from the checkpointer.
        snap = await graph.aget_state(cfg)
        final = snap.values if snap else {}
        md = final.get("draft_md") or ""
        fetched = final.get("fetched") or []
        findings = final.get("findings") or []
        paths = final.get("report_paths") or {}
        _copy_report_files(paths, qdir)

        record.update({
            "status": "ok",
            "report_md": md[:20000],  # cap for JSONL sanity
            "report_path": paths.get("md"),
            "n_sources": len(fetched),
            "n_findings": len(findings),
        })
    except asyncio.TimeoutError:
        record.update({
            "status": "timeout",
            "error": f"exceeded {timeout_s}s",
        })
    except Exception as e:
        logger.exception("runner: query %s failed", qid)
        # str(), not repr(): an OSError's repr drops the file name
        record.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
    record["duration_s"] = round(time.monotonic() - t0, 2)
    # The full final state — after a timeout or an error, the last checkpoint.
    snap = await graph.aget_state(cfg)
    qdir.mkdir(parents=True, exist_ok=True)
    if snap.created_at is not None:
        state_path = qdir / "state.json"
        state_path.write_text(json.dumps(snap.values, indent=2, ensure_ascii=False, default=str),
                              encoding="utf-8")
        record["state_path"] = str(state_path)
    (qdir / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    return record


async def _main_async(args: argparse.Namespace) -> int:
    queries_path = Path(args.queries)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("ARGUS_REPORTS_ROOT"):            # before anything reads Settings
        os.environ["ARGUS_REPORTS_ROOT"] = str(out_path.parent / "reports")

    with queries_path.open(encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    engines = await _engines_snapshot(args.profile)
    logger.info("bench: running %d queries (length=%s, depth=%s, profile=%s, timeout=%ds)",
                len(queries), args.length, args.depth or "preset", engines.profile, args.timeout)

    with out_path.open("w", encoding="utf-8") as out:
        for q in queries:
            logger.info("bench: [%s] %s", q["id"], q["query"][:60])
            rec = await _run_one(q, length=args.length, depth=args.depth, engines=engines,
                                 timeout_s=args.timeout, state_dir=out_path.parent)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            logger.info("bench: [%s] %s in %.1fs (sources=%d, findings=%d)",
                        q["id"], rec["status"], rec["duration_s"],
                        rec["n_sources"], rec["n_findings"])

    logger.info("bench: wrote %d records to %s", len(queries), out_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Argus Deep Research Bench runner.")
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--length", default="short", choices=VALID_LENGTHS)
    parser.add_argument("--depth", default=None, choices=list(PRESETS),
                        help="Research depth preset (default: the length's own)")
    parser.add_argument("--profile", default=None,
                        help="Engines profile (default: the active one in the engines files)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-query timeout in seconds (default 300)")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("ARGUS_BENCH_LOG", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())