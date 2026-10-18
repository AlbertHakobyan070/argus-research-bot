"""Argus Deep Research Bench — runner.

Drives the Argus graph against each query in queries.jsonl, captures
the final markdown report, and writes per-query JSONL records to
results/raw.jsonl.

Designed for comparison runs against the LIVE proxy + live search
(nothing is stubbed):
  - No model pinning: tiers resolve through llm.resolve_tier against
    the live proxy, so runs are comparable only while it serves the
    same models.
  - Auto-resumes through LangGraph interrupts (plan approval + report
    preview) without user input — the goal is to measure the graph's
    output, not the HITL UX.
  - Times each run end-to-end.

Usage:
    PYTHONPATH="" ./venv/Scripts/python.exe -m bench.runner \\
        --queries bench/queries.jsonl \\
        --out bench/results/raw.jsonl \\
        --length short

Notes:
    - Always uses a fresh checkpointer per query (MemorySaver), so
      thread_ids don't collide and previous state doesn't leak.
    - Pauses at the same two static interrupts the bot uses (plan gate
      after scout, report gate before deliver) and resumes each with
      ``Command(resume=True)``; LangGraph emits ``__interrupt__`` for
      static breakpoints in stream_mode="updates" (verified on 1.2.8).
    - ``report_md`` is compose's ``draft_md`` (pre-report_builder: no
      title block, no citation sanitising). The finished report.md is
      written under Settings.reports_root — the vault unless
      ARGUS_REPORTS_ROOT is set.
    - Output is one JSON object per line; downstream scorer.py reads
      this directly.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from argus.graph.graph import build_graph


logger = logging.getLogger("argus.bench.runner")


async def _run_one(query_obj: dict, *, length: str, timeout_s: int) -> dict:
    """Run a single query end-to-end. Auto-resume through interrupts."""
    from argus.config import get_settings

    qid = query_obj["id"]
    query = query_obj["query"]
    thread_id = f"bench-{qid}-{uuid.uuid4().hex[:8]}"
    user_id = 0  # bench user

    # Fresh checkpointer per query — no state leakage.
    graph = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": thread_id}}

    state_in = {
        "thread_id": thread_id,
        "user_id": user_id,
        "user_request": query,
        "mode": "deep",
        "length": length,
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
        "status": "started",
        "duration_s": 0.0,
        "report_md": None,
        "report_path": None,
        "n_sources": 0,
        "n_findings": 0,
        "interrupted_at": None,
        "error": None,
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
                async for ev in graph.astream(current_input, config=config, stream_mode="updates"):
                    for node_name, delta in ev.items():
                        if node_name == "__interrupt__":
                            interrupted = True
                            last_interrupt = delta
                            break
                    if interrupted:
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
    except Exception as e:  # pragma: no cover — defensive
        logger.exception("runner: query %s failed", qid)
        record.update({"status": "error", "error": repr(e)})
    record["duration_s"] = round(time.monotonic() - t0, 2)
    return record


async def _main_async(args: argparse.Namespace) -> int:
    queries_path = Path(args.queries)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with queries_path.open(encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    logger.info("bench: running %d queries (length=%s, timeout=%ds)",
                len(queries), args.length, args.timeout)

    with out_path.open("w", encoding="utf-8") as out:
        for q in queries:
            logger.info("bench: [%s] %s", q["id"], q["query"][:60])
            rec = await _run_one(q, length=args.length, timeout_s=args.timeout)
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
    parser.add_argument("--length", default="short",
                        choices=["quick", "short", "medium", "long", "lecture"])
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