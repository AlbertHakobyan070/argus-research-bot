"""Self-contained v3 end-to-end demo — drive the full research engine.

Strategy
--------
The two live-network pieces (the search wave and the page fetches) are
stubbed to a small controlled corpus so the run is deterministic and
doesn't depend on live search. Everything else is the REAL v3 engine
hitting the live FreeLLMAPI proxy: brief scoping, the per-source digest
that reads each document into evidence notes, the section writers, and
the review panel. The point is to show the engine producing a grounded,
cited report from evidence the LLM actually read.

What you should see
-------------------
1. Routing: cheap=<X> strong=<Y> judge=<Z>
2. Plan gate: pause after `scout` with the corpus sources shown.
3. Resume -> research (fetch + digest) -> outline -> compose -> panel
   (revise loop if the panel flags a section) -> report_builder.
4. Report gate: report.md + report.pdf on disk, every finding cited to a
   corpus URL; then a final resume runs to deliver.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Clean Python path so the argus venv isn't poisoned by the parent Hermes
# venv's site-packages (pydantic_core ABI mismatch).
os.environ.pop("PYTHONPATH", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEMO_TOPIC = "LLM agent benchmarks and primary-source evaluation suites"
DEMO_OUT = ROOT / "demo_output"
DEMO_OUT.mkdir(parents=True, exist_ok=True)

# A small controlled corpus. Each item becomes one stubbed search hit and
# one fetched markdown document the digest step reads.
CORPUS = [
    {
        "url": "https://arxiv.org/abs/2310.12931",
        "title": "Holistic Evaluation of Language Models (HELM)",
        "kind": "paper",
        "body": (
            "HELM is a holistic framework for evaluating foundation models "
            "across many scenarios. It measures accuracy, calibration, "
            "robustness, fairness, bias, toxicity, and efficiency across "
            "seven broad task categories, and standardizes prompting so "
            "results are comparable across models."
        ),
    },
    {
        "url": "https://github.com/openai/simple-evals",
        "title": "OpenAI simple-evals",
        "kind": "repo",
        "body": (
            "simple-evals is a lightweight library for evaluating language "
            "models. It contains MMLU, MATH, GPQA, DROP, and other evals, "
            "and reports zero-shot chain-of-thought accuracy to reduce "
            "prompt-engineering confounds across models."
        ),
    },
    {
        "url": "https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard",
        "title": "HuggingFace Open LLM Leaderboard",
        "kind": "official_doc",
        "body": (
            "The Open LLM Leaderboard ranks open-source LLMs across IFEval, "
            "BBH, MATH, GPQA, MUSR, and MMLU-Pro. It runs a fixed harness so "
            "community submissions are directly comparable, and reports "
            "normalized scores per benchmark."
        ),
    },
]


def _write_corpus() -> dict[str, str]:
    """Write each corpus item to a markdown file; return url -> path."""
    corpus_dir = DEMO_OUT / "_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for it in CORPUS:
        p = corpus_dir / (f"{abs(hash(it['url']))}.md")
        p.write_text(f"# {it['title']}\n\n{it['body']}\n", encoding="utf-8")
        paths[it["url"]] = str(p)
    return paths


def main() -> int:
    corpus_paths = _write_corpus()

    from argus.graph import research as research_mod
    from argus.graph import scout as scout_mod

    # Stub the search wave: return the corpus as hits tagged for every
    # plausible sub-question index. Used by both scout and follow-up waves.
    def fake_wave(queries, **kw):
        tags = sorted({i for q in queries for i in (q.get("sub_qs") or [])}) \
            or list(range(6))
        hits = [{
            "url": it["url"], "title": it["title"],
            "snippet": it["body"][:200], "kind": it["kind"],
            "provider": "demo", "sub_qs": tags, "text": "",
            "published": "", "source": "demo", "summary": it["body"][:200],
        } for it in CORPUS]
        return hits, []

    # Stub the fetch: point each source at its pre-written corpus markdown.
    def fake_fetch(src):
        from argus.graph.state import FetchedItem
        url = src.get("url", "")
        path = corpus_paths.get(url)
        if not path:
            return None, [f"demo: no corpus entry for {url}"]
        body = Path(path).read_text(encoding="utf-8")
        return FetchedItem(
            url=url, title=src.get("title", ""), markdown_path=path,
            section=src.get("kind", ""), excerpt=body[:600],
            sub_qs=src.get("sub_qs") or [], provider="demo",
        ).model_dump(), []

    scout_mod.run_query_wave = fake_wave
    research_mod.run_query_wave = fake_wave
    research_mod.fetch_one_source = fake_fetch

    # Reports into demo_output (not the vault).
    os.environ["ARGUS_REPORTS_ROOT"] = str(DEMO_OUT)
    from argus import config as cfg_mod
    cfg_mod._cached = None

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    from argus.config import get_settings
    from argus.graph import build_graph
    from argus.llm import resolve_tier

    s = get_settings()
    print("=== Argus v3 end-to-end demo ===")
    print(f"Topic: {DEMO_TOPIC}")
    print(f"Reports root: {s.reports_root}")
    print(f"FreeLLMAPI base: {s.freellmapi_base_url}")
    print(f"Routing -> cheap={resolve_tier('cheap')} "
          f"strong={resolve_tier('strong')} judge={resolve_tier('judge')}")
    print()

    g = build_graph(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "demo"}}
    state_in = {
        "thread_id": "demo", "user_id": 0, "user_request": DEMO_TOPIC,
        "length": "medium",
        "messages": [], "plan": None, "sources": [], "fetched": [],
        "findings": [], "draft_md": "", "revision_notes": [],
        "revision_rounds": 0, "model_calls": [], "hitl": {"pending": False},
    }

    print(">>> first leg: intake -> brief -> scout -> PLAN GATE")
    g.invoke(state_in, config=cfg)
    snap = g.get_state(cfg)
    assert tuple(snap.next) == ("research",), (
        f"expected the plan gate (next=research), got {snap.next!r}")
    cur = snap.values
    print(f"    brief sub-questions: "
          f"{len((cur.get('brief') or {}).get('sub_questions') or [])}")
    print(f"    scout sources: {len(cur.get('sources') or [])}")
    print()

    print(">>> APPROVE -> research/outline/compose/panel -> REPORT GATE")
    g.invoke(Command(resume=True), config=cfg)
    snap = g.get_state(cfg)
    # Drive any panel revise loops until the report-preview pause.
    guard = 0
    while snap.next and tuple(snap.next) != ("deliver",) and guard < 8:
        g.invoke(Command(resume=True), config=cfg)
        snap = g.get_state(cfg)
        guard += 1

    print(">>> SEND -> deliver -> END")
    while snap.next:
        g.invoke(Command(resume=True), config=cfg)
        snap = g.get_state(cfg)

    final = snap.values
    paths = final.get("report_paths") or {}
    print()
    print("=== FINAL ===")
    print(f"evidence notes: {len(final.get('evidence') or [])}")
    print(f"findings:       {len(final.get('findings') or [])}")
    print(f"fetched:        {len(final.get('fetched') or [])}")
    print(f"verdict:        {(final.get('review_verdict') or {}).get('verdict')}")
    print(f"md:             {paths.get('md')}")
    print(f"pdf:            {paths.get('pdf')}")
    print(f"calls:          {len(final.get('model_calls') or [])}")
    for c in (final.get("model_calls") or [])[:10]:
        print(f"  - {str(c.get('tier')):6s} req={c.get('requested_model')} "
              f"served={c.get('served_model')}")

    transcript = DEMO_OUT / "demo_transcript.json"
    transcript.write_text(json.dumps({
        "topic": DEMO_TOPIC,
        "report_paths": paths,
        "n_evidence": len(final.get("evidence") or []),
        "n_findings": len(final.get("findings") or []),
        "n_fetched": len(final.get("fetched") or []),
        "verdict": (final.get("review_verdict") or {}).get("verdict"),
        "coverage": final.get("coverage") or {},
        "model_calls": final.get("model_calls") or [],
        "ts": datetime.now().astimezone().isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTranscript saved to {transcript}")

    if paths.get("md"):
        text = Path(paths["md"]).read_text(encoding="utf-8")
        print("\n--- REPORT (first 2000 chars) ---")
        print(text[:2000])

    findings = final.get("findings") or []
    cited = [f for f in findings if (f.get("citation_urls") or [])]
    ok = bool(cited) and bool(paths.get("md")) and Path(
        paths.get("md", "")).exists()
    print(f"\nOVERALL: {'PASS' if ok else 'BLOCK'} "
          f"({len(cited)}/{len(findings)} findings cited)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
