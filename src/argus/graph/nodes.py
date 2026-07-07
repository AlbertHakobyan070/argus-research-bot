"""Argus LangGraph node implementations.

Each node is a function (state) -> partial state update. We deliberately
keep the implementations small and obvious rather than reaching for
prebuilt agents — the contract says "explicit node/subgraph, no black-box
prebuilt agent".
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .. import llm
from ..tools import (
    HarvestResult, harvest_sources, snatch_url, crawl_url,
    normalize_to_markdown, markdown_to_pdf,
)
from .state import (
    ArgusState, Finding, FetchedItem, PlannedSource,
    ResearchPlan, ReviewVerdict,
)

logger = logging.getLogger("argus.nodes")


# ---------------------------------------------------------------------------
# intake
# ---------------------------------------------------------------------------

INTAKE_SYSTEM = """You are the intake classifier for Argus, a research bot.

Given the user's raw request, decide:
- mode: "quick" for short factual / definitional questions; "deep" for
  anything that wants a report with sources.
- refined_query: a clean version of the user's request to feed the planner.

Return ONLY valid JSON matching the schema:
{"mode": "quick" | "deep", "refined_query": "..."}
"""


def intake_node(state: ArgusState) -> dict:
    """Classify request as quick vs deep and clean it up."""
    raw = state.get("user_request") or ""
    chat = llm.chat_for_tier("cheap", temperature=0.0, max_tokens=200)
    resp = chat.invoke([
        SystemMessage(content=INTAKE_SYSTEM),
        HumanMessage(content=raw[:4000]),
    ])
    rec = llm.record_from_response("cheap", "cheap", resp)
    try:
        parsed = _parse_json_obj(resp.content)
    except Exception:
        parsed = {"mode": "deep" if len(raw) > 80 else "quick",
                  "refined_query": raw.strip()}
    mode = parsed.get("mode", "deep")
    if mode not in ("quick", "deep"):
        mode = "deep"
    return {
        "mode": mode,
        "user_request": parsed.get("refined_query", raw).strip() or raw,
        "model_calls": [rec.model_dump()],
        "messages": [{"role": "user", "content": raw, "ts": _ts()}],
    }


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You are Argus's research planner.

Produce a structured research plan that maximises the chance of finding
*primary-source* evidence: arXiv preprints, HF models, official docs,
named engineers' blogs, GitHub repos, Hacker News threads. SEO content
farms are banned as evidence — never include them in `planned_sources`.

Return JSON only, matching:
{
  "sub_questions": ["...", "..."],
  "planned_sources": [
    {"kind": "paper|repo|news|blog|official_doc|search_result",
     "query": "search string or paper title",
     "target_url": "https://... (only if you have a specific URL)",
     "rationale": "why this source is primary"}
  ],
  "must_have_keywords": ["...", "..."],
  "summary": "1-2 sentence plan summary"
}

Aim for 4-7 sub_questions and 6-12 planned_sources. Mark each source's
kind correctly. Prefer primary kinds (paper/repo/official_doc) when you
know the right venue.
"""


def planner_node(state: ArgusState) -> dict:
    """LLM drafts a research plan. Triggers HITL interrupt."""
    chat = llm.chat_for_tier("strong", temperature=0.2, max_tokens=1200)
    resp = chat.invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content=(
            f"Topic / question: {state['user_request']}\n\n"
            "Draft the research plan."
        )),
    ])
    rec = llm.record_from_response("strong", llm.resolve_tier("strong"), resp)
    try:
        data = _parse_json_obj(resp.content)
        plan = ResearchPlan.model_validate(data)
    except Exception as e:
        logger.warning("Planner returned non-JSON (%s); falling back.", e)
        plan = ResearchPlan(
            sub_questions=[state["user_request"]],
            planned_sources=[PlannedSource(
                kind="search_result",
                query=state["user_request"],
                rationale="Fallback: a single web search.",
            )],
            summary=resp.content[:300],
        )
    return {
        "plan": plan.model_dump(),
        "model_calls": [rec.model_dump()],
        "messages": [{"role": "assistant", "content": (
            f"📋 Drafted plan with {len(plan.planned_sources)} sources. "
            "Awaiting your approval.")}, {"ts": _ts()}],
        # Set HITL pending; the bot layer will surface this and the
        # LangGraph interrupt will pause the run.
        "hitl": {"pending": True, "kind": "plan_approval",
                 "ctx": {"plan": plan.model_dump()}},
    }


# ---------------------------------------------------------------------------
# researcher
# ---------------------------------------------------------------------------

def researcher_node(state: ArgusState) -> dict:
    """Gather candidate URLs from intel-radar + arXiv + web search.

    Honours any pre-seeded ``state["sources"]`` (used by the demo to
    inject a static corpus) — we dedupe by URL, so re-adding them is
    a no-op.
    """
    plan = ResearchPlan.model_validate(state.get("plan") or {})
    sources: list[dict] = list(state.get("sources") or [])
    seen = {s["url"] for s in sources if s.get("url")}
    # Use harvest for primary-source radar (papers/repos/news/blogs).
    try:
        harvest = harvest_sources(hours=72, top=6,
                                  sections="papers,repos,news,blogs")
        for item in harvest.items[:25]:
            if item.url in seen:
                continue
            if _matches_plan(item.title + " " + item.summary, plan):
                sources.append({
                    "kind": item.section or "search_result",
                    "title": item.title,
                    "url": item.url,
                    "summary": item.summary,
                    "source": "intel-radar",
                })
                seen.add(item.url)
    except Exception as e:
        logger.warning("harvest failed: %s", e)

    # Also try arXiv search if any planned source is a "paper".
    if any(s.kind == "paper" for s in plan.planned_sources):
        try:
            arxiv_items = _arxiv_search(plan)
            for it in arxiv_items:
                if it["url"] in seen:
                    continue
                sources.append(it)
                seen.add(it["url"])
        except Exception as e:
            logger.warning("arxiv search failed: %s", e)

    # And any explicit target URLs from the plan.
    for s in plan.planned_sources:
        if s.target_url and s.target_url.startswith("http"):
            if s.target_url in seen:
                continue
            sources.append({
                "kind": s.kind,
                "title": s.query or s.target_url,
                "url": s.target_url,
                "summary": s.rationale,
                "source": "planner",
            })
            seen.add(s.target_url)

    sources = sources[:18]
    return {
        "sources": sources,
        "messages": [{"role": "assistant",
                      "content": f"🔍 {len(sources)} candidate sources."}],
    }


def _matches_plan(text: str, plan: ResearchPlan) -> bool:
    text = text.lower()
    keywords = [k.lower() for k in plan.must_have_keywords]
    if not keywords:
        return True
    hits = sum(1 for k in keywords if k in text)
    return hits >= max(1, len(keywords) // 3)


def _arxiv_search(plan: ResearchPlan) -> list[dict]:
    """Use the arxiv skill (httpx, no key) for a real arXiv search."""
    import httpx
    query = " ".join(plan.must_have_keywords[:5]) or plan.sub_questions[0]
    url = "https://export.arxiv.org/api/query"  # arXiv moved to HTTPS
    params = {
        "search_query": f"all:{query}",
        "start": 0, "max_results": 6,
        "sortBy": "relevance", "sortOrder": "descending",
    }
    with httpx.Client(timeout=15.0, follow_redirects=True) as c:
        r = c.get(url, params=params)
        r.raise_for_status()
    # Minimal Atom parse: extract <entry> blocks.
    entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.DOTALL)
    out: list[dict] = []
    for ent in entries:
        title_m = re.search(r"<title>(.*?)</title>", ent, re.DOTALL)
        link_m = re.search(r"<id>(.*?)</id>", ent)
        sum_m = re.search(r"<summary>(.*?)</summary>", ent, re.DOTALL)
        title = (title_m.group(1).strip() if title_m else "")
        link = (link_m.group(1).strip() if link_m else "")
        summary = (sum_m.group(1).strip()[:400] if sum_m else "")
        if title and link:
            out.append({
                "kind": "paper", "title": title, "url": link,
                "summary": summary, "source": "arxiv",
            })
    return out


# ---------------------------------------------------------------------------
# fetcher
# ---------------------------------------------------------------------------

def fetcher_node(state: ArgusState) -> dict:
    """Fetch each source URL into markdown via snatch/crawl/normalize."""
    fetched: list[dict] = []
    sources = state.get("sources") or []
    for src in sources:
        url = src["url"]
        kind = src.get("kind", "search_result")
        try:
            # Pick tool based on kind.
            if kind == "official_doc" or "github.com" in url or "arxiv.org" in url:
                # use normalize (article_convert) — works for both
                res = normalize_to_markdown(url)
                if res.ok and res.markdown_path:
                    fetched.append(FetchedItem(
                        url=url, title=res.title or src.get("title", ""),
                        markdown_path=res.markdown_path,
                        section=kind,
                        excerpt=(res.markdown_text or "")[:600],
                    ).model_dump())
                    continue
            if kind == "paper" and "arxiv.org/abs/" in url:
                # convert abs to pdf url, then snatch as paper
                pdf = url.replace("/abs/", "/pdf/") + ".pdf"
                res = snatch_url(pdf, kind="papers")
                if res.ok:
                    fetched.append(FetchedItem(
                        url=url, title=res.title or src.get("title", ""),
                        markdown_path=res.markdown_path,
                        section=kind,
                        excerpt="",
                    ).model_dump())
                    continue
            # default: try snatch first
            res = snatch_url(url, kind="auto")
            if res.ok and res.markdown_path:
                fetched.append(FetchedItem(
                    url=url, title=res.title or src.get("title", ""),
                    markdown_path=res.markdown_path,
                    section=kind,
                    excerpt=_read_excerpt(res.markdown_path),
                ).model_dump())
                continue
            # fallback: crawl
            res = crawl_url(url, deep=False, max_pages=4)
            if res.ok and res.markdown_path:
                fetched.append(FetchedItem(
                    url=url, title=src.get("title", ""),
                    markdown_path=res.markdown_path,
                    section=kind,
                    excerpt=_read_excerpt(res.markdown_path),
                ).model_dump())
        except Exception as e:
            logger.warning("fetch %s failed: %s", url, e)
    return {
        "fetched": fetched,
        "messages": [{"role": "assistant",
                      "content": f"📥 {len(fetched)} items fetched."}],
    }


def _read_excerpt(md_path: str | None, limit: int = 600) -> str:
    if not md_path:
        return ""
    try:
        text = Path(md_path).read_text(encoding="utf-8", errors="replace")
        return text[:limit]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# normalizer (alias for any not-yet-normalized items; minimal because the
# fetcher already normalizes via the underlying scripts).
# ---------------------------------------------------------------------------

def normalizer_node(state: ArgusState) -> dict:
    """Re-confirm every fetched item has a markdown_path on disk.

    Most work happens in fetcher; this node only patches missing ones.
    """
    patched = []
    for f in state.get("fetched") or []:
        item = FetchedItem.model_validate(f)
        if not item.markdown_path or not Path(item.markdown_path).exists():
            # try once more via normalize
            res = normalize_to_markdown(item.url)
            if res.ok and res.markdown_path:
                item.markdown_path = res.markdown_path
                item.excerpt = (res.markdown_text or "")[:600]
                item.title = item.title or res.title
        patched.append(item.model_dump())
    return {"fetched": patched}


# ---------------------------------------------------------------------------
# filter / rank
# ---------------------------------------------------------------------------

def filter_node(state: ArgusState) -> dict:
    """Drop low-signal items; keep provenance."""
    fetched = [FetchedItem.model_validate(f) for f in state.get("fetched") or []]
    plan = ResearchPlan.model_validate(state.get("plan") or {})
    keywords = [k.lower() for k in plan.must_have_keywords] or \
        [state["user_request"][:30].lower()]
    scored: list[FetchedItem] = []
    for f in fetched:
        haystack = (f.title + " " + f.excerpt).lower()
        score = sum(1.0 for k in keywords if k in haystack) / max(1, len(keywords))
        f.relevance_score = score
        if score >= 0.05 and f.markdown_path:
            scored.append(f)
    scored.sort(key=lambda x: x.relevance_score, reverse=True)
    return {"fetched": [f.model_dump() for f in scored[:14]]}


# ---------------------------------------------------------------------------
# synthesizer
# ---------------------------------------------------------------------------

SYNTH_SYSTEM = """You are Argus's senior research synthesizer.

You will be given a topic, the user's question, and a set of fetched
evidence items (each with a URL and a markdown excerpt). Produce a
grounded report:

1. Every factual claim MUST cite at least one evidence URL using
   the literal format [n] where n is the index in the evidence list.
2. Only use information that appears in the evidence. If the evidence
   is insufficient for a claim, omit it (do NOT fabricate).
3. Structure: ## TL;DR, ## Key Findings (each finding = claim + citations),
   ## Sources (numbered), ## Open Questions.
4. Keep prose tight; prefer numbered findings over paragraphs.

Return a JSON object:
{
  "findings": [
    {"claim": "...", "citation_urls": ["https://...", "..."],
     "confidence": "high|medium|low"},
    ...
  ],
  "draft_md": "# Title\\n\\n## TL;DR\\n...\\n## Sources\\n..."
}

Make sure every URL in citation_urls appears in the Sources section.
"""


def synthesizer_node(state: ArgusState) -> dict:
    chat = llm.chat_for_tier("strong", temperature=0.3, max_tokens=2500)
    fetched = [FetchedItem.model_validate(f) for f in state.get("fetched") or []]
    # Build evidence corpus (cap length).
    ev_lines: list[str] = []
    for i, f in enumerate(fetched, start=1):
        ev_lines.append(
            f"[{i}] {f.title or '(no title)'} — {f.url}\n"
            f"    excerpt: {(f.excerpt or '')[:400]}"
        )
    evidence = "\n".join(ev_lines) or "(no evidence fetched)"
    revision_notes = state.get("revision_notes") or []
    revision_block = ""
    if revision_notes:
        revision_block = (
            "\n\nPRIOR REVIEWER NOTES (must address):\n"
            + "\n".join(f"- {n}" for n in revision_notes)
        )

    user = (
        f"Topic: {state['user_request']}\n\n"
        f"Evidence ({len(fetched)} items):\n{evidence}\n"
        f"{revision_block}\n\n"
        "Write the JSON response now."
    )
    resp = chat.invoke([
        SystemMessage(content=SYNTH_SYSTEM),
        HumanMessage(content=user[:14000]),
    ])
    rec = llm.record_from_response("strong", llm.resolve_tier("strong"), resp)
    findings: list[Finding] = []
    draft_md = ""
    try:
        data = _parse_json_obj(resp.content)
        for fd in data.get("findings", []):
            try:
                findings.append(Finding.model_validate(fd))
            except Exception as e:
                logger.warning("bad finding: %s", e)
        draft_md = data.get("draft_md") or _draft_md_from_findings(
            state["user_request"], findings)
    except Exception as e:
        logger.warning("Synthesizer returned non-JSON: %s", e)
        draft_md = (
            f"# {state['user_request']}\n\n"
            "_Synthesis incomplete — model did not return structured JSON._\n\n"
            f"Raw model output:\n\n```\n{resp.content[:3000]}\n```\n"
        )
    return {
        "findings": [f.model_dump() for f in findings],
        "draft_md": draft_md,
        "model_calls": [rec.model_dump()],
        "messages": [{"role": "assistant",
                      "content": f"🧠 Synthesized {len(findings)} findings."}],
    }


def _draft_md_from_findings(topic: str, findings: list[Finding]) -> str:
    parts = [f"# {topic}", "", "## TL;DR", "",
             "Report generated from the synthesized findings below.", "",
             "## Key Findings", ""]
    for i, f in enumerate(findings, 1):
        cites = " ".join(f"[{_url_index(c, findings)}]" for c in f.citation_urls)
        parts.append(f"{i}. **{f.claim}** {cites} _(confidence: {f.confidence})_")
    parts += ["", "## Sources", ""]
    urls = []
    for f in findings:
        urls.extend(f.citation_urls)
    for i, u in enumerate(dict.fromkeys(urls), 1):
        parts.append(f"{i}. {u}")
    return "\n".join(parts)


def _url_index(url: str, findings: list[Finding]) -> int:
    """Return a stable 1-based index for a URL across the findings list."""
    urls = []
    for f in findings:
        for c in f.citation_urls:
            if c not in urls:
                urls.append(c)
    return urls.index(url) + 1 if url in urls else 0


# ---------------------------------------------------------------------------
# reviewer
# ---------------------------------------------------------------------------

REVIEW_SYSTEM = """You are Argus's adversarial reviewer.

You will see:
- The user's original question
- The list of findings (each with claim + citation URLs + confidence)
- The full draft markdown

Your job:
- Every factual claim in the draft MUST trace to a citation URL.
- Flag any claim that sounds fabricated, has no citation, or whose
  citation URL is not present in the fetched evidence list.
- Flag any sentence that introduces a number, date, or specific name
  not supported by the evidence.
- Verdict MUST be "pass" or "revise".
- If you say "revise", list specific actionable notes the synthesizer
  must follow.

You are from a DIFFERENT model family than the synthesizer. Do not
defer to the synthesizer's confidence labels — verify them yourself.

Return JSON only:
{
  "verdict": "pass" | "revise",
  "notes": ["...", "..."],
  "unsupported_claims": ["the exact claim text...", "..."],
  "fabrication_flags": ["the exact suspicious text...", "..."]
}
"""


def reviewer_node(state: ArgusState) -> dict:
    chat = llm.chat_for_tier("judge", temperature=0.1, max_tokens=1500)
    findings = state.get("findings") or []
    fetched = state.get("fetched") or []
    fetched_urls = {f["url"] for f in fetched}
    user = (
        f"Topic: {state['user_request']}\n\n"
        f"Findings JSON:\n{json.dumps(findings[:20], indent=2)[:6000]}\n\n"
        f"Draft markdown:\n```\n{(state.get('draft_md') or '')[:6000]}\n```\n\n"
        f"Allowed citation URLs ({len(fetched_urls)}):\n"
        + "\n".join(f"- {u}" for u in list(fetched_urls)[:30])
        + "\n\nReturn JSON verdict now."
    )
    resp = chat.invoke([
        SystemMessage(content=REVIEW_SYSTEM),
        HumanMessage(content=user[:14000]),
    ])
    rec = llm.record_from_response("judge", llm.resolve_tier("judge"), resp)
    try:
        data = _parse_json_obj(resp.content)
        verdict = ReviewVerdict.model_validate({
            "verdict": data.get("verdict", "revise"),
            "notes": data.get("notes", []),
            "unsupported_claims": data.get("unsupported_claims", []),
            "fabrication_flags": data.get("fabrication_flags", []),
        })
    except Exception as e:
        logger.warning("Reviewer non-JSON, defaulting to revise: %s", e)
        verdict = ReviewVerdict(
            verdict="revise",
            notes=["Reviewer did not return structured JSON; please tighten the draft."],
        )
    notes = list(state.get("revision_notes") or [])
    notes.extend(verdict.notes)
    notes.extend(f"Unsupported claim: {c}" for c in verdict.unsupported_claims)
    notes.extend(f"Fabrication flag: {f}" for f in verdict.fabrication_flags)
    rounds = int(state.get("revision_rounds") or 0) + (
        1 if verdict.verdict == "revise" else 0
    )
    return {
        "review_verdict": verdict.model_dump(),
        "revision_notes": notes,
        "revision_rounds": rounds,
        "model_calls": [rec.model_dump()],
        "messages": [{"role": "assistant",
                      "content": f"🔬 Reviewer verdict: {verdict.verdict}"}],
    }


def route_after_review(state: ArgusState) -> str:
    v = (state.get("review_verdict") or {}).get("verdict")
    rounds = int(state.get("revision_rounds") or 0)
    max_rounds = int(os.environ.get("ARGUS_MAX_REVISIONS", "3"))
    if v == "pass" or rounds >= max_rounds:
        return "report_builder"
    return "synthesizer"


# ---------------------------------------------------------------------------
# report_builder
# ---------------------------------------------------------------------------

def report_builder_node(state: ArgusState) -> dict:
    """Render markdown -> PDF; create per-run folder; trigger HITL preview."""
    from ..config import get_settings
    s = get_settings()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic = re.sub(r"[^A-Za-z0-9_-]+", "_", state["user_request"])[:50] or "report"
    folder = s.reports_root / f"{stamp}_{topic}"
    folder.mkdir(parents=True, exist_ok=True)
    md_path = folder / "report.md"
    pdf_path = folder / "report.pdf"
    md_text = state.get("draft_md") or "# (empty draft)"
    md_text += (
        "\n\n---\n\n_Generated by Argus • "
        f"{datetime.now().astimezone().isoformat()}_\n"
    )
    md_path.write_text(md_text, encoding="utf-8")
    pdf_ok = False
    try:
        # markdown_to_pdf uses playwright via the intel-stack helper. That
        # helper imports playwright at runtime, so if it isn't in the
        # argus venv we delegate to the intel-stack venv python.
        try:
            markdown_to_pdf(md_text, str(pdf_path), title=topic)
            pdf_ok = pdf_path.exists()
        except (ModuleNotFoundError, ImportError) as e:
            # Fallback: render via intel-stack's python + the _common helper.
            logger.info("argus venv missing playwright; delegating PDF render")
            import importlib.util
            import subprocess
            helper_path = (Path(os.environ.get(
                "INTEL_STACK_DIR",
                r"A:\Hermes\Agents\intel-stack\scripts")) / "_common.py")
            helper_path.parent.mkdir(parents=True, exist_ok=True)
            # We need a small wrapper: write the md to disk, call _common.
            tmp_py = folder / "_render_pdf.py"
            tmp_py.write_text(
                "import sys, os\n"
                f"sys.path.insert(0, r'{helper_path.parent}')\n"
                "os.environ.pop('PYTHONPATH', None)\n"
                f"from _common import markdown_to_pdf\n"
                f"markdown_to_pdf(open(r'{md_path}','r',encoding='utf-8').read(),\n"
                f"                  r'{pdf_path}', title=r'{topic}')\n",
                encoding="utf-8",
            )
            intel_py = os.environ.get(
                "INTEL_PYTHON",
                r"A:\Hermes\Agents\intel-stack\venv\Scripts\python.exe",
            )
            res = subprocess.run(
                [intel_py, str(tmp_py)],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "PYTHONPATH": ""},
            )
            if res.returncode != 0:
                logger.warning("PDF fallback failed: %s", res.stderr[-500:])
            pdf_ok = pdf_path.exists()
    except Exception as e:
        logger.warning("PDF render failed: %s", e)
    meta = {
        "topic": state["user_request"],
        "thread_id": state.get("thread_id"),
        "user_id": state.get("user_id"),
        "revision_rounds": state.get("revision_rounds", 0),
        "n_findings": len(state.get("findings") or []),
        "n_sources": len(state.get("fetched") or []),
        "model_calls": state.get("model_calls") or [],
    }
    (folder / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "report_paths": {
            "folder": str(folder),
            "md": str(md_path),
            "pdf": str(pdf_path) if pdf_ok else None,
        },
        "hitl": {"pending": True, "kind": "report_preview",
                 "ctx": {"paths": {
                     "folder": str(folder),
                     "md": str(md_path),
                     "pdf": str(pdf_path) if pdf_ok else None,
                 }}},
        "messages": [{"role": "assistant",
                      "content": "📝 Report ready. Awaiting your sign-off."}],
    }


# ---------------------------------------------------------------------------
# deliver
# ---------------------------------------------------------------------------

def deliver_node(state: ArgusState) -> dict:
    """Just records the paths; the Telegram bot layer does the actual send."""
    paths = state.get("report_paths") or {}
    return {
        "hitl": {"pending": False},
        "messages": [{"role": "assistant", "content": (
            f"✅ Delivered. Folder: {paths.get('folder')}")}],
    }


# ---------------------------------------------------------------------------
# quick_answer
# ---------------------------------------------------------------------------

def quick_answer_node(state: ArgusState) -> dict:
    """Cheap single-shot answer. Still routes through FreeLLMAPI, so it
    is grounded in the model's training (no fabrication guarantees)."""
    chat = llm.chat_for_tier("cheap", temperature=0.3, max_tokens=600)
    resp = chat.invoke([
        SystemMessage(content=(
            "You are Argus, a research assistant. Answer the user's question "
            "concisely and accurately. If you don't know, say so. No fake citations."
        )),
        HumanMessage(content=state["user_request"][:4000]),
    ])
    rec = llm.record_from_response("cheap", llm.resolve_tier("cheap"), resp)
    return {
        "quick_answer": resp.content,
        "model_calls": [rec.model_dump()],
        "messages": [{"role": "assistant", "content": resp.content}],
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_json_obj(text: str, *, min_keys: int = 1) -> dict:
    """Best-effort JSON parse; tolerates ```json fences and embedded JSON.

    Strategy:
      1. Strip outer fences and try direct parse.
      2. Balanced-brace match — find the outermost valid JSON object.
      3. Fall back to a ```json fenced block (the model may have
         embedded its real JSON inside a markdown code block).

    ``min_keys`` rejects trivially-empty objects like ``{}`` so callers
    can force the fallback path to fire when the LLM produced no useful
    structured output.
    """
    raw = text or ""

    def _accept(d: dict) -> dict | None:
        if not isinstance(d, dict):
            return None
        if len(d) < min_keys:
            return None
        return d

    # 1. Direct (after fence stripping).
    t = raw.strip()
    t = re.sub(r"^```(?:json)?", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        d = _accept(json.loads(t))
        if d is not None:
            return d
    except Exception:
        pass
    # 2. Balanced braces — find the first '{' whose matching '}' yields
    #    valid JSON.
    s = t
    for i, ch in enumerate(s):
        if ch == "{":
            depth = 0
            for j in range(i, len(s)):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = s[i:j + 1]
                        try:
                            d = _accept(json.loads(candidate))
                            if d is not None:
                                return d
                        except Exception:
                            break
    # 3. Try json-fenced block (the model sometimes dumps JSON inside
    #    a code fence that itself is in a string field).
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL |
                  re.IGNORECASE)
    if m:
        try:
            d = _accept(json.loads(m.group(1)))
            if d is not None:
                return d
        except Exception:
            pass
    raise ValueError(
        f"no useful JSON object found in: {raw[:200]!r} "
        f"(required >= {min_keys} keys)"
    )


def _ts() -> str:
    return datetime.now().astimezone().isoformat()