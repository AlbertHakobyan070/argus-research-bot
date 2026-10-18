"""Argus LangGraph state — TypedDict with append-only reducers for log fields.

The graph is checkpointed by ONE shared AsyncSqliteSaver (opened in the
bot's ``post_init``); every run gets its own thread
``tg:<chat_id>:<run8>``, so each run — not just one per chat — survives
a bot restart and can be re-attached with /continue.

``ArgusState`` is also the update contract: LangGraph silently DROPS any
key a node returns that is not declared here (verified on langgraph
1.2.8). The ``messages`` note below is one casualty of that rule;
compose's ``synthesis_outcome`` is another.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field
from typing_extensions import NotRequired


Mode = Literal["quick", "deep"]
Verdict = Literal["pass", "revise"]

# T7 length selector. Chosen at plan-approval time (HITL). Drives
# synthesizer prompt/template depth + report_builder decisions.
Length = Literal["tldr", "short", "medium", "long", "lecture"]
DEFAULT_LENGTH: Length = "short"
VALID_LENGTHS: tuple[Length, ...] = (
    "tldr", "short", "medium", "long", "lecture")


class FetchedItem(BaseModel):
    """A piece of evidence already fetched + normalized."""
    url: str
    title: str = ""
    markdown_path: str | None = None
    section: str = ""  # which planned-source bucket it came from
    excerpt: str = ""   # first 600 chars of the markdown
    relevance_score: float = 0.0
    credibility_score: float | None = None  # set by credibility scoring
    credibility_flag: str | None = None    # e.g. "low_credibility"
    # v3 — which brief sub-questions (0-based indices) this source was
    # gathered for, and the provider that surfaced it.
    sub_qs: list[int] = Field(default_factory=list)
    provider: str = ""


# ---------------------------------------------------------------------------
# v3 research-engine models (brief / evidence / outline / panel)
# ---------------------------------------------------------------------------

SubQuestionKind = Literal["paper", "repo", "web", "mixed"]


class SubQuestion(BaseModel):
    """One brief sub-question with a source-kind hint for provider routing."""
    q: str
    kind: SubQuestionKind = "mixed"


class ResearchBrief(BaseModel):
    """v3 scoping artifact (replaces the v2 ResearchPlan as the driver;
    a v2-compatible ``plan`` dict is still emitted for the Telegram
    plan-gate renderer)."""
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    must_have_keywords: list[str] = Field(default_factory=list)
    summary: str = ""
    success_criteria: list[str] = Field(default_factory=list)


class EvidenceClaim(BaseModel):
    """One atomic claim extracted from a source by the digest pass."""
    text: str
    quote: str = ""          # short supporting quote from the document
    confidence: Literal["high", "medium", "low"] = "medium"


class EvidenceNote(BaseModel):
    """Digest of ONE fetched source, keyed to the sub-questions it was
    gathered for. This is what the writers read — never raw excerpts.

    ``source_id`` is the 1-based index of the source in ``state["fetched"]``
    order; it is the [n] citation id used in the report body and the
    ## Sources block.
    """
    source_id: int
    source_url: str
    title: str = ""
    sub_qs: list[int] = Field(default_factory=list)
    relevance: int = 0       # 0-5, judged by the digest LLM from full text
    stance: Literal["supports", "mixed", "contradicts", "background"] = "background"
    claims: list[EvidenceClaim] = Field(default_factory=list)


class OutlineSection(BaseModel):
    """One planned report section."""
    title: str
    focus: str = ""
    sub_qs: list[int] = Field(default_factory=list)


class PanelVerdict(BaseModel):
    """Merged output of the 3-judge review panel."""
    verdict: Verdict = "pass"
    judge_verdicts: dict = Field(default_factory=dict)  # {judge: pass|revise}
    notes: list[str] = Field(default_factory=list)
    flagged_finding_ids: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    fabrication_flags: list[str] = Field(default_factory=list)
    revise_sections: list[str] = Field(default_factory=list)  # section titles


class Finding(BaseModel):
    """One cited claim in the synthesized report.

    ``id`` is assigned by compose (``"f0"``, ``"f1"``… in extraction
    order) and is re-assigned on every compose pass, so it is stable
    only within one draft. It is the join key between the panel's
    grounding judge (which flags findings by id) and report_builder's
    ``post_synthesis.quarantine_by_id`` — matching by id instead of
    substring-matching the prose means a sentence that recurs in
    several sections is routed exactly once.
    """
    id: str = ""  # set by compose; empty only for legacy findings
    claim: str
    citation_urls: list[str]  # at least one FetchedItem URL
    confidence: Literal["high", "medium", "low"] = "medium"
    # Section anchor: compose sets it to the title of the section the
    # claim was extracted from ("Report" in flat modes). The panel uses
    # it to turn flagged finding ids into the list of sections to
    # rewrite; build_assessment counts claims per section with it.
    section: str = ""


class ReviewVerdict(BaseModel):
    verdict: Verdict
    notes: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    fabrication_flags: list[str] = Field(default_factory=list)


class ValidatedAssessment(BaseModel):
    """T7.6 — per-section self-assessment of the synthesized report.

    v3: computed deterministically by ``compose.build_assessment`` from
    the extracted findings and per-sub-question coverage — no LLM
    self-grading. report_builder then merges the panel's unsupported /
    fabrication flags in (``merge_reviewer_into_assessment``). Surfaced
    on the title page so the reader can calibrate trust per section
    rather than trust the whole report as a monolith.
    """

    sections: list[dict] = Field(default_factory=list)  # each: {name, confidence, open_challenges, claim_count}
    overall_relevancy: Literal["high", "medium", "low"] = "medium"
    knowledge_density: dict = Field(default_factory=dict)  # {claims_per_section, citations_per_claim, conflict_markers}
    reviewer_unsupported: list[str] = Field(default_factory=list)
    reviewer_fabrication_flags: list[str] = Field(default_factory=list)


# Sentinel prefix the report_builder writes to mark the end of the
# deliverable block.
REPORT_MARKER = "<!-- ARGUS_REPORT_END -->"


class ArgusState(TypedDict, total=False):
    # Identity
    thread_id: str                  # "tg:<chat_id>:<run8>" — one thread per run
    user_id: int
    user_request: str               # raw text after the command
    mode: Mode                      # "quick" or "deep"

    # Working memory
    # NB: this MUST be a real reducer. The previous annotation was the
    # STRING "appended-only chat history", which LangGraph ignores —
    # the channel silently became last-write-wins and every node's
    # message overwrote the history it claimed to append to.
    messages: Annotated[list[dict], operator.add]
    length: Length                  # T7: chosen at plan-approval HITL
    plan: dict | None               # v2-shaped dict from brief.brief_to_plan (plan-gate renderer reads it)
    plan_approved: bool             # legacy: no node reads it (bot tracks approval in _inflight)
    plan_attempts: int              # legacy v2 (planner_reflect_node is gone; brief re-drafts inline)

    # Research pipeline
    sources: list[dict]             # candidate sources from scout/research
    fetched: list[dict]             # FetchedItem.model_dump() list
    findings: list[dict]            # Finding.model_dump() list
    draft_md: str                   # current draft markdown
    review_verdict: dict | None     # ReviewVerdict-compatible dict
    revision_notes: list[str]       # panel notes (grow each revise round) + user revise
                                    # requests, which bot.py appends LAST
    revision_rounds: int            # counter

    # v3 research engine
    brief: dict | None              # ResearchBrief.model_dump()
    queries: list[dict]             # planned search queries (scout + waves)
    evidence: list[dict]            # EvidenceNote.model_dump() list
    coverage: dict                  # {sub_q_idx(str): strength int}
    outline: dict | None            # {"sections": [OutlineSection...]}
    sections: list[dict]            # [{"title", "md"}] composed sections
    panel_verdict: dict | None      # PanelVerdict.model_dump()
    research_rounds: int            # waves executed inside research_node

    # Delivery
    report_paths: dict              # {"md": str, "pdf": str, "folder": str}
    quick_answer: str               # for /ask path
    validated_assessment: dict      # T7: ValidatedAssessment.model_dump()
    lecture_appendix: dict          # T7: methodology + tool calls log + density metrics

    # HITL control
    hitl: dict                      # {"pending": bool, "kind": str, "ctx": dict}
    extend_requested: bool          # Phase 2: user asked to deepen research at preview
    extend_rounds: int              # Phase 2: how many extend loops have run (capped)
    plan_feedback: str              # v2: user's plan-edit reply, consumed by planner
    revision_requested: bool        # v2: user asked to revise at report preview
    revise_rounds: int              # v2: user-driven revise loops (capped)
    append_only: bool               # v2: /continue ingests only appended sources
                                    #     (extend_prep skips the researcher subgraph)

    # Telemetry — append-only so each node can return just the new record.
    model_calls: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]
