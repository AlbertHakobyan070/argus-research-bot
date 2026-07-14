"""Argus LangGraph state — TypedDict with append-only reducers for log fields.

The graph is checkpointed with SqliteSaver keyed on thread_id = Telegram
chat_id, so each user's in-flight research run survives a bot restart.
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


class PlannedSource(BaseModel):
    """One source the planner intends to consult."""
    kind: Literal["paper", "repo", "news", "blog", "official_doc",
                  "search_result"] = "search_result"
    query: str = ""
    target_url: str | None = None
    rationale: str = ""


class ResearchPlan(BaseModel):
    sub_questions: list[str] = Field(default_factory=list)
    planned_sources: list[PlannedSource] = Field(default_factory=list)
    must_have_keywords: list[str] = Field(default_factory=list)
    summary: str = ""


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

    ``id`` is a stable per-run identifier assigned by the synthesizer
    (e.g. ``"f3"``). It is the join key for post-synthesis passes
    (quarantine, fabrication detector, grounding verifier) — those
    passes operate on ``finding_id`` rather than substring-matching the
    prose, so a sentence that recurs in multiple sections is matched
    exactly once instead of partially. See
    ``src/argus/post_synthesis.py`` for the consumers.
    """
    id: str = ""  # set by synthesizer; empty only for legacy findings
    claim: str
    citation_urls: list[str]  # at least one FetchedItem URL
    confidence: Literal["high", "medium", "low"] = "medium"
    # Optional section anchor for finding-id routing. Set by the
    # synthesizer when the LLM places the claim under a specific
    # heading (e.g. "## Current state"). Used by report_builder to
    # group findings under sections.
    section: str = ""


class ReviewVerdict(BaseModel):
    verdict: Verdict
    notes: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    fabrication_flags: list[str] = Field(default_factory=list)


class ValidatedAssessment(BaseModel):
    """T7.6 — per-section self-assessment of the synthesized report.

    The synthesizer LLM produces one entry per major section it wrote,
    rating confidence and flagging open challenges. The reviewer verifier
    annotates unsupported claims. Surfaced to the reader in the document
    so they can calibrate trust per section rather than trust the whole
    report as a monolith.
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
    thread_id: str                  # = telegram chat id
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
    plan: dict | None               # ResearchPlan.model_dump()
    plan_approved: bool             # set by HITL resume
    plan_attempts: int              # T8: counter for planner_reflect_node re-plans

    # Research pipeline
    sources: list[dict]             # candidate sources from scout/research
    fetched: list[dict]             # FetchedItem.model_dump() list
    findings: list[dict]            # Finding.model_dump() list
    draft_md: str                   # current draft markdown
    review_verdict: dict | None     # ReviewVerdict-compatible dict
    revision_notes: list[str]       # accumulated reviewer notes per round
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
