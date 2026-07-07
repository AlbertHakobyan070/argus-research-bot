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


class Finding(BaseModel):
    """One cited claim in the synthesized report."""
    claim: str
    citation_urls: list[str]  # at least one FetchedItem URL
    confidence: Literal["high", "medium", "low"] = "medium"


class ReviewVerdict(BaseModel):
    verdict: Verdict
    notes: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    fabrication_flags: list[str] = Field(default_factory=list)


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
    messages: Annotated[list[dict], "appended-only chat history"]  # type: ignore
    plan: dict | None               # ResearchPlan.model_dump()
    plan_approved: bool             # set by HITL resume

    # Research pipeline
    sources: list[dict]             # candidate sources from researcher
    fetched: list[dict]             # FetchedItem.model_dump() list
    findings: list[dict]            # Finding.model_dump() list
    draft_md: str                   # current draft markdown
    review_verdict: dict | None     # ReviewVerdict.model_dump()
    revision_notes: list[str]       # accumulated reviewer notes per round
    revision_rounds: int            # counter

    # Delivery
    report_paths: dict              # {"md": str, "pdf": str, "folder": str}
    quick_answer: str               # for /ask path

    # HITL control
    hitl: dict                      # {"pending": bool, "kind": str, "ctx": dict}

    # Telemetry — append-only so each node can return just the new record.
    model_calls: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]