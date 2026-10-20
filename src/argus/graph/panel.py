"""v4 review panel (spec §3 WP-C, §2.8 R1/R14): batched grounding, the
coverage judge, the precision scan and a deterministic merge that writes
``panel_verdict.next``.

- **grounding** (``panel.grounding``, ``skip_item``): batches of at most
  ``judge_batch_size`` findings grouped by section, each prompt holding
  exactly the batch's cited notes — at most 3 claims per note, chosen by
  word overlap with the batch, quotes up to 250 chars — within 24 000 chars
  (system + human); a batch over that is halved until it fits, and a single
  finding still over it is skipped. Batches run on a pool of 3. A batch that
  is skipped, fails or returns unparseable JSON — an id that, normalised
  (``[f3]``, ``F3``, ``3``, a ``finding_id`` key), names no finding of the
  batch included — leaves its findings **unjudged** (``judged < total``),
  never passed. When the judge served a model of a writer's family, the
  batches are rerun once on an engine of the binding from another family
  (``panel.diversity_rerun``); the rerun's verdict counts when it judged as
  much, else ``judge_not_diverse``.
- **coverage** (``panel.coverage``, ``skip_item``): weak sections and the
  sub-questions no section answers (``missing_sub_qs``).
- **precision** (``panel.precision_mode``): the deterministic
  ``precision.scan`` (``uncited_specifics``), no LLM.

``next`` is coverage first: an eligible gap (R14: a sub-question under
``coverage_min_strong`` or owned by no section, then ``missing_sub_qs``,
minus the sub-questions whose last gap wave added no strong note) goes back
to ``research`` while ``gap_waves < gap_waves_max``; else a wanted rewrite (a
flagged share ≥ ``rewrite_flag_ratio`` of the judged findings, or a weak
section) goes to ``compose`` while ``revision_rounds <
max_compose_revisions``; else ``report_builder``. The verdict is also written
as the v2 ``review_verdict``.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from .. import llm
from ..engines.registry import family
from . import precision, run_config
from .assemble import ANALYSIS_SECTION
from .jsonx import parse_json_obj
from .outline import _ints
from .research import _round
from .state import PanelVerdict, ResearchBrief

logger = logging.getLogger("argus.panel")

BATCH_BUDGET = 24_000             # chars of system + human per grounding batch (spec §2.8 R1)
_CLAIMS_PER_NOTE = 3
_QUOTE_CHARS = 250
_POOL = 3
_WRITER_ROLES = ("compose.section", "compose.flat", "compose.analyst")

GROUNDING_SYSTEM = """You are an adversarial grounding auditor for a
research report. You get a batch of the report's findings — each with its
id, its section, the source ids it cites and the sentence the report
states it in — and, for every cited source, its evidence claims with their
verbatim quotes.

Flag a finding when:
- its sentence is NOT supported by a claim/quote of a source it cites;
- it states a number, date or name absent from the cited evidence;
- a cited source contradicts it (unless the sentence presents the
  conflict).

Be strict but fair: condensed wording is fine, invented specifics are
not. Judge only the findings listed. Return ONLY JSON:
{"flagged": [{"id": "f2", "reason": "short reason"}, ...],
 "notes": ["actionable note for the writer", ...]}
"""

COVERAGE_SYSTEM = """You audit a research report for COVERAGE against its
brief. You get the brief (numbered sub-questions + success criteria) and
the report's section titles with short excerpts.

Judge: does the report substantively answer each sub-question and success
criterion? Name the sections that are thin or off-topic, and the numbers
of the sub-questions no section answers.

Return ONLY JSON:
{"verdict": "pass|revise",
 "weak_sections": ["exact section title", ...],
 "missing_sub_qs": [2, ...],
 "notes": ["actionable note", ...]}
Verdict is "revise" only for MATERIAL gaps (an unanswered sub-question),
not stylistic wishes.
"""

_CODE = re.compile(r"\b(429|5\d\d)\b")
_ERROR_TYPE = re.compile(r"\b([A-Z]\w*(?:Error|Exception|Timeout))\b")


def failure_reason(text: str) -> str:
    """The short reason of a skip (a facade message or a ``skip:`` line):
    "unparseable", else an HTTP 429/5xx code, else the error type, else the
    text itself."""
    if "(last: " in text:
        text = text.rsplit("(last: ", 1)[1]
    elif text.endswith(")") and "(" in text:
        text = text[text.index("(") + 1:]
    text = " ".join(text.rstrip(")").split())
    if text.startswith("unparseable"):
        return "unparseable"
    if m := _CODE.search(text):
        return m.group(1)
    if m := _ERROR_TYPE.search(text):
        return m.group(1)
    return text[:80]


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

def decide_next(*, gaps: list[int], wants_rewrite: bool, gap_waves: int, revision_rounds: int,
                rc: run_config.RunConfig) -> str:
    """Coverage first: research while the gap reserve lasts, then a rewrite
    while the revision budget lasts, else the report."""
    if gaps and rc.panel.coverage_to_research and gap_waves < rc.research.gap_waves_max:
        return "research"
    if wants_rewrite and revision_rounds < rc.panel.max_compose_revisions:
        return "compose"
    return "report_builder"


def gap_signal(state, rc: run_config.RunConfig, missing_sub_qs) -> list[int]:
    """R14: the sub-questions under ``coverage_min_strong`` or owned by no
    outline section (a sectioned report only), then the coverage judge's
    ``missing_sub_qs``; minus the zero-yield ones — a sub-question whose last
    gap wave (``gap_history``) added no strong note."""
    brief = ResearchBrief.model_validate(state.get("brief") or {})
    n = len(brief.sub_questions)
    coverage = state.get("coverage") or {}
    sections = (state.get("outline") or {}).get("sections") or []
    owned = {k for s in sections for k in s.get("sub_qs") or []}
    gaps = [k for k in range(n)
            if int(coverage.get(str(k)) or 0) < rc.research.coverage_min_strong or (sections and k not in owned)]
    gaps += [k for k in _ints(missing_sub_qs, lambda k: 0 <= k < n) if k not in gaps]
    last: dict[int, dict] = {}
    for entry in state.get("gap_history") or []:
        for k in entry.get("sub_qs") or []:
            last[int(k)] = entry
    dead = {k for k, e in last.items()
            if (e.get("new_strong_by_subq") or {}).get(str(k), e.get("new_strong")) == 0}
    return [k for k in gaps if k not in dead]


# ---------------------------------------------------------------------------
# Grounding batches
# ---------------------------------------------------------------------------

def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", (text or "").lower()) if len(w) >= 3}


def _batch_prompt(batch: list[dict], notes: dict[int, dict]) -> str:
    lines = ["FINDINGS:"]
    for f in batch:
        ids = "".join(f"[{i}]" for i in f.get("source_ids") or [])
        text = " ".join((f.get("sentence") or f.get("claim") or "").split())
        lines.append(f"[{f.get('id')}] (section: {f.get('section') or 'Report'} · cites {ids or 'nothing'}) {text}")
    words = set().union(*(_words(f.get("sentence") or f.get("claim") or "") for f in batch))
    lines += ["", "CITED SOURCES:"]
    for i in sorted({i for f in batch for i in f.get("source_ids") or []}):
        n = notes.get(i)
        if n is None:
            continue
        claims = [c for c in n.get("claims") or [] if c.get("quote_verified") is not False]
        best = sorted(range(len(claims)),
                      key=lambda k: -len(words & _words(f"{claims[k].get('text')} {claims[k].get('quote')}")))
        lines.append(f"[{i}] {n.get('title') or '(untitled)'} — {n.get('source_url')}")
        for k in sorted(best[:_CLAIMS_PER_NOTE]):
            c = claims[k]
            lines.append(f"   • {c.get('text')}")
            if c.get("quote"):
                lines.append(f'     quote: "{str(c["quote"])[:_QUOTE_CHARS]}"')
    return "\n".join([*lines, "", "Return the audit JSON."])


def _fit(batch: list[dict], notes: dict[int, dict]) -> list[tuple[list[dict], str | None]]:
    """``batch`` as ``(findings, prompt)`` pieces within the budget, halving
    an oversized one; a single finding still over it gets prompt None."""
    prompt = _batch_prompt(batch, notes)
    if len(GROUNDING_SYSTEM) + len(prompt) <= BATCH_BUDGET:
        return [(batch, prompt)]
    if len(batch) == 1:
        return [(batch, None)]
    mid = len(batch) // 2
    return _fit(batch[:mid], notes) + _fit(batch[mid:], notes)


def _batches(findings: list[dict], size: int, notes: dict[int, dict]) -> list[tuple[list[dict], str | None]]:
    """Section groups split into chunks of ≤ ``size``, packed in order, then fitted."""
    groups: dict[str, list[dict]] = {}
    for f in findings:
        groups.setdefault(f.get("section") or "", []).append(f)
    packed: list[list[dict]] = []
    for group in groups.values():
        for k in range(0, len(group), size):
            chunk = group[k:k + size]
            if packed and len(packed[-1]) + len(chunk) <= size:
                packed[-1] = packed[-1] + chunk
            else:
                packed.append(chunk)
    return [piece for b in packed for piece in _fit(b, notes)]


_FID = re.compile(r"\[?\s*f?\s*(\d+)\s*\]?", re.I)          # [f3]  f3  F3  3  → f3


def _parse_grounding(text: str, ids: set[str]) -> tuple[dict[str, str], list[str]]:
    """``(flagged id → reason, notes)``. Each entry's id (or ``finding_id``) is
    normalised; one that names no finding of the batch makes the reply
    unparseable — retried, then the batch is unjudged (spec R1), never passed."""
    data = parse_json_obj(text, require_any=("flagged",))
    raw = data.get("flagged")
    if not isinstance(raw, list):
        raise ValueError(f"'flagged' is {type(raw).__name__}, not a list")
    flagged: dict[str, str] = {}
    for item in raw:
        fid, reason = ((item.get("id", item.get("finding_id")), item.get("reason")) if isinstance(item, dict)
                       else (item, ""))
        m = _FID.fullmatch(str(fid).strip())
        key = f"f{int(m.group(1))}" if m else None
        if key not in ids:
            raise ValueError(f"flagged id {fid!r} names no finding of this batch")
        flagged.setdefault(key, " ".join(str(reason or "").split()) or "not supported by its sources")
    notes = data.get("notes")
    return flagged, [str(n) for n in notes if str(n).strip()] if isinstance(notes, list) else []


class _Judged:
    """One pass over every batch."""

    def __init__(self):
        self.flagged: dict[str, str] = {}
        self.notes: list[str] = []
        self.judged = 0
        self.batches = 0                      # this pass's batches
        self.skipped: list[str] = []          # skipped_judges entries
        self.errors: list[str] = []
        self.calls: list[dict] = []
        self.events: list[dict] = []
        self.families: set[str] = set()


def _judge(pieces: list[tuple[list[dict], str | None]], engines, rnd: int) -> _Judged:
    out = _Judged()
    out.batches = len(pieces)

    def one(i: int, piece):
        batch, prompt = piece
        if prompt is None:
            return None
        ids = {str(f.get("id")) for f in batch}
        try:
            return llm.complete("panel.grounding", [SystemMessage(content=GROUNDING_SYSTEM),
                                                    HumanMessage(content=prompt)],
                                engines=engines, temperature=0.1, max_tokens=1200,
                                validate=lambda text: _parse_grounding(text, ids), item=f"batch {i}")
        except llm.EngineSkipped as e:
            return e

    with ThreadPoolExecutor(max_workers=_POOL, thread_name_prefix="argus-panel") as ex:
        results = list(ex.map(one, range(1, len(pieces) + 1), pieces))
    for i, ((batch, _), res) in enumerate(zip(pieces, results), start=1):
        if res is None or isinstance(res, llm.EngineSkipped):
            reason = f"input over the {BATCH_BUDGET}-char budget" if res is None else failure_reason(str(res))
            out.skipped.append(f"grounding batch {i}: {reason}")
            out.errors.append(f"[r{rnd}] skip: panel.grounding batch {i} "
                              f"({res if res is not None else reason})")
            if res is not None:
                out.calls += [a.model_dump() for a in res.attempts]
            out.events.append({"name": "grounding", "batch": i, "verdict": "skipped", "skipped": True})
            continue
        flagged, notes = res.parsed
        out.calls += [r.model_dump() for r in res.records]
        out.families.add(family(res.record.served_model or res.record.requested_model))
        out.judged += len(batch)
        out.flagged.update(flagged)
        out.notes += notes
        out.events.append({"name": "grounding", "batch": i, "skipped": False,
                           "verdict": f"{len(flagged)} flagged" if flagged else "pass"})
    return out


def _diverse_engines(rc: run_config.RunConfig, writers: set[str]):
    """A copy of the snapshot whose ``panel.grounding`` binding is one engine
    of another family than the writers', or None."""
    binding = rc.engines.roles["panel.grounding"]
    other = next((e for e in [binding.primary, *binding.fallbacks] if e.model and family(e.model) not in writers),
                 None)
    if other is None:
        return None
    roles = {**rc.engines.roles, "panel.grounding": binding.model_copy(update={"primary": other, "fallbacks": []})}
    return rc.engines.model_copy(update={"roles": roles})


def _ground(state, rc: run_config.RunConfig, findings: list[dict], rnd: int) -> tuple[_Judged, bool]:
    """The grounding pass, with the diversity rerun: (result, judge_not_diverse)."""
    notes = {int(n["source_id"]): n for n in state.get("evidence") or []}
    pieces = _batches(findings, rc.panel.judge_batch_size, notes)
    first = _judge(pieces, rc.engines, rnd)
    writers = {family(c["served_model"]) for c in state.get("model_calls") or []
               if c.get("role") in _WRITER_ROLES and c.get("ok", True) and c.get("served_model")}
    if not first.families & writers:
        return first, False
    engines = _diverse_engines(rc, writers) if rc.panel.diversity_rerun else None
    if engines is None:
        return first, True
    rerun = _judge(pieces, engines, rnd)
    for c in rerun.calls:
        c["retry_reason"] = "diversity_rerun"
    if rerun.judged < first.judged:                        # a failed rerun: the first verdict stands
        first.calls += rerun.calls
        return first, True
    rerun.calls = first.calls + rerun.calls
    rerun.events = first.events
    return rerun, bool(rerun.families & writers)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def _parse_coverage(text: str) -> dict:
    data = parse_json_obj(text, require_any=("verdict", "weak_sections", "missing_sub_qs"))
    weak = data.get("weak_sections") or []
    return {"verdict": "revise" if data.get("verdict") == "revise" else "pass",
            "weak_sections": [str(t) for t in weak] if isinstance(weak, list) else [],
            "missing_sub_qs": _ints(data.get("missing_sub_qs")),
            "notes": [str(n) for n in data.get("notes") or [] if str(n).strip()]
            if isinstance(data.get("notes"), list) else []}


def _coverage(state, rc: run_config.RunConfig, rnd: int) -> tuple[dict | None, list[dict], list[str]]:
    brief = ResearchBrief.model_validate(state.get("brief") or {})
    subs = "\n".join(f"({k}) {sq.q}" for k, sq in enumerate(brief.sub_questions)) or "(none)"
    criteria = "\n".join(f"- {c}" for c in brief.success_criteria) or "- (none)"
    excerpts = "\n\n".join(f"### {s.get('title')}\n{(s.get('md') or '')[:400]}" for s in state.get("sections") or [])
    human = (f"SUB-QUESTIONS:\n{subs}\n\nSUCCESS CRITERIA:\n{criteria}\n\n"
             f"REPORT SECTIONS:\n{excerpts[:6000] or '(none)'}\n\nReturn the coverage JSON.")
    try:
        res = llm.complete("panel.coverage", [SystemMessage(content=COVERAGE_SYSTEM), HumanMessage(content=human)],
                           engines=rc.engines, temperature=0.1, max_tokens=900, validate=_parse_coverage,
                           item="report")
    except llm.EngineSkipped as e:
        return None, [a.model_dump() for a in e.attempts], [f"[r{rnd}] skip: panel.coverage report ({e})"]
    return res.parsed, [r.model_dump() for r in res.records], []


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

def _emit(runtime: Runtime | None, kind: str, data: dict) -> None:
    if runtime is not None:
        runtime.stream_writer({"kind": kind, "node": "panel", "data": data})


def panel_node(state, runtime: Runtime | None = None) -> dict:
    """Judge the draft and decide where the run goes next."""
    rc = run_config.require(state)
    findings = state.get("findings") or []
    rnd = _round(state)
    if not findings or not rc.panel.enabled:
        pv = PanelVerdict(judge_verdicts={"grounding": "off", "coverage": "off"} if findings else {})
        return {"panel_verdict": pv.model_dump(), "review_verdict": _to_review_verdict(pv), "coverage_gaps": [],
                "messages": [{"role": "assistant", "content": (
                    "🔬 Panel off." if findings else "🔬 Panel skipped (no findings to audit).")}]}

    calls: list[dict] = []
    errors: list[str] = []
    skipped: list[str] = []
    judge_verdicts: dict = {"grounding": "off", "coverage": "off"}
    flagged: dict[str, str] = {}
    judge_notes: list[str] = []
    judged = total = 0
    not_diverse = False
    if rc.panel.grounding:
        g, not_diverse = _ground(state, rc, findings, rnd)
        for event in g.events:
            _emit(runtime, "judge", event)
        calls += g.calls
        errors += g.errors
        skipped += g.skipped
        judged, total = g.judged, len(findings)
        flagged = {f["id"]: g.flagged[f["id"]] for f in findings if f.get("id") in g.flagged}
        judge_notes = g.notes
        judge_verdicts["grounding"] = "unjudged" if not judged else "revise" if flagged else "pass"
        judge_verdicts["flag_reasons"] = flagged
        judge_verdicts["grounding_batches"] = g.batches   # the degraded-run count is over this pass (§2.3)
    coverage = None
    if rc.panel.coverage:
        coverage, ccalls, cerrors = _coverage(state, rc, rnd)
        calls += ccalls
        errors += cerrors
        if coverage is None:
            skipped.append(f"coverage: {failure_reason(cerrors[0])}")
        judge_verdicts["coverage"] = coverage["verdict"] if coverage else "unjudged"
        _emit(runtime, "judge", {"name": "coverage", "verdict": judge_verdicts["coverage"],
                                 "skipped": coverage is None})
    uncited: list[str] = []
    if rc.panel.precision_mode == "deterministic":
        analysis = (state.get("analysis") or {}).get("md") or ""
        uncited = precision.scan("\n\n".join([*(s.get("md") or "" for s in state.get("sections") or []), analysis]))

    titles = [s.get("title") for s in state.get("sections") or []]
    weak = [t for t in (coverage or {}).get("weak_sections") or [] if t in titles]
    flagged_sections = {f.get("section") for f in findings if f.get("id") in flagged}
    revise_sections = [t for t in titles if t != ANALYSIS_SECTION and (t in flagged_sections or t in weak)]
    wants_rewrite = bool(flagged and judged and len(flagged) / judged >= rc.panel.rewrite_flag_ratio) or bool(weak)
    gaps = gap_signal(state, rc, (coverage or {}).get("missing_sub_qs") or [])
    rounds = int(state.get("revision_rounds") or 0)
    nxt = decide_next(gaps=gaps, wants_rewrite=wants_rewrite, gap_waves=int(state.get("gap_waves") or 0),
                      revision_rounds=rounds, rc=rc)
    by_id = {f.get("id"): f for f in findings}
    notes = [f'{by_id[i].get("section") or "Report"}: "{by_id[i].get("claim")}" is not supported by its sources '
             f"— {reason}" for i, reason in flagged.items()]
    notes += judge_notes + list((coverage or {}).get("notes") or [])
    pv = PanelVerdict(
        verdict="revise" if wants_rewrite else "pass", judge_verdicts=judge_verdicts, notes=notes,
        flagged_finding_ids=list(flagged), unsupported_claims=[by_id[i].get("claim") or "" for i in flagged],
        revise_sections=revise_sections, next=nxt, coverage_gaps=gaps, wants_rewrite=wants_rewrite,
        judge_not_diverse=not_diverse, uncited_specifics=uncited, skipped_judges=skipped, judged=judged,
        total=total)
    out = {
        "panel_verdict": pv.model_dump(),
        "review_verdict": _to_review_verdict(pv),
        "coverage_gaps": gaps if nxt == "research" else [],
        "model_calls": calls,
        "errors": errors,
        "messages": [{"role": "assistant", "content": (
            f"🔬 Panel: {pv.verdict} — judged {judged}/{total}, {len(flagged)} flagged → {nxt}.")}],
    }
    if nxt == "compose":
        out["revision_rounds"] = rounds + 1
    return out


def _to_review_verdict(pv: PanelVerdict) -> dict:
    """v2 ReviewVerdict-compatible dict (the bot reads it)."""
    return {
        "verdict": pv.verdict,
        "notes": list(pv.notes),
        "unsupported_claims": list(pv.unsupported_claims),
        "fabrication_flags": list(pv.fabrication_flags),
        "flagged_finding_ids": list(pv.flagged_finding_ids),
    }


def route_after_panel(state) -> str:
    """The panel's own decision (``panel_verdict.next``)."""
    return (state.get("panel_verdict") or {}).get("next") or "report_builder"


__all__ = ["panel_node", "route_after_panel", "decide_next", "gap_signal", "failure_reason",
           "GROUNDING_SYSTEM", "COVERAGE_SYSTEM", "BATCH_BUDGET"]
