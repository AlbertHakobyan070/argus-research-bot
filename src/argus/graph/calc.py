"""Worked numbers — calculations the code does, from figures the sources state.

A question that asks for numbers ("with calculations", cost, revenue, break-even,
growth…) needs arithmetic the analyst may not do: it may introduce no number the
findings lack, and a writer's own arithmetic is unchecked. Here one
``compose.calc`` call (JSON; degrade → ``omit_worked_numbers``) PROPOSES a few
calculations; the code checks and computes them:

- every input is either a FIGURE from a finding — its value must appear in that
  finding's claim (scale words and percentages understood), and it carries the
  finding's source ids — or an ASSUMPTION, labelled as one with its reason;
- the formula is parsed with ``ast`` and evaluated over numbers, the inputs'
  names, ``+ - * / ** %`` and ``min max round abs`` only — never ``eval``;
- a calculation with an input that fails the check, an unknown name or a result
  that is not a finite number is dropped, with an ``errors`` line.

report_builder renders the kept ones after the cuts as the "## Worked numbers"
appendix (a calculation whose cited finding was cut goes with it); the panel
never judges it — it is computation, labelled as such. It runs only when the
request asks for numbers (``compose.calculations: auto``) and some finding states
one.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import operator
import re

from langchain_core.messages import HumanMessage, SystemMessage

from .. import llm
from ..engines.roles import DEGRADABLE_ROLES
from .jsonx import parse_json_obj

logger = logging.getLogger("argus.calc")

ROLE = "compose.calc"
MAX_CALCS = 5
_MAX_FINDINGS = 60
_MAX_POWER = 100                   # an exponent above this is refused (2 ** 10**9 would never finish)
REL_TOL = 0.005                    # a figure matches a stated number within 0.5%

_ASKS = re.compile(r"calculat|comput|estimat|\bcost|pric(?:e|ing)|revenue|profit|margin|\broi\b|break-?even"
                   r"|cagr|growth rate|forecast|projection|how (?:much|many)|budget|valuation|per (?:month|year|user)"
                   r"|monetiz|monetis|unit economics|payback|\btco\b|market size|\btam\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-+]?\d+(?:\.\d+)?")
_SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9,
           "trillion": 1e12, "tn": 1e12, "t": 1e12}
_SCALE_AFTER = re.compile(r"\s*(thousand|million|billion|trillion|bn|mn|tn|[kmbt])\b|\s*(%|percent)", re.IGNORECASE)

CALC_SYSTEM = f"""You plan the worked calculations of a research report. Code
computes them; you only write them down.

You get the user's question and the report's findings (id, claim, source
ids). Propose 1-{MAX_CALCS} calculations that answer the question's numeric asks
(costs, revenue, margins, growth, break-even, sizes, per-unit figures).

Each calculation:
- "label": what it computes, e.g. "Monthly LLM cost at 1,000 reports";
- "inputs": the named numbers it uses, each EITHER
  a figure from a finding: {{"name": "price", "value": 20, "unit": "USD/month",
  "finding": "f3"}} — value exactly as the finding states it, as a plain number
  (952 million → 952000000, 30.3% → 0.303);
  OR an assumption: {{"name": "users", "value": 1000, "unit": "users",
  "assumption": "a small first-year user base"}} — only when no finding has it;
- "formula": an expression over the input names and numbers with + - * / **
  ( ) and min, max, round, abs — e.g. "price * users * 12";
- "unit" of the result, and a one-sentence "note" on what the result shows.

Cite a finding only for a figure an outside source STATES as a fact (a
competitor's price, a market size, an API's fee). A figure a finding merely
PROPOSES or projects ("could price at $15/mo", "would generate $240k") is not
a fact: enter it as an assumption, saying it is a proposal. A project that has
no revenue yet has no stated price, user count or volume — those are
assumptions too.

Never compute a result yourself. Use only figures the findings state; mark
everything else as an assumption. No calculation is better than a made-up one.

Return ONLY JSON: {{"calculations": [{{"label": "...", "inputs": [...],
"formula": "...", "unit": "...", "note": "..."}}]}}
"""


def wants_numbers(request: str) -> bool:
    """The request asks for numbers (a cost, a forecast, "with calculations"…)."""
    return bool(_ASKS.search(request or ""))


def stated_numbers(text: str) -> list[float]:
    """Every number ``text`` states, with its scale word or percentage applied —
    and the bare number too ("30.3%" → 0.303 and 30.3; "952 million" → 9.52e8 and 952)."""
    out: list[float] = []
    for m in _NUMBER.finditer(text or ""):
        try:
            value = float(m.group(0).replace(",", ""))
        except ValueError:
            continue
        out.append(value)
        after = _SCALE_AFTER.match(text, m.end())
        if after and after.group(1):
            out.append(value * _SCALES[after.group(1).lower()])
        elif after and after.group(2):
            out.append(value / 100)
    return out


def _matches(value: float, stated: list[float]) -> bool:
    return any(math.isclose(value, s, rel_tol=REL_TOL, abs_tol=1e-12) for s in stated)


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"min": min, "max": max, "abs": abs,
          "round": lambda x, digits=0: round(x, int(digits))}     # every number is a float; round needs an int


def evaluate(formula: str, names: dict[str, float]) -> float:
    """``formula`` over ``names``: numbers, the names, ``+ - * / ** %``, unary
    ``+ -`` and ``min max round abs``; anything else raises ``ValueError``."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"unknown name {node.id!r}")
            return float(names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER:
                raise ValueError(f"exponent {right:g} is too large")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
                and not node.keywords):
            return float(_FUNCS[node.func.id](*(ev(a) for a in node.args)))
        raise ValueError(f"{type(node).__name__} is not allowed in a formula")
    try:
        result = ev(ast.parse(formula, mode="eval"))
    except (SyntaxError, ZeroDivisionError, OverflowError, TypeError) as e:
        raise ValueError(f"{type(e).__name__}: {e}") from e
    if not isinstance(result, float) or not math.isfinite(result):
        raise ValueError("the result is not a finite real number")
    return result


def check(raw: dict, findings: dict[str, dict]) -> tuple[dict | None, str]:
    """One proposed calculation → ``(calculation, "")`` or ``(None, why it was dropped)``."""
    label = " ".join(str(raw.get("label") or "").split())[:140]
    formula = str(raw.get("formula") or "").strip()[:300]
    if not label or not formula:
        return None, "no label or formula"
    inputs, names = [], {}
    for i in raw.get("inputs") or []:
        if not isinstance(i, dict):
            continue
        name = str(i.get("name") or "")
        if not re.fullmatch(r"[A-Za-z_]\w{0,40}", name) or name in _FUNCS or name in names:
            return None, f"input name {name!r} is not a usable identifier"
        try:
            value = float(i.get("value"))
        except (TypeError, ValueError):
            return None, f"input {name}: value {i.get('value')!r} is not a number"
        if not math.isfinite(value):
            return None, f"input {name}: value is not finite"
        unit = " ".join(str(i.get("unit") or "").split())[:40]
        fid = str(i.get("finding") or "").strip()
        assumption = " ".join(str(i.get("assumption") or "").split())[:200]
        if fid:
            f = findings.get(fid)
            if f is None:
                return None, f"input {name}: finding {fid} does not exist"
            if not _matches(value, stated_numbers(f.get("claim", ""))):
                return None, f"input {name}: {value:g} is not stated in finding {fid}"
            inputs.append({"name": name, "value": value, "unit": unit, "finding": fid,
                           "source_ids": list(f.get("source_ids") or [])})
        elif assumption:
            inputs.append({"name": name, "value": value, "unit": unit, "assumption": assumption})
        else:
            return None, f"input {name}: neither a finding nor an assumption"
        names[name] = value
    if not inputs:
        return None, "no inputs"
    try:
        result = evaluate(formula, names)
    except ValueError as e:
        return None, f"formula {formula!r}: {e}"
    return {"label": label, "formula": formula, "inputs": inputs, "result": result,
            "unit": " ".join(str(raw.get("unit") or "").split())[:40],
            "note": " ".join(str(raw.get("note") or "").split())[:300]}, ""


def _parse(text: str) -> list[dict]:
    raw = parse_json_obj(text, require_any=("calculations",)).get("calculations")
    if not isinstance(raw, list):
        raise ValueError(f"'calculations' is {type(raw).__name__}, not a list")
    return [c for c in raw if isinstance(c, dict)]


def worked_numbers(state, findings: list[dict], *, rc, rnd: int) -> dict:
    """The run's calculations → ``{"calculations": [...], "model_calls", "errors"}``
    (``{}`` when the request asks for no numbers, the knob is off or no finding states one)."""
    request = state.get("user_request") or ""
    if rc.compose.calculations == "off" or rc.length == "tldr" or not wants_numbers(request):
        return {}
    numeric = [f for f in findings if f.get("claim") and f.get("id") and stated_numbers(f["claim"])]
    if not numeric:
        return {}
    if ROLE not in rc.engines.roles:          # a run whose engines snapshot predates the role
        return {"errors": [f"[r{rnd}] calc: skipped — this run's engines snapshot has no {ROLE} binding"]}
    by_id = {f["id"]: f for f in numeric}
    listing = "\n".join(f"{f['id']}: {f['claim']} {json.dumps(f.get('source_ids') or [])}"
                        for f in numeric[:_MAX_FINDINGS])
    try:
        res = llm.complete(ROLE, [SystemMessage(content=CALC_SYSTEM), HumanMessage(content=(
            f"Question: {request}\n\nFindings (id: claim [source ids]):\n{listing}\n\n"
            "Return the calculations JSON."))], engines=rc.engines, temperature=0.1, max_tokens=3000,
            validate=_parse)
    except llm.EngineDegraded as e:
        logger.warning("%s degraded (%s); no worked numbers", ROLE, e)
        return {"model_calls": [a.model_dump() for a in e.attempts],
                "errors": [f"[r{rnd}] degrade: {ROLE} → {DEGRADABLE_ROLES[ROLE]} ({e})"]}
    calcs, errors = [], []
    for raw in res.parsed[:MAX_CALCS]:
        calc, why = check(raw, by_id)
        if calc is None:
            errors.append(f"[r{rnd}] calc: dropped {str(raw.get('label') or '(unlabelled)')[:80]!r} — {why}")
        else:
            calcs.append(calc)
    return {"calculations": calcs, "model_calls": [r.model_dump() for r in res.records], "errors": errors}


def _fmt(x: float) -> str:
    if x != 0 and (abs(x) >= 1e15 or abs(x) < 1e-4):
        return f"{x:.3g}"
    if abs(x) >= 100 or x == int(x):
        return f"{x:,.0f}"
    return f"{x:,.4g}" if abs(x) < 1 else f"{x:,.2f}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_appendix(calcs: list[dict], kept_finding_ids: set[str]) -> tuple[str, list[str]]:
    """The "## Worked numbers" appendix of the calculations whose cited findings
    were all kept → ``(markdown, dropped labels)``; ``""`` when none is left."""
    kept, dropped = [], []
    for c in calcs or []:
        if all(not i.get("finding") or i["finding"] in kept_finding_ids for i in c["inputs"]):
            kept.append(c)
        else:
            dropped.append(c["label"])
    if not kept:
        return "", dropped
    rows = ["| Calculation | Formula | Inputs | Result |", "|---|---|---|---:|"]
    for c in kept:
        ins = "; ".join(
            f"{i['name']} = {_fmt(i['value'])}{(' ' + i['unit']) if i['unit'] else ''} "
            + ("".join(f"[{s}]" for s in i["source_ids"]) if i.get("finding")
               else f"(assumption: {i['assumption']})")
            for i in c["inputs"])
        rows.append(f"| {_cell(c['label'])} | `{_cell(c['formula'])}` | {_cell(ins)} | "
                    f"**{_fmt(c['result'])}**{(' ' + _cell(c['unit'])) if c['unit'] else ''} |")
    notes = [f"- **{c['label']}** — {c['note']}" for c in kept if c["note"]]
    return "\n\n".join(x for x in [
        "## Worked numbers",
        "_Computed by Argus from the figures shown: each input cites the source that states it, or is marked as "
        "an assumption; the formulas are evaluated by code, not by the model._",
        "\n".join(rows), "\n".join(notes)] if x), dropped


__all__ = ["worked_numbers", "render_appendix", "evaluate", "check", "stated_numbers", "wants_numbers",
           "CALC_SYSTEM", "ROLE"]
