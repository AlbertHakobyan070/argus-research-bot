"""Report archetypes (v4 spec §3 WP-I) — plain data, no machinery.

The brief classifies the request into one archetype; it shapes the report
WITHIN the length the user picked ("length wins"):

- ``sectioned``: (title, focus) pairs the outline offers as a hint, inside the
  length's section budget; a lecture keeps its Part I–IV names and applies
  these focuses inside the parts.
- ``flat``: ``body_emphasis`` for the flat (tldr/short) writer and
  ``analyst_emphasis`` for the short analysis.
- ``analyst_sections``: the analysis headings, in order; only forecast and
  decision add the scenarios section.
"""
from __future__ import annotations

TAKEAWAYS = "Key takeaways"
AGREEMENT = "Where sources agree and disagree"
SCENARIOS = "Scenarios and indicators"
IMPLICATIONS = "Implications"
OPEN_QUESTIONS = "Open questions and limits"

ANALYST_CORE = (TAKEAWAYS, AGREEMENT, IMPLICATIONS, OPEN_QUESTIONS)
ANALYST_WITH_SCENARIOS = (TAKEAWAYS, AGREEMENT, SCENARIOS, IMPLICATIONS, OPEN_QUESTIONS)

DEFAULT_ARCHETYPE = "explainer"

TEMPLATES: dict[str, dict] = {
    "explainer": {
        "sectioned": [
            ("Background & key concepts", "the terms, history and prerequisites a reader needs first"),
            ("How it works", "the mechanism step by step, with the evidence for each step"),
            ("Current state", "where things stand now: versions, adoption, recent results"),
            ("Trade-offs & limitations", "costs, failure modes and known weaknesses"),
            ("Practical guidance", "when and how to use it, as far as the evidence says"),
        ],
        "flat": {"body_emphasis": "explain what it is and how it works before the details",
                 "analyst_emphasis": "say what the evidence means for someone deciding whether to rely on it"},
        "analyst_sections": ANALYST_CORE,
    },
    "definition": {
        "sectioned": [
            ("Definition & origin", "the accepted definition or definitions and where the term comes from"),
            ("Key properties", "the defining characteristics and how they are recognised or measured"),
            ("Examples & non-examples", "concrete cases that fit the term and near-misses that do not"),
            ("Related concepts & common confusions", "neighbouring terms and frequent misunderstandings"),
        ],
        "flat": {"body_emphasis": "open with the definition in one sentence, then its key properties",
                 "analyst_emphasis": "note where sources define or use the term differently"},
        "analyst_sections": ANALYST_CORE,
    },
    "comparison": {
        "sectioned": [
            ("Context & what is compared", "the options, what each is for and the basis of comparison"),
            ("Comparison criteria", "a markdown table of the options against the key criteria, with citations"),
            ("Per-criterion analysis", "criterion by criterion, which option is stronger and why"),
            ("Decision guidance by scenario", "which option fits which situation"),
        ],
        "flat": {"body_emphasis": "lead with a compact markdown table comparing the options on the key criteria",
                 "analyst_emphasis": "say which option fits which situation and where the evidence is one-sided"},
        "analyst_sections": ANALYST_CORE,
    },
    "forecast": {
        "sectioned": [
            ("Current state & baseline", "where things stand today: levels, recent trend, consensus"),
            ("Drivers & mechanisms", "the forces that move the outcome and how they act"),
            ("Scenarios (base, upside, downside)", "the plausible paths, their triggers and their likelihood"),
            ("Leading indicators", "the signals that would show which scenario is unfolding"),
        ],
        "flat": {"body_emphasis": "state the current level and the main drivers before any outlook",
                 "analyst_emphasis": "give the base, upside and downside cases and the indicator that would "
                                     "confirm each"},
        "analyst_sections": ANALYST_WITH_SCENARIOS,
    },
    "sota_review": {
        "sectioned": [
            ("Taxonomy & landscape", "the families of approaches and how they relate"),
            ("Timeline & milestones", "the key results in order, with dates"),
            ("Leading approaches & results", "the current best methods and their reported numbers"),
            ("Open problems", "what remains unsolved and where results disagree"),
        ],
        "flat": {"body_emphasis": "name the leading approaches and their reported results first",
                 "analyst_emphasis": "say which results look robust and which rest on a single source"},
        "analyst_sections": ANALYST_CORE,
    },
    "howto": {
        "sectioned": [
            ("Prerequisites", "what is needed before starting: tools, versions, access"),
            ("Step-by-step", "the procedure in order, each step backed by the evidence"),
            ("Pitfalls & troubleshooting", "common failures and how to fix them"),
            ("Alternatives", "other ways to reach the same goal and when to prefer them"),
        ],
        "flat": {"body_emphasis": "give the steps in order before any background",
                 "analyst_emphasis": "flag the steps where sources disagree or the evidence is thin"},
        "analyst_sections": ANALYST_CORE,
    },
    "news_update": {
        "sectioned": [
            ("What happened", "the event itself: who, what, when, with dates"),
            ("Context", "the background that explains why it matters"),
            ("Reactions & analysis", "how the parties involved and observers responded"),
            ("What to watch", "the next decisions, dates and signals"),
        ],
        "flat": {"body_emphasis": "lead with what happened and when",
                 "analyst_emphasis": "say what is confirmed and what is still reported or disputed"},
        "analyst_sections": ANALYST_CORE,
    },
    "decision": {
        "sectioned": [
            ("Options", "the realistic choices and what each involves"),
            ("Criteria & evidence", "what the choice should rest on and what the evidence says per option"),
            ("Recommendation by scenario", "which option fits which situation and why"),
            ("Risks", "what could go wrong with each option"),
        ],
        "flat": {"body_emphasis": "lay out the options and the evidence for each before any recommendation",
                 "analyst_emphasis": "give the recommendation per scenario and what would change it"},
        "analyst_sections": ANALYST_WITH_SCENARIOS,
    },
}
