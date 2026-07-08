# Argus — Handoff 2026-07-09: research-QUALITY defects

**From:** coding-app (Opus 4.8), after Phase 1 (pipeline fixes) + Phase 2
(YouTube/scrapling/HITL) + the live Telegram markdown-crash fix.
**For:** next Hermes agent (kanban worker / fresh session).
**Branch:** `feat/argus-fixes` == `main` @ `55bcff9` (local only, no remote).
**Tests:** 193 passing (`export PYTHONPATH="" && ./venv/Scripts/python.exe -m pytest tests/ --ignore=tests/manual_e2e.py --ignore=tests/test_e2e_research.py -q`).

---

## 0. TL;DR

The bot now works **end to end** on Telegram: plan renders, progress streams,
report delivers (md+pdf), Extend/Send/Revise/Cancel buttons work, no more
"resume failed" crashes. **What remains are RESEARCH-QUALITY defects, not
plumbing.** Live evidence: a `/research GLM 5.2` run (report attached below as
§7) and a `/research metacognitive reinforcement learning transformers` run.

Six defects, all reproducible. Fix in the ROI order in §6.

---

## 1. P1 — Plan preview shows HALLUCINATED planner URLs (misleading)

**Symptom (metacognitive-RL run):** the plan proposal listed
`repo — https://github.com/DeepMind/metacognitive-rl-transformers`,
`official_doc — https://www.deepmind.com/research/.../metacognitive-reinforcement-learning`,
`blog — https://blog.ought.com/...`, `paper — https://arxiv.org/abs/2309.12345`
(a placeholder ID), etc. — all fabricated. The user replied "The sources are
wrong / Redo".

**Root cause:** `_format_plan` at `src/argus/bot.py:258` renders
`s.get("target_url")` first. But since the Phase-1 fix, `researcher_node`
delegates to the subgraph, which **ignores planner `target_url`s entirely** and
searches fresh (arxiv/github/ddgs). So the plan preview advertises URLs the bot
will never use — pure noise that makes the user distrust the run before it even
starts.

**Fix:** in `_format_plan`, stop showing raw `target_url`. Show the search
*intent* instead — e.g. `` `kind` — <query or sub-question>`` — and, if you keep
target_url at all, label it "candidate (will be verified)". Simplest correct
change: prefer `query`, drop `target_url` from the display. ~15 min, bot.py only.
Add/adjust a `tests/test_bot.py::test_format_plan_*` assertion.

---

## 2. P2 — Content-farm / low-quality sources reach the final report

**Symptom (GLM 5.2 report §7):** sources include
`[1] thetechbriefs.com/thudm-releases-glm-4-...` (SEO content farm) and
`[3] https://glm45.org/?` (parked/low-quality). These were cited as evidence.

**Root cause (two bugs):**
1. `src/argus/graph/credibility.py` `_DOMAIN_TABLE` (lines 41-82) has **only
   `primary` and `trusted` tiers — ZERO `low` entries**. So content farms match
   nothing, fall through to the `"neutral"` default (0.25, `tier_for` line 110),
   and the documented "content-farm penalty" never fires. The heuristic
   `_CONTENT_FARM_URL_HINTS` (line 147) only catches obvious spam slugs
   (`best-`, `top-10-`); `thetechbriefs.com` matches none.
2. `filter_node` (`src/argus/graph/nodes.py`, ~line 746) **ranks** by
   `(relevance, credibility)` and keeps top-14 but **never drops below
   `CREDIBILITY_FLOOR` (0.4)**. So a neutral-scored farm with high keyword
   overlap survives and gets cited.

**Fix:** (a) enforce the floor in `filter_node` — drop items with
`credibility_score < CREDIBILITY_FLOOR`, but keep a safety net (if that leaves
< N items, keep the best N so we never empty the report — reuse the Phase-1
"don't empty the report" principle). (b) Strengthen `credibility.py`: add a
curated `low` blocklist (content farms) AND a structural heuristic (domains with
3+ hyphens, `*briefs*`, `*-news.com`, single-post SEO domains). ~1-2h.
Tests in `tests/test_credibility.py`.

---

## 3. P3 — Empty dangling source slots (`[2]`, `[7]` with no URL)

**Symptom (GLM 5.2 report §7, Sources):** `[2] ` and `[7] ` are present but
blank. Findings that cite `[2]`/`[7]` now point at nothing.

**Root cause:** the synthesizer wrote a Sources section listing `[1]..[10]`, but
some of those URLs were **hallucinated by the synthesizer** (not in the fetched
evidence). `verify_citations` (`src/argus/citations.py`) correctly strips the
unregistered URL — but only the URL, leaving the `[N] ` label as residue. The
finding→source index mapping (`_url_index` in nodes.py) also breaks when a
source is stripped. This is the citation net *working* (fake URLs removed) but
leaving broken presentation + dangling `[n]` refs.

**Fix:** two options — (a) after `verify_citations`/`sanitize_report`,
post-process the Sources block: drop empty `[n]` lines and renumber remaining
references consistently in the body; OR (b) constrain the synthesizer to cite
**only** the numbered evidence it was given (evidence list is the fetched set),
so there is nothing to strip. (b) is cleaner long-term; (a) is a robust
belt-and-suspenders. Do both. ~30-60 min. Tests in `tests/test_citations.py`
(add a "strips URL AND removes the empty [n] slot" case).

---

## 4. P4 — Reviewer flags fabrication but the flagged claims STAY

**Symptom (GLM 5.2 report §7 quality summary):** "Reviewer flagged 3 unsupported
claim(s). Reviewer flagged 2 fabrication risk(s)." — yet those claims are still
in the body, presented as fact. `revisions: 3` = the loop hit its cap and
delivered anyway.

**Root cause:** the reviewer's `unsupported_claims` / `fabrication_flags` are
merged into the `validated_assessment` and printed in the title block (see
`report_builder_node` + `report_builder_helpers.merge_reviewer_into_assessment`),
but nothing **removes or quarantines** the offending sentences. After
`MAX` revisions the run proceeds regardless.

**Fix:** when the final verdict still carries fabrication flags, either (a)
strip/soften the flagged claims out of `draft_md` before writing, or (b) move
them into a clearly-labelled "⚠️ Unverified / flagged claims" appendix and drop
their confidence, so nothing flagged is presented as established fact. Also
surface the flag count more prominently (top of report, not just the summary
block). ~1-2h. Files: `nodes.py` (reviewer/report_builder), possibly a small
`quarantine_flagged_claims()` helper. Tests: extend `tests/test_reflexion.py`.

---

## 5. P5 — Confident fabrication for obscure / non-existent topics

**Symptom (GLM 5.2 run):** "GLM 5.2" is a bleeding-edge / possibly-nonexistent
model. The report confidently states it's "the latest flagship LLM from Z.ai,
1M-token context" citing `[8] build.nvidia.com/z-ai/glm-5.2/modelcard`, and
pads with GLM-4 / GLM-4.5 / GLM-4.1V facts as if continuous. The pipeline
manufactured a plausible narrative from thin+mixed evidence instead of saying
"insufficient reliable evidence for GLM 5.2 specifically."

**Root cause:** no "grounding gate". The synthesizer will always write a report;
`verify_citations` only checks URLs were *fetched*, never that the fetched
content actually *supports the claim about the queried entity*. For rare
entities the top web hits are farms + adjacent-topic pages, and the LLM bridges
the gap by inventing.

**Fix (highest-value, hardest):** add a grounding check before/inside synthesis:
does a threshold number of *credible* fetched sources actually mention the core
query entity (token/entity match against title+excerpt)? If not, the report
leads with an explicit "⚠️ Limited direct evidence found for <entity>; the
following is drawn from adjacent/uncertain sources" and the reviewer is
instructed to be stricter. Consider a dedicated `grounding_node` after
`credibility` or a guard inside `synthesizer_node`. ~2-3h. New
`tests/test_grounding.py`.

---

## 6. Suggested fix order (ROI)

| # | Defect | Effort | Files |
|---|---|---|---|
| 1 | P1 plan-preview fake URLs | 15 min | `bot.py:258`, `test_bot.py` |
| 2 | P3 empty source slots + dangling `[n]` | 30-60 min | `citations.py`, `nodes.py` synth, `test_citations.py` |
| 3 | P2 credibility floor + content-farm list | 1-2h | `credibility.py`, `nodes.py:filter_node`, `test_credibility.py` |
| 4 | P4 enforce reviewer fabrication flags | 1-2h | `nodes.py` reviewer/report_builder, `test_reflexion.py` |
| 5 | P5 grounding gate | 2-3h | new `grounding_node` / synth guard, `test_grounding.py` |

Do 1-3 first (they're the visible, cheap wins). 4-5 are the "trust" work.

---

## 7. Evidence — the GLM 5.2 report (verbatim, `report.md`, 2026-07-09 01:51)

```
# GLM 5.2
_Mode: **Long** • ... • findings: 13 • sources: 14 • revisions: 3_
> Reviewer flagged 3 unsupported claim(s). Reviewer flagged 2 fabrication risk(s).
...
### GLM-5.2 Model
GLM-5.2 is the latest flagship large language model from Z.ai (zai-org),
designed for long-horizon tasks with a solid 1M-token context window [8].
...
## Sources
[1] https://thetechbriefs.com/thudm-releases-glm-4-...   <- content farm
[2]                                                       <- EMPTY (P3)
[3] https://glm45.org/?                                   <- low quality
[5] http://arxiv.org/abs/2406.12793v2                     <- real GLM paper
[7]                                                       <- EMPTY (P3)
[8] https://build.nvidia.com/z-ai/glm-5.2/modelcard       <- claim basis for GLM 5.2
[9] http://arxiv.org/abs/2402.11651v2
[10] http://arxiv.org/abs/2312.10793v3
```

**Repro:** `/research GLM 5.2` (or any obscure/near-future model name) surfaces
P2/P3/P5. `/research metacognitive reinforcement learning transformers` surfaces
P1 (plan preview) — see the fabricated planner URLs.

---

## 8. What is ALREADY fixed (do not redo)

- Dead web search (`duckduckgo_search`→`ddgs`), subgraph wired in, planner URLs
  no longer trusted by the researcher, filter no longer empties reports,
  citation-integrity pass now runs (was silently crashing on dicts). — commits
  `bc44fa1`, `7acd9fa`.
- Phase 2: `/video` YouTube search, scrapling stealth fetch fallback, HITL
  Extend loop. — commit `669ceb5`.
- Telegram markdown crash ("Can't parse entities") + global error handler +
  `run.sh` auto-kills stale pollers. — commits `f245965`, `55bcff9`.

## 9. Known INFRA gaps (env, not code — carried forward)

- **Scrapling stealth is inert:** camoufox browser binary corrupted on this box
  ("Accessing a corrupted shared library" / spawn UNKNOWN). Try
  `A:\Hermes\Agents\intel-stack\venv\Scripts\python.exe -m camoufox remove` then
  `... -m camoufox fetch`; check Windows Defender isn't quarantining a DLL.
- **YouTube transcripts-as-evidence** need `curl_cffi` for yt-dlp impersonation,
  but the intel-stack venv has **no pip**. Bootstrap with
  `A:\Hermes\Agents\intel-stack\venv\Scripts\python.exe -m ensurepip` then
  `-m pip install curl_cffi`. (`/video` search itself works without this.)

## 10. Pointers

- Repo: `A:\Hermes\Agents\argus` (branch `feat/argus-fixes` == `main`).
- Launch: `cd /a/Hermes/Agents/argus && bash scripts/run.sh` (Git Bash; run.sh
  clears PYTHONPATH + kills stale pollers).
- Live pipeline trace (bypasses Telegram): pattern in
  `tests/manual_e2e.py`; the scratchpad `trace_aiq.py` / `trace_extend.py`
  drivers are good templates for reproducing quality issues off-Telegram.
- Prior handoffs: `tonight/HANDOFF-2026-07-08-AIQ-FIXES.md`,
  `tonight/HANDOFF-RESEARCH-2026-07-08.md`.
