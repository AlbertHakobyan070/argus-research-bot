# Argus — full command guide

Argus is a single-user Telegram research + media bot. It runs a multi-agent
LangGraph engine for deep research (grounded in **live** Exa / web / arXiv /
GitHub search, not the model's memory — every source is fetched and read
into cited evidence notes, and every claim in the report is checked against
those notes), downloads real media from YouTube / X / Reddit / Instagram into
your vault, transcribes it, and keeps a resumable history of every run.

Send `/help` for the compact quick-reference; this file is what `/help full`
delivers.

---

## Research

### `/research <topic>`
Runs the full deep-research loop with two human-in-the-loop (HITL) gates. A
live progress card (stage, counts, last event) follows the run while it works.

1. **Plan gate** — Argus drafts a research brief *and runs a live discovery
   wave first* (Exa/DDGS/arXiv/GitHub), then shows you the sub-questions plus
   the **real sources it found** (title + URL). The brief carries no URLs at
   all, so hallucinated links are structurally impossible — you approve
   actual evidence. The card also names the report's archetype, the recency
   window, the engines profile and an Exa cost estimate (when Exa is on).
   Buttons:
   - **Length** (TL;DR / Short / Medium / Long / Lecture) — the report's shape.
   - **Depth** (Quick / Standard / Deep / Exhaustive, or **Auto**, which
     follows the length) — how much research: waves, fetches, gap waves and
     rewrites. Deeper presets read more sources and take longer.
   - **⚙ profile** — cycles the engines profiles (see `/engines`); a profile
     that cannot run is marked ⚠ and Approve refuses it with the reason.
   - **🕒 recency** — cycles auto (the brief decides) / past day / week /
     month / year / any (no recency filter).
   - **✅ Approve** — research the plan with these choices.
   - **✏️ Edit** — reply with changes; Argus redrafts the plan and re-searches.
   - **❌ Cancel** — drop the run.
2. **Report preview gate** — after deep research (every source is read and
   digested into evidence notes), the outline, the section writers, the
   analyst and the review panel, you see the report head. It opens with a
   quality line (how many findings were judged and how many claims were
   cut), any warning banner, and the run's degradation notes.
   Buttons:
   - **📤 Send** — deliver the `.md` + `.pdf` (and finish the run).
   - **🔎 Extend** — another research pass to deepen it (capped rounds).
   - **🔁 Revise** — reply with what to change; Argus rewrites the report
     from the same evidence with your notes (capped rounds).
   - **❌ Cancel** — drop the report.

Pre-pin the length without the keyboard: `/research /length long <topic>`.

**Length modes**

| Mode | Shape |
|------|-------|
| `tldr` | one short paragraph |
| `short` | a compact answer (default) |
| `medium` | a few pages with sections |
| `long` | a longer sectioned report |
| `lecture` | Part I–IV + References + Appendix |

**What the report guarantees**
- **Cut, not flagged.** The grounding judge checks every finding against the
  quotes of the notes it cites. A claim it rejects is removed from the body
  (whole sentences) and listed under "⚠ Flagged claims (removed from the
  body)"; one that cannot be located stays and is listed as such.
- **TL;DR from the report itself.** The TL;DR is compiled from the kept,
  judged takeaways (or each section's first kept finding), never written by a
  model on its own.
- **Analysis.** Every report but a TL;DR gets an "Analysis" section with
  takeaways labelled high / medium / low confidence (unless the run's config
  turns the analyst off).
- **Banners.** "⚠ Not fact-checked" means some findings were not judged (the
  judge failed, or a section cites sources but had no finding to check);
  "⚠ Degraded run" means a per-item step (digests, finding extraction,
  section writers, judge batches) skipped a large share of its items. Both
  say why, and the preview repeats them.
- **Degradation notes** list every declared fallback the run took (for
  example keyword triage instead of the reranker), by round.
- `evidence.json` next to the report holds every source, note, quote check,
  finding and verdict (the console's tap-to-verify reads it).

### `/ask <question>`
A quick, single-shot answer from the model's own knowledge. No search, no
plan, no gates — use it for fast look-ups, not for claims you need cited.

---

## The console (Mini App)

The **Argus** menu button next to the chat's text field opens the console:
the runs list, a run's live progress, its plan gate (with the same knobs as
the chat card), the report with **tap-to-verify** citations (tap `[n]` to see
the source and its quotes) and **Send to chat**, a new-research form, and
the engines settings. It follows Telegram's light/dark theme. A session that
is too old shows "Session expired — reopen from the menu button".

The console is optional: if it is down, research and media in chat keep
working, and `/status` says why. With the default quick tunnel the console's
address changes on every restart; the menu button is always kept current.

---

## Media → vault

Downloads and transcripts are written into your DS vault:

```
…\Assets\Argus\media\<platform>\        (the .mp4 + .info.json)
…\Assets\Argus\transcripts\<platform>\  (.txt / .srt)
```

### `/find [yt|shorts|reddit] <query>`
Search for videos, then act on each result with buttons:
- **⬇** download to the vault
- **📝** transcript (captions for YouTube, speech-to-text otherwise)
- **⬇+📝** both

Default scope is YouTube; `shorts` biases to Shorts, `reddit` searches Reddit
video posts. The result pool expires after a while; search again if a button
stops working.

### `/fetch <url> [url…]`
Download one or more media links straight into the vault — no search step. You
can also just **paste** any supported link into the chat and Argus offers the
same download / transcript menu. Supported: YouTube (incl. Shorts), X/Twitter
status links, Reddit posts / `v.redd.it`, Instagram reels/posts.

> Instagram (and some X) downloads need login cookies. Public content works out
> of the box; for the rest, set `ARGUS_YTDLP_COOKIES` (a cookies.txt) or
> `ARGUS_YTDLP_COOKIES_BROWSER` (e.g. `firefox`). Files over Telegram's bot
> upload cap stay in the vault and Argus sends you the path.

### `/quality [auto|min|max|<height>]`
Global download quality. `auto` picks a sensible cap, `max`/`min` are the
extremes, or give a pixel height like `720`. With no argument it shows the
current setting and whether ffmpeg is available (needed to merge high-res
video+audio).

### `/transcript <indices> [format=txt|srt]`
Fetch YouTube captions for videos picked from your last `/find` (or `/video`)
results. Index syntax: `2`, `2,4`, `2 4`, `1-3`, `all`. `format=srt` keeps
timestamps. Auto-caption line duplication (the "rolling window" repeat) is
collapsed automatically.

### `/transcripts`
Browse the latest transcripts saved in the vault, with **📤** buttons to
re-send any of them. Each line shows its `asset:<N>` id for `/append`.

### Transcription engines
- **YouTube** → caption extraction first (fast, no media download), then
  speech-to-text if the captions fail. The order is configurable.
- **X / Reddit / Instagram** (no caption tracks) → speech-to-text over the
  downloaded clip.
- The speech-to-text engine is local **faster-whisper** by default (CPU; the
  first run downloads its model), or an OpenAI-compatible audio API such as
  Groq's Whisper. `/engines transcriber` shows and switches it (below).
- The transcript header names the engine that produced it.

---

## Research history (resume & extend)

Every run is registered in a SQLite library and gets its own checkpoint thread,
so runs survive bot restarts and can be picked back up. Buttons on messages
sent before a restart keep working.

### `/runs`
List this chat's runs (newest first) with their short ids and status glyphs.
Use the ids with `/continue` and `/append`.

### `/status`
Each tracked run's stage, elapsed time and last event (or the gate it waits
at), your recent runs and their report folders, and the console link with its
health.

### `/append <run-id> <url…|asset:N…>`
Queue extra sources onto an existing run — web URLs, or vault transcripts by
their `asset:<N>` id (see `/transcripts`). They are always ingested, outside
every budget, on the next `/continue` or Extend; one that cannot be fetched is
named to you with the reasons, and `/append`ing it again queues it again.

### `/continue <run-id>`
- If the run **waits** at the plan or preview gate (even after a restart), it
  re-attaches and re-shows that gate.
- If the run **stopped** mid-way (cancelled, interrupted by a restart, or
  failed), it resumes from its last saved step.
- If the run **finished**: with appended sources queued it does an *append-only*
  pass (ingests exactly your sources, no new searches); otherwise it runs a
  fresh search pass to deepen the report. Either way it ends at a new preview.

Appended local files are cited as `file:///…` sources and pinned through
ranking, so your own materials always make it into the report.

### `/cancel [run-id|all]`
Stop a run. A run that is working **finishes its current step**, saves its
checkpoint and stops (the card says "cancelling — finishing the current
step"); `/continue <id>` resumes it later. Research stops between fetches and
keeps what it already read. A run waiting at a gate is cancelled at once.
With one run in flight the id is optional; with several, pass the id or `all`.

Only one run researches at a time (free-tier limits and the Exa budget): a
second start or approve while one runs answers "busy" and names it. Runs
waiting at a gate do not count.

---

## `/engines`
Which model serves each step of the pipeline. Engines are grouped in
**profiles** (the shipped ones are in `argus_engines.json`; the default uses
the FreeLLMAPI proxy). Each role (brief drafting, digest, section writing,
grounding judge, …) has a primary engine, optional fallbacks and a declared
failure policy.

- `/engines` — the active profile, every role's binding, validation problems,
  the last run's served models, the judge-diversity check and this month's
  Exa spend.
- `/engines use <profile>` — switch profiles (refused, with the reason, if the
  profile cannot run: a missing API key, an unchosen model).
- `/engines tier <cheap|strong|judge> <provider>/<model>` — change a tier;
  every role that follows the tier moves with it.
- `/engines set <role> <provider>/<model>`, `/engines fallback <role>
  <provider>/<model>` — pin or extend one role.
- `/engines reset [profile]` — drop your changes.
- `/engines models <provider>` — the models a provider serves.
- `/engines transcriber [faster_whisper|openai_audio [provider/model]]` —
  show or switch the speech-to-text engine.

Provider definitions (URLs, key names) are file-only. The console's Engines
page does the same, one change at a time.

---

## `/delete`
Free up disk space. Pick a category (Runs / Media / Transcripts), multi-select
items (each shows its size), confirm the total, and Argus removes the files, the
report folders, the checkpoint threads, and the registry rows. In-flight runs
are refused, and only files strictly inside the Argus vault roots are ever
touched.

---

## Notes
- Reports are written to the vault research-history folder with a
  human-readable `run.md` sidecar, `metadata.json` and `evidence.json`, and
  registered in the library.
- The registry DB (`argus_library.sqlite`) and checkpoints
  (`argus_checkpoints.sqlite`) live in the project directory, **not** the synced
  vault, to avoid sync/lock corruption.
- Launch the bot from the desktop **Start Argus Bot** shortcut (it calls Git
  Bash + `scripts/run.sh`).
