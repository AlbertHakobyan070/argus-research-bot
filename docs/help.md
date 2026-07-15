# Argus — full command guide

Argus is a single-user Telegram research + media bot. It runs a multi-agent
LangGraph engine for deep research (grounded in **live** Exa / web / arXiv /
GitHub search, not the model's memory — every source is fetched and read
into cited evidence notes), downloads real media from YouTube / X / Reddit /
Instagram into your vault, transcribes it, and keeps a resumable history of
every run.

Send `/help` for the compact quick-reference; this file is what `/help full`
delivers.

---

## Research

### `/research <topic>`
Runs the full deep-research loop with two human-in-the-loop (HITL) gates:

1. **Plan gate** — Argus drafts a research brief *and runs a live discovery
   wave first* (Exa/DDGS/arXiv/GitHub), then shows you the sub-questions plus
   the **real sources it found** (title + URL). The brief carries no URLs at
   all, so hallucinated links are structurally impossible — you approve
   actual evidence.
   Buttons:
   - **Length** (TL;DR / Short / Medium / Long / Lecture) — pick output depth.
   - **✅ Approve** — fetch the shown sources and continue.
   - **✏️ Edit** — reply with changes; Argus redrafts the plan and re-searches.
   - **❌ Cancel** — drop the run.
2. **Report preview gate** — after deep research (every source is read and
   digested into evidence notes), sectioned composition, and a 3-judge review
   panel, you see the report head.
   Buttons:
   - **📤 Send** — deliver the `.md` + `.pdf`.
   - **🔎 Extend** — another research pass to deepen it (capped).
   - **🔁 Revise** — reply with what to change; Argus rewrites the report
     from the same evidence with your notes (capped).
   - **❌ Cancel** — drop the report.

Pre-pin the length without the keyboard: `/research /length long <topic>`.

**Length modes**

| Mode | Shape |
|------|-------|
| `tldr` | one short paragraph |
| `short` | ~300–700 chars (default) |
| `medium` | 2–3 pages, sub-headings |
| `long` | 5–8 pages, 10–15 findings |
| `lecture` | 8–10 pages, Part I–IV + References + Appendix |

### `/ask <question>`
A quick, single-shot grounded answer. No plan, no gates — use it for fast
look-ups rather than full reports.

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
video posts. The result pool lives ~30 minutes.

### `/fetch <url> [url…]`
Download one or more media links straight into the vault — no search step. You
can also just **paste** any supported link into the chat and Argus offers the
same download / transcript menu. Supported: YouTube (incl. Shorts), X/Twitter
status links, Reddit posts / `v.redd.it`, Instagram reels/posts.

> Instagram (and some X) downloads need login cookies. Public content works out
> of the box; for the rest, set `ARGUS_YTDLP_COOKIES` (a cookies.txt) or
> `ARGUS_YTDLP_COOKIES_BROWSER` (e.g. `firefox`). Files over ~50 MB stay in the
> vault (Telegram's bot upload cap) and Argus sends you the path.

### `/quality [auto|min|max|<height>]`
Global download quality. `auto` caps at 1080p, `max`/`min` are the extremes, or
give a pixel height like `720`. With no argument it shows the current setting
and whether ffmpeg is available (needed to merge high-res video+audio).

### `/transcript <indices> [format=txt|srt]`
Fetch YouTube captions for videos picked from your last `/find` (or `/video`)
results. Index syntax: `2`, `2,4`, `2 4`, `1-3`, `all`. `format=srt` keeps
timestamps. Auto-caption line duplication (the "rolling window" repeat) is
collapsed automatically.

### `/transcripts`
Browse the latest transcripts saved in the vault, with **📤** buttons to
re-send any of them. Each line shows its `asset:<N>` id for `/append`.

### Transcription backends
- **YouTube** → caption extraction (fast, no media download).
- **X / Reddit / Instagram** (no caption tracks) → local **faster-whisper**
  speech-to-text over the downloaded clip. The first ever run downloads the
  model (a few hundred MB); later runs reuse it. CPU-only, so a few minutes for
  a long clip is normal.

---

## Research history (resume & extend)

Every run is registered in a SQLite library and gets its own checkpoint thread,
so runs survive bot restarts and can be picked back up.

### `/runs`
List this chat's runs (newest first) with their short ids and status glyphs.
Use the ids with `/continue` and `/append`.

### `/status`
Show what's currently in flight plus your recent runs and their report folders.

### `/append <run-id> <url…|asset:N…>`
Queue extra sources onto an existing run — web URLs, or vault transcripts by
their `asset:<N>` id (see `/transcripts`). They're ingested on the next
`/continue`.

### `/continue <run-id>`
- If the run **paused** at the plan or preview gate (even after a restart), it
  re-attaches and re-shows that gate.
- If the run **finished**: with appended sources queued it does an *append-only*
  pass (ingests exactly your sources, no new searches); otherwise it runs a
  fresh search pass to deepen the report. Either way it ends at a new preview.

Appended local files are cited as `file:///…` sources and pinned through
ranking, so your own materials always make it into the report.

### `/cancel [run-id|all]`
Cancel an in-flight run. With one run in flight the id is optional; with several,
pass the id or `all`.

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
  human-readable `run.md` sidecar, and registered in the library.
- The registry DB (`argus_library.sqlite`) and checkpoints
  (`argus_checkpoints.sqlite`) live in the project directory, **not** the synced
  vault, to avoid sync/lock corruption.
- Launch the bot from the desktop **Start Argus Bot** shortcut (it calls Git
  Bash + `scripts/run.sh`).
