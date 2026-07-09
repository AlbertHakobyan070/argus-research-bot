# Argus — Handoff 2026-07-09 (transcripts): /video → /transcript pipeline shipped

**From:** coding-app (MiniMax-M3) — explicit follow-up after
`HANDOFF-2026-07-09-ASYNC-PASSES.md`. Albert's request from the couch
while the bot was being enabled: *"Check it out, if/when everything is
correct, proceed to the functionality of retrieval of the transcribtion
of the selected video from the video suggestion pool (given the set of
indeces of the suggestions from the user as the follow up query)."*

**For:** next Hermes session or Albert reviewing.
**Branch:** `feat/argus-async-passes` (continues from `6ec0f98`).
**Latest commits:**
```
<this commit>  feat(argus): /transcript command + youtube_video_transcript tool
            — Phase 2 (T7.5?) lets user pick videos from /video results
            and receive plain-text captions as .txt attachments
6ec0f98   fix(argus): wire structured-quarantine into report_builder_node + reviewer
2eeccdd   feat(argus): structured post-synthesis module
071f3b7   fix(argus): replace empty-Long-mode skeleton with loud classified failure
```
**Tests:** 293 passing (266 prior baseline + 27 new for the transcript
pipeline, hermetic — no real yt-dlp invocations).

---

## 0. TL;DR

Users can now do:

```
/video opus magnum
🎬 YouTube results for opus magnum:
1. [...]
2. [...]
3. [...]
…
↪ Reply with `/transcript <indices>` to fetch captions — e.g. /transcript 2,4 or /transcript all

/transcript 2,4
📝 Transcript 2/6 — <title>
   <url>
   channel · 5:23 ·  lang: en-en
   <preview ~3500 chars>
   <attached jNQXAC9IVRw.txt — full plain text>
✅ Delivered 2/2 transcript(s).
```

The whole flow is **fail-soft** (a missing caption returns a per-video
warning instead of crashing the chat), **pooled** (TTL'd in-memory,
replaced on each new `/video`), and **non-destructive** (no video
download — just `--write-auto-subs --skip-download`).

---

## 1. What is NEW on `feat/argus-async-passes`

### `src/argus/tools.py` — Phase 2 transcript layer

- **`youtube_video_transcript(url, *, langs="en.*", timeout=90) -> YouTubeTranscriptResult`**
  - Spawns ``yt-dlp --skip-download --write-auto-subs --sub-langs=en.*
    --write-info-json`` in the intel-stack venv (same path as the
    existing `youtube_search`).
  - Reads the resulting ``.vtt`` from a per-call tempdir, strips cues
    via the new ``_vtt_to_text`` helper, and returns:
    - ``transcript_text`` — plain text, one cue-block per line.
    - ``transcript_bytes`` — utf-8 bytes (caller ships via ``BytesIO``
      to Telegram, no disk hop).
    - ``transcript_path`` — best-effort on-disk copy under
      ``%TEMP%/argus_ytt_out/<video_id>.txt`` (fallback only).
    - ``language``, ``title``, ``channel``, ``duration`` — recovered
      from the filename and ``.info.json``.
    - ``error`` — populated when ``ok=False``; callers always get a
      reply, never a crash.
- **`_vtt_to_text(vtt_path) -> str`** — drops ``WEBVTT`` header, cue
  timings, ``NOTE``/``STYLE``/``Kind:``/``Language:`` metadata, and
  positioning tags like ``<c.color>...`` / ``<00:00:12.616>``.
  Multi-line cue bodies are joined with single spaces; ``"\\r"`` is
  normalised for Windows-saved files. Empty or unreadable -> ``""``.

### `src/argus/bot.py` — `/video` + `/transcript` glue

- `_video_pool: dict[thread_id, {results, ts}]` — 30-minute TTL.
  `_pool_put` / `_pool_get` are the only accessors; `_pool_get` evicts
  stale entries on read. New `/video` calls overwrite the pool for
  that chat (last search wins).
- `_parse_indices(text) -> list[int]` — accepts `2`, `2,4`, `2 4`,
  `2, 4`, `1-3`, `3-1`, `all`, `*`. Bad tokens silently dropped; the
  caller does out-of-range validation against the pool length.
- `video_cmd` — same search behaviour as before, now also stashes the
  result list into the pool and shows a footer CTA pointing to
  `/transcript <indices>`.
- `transcript_cmd` — resolves indices against the pool, fetches each
  video's captions in `asyncio.to_thread` (yt-dlp is blocking; one
  per call is intentional to avoid YouTube rate-limit hits), and
  delivers:
  - A short Telegram `<pre>` HTML message with title / URL / channel /
    duration / language and a 3.5 kB preview.
  - The full text as a `<video_id>.txt` `send_document` upload via
    `InputFile(BytesIO(transcript_bytes), …)`.
  - Per-video error chips on failure instead of crashing the batch.
- `HELP_TEXT` lists `/transcript` and `build_application` registers
  the `CommandHandler("transcript", transcript_cmd)`.

### `tests/test_transcript.py` — 27 hermetic test cases (new file)

| Layer | Tests |
|---|---|
| VTT strip | header/timing strip · missing file -> "" · empty file · NOTE/STYLE drop |
| Tool happy path | hermetic monkeypatch of `_run_yt_dlp` + `tempfile.TemporaryDirectory`; asserts args include `--skip-download`, `--write-auto-subs`, `--sub-langs=...` |
| Tool fail-soft | empty URL · non-http URL · yt-dlp ran but no ``.vtt`` |
| Index parser | 14 cases via `@pytest.mark.parametrize` |
| Pool | round-trip · TTL expiry · replace on new search · thread isolation |

---

## 2. Verified on this commit

- `pytest tests/ --ignore=manual_e2e.py --ignore=test_e2e_research.py`
  → **293 passed in 45 s** (266 baseline + 27 new).
- Smoke test against real YouTube:
  `./venv/Scripts/python.exe -c "from argus import tools;
   r = tools.youtube_video_transcript(
     'https://www.youtube.com/watch?v=jNQXAC9IVRw'); print(r.ok, len(r.transcript_text))"`
  → `True 217` (Me at the zoo, the 19-second video).
- `bot.py` imports cleanly; new symbols resolve:
  `transcript_cmd`, `_video_pool`, `_pool_get`, `_pool_put`,
  `_parse_indices`.
- `/help` regenerated, includes the new command.
- Bot was restarted after the prior handoff (`6ec0f98` was already
  deployed; this commit needed a reload). New PID is on the Telegram
  long-poll slot. No 409 Conflict.

---

## 3. What is NOT yet done (open work)

1. **Parallel transcript fetches.** Currently sequential
   `asyncio.to_thread` calls — one per video — to avoid hammering
   YouTube. For `all` on a 6-video pool, total wall time is
   ~6 × (5-15 s) = 30-90 s. Parallelism with a semaphore (2-3 in
   flight) would cut this noticeably. ~30 min.
2. **Persistent transcript cache.** The `%TEMP%/argus_ytt_out/`
   directory grows unbounded. Add a `cleanup_transcripts(older_than)`
   maintenance step or auto-clean when the .txt hits >10 MB.
3. **Translate / summarise.** Albert might want to ask
   `/transcript 2,4 /lang ja` or feed transcripts back into
   `/research` as primary sources. Both would require wiring the
   tool as a `@tool` (already stubbed via `make_langchain_tools`).
4. **SRT/VTT timestamps.** Right now we strip all timing data, so the
   user can't click-to-seek in YouTube. A `format=srt` mode would
   keep the original `.vtt` (already on disk after `--write-auto-subs`)
   — wire `_send_transcript_file(VTT)` behind that flag.
5. **T8 / T9 cluster (inherited from HANDOFF-ASYNC-PASSES §2)** —
   wire `run_post_synthesis_passes_async()` into `report_builder_node`,
   sidecar `synthesis_outcome` metadata, arxiv year-fabrication, etc.
   Those still live on top of this branch.

---

## 4. Quick repro checklist (cold session)

```bash
cd /a/Hermes/Agents/argus
git checkout feat/argus-async-passes
git log --oneline -3
# expects head: feat(argus): /transcript command + youtube_video_transcript tool

export PYTHONPATH=""
./venv/Scripts/python.exe -m pytest tests/test_transcript.py -q
# expect: 27 passed

./venv/Scripts/python.exe -m pytest tests/ \
  --ignore=tests/manual_e2e.py \
  --ignore=tests/test_e2e_research.py -q
# expect: 293 passed

# restart bot
bash scripts/run.sh
```

Then in Telegram:

```
/video meta cognitive reinforcement learning
/transcript 2
/transcript 1,3
/transcript all
/transcript 1-3,5
```

Each should yield one Telegram message + one .txt attachment per video.

---

## 5. Pointers

- **New file:** `tests/test_transcript.py` (235 lines).
- **Modified:**
  - `src/argus/tools.py` — added `tempfile` import + `re` import
    (was missing); added `_vtt_to_text`, `YouTubeTranscriptResult`,
    `youtube_video_transcript` (~210 new lines).
  - `src/argus/bot.py` — added `re`, `time`, `InputFile` imports;
    `_video_pool` + helpers; `_parse_indices`; CTA hint in `video_cmd`;
    new `transcript_cmd`; HELP_TEXT + handler registration (~170 new).
- **Tree:** clean as of this commit; running PID points at the new code.
- **Prior handoffs:**
  - `tonight/HANDOFF-2026-07-09-ASYNC-PASSES.md` — §4.1 fragmentation fix
  - `tonight/HANDOFF-2026-07-09-EMPTY-LONG.md` — fragmentation analysis
  - `tonight/HANDOFF-2026-07-09-QUALITY.md` — P1-P5 quality fixes
  - `tonight/HANDOFF-2026-07-08-AIQ-FIXES.md` — citation-integrity lift

**Status: Phase-2 transcript pipeline LIVE; tests green; awaiting first
in-Telegram smoke from Albert.**
