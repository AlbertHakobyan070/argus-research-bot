# Argus — Handoff 2026-07-09 (desktop shortcut & current build state)

**From:** coding-app (MiniMax-M3). Short task: create a desktop
shortcut to enable the bot (with prior taskkill), and write a state
handoff for a fresh session.

**For:** the next session, or Albert when rebooting.

---

## 0. What just shipped (today, 2026-07-09)

Two artefacts on the Windows desktop, both freshly created:

| File | Size | Purpose |
|---|---|---|
| `C:\Users\Albert\Desktop\Start Argus Bot.lnk` | 1687 B | Double-clickable shortcut to enable the bot. |
| `C:\Users\Albert\Desktop\start-argus.bat` | 2064 B | The launcher the shortcut targets. |

What the .lnk does (so the next session can audit / move / rewrite):

1. Opens a normal console window.
2. Runs a PowerShell `Get-CimInstance Win32_Process` filter killing any
   `python.exe` whose commandline contains `argus.bot` (this is the
   pre-flight taskkill — the same filter `scripts/run.sh` already runs
   internally, duplicated here so the shortcut works standalone).
3. Sleeps 2 s (lets Telegram release the previous long-poll slot).
4. `pushd "A:\Hermes\Agents\argus"` → `bash scripts/run.sh` →
   `./venv/Scripts/python.exe -m argus.bot` long-polls.
5. On exit, prints "Bot exited (code …)" and pauses so the console
   doesn't vanish silently.

Notes for anyone rewriting it:
- `IconLocation` points at `A:\Hermes\Agents\argus\venv\Scripts\python.exe,0` — uses the venv's python icon (no custom `.ico` was made).
- `WindowStyle=1` means "normal visible console" — change to 7 for a minimised launch.
- The .bat is **self-contained** with respect to the kill step: even if
  `scripts/run.sh` is renamed later, the shortcut's pre-flight kill
  still works.

---

## 1. Current build state

**Repo:** `A:\Hermes\Agents\argus`
**Branch:** `feat/argus-async-passes`
**HEAD:** `1d853f4 feat(argus): /transcript command + youtube_video_transcript tool`

### Last three commits

```
1d853f4  feat(argus): /transcript command + youtube_video_transcript tool
6ec0f98  fix(argus):  wire structured-quarantine into report_builder_node
2eeccdd  feat(argus):  structured post-synthesis module
071f3b7  fix(argus):  replace empty-Long-mode skeleton with loud classified failure
```

### Tests

```
./venv/Scripts/python.exe -m pytest tests/ \
  --ignore=tests/manual_e2e.py \
  --ignore=tests/test_e2e_research.py -q
# expect: 293 passed in ~45s
```

(Source-of-truth numbers: 266 baseline + 27 new for the transcript
pipeline; see `tests/test_transcript.py`.)

### Live process at handoff time

| Item | Value |
|---|---|
| Polling PID | `28056` (started 2026-07-09 23:49:12) |
| Telegram long-poll socket pair | 2 connections, `149.154.166.110:443` |
| HF hermes-cli profiles running | alfred (9212), coding-app (17632) — none touch argus's token |
| Conflicting argus processes | none (`28056` is sole poller) |

### Recently created/modified files (since 2026-07-08)

```
A  tests/test_transcript.py                                 +239
M  src/argus/tools.py                                       +215  (adds _vtt_to_text,
                                                                    YouTubeTranscriptResult,
                                                                    youtube_video_transcript)
M  src/argus/bot.py                                         +245  (adds _video_pool, _parse_indices,
                                                                    transcript_cmd, CTA hint,
                                                                    registry import of InputFile)
A  tonight/HANDOFF-2026-07-09-TRANSCRIPTS.md                +213  (feature-level handoff)
A  tonight/HANDOFF-2026-07-09-DESKTOP.md                    +this file
```

---

## 2. What this commit enabled (recap)

```
/video opus magnum
  -> 6 numbered YouTube results, stashed in _video_pool[thread_id] (30-min TTL)
  -> footer CTA pointing the user at /transcript

/transcript 2,4
  -> resolves indices against the pool (clamps / dedupes / ranges / "all")
  -> per video: spawns yt-dlp --skip-download --write-auto-subs --sub-langs=en.*
       in the intel-stack venv, reads the .vtt from a per-call tempdir,
       strips cues via _vtt_to_text, encodes for in-memory use.
  -> sends: (1) a Telegram <pre> HTML message with title / url / channel /
       duration / language and ~3.5 kB preview; (2) the full text as a
       <video_id>.txt BytesIO upload.
  -> per-video fail-soft: a missing-caption video yields a ⚠️ chip and
       the rest of the batch still ships.
```

Index syntax accepted by `_parse_indices`:
`2` · `2,4` · `2 4` · `2, 4` · `1-3` · `3-1` (range reversed) · `all` ·
`*` · `/transcript 2,4` (slashed prefix is just stripped).

---

## 3. How to enable the bot (post-reboot, or if it died)

```bash
# From PowerShell / cmd / WindowsTerminal:
cd "C:\Users\Albert\Desktop"
.\start-argus.bat
```

Or just double-click **Start Argus Bot** on the desktop — same thing.

If you want to do it from inside the repo (e.g. SSH or bash):

```bash
cd /a/Hermes/Agents/argus
bash scripts/run.sh
```

Either way:
1. Any prior `argus.bot` Python process is killed (PowerShell CIM kill).
2. `bash scripts/run.sh` clears `PYTHONPATH`, runs the same kill again
   (belt-and-suspenders), then `exec ./venv/Scripts/python.exe -m argus.bot`.
3. The bot long-polls Telegram. Logs stream to the console window.

---

## 4. How to verify it's healthy

```bash
# 1. Telegram long-poll is held by exactly ONE process.
netstat -ano | grep "149.154.166.110:443 " | grep ESTABLISHED

# 2. That PID's commandline contains "argus.bot".
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" |
  Where-Object { \$_.CommandLine -like '*argus.bot*' } |
  Format-Table ProcessId, CreationDate, CommandLine"

# 3. No "409 Conflict" warnings in the last minute.
#    (open the bot's console window; or, if you redirected, tail argus_bot.log)

# 4. From Telegram, /video YOUR_QUERY should return 4-6 numbered hits.
```

If conflict: only one process owns the long-poll — kill any other
`argus.bot` PIDs and restart via the shortcut.

---

## 5. Open work (in priority order)

1. **Wire `run_post_synthesis_passes_async()` into `report_builder_node`.**
   The async orchestrator exists and is tested, just not called by the
   report_builder yet. ~1-2 h. (Inherited from
   `HANDOFF-2026-07-09-ASYNC-PASSES.md` §2.)
2. **Sidecar `synthesis_outcome` reconciliation.** Easy fix in
   `report_builder_helpers.build_sidecar_metadata()`. ~30 min.
3. **arxiv year-fabrication heuristic** in `credibility.py` — flag
   IDs like `26XX.YYYY` as `fabricated_path`. ~1 h.
4. **Migration of reviewer to populate `flagged_finding_ids`
   reliably** via substring-derive fallback. ~1 h.
5. **Parallel /transcript fetches** with a 2-3-slot semaphore. ~30 min.
6. **Persistent transcript cache** — the `%TEMP%/argus_ytt_out`
   directory grows unbounded; add an ageing cleanup. ~15 min.
7. **`format=srt` mode** that ships the original `.vtt` (with
   timestamps) instead of stripped plain text. ~30 min.

---

## 6. Pointers

- `tonight/HANDOFF-2026-07-09-ASYNC-PASSES.md` — the §4.1
  fragmentation fix; explains the structured-quarantine model that
  made `/transcript` safe to build on.
- `tonight/HANDOFF-2026-07-09-TRANSCRIPTS.md` — Phase-2 transcript
  pipeline, file-by-file.
- `tonight/HANDOFF-2026-07-09-QUALITY.md` — earlier P1-P5 quality
  fixes (grounding, quarantine reliability).
- `tonight/HANDOFF-2026-07-08-AIQ-FIXES.md` — citation-integrity
  lift from NVIDIA AI-Q.
- `tests/test_transcript.py` — 27 hermetic cases (VTT strip, tool
  fail-soft, parser parametrisation, pool TTL/isolation).
- `src/argus/tools.py` — `youtube_video_transcript` (line ~1130),
  `_vtt_to_text`, `YouTubeTranscriptResult`.
- `src/argus/bot.py` — `_video_pool`, `_parse_indices`,
  `transcript_cmd` (line ~660), `_pool_put/_pool_get`.

---

**Status: bot ENABLED (PID 28056 polling), `/transcript` LIVE, 293 tests
green, desktop shortcut ready. Bring it back via the .lnk after any
reboot or process death.**
