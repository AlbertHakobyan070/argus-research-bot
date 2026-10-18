# Argus — MacBook handoff

Argus is a personal Telegram bot with two jobs: **downloading video** from
YouTube / X / Reddit / Instagram into a local vault, and running **multi-agent
research** (LangGraph) that produces Markdown + PDF reports.

This document moves it from the Windows box to a MacBook. The Windows instance
will be **off** while the Mac one runs — that is required, not optional (see
[One bot per token](#one-bot-per-token)).

> **If you are Claude Code reading this cold:** work through
> [Setup](#setup) in order. Do not skip [Step 5](#step-5--verify) — it is the
> only thing that distinguishes "installed" from "working". Everything stated
> here was verified on the Windows machine on 2026-08-07; things marked
> **UNVERIFIED ON MAC** could not be tested from Windows and need a real check.

---

## What changed right before this handoff

A round of download fixes landed (uncommitted in the working tree — see
[Git state](#git-state)). Context, because the symptoms will look familiar:

| Symptom | Cause | Fix |
|---|---|---|
| Bot said "✅ Saved <path>" but no video ever arrived in Telegram | python-telegram-bot defaults: `write_timeout` 5s, `media_write_timeout` 20s, `read_timeout` 5s. A 40 MB upload cannot finish in 20s; both `send_video` and the `send_document` fallback timed out and only logged | Generous timeouts on `Application.builder()`; both-sends-failed is now reported to the user instead of swallowed |
| Long X/Twitter videos died partway, leaving `.part` files | X serves long video as HLS with 800+ fragments, downloaded **serially**. A 72-min tweet needed ~40 min; the blanket 900s timeout killed it at fragment 871 | `--concurrent-fragments 8` (**measured 4.5× faster**: 30.1 MB → 135.9 MB in the same 45s window) + a *stall* timeout that measures silence, not total elapsed time |
| Completed videos on disk that the bot didn't know about | On timeout, `proc.kill()` killed yt-dlp but not its **ffmpeg child**, which kept writing after Argus reported failure | `_kill_tree()` (taskkill /T on Windows, `killpg` on macOS/Linux) + a finished file is now kept even if the exit code is non-zero |
| Vertical videos downloaded at reduced quality | The `auto` quality filter capped `height<=1080`, which rejects a normal 1080p **portrait** video (1080×1922) — i.e. most Reels/Shorts. Verified: a 1080×1222 tweet came down as 720×814 | Cap the **short side** via yt-dlp's `res` sort (`min(width,height)`); verified 1080×1222 and 1080×1440 now |
| Unhelpful errors | Raw yt-dlp stderr ("No video could be found in this tweet") with no next step | `explain_error()` appends actionable hints (cookies, `/quality auto`, resume) |
| `/find` needed a sibling repo | `_run_yt_dlp` shelled out to the `intel-stack` venv's yt-dlp although argus declares yt-dlp itself | Uses the argus venv — **this is why `/find` works on the Mac at all** |

New maintenance tool: `scripts/reconcile_vault.py` (see
[Step 4](#step-4--bring-the-vault-across-optional)).

---

## Prerequisites

| Thing | Why | Check |
|---|---|---|
| **Python 3.12** (not 3.13) | `pyproject.toml` pins `>=3.12,<3.13` | `python3.12 -V` |
| **uv** | package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **git** | repo | `git --version` |
| **ffmpeg** | merging video+audio | Comes free — `imageio-ffmpeg` ships an arm64 macOS wheel (verified on PyPI). Homebrew's is optional. |
| **FreeLLMAPI server** | `/research` and `/ask` only | ⚠️ See [Known gaps](#known-gaps-on-mac) |
| Telegram bot token | the bot | already in the Windows `.env` |

Media downloads need **none** of the LLM stack. If FreeLLMAPI never gets set
up on the Mac, `/fetch` and link-pasting still work.

---

## Setup

### Step 1 — unpack

```bash
mkdir -p ~/Agents && cd ~/Agents
tar -xzf ~/Downloads/argus-macbook-handoff.tar.gz
cd argus
```

### Step 2 — create the venv and install

The venv **must** be called `venv` in the repo root — `scripts/run_mac.sh`
looks for `./venv/bin/python`.

```bash
uv venv venv --python 3.12
VIRTUAL_ENV=venv uv pip install -e '.[dev,asr]'
```

`asr` pulls `faster-whisper`, needed for transcripts of X / Reddit / Instagram
video (YouTube uses captions instead). Drop it if you don't want a ~1 GB
model download on first use.

### Step 3 — secrets

The archive deliberately contains **no secrets**. Copy the five values from the
Windows `.env` (they are not reconstructible from anything in the archive):

```bash
cp .env.mac.example .env
```

Then fill in, from the Windows machine's `A:\Hermes\Agents\argus\.env`:

- `FREELLMAPI_API_KEY` — starts `freellmapi-`
- `TELEGRAM_BOT_TOKEN` — `<id>:<secret>`
- `TELEGRAM_ALLOWED_USER_ID` — your numeric Telegram id
- `EXA_API_KEY` — neural search (optional; falls back to DDGS)
- `FREELLMAPI_BASE_URL` — usually `http://127.0.0.1:3001/v1`

The bot refuses to start if the API key or bot token is missing or malformed,
and answers **only** `TELEGRAM_ALLOWED_USER_ID`.

### Step 4 — where downloads land 📁

This is the one setting you must get right. Everything Argus writes lives under
a single root, set by `ARGUS_VAULT_ROOT` in `.env`:

```
$ARGUS_VAULT_ROOT/
├── media/
│   ├── youtube/      ← downloaded video + .info.json sidecars
│   ├── x/
│   ├── reddit/
│   └── instagram/
├── transcripts/      ← whisper + caption text
└── research history/ ← reports (Markdown + PDF)
```

A leading `~` **is** expanded (that was fixed for this handoff — an
unexpanded `~` used to create a literal `~` folder inside the repo).

```bash
# .env — mirrors the Windows layout, so a copied vault drops straight in
ARGUS_VAULT_ROOT=~/DS_Vault/DS Main Vault/DS_AUA_Toolkit/Assets/Argus

# ...or somewhere simpler if you're starting fresh. Subfolders are created
# on first use; no need to mkdir anything.
ARGUS_VAULT_ROOT=~/Argus/vault
```

Do **not** point it at iCloud Drive / Dropbox. Large video files plus a
SQLite database in a sync folder is a corruption and upload-storm hazard.
An external disk is fine — but if it is unplugged mid-download, the download
fails, so prefer internal storage and move files later.

The two databases stay in the repo root by default
(`argus_library.sqlite`, `argus_checkpoints.sqlite`); override with
`ARGUS_LIBRARY_DB` / `ARGUS_CHECKPOINT_DB` if you want them elsewhere. Keep
them off synced folders too.

### Step 4b — bring the vault across (optional)

Only if you want the existing ~3.4 GB of downloads and past reports. The
library database in the archive already lists them with **Windows** paths, so
after copying you must remap:

```bash
# 1. copy the media/transcripts/reports from the Windows box (rsync/USB/etc.)
#    into whatever you set as ARGUS_VAULT_ROOT

# 2. rewrite the stored paths (old root -> new root)
./venv/bin/python scripts/reconcile_vault.py \
    --remap 'A:\DS_Vault' "$HOME/DS_Vault"

# 3. confirm every row resolves and nothing is orphaned
./venv/bin/python scripts/reconcile_vault.py
```

Step 3 prints anything on disk but missing from the library (`--register`
adopts it) and any library row whose file is gone. If you are **not** bringing
the vault, skip all of this — just delete `argus_library.sqlite` for a clean
slate, or keep it and let the reconciler show the old rows as missing.

`argus_checkpoints.sqlite` is **not** in the archive (53 MB of in-flight
LangGraph state that does not travel meaningfully). Any research run that was
mid-flight on Windows is gone; the bot recreates the file on first start and
tells you about orphaned runs.

### Step 5 — verify

Run these in order. Do not report success before all four pass.

```bash
# 1. test suite
PYTHONPATH="" ./venv/bin/python -m pytest -q
```

Expect **~489 passed**. The 6 `test_llm.py` / `test_graph.py` failures are
expected **iff** the FreeLLMAPI server isn't running — they are live-LLM tests
and fail with `httpx.ConnectError` to `127.0.0.1:3001`. Any *other* failure is
real and must be fixed before continuing.

```bash
# 2. downloader + ffmpeg resolve
PYTHONPATH="" ./venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from argus.media import resolve_ffmpeg
from yt_dlp.version import __version__ as v
print('yt-dlp :', v)
print('ffmpeg :', resolve_ffmpeg())"
```

Both must print a real value; `ffmpeg: None` means merged video+audio is
impossible and quality silently degrades.

```bash
# 3. a real download into the real vault  ← the actual proof
PYTHONPATH="" ./venv/bin/python -c "
import asyncio, sys; sys.path.insert(0,'src')
from argus.media import download_media
from argus.config import get_settings
r = asyncio.run(download_media(
    'https://www.youtube.com/watch?v=jNQXAC9IVRw',
    dest_root=get_settings().media_root, quality='auto'))
print('ok:', r.ok, '| err:', r.error)
print('path:', r.path)
print('%.2f MB in %.1fs' % (r.size_bytes/1e6, r.elapsed_s))"
```

```bash
# 4. start the bot, then send it a link in Telegram
./scripts/run_mac.sh
```

Startup log should show `ffmpeg: …`, `yt-dlp: …`, and
`Argus bot starting; long-polling…`. In Telegram, paste any YouTube/X link →
you get a download menu → tap `⬇ 1` → the file lands in the vault and (if
under 50 MB) is sent back to you.

---

## Running day to day

```bash
cd ~/Agents/argus && ./scripts/run_mac.sh
```

`run_mac.sh` kills any existing `argus.bot` poller, asserts it is gone, then
execs the new one. If it refuses to start it will tell you what is still alive.

### One bot per token

Telegram allows exactly one long-poller per token. A second instance makes
Telegram answer one and return **409 Conflict** to the other — and the loser
may be the one you just fixed, so a change appears not to take effect. Before
starting on the Mac, confirm the Windows bot is stopped. Never run both.

### Useful commands

| Command | Does |
|---|---|
| paste a link | download menu (video / transcript / both) |
| `/fetch <url…>` | download directly |
| `/find <query>` | search YouTube, pick from results |
| `/quality auto\|min\|max\|<height>` | download quality (see below) |
| `/research <topic>` | full multi-agent report (needs FreeLLMAPI) |
| `/ask <question>` | quick answer (needs FreeLLMAPI) |
| `/transcript`, `/transcripts` | speech-to-text |
| `/runs`, `/status`, `/delete` | manage runs and stored assets |
| `/help full` | the detailed guide (`docs/help.md`) |

### About `/quality`

Currently set to **`max`** (it is stored in the library DB and travels with it).
`max` takes the largest available stream, which is why past downloads hit
647 MB and 676 MB. Telegram's bot upload cap is 50 MB, so at `max` most long
videos are saved to the vault but **cannot** be delivered in chat — that is
expected, not a bug, and the bot now says so explicitly.

`/quality 720` or `/quality auto` keeps most clips deliverable. `auto` caps
the short side at 1080 and is orientation-aware.

---

## Known gaps on Mac

Read this before concluding something is broken.

### 1. FreeLLMAPI server — `/research` and `/ask` will not work without it

`.env` points at `http://127.0.0.1:3001/v1`, a **local** server that is not
part of this repo and is **not** in the archive. How it is installed on the
Windows box is not recorded anywhere in this project, so it could not be
documented here.

- **Impact:** all LLM features. Media download and `/find` are unaffected.
- **Action:** find how FreeLLMAPI was installed on Windows and repeat it, or
  point `FREELLMAPI_BASE_URL`/`FREELLMAPI_API_KEY` at any OpenAI-compatible
  endpoint. The bot only needs a well-formed key string to *start*; it needs a
  live server to actually answer.
- **UNVERIFIED ON MAC.**

### 2. `intel-stack` — degrades research source-fetching

`src/argus/tools.py` shells out to a sibling repo at
`A:\Hermes\Agents\intel-stack` for three scripts: `snatch.py`, `crawl.py`,
`article_convert.py` (plus `harvest.py`). Those live in their own venv with
`crawl4ai` / `markitdown` / `feedparser`.

- **Works anyway:** all media downloads, `/find`, YouTube transcripts, DDGS and
  Exa web search, PDF rendering (ReportLab, in-venv).
- **Degraded:** `/research` fetches source *pages* through
  `snatch_url` / `crawl_url` / `normalize_to_markdown`. Without intel-stack
  these fail, and the research engine falls back to search-result snippets
  instead of full page text — thinner reports, not a crash.
- **Action if you want full research:** port `intel-stack` too, then set both
  in `.env`:
  ```bash
  INTEL_STACK_DIR=~/Agents/intel-stack/scripts
  INTEL_PYTHON=~/Agents/intel-stack/venv/bin/python
  ```
  (Both are env-overridable as of this handoff; they used to be hardcoded.)

### 3. Login-walled content

Private X videos, some Instagram posts and age-restricted YouTube need cookies.
Set **one** of these in `.env`:

```bash
ARGUS_YTDLP_COOKIES_BROWSER=chrome     # or firefox / edge / safari / brave
ARGUS_YTDLP_COOKIES=~/argus-cookies.txt
```

macOS specifics: Safari is supported by yt-dlp (verified in the installed
version). Chrome **locks its cookie database while running** — quit it first.
Reading Safari's cookies may prompt for Full Disk Access for your terminal.
The bot now tells you when an error looks cookie-shaped.

### 4. Whisper on Apple Silicon

`faster-whisper` is configured for **CPU int8** (`transcribe.py`). It works on
Apple Silicon but does not use the GPU, so long audio is slow. First run
downloads the model (`ARGUS_WHISPER_MODEL`, default `small`).
**UNVERIFIED ON MAC.**

---

## Troubleshooting

| Symptom | Do this |
|---|---|
| `409 Conflict` in the log | Another poller has the token. Stop the Windows bot, and any second Mac terminal. |
| Bot starts, downloads nothing, `ffmpeg: None` in log | `VIRTUAL_ENV=venv uv pip install -U imageio-ffmpeg`, or set `ARGUS_FFMPEG=/opt/homebrew/bin/ffmpeg` |
| "No video could be found in this tweet" | Post is deleted/protected, or needs login → set `ARGUS_YTDLP_COOKIES_BROWSER`. The error message now says this. |
| A platform suddenly stops working | yt-dlp lags platform changes. `VIRTUAL_ENV=venv uv pip install -U --pre yt-dlp` (`--pre` gets nightlies, which is where extractor fixes land first). |
| Download "stalled — no progress for 300s" | Connection dropped. Re-send the same link: yt-dlp **resumes** from the `.part` file. |
| Big file saved but not sent to Telegram | Over the 50 MB bot cap. Expected. Use `/quality 720`, or open it from the vault. |
| Vault and library disagree | `./venv/bin/python scripts/reconcile_vault.py` (add `--register` to adopt orphans) |
| `.part` files piling up | Same tool lists them with age. `--prune-partials --older-than 7` deletes them — this destroys resume capability, so only for downloads you have abandoned. |
| `pydantic_core` import explosion | A foreign `PYTHONPATH` leaked in. `run_mac.sh` clears it; if running python directly, prefix `PYTHONPATH=""`. |

### Tuning knobs (all optional, all in `.env`)

```bash
ARGUS_MEDIA_FRAGMENTS=8         # parallel HLS fragments (the 4.5x win)
ARGUS_MEDIA_CONCURRENCY=2       # simultaneous downloads
ARGUS_MEDIA_STALL_TIMEOUT=300   # kill after N seconds of silence
ARGUS_MEDIA_TIMEOUT=5400        # absolute ceiling per download
ARGUS_TG_MEDIA_WRITE_TIMEOUT=1800   # raise on a slow uplink
```

---

## Git state

The download fixes described above are **uncommitted** in the working tree of
the archive. `git status` on the Mac will show modified `src/argus/*.py`,
`tests/*.py` and new files (`scripts/run_mac.sh`,
`scripts/reconcile_vault.py`, `.env.mac.example`, `handoff.md`).

`git log` still ends at `3e8c8eb fix(argus): PDF cosmetics + old-money
editorial redesign (#17)`. Commit them on the Mac once you have confirmed
Step 5 passes there — that way the commit records a state actually verified on
the machine it runs on.

## What is NOT in the archive, and why

| Excluded | Why |
|---|---|
| `.env` | live secrets; copy the 5 values by hand (Step 3) |
| `venv/` | Windows binaries; rebuilt in Step 2 |
| `argus_checkpoints.sqlite` | 53 MB of in-flight run state, not portable |
| `media/`, `transcripts/`, `research history/` | the 3.4 GB vault — copy separately if you want it (Step 4b) |
| `__pycache__/`, `.pytest_cache/`, `demo_output/` | build/run artifacts |

`argus_library.sqlite` (94 KB, 56 assets / 13 runs) **is** included — it is
your history and your `/quality` setting. Its WAL was checkpointed and
integrity-checked before packing.
