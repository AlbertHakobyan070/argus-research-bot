# Argus — handoff (running it, moving it, what v4 changed)

Argus is a personal Telegram bot with two jobs: **downloading video** from
YouTube / X / Reddit / Instagram into a local vault, and running **multi-agent
research** (LangGraph) that produces Markdown + PDF reports with an
`evidence.json`. A Telegram Mini App console (the chat's **Argus** menu
button) shows the same runs.

This document covers running it day to day, moving it to another machine
(it was written for the Windows → MacBook move and still follows that
order), and what the v4 rebuild changed for an operator. Only one machine
may run the bot at a time — that is required, not optional (see
[One bot per token](#one-bot-per-token)).

> **If you are Claude Code reading this cold:** work through
> [Setup](#setup) in order. Do not skip [Step 5](#step-5--verify) — it is the
> only thing that distinguishes "installed" from "working". Items marked
> **UNVERIFIED ON MAC** could not be tested from Windows and need a real check.

---

## What v4 changed (operator view)

The design is `docs/argus-v4-design.md`; what each work package built, with
its decisions and evidence, is in `docs/v4-notes/` (start at its
`README.md`). In short:

**Engines (which model does what)**
- Every LLM call names a pipeline role; `argus_engines.json` binds each role
  to a provider and model through **profiles**, with explicit fallbacks and a
  declared failure policy (`raise` / `skip_item` / `degrade`). The default
  profile reproduces the old FreeLLMAPI setup; others use paid APIs (keys by
  env var name) or local servers. `/engines` and the console show and change
  the bindings; provider definitions stay file-only.
- The bot starts with the proxy down or `FREELLMAPI_API_KEY` unset: engines
  problems are logged and DM'd once; media, `/find` and `/runs` keep working;
  `/research`, `/ask`, the gate buttons and `/continue` answer with the
  engines error until it is fixed. A malformed key now shows at a run's first
  call, not at startup.
- Each engine gets a few tries with backoff (honouring `Retry-After` up to a
  cap) and the SDK itself never retries, so under a quota burst a call gives
  up sooner than in v3. `model_calls` records role, requested vs served model,
  routing, attempt and retry reason; `metadata.json["engines"]` summarises
  them, and `/engines` shows the last run's served models and the
  judge/writer family check.
- `argus_engines.json` and the local file must be UTF-8 (a BOM is fine);
  anything else is a startup problem naming the file.

**Research depth**
- Depth presets (`quick` / `standard` / `deep` / `exhaustive`, chosen on the
  plan card or Auto from the length) drive the waves, fetches, the
  per-sub-question quota, the first-pass cap, the gap reserve and the per-run
  Exa budget. Each user Extend gets a fresh allowance. All spend goes through
  one ledger, and the run's Exa cost lands in the library (`/engines` shows
  the month to date).
- Research no longer needs the intel-stack: a native fetcher reads HTML,
  PDFs, arXiv and GitHub READMEs. It refuses private and loopback targets
  (the proxy and the console included), connects only to the address it
  checked (or by name through an environment HTTP(S) proxy), and stops a
  download at the configured byte cap. The intel-stack rungs still join the
  chain when both `INTEL_*` paths exist.
- Duplicate URL forms are fetched once; candidates are reranked before they
  are fetched; long documents are digested in parts; every quote is checked
  verbatim against its document (`quote_verified`).
- `/append` sources are always ingested, outside every budget; one that
  cannot be fetched is named with the rungs tried, and appending it again
  queues it again.
- A run whose fetched sources all fail to digest stops with an error instead
  of shipping an ungrounded report.

**Trust (the report)**
- A grounding judge checks every finding, in batches, against the full
  quotes of the notes it cites. Rejected claims are **cut** (whole sentences)
  and listed; an unlocated flag stays and is listed as such. Any unjudged
  finding gives "⚠ Not fact-checked"; a per-item step that skipped a large
  share of its items gives "⚠ Degraded run". Every declared fallback is a
  line in "## Degradation notes".
- The TL;DR is compiled deterministically from kept takeaways or findings;
  no model writes it. An analyst step adds "## Analysis" with confidence
  labels. "## Uncited specifics" lists numeric sentences without a citation.
- The panel can send a run back for a targeted gap wave, or rewrite only the
  weak sections, within the preset's bounds.
- New report files: `evidence.json` beside `report.md`; `metadata.json` gains
  `run_config`, `degradations`, `degraded`, `exa_usd`, `engines`.
- A missing bundled font fails the PDF (the markdown still ships, with a
  `pdf:` note); there is no Chromium fallback.

**Runs (cancel, continue, status)**
- One live drive per run, and one researching run at a time: a second start
  or resume answers "busy" naming the running one; a second tap on a running
  run answers "already running".
- `/cancel` **drains**: the run finishes its current step, saves its
  checkpoint and stops (research stops between fetches and keeps what it
  read); `/continue <id>` resumes it. Only after `ARGUS_CANCEL_DRAIN_TIMEOUT_S`
  is the task cancelled outright (logged). A run at a gate is cancelled at once.
- Shutdown drains live drives for at most `ARGUS_SHUTDOWN_DRAIN_S`; new
  actions are refused meanwhile. A drained run is `cancelled` and resumable.
- Buttons on messages sent before a restart keep working (the run attaches on
  the tap). A run interrupted mid-step by a restart is marked `cancelled` and
  DM'd with the `/continue` hint.
- `/research` and `/ask` show a live progress card. The plan card has depth,
  engines-profile and recency knobs and an Exa estimate; the preview opens
  with a quality line, the banners and the degradation notes. Send finishes
  the run, so `/continue` on a sent run extends it.
- `/status` shows each run's stage and last event, and the console's URL and
  health.

**Console (Mini App)**
- The bot serves it on `127.0.0.1:${ARGUS_WEBAPP_PORT}` and, with the default
  quick tunnel, starts `tools/bin/cloudflared(.exe)` (or `ARGUS_CLOUDFLARED`),
  registers the chat's **Argus** menu button and logs the tunnel URL;
  cloudflared's output goes to `logs/cloudflared.log`. The URL changes on every
  start; the menu button follows it.
- It is optional: a taken port, a missing cloudflared, a dead tunnel or a
  failed menu button is logged, DM'd ("⚠️ Console down — …") and shown in
  `/status`, and chat keeps working. A killed cloudflared is restarted with
  backoff and the new URL DM'd. Every API call is authenticated with
  Telegram's signed `initData` for `TELEGRAM_ALLOWED_USER_ID` only; report
  HTML is sanitised, and the CSP allows no inline script.
- To refresh the UI kit snapshot: `./venv/Scripts/python.exe
  scripts/sync_arsenal.py`, then commit `src/argus/webapp/static/vendor/arsenal/`.

**Transcription**
- The engine is configured in `argus_engines.json` `transcriber`: local
  faster-whisper (model, device and compute type in its block) or an
  OpenAI-compatible audio API (`/engines transcriber openai_audio groq/<model>`,
  needs `GROQ_API_KEY`). YouTube tries captions, then ASR, in the configured
  order. Transcript headers name the engine (`faster_whisper:<model>`,
  `openai_audio:<provider>/<model>`, `captions→…`).

**Removed env knobs** (a startup WARNING names each one still set; the list
is `REMOVED_ENV_KNOBS` in `src/argus/engines/registry.py`): `ARGUS_TIMEOUT`,
`ARGUS_LLM_RETRIES`, `ARGUS_RESEARCH_WAVES`, `ARGUS_FETCH_PER_WAVE`,
`ARGUS_TOTAL_FETCH_CAP`, `ARGUS_EXA_MAX_CALLS`, `ARGUS_MAX_REVISIONS`,
`ARGUS_FETCH_CONCURRENCY`, `ARGUS_DIGEST_CONCURRENCY`,
`ARGUS_COMPOSE_CONCURRENCY`, `ARGUS_WHISPER_MODEL`, `PERPLEXITY_API_KEY`.
New ones are in `.env.example`.

**Migration and compatibility**
- The first start on v4 adds the new library columns (existing rows are
  untouched). Checkpoints keep loading: a pre-v4 run resumed through any
  button or `/continue` gets its length's legacy preset (`origin="legacy"`),
  recorded once in its errors and its report's notes. A run config stored
  during the v4 build that still names the retired `legacy.*` roles loads with
  them dropped. Older-topology (v2) checkpoints can be re-attached and sent,
  not resumed.
- Report file names are unchanged; `evidence.json` is new. The vault layout
  is unchanged.

**Scripts**
- `scripts/live_check.py` (graph-direct, prints an acceptance table, exits
  non-zero on any failed check) replaces `tests/manual_e2e.py`;
  `scripts/verify_live.sh` runs it after the live tests. `verify_live.sh`
  needs a valid `.env`, `TELEGRAM_BOT_TOKEN` included: it reads the proxy URL
  through `get_settings()`, which refuses a missing or malformed token. The old opt-in
  `tests/test_e2e_research.py` is gone.
- `python -m bench.metrics` / `python -m bench.compare` measure runs
  deterministically; bench runs write under `bench/results/` (gitignored).
- `scripts/engines_probe.py` checks a provider's request shapes;
  `scripts/webapp_dev_url.py` prints a signed local console URL (never the
  token).
- `scripts/run.sh` / `run_mac.sh` pre-flight now also stops this repo's
  cloudflared (see [Running day to day](#running-day-to-day)).

---

## History: the 2026-08-07 download fixes

Dated record of the fixes made right before the first Mac handoff (the
numbers are what was measured then):

| Symptom | Cause | Fix |
|---|---|---|
| Bot said "✅ Saved <path>" but no video ever arrived in Telegram | python-telegram-bot's default write/read timeouts were far too short for a large upload; both `send_video` and the `send_document` fallback timed out and only logged | Generous timeouts on `Application.builder()`; both-sends-failed is now reported to the user instead of swallowed |
| Long X/Twitter videos died partway, leaving `.part` files | X serves long video as HLS with hundreds of fragments, downloaded **serially**, and a blanket timeout killed it | Concurrent fragments (measured several times faster) + a *stall* timeout that measures silence, not total elapsed time |
| Completed videos on disk that the bot didn't know about | On timeout, `proc.kill()` killed yt-dlp but not its **ffmpeg child**, which kept writing after Argus reported failure | `_kill_tree()` (taskkill /T on Windows, `killpg` on macOS/Linux) + a finished file is now kept even if the exit code is non-zero |
| Vertical videos downloaded at reduced quality | The `auto` quality filter capped the height, which rejects a normal full-HD **portrait** video (most Reels/Shorts) | Cap the **short side** via yt-dlp's `res` sort (`min(width,height)`) |
| Unhelpful errors | Raw yt-dlp stderr ("No video could be found in this tweet") with no next step | `explain_error()` appends actionable hints (cookies, `/quality auto`, resume) |
| `/find` needed a sibling repo | `_run_yt_dlp` shelled out to the `intel-stack` venv's yt-dlp although argus declares yt-dlp itself | Uses the argus venv — **this is why `/find` works on the Mac at all** |

Maintenance tool from that round: `scripts/reconcile_vault.py` (see
[Step 4b](#step-4b--bring-the-vault-across-optional)).

---

## Prerequisites

| Thing | Why | Check |
|---|---|---|
| **Python 3.12** (not 3.13) | `pyproject.toml` pins the 3.12 series | `python3.12 -V` |
| **uv** | package manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **git** | repo | `git --version` |
| **ffmpeg** | merging video+audio | Comes free — `imageio-ffmpeg` ships a macOS wheel. Homebrew's is optional. |
| **An LLM engine** | `/research` and `/ask` only | the default profile needs the FreeLLMAPI server — ⚠️ see [Known gaps](#known-gaps-on-mac); or switch to a profile you have keys for |
| **cloudflared** | the console's tunnel only | optional — see [Known gaps](#known-gaps-on-mac) |
| Telegram bot token | the bot | in the old machine's `.env` |

Media downloads need **none** of the LLM stack. If no engine is ever set up,
`/fetch`, `/find` and link-pasting still work.

---

## Setup

### Step 1 — unpack

```bash
mkdir -p ~/Agents && cd ~/Agents
tar -xzf ~/Downloads/argus-macbook-handoff.tar.gz     # or git clone
cd argus
```

### Step 2 — create the venv and install

The venv **must** be called `venv` in the repo root — `scripts/run_mac.sh`
looks for `./venv/bin/python`.

```bash
uv venv venv --python 3.12
VIRTUAL_ENV=venv uv pip install -e '.[dev,asr]'
```

`asr` pulls `faster-whisper`, needed for local transcripts of X / Reddit /
Instagram video (YouTube uses captions first). Drop it if you don't want a
model download on first use, and switch `/engines transcriber` to an API.

### Step 3 — secrets

The archive deliberately contains **no secrets**. Copy the values from the
old machine's `.env` (they are not reconstructible from anything in the
archive):

```bash
cp .env.example .env
```

Then fill in, from the old `.env`:

- `TELEGRAM_BOT_TOKEN` — `<id>:<secret>`
- `TELEGRAM_ALLOWED_USER_ID` — your numeric Telegram id
- `FREELLMAPI_API_KEY` and `FREELLMAPI_BASE_URL` — for the default profile
- `EXA_API_KEY` — Exa search (optional; DDGS takes its queries without it)
- any paid-provider key a profile you use binds (`GROQ_API_KEY`, …)

The bot refuses to start only without a well-formed bot token. It answers
**only** `TELEGRAM_ALLOWED_USER_ID` (unset, it answers nobody).

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
├── transcripts/      ← speech-to-text + caption text
└── research history/ ← reports (Markdown + PDF + evidence.json)
```

A leading `~` **is** expanded (an unexpanded `~` used to create a literal `~`
folder inside the repo).

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

Only if you want the existing downloads and past reports. The library
database in the archive lists them with the **old machine's** paths, so after
copying you must remap:

```bash
# 1. copy the media/transcripts/reports from the old box (rsync/USB/etc.)
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

`argus_checkpoints.sqlite` does not travel meaningfully (in-flight LangGraph
state). Without it, a research run that was mid-flight on the old machine is
gone; the bot recreates the file on first start and tells you about orphaned
runs.

### Step 5 — verify

Run these in order. Do not report success before all four pass.

```bash
# 1. test suite
PYTHONPATH="" ./venv/bin/python -m pytest -q
```

Everything must pass except the live-proxy tests (`tests/test_llm.py`,
`tests/test_graph.py`) when the FreeLLMAPI server isn't running — they resolve
the shipped engines config against it and fail with an `EngineConfigError`
("cannot list models at …"). Any *other* failure is real and must be fixed
before continuing.

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
`Argus bot starting; long-polling…`, plus the console lines
(`webapp: listening on 127.0.0.1:<port>`, then the tunnel URL) unless the
console is disabled. In Telegram, paste any YouTube/X link → you get a
download menu → tap `⬇ 1` → the file lands in the vault and (if under
Telegram's bot upload cap) is sent back to you. With an engine available,
`scripts/verify_live.sh` then checks research end to end (it needs the same
valid `.env`, bot token included).

---

## Running day to day

```bash
cd ~/Agents/argus && ./scripts/run_mac.sh      # Windows: ./scripts/run.sh (Git Bash)
```

The pre-flight kills any existing `argus.bot` poller **and any orphaned
cloudflared of this bot**, asserts none is left, then execs the new bot. A
bot that was hard-killed leaves its cloudflared child behind (seen on
Windows: the old tunnel kept serving the console port), which is why. "Of
this bot" means either this repo's binary (`tools/bin/cloudflared(.exe)`,
whatever its arguments), or the `ARGUS_CLOUDFLARED` binary **only when it runs
the bot's own command line** (`tunnel --url http://127.0.0.1:<ARGUS_WEBAPP_PORT>
--no-autoupdate`). So a shared `ARGUS_CLOUDFLARED` (say, Homebrew's) running
your other tunnels is left alone; only a quick tunnel to the console's own
port on that binary would be taken for the bot's. Any other cloudflared is
never touched. A failed process query (WMI on Windows) stops the script
instead of reading as "none left". If the script refuses to start it tells
you what is still alive.

### One bot per token

Telegram allows exactly one long-poller per token. A second instance makes
Telegram answer one and return **409 Conflict** to the other — and the loser
may be the one you just fixed, so a change appears not to take effect. Before
starting on a new machine, confirm the old bot is stopped. Never run both.

### Useful commands

| Command | Does |
|---|---|
| paste a link | download menu (video / transcript / both) |
| `/fetch <url…>` | download directly |
| `/find <query>` | search YouTube, pick from results |
| `/quality auto\|min\|max\|<height>` | download quality (see below) |
| `/research <topic>` | full multi-agent report (needs an engine) |
| `/ask <question>` | quick answer, no search (needs an engine) |
| `/transcript`, `/transcripts` | captions / speech-to-text |
| `/runs`, `/status`, `/cancel`, `/continue`, `/delete` | manage runs and stored assets |
| `/engines` | engines profiles, roles, the transcriber |
| menu button **Argus** | the console |
| `/help full` | the detailed guide (`docs/help.md`) |

### About `/quality`

The setting is stored in the library DB and travels with it. `max` takes the
largest available stream, so long videos are saved to the vault but often
**cannot** be delivered in chat (Telegram's bot upload cap) — expected, not a
bug, and the bot says so explicitly. `/quality 720` or `/quality auto` keeps
most clips deliverable; `auto` caps the short side and is orientation-aware.

---

## Known gaps on Mac

Read this before concluding something is broken.

### 1. FreeLLMAPI server — the default profile needs it

The default engines profile points at `FREELLMAPI_BASE_URL`, a **local**
server that is not part of this repo and is **not** in the archive. How it is
installed on the Windows box is not recorded anywhere in this project.

- **Impact:** research and `/ask` on the default profile. Media download and
  `/find` are unaffected; the bot starts and DMs the engines problem.
- **Action:** repeat the FreeLLMAPI install, point `FREELLMAPI_BASE_URL` at
  any OpenAI-compatible endpoint, or switch profile (`/engines use
  <profile>`) to one whose provider keys you have. `scripts/engines_probe.py
  --provider <name>` checks a provider before you rely on it.
- **UNVERIFIED ON MAC.**

### 2. `intel-stack` — optional

Research fetches pages with its own native fetcher. The intel-stack sibling
repo (`snatch.py`, `crawl.py`, `article_convert.py`, in their own venv) only
adds extra fetch rungs for pages the native one cannot read. To use it, port
it and set both in `.env`:

```bash
INTEL_STACK_DIR=~/Agents/intel-stack/scripts
INTEL_PYTHON=~/Agents/intel-stack/venv/bin/python
```

### 3. cloudflared — the console's tunnel

`tools/bin/` is gitignored, so the binary does not come with the repo. Put a
macOS `cloudflared` at `tools/bin/cloudflared` (or set `ARGUS_CLOUDFLARED`;
the pre-flight then stops that binary only when it runs the bot's own
command line, see [Running day to day](#running-day-to-day)).
Without it the console reports "cloudflared not found" and the chat works as
before; `ARGUS_WEBAPP_TUNNEL=none` keeps the console local only, and
`ARGUS_WEBAPP_ENABLED=false` turns it off. **UNVERIFIED ON MAC.**

### 4. Login-walled content

Private X videos, some Instagram posts and age-restricted YouTube need cookies.
Set **one** of these in `.env`:

```bash
ARGUS_YTDLP_COOKIES_BROWSER=chrome     # or firefox / edge / safari / brave
ARGUS_YTDLP_COOKIES=~/argus-cookies.txt
```

macOS specifics: Safari is supported by yt-dlp. Chrome **locks its cookie
database while running** — quit it first. Reading Safari's cookies may prompt
for Full Disk Access for your terminal. The bot tells you when an error looks
cookie-shaped.

### 5. Whisper on Apple Silicon

`faster-whisper` runs on the CPU here; it does not use the GPU, so long audio
is slow. First run downloads the model. The model, device and compute type
come from `transcriber.faster_whisper` in `argus_engines.json` (override them
in the gitignored `argus_engines.local.json`); `/engines transcriber` picks
the engine (an audio API such as Groq's Whisper is the fast alternative).
`ARGUS_WHISPER_MODEL` is no longer read; startup warns if it is set.
**UNVERIFIED ON MAC.**

---

## Troubleshooting

| Symptom | Do this |
|---|---|
| `409 Conflict` in the log | Another poller has the token. Stop the other machine's bot, and any second terminal. |
| Console still answers after the bot died | An orphaned cloudflared from a hard-killed bot. `run.sh` / `run_mac.sh` stop it on the next start. |
| "⚠️ Console down — …" DM | The reason follows the dash, and `/status` repeats it; the chat is unaffected. `logs/cloudflared.log` has the tunnel's output. |
| `/research` answers with an engines error | The active profile cannot run (no key, no model, proxy down). `/engines` shows the problem; fix it or `/engines use` another profile. |
| Bot starts, downloads nothing, `ffmpeg: None` in log | `VIRTUAL_ENV=venv uv pip install -U imageio-ffmpeg`, or set `ARGUS_FFMPEG=/opt/homebrew/bin/ffmpeg` |
| "No video could be found in this tweet" | Post is deleted/protected, or needs login → set `ARGUS_YTDLP_COOKIES_BROWSER`. The error message says this. |
| A platform suddenly stops working | yt-dlp lags platform changes. `VIRTUAL_ENV=venv uv pip install -U --pre yt-dlp` (`--pre` gets nightlies, which is where extractor fixes land first). |
| Download "stalled — no progress for …" | Connection dropped. Re-send the same link: yt-dlp **resumes** from the `.part` file. |
| Big file saved but not sent to Telegram | Over Telegram's bot upload cap. Expected. Use `/quality 720`, or open it from the vault. |
| Vault and library disagree | `./venv/bin/python scripts/reconcile_vault.py` (add `--register` to adopt orphans) |
| `.part` files piling up | Same tool lists them with age. `--prune-partials --older-than <days>` deletes them — this destroys resume capability, so only for downloads you have abandoned. |
| `pydantic_core` import explosion | A foreign `PYTHONPATH` leaked in. `run_mac.sh` clears it; if running python directly, prefix `PYTHONPATH=""`. |

### Tuning knobs (all optional, all in `.env`)

The media knobs (`ARGUS_MEDIA_FRAGMENTS`, `ARGUS_MEDIA_CONCURRENCY`,
`ARGUS_MEDIA_STALL_TIMEOUT`, `ARGUS_MEDIA_TIMEOUT`), the Telegram client
timeouts (`ARGUS_TG_*_TIMEOUT`; raise `ARGUS_TG_MEDIA_WRITE_TIMEOUT` on a slow
uplink), the console and run-driver knobs: see `.env.example`, which says what
each does. Research depth is not an env knob any more: it is the depth preset
on the plan card (or the console's overrides).

---

## Git state

Check `git status` / `git log` on the machine itself. Commit only a state
verified on the machine it runs on (Step 5).

## What is NOT in the archive, and why

| Excluded | Why |
|---|---|
| `.env` | live secrets; copy the values by hand (Step 3) |
| `venv/` | platform binaries; rebuilt in Step 2 |
| `tools/bin/` | platform binaries (cloudflared); see Known gaps |
| `argus_checkpoints.sqlite` | in-flight run state, not portable |
| `media/`, `transcripts/`, `research history/` | the vault — copy separately if you want it (Step 4b) |
| `__pycache__/`, `.pytest_cache/`, `demo_output/`, `logs/`, `bench/results/` | build/run artifacts |

`argus_library.sqlite` **is** included — it is your history and your
`/quality` setting. Checkpoint its WAL and check its integrity before packing.
