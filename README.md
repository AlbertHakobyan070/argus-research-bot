# Argus

A single-user Telegram bot for research you can check, and for saving media.

Ask it a question and it plans the research with you, searches live (Exa,
DDGS, arXiv, GitHub), reads every source into notes with verbatim quotes,
writes a cited report, has a separate judge check each claim against the
quotes it cites, and **cuts** what fails. You get Markdown, a PDF and an
`evidence.json`.

**Documentation: <https://alberthakobyan070.github.io/argus-research-bot/>**
([user guide](https://alberthakobyan070.github.io/argus-research-bot/help/) ·
[how it works](https://alberthakobyan070.github.io/argus-research-bot/how-it-works/))

## What it does

- **Two checkpoints with you.** A plan gate over the real sources before
  anything is fetched — only the links you put in the question are read
  first, so the plan is about them (approve, edit, set length, depth, engines
  and recency), and a report gate before delivery (send, extend, revise, cancel).
- **Trust first.** Links come only from live search or from you, quotes are checked word
  for word, unsupported claims leave the body for an appendix, and anything
  unchecked is announced at the top of the report.
- **Swappable engines.** `argus_engines.json` binds every pipeline role to a
  provider and model: the free FreeLLMAPI proxy (default), paid APIs
  (OpenAI-compatible, Anthropic, Gemini, Groq, OpenRouter) or local servers
  (Ollama, LM Studio). Switch with `/engines`; each report records which model
  actually served each call.
- **A console in Telegram.** The Argus menu button opens a Mini App, served
  from your machine through a cloudflared quick tunnel: runs, live progress,
  plan controls, a report reader with tap-to-verify citations, and engines
  settings.
- **Media.** YouTube, X, Reddit and Instagram downloads into your vault, with
  transcription (local faster-whisper or an OpenAI-compatible audio API).

## Quick start

```bash
# 1) secrets
cp .env.example .env
# Edit .env: TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USER_ID (your numeric
#   Telegram id), and the key of the engines profile you use
#   (FREELLMAPI_API_KEY for the default profile). EXA_API_KEY is optional.

# 2) install (from the repo root)
uv venv --python 3.12 venv
VIRTUAL_ENV=venv uv pip install -e ".[dev]"     # add ",asr" for local faster-whisper

# 3) start the bot (Windows: Git Bash; macOS/Linux: scripts/run_mac.sh)
./scripts/run.sh
```

Then in Telegram: `/research <topic>`, or `/ask <question>` for a quick
answer without search. `/help` lists every command.

For the console, put the cloudflared binary in `tools/bin/` (gitignored) or
set `ARGUS_CLOUDFLARED`; without it the chat keeps working and the console
reports "cloudflared not found".

> **PYTHONPATH:** the argus venv must run with an empty `PYTHONPATH` (a parent
> venv's `site-packages` breaks `pydantic_core`). The `scripts/run*.sh`
> scripts clear it; if you bypass them, `unset PYTHONPATH` first.

## API access for other programs

A script, a cron job or another bot can start and follow runs through the
console's API with a service token instead of Telegram's `initData`:
`Authorization: Bearer <token>`. It is off until you set the token.

| Variable | Default | Meaning |
|---|---|---|
| `ARGUS_SERVICE_TOKEN` | unset (off) | The token; unset or empty refuses every `Bearer`. |
| `ARGUS_SERVICE_HOSTS` | `host.docker.internal:8787` | Comma list of `Host` headers a `Bearer` is accepted with, so it never works through the public tunnel. |
| `ARGUS_SERVICE_ORIGIN` | `api` | The `origin` label of the runs the token starts (`[a-z0-9_-]{1,32}`, not one of Argus's own origins). |

The token may call only `POST /api/runs`, `GET /api/runs/{id}` (and its
`/events`, `/evidence`, `/report`), `POST /api/runs/{id}/plan` and
`POST /api/runs/{id}/cancel`, and only on the runs it started: any other run
answers 404, any other route 401 (a `Bearer` on a Host outside
`ARGUS_SERVICE_HOSTS` is answered as if the token were off). It cannot Send a
report to the chat. At start it may set `length`, `depth` and `recency` (a
search filter, not a cost knob), but not `overrides` or an engines profile; on
approve it may set none of `length`, `depth` or an engines profile. Its actions
are recorded with the actor `service`, never as the console's.

`POST /api/runs` takes these start options beside `topic` (from the console
too):

- `notify_chat` (default `true`): `false` posts nothing to your chat for the
  run (no progress card, plan card, report preview or restart notice); your
  own `/continue` of it still shows its card.
- `external_ref`: an idempotency key, 1-128 characters of `A-Z a-z 0-9 . _ : -`.
  A key already used by a run of the same origin answers `200` with
  `"outcome": "existing"` and that run's id, live or finished.
- `auto_approve_plan`, `auto_send_report` (default `false`): pass that gate
  unattended, so a run can go end to end on its own (Send posts the report
  files to your chat only when `notify_chat` is on; the files are under the
  run's report folder either way). A run resting at such a gate after a
  restart gets that step again at startup.

`/runs` and the console's run list hide the token's runs; `/runs all` and
`GET /api/runs?origin=all` show them.

## More

- Running, the console and tunnel, migration and troubleshooting: `handoff.md`.
- The v4 design and what each work package built: `docs/argus-v4-design.md`,
  `docs/v4-notes/`.
- Tests: `./scripts/run_tests.sh -q`. Green tests are not a working bot;
  `scripts/verify_live.sh` drives the real graph end to end.
- Secrets live in `.env` only (gitignored); `argus_engines.json` holds env var
  names, never values.

### Building the docs locally

```bash
VIRTUAL_ENV=venv uv pip install "mkdocs-material>=9.5,<10" "mkdocs>=1.6,<2"
./venv/Scripts/mkdocs serve      # macOS/Linux: ./venv/bin/mkdocs serve
```

The site publishes itself to GitHub Pages on every push to `main` that touches
`docs/` or `mkdocs.yml` (`.github/workflows/docs.yml`).
