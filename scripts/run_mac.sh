#!/usr/bin/env bash
# Argus runner — macOS / Linux.
#
# The Windows sibling (run.sh) exists because a second poller on the same
# Telegram token makes both instances fight over getUpdates: Telegram
# answers one and 409s the other, so a freshly-fixed bot can lose the
# long-poll to a stale one still serving old code from memory. Same
# hazard here, so the same pre-flight: kill any existing argus.bot, then
# ASSERT it is gone before exec'ing the new one.
set -euo pipefail
cd "$(dirname "$0")/.."

# The parent venv's site-packages must not leak in (pydantic_core ABI).
export PYTHONPATH=""

PYBIN="./venv/bin/python"
if [ ! -x "$PYBIN" ]; then
  echo "FATAL: $PYBIN not found. Create the venv first:" >&2
  echo "  uv venv venv --python 3.12" >&2
  echo "  VIRTUAL_ENV=venv uv pip install -e '.[dev,asr]'" >&2
  exit 1
fi

echo "[run_mac.sh] pre-flight: killing any prior argus.bot pollers…"
# -f matches the full command line; exclude this script's own pgrep.
PIDS="$(pgrep -f '[p]ython.*argus\.bot' || true)"
if [ -n "$PIDS" ]; then
  echo "$PIDS" | while read -r pid; do
    echo "  killing PID $pid"
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
  # Anything that ignored SIGTERM gets SIGKILL.
  echo "$PIDS" | while read -r pid; do
    kill -KILL "$pid" 2>/dev/null || true
  done
  sleep 1
fi

REMAINING="$(pgrep -f '[p]ython.*argus\.bot' | wc -l | tr -d ' ')"
if [ "$REMAINING" != "0" ]; then
  echo "" >&2
  echo "FATAL: $REMAINING argus.bot process(es) survived the pre-flight kill." >&2
  echo "Something is respawning the bot (an IDE run config, a launchd job," >&2
  echo "or a terminal tab). Find it with:" >&2
  echo "  ps -ax -o pid,command | grep argus" >&2
  echo "Stop it, then re-run this script." >&2
  exit 1
fi

# A hard-killed bot can leave its cloudflared child running (seen on
# Windows: the old tunnel kept serving the console port). Stop every
# cloudflared that is
#   (a) this repo's binary (argv[0] = tools/bin/cloudflared), whatever its args, or
#   (b) the ARGUS_CLOUDFLARED binary exactly as the bot spawns it, ONLY with
#       the bot's own argv (webapp/tunnel.py command():
#       tunnel --url http://127.0.0.1:<ARGUS_WEBAPP_PORT> --no-autoupdate),
# then assert none is left. Paths and argv come from the bot's own code
# (.env included). Any other cloudflared — another path, or the same shared
# binary running someone else's tunnel — is never touched.
CF_TARGETS="$("$PYBIN" -c "import os; from argus.config import _PROJECT_ROOT as r, Settings; from argus.webapp import tunnel; port = int(os.environ.get('ARGUS_WEBAPP_PORT') or Settings.webapp_port); print(r / 'tools' / 'bin' / 'cloudflared'); print(tunnel.binary()); print(' '.join(tunnel.command(tunnel.binary(), port)[1:]))")"
{ IFS= read -r CF_REPO; IFS= read -r CF_BIN; IFS= read -r CF_ARGS; } <<< "$CF_TARGETS"
if [ -z "$CF_REPO" ] || [ -z "$CF_BIN" ] || [ -z "$CF_ARGS" ]; then
  echo "FATAL: pre-flight could not resolve this repo's cloudflared targets." >&2
  exit 1
fi
cf_ere() { printf '%s' "$1" | sed 's/[][\.*^$()+?{}|]/\\&/g'; }   # regex metacharacters escaped
cf_pids() {
  # anchored EREs over the full command line (pgrep -f)
  { pgrep -f "^$(cf_ere "$CF_REPO")( |\$)" || true
    pgrep -f "^$(cf_ere "$CF_BIN $CF_ARGS")\$" || true
  } | sort -u
}
echo "[run_mac.sh] pre-flight: killing any orphaned cloudflared of this repo…"
CF_PIDS="$(cf_pids)"
if [ -n "$CF_PIDS" ]; then
  echo "$CF_PIDS" | while read -r pid; do
    echo "  killing cloudflared PID $pid"
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
  echo "$CF_PIDS" | while read -r pid; do
    kill -KILL "$pid" 2>/dev/null || true
  done
  sleep 1
fi
CF_REMAINING="$(cf_pids | grep -c . || true)"
if [ "$CF_REMAINING" != "0" ]; then
  echo "" >&2
  echo "FATAL: $CF_REMAINING cloudflared process(es) of this repo survived the pre-flight kill." >&2
  echo "Find them with:" >&2
  echo "  ps -ax -o pid,command | grep cloudflared" >&2
  exit 1
fi
echo "[run_mac.sh] pre-flight: clear."

exec "$PYBIN" -m argus.bot
