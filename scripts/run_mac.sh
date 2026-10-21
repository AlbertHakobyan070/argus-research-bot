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
# cloudflared whose argv[0] is this repo's binary (tools/bin/cloudflared) or
# the ARGUS_CLOUDFLARED path exactly as the bot spawns it (resolved by the
# bot's own code, .env included), then assert none is left. A cloudflared
# started from any other path is never touched.
CF_PATHS="$("$PYBIN" -c "from argus.config import _PROJECT_ROOT as r; from argus.webapp import tunnel; print(r / 'tools' / 'bin' / 'cloudflared'); print(tunnel.binary())")"
cf_pids() {
  # each path as an anchored ERE (regex metacharacters escaped), then pgrep -f
  printf '%s\n' "$CF_PATHS" | sort -u | while IFS= read -r p; do
    [ -n "$p" ] || continue
    pgrep -f "^$(printf '%s' "$p" | sed 's/[][\.*^$()+?{}|]/\\&/g')( |\$)" || true
  done | sort -u
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
