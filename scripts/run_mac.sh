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
echo "[run_mac.sh] pre-flight: clear."

exec "$PYBIN" -m argus.bot
