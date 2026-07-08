#!/usr/bin/env bash
# Argus runner — clears PYTHONPATH so the parent Hermes venv doesn't
# poison the argus venv (pydantic_core ABI mismatch).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=""

# Pre-flight: stop any bot already polling this token. Telegram allows only
# one getUpdates poller per token; a leftover instance causes an endless
# 409 Conflict storm and steals your updates. This kills prior argus.bot
# processes (never this script's own children — we start after).
if command -v powershell.exe >/dev/null 2>&1; then
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*argus.bot*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1 || true
  sleep 1  # give Telegram a moment to release the old long-poll
fi

exec ./venv/Scripts/python.exe -m argus.bot
