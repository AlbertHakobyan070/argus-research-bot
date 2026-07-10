#!/usr/bin/env bash
# Argus runner — clears PYTHONPATH so the parent Hermes venv doesn't
# poison the argus venv (pydantic_core ABI mismatch).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=""

# Pre-flight v2: kill ALL argus-related python processes and ASSERT
# nothing is left before we exec the new bot.
#
# Why v2 (2026-07-10): the original filter only matched `python.exe`
# whose CommandLine contained the literal substring `argus.bot`. An
# IDE-launched bot on a different Python interpreter (e.g. the
# uv-managed `cpython-3.12-windows-x86_64-none\python.exe`) can slip
# through if its launch command doesn't include the substring, or if
# the IDE auto-respawns the bot microseconds after `Stop-Process`.
# When that happens, the new bot from this script loses the Telegram
# long-poll and falls into a 409 Conflict loop, while the IDE-launched
# bot keeps serving callbacks with whatever code was loaded into
# memory at startup — so a freshly-pushed fix never reaches the user.
#
# The hardening below: (1) matches any python interpreter, not just
# `python.exe`; (2) waits and re-asserts; (3) fails the script loudly
# if anything is still alive so the user can intervene (close the IDE,
# kill the rogue process) before the new bot launches.
if command -v powershell.exe >/dev/null 2>&1; then
  echo "[run.sh] pre-flight: killing any prior argus.bot pollers…"
  powershell.exe -NoProfile -Command "
    Get-CimInstance Win32_Process |
      Where-Object { \$_.Name -match '^python[wx]?\.exe$' -and \$_.CommandLine -like '*argus.bot*' } |
      ForEach-Object {
        Write-Output \"  killing PID \$(\$_.ProcessId) (\$(\$_.CommandLine))\"
        Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue
      }
  "
  sleep 2

  # Assertion — fail loudly if anything is still alive.
  REMAINING=$(powershell.exe -NoProfile -Command "
    (Get-CimInstance Win32_Process |
      Where-Object { \$_.Name -match '^python[wx]?\.exe$' -and \$_.CommandLine -like '*argus.bot*' } |
      Measure-Object).Count
  " | tr -d '\r\n[:space:]')
  if [ -z "$REMAINING" ]; then
    # PowerShell returned nothing — the cmdlet failed or wasn't found.
    # Treat as fatal so we don't silently launch a bot into a
    # contested long-poll.
    echo "" >&2
    echo "FATAL: pre-flight assertion could not query argus.bot processes." >&2
    echo "PowerShell / WMI may be unavailable in this shell. Re-run from" >&2
    echo "Git Bash with admin privileges, or kill any rogue processes manually:" >&2
    echo "  taskkill /F /IM python.exe /FI \"WINDOWTITLE eq argus*\"" >&2
    exit 1
  fi
  if [ "$REMAINING" != "0" ]; then
    echo "" >&2
    echo "FATAL: $REMAINING argus.bot process(es) survived the pre-flight kill." >&2
    echo "An IDE or background runner is respawning the bot. Diagnose with:" >&2
    echo "  powershell.exe -NoProfile -Command \"Get-CimInstance Win32_Process | Where-Object { \\\$_.CommandLine -like '*argus*' } | Select-Object ProcessId, Name, CommandLine\"" >&2
    echo "" >&2
    echo "Likely causes: a Cursor / VS Code / PyCharm 'Run File' config for" >&2
    echo "src/argus/bot.py, or an LLM agent that re-execs the bot on crash." >&2
    echo "Stop the offending tool, then re-run this script." >&2
    exit 1
  fi
  echo "[run.sh] pre-flight: clear."
fi

exec ./venv/Scripts/python.exe -m argus.bot