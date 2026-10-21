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
  # -ErrorAction Stop: a failed WMI query must read as "no answer" (FATAL
  # below), never as a count of 0.
  REMAINING=$(powershell.exe -NoProfile -Command "
    (Get-CimInstance Win32_Process -ErrorAction Stop |
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
    echo "src/argus/bot/, or an LLM agent that re-execs the bot on crash." >&2
    echo "Stop the offending tool, then re-run this script." >&2
    exit 1
  fi

  # Pre-flight v3 (2026-10-21): a hard-killed bot leaves its cloudflared
  # child orphaned on Windows, and the old tunnel keeps serving the console
  # port. Stop every cloudflared that is
  #   (a) this repo's binary (tools/bin/cloudflared.exe), whatever its args, or
  #   (b) the ARGUS_CLOUDFLARED binary the bot would spawn, ONLY when its
  #       command line ends with the bot's own argv (webapp/tunnel.py
  #       command(): tunnel --url http://127.0.0.1:<ARGUS_WEBAPP_PORT> --no-autoupdate),
  # then assert none is left. Paths and argv come from the bot's own code
  # (.env included). Any other cloudflared — another binary, or the same
  # shared binary running someone else's tunnel — is never touched.
  # No pipe here: set -e must see a Python failure, never an empty target.
  CF_TARGETS=$(./venv/Scripts/python.exe -c "import os; from argus.config import _PROJECT_ROOT as r, Settings; from argus.webapp import tunnel; port = int(os.environ.get('ARGUS_WEBAPP_PORT') or Settings.webapp_port); print((r / 'tools' / 'bin' / 'cloudflared.exe').resolve()); print(tunnel.binary().resolve()); print(' '.join(tunnel.command(tunnel.binary(), port)[1:]))")
  CF_TARGETS=${CF_TARGETS//$'\r'/}
  { IFS= read -r ARGUS_PREFLIGHT_CF_REPO; IFS= read -r ARGUS_PREFLIGHT_CF_BIN; IFS= read -r ARGUS_PREFLIGHT_CF_ARGS; } <<< "$CF_TARGETS"
  if [ -z "$ARGUS_PREFLIGHT_CF_REPO" ] || [ -z "$ARGUS_PREFLIGHT_CF_BIN" ] || [ -z "$ARGUS_PREFLIGHT_CF_ARGS" ]; then
    echo "FATAL: pre-flight could not resolve this repo's cloudflared targets." >&2
    exit 1
  fi
  export ARGUS_PREFLIGHT_CF_REPO ARGUS_PREFLIGHT_CF_BIN ARGUS_PREFLIGHT_CF_ARGS
  # PowerShell, single-quoted so bash leaves it alone; [char]32 is a space.
  # -ErrorAction Stop: a failed WMI query fails loudly, never a count of 0.
  CF_FILTER='$repo = [System.IO.Path]::GetFullPath($env:ARGUS_PREFLIGHT_CF_REPO); $bin = [System.IO.Path]::GetFullPath($env:ARGUS_PREFLIGHT_CF_BIN); $tail = [string][char]32 + $env:ARGUS_PREFLIGHT_CF_ARGS; $cf = @(Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { $_.ExecutablePath -and ((([System.IO.Path]::GetFullPath($_.ExecutablePath)) -ieq $repo) -or ((([System.IO.Path]::GetFullPath($_.ExecutablePath)) -ieq $bin) -and $_.CommandLine -and $_.CommandLine.TrimEnd().EndsWith($tail))) })'
  echo "[run.sh] pre-flight: killing any orphaned cloudflared of this repo…"
  powershell.exe -NoProfile -Command "$CF_FILTER; \$cf | ForEach-Object {
      Write-Output \"  killing cloudflared PID \$(\$_.ProcessId) (\$(\$_.ExecutablePath))\"
      Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue
    }" || { echo "FATAL: pre-flight could not query cloudflared processes (see the error above)." >&2; exit 1; }
  sleep 1
  CF_REMAINING=$(powershell.exe -NoProfile -Command "$CF_FILTER; \$cf.Count" | tr -d '\r\n[:space:]')
  if [ -z "$CF_REMAINING" ]; then
    echo "" >&2
    echo "FATAL: pre-flight assertion could not query cloudflared processes." >&2
    echo "Kill this repo's cloudflared manually (Task Manager → cloudflared.exe," >&2
    echo "path $ARGUS_PREFLIGHT_CF_REPO), then re-run this script." >&2
    exit 1
  fi
  if [ "$CF_REMAINING" != "0" ]; then
    echo "" >&2
    echo "FATAL: $CF_REMAINING cloudflared process(es) of this repo survived the pre-flight kill." >&2
    echo "Diagnose with:" >&2
    echo "  powershell.exe -NoProfile -Command \"Get-CimInstance Win32_Process -Filter \\\"Name='cloudflared.exe'\\\" | Select-Object ProcessId, ExecutablePath, CommandLine\"" >&2
    exit 1
  fi
  echo "[run.sh] pre-flight: clear."
fi

exec ./venv/Scripts/python.exe -m argus.bot