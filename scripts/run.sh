#!/usr/bin/env bash
# Argus runner — clears PYTHONPATH so the parent Hermes venv doesn't
# poison the argus venv (pydantic_core ABI mismatch).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=""
exec ./venv/Scripts/python.exe -m argus.bot
