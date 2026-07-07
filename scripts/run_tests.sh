#!/usr/bin/env bash
# Argus test runner — same PYTHONPATH reset.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=""
exec ./venv/Scripts/python.exe -m pytest tests/ "$@"
