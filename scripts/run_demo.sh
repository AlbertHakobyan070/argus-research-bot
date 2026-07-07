#!/usr/bin/env bash
# Argus demo runner — same PYTHONPATH reset.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=""
exec ./venv/Scripts/python.exe scripts/demo_run.py
