#!/usr/bin/env bash
# Post-proxy live verification for the Argus v2 rebuild.
# Run this once the FreeLLMAPI proxy (127.0.0.1:3001) is back up. It drives
# the REAL graph end-to-end (live LLM + live search) to prove the research
# path — including the grounded plan gate, the revise loop, and parallel
# fetch — actually works, not just the hermetic suite.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=""
export PYTHONIOENCODING=utf-8

echo "== 1. proxy reachable? =="
curl -s -m 5 http://127.0.0.1:3001/v1/models -o /dev/null -w "  HTTP %{http_code}\n" \
  || { echo "  proxy DOWN — start it first"; exit 1; }

echo "== 2. live LLM tier + graph tests (need the proxy) =="
./venv/Scripts/python.exe -m pytest tests/test_llm.py tests/test_graph.py \
  tests/test_reflexion.py -q

echo "== 3. graph-direct deep run to report (manual_e2e) =="
./venv/Scripts/python.exe tests/manual_e2e.py

echo
echo "All live checks passed. Now drive it from Telegram:"
echo "  /research langgraph vs langchain   -> tap Edit, reply 'focus on"
echo "     streaming', confirm the plan REDRAFTS with your angle"
echo "  Approve -> at the preview tap Revise, reply 'add a comparison"
echo "     table', confirm the report is re-synthesized"
