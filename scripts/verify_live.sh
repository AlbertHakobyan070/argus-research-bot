#!/usr/bin/env bash
# Live verification for Argus v4.
# Run this once the FreeLLMAPI proxy (FREELLMAPI_BASE_URL, the default engines
# profile's provider) is up. It drives the REAL graph end-to-end (live LLM +
# live search) to prove the research path — the grounded plan gate, research
# and digest, compose, analyst, the judged panel and the report — actually
# works, not just the hermetic suite.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=""
export PYTHONIOENCODING=utf-8

# The proxy URL the bot uses (.env, or the code default), not a hardcoded port.
BASE=$(./venv/Scripts/python.exe -c "from argus.config import get_settings; print(get_settings().freellmapi_base_url)")
BASE=${BASE//$'\r'/}
echo "== 1. proxy reachable? ($BASE) =="
curl -s -m 5 "$BASE/models" -o /dev/null -w "  HTTP %{http_code}\n" \
  || { echo "  proxy DOWN — start it first"; exit 1; }

echo "== 2. live LLM tier + graph tests (need the proxy) =="
./venv/Scripts/python.exe -m pytest tests/test_llm.py tests/test_graph.py \
  tests/test_reflexion.py -q

echo "== 3. graph-direct live check: q010 medium, acceptance table (scripts/live_check.py) =="
./venv/Scripts/python.exe scripts/live_check.py --query q010 --length medium

echo
echo "All live checks passed. Now drive it from Telegram:"
echo "  /research langgraph vs langchain   -> tap Edit, reply 'focus on"
echo "     streaming', confirm the plan REDRAFTS with your angle"
echo "  Approve -> at the preview tap Revise, reply 'add a comparison"
echo "     table', confirm the report is re-synthesized"
