"""Argus Deep Research Bench.

A eval harness for measuring whether Argus graph improvements actually
move the needle on report quality. Pipeline:

    queries.jsonl  ──►  runner.py  ──►  results/raw.jsonl
                                              │
                                              ▼
                                        scorer.py
                                              │
                                              ▼
                                  results/score.json + score.md
"""