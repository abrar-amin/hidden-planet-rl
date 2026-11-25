#!/usr/bin/env bash
# Recompute the simulator-level baselines for the hard level (val 0-59, test 0-149). Resumes if files exist.
cd "$(dirname "$0")/.."
.venv/bin/python scripts/run_baselines.py val 0 60 results/hard_val.jsonl --level 5 --procs 3
.venv/bin/python scripts/run_baselines.py test 0 150 results/hard_test.jsonl --level 5 --procs 3
echo BASELINES_DONE
