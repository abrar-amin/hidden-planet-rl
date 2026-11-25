#!/usr/bin/env bash
# Free, local: demonstrator data and scripted references for the event (strict0, cost 0.008, hard level).
# Dev set = test seeds 3000-3191. Selection set = test seeds 4000-4047 (no references needed). The sealed set is never touched here.
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export OMP_NUM_THREADS=1
RW="--reward-mode strict0 --wrong-penalty 1.0 --cost-per-unit 0.008"
$PY scripts/dry_run.py --agent fixed --level hard --split train --n 700 $RW --diverse --dump data/expert_fixed_hard_strict0.jsonl --json results/text_fixed_hard_train_strict0.json > /dev/null 2>&1
$PY scripts/run_baselines.py test 3000 192 results/hard_dev.jsonl --level 5 --procs 3
[ -f results/dev192/refs_dev_summary.json ] || $PY scripts/eval_refs.py --start 3000 --n 192 --agents teacher eig compact compact_smart fixed lite_teacher random --out-prefix results/dev192/refs_dev
$PY scripts/make_dev_reference.py
echo EVENT_REFS_DONE
