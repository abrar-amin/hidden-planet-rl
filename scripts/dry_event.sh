#!/usr/bin/env bash
# Free plumbing check of the whole pre-registered plan against the fake service, at tiny sizes. Writes only under runs/event_dry_*.
# The mock policy is random-ish, so its gate may FAIL: the first RL call is made WITHOUT FORCE to show the refusal, then with FORCE=1.
cd "$(dirname "$0")/.."
export DRY=1 QUICK=1 EVENT_DIR=runs/event_dry_$(date +%H%M%S)
for step in "sft fixed" "sft fit" "sft compact" "sft lite"; do echo "=== event.sh $step ==="; bash scripts/event.sh $step > /dev/null 2>&1 || echo "STEP FAILED: $step"; done
echo "gates: $(for d in fixed fit compact lite; do echo -n "$d=$(cat $EVENT_DIR/gate_sft_$d.txt) "; done)"
echo "=== v3.1: event.sh final before any arm has finished (must refuse) ==="; bash scripts/event.sh final > /dev/null 2>&1; echo "exit code $? (2 = refused, as required)"
echo "=== event.sh rl fixed 0 (no FORCE) ==="; bash scripts/event.sh rl fixed 0 > /dev/null 2>&1; echo "exit code $? (2 = refused because the gate failed; 0 = gate had passed and RL ran)"
for step in "rl fixed 0" "rl fixed 1" "control fixed" "rl fit 0" "rl compact 0" "rl lite 0"; do echo "=== FORCE=1 event.sh $step ==="; FORCE=1 bash scripts/event.sh $step > /dev/null 2>&1 || echo "STEP FAILED: $step"; done
echo "=== v3 checks: training reward, shape and futility rule ==="
grep -h "train-reward expected" $EVENT_DIR/rl_fixed_s0/run.log > /dev/null && echo "rl_fixed_s0 trained on the expected score: yes" || echo "ERROR: rl_fixed_s0 not trained on the expected score"
for D in $EVENT_DIR/rl_fixed_s0 $EVENT_DIR/rl_fixed_s1; do echo "$(basename $D) futility check: $(cat $D/rl/futility_check.txt 2>/dev/null || echo MISSING)"; done
if [ ! -f $EVENT_DIR/rl_fixed_s0/rl/futility_stop.txt ] && [ ! -f $EVENT_DIR/rl_fixed_s1/rl/futility_stop.txt ]; then
  echo "DRY-RUN SIMULATED futility stop (to exercise the final/verdict path)" > $EVENT_DIR/rl_fixed_s1/rl/futility_stop.txt; echo "no arm stopped by itself: simulated a futility stop for rl_fixed_s1"; fi
echo "=== event.sh final ==="; bash scripts/event.sh final > /dev/null 2>&1 || echo "STEP FAILED: final"
echo "=== v3.1 fault injection: the verdict given only one of the two pre-registered arms (must be NOT MET, no sealed arm) ==="
FI=${EVENT_DIR}_fi; rm -rf $FI; cp -R $EVENT_DIR $FI; rm -f $FI/rl_fixed_s*/rl/futility_stop.txt
for c in rl_fixed_s0 rl_fixed_s1; do ls $FI/final_dev_sampled_${c}_episodes.jsonl > /dev/null 2>&1 || { cp $FI/final_dev_sampled_sft_fixed_episodes.jsonl $FI/final_dev_sampled_${c}_episodes.jsonl; cp $FI/final_dev_sampled_sft_fixed_traces.jsonl $FI/final_dev_sampled_${c}_traces.jsonl; }; done
.venv/bin/python scripts/final_verdict.py $FI --set dev --dec sampled --init sft_fixed --demo fixed_diverse --control control_fixed --arms rl_fixed_s0 --expect-arms 2 --boot 2000 2>&1 | grep -E "PRE-REGISTERED RESULT|arm for the sealed|FAIL  1 arm"
.venv/bin/python scripts/final_verdict.py $FI --set dev --dec sampled --init sft_fixed --demo fixed_diverse --control control_fixed --arms rl_fixed_s0 rl_fixed_s1x --expect-arms 2 --boot 2000 2>&1 | grep -E "PRE-REGISTERED RESULT|no eval files"
rm -rf $FI
echo "=== v3.1: event.sh rl on a futility-stopped arm (must refuse) ==="; FS=""; for c in rl_fixed_s0 rl_fixed_s1; do [ -f $EVENT_DIR/$c/rl/futility_stop.txt ] && FS=$c; done
[ -n "$FS" ] && { FORCE=1 bash scripts/event.sh rl fixed ${FS##*_s} > /dev/null 2>&1; echo "exit code $? for $FS (2 = refused, as required)"; }
SA=""; for c in rl_fixed_s0 rl_fixed_s1; do [ -f $EVENT_DIR/$c/rl/futility_stop.txt ] || { SA=$c; break; }; done
if [ -n "$SA" ]; then echo "=== event.sh sealed (dry: seeds 9000+, arm forced to $SA) ==="; SEALED_ARM=$SA bash scripts/event.sh sealed > /dev/null 2>&1 || echo "STEP FAILED: sealed"
else echo "both P arms stopped for futility: the sealed step is skipped (as the rule says)"; fi
echo "EVENT_DIR=$EVENT_DIR"
grep -E "GATE|PRE-REGISTERED RESULT|arm for the sealed|Traceback|WARNING|FAILED|refusing|sealed evaluation|FUTILITY" "$EVENT_DIR/event.log" | cut -c1-200
ls "$EVENT_DIR" | wc -l
[ -f results/SEALED_LOCK.json ] && echo "ERROR: a dry run created the sealed lock" || echo "sealed lock untouched"
echo DRY_EVENT_DONE
