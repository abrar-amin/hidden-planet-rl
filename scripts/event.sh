#!/usr/bin/env bash
# The pre-registered experiment: arms, gate, paired comparisons and the sealed evaluation (background: diagnosis/DIAGNOSIS.md, audit_v3/AUDIT_V3.md).
#
#   bash scripts/event.sh sft fixed            # SFT on the primary demonstrator + dev evals + GATE     (also: sft fit | compact | lite)
#   bash scripts/event.sh rl fixed 0           # RL from that SFT, seed 0; refuses unless the gate passed (also: rl fixed 1; rl fit 0 is optional)
#                                              # 24 systems x 4 rollouts, AMBIG 0.3, trained on the expected score, futility stop at step 15
#   bash scripts/event.sh control fixed        # no-RL control: token-matched extra SFT steps from the same state, fresh batches
#   bash scripts/event.sh final                # dev evals (192 systems) of every finished arm, paired comparisons, verdicts, figure
#   bash scripts/event.sh sealed               # ONCE: the arm chosen by the pre-registered rule, on the sealed set
# DRY=1 runs against the fake service; QUICK=1 uses tiny sizes. TAG=b gives fresh arm directories (e.g. after changing RL_BATCH).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HIDDEN_PLANET_RV=ideal
MODEL=${MODEL:-Qwen/Qwen3.5-9B}; E=${EVENT_DIR:-runs/event}; PRIMARY=${PRIMARY:-fixed}; TAG=${TAG:-}
RW0="--reward-mode strict0 --wrong-penalty 1.0 --cost-per-unit 0.008"
DEV_START=3000; DEV_N=${DEV_N:-192}          # primary analysis set (test seeds 3000-3191)
SEL_START=4000; SEL_N=${SEL_N:-48}           # in-run evals and checkpoint selection (test seeds 4000-4047): disjoint from the dev set
SFT_STEPS=${SFT_STEPS:-15}; SFT_BATCH=${SFT_BATCH:-32}
RL_STEPS=${RL_STEPS:-30}; RL_BATCH=${RL_BATCH:-24}; RL_GROUP=${RL_GROUP:-4}; RL_LR=${RL_LR:-2e-5}; RL_MAX_USD=${RL_MAX_USD:-24}; RL_MAX_MIN=${RL_MAX_MIN:-40}
AMBIG=${AMBIG:-0.3}; TRAIN_REWARD=${TRAIN_REWARD:-expected}; FUTILITY_STEP=${FUTILITY_STEP:-15}   # as pre-registered
REFS=${REFS:-results/dev192_avg3}           # scripted references averaged over 3 policy seeds
CTRL_BATCH=${CTRL_BATCH:-128}                # about the same trained tokens per step as one RL step of 96 episodes
if [ "${QUICK:-0}" = "1" ]; then DEV_N=6; SEL_N=4; SFT_STEPS=2; SFT_BATCH=8; RL_STEPS=${RL_STEPS_Q:-3}; RL_BATCH=2; RL_GROUP=4; CTRL_BATCH=8; FUTILITY_STEP=${FUTILITY_STEP_Q:-2}; fi
DRYFLAG=""; if [ "${DRY:-0}" = "1" ]; then DRYFLAG="--dry-run"; fi
CMD=${1:-help}
if [ -z "$DRYFLAG" ] && [ -z "${TINKER_API_KEY:-}" ] && [ "$CMD" != "refs" ] && [ "$CMD" != "help" ]; then echo "TINKER_API_KEY is not exported (or use DRY=1)"; exit 1; fi
mkdir -p "$E"; LOG="$E/event.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { say "+ $*"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
level_of() { if [ "$1" = "lite" ]; then echo lite; else echo oracle; fi; }       # the 'lite' demonstrator runs with TOOL_LEVEL=lite
demo_of() { echo "$1" | cut -d_ -f2; }                                           # sft_fixed / rl_fixed_s0 / control_fixed -> fixed
ref_of() { case "$1" in fit) echo teacher;; lite) echo lite_teacher;; fixed) echo fixed_diverse;; *) echo "$1";; esac; }   # D2 for arm P uses the script that made the SFT data
sampler_of() { $PY -c "import json;print(json.load(open('$1'))['sampler_path'])"; }

evals() {   # evals <arm> <sampler> <set: dev|sealed> [sealed arm]
  local arm=$1 sampler=$2 set=$3 sel RW DEC dec out
  RW="$RW0 --tool-level $(level_of "$(demo_of "$arm")")"
  if [ "$set" = "sealed" ]; then sel="--eval-set sealed --sealed-arm ${4:-}"; else sel="--start $DEV_START --n $DEV_N"; fi
  if [ "$set" = "sealed" ] && [ -n "$DRYFLAG" ]; then sel="--start 9000 --n $DEV_N"; fi      # dry runs never touch the sealed seeds
  for dec in sampled greedy; do
    if [ "$dec" = "greedy" ]; then DEC="--temperature 0 --samples 1"; else DEC="--temperature 1 --samples 2"; fi
    out="$E/final_${set}_${dec}_${arm}.json"
    [ -f "$out" ] || run $PY scripts/eval_zero_shot.py $DRYFLAG --model "$MODEL" --checkpoint "$sampler" --level hard $sel $DEC $RW --out "$out" || { say "WARNING: eval $arm $set $dec failed"; return 1; }
  done
}
gate() {    # same per-episode files and the same no-answer definition as final_verdict.py
  $PY - "$E" "$1" <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
from hidden_planet.expected_score import es_from_metrics
E, arm = sys.argv[1:3]
ok = True
for dec in ("sampled", "greedy"):
    rows = [json.loads(l) for l in open(f"{E}/final_dev_{dec}_{arm}_episodes.jsonl")]
    m = lambda k: sum(r.get(k, 0.0) for r in rows) / len(rows)
    es = sum(es_from_metrics(r) for r in rows) / len(rows)
    print(f"{arm} {dec}: n={len(rows)} expected score {es:+.3f}, reward {m('reward'):+.3f}, fully correct {m('correct'):.2f}, no answer {m('invalid'):.3f}, abstains {m('verdict_cannot'):.2f}, "
          f"commits with posterior <= 0.5: {m('post_lowconf_commit'):.3f}, RV buying {m('rv_buy_rate'):.2f}, spent {m('spent'):.1f}")
    ok &= m("invalid") <= 0.05
print("GATE (no-answer rate <= 0.05 in both decodings):", "PASS" if ok else "FAIL - RL will not start from this init (inspect final_dev_*_traces.jsonl)")
open(f"{E}/gate_{arm}.txt", "w").write("PASS\n" if ok else "FAIL\n")
PYEOF
}
select_checkpoint() {   # written ONCE per arm, at `final`; `sealed` reads the same file. Selection uses the in-run evals (seeds 4000+), never the dev set.
  local D=$1
  [ -f "$D/selected_checkpoint.json" ] || $PY - "$D" <<'PYEOF'
import json, csv, sys, os
run = sys.argv[1]
ck = {int(json.loads(l)["batch"]): json.loads(l).get("sampler_path") for l in open(os.path.join(run, "rl", "checkpoints.jsonl")) if l.strip()}
best = None
try:
    for r in csv.DictReader(open(os.path.join(run, "rl", "metrics.csv"))):
        st = int(float(r["step"]))
        if r.get("eval_es") not in (None, "") and ck.get(st):         # selection by the in-run expected score
            v = float(r["eval_es"])
            if best is None or v > best[0]:
                best = (v, st, ck[st])
except Exception:
    pass
last = max(k for k, v in ck.items() if v)
if best is None:
    best = (float("nan"), last, ck[last])
json.dump(dict(selection_eval_reward=best[0], step=best[1], sampler_path=best[2], last_step=last), open(os.path.join(run, "selected_checkpoint.json"), "w"))
PYEOF
  $PY -c "import json;print(json.load(open('$D/selected_checkpoint.json'))['sampler_path'])"
}

case "$CMD" in
refs)
  [ -f $REFS/refs_dev_summary.json ] || { mkdir -p $REFS; run $PY scripts/eval_refs.py --start $DEV_START --n 192 --policy-seeds 3 --agents teacher eig compact compact_smart fixed fixed_diverse lite_teacher random --out-prefix $REFS/refs_dev; }
  cp $REFS/refs_dev_*_episodes.jsonl $REFS/refs_dev_*_traces.jsonl "$E"/
  say "scripted references on the dev set (test seeds 3000-3191, averaged over 3 policy seeds) copied from $REFS to $E" ;;
sft)
  DEMO=${2:?usage: event.sh sft fixed|fit|compact|lite}; D="$E/sft_$DEMO"; RW="$RW0 --tool-level $(level_of "$DEMO")"
  # every demonstrator is cloned UNFILTERED (the demonstrator as it is, wrong answers included)
  [ -f "$D/sft.json" ] || run $PY scripts/sft_warmstart.py $DRYFLAG --model "$MODEL" --data "data/expert_${DEMO}_hard_strict0.jsonl" --min-reward -9 $RW --steps "$SFT_STEPS" --batch "$SFT_BATCH" --max-dollars 4 --log "$D" || { say "SFT FAILED"; exit 2; }
  evals "sft_$DEMO" "$(sampler_of "$D/sft.json")" dev || exit 3
  gate "sft_$DEMO" 2>&1 | tee -a "$LOG" ;;
rl)
  DEMO=${2:?usage: event.sh rl <demo> SEED}; SEED=${3:?seed}; ARM="rl_${DEMO}_s${SEED}${TAG}"
  [ -f "$E/sft_$DEMO/sft.json" ] || { say "run 'event.sh sft $DEMO' first"; exit 2; }
  if [ "$(cat "$E/gate_sft_$DEMO.txt" 2>/dev/null)" != "PASS" ] && [ "${FORCE:-0}" != "1" ]; then say "GATE not passed for sft_$DEMO: refusing to start RL (FORCE=1 overrides and must be reported)"; exit 2; fi
  # an arm stopped by the futility rule is final; it is never resumed (no FORCE override)
  [ -f "$E/$ARM/rl/futility_stop.txt" ] && { say "$ARM was stopped by the pre-registered futility rule: not resuming ($(cat "$E/$ARM/rl/futility_stop.txt"))"; exit 2; }
  mkdir -p "$E/$ARM"; rm -f "$E/$ARM/finished.txt"
  QV=""; [ "${QUICK:-0}" = "1" ] && QV="EVAL_EVERY=1 SAVE_EVERY=1"
  env DRY=${DRY:-0} RUN="$E/$ARM" SEED=$SEED INIT_FROM="$E/sft_$DEMO/sft.json" LEVEL=hard REWARD_MODE=strict0 WRONG=1.0 COST=0.008 SKIP_BASE=1 SKIP_FINAL=1 \
    TOOL_LEVEL=$(level_of "$DEMO") RL_STEPS=$RL_STEPS RL_BATCH=$RL_BATCH RL_GROUP=$RL_GROUP RL_LR=$RL_LR RL_MAX_USD=$RL_MAX_USD RL_MAX_MIN=$RL_MAX_MIN AMBIG=$AMBIG KL_COEF=${KL_COEF:-0} TRAIN_REWARD=$TRAIN_REWARD FUTILITY_STEP=$FUTILITY_STEP \
    EVAL_N=$SEL_N EVAL_EVERY=${EVAL_EVERY:-5} SAVE_EVERY=${SAVE_EVERY:-5} EVAL_START=$SEL_START COLLAPSE_GUARD=${COLLAPSE_GUARD:-10} $QV bash scripts/run_arm.sh
  RC=$?; echo "exit $RC $(date +%H:%M:%S)" > "$E/$ARM/finished.txt"; say "$ARM finished (exit $RC)" ;;
control)
  DEMO=${2:?usage: event.sh control <demo>}; D="$E/control_$DEMO"; RW="$RW0 --tool-level $(level_of "$DEMO")"
  SE=5; [ "${QUICK:-0}" = "1" ] && SE=1
  [ -f "$D/sft.json" ] || run $PY scripts/sft_warmstart.py $DRYFLAG --model "$MODEL" --init "$E/sft_$DEMO/sft.json" --data "data/expert_${DEMO}_hard_strict0.jsonl" --min-reward -9 $RW \
      --steps "$RL_STEPS" --batch "$CTRL_BATCH" --lr "$RL_LR" --save-every $SE --max-dollars 20 --log "$D" || say "WARNING: control failed" ;;
final)
  # the primary arms are the explicit pre-registered list (rl_<primary>_s0$TAG, rl_<primary>_s1$TAG), never a glob,
  # and `final` refuses to start until each of them and the control has finished (FORCE_FINAL=1 overrides; a missing arm then FAILS the verdict).
  PARMS="rl_${PRIMARY}_s0${TAG} rl_${PRIMARY}_s1${TAG}"
  if [ "${FORCE_FINAL:-0}" != "1" ]; then
    NOTDONE=""
    for A in $PARMS; do [ -f "$E/$A/finished.txt" ] || [ -f "$E/$A/rl/futility_stop.txt" ] || NOTDONE="$NOTDONE $A"; done
    [ -f "$E/control_$PRIMARY/sft.json" ] || NOTDONE="$NOTDONE control_$PRIMARY"
    [ -z "$NOTDONE" ] || { say "final: not finished yet:$NOTDONE. Wait for them (or FORCE_FINAL=1, which fails the verdict for any missing arm and must be reported)"; exit 2; }
  fi
  bash "$0" refs > /dev/null 2>&1
  for DEMO in fixed fit compact lite; do
    [ -f "$E/final_dev_sampled_sft_${DEMO}_episodes.jsonl" ] || continue
    ARMS=""; FUTARMS=""; LISTED=""; MAXSTEP=0
    for A in rl_${DEMO}_s0${TAG} rl_${DEMO}_s1${TAG}; do
      D="$E/$A"
      if [ "$DEMO" = "$PRIMARY" ]; then LISTED="$LISTED $A"; else [ -d "$D" ] && LISTED="$LISTED $A"; fi
      if [ -f "$D/rl/futility_stop.txt" ]; then say "$(basename "$D"): FUTILITY STOP (pre-registered); no dev evaluation: $(cat "$D/rl/futility_stop.txt")"; FUTARMS="$FUTARMS $(basename "$D")"; continue; fi
      [ -f "$D/rl/checkpoints.jsonl" ] || { [ -d "$D" ] && say "$A: no checkpoints; it fails the verdict"; continue; }
      S=$(select_checkpoint "$D") || continue
      evals "$(basename "$D")" "$S" dev && ARMS="$ARMS $(basename "$D")"
      L=$($PY -c "import json;print(json.load(open('$D/selected_checkpoint.json'))['last_step'])"); [ "$L" -gt "$MAXSTEP" ] && MAXSTEP=$L
    done
    [ -n "$LISTED" ] || continue
    if [ -n "$ARMS" ] && [ -f "$E/control_$DEMO/sft.json" ] && [ -f "$E/control_$DEMO/eval_sampler.json" ]; then
      evals "control_$DEMO" "$(sampler_of "$E/control_$DEMO/eval_sampler.json")" dev    # chosen once, at the first `final`; never rewritten
    elif [ -n "$ARMS" ] && [ -f "$E/control_$DEMO/sft.json" ]; then
      # the control is compared at the largest step any RL arm of this demonstrator reached (never fewer updates than an RL checkpoint could have)
      CS=$($PY - "$E/control_$DEMO/sft.json" "$MAXSTEP" "$E/control_$DEMO/eval_sampler.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); want = int(sys.argv[2]); ck = {int(k): v for k, v in d.get("checkpoints", {}).items()}
ok = [k for k in ck if k <= max(want, min(ck))] or list(ck)
step = max(ok); json.dump(dict(step=step, sampler_path=ck[step]["sampler_path"], rl_max_step=want), open(sys.argv[3], "w")); print(ck[step]["sampler_path"])
PYEOF
)
      evals "control_$DEMO" "$CS" dev
    fi
    for dec in sampled greedy; do
      XA=""; [ "$DEMO" = "$PRIMARY" ] && XA="--expect-arms 2"
      run $PY scripts/final_verdict.py "$E" --set dev --dec $dec --init "sft_$DEMO" --demo "$(ref_of "$DEMO")" --control "control_$DEMO" --arms $LISTED $XA --n-expected $DEV_N
    done
    for A in $ARMS; do
      [ -f "$E/final_dev_greedy_${A}_traces.jsonl" ] && run $PY scripts/decision_quality.py --traces "$E/final_dev_greedy_sft_${DEMO}_traces.jsonl" "$E/final_dev_greedy_${A}_traces.jsonl" "$E/refs_dev_$(ref_of "$DEMO")_traces.jsonl" \
          --labels "sft_$DEMO" "$A" "$(ref_of "$DEMO") script" --json "$E/decision_quality_dev_${A}.json"
    done
  done
  RUNS=$(for D in "$E"/rl_*; do [ -f "$D/rl/metrics.csv" ] && echo "--run $D/rl --label $(basename "$D")"; done | tr '\n' ' ')
  [ -n "$RUNS" ] && run $PY scripts/plot.py $RUNS --level hard --ref results/reference_dev_strict0.json --out "$E/learning_curve.png"
  say "PRIMARY result: $E/final_verdict_dev_sampled_sft_${PRIMARY}.json" ;;
sealed)
  V="$E/final_verdict_dev_sampled_sft_${PRIMARY}.json"
  [ -f "$V" ] || { say "run 'event.sh final' first"; exit 2; }
  ARM=$($PY -c "import json;print(json.load(open('$V')).get('best_arm') or '')")
  [ -n "$DRYFLAG" ] && ARM=${SEALED_ARM:-$ARM}           # dry runs only: exercise the code path (on seeds 9000+) even when the mock policy did not succeed
  [ -n "$ARM" ] || { say "no arm succeeded on the dev set: by the pre-registered rule the sealed set is NOT used"; exit 1; }
  DEMO=$(demo_of "$ARM"); REFDEMO=$(ref_of "$DEMO"); NSEAL=96; [ -n "$DRYFLAG" ] && NSEAL=$DEV_N
  say "sealed evaluation for the pre-registered best arm: $ARM"
  OK=1
  if [ -n "$DRYFLAG" ]; then run $PY scripts/eval_refs.py --start 9000 --n $DEV_N --policy-seeds 3 --agents $REFDEMO fixed teacher --out-prefix "$E/refs_sealed" || OK=0
  else run $PY scripts/eval_refs.py --eval-set sealed --sealed-arm "$ARM" --policy-seeds 3 --agents $REFDEMO fixed teacher eig --out-prefix "$E/refs_sealed" || OK=0; fi
  evals "sft_$DEMO" "$(sampler_of "$E/sft_$DEMO/sft.json")" sealed "$ARM" || OK=0
  [ -f "$E/control_$DEMO/eval_sampler.json" ] && { evals "control_$DEMO" "$(sampler_of "$E/control_$DEMO/eval_sampler.json")" sealed "$ARM" || OK=0; }
  evals "$ARM" "$(sampler_of "$E/$ARM/selected_checkpoint.json")" sealed "$ARM" || OK=0        # the checkpoint chosen at `final`, not re-selected
  for dec in sampled greedy; do run $PY scripts/final_verdict.py "$E" --set sealed --dec $dec --init "sft_$DEMO" --demo "$REFDEMO" --control "control_$DEMO" --arms "$ARM" --n-expected $NSEAL || OK=0; done
  if [ "$OK" = "1" ] && [ -z "$DRYFLAG" ]; then $PY -m hidden_planet.sealed finish "$ARM" | tee -a "$LOG"; fi
  [ "$OK" = "1" ] || say "sealed evaluation incomplete: re-run 'event.sh sealed' (finished parts are skipped; the lock allows only arm $ARM)" ;;
*) sed -n 2,10p "$0" ;;
esac
