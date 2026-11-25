#!/usr/bin/env bash
# Per-run worker used by event.sh:  SFT on expert episodes -> eval -> RL from the SFT LoRA (capped) -> eval -> figure.
#
#   bash scripts/run_arm.sh                 # real run (TINKER_API_KEY must already be exported)
#   DRY=1 bash scripts/run_arm.sh           # same pipeline against the fake service: no network, no credits
#   MODE=rl_base bash scripts/run_arm.sh    # comparison arm: RL straight from the base model (no SFT)
#
# Re-running with the same RUN name resumes: finished stages are skipped, RL continues from its last checkpoint.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

MODEL=${MODEL:-Qwen/Qwen3.5-9B}
MODE=${MODE:-sft_rl}                 # sft_rl | rl_base
SEED=${SEED:-0}                      # selects the RL training systems; the eval sets do not depend on it
RUN=${RUN:-runs/arm_${MODE}_s${SEED}}
SEALED=0                             # the sealed set is handled only by scripts/event.sh sealed (one-use lock in results/SEALED_LOCK.json)
LEVEL=${LEVEL:-hard}                 # training + main eval level (easy is already solved by SFT alone)
SFT_STEPS=${SFT_STEPS:-15}; SFT_BATCH=${SFT_BATCH:-32}; SFT_MAX_USD=${SFT_MAX_USD:-10}
RL_STEPS=${RL_STEPS:-40};   RL_BATCH=${RL_BATCH:-16};   RL_GROUP=${RL_GROUP:-8}
COST=${COST:-0.008}                  # observation cost per unit in the RL reward
REWARD_MODE=${REWARD_MODE:-strict0}  # strict0: right +1, wrong -WRONG, "cannot" 0 (the pre-registered setting)
WRONG=${WRONG:-1.0}                   # with -1.0 a coin-flip alias guess has negative expected value (results/reward_mode_ev.txt)
RV_AVAIL=${RV_AVAIL:-ideal}           # realistic: RV precision from host brightness/activity, ~1 in 8 stars has no RVs (validated offline only)
export HIDDEN_PLANET_RV=$RV_AVAIL
KL_COEF=${KL_COEF:-0}                # >0: KL penalty towards the SFT policy (cookbook kl_penalty_coef); untested against the live service, falls back to 0 if RL fails to start
RL_LR=${RL_LR:-2e-5}                 # gentler than the cookbook default/10 (4.7e-5); abstention fell at both rates in live runs
AMBIG=${AMBIG:-0.6}                  # share of RL training systems from the ambiguous-first-fit pool
PILOT_STOP=${PILOT_STOP:-0}          # e.g. 5: stop after the step-5 eval and print the pre-registered abort verdict
OOC_MIX=${OOC_MIX:-0}                # >0: parallel arm with the "other" verdict on an out-of-class mixture
RL_MAX_USD=${RL_MAX_USD:-60}; RL_MAX_MIN=${RL_MAX_MIN:-55}
EVAL_N=${EVAL_N:-48}; EVAL_EVERY=${EVAL_EVERY:-5}; SAVE_EVERY=${SAVE_EVERY:-5}
RVTAG=""; if [ "$RV_AVAIL" = "realistic" ]; then RVTAG="_rv"; fi
DATA=${DATA:-"data/expert_fit_${LEVEL}_${REWARD_MODE}${RVTAG}.jsonl"}   # fit-following expert: every action and answer is a function of visible tool output
EXTRA="--cost-per-unit $COST --reward-mode $REWARD_MODE --wrong-penalty $WRONG"; EXTRA="$EXTRA --tool-level ${TOOL_LEVEL:-oracle}"; RLEXTRA="--ambiguous-frac $AMBIG --collapse-guard ${COLLAPSE_GUARD:-0} --eval-start ${EVAL_START:-0} --train-reward ${TRAIN_REWARD:-realised} --futility-step ${FUTILITY_STEP:-0}"; if [ "$OOC_MIX" != "0" ]; then EXTRA="$EXTRA --allow-other --ooc-mix $OOC_MIX"; fi
LRFLAG=""; if [ -n "$RL_LR" ]; then LRFLAG="--lr $RL_LR"; fi
if [ "$PILOT_STOP" != "0" ]; then LRFLAG="$LRFLAG --pilot-stop-step $PILOT_STOP"; fi
INIT_FROM=${INIT_FROM:-}              # existing SFT state path or sft.json: skips the SFT stage
DRYFLAG=""; if [ "${DRY:-0}" = "1" ]; then DRYFLAG="--dry-run"; fi
if [ -z "$DRYFLAG" ] && [ -z "${TINKER_API_KEY:-}" ]; then echo "TINKER_API_KEY is not exported (or use DRY=1)"; exit 1; fi

mkdir -p "$RUN"
LOG="$RUN/run.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { say "+ $*"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

say "=== run_arm.sh MODE=$MODE MODEL=$MODEL LEVEL=$LEVEL COST=$COST RL=${RL_STEPS}x${RL_BATCH}x${RL_GROUP} OOC_MIX=$OOC_MIX RUN=$RUN ${DRYFLAG:+(DRY RUN)} ==="

# 0. zero-shot baseline of the base model (cheap; skipped if already there)
[ "${SKIP_BASE:-0}" = "1" ] || [ -f "$RUN/eval_base_$LEVEL.json" ] || run $PY scripts/eval_zero_shot.py $DRYFLAG --model "$MODEL" --level "$LEVEL" --n "$EVAL_N" $EXTRA --out "$RUN/eval_base_$LEVEL.json" \
  || say "WARNING: base eval failed; continuing"

INIT=""
if [ -n "$INIT_FROM" ]; then
  INIT="--init-from $INIT_FROM"; say "using existing init checkpoint $INIT_FROM (SFT stage skipped)"
  if [ "$KL_COEF" != "0" ] && [ "${INIT_FROM##*.}" = "json" ]; then INIT="$INIT --kl-coef $KL_COEF --kl-ref $INIT_FROM"; fi
elif [ "$MODE" = "sft_rl" ]; then
  # 1. SFT warm start on expert episodes (a few minutes)
  if [ ! -f "$RUN/sft/sft.json" ]; then
    run $PY scripts/sft_warmstart.py $DRYFLAG --model "$MODEL" --data $DATA --steps "$SFT_STEPS" --batch "$SFT_BATCH" --min-reward 0.2 --reward-mode "$REWARD_MODE" --wrong-penalty "$WRONG" --cost-per-unit "$COST" --max-dollars "$SFT_MAX_USD" --log "$RUN/sft" \
      || { say "SFT FAILED - fix and re-run (nothing to resume yet)"; exit 2; }
  else say "SFT already done: $RUN/sft/sft.json"; fi
  SAMPLER=$($PY -c "import json;print(json.load(open('$RUN/sft/sft.json'))['sampler_path'])")
  # 2. eval the SFT model
  [ -f "$RUN/eval_sft_$LEVEL.json" ] || run $PY scripts/eval_zero_shot.py $DRYFLAG --model "$MODEL" --checkpoint "$SAMPLER" --level "$LEVEL" --n "$EVAL_N" $EXTRA --out "$RUN/eval_sft_$LEVEL.json" \
    || say "WARNING: SFT eval failed; continuing"
  INIT="--init-from $RUN/sft/sft.json"
  if [ "$KL_COEF" != "0" ]; then INIT="$INIT --kl-coef $KL_COEF --kl-ref $RUN/sft/sft.json"; fi
fi

# 3. RL (resumes from $RUN/rl/checkpoints.jsonl if present; cancelled cleanly at the dollar / minute caps)
rl() { run $PY scripts/train_rl.py $DRYFLAG --model "$MODEL" "$@" $EXTRA $RLEXTRA $LRFLAG --seed "$SEED" --schedule "$LEVEL" --eval-level "$LEVEL" --steps "$RL_STEPS" --batch "$RL_BATCH" --group "$RL_GROUP" \
  --eval-n "$EVAL_N" --eval-every "$EVAL_EVERY" --save-every "$SAVE_EVERY" --max-dollars "$RL_MAX_USD" --max-minutes "$RL_MAX_MIN" --log "$RUN/rl"; }
rl $INIT
RC=$?
if [ ! -s "$RUN/rl/metrics.jsonl" ] && [ "$KL_COEF" != "0" ]; then          # the KL path is untested against the live service: fall back rather than lose the run
  say "WARNING: RL produced no step with KL_COEF=$KL_COEF (exit $RC); retrying WITHOUT the KL penalty"
  INIT_NOKL=$(echo "$INIT" | sed -E 's/--kl-coef [^ ]+ --kl-ref [^ ]+//')
  rm -rf "$RUN/rl"; rl $INIT_NOKL; RC=$?; echo "KL disabled by fallback" > "$RUN/kl_fallback.txt"
fi
[ "$RC" = "0" ] || say "WARNING: RL exited non-zero; the last checkpoint is in $RUN/rl/checkpoints.jsonl"

# 4. final evals: last checkpoint on the fixed set (easy and hard); with SEALED=1 the BEST checkpoint (by fixed-set eval) and the SFT init on the sealed set
EVALFLAGS="--cost-per-unit $COST --reward-mode $REWARD_MODE --wrong-penalty $WRONG --tool-level ${TOOL_LEVEL:-oracle}"
LAST=$($PY -c "
import json
try:
    rows=[json.loads(l) for l in open('$RUN/rl/checkpoints.jsonl')]
    print([r['sampler_path'] for r in rows if r.get('sampler_path')][-1])
except Exception: pass")
BEST=$($PY -c "
import json, csv
try:
    ck={int(json.loads(l)['batch']): json.loads(l).get('sampler_path') for l in open('$RUN/rl/checkpoints.jsonl')}
    best=None
    for r in csv.DictReader(open('$RUN/rl/metrics.csv')):
        st=int(float(r['step']))
        if r.get('eval_score') not in (None,'') and ck.get(st):
            v=float(r['eval_score'])-$COST*float(r['eval_spent'])
            if best is None or v>best[0]: best=(v,st,ck[st])
    print(best[2] if best else '')
    open('$RUN/best_checkpoint.txt','w').write(str(best))
except Exception: pass")
if [ "${SKIP_FINAL:-0}" = "1" ]; then say "SKIP_FINAL=1: final evals skipped"
elif [ -n "$LAST" ]; then
  for lv in easy hard; do
    [ -f "$RUN/eval_rl_$lv.json" ] || run $PY scripts/eval_zero_shot.py $DRYFLAG --model "$MODEL" --checkpoint "$LAST" --level $lv --n "$EVAL_N" $EVALFLAGS --out "$RUN/eval_rl_$lv.json" || say "WARNING: final eval ($lv) failed"
  done
  if [ "${FINAL_PAIRED:-$SEALED}" = "1" ]; then
    SJ="$RUN/sft/sft.json"; [ -f "$SJ" ] || SJ="$INIT_FROM"
    if [ -f "$SJ" ]; then
      A_SAMPLER=$($PY -c "import json;print(json.load(open('$SJ'))['sampler_path'])")
      B_SAMPLER="${BEST:-$LAST}"
      SETS="paired:--start 2000 --n 96"; if [ "$SEALED" = "1" ]; then SETS="$SETS|sealed:--eval-set sealed"; fi
      say "FINAL paired evaluation: A = SFT init, B = best RL checkpoint $(cat "$RUN/best_checkpoint.txt" 2>/dev/null)"
      OLDIFS=$IFS; IFS='|'
      for spec in $SETS; do
        IFS=$OLDIFS
        name=${spec%%:*}; sel=${spec#*:}
        for dec in greedy sampled; do
          if [ "$dec" = "greedy" ]; then DFLAG="--temperature 0 --samples 1"; else DFLAG="--temperature 1 --samples 2"; fi
          A="$RUN/final_${name}_${dec}_A"; B="$RUN/final_${name}_${dec}_B"
          if [ ! -f "$A.json" ] && [ -n "${A_FROM:-}" ] && [ -f "$A_FROM/final_${name}_${dec}_A.json" ]; then cp "$A_FROM"/final_${name}_${dec}_A* "$RUN"/; say "eval A ($name, $dec) copied from $A_FROM"; fi
          [ -f "$A.json" ] || run $PY scripts/eval_zero_shot.py $DRYFLAG --model "$MODEL" --checkpoint "$A_SAMPLER" --level "$LEVEL" $sel $DFLAG $EVALFLAGS --out "$A.json" || say "WARNING: eval A ($name, $dec) failed"
          [ -f "$B.json" ] || run $PY scripts/eval_zero_shot.py $DRYFLAG --model "$MODEL" --checkpoint "$B_SAMPLER" --level "$LEVEL" $sel $DFLAG $EVALFLAGS --out "$B.json" || say "WARNING: eval B ($name, $dec) failed"
          if [ -f "${A}_episodes.jsonl" ] && [ -f "${B}_episodes.jsonl" ]; then
            say "--- paired comparison: $name set, $dec decoding ---"
            run $PY scripts/paired_compare.py "${A}_episodes.jsonl" "${B}_episodes.jsonl" --label-a "SFT" --label-b "RL best" --json "$RUN/final_${name}_${dec}_paired.json"
            run $PY scripts/decision_quality.py --traces "${A}_traces.jsonl" "${B}_traces.jsonl" --labels "SFT" "RL best" --json "$RUN/final_${name}_${dec}_decision_quality.json"
          fi
        done
        IFS='|'
      done
      IFS=$OLDIFS
    else say "no SFT json found; skipping the paired final evaluation"; fi
  fi
else say "no RL sampler checkpoint found; skipping final evals"; fi
ZS=""; [ -f "$RUN/eval_base_$LEVEL.json" ] && ZS="--zero-shot $RUN/eval_base_$LEVEL.json"
[ -f "$RUN/eval_sft_$LEVEL.json" ] && ZS="$ZS --sft $RUN/eval_sft_$LEVEL.json"
REF="results/reference_${LEVEL}_${REWARD_MODE}${RVTAG}.json"; [ -f "$REF" ] || REF="results/reference_${LEVEL}.json"
run $PY scripts/plot.py --run "$RUN/rl" --label "$MODE" --level "$LEVEL" --ref "$REF" $ZS --out "$RUN/learning_curve.png" || say "WARNING: plot failed"
[ -f "$RUN/rl/pilot_verdict.txt" ] && say "$(cat "$RUN/rl/pilot_verdict.txt")"
say "=== done. spend estimate: $(cat "$RUN/rl/spend.txt" 2>/dev/null) | figure: $RUN/learning_curve.png | log: $LOG ==="
