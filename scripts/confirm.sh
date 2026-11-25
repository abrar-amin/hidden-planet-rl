#!/usr/bin/env bash
# One-command live confirmation of the strict0 reward (about $5):
#   fresh SFT on strict0 teacher data -> eval A (SFT model) -> RL 10 steps, no in-run evals -> eval B (step-10 model) -> paired comparison.
# Both evals use the SAME 96 systems, greedy (temperature 0), 1 sample (test seeds 2000-2095; not the tuning seeds 0-149, not the retired sealed range 10000-10095).
#   bash scripts/confirm.sh            DRY=1 bash scripts/confirm.sh   (fake service, reduced sizes with QUICK=1)
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HIDDEN_PLANET_RV=ideal
MODEL=${MODEL:-Qwen/Qwen3.5-9B}; RUN=${RUN:-runs/confirm_strict0}; SEED=${SEED:-0}
MODE=strict0; WRONG=1.0; COST=0.008
SFT_STEPS=${SFT_STEPS:-10}; SFT_BATCH=${SFT_BATCH:-16}; SFT_MAX_USD=${SFT_MAX_USD:-0.8}
RL_STEPS=${RL_STEPS:-10}; RL_BATCH=${RL_BATCH:-5}; RL_GROUP=${RL_GROUP:-8}; RL_LR=${RL_LR:-4.7e-5}; RL_MAX_USD=${RL_MAX_USD:-2.9}; RL_MAX_MIN=${RL_MAX_MIN:-25}
EVAL_START=${EVAL_START:-2000}; EVAL_SYS=${EVAL_SYS:-96}; EVAL_SAMPLES=${EVAL_SAMPLES:-1}; EVAL_TEMP=${EVAL_TEMP:-0}   # greedy, 1 sample: an unchanged policy gives identical episodes
AMBIG=${AMBIG:-0.6}                   # share of RL training systems from the ambiguous-first-fit pool (data/curriculum_hard.json)
INIT_FROM=${INIT_FROM:-}
if [ "${QUICK:-0}" = "1" ]; then SFT_STEPS=2; SFT_BATCH=8; RL_STEPS=2; RL_BATCH=2; RL_GROUP=4; EVAL_SYS=6; fi
DRYFLAG=""; if [ "${DRY:-0}" = "1" ]; then DRYFLAG="--dry-run"; fi
if [ -z "$DRYFLAG" ] && [ -z "${TINKER_API_KEY:-}" ]; then echo "TINKER_API_KEY is not exported (or use DRY=1)"; exit 1; fi
if [ "$EVAL_START" -le 10095 ] && [ $((EVAL_START + EVAL_SYS)) -gt 10000 ]; then echo "eval seeds overlap the sealed set"; exit 1; fi
mkdir -p "$RUN"; LOG="$RUN/confirm.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
run() { say "+ $*"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
RW="--reward-mode $MODE --wrong-penalty $WRONG --cost-per-unit $COST"
EV="--level hard --start $EVAL_START --n $EVAL_SYS --samples $EVAL_SAMPLES --temperature $EVAL_TEMP $RW"
say "=== confirm.sh strict0: SFT ${SFT_STEPS}x${SFT_BATCH} -> RL ${RL_STEPS} steps ${RL_BATCH}x${RL_GROUP} lr $RL_LR ambiguous-frac $AMBIG temp $EVAL_TEMP -> paired eval on seeds $EVAL_START..$((EVAL_START + EVAL_SYS - 1)) x $EVAL_SAMPLES ${DRYFLAG:+(DRY RUN)} ==="

# 1. SFT (skipped if INIT_FROM is given or already done)
SJ="$RUN/sft/sft.json"
if [ -n "$INIT_FROM" ]; then SJ="$INIT_FROM"; say "using existing SFT $SJ"
elif [ ! -f "$SJ" ]; then
  run $PY scripts/sft_warmstart.py $DRYFLAG --model "$MODEL" --data data/expert_fit_hard_strict0.jsonl --min-reward 0.2 $RW --steps "$SFT_STEPS" --batch "$SFT_BATCH" --max-dollars "$SFT_MAX_USD" --log "$RUN/sft" || { say "SFT FAILED"; exit 2; }
fi
SAMPLER_A=$($PY -c "import json;print(json.load(open('$SJ'))['sampler_path'])")

# 2. eval A: the SFT model (= RL step 0)
if [ ! -f "$RUN/evalA.json" ]; then
  run $PY scripts/eval_zero_shot.py $DRYFLAG --model "$MODEL" --checkpoint "$SAMPLER_A" $EV --out "$RUN/evalA.json" --episodes-out "$RUN/evalA_episodes.jsonl" --traces-out "$RUN/evalA_traces.jsonl"
  if [ ! -f "$RUN/evalA.json" ] && [ "$EVAL_TEMP" = "0" ]; then        # fall back if the service rejects temperature 0
    say "eval A failed at temperature 0; retrying both evals at temperature 0.05"
    EVAL_TEMP=0.05; EV="--level hard --start $EVAL_START --n $EVAL_SYS --samples $EVAL_SAMPLES --temperature $EVAL_TEMP $RW"
    run $PY scripts/eval_zero_shot.py $DRYFLAG --model "$MODEL" --checkpoint "$SAMPLER_A" $EV --out "$RUN/evalA.json" --episodes-out "$RUN/evalA_episodes.jsonl" --traces-out "$RUN/evalA_traces.jsonl"
  fi
  [ -f "$RUN/evalA.json" ] || { say "eval A FAILED"; exit 3; }
fi
EVAL_TEMP=$($PY -c "import json;print(json.load(open('$RUN/evalA.json'))['temperature'])")       # eval B always uses eval A's temperature
EV="--level hard --start $EVAL_START --n $EVAL_SYS --samples $EVAL_SAMPLES --temperature $EVAL_TEMP $RW"

# 3. RL without in-run evals
run $PY scripts/train_rl.py $DRYFLAG --model "$MODEL" --init-from "$SJ" $RW --ambiguous-frac "$AMBIG" --lr "$RL_LR" --seed "$SEED" --schedule hard --eval-level hard --steps "$RL_STEPS" \
  --batch "$RL_BATCH" --group "$RL_GROUP" --eval-n 2 --eval-every 0 --save-every 5 --max-dollars "$RL_MAX_USD" --max-minutes "$RL_MAX_MIN" --log "$RUN/rl" \
  || say "WARNING: RL exited non-zero"
SAMPLER_B=$($PY -c "
import json
rows=[json.loads(l) for l in open('$RUN/rl/checkpoints.jsonl')]
r=[x for x in rows if x.get('sampler_path')][-1]; print(r['sampler_path']); open('$RUN/evalB_checkpoint.txt','w').write(json.dumps(r))")
B_STEP=$($PY -c "import json;print(json.load(open('$RUN/evalB_checkpoint.txt'))['batch'])")
say "eval B uses checkpoint: $(cat "$RUN/evalB_checkpoint.txt" | cut -c1-120)"

# 4. eval B: the RL model, same systems and noise
[ -f "$RUN/evalB.json" ] || run $PY scripts/eval_zero_shot.py $DRYFLAG --model "$MODEL" --checkpoint "$SAMPLER_B" $EV --out "$RUN/evalB.json" --episodes-out "$RUN/evalB_episodes.jsonl" --traces-out "$RUN/evalB_traces.jsonl" || { say "eval B FAILED"; exit 4; }

# 5. paired comparison + pre-registered verdict
run $PY scripts/paired_compare.py "$RUN/evalA_episodes.jsonl" "$RUN/evalB_episodes.jsonl" --label-a "SFT (step 0)" --label-b "RL step $B_STEP" --json "$RUN/paired.json"
run $PY scripts/decision_quality.py --traces "$RUN/evalA_traces.jsonl" "$RUN/evalB_traces.jsonl" --labels "SFT (step 0)" "RL step $B_STEP" --refs teacher eig --start "$EVAL_START" --n "$EVAL_SYS" --json "$RUN/decision_quality.json"
run $PY scripts/confirm_verdict.py "$RUN"
say "=== done: $RUN/paired.json | RL spend estimate: $(cat "$RUN/rl/spend.txt" 2>/dev/null) ==="
