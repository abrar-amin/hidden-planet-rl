"""RL training on the hidden-planet env with tinker-cookbook (GRPO-style groups, LoRA).

Real run (spends Tinker credits; needs TINKER_API_KEY):
  python scripts/train_rl.py --model Qwen/Qwen3.5-9B --steps 30 --batch 16 --group 8 --log runs/qwen9b_easy
Dry run (mock sampler, no API calls, no credits):
  python scripts/train_rl.py --dry-run --steps 6 --batch 4 --group 4 --log runs/dry

Both modes run tinker_cookbook.rl.train.main on a standard train.Config (checkpoints, periodic
held-out eval, metrics.jsonl). --dry-run swaps tinker.ServiceClient for hidden_planet.fake_tinker
(sampling / forward_backward / optim_step / saves are faked; everything else is the real path).
A watchdog mirrors metrics.jsonl to metrics.csv and cancels the run at --max-dollars / --max-minutes.
"""
import os, sys, json, time, argparse, asyncio
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hidden_planet.tasks import LEVEL_NAMES
from hidden_planet.tinker_env import HiddenPlanetDatasetBuilder
from hidden_planet import rollout_utils as ru

MODELS = {   # short name -> (tinker id, renderer with thinking off, renderer with thinking on)
    "qwen9b": ("Qwen/Qwen3.5-9B", "qwen3_5_disable_thinking", "qwen3_5"),
    "nemotron30b": ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "nemotron3_ultra_disable_thinking", "nemotron3_ultra"),
    "qwen35b": ("Qwen/Qwen3.6-35B-A3B", "qwen3_5_disable_thinking", "qwen3_5"),
}


def resolve_model(a):
    if a.model in MODELS:
        mid, r_off, r_on = MODELS[a.model]
    else:
        match = [v for v in MODELS.values() if v[0] == a.model]
        mid, r_off, r_on = match[0] if match else (a.model, None, None)
    renderer = a.renderer or (r_on if a.thinking else r_off)
    return mid, renderer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="tinker model id or one of: " + ", ".join(MODELS))
    ap.add_argument("--renderer", default=None); ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--steps", type=int, default=30); ap.add_argument("--batch", type=int, default=16, help="systems (groups) per step")
    ap.add_argument("--group", type=int, default=8, help="rollouts per system")
    ap.add_argument("--schedule", default="easy", choices=["easy", "hard", "mix"])
    ap.add_argument("--eval-level", default="easy", choices=list(LEVEL_NAMES)); ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=5); ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--lr", type=float, default=None, help="default: tinker_cookbook.hyperparam_utils.get_lr(model) (LoRA)")
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--max-tokens", type=int, default=512, help="per-turn generation cap")
    ap.add_argument("--loss", default="importance_sampling"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="runs/hidden_planet"); ap.add_argument("--no-abstain-aware", action="store_true")
    ap.add_argument("--init-from", "--load-checkpoint", dest="load_checkpoint", default=None,
                    help="warm start: a tinker:// STATE path (not a sampler path), or the sft.json written by sft_warmstart.py")
    ap.add_argument("--reward-mode", default="partial", choices=["partial", "strict", "strict0"], help="strict: 'planet' = +1 only if period AND mass are right, else -wrong-penalty")
    ap.add_argument("--wrong-penalty", type=float, default=None, help="penalty for a wrong 'planet'/'none' in strict mode (default 0.5)")
    ap.add_argument("--tool-level", default="oracle", choices=["oracle", "lite"], help="lite: fit_model prints chi2 / BIC only, no posterior probabilities")
    ap.add_argument("--eval-start", type=int, default=0, help="first test-split seed of the in-run eval set (the event uses the dev set, 3000)")
    ap.add_argument("--ambiguous-frac", type=float, default=0.0, help="share of RL training systems drawn from data/curriculum_hard.json (ambiguous first fit)")
    ap.add_argument("--kl-coef", type=float, default=0.0, help="cookbook KL penalty towards a reference policy (0 = off)")
    ap.add_argument("--kl-ref", default=None, help="KL reference: a tinker:// SAMPLER path or an sft.json; default = the base model")
    ap.add_argument("--cost-per-unit", type=float, default=None, help="observation cost in the reward (default 0.005)")
    ap.add_argument("--allow-other", action="store_true", help="enable the 4th verdict 'other'")
    ap.add_argument("--ooc-mix", type=float, default=0.0, help="fraction of out-of-class systems in train and eval (use with --allow-other)")
    ap.add_argument("--train-reward", default="realised", choices=["realised", "expected"], help="expected: train on the posterior-expected score (hidden_planet/expected_score.py); evals stay realised")
    ap.add_argument("--futility-step", type=int, default=0, help="futility stop: at the first in-run eval at or after this step, stop if its expected score is not above step 0's (0 = off)")
    ap.add_argument("--max-dollars", type=float, default=60.0, help="cancel when the conservative spend estimate reaches this")
    ap.add_argument("--collapse-guard", type=int, default=0, help="from this step on, stop if in-run eval commits with posterior <= 0.5 rise by > 0.08 over step 0 (0 = off)")
    ap.add_argument("--pilot-stop-step", type=int, default=0, help="stop after the eval at this step and print the pre-registered abort-rule verdict")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="cancel after this much wall clock (0 = off)")
    ap.add_argument("--rv-availability", default=None, choices=["ideal", "realistic"], help="realistic: RV precision from host brightness/activity; faint stars have no RVs")
    ap.add_argument("--dry-run", action="store_true", help="fake tinker service: real train.main, no network, no credits")
    return ap.parse_args()


def default_lr(model_id):
    try:
        from tinker_cookbook.hyperparam_utils import get_lr
        return float(get_lr(model_id, is_lora=True)) / 10.0     # RL: 10x below the SL LoRA recommendation
    except Exception:
        return 4e-5


def make_builder(a, model_id, renderer):
    return HiddenPlanetDatasetBuilder(model_name_for_tokenizer=model_id, renderer_name=renderer, batch_size=a.batch,
                                      group_size=a.group, n_batches=a.steps, n_test_systems=a.eval_n, seed=a.seed,
                                      schedule=a.schedule, eval_level=LEVEL_NAMES[a.eval_level], abstain_aware=not a.no_abstain_aware,
                                      cost_per_unit=a.cost_per_unit, allow_other=a.allow_other, ooc_mix=a.ooc_mix,
                                      reward_mode=a.reward_mode, wrong_penalty=a.wrong_penalty, ambiguous_frac=a.ambiguous_frac, eval_start=a.eval_start, tool_level=a.tool_level,
                                      train_reward=a.train_reward)


# ------------------------------------------------------------------ run (real or faked service)
PRICES = {  # USD / 1M tokens (prefill, cached prefill, sample, train), tinker-docs models page, fetched 2026-11-15
    "Qwen/Qwen3.5-9B": (0.30, 0.06, 0.807, 1.463),
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": (0.184, 0.0368, 0.495, 0.44),
    "Qwen/Qwen3.6-35B-A3B": (0.45, 0.09, 1.335, 1.177),
}
SAFETY = 1.10     # guard margin on top of the cache-aware estimate


def spend_estimate(log_dir, model_id):
    """Running cost from the cookbook's per-step token counters in metrics.jsonl, pricing cache-hit
    prompt tokens at the cached rate (multi-turn episodes are mostly cache hits; pricing everything
    uncached overstated real billing by ~30 % in a live calibration: estimated $9.7, billed $6.81).
    Uses the larger of (a) the cookbook's own tracker `cost/cumulative_usd` and (b) this estimate, x SAFETY.
    On resume the cookbook drops metrics rows at or after the checkpoint step, after copying the file to
    metrics.jsonl.pre_resume_<step>; those dropped rows were really spent, so they are added back here (caps are cumulative)."""
    import glob
    pre, prc, smp, trn = PRICES.get(model_id, (0.7, 0.14, 5.6, 4.2))
    usd = book = 0.0
    path = os.path.join(log_dir, "metrics.jsonl")
    sources = [(path, None)] if os.path.exists(path) else []
    for old in sorted(glob.glob(path + ".pre_resume_*")):
        try:
            sources.append((old, int(old.rsplit("_", 1)[1])))
        except ValueError:
            continue
    for src, from_step in sources:
      for line in open(src):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if from_step is not None and not (isinstance(r.get("step"), int) and r["step"] >= from_step):
            continue                                   # kept rows of an old file are already in metrics.jsonl
        book = max(book, float(r.get("cost/cumulative_usd", 0.0) or 0.0))
        if "tokens/prefill_uncached" in r:            # training-step usage, exact token classes
            usd += (r.get("tokens/prefill_uncached", 0) * pre + r.get("tokens/prefill_cached", 0) * prc
                    + r.get("tokens/sampled", 0) * smp + r.get("tokens/trained", 0) * trn) / 1e6
        else:
            ob, ac, ca = r.get("env/all/total_ob_tokens", 0.0), r.get("env/all/total_ac_tokens", 0.0), r.get("env/all/total_cached_ob_tokens", 0.0)
            usd += ((ob - ca) * pre + ca * prc + ac * smp) / 1e6
        ob, ac, ca = r.get("test/env/all/total_ob_tokens", 0.0), r.get("test/env/all/total_ac_tokens", 0.0), r.get("test/env/all/total_cached_ob_tokens", 0.0)
        if ca == 0 and ob:                             # eval rows may not report cache hits: assume the repeated prefix is cached
            ca = ob * (1 - 1 / max(r.get("test/env/all/turns_per_episode", 1.0), 1.0)) * 0.9
        usd += ((ob - ca) * pre + ca * prc + ac * smp) / 1e6
    return SAFETY * max(usd, book)


def eval_rows(log_dir):
    """[(step, eval reward at standard cost, eval invalid fraction)] from metrics.jsonl."""
    out = []
    path = os.path.join(log_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = "test/env/all/reward_std_cost" if "test/env/all/reward_std_cost" in r else "test/env/all/reward/total"
        if key in r:
            out.append((int(r.get("step", r.get("progress/batch", 0))), float(r[key]), float(r.get("test/env/all/invalid", 0.0)),
                        float(r.get("test/env/all/verdict_cannot", 0.0)), float(r.get("test/env/all/post_lowconf_commit", r.get("test/env/all/lowconf_commit", 0.0)))))
    return sorted(out)


def eval_es_rows(log_dir):
    """[(step, in-run eval expected score)] from metrics.jsonl (test/env/all/es, logged by the grader)."""
    out = []
    path = os.path.join(log_dir, "metrics.jsonl")
    if not os.path.exists(path):
        return out
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "test/env/all/es" in r:
            out.append((int(r.get("step", r.get("progress/batch", 0))), float(r["test/env/all/es"])))
    return sorted(out)


def futility_verdict(rows, step):
    """Pre-registered futility rule: at the first in-run eval at or after `step`, STOP unless its expected score is
    strictly above the step-0 eval's (the SFT init, same selection-set systems). None until both evals exist."""
    r0 = next((r for r in rows if r[0] == 0), None); rs = next((r for r in rows if r[0] >= step), None)
    if r0 is None or rs is None:
        return None
    return dict(step0_es=r0[1], step=rs[0], step_es=rs[1], delta=rs[1] - r0[1], verdict="STOP" if rs[1] <= r0[1] else "CONTINUE")


def pilot_verdict(rows, step, drop=0.2, max_invalid=0.3):
    """Pre-registered abort rule (docs/archive/PILOT_PLAN.md): abort if the eval reward at `step` is more than
    `drop` below step 0, or more than `max_invalid` of eval episodes have no valid answer."""
    r0 = next((r for r in rows if r[0] == 0), None); rs = next((r for r in rows if r[0] >= step), None)
    if r0 is None or rs is None:
        return None
    abort = (rs[1] < r0[1] - drop) or (rs[2] > max_invalid)
    return dict(step0_reward=r0[1], step_reward=rs[1], delta=rs[1] - r0[1], invalid=rs[2], verdict="ABORT" if abort else "CONTINUE")


def first_start(log_dir):
    """Wall-clock time of the FIRST start of this run directory, kept across resumes, so the minute cap is cumulative."""
    p = os.path.join(log_dir, "cap_clock.json")
    if os.path.exists(p):
        try:
            return float(json.load(open(p))["first_start"])
        except Exception:
            pass
    t = time.time()
    os.makedirs(log_dir, exist_ok=True)
    json.dump(dict(first_start=t), open(p, "w"))
    return t


def over_cap_at_start(a, model_id):
    """Refuse to (re)start a run whose cumulative spend or minutes already reach its caps. Returns a message or None."""
    usd, mins = spend_estimate(a.log, model_id), (time.time() - first_start(a.log)) / 60
    if a.max_dollars and usd >= a.max_dollars:
        return f"cumulative spend estimate ${usd:.2f} already reaches the cap ${a.max_dollars}"
    if a.max_minutes and mins >= a.max_minutes:
        return f"{mins:.1f} min since this run's first start already reach the cap {a.max_minutes} min"
    return None


def logged_steps(log_dir):
    path = os.path.join(log_dir, "metrics.jsonl")
    return sum(1 for _ in open(path)) if os.path.exists(path) else 0


async def watchdog(a, model_id, train_task):
    """Mirror metrics.jsonl -> metrics.csv and enforce --max-dollars / --max-minutes.
    The dollar check is PROJECTED: as soon as a step is logged, the run is cancelled if one more step
    (costed at the most expensive step so far, +10 %) would cross the cap, so the cap is not overshot.
    The cookbook checkpoints every --save-every steps, so a cancelled run can be resumed or evaluated."""
    from scripts_common import jsonl_to_csv
    t0 = first_start(a.log)                     # cumulative across resumes
    n_seen, last_usd, step_max, step_secs, t_last = 0, 0.0, 0.0, 0.0, time.time()
    while not train_task.done():
        await asyncio.sleep(0.5 if a.dry_run else 5.0)
        n = logged_steps(a.log)
        if n == n_seen:
            continue
        jsonl_to_csv(a.log)
        usd, mins = spend_estimate(a.log, model_id), (time.time() - t0) / 60
        step_max = max(step_max, (usd - last_usd) / max(n - n_seen, 1))
        step_secs = max(step_secs, (time.time() - t_last) / max(n - n_seen, 1))
        n_seen, last_usd, t_last = n, usd, time.time()
        with open(os.path.join(a.log, "spend.txt"), "w") as f:
            f.write(f"estimated_usd={usd:.2f} minutes={mins:.1f} max_step_usd={step_max:.2f}\n")
        over_usd = a.max_dollars and usd + 1.1 * step_max >= a.max_dollars
        over_min = a.max_minutes and mins + step_secs / 60 >= a.max_minutes
        if a.collapse_guard:                      # stop when the in-run eval shows over-commitment
            ev = eval_rows(a.log)
            if len(ev) >= 2 and ev[0][0] == 0 and ev[-1][0] >= a.collapse_guard:
                e0, e1 = ev[0], ev[-1]
                if e1[4] > e0[4] + 0.08:
                    msg = (f"COLLAPSE GUARD at step {e1[0]}: commits with posterior probability <= 0.5 rose from {e0[4]:.2f} to {e1[4]:.2f} "
                           f"(more than +0.08) on the in-run eval. Stopping; use the best earlier checkpoint.")
                    print("\n*** " + msg + " ***", flush=True)
                    open(os.path.join(a.log, "collapse_guard.txt"), "w").write(msg + "\n")
                    train_task.cancel()
                    break
        if a.futility_step and not os.path.exists(os.path.join(a.log, "futility_check.txt")):
            fv = futility_verdict(eval_es_rows(a.log), a.futility_step)
            if fv is not None:
                msg = (f"FUTILITY CHECK at step {fv['step']}: selection-set expected score {fv['step0_es']:+.3f} (step 0) -> {fv['step_es']:+.3f} "
                       f"(delta {fv['delta']:+.3f}) => {fv['verdict']} (rule: stop unless strictly above step 0)")
                print("\n*** " + msg + " ***", flush=True)
                open(os.path.join(a.log, "futility_check.txt"), "w").write(json.dumps(fv) + "\n")
                if fv["verdict"] == "STOP":
                    open(os.path.join(a.log, "futility_stop.txt"), "w").write(msg + "\n")
                    train_task.cancel()
                    break
        if a.pilot_stop_step:
            pv = pilot_verdict(eval_rows(a.log), a.pilot_stop_step)
            if pv is not None:
                msg = (f"PILOT CHECK at step {a.pilot_stop_step}: eval reward (std cost) {pv['step0_reward']:+.3f} -> {pv['step_reward']:+.3f} "
                       f"(delta {pv['delta']:+.3f}), invalid {pv['invalid']:.2f}  =>  {pv['verdict']} (rule: abort if delta < -0.2 or invalid > 0.3)")
                print("\n*** " + msg + " ***", flush=True)
                open(os.path.join(a.log, "pilot_verdict.txt"), "w").write(msg + "\n")
                ck = os.path.join(a.log, "checkpoints.jsonl")           # cancel as soon as the step's checkpoint exists
                for _ in range(2 if a.dry_run else 45):
                    if os.path.exists(ck) and any(json.loads(l).get("batch", -1) >= a.pilot_stop_step for l in open(ck) if l.strip()):
                        break
                    await asyncio.sleep(1.0)
                train_task.cancel()
                break
        if over_usd or over_min:
            print(f"\n*** GUARD: est. ${usd:.2f} after {mins:.1f} min; one more step (~${step_max:.2f}, ~{step_secs:.0f} s) would cross "
                  f"the cap (${a.max_dollars}, {a.max_minutes} min). Cancelling. Last checkpoint: {a.log}/checkpoints.jsonl ***", flush=True)
            train_task.cancel()
            break
    jsonl_to_csv(a.log)


async def run(a, model_id, renderer_name):
    from tinker_cookbook.rl import train
    lr = a.lr or default_lr(model_id)
    restore = None
    if a.dry_run:
        from hidden_planet import fake_tinker
        restore = fake_tinker.install(model_id, renderer_name, seed=a.seed, skill0=0.25, skill_steps=max(1, a.steps - 1))
    kl = {}
    if a.kl_coef > 0:
        ref = a.kl_ref
        if ref and ref.endswith(".json"):
            ref = json.load(open(ref))["sampler_path"]
        kl = dict(kl_penalty_coef=a.kl_coef, kl_reference_config=train.KLReferenceConfig(base_model=model_id, load_checkpoint_path=ref))
    cfg = train.Config(model_name=model_id, recipe_name="hidden_planet_rl", renderer_name=renderer_name, log_path=a.log,
                       dataset_builder=make_builder(a, model_id, renderer_name), learning_rate=lr, max_tokens=a.max_tokens,
                       eval_every=a.eval_every, save_every=a.save_every, lora_rank=a.rank, loss_fn=a.loss,
                       max_steps=a.steps, load_checkpoint_path=a.load_checkpoint,
                       rollout_error_tolerance=True, **kl)   # retry a failed trajectory instead of crashing the run
    print(f"{'DRY RUN (fake service, no credits) ' if a.dry_run else ''}model {model_id} renderer {renderer_name} lr {lr:.2e} "
          f"rank {a.rank} steps {a.steps} batch {a.batch}x{a.group} init {a.load_checkpoint or 'base'} guard ${a.max_dollars}/{a.max_minutes}min -> {a.log}", flush=True)
    msg = over_cap_at_start(a, model_id)
    if msg:
        print(f"*** GUARD at start: {msg}. Not training (caps are cumulative across resumes). ***", flush=True)
        if restore:
            restore()
        return
    task = asyncio.create_task(train.main(cfg))
    wd = asyncio.create_task(watchdog(a, model_id, task))
    try:
        await task
    except asyncio.CancelledError:
        print("training cancelled by guard", flush=True)
    finally:
        await wd
        if restore:
            from hidden_planet import fake_tinker
            u = fake_tinker.FakeServiceClient.last
            print("fake-service usage:", u.usage, "checkpoints:", u.checkpoints)
            restore()
    print(f"metrics: {a.log}/metrics.csv   estimated spend: ${spend_estimate(a.log, model_id):.2f}")


if __name__ == "__main__":
    a = parse_args()
    if a.rv_availability:
        from hidden_planet.tasks import set_rv_mode
        set_rv_mode(a.rv_availability)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    model_id, renderer = resolve_model(a)
    os.makedirs(a.log, exist_ok=True)
    if a.load_checkpoint and a.load_checkpoint.endswith(".json"):
        a.load_checkpoint = json.load(open(a.load_checkpoint))["state_path"]
    if a.load_checkpoint and "sampler" in a.load_checkpoint and not a.dry_run:
        sys.exit("--init-from needs the STATE path (trainable weights), not the sampler path; see <sft log>/sft.json")
    if not a.dry_run and "TINKER_API_KEY" not in os.environ:
        sys.exit("TINKER_API_KEY is not set in the environment. Use --dry-run to test without credits.")
    asyncio.run(run(a, model_id, renderer or "qwen3_5_disable_thinking"))
