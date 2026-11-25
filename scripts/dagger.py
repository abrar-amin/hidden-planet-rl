"""DAgger round: roll out the CURRENT model on training systems, relabel every state
it visited with the teacher's action (teacher_decide: a function of the visible tool outputs only), and
fine-tune on those (state, teacher action) pairs.

  python scripts/dagger.py --init runs/confirm_strict0/sft/sft.json --n-systems 100 --steps 10 --batch 32 --log runs/dagger1
  python scripts/dagger.py --dry-run --init runs/confirm_dry/sft/sft.json --n-systems 6 --steps 2 --batch 8 --log runs/dagger_dry
Output: <log>/sft.json with the new state/sampler paths (usable as --init-from for train_rl.py, --init for another
round, or --checkpoint for eval_zero_shot.py), plus the model-teacher agreement rate per turn.
Expected live cost for the default sizes: rollouts 100 episodes ~ $0.4-0.5, training 10 x 32 datums of ~3k tokens ~ $1.4.
Exercised with --dry-run only; it has not been run against the live service."""
import os, sys, json, argparse, asyncio, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
np.seterr(all="ignore")
import tinker
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import get_renderer, TrainOnWhat
from tinker_cookbook.renderers.base import ToolCall
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from hidden_planet.tasks import sample_system
from hidden_planet.text_env import HiddenPlanetSession
from hidden_planet.text_policies import teacher_decide
from hidden_planet.tinker_env import build_env, _curriculum_pool
from train_rl import MODELS


def relabel(session, history, n_prefix, renderer, rng, mode, W):
    """(datums, n_turns, n_agree) for one model episode."""
    datums, turns, n_turns, n_agree = [], [], 0, 0
    i = n_prefix
    while i < len(history):
        m = history[i]
        if m["role"] != "assistant":
            i += 1
            continue
        thought, calls = teacher_decide(session.user_prompt(), turns, rng, session.max_turns, mode, W)
        tmsg = {"role": "assistant", "content": thought, "tool_calls": [
            ToolCall(function=ToolCall.FunctionBody(name=n, arguments=json.dumps(a)), id=f"dagger_{i}_{k}") for k, (n, a) in enumerate(calls)]}
        mi, w = renderer.build_supervised_example(history[:i] + [tmsg], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
        datums.append(datum_from_model_input_weights(mi, w, 16000))
        mine = [(c.function.name, json.loads(c.function.arguments)) for c in (m.get("tool_calls") or [])]
        key = lambda cs: tuple((n, a.get("verdict")) if n == "submit_answer" else (n, None) for n, a in cs)
        n_turns += 1; n_agree += key(mine) == key(calls)
        res = []
        j = i + 1
        while j < len(history) and history[j]["role"] == "tool":
            try:
                res.append(json.loads(history[j]["content"]))
            except Exception:
                res.append({})
            j += 1
        turns.append((mine, res))
        i = j
    return datums, n_turns, n_agree


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="sft.json (state_path + sampler_path) of the current model")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B"); ap.add_argument("--renderer", default=None)
    ap.add_argument("--n-systems", type=int, default=100); ap.add_argument("--start", type=int, default=300000, help="first TRAIN seed (or the curriculum pool with --ambiguous-frac)")
    ap.add_argument("--ambiguous-frac", type=float, default=0.6); ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=10); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--reward-mode", default="strict0"); ap.add_argument("--wrong-penalty", type=float, default=1.0); ap.add_argument("--cost-per-unit", type=float, default=0.008)
    ap.add_argument("--max-tokens", type=int, default=512); ap.add_argument("--log", default="runs/dagger"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    mid, r_off, _ = MODELS.get(a.model, next((v for v in MODELS.values() if v[0] == a.model), (a.model, None, None)))
    rname = a.renderer or r_off or "qwen3_5_disable_thinking"
    init = json.load(open(a.init))
    if a.dry_run:
        from hidden_planet import fake_tinker
        fake_tinker.install(mid, rname, skill0=0.6)
    elif "TINKER_API_KEY" not in os.environ:
        sys.exit("TINKER_API_KEY is not set in the environment. Use --dry-run to test without credits.")
    os.makedirs(a.log, exist_ok=True)
    tok = tokenizer_utils.get_tokenizer(mid); renderer = get_renderer(rname, tok)
    service = tinker.ServiceClient()
    sc = service.create_sampling_client(model_path=init["sampler_path"])
    if hasattr(sc, "result_async"):
        sc = await sc.result_async()
    policy = TinkerTokenCompleter(sc, max_tokens=a.max_tokens, temperature=a.temperature)
    pool = _curriculum_pool()
    rng = np.random.default_rng(0)
    seeds = [int(pool[(7 * i + 3) % len(pool)]) if (pool and rng.random() < a.ambiguous_frac) else a.start + i for i in range(a.n_systems)]

    async def one(seed):
        session = HiddenPlanetSession(sample_system(seed, 5, "train"), reward_mode=a.reward_mode, wrong_penalty=a.wrong_penalty, cost_per_unit=a.cost_per_unit)
        env = build_env(session, renderer)
        await env.initial_observation()
        n_prefix = len(env.message_env.history)
        env2 = build_env(session, renderer)            # fresh wrapper around the same (unused) session for the actual rollout
        try:
            await do_single_rollout(policy, env2)
        except Exception as e:
            print(f"rollout failed for seed {seed}: {type(e).__name__}", flush=True)
            return [], 0, 0, None
        d, nt, na = relabel(session, env2.message_env.history, n_prefix, renderer, np.random.default_rng(seed + 5), a.reward_mode, a.wrong_penalty)
        r, _ = session.grade()
        return d, nt, na, r

    t0 = time.time()
    sem = asyncio.Semaphore(32)

    async def guarded(s):
        async with sem:
            return await one(s)
    out = await asyncio.gather(*[guarded(s) for s in seeds])
    datums = [d for o in out for d in o[0]]
    nt, na = sum(o[1] for o in out), sum(o[2] for o in out)
    rew = [o[3] for o in out if o[3] is not None]
    print(f"rolled out {len(rew)} episodes in {time.time() - t0:.0f} s: mean reward {np.mean(rew):+.3f}; {nt} visited states relabelled; "
          f"model action agreed with the teacher in {na / max(nt, 1):.1%} of them; {len(datums)} datums, {sum(d.model_input.length for d in datums)} tokens")
    tc = service.create_training_client_from_state(init["state_path"])
    tc = (await tc.result_async()) if hasattr(tc, "result_async") else tc
    for step in range(a.steps):
        batch = [datums[i] for i in rng.choice(len(datums), min(a.batch, len(datums)), replace=False)]
        fb = tc.forward_backward(batch, loss_fn="cross_entropy"); op = tc.optim_step(tinker.AdamParams(learning_rate=a.lr, beta1=0.9, beta2=0.95, eps=1e-8))
        res = fb.result(); op.result()
        num = den = 0.0
        for o, d in zip(res.loss_fn_outputs, batch):
            lp = np.asarray(o["logprobs"].to_torch(), float); w = np.asarray(d.loss_fn_inputs["weights"].to_torch(), float)
            n = min(len(lp), len(w)); num += float(-(lp[:n] * w[:n]).sum()); den += float(w[:n].sum())
        print(f"dagger step {step:3d} nll/token {num / max(den, 1e-9):.4f} tokens {sum(d.model_input.length for d in batch)}", flush=True)
    st = tc.save_state("dagger_final").result(); sm = tc.save_weights_for_sampler("dagger_final").result()
    json.dump(dict(state_path=st.path, sampler_path=sm.path, parent=a.init, agreement=na / max(nt, 1), n_states=nt, model=mid, renderer=rname), open(os.path.join(a.log, "sft.json"), "w"))
    print("new state:", st.path, "\nnew sampler:", sm.path, f"\nwrote {a.log}/sft.json")

if __name__ == "__main__":
    asyncio.run(main())
