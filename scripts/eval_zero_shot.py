"""Evaluate a model (base = zero-shot, or a trained checkpoint) on a fixed held-out set.

Real (spends credits; TINKER_API_KEY must be in the environment):
  python scripts/eval_zero_shot.py --model Qwen/Qwen3.5-9B --level easy --n 32 --out results/zero_shot_easy.json
  python scripts/eval_zero_shot.py --model Qwen/Qwen3.5-9B --checkpoint tinker://.../sampler/final --level hard
Dry run (fake service, no credits):
  python scripts/eval_zero_shot.py --dry-run --level easy --n 8 --show 1
Out-of-class table:  add --ooc-kind drift|activity|two_planets --allow-other
"""
import os, sys, json, argparse, asyncio, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import tinker
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.completers import TinkerTokenCompleter
from hidden_planet.tasks import LEVEL_NAMES
from hidden_planet import rollout_utils as ru
from train_rl import MODELS, PRICES

from hidden_planet import sealed as _sealed
SEALED_START, SEALED_N = _sealed.SEALED_START, _sealed.SEALED_N


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B"); ap.add_argument("--renderer", default=None)
    ap.add_argument("--thinking", action="store_true"); ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--level", default="easy", choices=list(LEVEL_NAMES)); ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--eval-set", default="fixed", choices=["fixed", "sealed"], help="sealed: reserved for the final confirmation; needs --sealed-arm and respects the one-use lock"); ap.add_argument("--sealed-arm", default=None); ap.add_argument("--max-tokens", type=int, default=512, help="per-turn generation cap")
    ap.add_argument("--ooc-kind", default=None, choices=[None, "two_planets", "activity", "drift"])
    ap.add_argument("--allow-other", action="store_true"); ap.add_argument("--cost-per-unit", type=float, default=None); ap.add_argument("--reward-mode", default="partial"); ap.add_argument("--tool-level", default="oracle", choices=["oracle", "lite"]); ap.add_argument("--wrong-penalty", type=float, default=None); ap.add_argument("--ooc-mix", type=float, default=0.0); ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--show", type=int, default=0); ap.add_argument("--out", default=None)
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature; 0 = greedy (identical policy -> identical episodes)")
    ap.add_argument("--samples", type=int, default=1, help="rollouts per system (same system, same noise)"); ap.add_argument("--episodes-out", default=None, help="per-episode JSONL for paired_compare.py")
    ap.add_argument("--traces-out", default=None, help="per-episode step logs (tool calls and results) for decision_quality.py")
    ap.add_argument("--replay", type=int, default=0, help="archival Kepler replay: this many anonymised windows per target (5 targets)")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--mock-skill", type=float, default=0.3)
    ap.add_argument("--rv-availability", default=None, choices=["ideal", "realistic"])
    a = ap.parse_args()
    if a.rv_availability:
        from hidden_planet.tasks import set_rv_mode
        set_rv_mode(a.rv_availability)
    if a.eval_set == "sealed":
        if a.dry_run:      # dry runs never generate the sealed systems (event.sh uses seeds 9000+)
            sys.exit("--dry-run with --eval-set sealed is refused: dry runs must not generate the sealed systems")
        a.start, a.n = SEALED_START, SEALED_N
        _sealed.claim(a.sealed_arm)
    elif _sealed.overlaps(a.start, a.n):
        sys.exit("these seeds overlap the sealed set; use --eval-set sealed --sealed-arm <arm> (never in a dry run)")
    mid, r_off, r_on = MODELS.get(a.model, next((v for v in MODELS.values() if v[0] == a.model), (a.model, None, None)))
    renderer = a.renderer or (r_on if a.thinking else r_off) or "qwen3_5_disable_thinking"
    if a.dry_run:
        from hidden_planet import fake_tinker
        fake_tinker.install(mid, renderer, skill0=a.mock_skill)
    elif "TINKER_API_KEY" not in os.environ:
        sys.exit("TINKER_API_KEY is not set in the environment. Use --dry-run to test without credits.")
    service = tinker.ServiceClient()
    sc = service.create_sampling_client(model_path=a.checkpoint) if a.checkpoint else service.create_sampling_client(base_model=mid)
    if hasattr(sc, "result_async"):
        sc = await sc.result_async()
    elif hasattr(sc, "__await__"):
        sc = await sc
    policy = TinkerTokenCompleter(sc, max_tokens=a.max_tokens, temperature=a.temperature)
    level = LEVEL_NAMES["hard"] if a.ooc_kind else LEVEL_NAMES[a.level]
    if a.replay > 0:
        from hidden_planet.replay import TARGETS
        from hidden_planet.tinker_env import ReplayEnvGroupBuilder
        builders = [ReplayEnvGroupBuilder(k, sd, mid, renderer, a.allow_other, a.cost_per_unit, a.reward_mode, a.wrong_penalty) for k in TARGETS for sd in range(a.replay)]
    elif a.ooc_mix > 0:                      # same fixed out-of-class mixture as train_rl.py --ooc-mix uses for its eval set
        from hidden_planet.tinker_env import HiddenPlanetDataset
        builders = HiddenPlanetDataset(1, a.n, 1, "test", mid, renderer, 0, "hard", level, allow_other=True, ooc_mix=a.ooc_mix,
                                       cost_per_unit=a.cost_per_unit, reward_mode=a.reward_mode, wrong_penalty=a.wrong_penalty).get_batch(0)
    else:
        builders = ru.eval_builders(level, a.n, mid, renderer, "test", a.start, a.samples, allow_other=a.allow_other, ooc_kind=a.ooc_kind,
                                    cost_per_unit=a.cost_per_unit, reward_mode=a.reward_mode, wrong_penalty=a.wrong_penalty, tool_level=a.tool_level)
    t0 = time.time()
    groups = await ru.rollout_builders(builders, policy, a.concurrency)
    s = ru.summarize_groups([g for g in groups if g is not None])
    if a.out:                               # always keep per-episode results and transcripts next to the summary
        base = a.out[:-5] if a.out.endswith(".json") else a.out
        a.episodes_out = a.episodes_out or base + "_episodes.jsonl"
        a.traces_out = a.traces_out or base + "_traces.jsonl"
    if a.episodes_out:
        with open(a.episodes_out, "w") as f:
            for row in ru.episode_rows(groups, builders):
                f.write(json.dumps(row) + "\n")
    if a.traces_out:
        with open(a.traces_out, "w") as f:
            for gi, g in enumerate(groups):
                if g is None:
                    continue
                for si, traj in enumerate(g.trajectories_G):
                    f.write(json.dumps(dict(group_idx=gi, traj_idx=si, seed=getattr(builders[gi], "seed", gi),
                                            steps=[dict(logs={k: str(v) for k, v in (tr.logs or {}).items()},
                                                        metrics={k: v for k, v in (tr.metrics or {}).items() if isinstance(v, (int, float))}) for tr in traj.transitions])) + "\n")
    if a.replay > 0:                        # per-target breakdown
        per = {}
        for b, g in zip(builders, groups):
            per.setdefault(b.koi, []).append(g)
        s["per_target"] = {k: {x: round(v, 3) for x, v in ru.summarize_groups([g for g in gs if g is not None]).items()
                               if x in ("reward", "correct", "spent", "verdict_none", "verdict_planet", "verdict_cannot", "verdict_other", "invalid")} for k, gs in per.items()}
    s.update(temperature=a.temperature, samples=a.samples)
    s.update(eval_set=a.eval_set, first_seed=a.start, reward_mode=a.reward_mode, cost_per_unit=a.cost_per_unit)
    s.update(model=mid, checkpoint=a.checkpoint, renderer=renderer, level=a.level, ooc_kind=a.ooc_kind, allow_other=a.allow_other,
             dry_run=a.dry_run, seconds=time.time() - t0)
    pre, prc, smp, _ = PRICES.get(mid, (0.7, 0.14, 5.6, 4.2))
    s["usd_per_episode_uncached"] = (s["prefill_tokens"] * pre + s["sampled_tokens"] * smp) / 1e6
    s["usd_per_episode_cache_aware"] = ((s["prefill_tokens"] - s["cached_prefill_tokens"]) * pre + s["cached_prefill_tokens"] * prc + s["sampled_tokens"] * smp) / 1e6
    tok = tokenizer_utils.get_tokenizer(mid)
    for g in [g for g in groups if g is not None][:a.show]:
        tr = g.trajectories_G[0].transitions
        print("=" * 100); print(tok.decode(tr[-1].ob.to_ints()) + tok.decode(tr[-1].ac.tokens))
        print(f"--> reward {g.get_total_rewards()[0]:+.3f}")
    print(json.dumps(s, indent=1))
    if a.out:
        json.dump(s, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    asyncio.run(main())
