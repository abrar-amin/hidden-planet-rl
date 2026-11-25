"""Shared rollout / metric helpers for train_rl.py and eval_zero_shot.py (real or mock sampler)."""
from __future__ import annotations

import asyncio
import csv
import os

import numpy as np
from tinker_cookbook.rl.rollouts import do_group_rollout

from .tinker_env import HiddenPlanetEnvGroupBuilder

KEYS = ("reward", "reward_std_cost", "score", "correct", "spent", "verdict_none", "verdict_planet", "verdict_cannot", "verdict_other", "invalid", "abstain_correct",
        "determinable", "has_planet", "n_transits", "n_rvs", "tool_errors", "rv_buy_rate", "submit_conf", "lowconf_planet", "lowconf_commit", "post_lowconf_commit", "post_conf", "committed", "used_fit_model")


def episode_rows(groups, builders=None):
    """One dict per episode (metrics, reward, system seed, sample index) for paired comparisons."""
    rows = []
    for gi, g in enumerate(groups):
        if g is None:
            continue
        for si, (traj, R) in enumerate(zip(g.trajectories_G, g.get_total_rewards())):
            m = {}
            for tr in traj.transitions:
                m.update({k: v for k, v in tr.metrics.items() if isinstance(v, (int, float))})
            if "score" not in m:                    # ended by the framework (context overflow, parse failure): counts as no answer everywhere
                m["invalid"] = 1.0
            m.update(reward=float(R), sample=si, seed=getattr(builders[gi], "seed", gi) if builders else gi, turns=len(traj.transitions))
            rows.append(m)
    return rows


def summarize_groups(groups):
    """Per-episode means over a list of TrajectoryGroup, plus token usage."""
    rows = []
    for g in groups:
        totals = g.get_total_rewards()
        for traj, R in zip(g.trajectories_G, totals):
            m = {}
            for tr in traj.transitions:
                m.update({k: v for k, v in tr.metrics.items() if isinstance(v, (int, float))})
            obs = [tr.ob.length for tr in traj.transitions]
            acs = [len(tr.ac.tokens) for tr in traj.transitions]
            ctx = [o + a for o, a in zip(obs, acs)]
            m.update(reward=float(R), turns=len(obs), prefill_tokens=float(sum(obs)), sampled_tokens=float(sum(acs)),
                     cached_prefill_tokens=float(sum(min(c, o) for c, o in zip([0] + ctx[:-1], obs))),
                     final_context_tokens=float(ctx[-1] if ctx else 0), answered=float("score" in m))
            m.setdefault("invalid", 1.0 if "score" not in m else m.get("invalid", 0.0))
            rows.append(m)
    out = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in KEYS + ("turns", "prefill_tokens", "sampled_tokens", "cached_prefill_tokens", "final_context_tokens", "answered")}
    r = np.array([x["reward"] for x in rows])
    out["reward_ci95"] = float(1.96 * r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0
    ab = np.array([x.get("verdict_cannot", 0.0) for x in rows]); det = np.array([x.get("determinable", 1.0) for x in rows])
    out["abstain_precision"] = float((1 - det[ab > 0]).mean()) if (ab > 0).any() else float("nan")
    out["n_episodes"] = len(rows)
    return out


async def rollout_builders(builders, policy, concurrency=64):
    sem = asyncio.Semaphore(concurrency)

    async def one(b):
        async with sem:
            try:
                return await do_group_rollout(b, policy)
            except Exception as e:                 # one bad rollout must not abort an eval
                print(f"rollout failed and was skipped: {type(e).__name__}: {str(e)[:200]}", flush=True)
                return None
    return await asyncio.gather(*[one(b) for b in builders])


def eval_builders(level, n, model_name, renderer_name, split="test", start=0, group_size=1, **kw):
    return [HiddenPlanetEnvGroupBuilder(start + i, level, split, group_size, model_name, renderer_name, **kw) for i in range(n)]


def append_csv(path, row: dict):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
