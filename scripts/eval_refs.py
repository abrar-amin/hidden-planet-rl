"""Scripted references on EXACTLY the systems a model eval used, through the text interface (free, local).
Writes per-episode files in the same format as eval_zero_shot.py, so paired_compare.py and decision_quality.py
can compare a model against the teacher or EIG system by system.

usage: eval_refs.py --start 2000 --n 96 --out-prefix runs/x/refs_paired [--agents teacher eig compact random]
       eval_refs.py --eval-set sealed --out-prefix runs/x/refs_sealed          (run this only at the final step)
"""
import os, sys, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")
from hidden_planet.tasks import sample_system
from hidden_planet.text_env import HiddenPlanetSession
from hidden_planet.text_policies import fit_following_agent, eig_text_agent, compact_text_agent, random_text_agent, lite_teacher_agent, fixed_fit_agent

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=2000); ap.add_argument("--n", type=int, default=96); ap.add_argument("--eval-set", default="fixed", choices=["fixed", "sealed"])
ap.add_argument("--sealed-arm", default=None); ap.add_argument("--agents", nargs="+", default=["teacher", "eig", "compact", "random"]); ap.add_argument("--out-prefix", required=True)
ap.add_argument("--policy-seeds", type=int, default=1, help="play each randomised script K times per system (policy RNG offsets 17, 1017, ...); rows get sample=k and paired comparisons average them per system");
ap.add_argument("--reward-mode", default="strict0"); ap.add_argument("--wrong-penalty", type=float, default=1.0); ap.add_argument("--cost-per-unit", type=float, default=0.008)
a = ap.parse_args()
from hidden_planet import sealed as _sealed
if a.eval_set == "sealed":
    a.start, a.n = _sealed.SEALED_START, _sealed.SEALED_N
    _sealed.claim(a.sealed_arm)
elif _sealed.overlaps(a.start, a.n):
    sys.exit("these seeds overlap the sealed set; use --eval-set sealed --sealed-arm <arm>")
kw = dict(reward_mode=a.reward_mode, wrong_penalty=a.wrong_penalty, cost_per_unit=a.cost_per_unit)


def play(agent, seed, k=0):
    s = HiddenPlanetSession(sample_system(seed, 5, "test"), **kw, tool_level="lite" if agent == "lite_teacher" else "oracle")
    rng = np.random.default_rng(seed + 17 + 1000 * k)
    if agent in ("always_none", "always_cannot"):
        s.call("submit_answer", {"verdict": "none" if agent == "always_none" else "cannot_determine"})
        r, m = s.grade()
        return r, m, []
    gen = {"teacher": lambda: fit_following_agent(s.user_prompt(), rng, max_turns=s.max_turns, mode=a.reward_mode, wrong_penalty=a.wrong_penalty),
           "eig": lambda: eig_text_agent(s.user_prompt(), rng, max_turns=s.max_turns, mode=a.reward_mode, wrong_penalty=a.wrong_penalty),
           "compact": lambda: compact_text_agent(s.user_prompt(), rng), "compact_smart": lambda: compact_text_agent(s.user_prompt(), rng, smart=True),
           "lite_teacher": lambda: lite_teacher_agent(s.user_prompt(), rng, max_turns=s.max_turns),
           "fixed": lambda: fixed_fit_agent(s.user_prompt(), rng, max_turns=s.max_turns),
           "fixed_diverse": lambda: fixed_fit_agent(s.user_prompt(), rng, max_turns=s.max_turns, diverse=True),   # the script that produced the SFT data (14-18 transits, 7-9 RVs)
           "random": lambda: random_text_agent(s.user_prompt(), rng)}[agent]()
    steps = []
    step = next(gen)
    while True:
        res = [s.call(nm, ar)[0] for nm, ar in step[1]]
        logs = {}
        for i, ((nm, ar), r) in enumerate(zip(step[1], res)):
            logs[f"tool_call_{i}"] = f"{nm}({json.dumps(ar)})"; logs[f"tool_result_{i}"] = r
        steps.append(dict(logs=logs, metrics={}))
        if s.done:
            break
        try:
            step = gen.send([json.loads(r) for r in res])
        except (StopIteration, KeyError, IndexError, TypeError):
            break
    r, m = s.grade()
    if steps:
        steps[-1]["metrics"] = m
    return r, m, steps


summary = {}
for agent in a.agents + ["always_none", "always_cannot"]:
    rows, traces = [], []
    K = 1 if agent in ("always_none", "always_cannot") else a.policy_seeds      # deterministic: one play is exact
    for i in range(a.n):
        for k in range(K):
            r, m, steps = play(agent, a.start + i, k)
            rows.append(dict(m, reward=float(r), sample=k, seed=a.start + i, turns=len(steps)))
            traces.append(dict(group_idx=i, traj_idx=k, seed=a.start + i, steps=steps))
    with open(f"{a.out_prefix}_{agent}_episodes.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in rows)
    with open(f"{a.out_prefix}_{agent}_traces.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in traces)
    x = np.array([r["reward"] for r in rows])
    from hidden_planet.expected_score import es_from_metrics
    summary[agent] = dict(es=float(np.mean([es_from_metrics(r, a.cost_per_unit) for r in rows])), reward=float(x.mean()), ci=float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0, correct=float(np.mean([r.get("correct", 0) for r in rows])),
                          spent=float(np.mean([r.get("spent", 0) for r in rows])), cannot=float(np.mean([r.get("verdict_cannot", 0) for r in rows])),
                          rv_buy_rate=float(np.mean([r.get("rv_buy_rate", 0) for r in rows])), n=len(rows))
    print(f"{agent:14s} ES {summary[agent]['es']:+.3f}  reward {summary[agent]['reward']:+.3f} ± {summary[agent]['ci']:.3f}  correct {summary[agent]['correct']:.2f}  spent {summary[agent]['spent']:5.1f}  cannot {summary[agent]['cannot']:.2f}")
summary["_meta"] = dict(first_seed=a.start, n=a.n, policy_seeds=a.policy_seeds, eval_set=a.eval_set, reward_mode=a.reward_mode, cost=a.cost_per_unit, level="hard",
                        frac_no_planet=float(np.mean([not sample_system(a.start + i, 5, "test").has_planet for i in range(a.n)])))
json.dump(summary, open(f"{a.out_prefix}_summary.json", "w"), indent=1)
print(summary["_meta"])
