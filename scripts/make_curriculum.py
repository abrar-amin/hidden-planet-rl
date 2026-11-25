"""Curriculum pool for RL: TRAIN-split seeds whose first transit-only fit is ambiguous
(max(p_both_right, p_no_perturber) in [0.3, 0.8] after the teacher's opening 14 transits).
Only the train split is touched; eval seeds live in the test split.
usage: make_curriculum.py [N=2000] [START=100000]   -> data/curriculum_hard.json"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")
from multiprocessing import Pool
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
START = int(sys.argv[2]) if len(sys.argv) > 2 else 100000


def first_fit(seed):
    from hidden_planet.text_env import HiddenPlanetSession
    from hidden_planet.text_policies import fit_following_agent
    from hidden_planet.tasks import sample_system
    s = HiddenPlanetSession(sample_system(seed, 5, "train"))
    step = next(fit_following_agent(s.user_prompt(), np.random.default_rng(seed + 17), max_turns=s.max_turns))
    out = [json.loads(s.call(n, a)[0]) for n, a in step[1]]
    f = out[1]
    return seed, max(f["best_perturber"]["p_both_right"], f["p_no_perturber"]), bool(s.sp.has_planet)


if __name__ == "__main__":
    with Pool(4) as p:
        rows = p.map(first_fit, range(START, START + N), chunksize=25)
    amb = [s for s, pm, _ in rows if 0.3 <= pm <= 0.8]
    pm = np.array([r[1] for r in rows])
    out = dict(level=5, split="train", start=START, n=N, ambiguous=amb, frac_ambiguous=len(amb) / N,
               frac_below_0p3=float((pm < 0.3).mean()), frac_above_0p8=float((pm > 0.8).mean()),
               frac_ambiguous_with_planet=float(np.mean([h for s, p_, h in rows if 0.3 <= p_ <= 0.8])))
    json.dump(out, open("data/curriculum_hard.json", "w"))
    print({k: (v if k != "ambiguous" else f"{len(v)} seeds") for k, v in out.items()})
