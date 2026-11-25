"""Real scripted teacher vs fixed demonstrator through the real text env: how stable is the +0.18 dev gap across policy RNG seeds and system sets?"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")
from multiprocessing import Pool
from hidden_planet.tasks import sample_system
from hidden_planet.text_env import HiddenPlanetSession
from hidden_planet.text_policies import fit_following_agent, fixed_fit_agent

def play(arg):
    agent, seed, split, off = arg
    s = HiddenPlanetSession(sample_system(seed, 5, split), reward_mode="strict0", wrong_penalty=1.0, cost_per_unit=0.008)
    rng = np.random.default_rng(seed + off)
    gen = (fit_following_agent(s.user_prompt(), rng, max_turns=s.max_turns, mode="strict0", wrong_penalty=1.0) if agent == "teacher"
           else fixed_fit_agent(s.user_prompt(), rng, max_turns=s.max_turns))
    step = next(gen)
    while True:
        res = [s.call(nm, ar)[0] for nm, ar in step[1]]
        if s.done: break
        try: step = gen.send([json.loads(r) for r in res])
        except (StopIteration, KeyError, IndexError, TypeError): break
    r, m = s.grade()
    return agent, seed, split, off, float(r), m["spent"]

if __name__ == "__main__":
    sets = {"test 2000-2095 + 4000-4047 (old confirm + selection sets)": [(s, "test") for s in list(range(2000, 2096)) + list(range(4000, 4048))], "more fresh train 500260-500499": [(s, "train") for s in range(500260, 500500)]}
    offs = [17, 1017]
    jobs = [(a, s, sp, o) for nm, L in sets.items() for (s, sp) in L for a in ("teacher", "fixed") for o in offs]
    with Pool(8) as p: out = p.map(play, jobs, chunksize=8)
    R = {(a, s, sp, o): r for a, s, sp, o, r, _ in out}
    for nm, L in sets.items():
        for o in offs:
            t = np.array([R[("teacher", s, sp, o)] for s, sp in L]); f = np.array([R[("fixed", s, sp, o)] for s, sp in L]); d = t - f
            print(f"{nm:60s} policy-rng offset {o:5d}: teacher {t.mean():+.3f} fixed {f.mean():+.3f} gap {d.mean():+.3f} +- {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f}")
        t = np.mean([[R[("teacher", s, sp, o)] for s, sp in L] for o in offs], 0); f = np.mean([[R[("fixed", s, sp, o)] for s, sp in L] for o in offs], 0); d = t - f
        print(f"{nm:60s} mean of policy seeds: teacher {t.mean():+.3f} fixed {f.mean():+.3f} gap {d.mean():+.3f} +- {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.3f}")
