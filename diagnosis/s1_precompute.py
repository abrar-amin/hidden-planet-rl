"""Precompute, with the REAL environment (HiddenPlanetSession: TTVFast truth, real fit_model, real strict0 grader),
the outcome of every node of a small purchase lattice for many systems, so that policy-gradient experiments are table lookups.
Node = (opening n0 in {10,16}, extra transit batches of 8: 0..4, RV batches of 4: 0..4) within the budget of 60.
The j-th extra transit / j-th RV of a system is always the same epoch / time, so the data depend only on the counts.
Stored per node: p_none, p_both (as printed by fit_model), score if 'planet' is submitted with the fit's numbers, score if 'none', spent."""
import os, sys, json, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")
from multiprocessing import Pool
from hidden_planet.tasks import sample_system, NMAX, BUDGET
from hidden_planet.text_env import HiddenPlanetSession
from hidden_planet.reward import score

OPEN = (10, 16); TB, RB = 8, 4; MAXB = 4

def nodes():
    return [(o, a, b) for o in OPEN for a in range(MAXB + 1) for b in range(MAXB + 1) if o + TB * a + 3 * RB * b <= BUDGET]

def one(arg):
    seed, split = arg
    sp = sample_system(seed, 5, split)
    rng = np.random.default_rng(seed + 17)
    out = {}
    opens = {o: [int(x) for x in np.unique(np.clip(np.linspace(0, NMAX, o).round().astype(int) + rng.integers(-4, 5, o), 0, NMAX))] for o in OPEN}
    perm = [int(x) for x in rng.permutation(NMAX + 1)]
    rvt = [round(float(t), 2) for t in rng.uniform(0, NMAX * sp.P_cat, 4 * MAXB)]
    for (o, a, b) in nodes():
        s = HiddenPlanetSession(sp, reward_mode="strict0", wrong_penalty=1.0, cost_per_unit=0.008)
        s.call("observe_transits", {"epochs": opens[o]})
        extra = [n for n in perm if n not in opens[o]][:TB * a]
        for i in range(0, len(extra), 16): s.call("observe_transits", {"epochs": extra[i:i + 16]})
        if b: s.call("observe_rvs", {"times_days": rvt[:RB * b]})
        f = json.loads(s.call("fit_model", {})[0])
        if "best_perturber" not in f:
            out[(o, a, b)] = None; continue
        bp = f["best_perturber"]
        sc_p = score(sp, {"verdict": "planet", "period_days": bp["period_days"], "mass_earth": bp["mass_earth"]}, 0, 1, 1, mode="strict0", wrong_penalty=1.0)["score"]
        sc_n = score(sp, {"verdict": "none"}, 0, 1, 1, mode="strict0", wrong_penalty=1.0)["score"]
        out[(o, a, b)] = (f["p_no_perturber"], bp["p_both_right"], sc_p, sc_n, float(s.ep.spent))
    return (split, seed), dict(nodes=out, has_planet=bool(sp.has_planet))

if __name__ == "__main__":
    amb = json.load(open("data/curriculum_hard.json"))["ambiguous"][:320]
    jobs = [(s, "test") for s in range(3000, 3192)] + [(s, "train") for s in amb] + [(s, "train") for s in range(500000, 500260)]
    with Pool(8) as p:
        res = {}
        for i, (k, v) in enumerate(p.imap_unordered(one, jobs, chunksize=2)):
            res[k] = v
            if i % 50 == 0: print("done", i, flush=True)
    pickle.dump(dict(res=res, amb=amb, fresh=list(range(500000, 500260)), dev=list(range(3000, 3192))), open("diagnosis/lattice.pkl", "wb"))
    print("systems", len(res), "nodes/system", len(nodes()))
