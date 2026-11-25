"""Offline headroom of TOOL_LEVEL=lite (free, local). Hard level, strict0, cost 0.008.

1. Tune the lite teacher's delta-chi2 thresholds on TRAIN seeds only (grid).
2. On the DEV set (test seeds 3000-3095) report, paired with cluster-bootstrap CIs:
   a  lite teacher (tuned thresholds)
   b  same purchases, final decision by a logistic rule on the lite statistics (fitted on train seeds)
   c  same purchases, final decision by the exact posterior (commit the fit's top option iff its probability > 0.5) -- the
      best achievable decision at those data states; c1 must submit the lite tool's numbers, c2 may submit the posterior's
   d  oracle-level references on the same systems: fit-teacher and EIG through the text tools (from results/dev)
usage: lite_headroom.py [N_TRAIN=300]"""
import os, sys, json, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")
from multiprocessing import Pool
from scipy.optimize import minimize

COST = 0.008
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TRAIN0, DEV0, NDEV = 200000, 3000, 96


def play(args):
    seed, split, thr = args
    from hidden_planet.text_env import HiddenPlanetSession
    from hidden_planet.text_policies import lite_teacher_agent
    from hidden_planet.tasks import sample_system
    from hidden_planet.inference import Posterior, Prior
    from hidden_planet.reward import score
    s = HiddenPlanetSession(sample_system(seed, 5, split), reward_mode="strict0", wrong_penalty=1.0, cost_per_unit=COST, tool_level="lite")
    gen = lite_teacher_agent(s.user_prompt(), np.random.default_rng(seed + 17), max_turns=s.max_turns, thr=thr)
    step = next(gen); lite = None
    while True:
        if step[1][0][0] == "submit_answer":
            ans = dict(step[1][0][1]); break
        res = [json.loads(s.call(nm, ar)[0]) for nm, ar in step[1]]
        if len(res) > 1 and "aliases" in res[1]:
            lite = res[1]
        step = gen.send(res)
    sp, spent = s.sp, s.ep.spent
    sc = lambda a: score(sp, a, spent, 1, 1, lam=COST, mode="strict0", wrong_penalty=1.0)["reward"]
    rec = dict(seed=seed, spent=spent, teacher=sc(ans), teacher_verdict=ans["verdict"], n_rv=len(s.ep.rvs), n_tr=len(s.ep.transits))
    a0 = lite["aliases"][0]
    best_lite = {"verdict": "planet", "period_days": a0["period_days"], "mass_earth": a0["mass_earth"]}
    rec.update(r_planet=sc(best_lite), r_none=sc({"verdict": "none"}), r_cannot=sc({"verdict": "cannot_determine"}))
    snr = (a0["K2_ms"] / a0["K2_sigma_ms"]) if a0.get("K2_sigma_ms") else 0.0
    rec["feat"] = [np.log1p(max(a0["delta_chi2_vs_none"], 0)), np.log1p(max(lite.get("delta_chi2_between_aliases", 0.0), 0)), min(snr, 15.0), len(s.ep.rvs) / 10.0,
                   abs(np.log(max(a0.get("mass_from_rv", a0["mass_if_circular"]), 0.1) / max(a0["mass_if_circular"], 0.1))), np.log(sp.sig_rv), np.log(sp.sig_t * 1440)]
    post = Posterior(Prior.from_system(sp), *s.ep.data()); su = post.summary()
    p_lite = post.prob_answer(a0["period_days"], a0["mass_earth"])[1]                 # posterior probability that the lite numbers are right
    # c1: oracle decision, lite numbers
    opts = [(p_lite, rec["r_planet"]), (su["p_none"], rec["r_none"])]
    pbest, rbest = max(opts)
    rec["oracle_lite_numbers"] = rbest if pbest > 0.5 else rec["r_cannot"]
    # c2: oracle decision, oracle numbers
    r_or_pl = sc({"verdict": "planet", "period_days": su["period"], "mass_earth": su["mass"]})
    pbest, rbest = max([(su["p_both"], r_or_pl), (su["p_none"], rec["r_none"])])
    rec["oracle_full"] = rbest if pbest > 0.5 else rec["r_cannot"]
    rec["p_lite"], rec["p_none"] = float(p_lite), float(su["p_none"])
    return rec


def run(seeds, split, thr, pool):
    return pool.map(play, [(s, split, thr) for s in seeds], chunksize=8)


def fit_logistic(X, y):
    X1 = np.c_[np.ones(len(X)), X]
    f = lambda w: np.mean(np.log1p(np.exp(-(2 * y - 1) * (X1 @ w)))) + 1e-3 * np.sum(w[1:] ** 2)
    w = minimize(f, np.zeros(X1.shape[1]), method="BFGS").x
    return lambda Z: 1 / (1 + np.exp(-(np.c_[np.ones(len(Z)), Z] @ w)))


def ci(d, B=20000):
    d = np.asarray(d, float); m = d[np.random.default_rng(0).integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), *map(float, np.percentile(m, [2.5, 97.5]))


if __name__ == "__main__":
    train = list(range(TRAIN0, TRAIN0 + N_TRAIN)); dev = list(range(DEV0, DEV0 + NDEV))
    with Pool(4) as pool:
        grid = [dict(det=d, sep=s, ksnr=k, sep_low=sl) for d, s, k, sl in itertools.product((20.0, 30.0), (6.0, 9.0, 14.0), (2.0, 3.0, 4.0), (4.0, 6.0)) if sl <= s]
        best = None
        for thr in grid:
            m = float(np.mean([r["teacher"] for r in run(train, "train", thr, pool)]))
            if best is None or m > best[0]:
                best = (m, thr)
        thr = best[1]
        print(f"thresholds tuned on {N_TRAIN} train seeds ({len(grid)} settings): {thr}  (train reward {best[0]:+.3f})")
        tr = run(train, "train", thr, pool); dv = run(dev, "test", thr, pool)
    # logistic decision rule on the lite statistics, fitted on train
    X = np.array([r["feat"] for r in tr]); Xd = np.array([r["feat"] for r in dv])
    p_pl = fit_logistic(X, np.array([r["r_planet"] > 0.5 - COST * 60 for r in tr], float))      # 'planet with lite numbers' would be right
    p_no = fit_logistic(X, np.array([r["r_none"] > 0.5 - COST * 60 for r in tr], float))
    pp, pn = p_pl(Xd), p_no(Xd)
    logi = [(r["r_planet"] if (a >= b and a > 0.5) else (r["r_none"] if (b > a and b > 0.5) else r["r_cannot"])) for r, a, b in zip(dv, pp, pn)]
    T = np.array([r["teacher"] for r in dv])
    out = dict(thresholds=thr, n_train=N_TRAIN, n_dev=NDEV)
    refs = {}
    for k in ("teacher", "eig", "compact"):
        p = f"results/dev/refs_dev_{k}_episodes.jsonl"
        if os.path.exists(p):
            refs[k] = np.array([json.loads(l)["reward"] for l in open(p)])
    print(f"\nDEV set (test seeds {DEV0}-{DEV0 + NDEV - 1}), strict0, cost {COST}:")
    rows = [("a  lite teacher (threshold rule)", T), ("b  same purchases + logistic decision on lite statistics", np.array(logi)),
            ("c1 same purchases + exact-posterior decision, lite numbers", np.array([r["oracle_lite_numbers"] for r in dv])),
            ("c2 same purchases + exact-posterior decision and numbers", np.array([r["oracle_full"] for r in dv]))]
    rows += [(f"d  oracle-level {k} (text tools, own purchases)", v) for k, v in refs.items()]
    for name, v in rows:
        m, lo, hi = ci(v); d, dlo, dhi = ci(v - T)
        print(f"  {name:62s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]   minus lite teacher: {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
        out[name[:2].strip()+ ("_" + name.split()[2] if name.startswith("d") else "")] = dict(reward=m, ci=[lo, hi], minus_lite_teacher=[d, dlo, dhi])
    v = [r["teacher_verdict"] for r in dv]
    print(f"  lite teacher on dev: planet {v.count('planet')}, none {v.count('none')}, cannot {v.count('cannot_determine')}; spent {np.mean([r['spent'] for r in dv]):.1f}; "
          f"mean posterior probability of its planet answers {np.mean([r['p_lite'] for r in dv if r['teacher_verdict'] == 'planet']):.2f}")
    json.dump(out, open("results/lite_headroom.json", "w"), indent=1)
