"""Run every scripted policy on a block of systems. One JSON line per system.

usage: run_baselines.py SPLIT START N OUT [--level L] [--procs K] [--backend analytic|nbody]
"""
import os, sys, json, time, argparse
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hidden_planet.tasks import sample_system, Episode, NMAX
from hidden_planet import policies as pol
from hidden_planet.reward import score

SIMPLE = dict(proto_uniform_transits=pol.proto_uniform_transits, proto_uniform_mixed=pol.proto_uniform_mixed,
              proto_adaptive=pol.proto_adaptive, bayes_uniform_transits=pol.uniform_transits_bayes,
              bayes_uniform_mixed=pol.uniform_mixed_bayes, bayes_rv_heavy=pol.rv_heavy_bayes)


def run_system(args):
    seed, split, level, backend = args
    np.seterr(all="ignore")
    t0 = time.time()
    try:
        sp = sample_system(seed, level, split)
        Episode(sp, backend=backend)       # may update sp.P2 (nbody) / raise if unstable
    except RuntimeError as e:
        return dict(seed=seed, split=split, error=str(e))
    qref = pol.reference_identifiability(sp, backend)
    rec = dict(seed=seed, split=split, level=level, backend=backend, has=sp.has_planet, q_ref=qref,
               truth=dict(P2=sp.P2, m2=sp.m2, j=sp.j, delta=sp.delta, sig_rv=sp.sig_rv, sig_t=sp.sig_t, rv=sp.extra.get("rv")), pol={})
    for name, fn in SIMPLE.items():
        ep = Episode(sp, backend=backend)
        ans = fn(ep, np.random.default_rng(seed * 7 + 1))
        post = pol.posterior_of(ep)
        s = score(sp, ans, ep.spent, post.prob_truth(sp), qref)
        s["p_conf"] = max(post.p_none, post.summary()["p_both"])
        rec["pol"][name] = s
        if name == "bayes_uniform_mixed":      # degenerate policies that reuse this full-spend data
            qa = post.prob_truth(sp)
            rec["pol"]["maxspend_cannot"] = score(sp, pol.CANNOT, ep.spent, qa, qref)
            rec["pol"]["maxspend_none"] = score(sp, pol.NONE, ep.spent, qa, qref)
    # zero-spend degenerate policies: q_agent = prior probability of the truth
    ep0 = Episode(sp, backend=backend)
    q0 = pol.posterior_of(ep0).prob_truth(sp)
    rg = np.random.default_rng(seed + 99)
    rand = [pol.NONE, pol.CANNOT, pol.planet(sp.P_cat * rg.choice([2, 1.5, 4 / 3]) * (1 + rg.choice([-1, 1]) * rg.uniform(.01, .05)),
                                              np.exp(rg.uniform(np.log(2), np.log(80))))][rg.integers(3)]
    rec["pol"]["always_none"] = score(sp, pol.NONE, 0, q0, qref)
    rec["pol"]["always_cannot"] = score(sp, pol.CANNOT, 0, q0, qref)
    rec["pol"]["random_guess"] = score(sp, rand, 0, q0, qref)
    rec["pol"]["always_planet_guess"] = score(sp, pol.planet(sp.P_cat * 2 * 1.02, 15.0), 0, q0, qref)
    # EIG: full-budget trajectory, stop rules replayed post hoc
    ep = Episode(sp, backend=backend)
    log = pol.eig_trajectory(ep, np.random.default_rng(seed * 13 + 5))
    keys = ("spent", "p_none", "p_period", "p_both", "p_mass", "period", "mass", "q_true", "rate", "rate_t", "rate_r", "n_tr", "n_rv")
    rec["eig_log"] = [{k: round(float(e[k]), 5) for k in keys} for e in log]
    rec["sec"] = time.time() - t0
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("start", type=int); ap.add_argument("n", type=int); ap.add_argument("out")
    ap.add_argument("--level", type=int, default=5); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--backend", default=None, help="ttvfast (default) | rebound | analytic")
    ap.add_argument("--rv-availability", default=None, choices=["ideal", "realistic"])
    a = ap.parse_args()
    if a.rv_availability:
        from hidden_planet.tasks import set_rv_mode
        set_rv_mode(a.rv_availability)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["seed"] for l in open(a.out)}
    jobs = [(s, a.split, a.level, a.backend) for s in range(a.start, a.start + a.n) if s not in done]
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.procs) as p, open(a.out, "a") as f:
        for i, rec in enumerate(p.imap_unordered(run_system, jobs)):
            f.write(json.dumps(rec, default=float) + "\n"); f.flush()
            if (i + 1) % 25 == 0:
                print(f"{a.out}: {i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    print("DONE", a.out, flush=True)
