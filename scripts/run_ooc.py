"""Decisive test: unchanged in-class EIG scheduler on out-of-class truth, plus scripted fixes.
usage: run_ooc.py N OUT [--procs 4] [--start 0]
"""
import os, sys, json, time, argparse
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hidden_planet.tasks import Episode, NMAX
from hidden_planet import policies as pol, ooc

OTHER = {"verdict": "other"}
MIN_T, MIN_RV = 16, 6


def topup(ep, rng):
    need = MIN_T - len(ep.transits)
    if need > 0:
        free = [n for n in np.linspace(0, NMAX, 4 * MIN_T).round().astype(int) if n not in ep.transits]
        for n in rng.choice(free, min(need, len(free)), replace=False):
            ep.observe_transit(int(n))
    for t in np.sort(rng.uniform(0, ep.t_max, max(0, MIN_RV - len(ep.rvs)))):
        ep.observe_rv(float(t))


def run_one(args):
    kind, seed = args
    np.seterr(all="ignore")
    t0 = time.time()
    sp = ooc.sample_ooc_system(seed, kind)
    qref = pol.reference_identifiability(sp) if kind == "in_class" else 0.0
    ep = Episode(sp)
    rng = np.random.default_rng(seed * 13 + 5)
    log = pol.eig_trajectory(ep, rng, stop_fn=pol.make_stop_rule(0.9, 0.02, 12.0))
    post0 = pol.posterior_of(ep)
    s0 = post0.summary(); ans0 = pol.bayes_answer(s0, 0.15)
    qa = post0.prob_truth(sp) if kind == "in_class" else 0.0
    spent0 = ep.spent
    rec = dict(kind=kind, seed=seed, has=sp.has_planet, conf=max(s0["p_none"], s0["p_both"]), n_tr0=len(ep.transits), n_rv0=len(ep.rvs),
               ooc={k: v for k, v in (sp.extra.get("ooc") or {}).items() if isinstance(v, (int, float, str))}, res={})
    rec["res"]["eig"] = ooc.score_ooc(sp, ans0, spent0, qa, qref)
    cA = ooc.chi2_check(post0)
    rec["res"]["fixA_chi2"] = ooc.score_ooc(sp, OTHER if cA["flag"] else ans0, spent0, qa, qref)
    cA2 = ooc.structured_check(post0, sp)
    rec["res"]["fixA2_structured_no_topup"] = ooc.score_ooc(sp, OTHER if cA2["flag"] else ans0, spent0, qa, qref)
    topup(ep, rng)
    post1 = pol.posterior_of(ep); s1 = post1.summary(); ans1 = pol.bayes_answer(s1, 0.15)
    qa1 = post1.prob_truth(sp) if kind == "in_class" else 0.0
    rec["res"]["topup_only"] = ooc.score_ooc(sp, ans1, ep.spent, qa1, qref)
    cB = ooc.chi2_check(post1)
    rec["res"]["fixB_topup_chi2"] = ooc.score_ooc(sp, OTHER if cB["flag"] else ans1, ep.spent, qa1, qref)
    cC = ooc.structured_check(post1, sp)
    rec["res"]["fixC_topup_structured"] = ooc.score_ooc(sp, OTHER if cC["flag"] else ans1, ep.spent, qa1, qref)
    rec["checks"] = dict(A_pt=cA["p_t"], A_pr=cA["p_r"], B_pt=cB["p_t"], B_pr=cB["p_r"], quad=cC["quad"], tsin=cC["tsin"], rsin=cC["rsin"])
    rec["sec"] = time.time() - t0
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("n", type=int); ap.add_argument("out")
    ap.add_argument("--procs", type=int, default=4); ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {(r["kind"], r["seed"]) for r in map(json.loads, open(a.out))}
    jobs = [(k, s) for s in range(a.start, a.start + a.n) for k in ("in_class",) + ooc.KINDS if (k, s) not in done]
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.procs) as p, open(a.out, "a") as f:
        for i, rec in enumerate(p.imap_unordered(run_one, jobs)):
            f.write(json.dumps(rec, default=float) + "\n"); f.flush()
            if (i + 1) % 40 == 0:
                print(f"{i+1}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    print("DONE", flush=True)
