"""Informed matched-filter detectability of each out-of-class signal: a full-budget reference
design (24 transits + 12 RVs) tested with the TRUE extra-physics template (known family and known
period where applicable). This is what an analyst who guessed the right alternative could detect.
usage: ooc_detectability.py N OUT
"""
import os, sys, json
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hidden_planet.tasks import Episode, NMAX
from hidden_planet import policies as pol, ooc, physics as ph

TWO_PI = 2 * np.pi


def one(a):
    kind, seed = a
    np.seterr(all="ignore")
    sp = ooc.sample_ooc_system(seed, kind)
    ep = Episode(sp, noise_seed=seed + 4242)
    pol.run_design(ep, 24, 12, np.random.default_rng(seed + 1))
    post = pol.posterior_of(ep)
    f = ooc.class_best_fit(post)
    nt, nr = f["nt"], f["nr"]; N = nt + nr
    o = sp.extra["ooc"]
    if kind == "drift":
        col = np.zeros((N, 1)); col[:nt, 0] = (post.n / NMAX) ** 2
        d, thr = ooc._dchi_extra(f, col), 9.0
    elif kind == "activity":
        d = 0.0
        for P in o["prot"] * np.linspace(0.9, 1.1, 21):
            cols = np.zeros((N, 2)); a_ = TWO_PI * post.t / P
            cols[nt:, 0] = np.sin(a_); cols[nt:, 1] = np.cos(a_)
            d = max(d, ooc._dchi_extra(f, cols))
        thr = 16.0
    else:
        nu = ph.ttv_frequency_per_epoch(3, o["delta"])
        cols = np.zeros((N, 2)); cols[:nt, 0] = np.sin(TWO_PI * nu * post.n); cols[:nt, 1] = np.cos(TWO_PI * nu * post.n)
        d1 = ooc._dchi_extra(f, cols)
        P3 = sp.P1 * 1.5 * (1 + o["delta"])
        cols = np.zeros((N, 2)); cols[nt:, 0] = np.sin(TWO_PI * post.t / P3); cols[nt:, 1] = np.cos(TWO_PI * post.t / P3)
        d, thr = max(d1, ooc._dchi_extra(f, cols)), 14.0
    return dict(kind=kind, seed=seed, dchi=float(d), detectable=bool(d > thr))


if __name__ == "__main__":
    from multiprocessing import Pool
    n = int(sys.argv[1])
    jobs = [(k, s) for s in range(n) for k in ooc.KINDS]
    with Pool(4) as p:
        out = p.map(one, jobs, chunksize=8)
    json.dump(out, open(sys.argv[2], "w"))
    for k in ooc.KINDS:
        print(k, "informed-detectable fraction:", np.mean([o["detectable"] for o in out if o["kind"] == k]).round(3))
