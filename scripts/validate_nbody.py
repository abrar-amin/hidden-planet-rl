"""Validate analytic TTVs/RVs against REBOUND (IAS15) across the env's parameter range.

Three analytic variants are compared with the N-body transit-time residuals (linear ephemeris
removed), using mean periods/phases measured from the N-body run:
  A  LXW12 single resonant term, tabulated f,g           (what the env's analytic backend uses)
  B  single term, alpha-dependent f(alpha), g(alpha)
  C  sum of first-order terms j=2..6, alpha-dependent
usage: validate_nbody.py N_PER_CASE [mass_max]
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hidden_planet import physics as ph
from hidden_planet.laplace import multi_resonance_ttv, fg

NE = 150


def one(rng, sigma_e, j=None, mmax=80.0):
    P1 = rng.uniform(5, 20)
    j = int(rng.choice([2, 3, 4])) if j is None else j
    delta = rng.choice([-1, 1]) * rng.uniform(0.01, 0.05)
    mu1 = np.exp(rng.uniform(np.log(3), np.log(30))) * ph.ME
    mu2 = np.exp(rng.uniform(np.log(2), np.log(mmax))) * ph.ME
    P2 = P1 * j / (j - 1) * (1 + delta)
    zf1 = rng.rayleigh(sigma_e) * np.exp(1j * rng.uniform(0, 2 * np.pi)) if sigma_e > 0 else 0j
    zf2 = rng.rayleigh(sigma_e) * np.exp(1j * rng.uniform(0, 2 * np.pi)) if sigma_e > 0 else 0j
    lam2_0 = rng.uniform(0, 2 * np.pi)
    f1, f2 = ph.forced_ecc(j, delta, mu1, mu2, P1, P2, j * lam2_0)
    z1, z2 = zf1 + f1, zf2 + f2
    rv_t = np.sort(rng.uniform(0, NE * P1, 40))
    out = ph.nbody_simulate(P1, mu1, z1, P2, mu2, z2, 0.0, lam2_0, NE, rv_t)
    rec = dict(P1=P1, j=j, delta=delta, m1=mu1 / ph.ME, m2=mu2 / ph.ME, e1=abs(zf1), e2=abs(zf2), stable=bool(out["stable"]))
    if not out["stable"] or len(out["t2"]) < 5:
        rec["stable"] = False
        return rec
    T0, P1m, res = ph.linear_ephemeris(out["t1"])
    T20, P2m, _ = ph.linear_ephemeris(out["t2"])
    lam2_at = 2 * abs(z2) * np.sin(np.angle(z2)) + 2 * np.pi * (T0 - T20) / P2m
    dm = (P2m / P1m) * (j - 1) / j - 1
    n = np.arange(NE)
    A = np.c_[np.ones(NE), n]
    detr = lambda x: x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    fa, ga = fg(j, P2m / P1m)
    preds = dict(
        A=ph.analytic_ttv(n, P1m, j, dm, mu2, ph.zfree_combo(j, zf1, zf2), lam2_at),
        B=multi_resonance_ttv(n, P1m, P2m, mu2, zf1, zf2, lam2_at, js=(j,)),
        C=multi_resonance_ttv(n, P1m, P2m, mu2, zf1, zf2, lam2_at))
    rec.update(amp_nb_min=np.sqrt(2) * np.std(res) * 1440, delta_m=dm)
    for k, p in preds.items():
        p = detr(p)
        rec[f"rel_{k}"] = float(np.std(res - p) / max(np.std(res), 1e-12))
        rec[f"abs_{k}_min"] = float(np.std(res - p) * 1440)
    # fitted resonant-sinusoid amplitude ratio (analytic A / nbody) : mass bias of the env model
    nu = ph.ttv_frequency_per_epoch(j, dm)
    X = np.c_[A, np.sin(2 * np.pi * nu * n), np.cos(2 * np.pi * nu * n)]
    c = np.linalg.lstsq(X, out["t1"], rcond=None)[0]
    W_nb = c[2] + 1j * c[3]
    rec["nonres_frac"] = float(np.std(out["t1"] - X @ c) / max(abs(W_nb) / np.sqrt(2), 1e-12))
    for k, (f_, g_) in dict(A=(ph.F_COEF[j], ph.G_COEF[j]), B=(fa, ga)).items():
        W = ph.ttv_prefactor(P1m, j, dm) * mu2 * (-f_ - 1.5 * np.conj(f_ * zf1 + g_ * zf2) / dm) * np.exp(1j * j * lam2_at)
        rec[f"ampratio_{k}"] = float(abs(W) / abs(W_nb)); rec[f"dphase_{k}"] = float(np.angle(W / W_nb))
    K1 = ph.rv_semi_amplitude(mu1, P1m, abs(z1)); K2 = ph.rv_semi_amplitude(mu2, P2m, abs(z2))
    rv_an = (ph.kepler_rv(rv_t, K1, P1m, ph.mean_longitude_at_transit(abs(z1), np.angle(z1)), abs(z1), np.angle(z1), t0=T0)
             + ph.kepler_rv(rv_t, K2, P2m, lam2_at, abs(z2), np.angle(z2), t0=T0))
    d = out["rv"] - rv_an
    rec.update(rv_rms_err=float(np.std(d - d.mean())), K2=float(K2))
    return rec


def summarize(recs, label):
    ok = [r for r in recs if r["stable"]]
    g = lambda k: np.array([r[k] for r in ok])
    p = lambda x: np.percentile(x, [50, 90]).round(3).tolist()
    s = dict(label=label, n=len(recs), unstable=len(recs) - len(ok), amp_nb_min=p(g("amp_nb_min")),
             rel_A=p(g("rel_A")), rel_B=p(g("rel_B")), rel_C=p(g("rel_C")), abs_A_min=p(g("abs_A_min")), abs_C_min=p(g("abs_C_min")),
             ampratio_A=np.percentile(g("ampratio_A"), [10, 50, 90]).round(3).tolist(),
             ampratio_B=np.percentile(g("ampratio_B"), [10, 50, 90]).round(3).tolist(),
             dphase_A_abs=p(np.abs(g("dphase_A"))), nonres_frac=p(g("nonres_frac")), rv_rms_err=p(g("rv_rms_err")))
    print(f"[{label}] n={s['n']} unstable={s['unstable']}  N-body TTV amp med/p90 {s['amp_nb_min']} min")
    print(f"   rel rms err (med, p90): A {s['rel_A']}  B {s['rel_B']}  C {s['rel_C']} | abs A {s['abs_A_min']} min, C {s['abs_C_min']} min")
    print(f"   resonant amp ratio analytic/nbody p10/50/90: A {s['ampratio_A']}  B {s['ampratio_B']} | |dphase| A {s['dphase_A_abs']} rad")
    print(f"   non-resonant residual / resonant rms: {s['nonres_frac']} | RV rms err {s['rv_rms_err']} m/s", flush=True)
    return s


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    out = {}
    for se in (0.0, 0.015):
        for j in (2, 3, 4):
            rng = np.random.default_rng(100 * j + int(se * 1000))
            recs = [one(rng, se, j) for _ in range(N)]
            label = f"j={j} sigma_e={se}"
            out[label] = dict(summary=summarize(recs, label), recs=recs)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "nbody_validation.json"), "w"), default=float)
