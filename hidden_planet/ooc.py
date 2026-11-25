"""Out-of-class ("hidden physics") generators, grading, and scripted model-adequacy checks.

Truth generators (analytic backend, 2:1-only level 5):
  two_planets : the 2:1 perturber plus a second hidden planet near 3:2 (extra TTV sinusoid + RV)
  activity    : quasi-periodic stellar-rotation RV signal (no TTV); perturber present with the
                usual prior probability
  drift       : quadratic transit-time drift (tidal decay / apsidal precession), no perturber
The inference engine / EIG scheduler know none of these.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from . import physics as ph
from .tasks import sample_system, _rng, NMAX, SIGMA_P, SIGMA_T0, GAMMA_SD, K1_SD
from .reward import score, LAMBDA, A_ABSTAIN

KINDS = ("two_planets", "activity", "drift")
OOC_LEVEL = 5
TWO_PI = 2 * np.pi


def sample_ooc_system(seed: int, kind: str, split: str = "test"):
    sp = sample_system(seed, OOC_LEVEL, split)
    if kind == "in_class":
        return sp
    r = _rng(seed, f"ooc/{kind}/{split}")
    if kind == "two_planets":
        if not sp.has_planet:                      # force the in-class planet to exist
            for k in range(1, 200):
                sp2 = sample_system(seed + 7_000_000 * k, OOC_LEVEL, split)
                if sp2.has_planet:
                    sp2.seed = seed; sp = sp2; break
        d3 = float(r.choice([-1, 1]) * r.uniform(0.01, 0.05))
        sp.extra["ooc"] = dict(kind=kind, j=3, delta=d3, m3=float(np.exp(r.uniform(np.log(4), np.log(80)))),
                               lam3=float(r.uniform(0, TWO_PI)), z=complex(r.normal(0, 0.03), r.normal(0, 0.03)))
    elif kind == "activity":
        prot = float(r.uniform(8, 35))
        sp.extra["ooc"] = dict(kind=kind, prot=prot, amp=float(np.exp(r.uniform(np.log(2), np.log(12)))),
                               periods=(prot * (1 + r.uniform(-0.07, 0.07, 5))).tolist(),
                               phases=r.uniform(0, TWO_PI, 5).tolist(), h2=r.uniform(0, TWO_PI, 5).tolist())
    elif kind == "drift":
        sp.has_planet = False; sp.P2 = 0.0; sp.m2 = 0.0
        sag = float(r.choice([-1, 1]) * np.exp(r.uniform(np.log(3), np.log(60))))    # minutes over the baseline
        sp.extra["ooc"] = dict(kind=kind, sag_min=sag)
    else:
        raise ValueError(kind)
    return sp


def ttv_extra(sp, n):
    o = sp.extra["ooc"]
    n = np.asarray(n, float)
    if o["kind"] == "two_planets":
        return ph.analytic_ttv(n, sp.P1, o["j"], o["delta"], o["m3"] * ph.ME, o["z"], o["lam3"])
    if o["kind"] == "drift":
        x = n / NMAX - 0.5                           # parabola with zero mean slope, peak-to-chord = sag
        return (o["sag_min"] / 1440.0) * 4 * (x * x - 1 / 12.0)
    return 0.0 * n


def rv_extra(sp, t):
    o = sp.extra["ooc"]
    t = np.asarray(t, float)
    if o["kind"] == "two_planets":
        P3 = sp.P1 * 1.5 * (1 + o["delta"])
        return ph.kepler_rv(t, ph.rv_semi_amplitude(o["m3"] * ph.ME, P3), P3, o["lam3"], 0.0, 0.0, t0=sp.T0)
    if o["kind"] == "activity":
        v = 0.0 * t
        for P, p1, p2 in zip(o["periods"], o["phases"], o["h2"]):
            v = v + np.cos(TWO_PI * t / P + p1) + 0.5 * np.cos(2 * TWO_PI * t / P + p2)
        return o["amp"] * v / math.sqrt(5 * 0.625)     # rms = amp
    return 0.0 * t


def score_ooc(sp, answer, spent, q_agent=0.0, q_ref=0.0, lam=LAMBDA, mode="partial", wrong_penalty=None):
    """Grading with the extra verdict 'other' (= the single-perturber model does not explain the data).
    In-class: 'other' is wrong (-0.5). Out-of-class: 'other' +1, 'cannot_determine' +0.4;
    'planet' is graded against the 2:1 planet where one exists; 'none' is right only for
    activity-without-planet (there really is no perturber and the transit times are clean)."""
    v = (answer or {}).get("verdict")
    kind = (sp.extra.get("ooc") or {}).get("kind", "in_class")
    if mode in ("strict", "strict0"):
        # same right / wrong / abstain structure as the in-class reward: 'other' is right iff the truth is out of class
        W = (1.0 if mode == "strict0" else 0.5) if wrong_penalty is None else wrong_penalty
        if kind == "in_class" and v != "other":
            s = score(sp, answer, spent, q_agent, q_ref, lam=lam, mode=mode, wrong_penalty=wrong_penalty)
        else:
            if v == "other":
                sc = 1.0 if kind != "in_class" else -W
            elif v == "cannot_determine":
                sc = 0.0 if mode == "strict0" else (A_ABSTAIN if kind != "in_class" else -0.3)
            elif v == "none":
                sc = 1.0 if (kind == "activity" and not sp.has_planet) else -W
            elif v == "planet" and kind in ("two_planets", "activity") and sp.has_planet:
                sc = score(sp, answer, spent, 1.0, 1.0, mode=mode, wrong_penalty=wrong_penalty)["score"]
            else:
                sc = -W if v in ("planet",) else -1.0
            s = dict(verdict=v if v in ("other", "cannot_determine", "none", "planet") else "invalid", score=sc, correct=float(sc > 0.99))
        s["spent"] = spent; s["reward"] = s["score"] - lam * spent; s["wrong"] = float(s["score"] < 0)
        return s
    if kind == "in_class":
        if v == "other":
            return dict(verdict=v, score=-0.5, reward=-0.5 - lam * spent, correct=0.0, wrong=1.0, spent=spent)
        s = score(sp, answer, spent, q_agent, q_ref, lam=lam)
    elif v == "other":
        s = dict(verdict=v, score=1.0, correct=1.0)
    elif v == "cannot_determine":
        s = dict(verdict=v, score=A_ABSTAIN, correct=0.0)
    elif v == "none":
        ok = kind == "activity" and not sp.has_planet
        s = dict(verdict=v, score=1.0 if ok else -0.5, correct=float(ok))
    elif v == "planet" and kind in ("two_planets", "activity") and sp.has_planet:
        s = score(sp, answer, spent, 1.0, 1.0)
    else:
        s = dict(verdict=v, score=-0.5, correct=0.0)
    s["spent"] = spent
    s["reward"] = s["score"] - lam * spent
    s["wrong"] = float(s["score"] < 0)
    return s


# ------------------------------------------------------------------ model-adequacy checks
def class_best_fit(post):
    """Maximum-likelihood fit of the in-class model family (free W, K2-vector; local continuous
    scan in Delta around the best grid cell). Returns design pieces and residuals."""
    pr = post.prior
    nt, nr = len(post.n), len(post.t)
    _, _, _, bound, _ = post._cell_stats(post.js, post.deltas)
    ib = int(np.argmax(bound))
    j, d0, w = int(post.js[ib]), float(post.deltas[ib]), float(post.widths[ib])
    best = None
    ridge = np.diag([1 / SIGMA_T0 ** 2, 1 / SIGMA_P ** 2, 1 / GAMMA_SD ** 2, 1 / K1_SD ** 2, 0, 0, 0, 0]) + 1e-12 * np.eye(8)
    for d in d0 + np.linspace(-1.5, 1.5, 61) * w:
        Xu, _ = post._design_u(np.array([j]), np.array([d]))
        X = np.c_[post.Xe, Xu[0]]
        c = np.linalg.solve(X.T @ X + ridge, X.T @ post.yw)
        res = post.yw - X @ c
        chi = float(res @ res)
        if best is None or chi < best["chi2"]:
            best = dict(chi2=chi, X=X, res=res, delta=d, j=j, coef=c)
    best.update(nt=nt, nr=nr, ridge=ridge)
    return best


def chi2_check(post, alpha=0.005):
    """Fix A: posterior-predictive style goodness of fit of the best in-class model."""
    f = class_best_fit(post)
    nt, nr = f["nt"], f["nr"]
    chi_t = float(np.sum(f["res"][:nt] ** 2)); chi_r = float(np.sum(f["res"][nt:] ** 2))
    dof_t, dof_r = nt - 5, nr - 4
    p_t = stats.chi2.sf(chi_t, dof_t) if dof_t >= 3 else 1.0
    p_r = stats.chi2.sf(chi_r, dof_r) if dof_r >= 3 else 1.0
    return dict(flag=bool(min(p_t, p_r) < alpha), p_t=float(p_t), p_r=float(p_r), fit=f)


def _dchi_extra(f, cols):
    X = np.c_[f["X"], cols]
    k = X.shape[1]
    ridge = np.zeros((k, k)); ridge[:8, :8] = f["ridge"]; ridge += 1e-12 * np.eye(k)
    c = np.linalg.solve(X.T @ X + ridge, X.T @ (f["res"] + f["X"] @ f["coef"]))
    r = f["res"] + f["X"] @ f["coef"] - X @ c
    return f["chi2"] - float(r @ r)


def structured_check(post, sp_pub, thr_quad=16.0, thr_sin=30.0):
    """Fix C: targeted residual tests a skeptic would add: quadratic timing term, an extra sinusoid
    in the transit times, an extra sinusoid in the RVs (periodogram over 3-60 d)."""
    c = chi2_check(post)
    f = c["fit"]; nt, nr = f["nt"], f["nr"]
    out = dict(chi2_flag=c["flag"], quad=0.0, tsin=0.0, rsin=0.0)
    N = nt + nr
    if nt >= 8:
        col = np.zeros(N); col[:nt] = (post.n / NMAX) ** 2 / sp_pub.sig_t * 0.01
        out["quad"] = _dchi_extra(f, col[:, None])
    if nt >= 14:
        best = 0.0
        for fr in np.linspace(1 / 300.0, 0.5, 300):
            cols = np.zeros((N, 2)); a = TWO_PI * fr * post.n
            cols[:nt, 0] = np.sin(a); cols[:nt, 1] = np.cos(a)
            best = max(best, _dchi_extra(f, cols))
        out["tsin"] = best
    if nr >= 10:
        best = 0.0
        for P in np.exp(np.linspace(np.log(3), np.log(60), 300)):
            cols = np.zeros((N, 2)); a = TWO_PI * post.t / P
            cols[nt:, 0] = np.sin(a); cols[nt:, 1] = np.cos(a)
            best = max(best, _dchi_extra(f, cols))
        out["rsin"] = best
    out["flag"] = bool(c["flag"] or out["quad"] > thr_quad or out["tsin"] > thr_sin or out["rsin"] > thr_sin)
    return out
