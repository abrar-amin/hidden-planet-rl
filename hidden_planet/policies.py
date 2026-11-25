"""Scripted policies: prototype heuristics, fixed designs + Bayes answer, and a Bayesian EIG scheduler."""
from __future__ import annotations

import numpy as np

from . import physics as ph
from .inference import Posterior, Prior, label_eig
from .reward import A_ABSTAIN, B_ABSTAIN, TAU
from .tasks import NMAX, COST, LEVELS, Episode, rv_available

NONE = {"verdict": "none"}
CANNOT = {"verdict": "cannot_determine"}


def planet(P, m):
    return {"verdict": "planet", "period_days": float(P), "mass_earth": float(m)}


def posterior_of(ep: Episode) -> Posterior:
    return Posterior(Prior.from_system(ep.sp), *ep.data())


def bayes_answer(summ, c_abst=0.15, mode="partial", wrong_penalty=0.5):
    """Expected-score-maximising answer from a posterior summary. c_abst = assumed value of abstaining
    when the agent's own posterior is below TAU (it cannot know q_ref)."""
    e_none = 1.5 * summ["p_none"] - 0.5
    e_pl = 0.75 * summ["p_period"] + 0.75 * summ["p_mass"] - 0.5
    if mode in ("strict", "strict0"):
        W = wrong_penalty
        e_none = (1 + W) * summ["p_none"] - W
        e_pl = (1 + W) * summ["p_both"] - W
    if mode == "strict0":                      # abstaining is worth exactly 0: answer iff P(correct) > W / (1 + W)
        e_cannot = 0.0
        best = max(e_none, e_pl, e_cannot)
        if best == e_cannot:
            return dict(CANNOT)
        return dict(NONE) if best == e_none else planet(summ["period"], summ["mass"])
    e_cannot = c_abst if max(summ["p_none"], summ["p_both"]) < TAU else -B_ABSTAIN
    best = max(e_none, e_pl, e_cannot)
    if best == e_cannot:
        return dict(CANNOT)
    if best == e_none:
        return dict(NONE)
    return planet(summ["period"], summ["mass"])


# ------------------------------------------------------------------ fixed designs
def design_epochs(k, rng=None, jitter=0):
    e = np.linspace(0, NMAX, k).round().astype(int)
    if rng is not None and jitter:
        e = np.clip(e + rng.integers(-jitter, jitter + 1, k), 0, NMAX)
    return np.unique(e)


def run_design(ep, n_transit, n_rv, rng, jitter=2):
    for n in design_epochs(n_transit, rng, jitter):
        ep.observe_transit(int(n))
    if n_rv:
        for t in np.sort(rng.uniform(0, ep.t_max, n_rv)):
            ep.observe_rv(float(t))


def uniform_transits_bayes(ep, rng):
    run_design(ep, 60, 0, rng, 0)
    return bayes_answer(posterior_of(ep).summary())


def uniform_mixed_bayes(ep, rng):
    run_design(ep, 30, 10, rng, 2)
    return bayes_answer(posterior_of(ep).summary())


def rv_heavy_bayes(ep, rng):
    run_design(ep, 15, 15, rng, 2)
    return bayes_answer(posterior_of(ep).summary())


REFERENCE_DESIGNS = ((24, 12), (12, 16), (36, 8))


def reference_identifiability(sp, backend=None):
    """q_ref: max posterior probability of the truth over full-budget reference designs."""
    best = 0.0
    designs = REFERENCE_DESIGNS if rv_available(sp) else ((60, 0), (40, 0))      # only observation types this star allows
    for i, (nt, nr) in enumerate(designs):
        ep = Episode(sp, noise_seed=sp.seed + 7919 * (i + 1), backend=backend)
        run_design(ep, nt, nr, np.random.default_rng(sp.seed * 31 + i), 2)
        best = max(best, posterior_of(ep).prob_truth(sp))
    return best


# ------------------------------------------------------------------ prototype heuristics
DET = 25.0


def _transit_fit(ep):
    n, y, _, _ = ep.data()
    if len(n) < 6:
        return None
    n = n.astype(float); w = 1 / ep.sp.sig_t
    A0 = np.c_[np.ones_like(n), n]
    c0 = np.linalg.lstsq(A0, y, rcond=None)[0]
    chi0 = np.sum(((y - A0 @ c0) * w) ** 2)
    best = (chi0, None, None)
    for Ntt in np.exp(np.linspace(np.log(4), np.log(2000), 1500)):
        A = np.c_[A0, np.sin(2 * np.pi * n / Ntt), np.cos(2 * np.pi * n / Ntt)]
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        chi = np.sum(((y - A @ c) * w) ** 2)
        if chi < best[0]:
            best = (chi, Ntt, np.hypot(c[2], c[3]))
    return dict(dchi=chi0 - best[0], Ntt=best[1], amp=best[2], n=len(n))


def _candidates(P, Ntt, amp, res=(2, 3, 4)):
    out = []
    for j in res:
        for sg in (-1, 1):
            den = (j - 1) * Ntt - sg
            if den <= 0:
                continue
            D = sg / den
            if abs(D) > 0.1:
                continue
            Pp = P * j / (j - 1) * (1 + D)
            mu = amp * np.pi * j ** (2 / 3) * (j - 1) ** (1 / 3) * abs(D) / (P * abs(ph.F_COEF[j]))
            out.append(dict(j=j, D=D, Pp=Pp, mu=mu))
    return out


def _rv_fit(ep, Pp):
    _, _, t, v = ep.data()
    if len(t) < 5:
        return None
    t = t - ep.sp.T0_cat
    A = np.c_[np.ones_like(t), np.sin(2 * np.pi * t / ep.sp.P_cat), np.sin(2 * np.pi * t / Pp), np.cos(2 * np.pi * t / Pp)]
    c = np.linalg.lstsq(A, v, rcond=None)[0]
    return np.sum(((v - A @ c) / ep.sp.sig_rv) ** 2), np.hypot(c[2], c[3])


def _decide(ep, margin=9.0):
    f = _transit_fit(ep)
    if f is None or f["dchi"] < DET:
        return dict(NONE)
    cs = _candidates(ep.sp.P_cat, f["Ntt"], f["amp"], LEVELS[ep.sp.level]["res"])
    fits = [(_rv_fit(ep, c["Pp"]), c) for c in cs]
    fits = [(r, c) for r, c in fits if r is not None]
    if not fits:
        return dict(CANNOT)
    fits.sort(key=lambda z: z[0][0])
    (chi, K), c = fits[0]
    if len(fits) > 1 and fits[1][0][0] - chi < margin:
        return dict(CANNOT)
    return planet(c["Pp"], ph.k_to_mass_earth(K, c["Pp"]))


def proto_uniform_transits(ep, rng):
    for n in design_epochs(60):
        ep.observe_transit(int(n))
    f = _transit_fit(ep)
    if f is None or f["dchi"] < DET:
        return dict(NONE)
    cs = _candidates(ep.sp.P_cat, f["Ntt"], f["amp"], LEVELS[ep.sp.level]["res"])
    c = [c for c in cs if c["j"] == 2 and c["D"] > 0] or cs
    if not c:
        return dict(CANNOT)
    return planet(c[0]["Pp"], c[0]["mu"] / ph.ME)


def proto_uniform_mixed(ep, rng):
    for n in design_epochs(30):
        ep.observe_transit(int(n))
    for t in np.linspace(0, ep.t_max, 10):
        ep.observe_rv(float(t))
    return _decide(ep)


def proto_adaptive(ep, rng):
    for n in design_epochs(16, rng, 3):
        ep.observe_transit(int(n))
    f = _transit_fit(ep)
    if f is None or f["dchi"] < DET:
        free = [n for n in range(NMAX + 1) if n not in ep.transits]
        for n in rng.choice(free, 10, replace=False):
            ep.observe_transit(int(n))
        f = _transit_fit(ep)
        if f["dchi"] < DET:
            return dict(NONE)
    cs = _candidates(ep.sp.P_cat, f["Ntt"], f["amp"], LEVELS[ep.sp.level]["res"])
    if not cs or not rv_available(ep.sp):
        return dict(CANNOT)
    tgrid = np.linspace(0, ep.t_max, 4000)
    for t in tgrid[rng.choice(len(tgrid), 5, replace=False)]:
        ep.observe_rv(float(t))
    while ep.remaining >= 3 * COST["rv"]:
        fits = sorted([(_rv_fit(ep, c["Pp"])[0], c) for c in cs], key=lambda z: z[0])
        if len(fits) == 1 or fits[1][0] - fits[0][0] > 16:
            break
        top = [c["Pp"] for _, c in fits[:3]]
        var = np.array([np.sin(2 * np.pi * tgrid / p) for p in top]).var(0)
        before = ep.spent
        for t in tgrid[np.argsort(-var)[:150:50]]:
            if ep.observe_rv(float(t))[0] is None:           # duplicate time: jitter it
                ep.observe_rv(float(np.clip(t + rng.uniform(-1, 1), 0, ep.t_max)))
        if ep.spent == before:
            break
    return _decide(ep)


# ------------------------------------------------------------------ Bayesian EIG scheduler
def eig_trajectory(ep, rng, batch=3, m=160, n_cand=48, stop_fn=None):
    """Greedy expected-information-gain-per-cost acquisition about the answer label.
    Logs a posterior summary before every purchase so any stopping rule can be replayed post hoc.
    stop_fn(entry) -> bool lets a rule stop online. Returns the log (list of dicts)."""
    sp = ep.sp
    log = []
    while True:
        post = posterior_of(ep)
        entry = post.summary()
        entry.update(spent=ep.spent, q_true=post.prob_truth(sp), n_tr=len(ep.transits), n_rv=len(ep.rvs))
        rate_t = rate_r = 0.0
        if ep.remaining >= COST["transit"] - 1e-9:
            S = post.sample(m, rng)
            free = np.array([n for n in range(NMAX + 1) if n not in ep.transits])
            epochs = rng.choice(free, min(n_cand, len(free)), replace=False)
            times = rng.uniform(0, ep.t_max, n_cand)
            tr, rv = post.predict(S, epochs, times)
            eig_t = label_eig(tr, S["label"], sp.sig_t) / COST["transit"]
            rate_t = float(eig_t.max())
            if ep.remaining >= COST["rv"] - 1e-9 and rv_available(sp):
                eig_r = label_eig(rv, S["label"], sp.sig_rv) / COST["rv"]
                rate_r = float(eig_r.max())
        entry.update(rate=max(rate_t, rate_r), rate_t=rate_t, rate_r=rate_r)
        log.append(entry)
        if ep.remaining < COST["transit"] - 1e-9 or (stop_fn is not None and stop_fn(entry)):
            break
        if rate_r > rate_t:
            order = np.argsort(-eig_r); chosen = []
            for i in order:
                if all(abs(times[i] - times[k]) > 0.1 * sp.P_cat for k in chosen):
                    chosen.append(i)
                if len(chosen) >= max(1, batch // 2) or (len(chosen) + 1) * COST["rv"] > ep.remaining:
                    break
            for i in chosen:
                ep.observe_rv(float(times[i]))
        else:
            order = np.argsort(-eig_t); chosen = []
            for i in order:
                if all(abs(epochs[i] - epochs[k]) >= 2 for k in chosen):
                    chosen.append(i)
                if len(chosen) >= batch or len(chosen) + 1 > ep.remaining:
                    break
            for i in chosen:
                ep.observe_transit(int(epochs[i]))
    return log


def make_stop_rule(p_stop=0.9, rate_thr=0.004, min_spend=8.0):
    def stop(e):
        if e["spent"] < min_spend:
            return False
        return max(e["p_none"], e["p_both"]) >= p_stop or e["rate"] < rate_thr
    return stop


def replay_stop(log, stop_fn):
    for e in log:
        if stop_fn(e):
            return e
    return log[-1]


def eig_policy(ep, rng, p_stop=0.9, rate_thr=0.004, min_spend=8.0, c_abst=0.15):
    log = eig_trajectory(ep, rng, stop_fn=make_stop_rule(p_stop, rate_thr, min_spend))
    return bayes_answer(log[-1], c_abst)
