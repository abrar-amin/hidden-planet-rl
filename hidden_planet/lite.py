"""TOOL_LEVEL=lite: a fit tool that reports best-fit parameters and chi-square statistics only.

The 'oracle' fit_model prints calibrated posterior probabilities, which reduces the agent's job to thresholding them.
fit_model in lite mode prints, for the no-perturber model and for each resonance alias, the maximum-likelihood fit
(period, mass), its chi2, delta-chi2 against no perturber, and BIC. Turning those into commit / abstain / buy-more
decisions is left to the agent.
"""
from __future__ import annotations

import math

import numpy as np

from . import physics as ph
from .inference import Posterior, Prior
from .tasks import SIGMA_T0, SIGMA_P, GAMMA_SD, K1_SD

_r = lambda x, k=3: round(float(x), k)


def lite_fit(sp, n, y, t, v):
    """dict for the lite tool, or {'error': ...}. Uses the same single-perturber model as the oracle, without priors on
    the perturber (free TTV amplitude/phase and free RV amplitude/phase at each trial period)."""
    nt, nr = len(n), len(t)
    if nt + nr < 3:
        return {"error": "buy at least 3 observations first"}
    post = Posterior(Prior.from_system(sp), n, y, t, v, refine=False, prune=False, evidence=False)
    N = nt + nr
    k0 = 2 + (2 if nr else 0)
    chi0 = float(post.c0)
    out = {"n_transits": int(nt), "n_rvs": int(nr),
           "no_perturber": {"chi2": _r(chi0, 1), "n_params": k0, "bic": _r(chi0 + k0 * math.log(max(N, 2)), 1)}}
    _, _, _, bound, _ = post._cell_stats(post.js, post.deltas)
    ridge = np.diag([1 / SIGMA_T0 ** 2, 1 / SIGMA_P ** 2, 1 / GAMMA_SD ** 2, 1 / K1_SD ** 2, 0, 0, 0, 0]) + 1e-10 * np.eye(8)
    aliases = []
    for j in sorted(set(int(x) for x in post.js)):
        for sg in (1, -1):
            m = np.flatnonzero((post.js == j) & (np.sign(post.deltas) == sg))
            if len(m) == 0:
                continue
            ib = m[int(np.argmax(bound[m]))]
            d0, w = float(post.deltas[ib]), float(post.widths[ib])
            best = None
            for d in d0 + np.linspace(-1.0, 1.0, 21) * w:
                if not (0.008 <= abs(d) <= 0.052) or np.sign(d) != sg:
                    continue
                Xu, P2 = post._design_u(np.array([j]), np.array([d]))
                X = np.c_[post.Xe, Xu[0]]
                G = X.T @ X + ridge
                c = np.linalg.solve(G, X.T @ post.yw)
                res = post.yw - X @ c
                chi = float(res @ res)
                if best is None or chi < best[0]:
                    best = (chi, d, float(P2[0]), c, G)
            chi, d, P2, c, G = best
            k = k0 + 3 + (2 if nr else 0)
            amp = float(np.hypot(c[4], c[5]))
            f = ph.coef_fg(j, d)[0]
            m_ttv = amp / (abs(ph.ttv_prefactor(sp.P_cat, j, d)) * ph.ME * abs(f))
            rec = {"j": j, "delta_sign": "+" if sg > 0 else "-", "period_days": _r(P2, 4), "chi2": _r(chi, 1), "delta_chi2_vs_none": _r(chi0 - chi, 1),
                   "n_params": k, "bic": _r(chi + k * math.log(max(N, 2)), 1), "ttv_amplitude_min": _r(amp * 1440, 2), "mass_if_circular": _r(m_ttv, 1)}
            mass, src = m_ttv, "ttv_amplitude_assuming_circular"
            if nr >= 5:
                K2 = float(np.hypot(c[6], c[7]))
                try:
                    cov = np.linalg.inv(G); sK = float(np.sqrt(max(0.5 * (cov[6, 6] + cov[7, 7]), 0.0)))
                except np.linalg.LinAlgError:
                    sK = float("inf")
                m_rv = ph.k_to_mass_earth(K2, P2)
                rec.update(K2_ms=_r(K2, 2), K2_sigma_ms=_r(sK, 2), mass_from_rv=_r(m_rv, 1))
                if sK > 0 and K2 / sK >= 2.0:
                    mass, src = m_rv, "rv_amplitude"
            rec.update(mass_earth=_r(mass, 1), mass_from=src)
            aliases.append(rec)
    aliases.sort(key=lambda a: a["chi2"])
    out["aliases"] = aliases
    if len(aliases) >= 2:
        out["delta_chi2_between_aliases"] = _r(aliases[1]["chi2"] - aliases[0]["chi2"], 1)
    out["note"] = ("No probabilities are given. Rules of thumb: delta_chi2_vs_none above about 25 is a clear detection (three or more extra parameters, many trial periods); "
                   "the aliases are told apart when delta_chi2_between_aliases is about 10 or more; BIC = chi2 + n_params ln(N), lower is better; "
                   "a mass from the RV amplitude is trustworthy when K2_ms is several times K2_sigma_ms.")
    return out
