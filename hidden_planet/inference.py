"""Grid-based Bayesian inference inside the known hypothesis class.

H0: no perturber.   H1: perturber near a j:j-1 resonance, (alias=(j, sign), |Delta|, mass, phase).
Given (alias, Delta) the data model is linear in
    eta = (dT0, dP, gamma, K1)             [Gaussian priors, shared by H0 and H1]
    u   = (W_re, W_im, K2 cos phi, K2 sin phi)
with transit residual  r_n = dT0 + dP*n + W_re sin(2 pi nu n) + W_im cos(2 pi nu n)
and  rv(t) = gamma - K1 sin(2 pi t'/P1) - K2 sin(phi + 2 pi t'/P2),  t' = t - T0.
W = V exp(i j phi) where V is the LXW complex TTV amplitude. With Rayleigh free eccentricities,
W | (mass, phi) is a 2-d Gaussian (mean from the -f term, isotropic sd from the Zfree term): this
is exactly the mass-eccentricity degeneracy. eta and W are marginalised analytically; (alias,
Delta, mass, phi) live on a grid, refined in Delta around likelihood peaks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import physics as ph
from .laplace import fg_array
from .tasks import (NMAX, SIGMA_T0, SIGMA_P, GAMMA_SD, K1_SD, RESONANCES, DELTA_RANGE, MASS_RANGE,
                    P_NONE)

N_DELTA = 160
N_MASS = 28
N_PHI = 24          # coarse phase grid (weak signal)
N_PHI_FINE = 96     # used when few cells survive pruning (strong signal -> sharp phase)
ECC_FLOOR = 0.002
PRUNE = 14.0        # drop cells whose max-likelihood bound is this far below the best evidence
PERIOD_TOL = 0.005
MASS_TOL = np.log(1.5)
TWO_PI = 2 * np.pi


@dataclass
class Prior:
    P_cat: float
    T0_cat: float
    sig_t: float
    sig_rv: float
    sigma_e: float
    p_none: float = P_NONE
    model_err_frac: float = 0.07     # systematic floor: the first-order model is only good to a few % of the TTV amplitude
    resonances: tuple = RESONANCES
    mass_range: tuple = MASS_RANGE

    @classmethod
    def from_system(cls, sp):
        from .tasks import LEVELS
        L = LEVELS[sp.level]
        return cls(sp.P_cat, sp.T0_cat, sp.sig_t, sp.sig_rv, sp.sigma_e, resonances=tuple(L["res"]),
                   mass_range=tuple(L["mass"]))


class Posterior:
    """Grid posterior for one data set."""

    def __init__(self, prior: Prior, n, y, t, v, refine: bool = True, prune: bool = True, evidence: bool = True):
        self.prior = pr = prior
        self.n = np.asarray(n, float)
        self.r = np.asarray(y, float) - (pr.T0_cat + self.n * pr.P_cat)
        self.t = np.asarray(t, float) - pr.T0_cat
        self.v = np.asarray(v, float)
        self.masses = np.exp(np.linspace(np.log(pr.mass_range[0]), np.log(pr.mass_range[1]), N_MASS))
        self.sig_t = pr.sig_t
        self._nuisance()
        d = np.linspace(DELTA_RANGE[0], DELTA_RANGE[1], N_DELTA + 1)
        dc, dw = 0.5 * (d[1:] + d[:-1]), np.diff(d)
        js, deltas, widths = [], [], []
        for j in pr.resonances:
            for sg in (-1, 1):
                js.append(np.full(N_DELTA, j)); deltas.append(sg * dc); widths.append(dw)
        js, deltas, widths = map(np.concatenate, (js, deltas, widths))
        Gt, bt, _, bound, _ = self._cell_stats(js, deltas)
        if pr.model_err_frac > 0 and len(self.n) >= 6 and bound.max() > 12.0:
            # strong signal: add a model-error floor proportional to the fitted TTV amplitude, then redo the stats
            ib = int(np.argmax(bound))
            w = np.linalg.solve(Gt[ib] + 1e-9 * np.trace(Gt[ib]) * np.eye(4), bt[ib])[:2]
            self.sig_t = float(np.hypot(pr.sig_t, pr.model_err_frac * np.hypot(w[0], w[1])))
            self._nuisance()
            bound = self._cell_stats(js, deltas)[3]
        if refine and len(self.n) >= 4:
            js, deltas, widths = self._refine(js, deltas, widths, bound)
        self.js, self.deltas, self.widths = js, deltas, widths
        if evidence:                 # the lite tool needs only the maximum-likelihood statistics
            self._evidence(prune)

    # ---- nuisance block (same for all hypotheses) ----
    def _nuisance(self):
        pr = self.prior
        nt, nr = len(self.n), len(self.t)
        Xe = np.zeros((nt + nr, 4))
        Xe[:nt, 0] = 1 / self.sig_t; Xe[:nt, 1] = self.n / self.sig_t
        Xe[nt:, 2] = 1 / pr.sig_rv; Xe[nt:, 3] = -np.sin(TWO_PI * self.t / pr.P_cat) / pr.sig_rv
        self.yw = np.concatenate([self.r / self.sig_t, self.v / pr.sig_rv])
        self.Xe = Xe
        L0 = np.diag([1 / SIGMA_T0 ** 2, 1 / SIGMA_P ** 2, 1 / GAMMA_SD ** 2, 1 / K1_SD ** 2])
        self.M = Xe.T @ Xe + L0
        self.Minv = np.linalg.inv(self.M)
        self.be = Xe.T @ self.yw
        self.c0 = float(self.yw @ self.yw - self.be @ self.Minv @ self.be)
        # log evidence of H0 up to a constant shared with H1
        self.logL0 = -0.5 * self.c0

    def _design_u(self, js, deltas):
        """Weighted design for u, shape (C, N, 4)."""
        pr = self.prior
        nt, nr = len(self.n), len(self.t)
        nu = -(js - 1) * deltas / (1 + deltas)
        P2 = pr.P_cat * js / (js - 1) * (1 + deltas)
        Xu = np.zeros((len(js), nt + nr, 4))
        a = TWO_PI * nu[:, None] * self.n[None, :]
        Xu[:, :nt, 0] = np.sin(a) / self.sig_t; Xu[:, :nt, 1] = np.cos(a) / self.sig_t
        w = TWO_PI * self.t[None, :] / P2[:, None]
        # -K2 sin(phi + w) = kc * (-sin w) + ks * (-cos w)
        Xu[:, nt:, 2] = -np.sin(w) / pr.sig_rv; Xu[:, nt:, 3] = -np.cos(w) / pr.sig_rv
        return Xu, P2

    def _cell_stats(self, js, deltas):
        Xu, P2 = self._design_u(js, deltas)
        Guu = np.einsum("cni,cnj->cij", Xu, Xu)
        Gue = np.einsum("cni,nj->cij", Xu, self.Xe)
        bu = np.einsum("cni,n->ci", Xu, self.yw)
        T = Gue @ self.Minv
        Gt = Guu - np.einsum("cij,ckj->cik", T, Gue)
        bt = bu - T @ self.be
        # upper bound of the likelihood over u (weak ridge for stability)
        ridge = np.eye(4) * 1e-9 * (1 + np.trace(Gt, axis1=1, axis2=2))[:, None, None]
        sol = np.linalg.solve(Gt + ridge, bt[..., None])[..., 0]
        bound = 0.5 * np.einsum("ci,ci->c", bt, sol)
        return Gt, bt, P2, bound, Gue

    def _refine(self, js, deltas, widths, bound):
        """Replace clusters of high-likelihood cells with a finer Delta grid (same prior mass)."""
        if bound.max() < 12.0:
            return js, deltas, widths
        keep = bound > bound.max() - 12.0
        if keep.sum() > 60:
            return js, deltas, widths
        new = [(js[~keep], deltas[~keep], widths[~keep])]
        for i in np.flatnonzero(keep):
            k = 9
            off = (np.arange(k) - (k - 1) / 2) / k * widths[i]
            new.append((np.full(k, js[i]), deltas[i] + np.sign(deltas[i]) * off, np.full(k, widths[i] / k)))
        return tuple(np.concatenate([x[m] for x in new]) for m in range(3))

    # ---- evidence on the (cell, mass, phi) grid ----
    def _evidence(self, prune=True):
        pr = self.prior
        Gt, bt, P2, bound, Gue = self._cell_stats(self.js, self.deltas)
        # prune: a cell's evidence <= -c0/2 + bound, while H0 achieves exactly -c0/2 and the best
        # cell achieves at least (roughly) its bound minus an Occam factor. Dropped mass < ~1e-3.
        n_alias = 2 * len(pr.resonances)
        wcell_all = self.widths / (DELTA_RANGE[1] - DELTA_RANGE[0]) / n_alias
        if prune:
            keep = bound > max(bound.max() - 8.0, 0.0) - PRUNE
            if keep.sum() < len(keep):
                Gt, bt, P2, Gue = Gt[keep], bt[keep], P2[keep], Gue[keep]
                self.js, self.deltas, self.widths = self.js[keep], self.deltas[keep], self.widths[keep]
                wcell_all = wcell_all[keep]
        js, deltas = self.js, self.deltas
        self.P2, self.Gt, self.bt, self.Gue = P2, Gt, bt, Gue
        C = len(js)
        n_phi = N_PHI_FINE if C <= 400 else N_PHI
        self.phis = (np.arange(n_phi) + 0.5) * TWO_PI / n_phi
        # alpha-dependent resonance coefficients (the tabulated LXW values are exact only at Delta = 0)
        fj = np.empty(C); gj = np.empty(C)
        for jv in np.unique(js):
            m = js == jv
            fj[m], gj[m] = fg_array(int(jv), P2[m] / pr.P_cat)
        c = ph.ttv_prefactor(pr.P_cat, js, deltas) * ph.ME            # days per Earth mass
        kap = ph.rv_semi_amplitude(ph.ME, P2)                          # m/s per Earth mass
        # ECC_FLOOR: even 'circular' systems have forced/residual eccentricities; never treat the
        # TTV phase/amplitude <-> RV link as exact (it is what breaks first under N-body truth)
        sigZ = np.sqrt(fj ** 2 + gj ** 2) * np.hypot(pr.sigma_e, ECC_FLOOR)
        self.c, self.kap, self.fj, self.sigZ = c, kap, fj, sigZ
        mu = self.masses[None, :, None]; phi = self.phis[None, None, :]
        cc = c[:, None, None]; ff = fj[:, None, None]; jj = js[:, None, None]
        # prior mean of W = c mu (-f) e^{i j phi}; isotropic sd s
        mr = cc * mu * (-ff) * np.cos(jj * phi); mi = cc * mu * (-ff) * np.sin(jj * phi)
        s2 = (np.abs(c) * 1.5 * sigZ / np.abs(deltas))[:, None, None] ** 2 * mu ** 2
        kc = kap[:, None, None] * mu * np.cos(phi); ks = kap[:, None, None] * mu * np.sin(phi)
        g = lambda a, b: Gt[:, a, b][:, None, None]
        b_ = lambda a: bt[:, a][:, None, None]
        # c_k = k^T Gkk k - 2 bk^T k ; d = bw - Gwk k
        ck = g(2, 2) * kc ** 2 + 2 * g(2, 3) * kc * ks + g(3, 3) * ks ** 2 - 2 * (b_(2) * kc + b_(3) * ks)
        d0 = b_(0) - g(0, 2) * kc - g(0, 3) * ks
        d1 = b_(1) - g(1, 2) * kc - g(1, 3) * ks
        # B = s2 * Gww + I (2x2), inverse analytic
        B00 = s2 * g(0, 0) + 1; B01 = s2 * g(0, 1); B11 = s2 * g(1, 1) + 1
        det = B00 * B11 - B01 ** 2
        i00, i01, i11 = B11 / det, -B01 / det, B00 / det
        Gm0 = g(0, 0) * mr + g(0, 1) * mi; Gm1 = g(0, 1) * mr + g(1, 1) * mi
        t1 = mr * (i00 * Gm0 + i01 * Gm1) + mi * (i01 * Gm0 + i11 * Gm1)           # m^T B^-1 G m
        t2 = s2 * (d0 * (i00 * d0 + i01 * d1) + d1 * (i01 * d0 + i11 * d1))        # s2 d^T B^-1 d
        t3 = 2 * (d0 * (i00 * mr + i01 * mi) + d1 * (i01 * mr + i11 * mi))         # 2 d^T B^-1 m
        logL = -0.5 * (self.c0 + ck + t1 - t2 - t3) - 0.5 * np.log(det)
        self._cache = dict(s2=s2, mr=mr, mi=mi, kc=kc, ks=ks, d0=d0, d1=d1, i00=i00, i01=i01, i11=i11)
        # priors: alias uniform, Delta uniform (cell width), log-mass uniform, phi uniform
        logprior = np.log(wcell_all)[:, None, None] - np.log(N_MASS * n_phi)
        lw1 = logL + logprior + np.log(1 - pr.p_none)
        lw0 = self.logL0 + np.log(pr.p_none)
        mx = max(lw1.max(), lw0)
        w1 = np.exp(lw1 - mx); w0 = np.exp(lw0 - mx)
        Zt = w1.sum() + w0
        self.p_none = float(w0 / Zt)
        self.w = w1 / Zt                                # (C, N_MASS, N_PHI), sums to 1 - p_none
        self.w_cm = self.w.sum(-1)                      # (C, N_MASS)
        self.log_bf = float(np.log(max(w1.sum(), 1e-300) / (1 - pr.p_none)) - np.log(max(w0, 1e-300) / pr.p_none))

    # ---- answer-level summaries ----
    def _mass_window(self, m_center):
        return np.abs(np.log(self.masses / m_center)) < MASS_TOL

    def prob_answer(self, period, mass):
        """Posterior prob. that (period within tol) and that (period & mass within tol)."""
        pm = np.abs(self.P2 / period - 1) < PERIOD_TOL
        pp = float(self.w_cm[pm].sum())
        mm = self._mass_window(mass)
        return pp, float(self.w_cm[pm][:, mm].sum()), float(self.w_cm[:, mm].sum())

    def best_planet_answer(self):
        """(period, mass, p_period, p_both, p_mass) maximising posterior mass in the tolerance box."""
        wc = self.w_cm.sum(1)
        if wc.sum() <= 0:
            return self.prior.P_cat * 2, 10.0, 0.0, 0.0, 0.0
        order = np.argsort(self.P2)
        Ps, ws = self.P2[order], wc[order]
        cs = np.concatenate([[0], np.cumsum(ws)])
        lo = np.searchsorted(Ps, Ps / (1 + PERIOD_TOL * 0.98)); hi = np.searchsorted(Ps, Ps * (1 + PERIOD_TOL * 0.98), "right")
        win = cs[hi] - cs[lo]
        ib = int(np.argmax(win))
        period = float(Ps[ib])
        pm = np.abs(self.P2 / period - 1) < PERIOD_TOL
        wm = self.w_cm[pm].sum(0)
        lm = np.log(self.masses)
        tot = np.array([wm[np.abs(lm - x) < MASS_TOL].sum() for x in lm])
        im = int(np.argmax(tot))
        # centre the mass estimate on the weighted log-mean inside the best window
        sel = np.abs(lm - lm[im]) < MASS_TOL
        mass = float(np.exp((wm[sel] * lm[sel]).sum() / max(wm[sel].sum(), 1e-300)))
        pp, pb, pmass = self.prob_answer(period, mass)
        return period, mass, pp, pb, pmass

    def prob_truth(self, sp):
        """Posterior probability of the *true* answer (the identifiability statistic q)."""
        if not sp.has_planet:
            return self.p_none
        return self.prob_answer(sp.P2, sp.m2)[1]

    def summary(self):
        period, mass, pp, pb, pmass = self.best_planet_answer()
        return dict(p_none=self.p_none, period=period, mass=mass, p_period=pp, p_both=pb, p_mass=pmass,
                    log_bf=self.log_bf)

    # ---- posterior sampling (for EIG) ----
    def sample(self, m, rng):
        """Draw m joint samples. Returns dict with label, and callables' ingredients."""
        C, NM, NP = self.w.shape
        flat = np.concatenate([[self.p_none], self.w.ravel()])
        flat = flat / flat.sum()
        idx = rng.choice(len(flat), size=m, p=flat)
        is0 = idx == 0
        k = np.maximum(idx - 1, 0)
        ci, mi_, pi_ = np.unravel_index(k, (C, NM, NP))
        ca = self._cache
        gi = lambda name: ca[name][ci, mi_ if ca[name].shape[1] > 1 else 0, pi_ if ca[name].shape[2] > 1 else 0]
        s2 = gi("s2"); mr = gi("mr"); mi2 = gi("mi"); d0 = gi("d0"); d1 = gi("d1")
        i00 = gi("i00"); i01 = gi("i01"); i11 = gi("i11")
        # W | rest ~ N(B^-1 (s2 d + m), s2 B^-1)
        h0 = s2 * d0 + mr; h1 = s2 * d1 + mi2
        w_mean0 = i00 * h0 + i01 * h1; w_mean1 = i01 * h0 + i11 * h1
        z = rng.normal(size=(m, 2))
        l00 = np.sqrt(np.maximum(s2 * i00, 0)); l10 = np.where(l00 > 0, s2 * i01 / np.maximum(l00, 1e-300), 0.0)
        l11 = np.sqrt(np.maximum(s2 * i11 - l10 ** 2, 0))
        Wr = w_mean0 + l00 * z[:, 0]; Wi = w_mean1 + l10 * z[:, 0] + l11 * z[:, 1]
        mass = self.masses[mi_]; phi = self.phis[pi_]
        K2 = self.kap[ci] * mass
        u = np.stack([Wr, Wi, K2 * np.cos(phi), K2 * np.sin(phi)], 1)
        u[is0] = 0.0
        # eta | u ~ N(Minv (be - Geu u), Minv)
        Gue = self.Gue[ci]                                       # (m, 4u, 4e)
        mean_e = (self.be[None, :] - np.einsum("mij,mi->mj", Gue, u)) @ self.Minv
        Lc = np.linalg.cholesky(self.Minv + 1e-18 * np.eye(4))
        eta = mean_e + rng.normal(size=(m, 4)) @ Lc.T
        js = self.js[ci]; deltas = self.deltas[ci]
        # label: 0 for none, else 1 + alias_id * NB + mass bin
        alias_id = ((js - 2) * 2 + (deltas > 0)).astype(int)
        mbin = np.minimum((np.log(mass / self.masses[0]) / (2 * MASS_TOL / 1.5)).astype(int), 15)
        label = np.where(is0, 0, 1 + alias_id * 16 + mbin)
        return dict(is0=is0, eta=eta, u=u, nu=-(js - 1) * deltas / (1 + deltas), P2=self.P2[ci],
                    label=label, mass=mass)

    def predict(self, S, epochs, times):
        """Noise-free predictions for candidate transits (residual, days) and RVs (m/s). (m, n_cand)."""
        pr = self.prior
        e = np.asarray(epochs, float)[None, :]; t = np.asarray(times, float)[None, :] - pr.T0_cat
        a = TWO_PI * S["nu"][:, None] * e
        tr = S["eta"][:, 0:1] + S["eta"][:, 1:2] * e + S["u"][:, 0:1] * np.sin(a) + S["u"][:, 1:2] * np.cos(a)
        w = TWO_PI * t / S["P2"][:, None]
        rv = (S["eta"][:, 2:3] - S["eta"][:, 3:4] * np.sin(TWO_PI * t / pr.P_cat)
              - S["u"][:, 2:3] * np.sin(w) - S["u"][:, 3:4] * np.cos(w))
        return tr, rv


def label_eig(pred, labels, sigma):
    """Mutual information I(y; label) for y = pred + N(0, sigma^2), pairwise (Kolchinsky-Tracey,
    KL variant) mixture-entropy estimator. pred: (m, n_cand). Returns (n_cand,) in nats."""
    m = pred.shape[0]
    d2 = (pred[:, None, :] - pred[None, :, :]) ** 2 / (2 * sigma ** 2)
    Kmat = np.exp(-d2)                                           # (m, m, n_cand)
    same = (labels[:, None] == labels[None, :]).astype(float)
    n_l = same.sum(1)
    h_all = -np.log(Kmat.mean(1))                                # (m, n_cand)
    h_grp = -np.log(np.einsum("ijc,ij->ic", Kmat, same) / n_l[:, None])
    return np.maximum((h_all - h_grp).mean(0), 0.0)
