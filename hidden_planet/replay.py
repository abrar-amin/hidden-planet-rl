"""'Real replay' evaluation: the transit times are ARCHIVAL Kepler measurements (Holczer et al. 2016,
ApJS 225, 9, table 3; copies in data/real/), not simulations.

The agent can only reveal epochs that exist in the archive (missing epochs are refused at no cost) and
no RVs are available. Targets are anonymised: a random 151-epoch window, epochs renumbered from 0, and
all times multiplied by a random factor so the period does not identify the system.

Targets: KOI-142 / Kepler-88 (a real perturber near 2:1, but 0.6 Jupiter masses: outside the env's
mass range and far beyond the first-order regime), three quiet hot Jupiters (TrES-2, HAT-P-7, Kepler-6:
no perturber), and Kepler-17 (spot-induced timing noise, no perturber).
"""
from __future__ import annotations

import os

import numpy as np

from .reward import LAMBDA, INVALID
from .tasks import NMAX, BUDGET, COST, SystemParams, _rng
from .text_env import HiddenPlanetSession

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real")
TARGETS = {   # koi: (kind, true perturber period [d], mass [Earth], description)
    "142.01": ("planet", 22.34, 199.0, "Kepler-88 b: real 2:1 perturber (Kepler-88 c)"),
    "1.01": ("null", None, None, "TrES-2 b: quiet"),
    "2.01": ("null", None, None, "HAT-P-7 b: quiet"),
    "17.01": ("null", None, None, "Kepler-6 b: quiet"),
    "203.01": ("activity", None, None, "Kepler-17 b: starspot timing noise, no perturber"),
}


def load_koi(koi):
    rows = []
    for line in open(os.path.join(DATA_DIR, f"koi_{koi}.tsv")):
        if line.startswith("#") or not line.strip():
            continue
        c = line.rstrip("\n").split("\t")
        try:
            n, tn, oc, e = int(c[2]), float(c[3]), float(c[4]), float(c[5])
            out = int(c[-2]) if c[-2].strip() else 0
        except (ValueError, IndexError):
            continue
        if out == 0 and e > 0:
            rows.append((n, tn + oc / 1440.0, e / 1440.0))
    a = np.array(rows)
    return a[:, 0].astype(int), a[:, 1], a[:, 2]


class ReplayEpisode:
    """Same interface as tasks.Episode, backed by archival transit times."""

    def __init__(self, koi, seed, budget=BUDGET):
        self.koi = koi
        r = _rng(seed, f"replay/{koi}")
        N, t, e = load_koi(koi)
        P_true = np.polyfit(N, t, 1)[0]
        n0 = int(r.integers(N.min(), max(N.max() - NMAX, N.min()) + 1))
        m = (N >= n0) & (N <= n0 + NMAX)
        self.scale = float(r.uniform(6.0, 18.0) / P_true)
        shift = float(r.uniform(0.3, 1.0))
        self._pool = {int(k - n0): float((tt - t[m][0]) * self.scale + shift + (N[m][0] - n0) * P_true * self.scale) for k, tt in zip(N[m], t[m])}
        ks = np.array(sorted(self._pool)); ts = np.array([self._pool[k] for k in ks])
        P_fit, T0_fit = np.polyfit(ks, ts, 1)
        self.P_cat = float(round(P_fit + r.normal(0, 1e-4), 5)); self.T0_cat = float(round(T0_fit + r.normal(0, 2e-3), 4))
        self.sig_t = float(np.median(e[m]) * self.scale)
        self.budget, self.spent = budget, 0.0
        self.transits, self.rvs = {}, []
        self.t_max = NMAX * self.P_cat

    @property
    def remaining(self):
        return self.budget - self.spent

    def observe_transit(self, n):
        n = int(n)
        if not (0 <= n <= NMAX):
            return None, f"epoch {n} outside 0..{NMAX}"
        if n in self.transits:
            return None, f"epoch {n} already observed"
        if n not in self._pool:
            return None, f"no archival transit at epoch {n} (not charged)"
        if self.spent + COST["transit"] > self.budget + 1e-9:
            return None, "insufficient budget"
        self.spent += COST["transit"]
        self.transits[n] = self._pool[n]
        return self._pool[n], None

    def observe_rv(self, t):
        return None, "no archival RVs exist for this target (not charged)"

    def data(self):
        n = np.array(sorted(self.transits), int)
        return n, np.array([self.transits[k] for k in n], float), np.zeros(0), np.zeros(0)


class ReplaySession(HiddenPlanetSession):
    def __init__(self, koi, seed, allow_other=False, cost_per_unit=None, max_turns=None, reward_mode="partial", wrong_penalty=None):
        ep = ReplayEpisode(koi, seed)
        kind, P2, m2, _ = TARGETS[koi]
        sp = SystemParams(seed=seed, level=5, P_cat=ep.P_cat, T0_cat=ep.T0_cat, P1=ep.P_cat, T0=ep.T0_cat, sig_t=ep.sig_t, sig_rv=3.0,
                          sigma_e=0.015, m1=10.0, e1=0.0, w1=0.0, gamma=0.0, has_planet=kind == "planet")
        if kind == "planet":
            sp.P2, sp.m2, sp.j = P2 * ep.scale, m2, 2
            sp.delta = sp.P2 / sp.P1 / 2 - 1
        sp.extra["replay"] = koi; sp.extra["split"] = f"replay-{koi}"
        sp.__dict__["_sim_cache"] = {"replay": None}
        self.sp, self.ep, self.backend, self.kind = sp, ep, "replay", kind
        self._q_ref, self.abstain_aware = 1.0, True
        self.max_turns = max_turns or 14
        self.lam = LAMBDA if cost_per_unit is None else float(cost_per_unit)
        self.allow_other, self.answer, self.done, self.n_calls, self.n_errors = allow_other, None, False, 0, 0
        self.reward_mode, self.wrong_penalty, self._last_fit = reward_mode, wrong_penalty, None
        self.tool_level = "oracle"

    def user_prompt(self):
        return (super().user_prompt()
                + " Archive-only target: some epochs were never observed and are refused at no cost, and no RVs can be taken.")

    def _grade(self):
        a = self.answer or {}
        v = a.get("verdict")
        correct = 0.0
        strict0 = self.reward_mode == "strict0"
        W = 1.0 if self.wrong_penalty is None else self.wrong_penalty
        if v not in ("none", "planet", "cannot_determine") + (("other",) if self.allow_other else ()):
            s, v = INVALID, "invalid"
        elif self.kind == "planet":
            if v == "planet":
                try:
                    pok = abs(float(a["period_days"]) / self.sp.P2 - 1) < 0.005
                    mok = abs(np.log(float(a["mass_earth"]) / self.sp.m2)) < np.log(1.5)
                    s = (0.5 if pok else -0.25) + (0.5 if mok else -0.25); correct = float(pok and mok)
                    if strict0:
                        s = 1.0 if (pok and mok) else -W
                except (KeyError, TypeError, ValueError):
                    s, v = INVALID, "invalid"
            else:
                s = {"none": -0.5, "cannot_determine": 0.4, "other": 0.4}[v]     # without RVs the alias and mass are open
                if strict0:
                    s = {"none": -W, "cannot_determine": 0.0, "other": 0.0}[v]
        elif self.kind == "null":
            s = {"none": 1.0, "cannot_determine": -0.3, "planet": -0.5, "other": -0.5}[v]; correct = float(v == "none")
            if strict0:
                s = {"none": 1.0, "cannot_determine": 0.0, "planet": -W, "other": -W}[v]
        else:
            s = {"none": 1.0, "other": 1.0, "cannot_determine": 0.4, "planet": -0.5}[v]; correct = float(v in ("none", "other"))
            if strict0:
                s = {"none": 1.0, "other": 1.0, "cannot_determine": 0.0, "planet": -W}[v]
        m = {"score": float(s), "spent": float(self.ep.spent), "correct": correct, "reward_std_cost": float(s - LAMBDA * self.ep.spent),
             "determinable": 1.0, "has_planet": float(self.kind == "planet"), "verdict_none": float(v == "none"), "verdict_planet": float(v == "planet"),
             "verdict_cannot": float(v == "cannot_determine"), "verdict_other": float(v == "other"), "invalid": float(v == "invalid"),
             "abstain_correct": 0.0, "n_transits": float(len(self.ep.transits)), "n_rvs": 0.0, "tool_errors": float(self.n_errors),
             "false_planet_claim": float(v == "planet" and self.kind != "planet")}
        return float(s - self.lam * self.ep.spent), m


def replay_sessions(n_per_target=4, **kw):
    return [ReplaySession(koi, seed, **kw) for koi in TARGETS for seed in range(n_per_target)]
