"""Procedural task generator: priors, difficulty curriculum, deterministic seeding, splits."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, asdict

import numpy as np

from . import physics as ph

NMAX = 150                 # last observable transit epoch (epochs 0..NMAX)
BUDGET = 60.0
COST = {"transit": 1.0, "rv": 3.0}
SIGMA_T0 = 0.003           # catalogue ephemeris uncertainties (days)
SIGMA_P = 2.0e-4
GAMMA_SD = 10.0            # RV zero-point prior sd (m/s)
K1_SD = 15.0               # inner-planet K prior sd used by the inference engine (m/s)
RESONANCES = (2, 3, 4)
DELTA_RANGE = (0.01, 0.05)
MASS_RANGE = (4.0, 80.0)   # perturber mass prior, Earth masses (log-uniform)
P_NONE = 0.35

# Difficulty curriculum. sigma_e = Rayleigh scale of free eccentricities (both planets).
LEVELS = {
    # level 0 = EASY: circular 2:1, quiet star, massive perturbers; solvable in ~5 tool calls
    0: dict(sigma_e=0.0, sig_rv=(1.0, 2.0), sig_t_min=(0.5, 1.5), mass=(10.0, 80.0), res=(2,), max_turns=8),
    # levels 1-2: 3:2 and 4:3 aliases as well (experimental: the first-order model is not adequate there)
    1: dict(sigma_e=0.0, sig_rv=(1.0, 3.0), sig_t_min=(1.0, 3.0), mass=(4.0, 80.0), res=(2, 3, 4)),
    2: dict(sigma_e=0.015, sig_rv=(1.0, 5.0), sig_t_min=(0.7, 3.0), mass=(4.0, 80.0), res=(2, 3, 4)),
    # level 5 = HARD (the default): level 2 restricted to the 2:1 resonance, the only regime where the
    # analytic TTVs were validated against N-body
    5: dict(sigma_e=0.015, sig_rv=(1.0, 5.0), sig_t_min=(0.7, 3.0), mass=(4.0, 80.0), res=(2,), max_turns=14),
    # level 3: noisier RVs + lighter perturbers -> ~1/4 of systems are genuinely undeterminable
    3: dict(sigma_e=0.015, sig_rv=(2.0, 8.0), sig_t_min=(0.5, 2.0), mass=(3.0, 80.0), res=(2, 3, 4)),
}
EASY, HARD = 0, 5
DEFAULT_LEVEL = HARD
LEVEL_NAMES = {"easy": EASY, "hard": HARD}       # levels 1-3 (3:2, 4:3 aliases) are experimental
SPLIT_OFFSETS = {"train": 0, "val": 1_000_000_000, "test": 2_000_000_000}


@dataclass
class SystemParams:
    seed: int
    level: int
    P_cat: float               # catalogue period (days)  -- shown to the agent
    T0_cat: float              # catalogue epoch (days)   -- shown to the agent
    P1: float                  # true mean period
    T0: float                  # true mean epoch
    sig_t: float               # transit-time noise (days)
    sig_rv: float              # RV noise (m/s)
    sigma_e: float             # eccentricity scale (shown to the agent)
    m1: float                  # inner mass (Earth)
    e1: float
    w1: float
    gamma: float               # RV zero point
    has_planet: bool
    j: int = 2
    delta: float = 0.02
    m2: float = 0.0            # perturber mass (Earth)
    e2: float = 0.0
    w2: float = 0.0
    lam2_0: float = 0.0        # perturber mean longitude at t = T0
    P2: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def z1(self):
        return self.e1 * np.exp(1j * self.w1)

    @property
    def z2(self):
        return self.e2 * np.exp(1j * self.w2)

    def to_dict(self):
        return asdict(self)


def _rng(seed: int, stream: str) -> np.random.Generator:
    h = hashlib.sha256(f"hidden-planet/{stream}/{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


# RV availability. "ideal" (default): RVs always available at the level's precision.
# "realistic": each host star gets a V magnitude and an activity jitter (both shown to the agent); RV precision is
# photon noise (1 m/s at V = 11.5, i.e. a large-telescope spectrograph, scaling as 10^(0.2 (V - 11.5))) plus jitter in quadrature, and stars fainter than
# V = 14.5 cannot be observed in RV at all. Select with the env var HIDDEN_PLANET_RV=realistic or set_rv_mode().
RV_MODE = os.environ.get("HIDDEN_PLANET_RV", "ideal")
RV_VMAG_LIMIT = 14.5


def set_rv_mode(mode: str):
    global RV_MODE
    if mode not in ("ideal", "realistic"):
        raise ValueError(mode)
    RV_MODE = mode
    os.environ["HIDDEN_PLANET_RV"] = mode          # inherited by worker processes


def _apply_rv_realism(sp: SystemParams, seed: int, split: str):
    r = _rng(seed + SPLIT_OFFSETS[split], f"rvstar/L{sp.level}")
    v = 0.0
    while not (8.0 <= v <= 16.0):                   # Kepler/TESS-like mix: most hosts are V = 10.5-14.5
        v = float(r.normal(12.5, 2.0))
    jitter = float(np.clip(np.exp(r.normal(np.log(1.5), 0.7)), 0.5, 20.0))
    phot = 10 ** (0.2 * (v - 11.5))
    sp.sig_rv = float(np.hypot(phot, jitter))
    sp.extra["rv"] = dict(vmag=round(v, 1), jitter=round(jitter, 1), available=bool(v <= RV_VMAG_LIMIT))


def rv_available(sp: SystemParams) -> bool:
    return bool(sp.extra.get("rv", {}).get("available", True))


def sample_system(seed: int, level: int = DEFAULT_LEVEL, split: str = "train") -> SystemParams:
    """Deterministic: (seed, level, split) -> system. Splits use disjoint seed spaces."""
    L = LEVELS[level]
    rng = _rng(seed + SPLIT_OFFSETS[split], f"system/L{level}")
    P_cat = float(rng.uniform(5, 20))
    T0_cat = float(rng.uniform(0, 1))
    P1 = P_cat + float(rng.normal(0, SIGMA_P))
    T0 = T0_cat + float(rng.normal(0, SIGMA_T0))
    sig_t = float(rng.uniform(*L["sig_t_min"])) / 1440.0
    sig_rv = float(np.exp(rng.uniform(np.log(L["sig_rv"][0]), np.log(L["sig_rv"][1]))))
    se = L["sigma_e"]
    m1 = float(np.exp(rng.uniform(np.log(3), np.log(30))))
    e1 = float(min(rng.rayleigh(se), 0.08)) if se > 0 else 0.0
    w1 = float(rng.uniform(0, 2 * np.pi))
    gamma = float(rng.normal(0, GAMMA_SD))
    has = bool(rng.random() > P_NONE)
    j = int(rng.choice(L["res"]))
    delta = float(rng.choice([-1, 1]) * rng.uniform(*DELTA_RANGE))
    m2 = float(np.exp(rng.uniform(np.log(L["mass"][0]), np.log(L["mass"][1]))))
    e2 = float(min(rng.rayleigh(se), 0.08)) if se > 0 else 0.0
    w2 = float(rng.uniform(0, 2 * np.pi))
    lam2_0 = float(rng.uniform(0, 2 * np.pi))
    sp = SystemParams(seed=seed, level=level, P_cat=P_cat, T0_cat=T0_cat, P1=P1, T0=T0, sig_t=sig_t,
                      sig_rv=sig_rv, sigma_e=se, m1=m1, e1=e1, w1=w1, gamma=gamma, has_planet=has)
    if has:
        sp.j, sp.delta, sp.m2, sp.e2, sp.w2, sp.lam2_0 = j, delta, m2, e2, w2, lam2_0
        sp.P2 = P1 * j / (j - 1) * (1 + delta)
    sp.extra["split"] = split
    if RV_MODE == "realistic":
        _apply_rv_realism(sp, seed, split)
    return sp


def target_id(sp: SystemParams) -> str:
    """Anonymous label shown to the agent (never the generator seed)."""
    h = hashlib.sha256(f"hidden-planet/target/{sp.extra.get('split', 'train')}/{sp.level}/{sp.seed}".encode()).hexdigest()
    return "T-" + h[:8].upper()


DEFAULT_BACKEND = "ttvfast"      # ground truth: 'ttvfast' (N-body, default) | 'rebound' (N-body, IAS15) | 'analytic'
BACKENDS = ("ttvfast", "rebound", "analytic")


def resolve_backend(backend):
    b = DEFAULT_BACKEND if backend in (None, "default") else ("rebound" if backend == "nbody" else backend)
    if b not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    return b


class Simulator:
    """Noise-free observables for a system.

    backend 'ttvfast' / 'rebound': the truth is an N-body integration (TTVFast, Deck et al. 2014; or
    REBOUND IAS15). 'analytic': first-order LXW TTVs + Keplerian RVs (the model the inference engine,
    the EIG scheduler and the agent-facing fit tools assume). Systems without a perturber are exactly
    Keplerian in every backend."""

    def __init__(self, sp: SystemParams, backend: str | None = None):
        backend = resolve_backend(backend)
        self.sp, self.backend = sp, backend
        n = np.arange(NMAX + 1)
        self._nb = None
        if backend != "analytic" and sp.has_planet:
            try:
                self._init_nbody()
            except RuntimeError:                 # unstable / missed transits: fall back, and say so
                sp.extra["backend_fallback"] = "analytic"
                self.backend = backend = "analytic"
        if backend == "analytic" or not sp.has_planet:
            ttv = 0.0 * n
            if sp.has_planet:
                Z = ph.zfree_combo(sp.j, sp.z1, sp.z2, sp.delta)
                ttv = ph.analytic_ttv(n, sp.P1, sp.j, sp.delta, sp.m2 * ph.ME, Z, sp.lam2_0)
            self.t_transit = sp.T0 + n * sp.P1 + ttv
        if sp.extra.get("ooc"):
            from . import ooc
            self.t_transit = self.t_transit + ooc.ttv_extra(sp, n)

    def _init_nbody(self):
        """N-body truth. The run's own mean inner ephemeris is mapped onto (T0, P1) so the catalogue
        ephemeris stays valid; time is rescaled by P1/P1_mean (a ~1e-4 change) and the scoring
        truth P2 becomes the perturber's measured mean period on that time axis."""
        sp = self.sp
        if "P2_osc" in sp.extra:                 # already converted once: restart from the osculating values
            sp.P2 = sp.extra["P2_osc"]; sp.delta = sp.extra["delta_osc"]
        mu1, mu2 = sp.m1 * ph.ME, sp.m2 * ph.ME
        # start 2.5 d before inner epoch 0 so that every observable time lies inside the integration
        lead = 2.5
        lam1_s = -2 * np.pi * lead / sp.P1
        lam2_s = sp.lam2_0 - 2 * np.pi * lead / sp.P2
        f1, f2 = ph.forced_ecc(sp.j, sp.delta, mu1, mu2, sp.P1, sp.P2, sp.j * lam2_s - (sp.j - 1) * lam1_s)
        self._z1, self._z2 = sp.z1 + f1, sp.z2 + f2
        grid = np.linspace(1.0, (NMAX + 1) * sp.P1 + 2 * lead, 6000)   # first time > one step: see the TTVFast RV quirk
        simulate = ph.ttvfast_simulate if self.backend == "ttvfast" else ph.nbody_simulate
        out = simulate(sp.P1, mu1, self._z1, sp.P2, mu2, self._z2, lam1_s, lam2_s, NMAX + 1, grid)
        if out["stable"] and not np.any(out["rv"] != 0.0):
            raise RuntimeError("N-body backend returned no RVs")
        if not out["stable"] or len(out["t2"]) < 4:
            raise RuntimeError("unstable system")
        T0m, P1m, res = ph.linear_ephemeris(out["t1"])
        self._scale = sp.P1 / P1m
        self.t_transit = sp.T0 + np.arange(NMAX + 1) * sp.P1 + res * self._scale
        self._T0m = T0m
        self._nb = out
        _, P2m, _ = ph.linear_ephemeris(out["t2"])
        sp.extra["P2_osc"] = sp.P2; sp.extra["delta_osc"] = sp.delta
        sp.P2 = float(P2m * self._scale)
        sp.delta = float(sp.P2 / sp.P1 * (sp.j - 1) / sp.j - 1)

    def rv(self, t):
        sp = self.sp
        t = np.asarray(t, float)
        if self._nb is not None:
            v = sp.gamma + np.interp((t - sp.T0) / self._scale + self._T0m, self._nb["rv_times"], self._nb["rv"])
            if sp.extra.get("ooc"):
                from . import ooc
                v = v + ooc.rv_extra(sp, t)
            return v
        K1 = ph.rv_semi_amplitude(sp.m1 * ph.ME, sp.P1, sp.e1)
        v = sp.gamma + ph.kepler_rv(t, K1, sp.P1, ph.mean_longitude_at_transit(sp.e1, sp.w1),
                                    sp.e1, sp.w1, t0=sp.T0)
        if sp.has_planet:
            K2 = ph.rv_semi_amplitude(sp.m2 * ph.ME, sp.P2, sp.e2)
            v = v + ph.kepler_rv(t, K2, sp.P2, sp.lam2_0, sp.e2, sp.w2, t0=sp.T0)
        if sp.extra.get("ooc"):
            from . import ooc
            v = v + ooc.rv_extra(sp, t)
        return v


class Episode:
    """One observing campaign. Deterministic given (system, noise_seed, action sequence)."""

    def __init__(self, sp: SystemParams, noise_seed: int | None = None, backend: str | None = None,
                 budget: float = BUDGET):
        self.sp = sp
        backend = resolve_backend(backend)
        cache = sp.__dict__.setdefault("_sim_cache", {})      # noise-free truth, shared by episodes
        if len(cache) and backend not in cache:
            raise ValueError("a SystemParams object is bound to one backend; sample the system again for another")
        if backend not in cache:
            cache[backend] = Simulator(sp, backend)
        self.sim = cache[backend]
        # noise depends on (seed, split), so train seed k and test seed k draw different noise
        ns = (sp.seed + SPLIT_OFFSETS.get(sp.extra.get("split", "train"), 0)) if noise_seed is None else noise_seed
        r = _rng(ns, f"noise/L{sp.level}")
        self._tnoise = r.normal(0, sp.sig_t, NMAX + 1)       # fixed per epoch (order independent)
        self._noise_seed = ns
        self.budget, self.spent = budget, 0.0
        self.transits: dict[int, float] = {}
        self.rvs: list[tuple[float, float]] = []
        self.t_max = NMAX * sp.P_cat

    @property
    def remaining(self):
        return self.budget - self.spent

    def observe_transit(self, n: int):
        n = int(n)
        if not (0 <= n <= NMAX):
            return None, f"epoch {n} outside 0..{NMAX}"
        if n in self.transits:
            return None, f"epoch {n} already observed"
        if self.spent + COST["transit"] > self.budget + 1e-9:
            return None, "insufficient budget"
        self.spent += COST["transit"]
        y = float(self.sim.t_transit[n] + self._tnoise[n])
        self.transits[n] = y
        return y, None

    def observe_rv(self, t: float):
        if not rv_available(self.sp):
            return None, "RVs are not possible for this star (too faint); not charged"
        t = float(t)
        if any(round(t, 4) == round(r[0], 4) for r in self.rvs):       # same key as the noise draw
            return None, f"time {t:.2f} already observed"
        if not (0.0 <= t <= self.t_max):
            return None, f"time {t:.2f} outside 0..{self.t_max:.1f}"
        if self.spent + COST["rv"] > self.budget + 1e-9:
            return None, "insufficient budget"
        self.spent += COST["rv"]
        # RV noise is a function of (system, time), not of purchase order: two policies buying the same time see the same value
        v = float(self.sim.rv(t) + _rng(self._noise_seed, f"rvnoise/L{self.sp.level}/{round(t, 4)}").normal(0, self.sp.sig_rv))
        self.rvs.append((t, v))
        return v, None

    def data(self):
        n = np.array(sorted(self.transits), int)
        y = np.array([self.transits[k] for k in n], float)
        t = np.array([r[0] for r in self.rvs], float)
        v = np.array([r[1] for r in self.rvs], float)
        return n, y, t, v
