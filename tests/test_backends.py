import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(all="ignore")
from hidden_planet import physics as ph
from hidden_planet.tasks import sample_system, Simulator, Episode, NMAX, DEFAULT_BACKEND, resolve_backend


def _planet_seed(level=5):
    return next(s for s in range(100) if sample_system(s, level, "val").has_planet)


def test_default_backend_and_switch():
    assert DEFAULT_BACKEND == "ttvfast" and resolve_backend(None) == "ttvfast" and resolve_backend("nbody") == "rebound"
    with pytest.raises(ValueError):
        resolve_backend("bogus")
    sp = sample_system(_planet_seed(), 5, "val"); Episode(sp)
    with pytest.raises(ValueError):
        Episode(sp, backend="analytic")                    # one SystemParams object, one backend


def test_ttvfast_rv_matches_kepler_for_negligible_perturber():
    s = _planet_seed()
    a = sample_system(s, 5, "val"); a.m2 = 1e-4
    b = sample_system(s, 5, "val"); b.m2 = 1e-4
    A, B = Simulator(a, "ttvfast"), Simulator(b, "analytic")
    t = np.linspace(0, NMAX * a.P_cat, 300)
    assert np.std(A.rv(t) - B.rv(t)) < 0.02                # m/s: sign, units, phase, time mapping
    assert np.max(np.abs(A.t_transit - B.t_transit)) * 86400 < 1.0


def test_ttvfast_matches_rebound_in_test_particle_limit():
    sp = sample_system(_planet_seed(), 5, "val")
    rv_t = np.linspace(1, 100 * sp.P1, 30)
    args = (sp.P1, 1e-12, sp.z1, sp.P2, sp.m2 * ph.ME, sp.z2, -0.6, sp.lam2_0, 110, rv_t)
    A, B = ph.ttvfast_simulate(*args), ph.nbody_simulate(*args)
    assert np.max(np.abs(A["t1"] - B["t1"])) * 86400 < 5.0 and np.max(np.abs(A["rv"] - B["rv"])) < 0.01


def test_nbody_truth_is_deterministic_and_complete():
    s = _planet_seed()
    a, b = sample_system(s, 5, "val"), sample_system(s, 5, "val")
    A, B = Simulator(a), Simulator(b)
    assert len(A.t_transit) == NMAX + 1 and np.all(np.diff(A.t_transit) > 0) and np.array_equal(A.t_transit, B.t_transit)
    assert a.P2 == b.P2 and abs(a.P2 / a.extra["P2_osc"] - 1) < 0.004 and "backend_fallback" not in a.extra
    n = sample_system(next(k for k in range(100) if not sample_system(k, 5, "val").has_planet), 5, "val")
    N = Simulator(n)
    assert np.allclose(np.diff(N.t_transit), n.P1)


def test_ttvfast_rvs_nonzero_across_period_range():
    """Regression: TTVFast zeroes every RV if one RV time lies inside the first step (all systems with P1 > ~8 d)."""
    for seed in range(60):
        sp = sample_system(seed, 5, "val")
        if not sp.has_planet:
            continue
        S = Simulator(sp)
        t = np.linspace(0, NMAX * sp.P_cat, 200)
        K = ph.rv_semi_amplitude(sp.m2 * ph.ME, sp.P2)
        assert S.backend == "ttvfast" and np.std(S.rv(t)) > 0.3 * K
    with pytest.raises(ValueError):
        ph.ttvfast_simulate(10.0, 3e-5, 0j, 20.4, 7e-5, 0j, -0.5, 1.0, 20, np.array([0.1, 5.0]))
