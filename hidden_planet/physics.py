"""Physics for the hidden-planet environment.

Three backends produce the same observables (inner-planet transit times, stellar RVs):

* analytic: first-order near-resonant TTVs of Lithwick, Xie & Wu (2012, "LXW") plus Keplerian RVs.
* ttvfast:  symplectic N-body integration with TTVFast (Deck et al. 2014); the default ground truth.
* rebound:  direct N-body integration with REBOUND (IAS15), used for validation and as an optional backend.

Conventions
-----------
Units: days, AU, solar masses. Star mass 1 Msun. The observer sits on the +x axis, so a planet
transits when its true longitude (measured from the line of sight) is 0. Complex eccentricity
z = e * exp(i * pomega). lambda^j = j*lambda2 - (j-1)*lambda1. Near a j:j-1 resonance,
Delta = (P2/P1)*(j-1)/j - 1.

LXW (their eqs. 8-11), inner-planet TTV:
    dt1 = |V1| sin(lambda^j + arg V1) = Im[V1 exp(i lambda^j)]
    V1  = P1 * mu2 / (pi j^(2/3) (j-1)^(1/3) Delta) * (-f - 1.5 conj(Zfree)/Delta)
    Zfree = f z1_free + g z2_free
RV sign: positive = star receding = -vx_star for an observer on +x:
    rv = -K [sin(theta) + e sin(pomega)],  theta = true longitude.
"""
from __future__ import annotations

import numpy as np

ME = 3.0035e-6          # Earth mass / solar mass
MJ = 9.5458e-4          # Jupiter mass / solar mass
# LXW12 Table 3 coefficients (evaluated at exact commensurability).
F_COEF = {2: -1.190, 3: -2.025, 4: -2.840, 5: -3.650}
G_COEF = {2: 0.4284, 3: 2.484, 4: 3.283, 5: 4.084}
TWO_PI = 2.0 * np.pi


def ttv_prefactor(P1, j, delta):
    """c such that V1 = c * mu2 * (-f - 1.5 conj(Z)/Delta)  [days per unit mass ratio]."""
    return P1 / (np.pi * j ** (2.0 / 3.0) * (j - 1) ** (1.0 / 3.0) * delta)


def coef_fg(j, delta, table=False):
    """(f, g) for the j:j-1 term. Default: evaluated at the actual period ratio (Laplace coefficients);
    table=True gives the LXW12 tabulated values at exact commensurability."""
    if table:
        return F_COEF[j], G_COEF[j]
    from .laplace import fg_array
    f, g = fg_array(j, j / (j - 1) * (1 + np.asarray(delta, float)))
    return (float(f), float(g)) if np.ndim(delta) == 0 else (f, g)


def ttv_complex_amplitude(P1, j, delta, mu2, zfree, f=None):
    f = coef_fg(j, delta)[0] if f is None else f
    return ttv_prefactor(P1, j, delta) * mu2 * (-f - 1.5 * np.conj(zfree) / delta)


def zfree_combo(j, z1, z2, delta=None):
    f, g = (F_COEF[j], G_COEF[j]) if delta is None else coef_fg(j, delta)
    return f * z1 + g * z2


def forced_ecc(j, delta, mu1, mu2, P1, P2, lamj):
    """LXW forced complex eccentricities (cf. their eq. A28); used to initialise N-body runs."""
    ph = np.exp(1j * lamj)
    f, g = coef_fg(j, delta)
    z1 = -mu2 * f / (j * delta) * (P1 / P2) ** (1.0 / 3.0) * ph
    z2 = -mu1 * g / (j * delta) * ph
    return z1, z2


def ttv_frequency_per_epoch(j, delta):
    """d(lambda^j)/dn in cycles per inner epoch (negative for Delta>0)."""
    return -(j - 1) * delta / (1.0 + delta)


def lambda_j(n, j, delta, lam2_0):
    """lambda^j at inner transit epoch n. lam2_0 = mean longitude of the perturber at epoch 0."""
    return j * lam2_0 + TWO_PI * ttv_frequency_per_epoch(j, delta) * np.asarray(n, float)


def analytic_ttv(n, P1, j, delta, mu2, zfree, lam2_0):
    V = ttv_complex_amplitude(P1, j, delta, mu2, zfree)
    return np.imag(V * np.exp(1j * lambda_j(n, j, delta, lam2_0)))


def rv_semi_amplitude(mu, P, e=0.0):
    """K in m/s for mass ratio mu (Mstar = 1 Msun, sin i = 1), period P in days."""
    return 28.4329 * (mu / MJ) * (P / 365.25) ** (-1.0 / 3.0) / np.sqrt(1.0 - e * e)


def k_to_mass_earth(K, P):
    return K / 28.4329 * MJ * (P / 365.25) ** (1.0 / 3.0) / ME


def solve_kepler(M, e, iters=8):
    E = M + e * np.sin(M)
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def true_longitude(lam, e, pomega):
    """True longitude from mean longitude for a Keplerian orbit."""
    if e == 0.0:
        return lam
    M = lam - pomega
    E = solve_kepler(M, e)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return pomega + nu


def mean_longitude_at_transit(e, pomega):
    """Mean longitude when the true longitude is 0 (i.e. at mid-transit)."""
    if e == 0.0:
        return 0.0
    nu = -pomega
    E = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    return pomega + E - e * np.sin(E)


def kepler_rv(t, K, P, lam_at_t0, e, pomega, t0=0.0):
    lam = lam_at_t0 + TWO_PI * (np.asarray(t, float) - t0) / P
    th = true_longitude(lam, e, pomega)
    return -K * (np.sin(th) + e * np.sin(pomega))


# ----------------------------------------------------------------------------------------------
# N-body (REBOUND)
# ----------------------------------------------------------------------------------------------
def nbody_simulate(P1, mu1, z1, P2, mu2, z2, lam1_0, lam2_0, n_epochs, rv_times=None,
                   integrator="ias15"):
    """Integrate star + 2 coplanar planets. Elements are osculating (Jacobi) at t=0.

    Returns dict with inner transit times (first n_epochs), outer conjunction times, RVs at
    rv_times (m/s, barycentric, receding positive), and a `stable` flag.
    """
    import rebound

    sim = rebound.Simulation()
    sim.units = ("day", "AU", "Msun")
    sim.integrator = integrator
    sim.add(m=1.0)
    for P, mu, z, lam in ((P1, mu1, z1, lam1_0), (P2, mu2, z2, lam2_0)):
        e = abs(z)
        sim.add(m=mu, P=P, e=e, pomega=float(np.angle(z)) if e > 0 else 0.0, l=float(lam))
    sim.move_to_com()
    if integrator == "whfast":
        sim.dt = P1 / 50.0
    ps = sim.particles
    t_end = (n_epochs + 2) * P1 * 1.02
    rv_times = np.sort(np.asarray(rv_times, float)) if rv_times is not None else np.array([])
    rv_out = np.zeros(len(rv_times))
    rv_i = 0
    AU_DAY_TO_MS = 1.495978707e11 / 86400.0
    tr = {1: [], 2: []}
    step = P1 / 20.0
    yold = {k: ps[k].y - ps[0].y for k in (1, 2)}
    told = sim.t
    stable = True
    a0 = (ps[1].a, ps[2].a)
    while sim.t < t_end and len(tr[1]) < n_epochs + 1:
        tnext = sim.t + step
        while rv_i < len(rv_times) and rv_times[rv_i] <= tnext:
            sim.integrate(rv_times[rv_i])
            rv_out[rv_i] = -ps[0].vx * AU_DAY_TO_MS
            rv_i += 1
        sim.integrate(tnext)
        tnew = sim.t
        for k in (1, 2):
            ynew = ps[k].y - ps[0].y
            if yold[k] < 0.0 <= ynew and (ps[k].x - ps[0].x) > 0.0:
                lo, hi = told, tnew
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    sim.integrate(mid)
                    if ps[k].y - ps[0].y < 0.0:
                        lo = mid
                    else:
                        hi = mid
                    if hi - lo < 1e-9:
                        break
                tr[k].append(0.5 * (lo + hi))
                sim.integrate(tnew)
            yold[k] = ps[k].y - ps[0].y
        told = tnew
        if not (0.5 * a0[0] < ps[1].a < 2 * a0[0] and 0.5 * a0[1] < ps[2].a < 2 * a0[1]):
            stable = False
            break
    while stable and rv_i < len(rv_times):
        sim.integrate(rv_times[rv_i])
        rv_out[rv_i] = -ps[0].vx * AU_DAY_TO_MS
        rv_i += 1
    return dict(t1=np.array(tr[1][:n_epochs]), t2=np.array(tr[2]), rv=rv_out, rv_times=rv_times,
                stable=stable and len(tr[1]) >= n_epochs)


def linear_ephemeris(times):
    n = np.arange(len(times))
    A = np.c_[np.ones_like(n, float), n]
    c = np.linalg.lstsq(A, times, rcond=None)[0]
    return c[0], c[1], times - A @ c


# ----------------------------------------------------------------------------------------------
# N-body (TTVFast; Deck, Agol, Holman & Nesvorny 2014, ApJ 787, 132)
# ----------------------------------------------------------------------------------------------
AU_DAY_TO_MS = 1.495978707e11 / 86400.0


def ttvfast_simulate(P1, mu1, z1, P2, mu2, z2, lam1_0, lam2_0, n_epochs, rv_times=None, steps_per_orbit=40):
    """Same contract as nbody_simulate, using TTVFast (symplectic, Jacobi elements at t=0).

    TTVFast puts the observer on +z with the sky in the x-y plane; for an edge-on orbit
    (i = 90 deg, Omega = 0) a transit happens at omega + f = 90 deg, so our line-of-sight
    longitude theta maps to omega_ttvfast = pomega + 90 deg and M = lambda - pomega.
    Both planets are integrated edge-on and coplanar; the perturber's own "transits" are used
    only to measure its mean period (it is declared non-transiting to the agent).
    RVs: TTVFast returns AU/day with the receding-positive sign convention, so m/s = AU_DAY_TO_MS * rv
    (checked against the Keplerian formula in tests/test_backends.py). Quirk of TTVFast: an RV time
    inside the first integration step silently zeroes ALL RV outputs, so every RV time must be
    later than one step after the start (enforced here).
    """
    import ttvfast

    def planet(P, mu, z, lam):
        e = float(abs(z)); pom = float(np.angle(z)) if e > 0 else 0.0
        return ttvfast.models.Planet(mass=float(mu), period=float(P), eccentricity=e, inclination=90.0, longnode=0.0,
                                     argument=float(np.degrees(pom) + 90.0) % 360.0,
                                     mean_anomaly=float(np.degrees(lam - pom)) % 360.0)

    total = (n_epochs + 3) * P1 * 1.02
    rv_times = np.sort(np.asarray(rv_times, float)) if rv_times is not None else np.array([])
    dt = P1 / steps_per_orbit
    if len(rv_times) and (rv_times[0] <= 1.5 * dt or rv_times[-1] > total):
        raise ValueError("rv_times must lie after the first 1.5 integration steps and before the end")
    kw = dict(rv_times=[float(t) for t in rv_times]) if len(rv_times) else {}
    res = ttvfast.ttvfast([planet(P1, mu1, z1, lam1_0), planet(P2, mu2, z2, lam2_0)], 1.0, 0.0, P1 / steps_per_orbit, total, **kw)
    idx, epoch, times = (np.asarray(res["positions"][k]) for k in range(3))
    ok = times > -1.5                                   # unused slots are -2
    idx, epoch, times = idx[ok], epoch[ok], times[ok]
    t1 = np.sort(times[idx == 0]); t2 = np.sort(times[idx == 1])
    # a missed/duplicated transit shows up as an irregular spacing
    stable = len(t1) >= n_epochs and len(t2) >= 4 and bool(np.all(np.abs(np.diff(t1[:n_epochs]) / P1 - 1) < 0.2))
    rv = np.asarray(res["rv"], float) * AU_DAY_TO_MS if len(rv_times) else np.zeros(0)
    return dict(t1=t1[:n_epochs], t2=t2, rv=rv, rv_times=rv_times, stable=stable)
