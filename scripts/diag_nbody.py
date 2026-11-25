import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hidden_planet import physics as ph
NE = 150
rng = np.random.default_rng(5)
print(" j  delta   m2   | |ratio| arg(ratio)[rad] | chop_rms/amp | relrms_after_sin_fit")
for j in (2, 3, 4):
    for delta in (-0.04, -0.015, 0.015, 0.04):
        for m2 in (5, 40):
            P1 = 10.0; mu1 = 10 * ph.ME; mu2 = m2 * ph.ME
            P2 = P1 * j / (j - 1) * (1 + delta); lam2 = rng.uniform(0, 2 * np.pi)
            f1, f2 = ph.forced_ecc(j, delta, mu1, mu2, P1, P2, j * lam2)
            out = ph.nbody_simulate(P1, mu1, f1, P2, mu2, f2, 0.0, lam2, NE)
            T0, P1m, res = ph.linear_ephemeris(out["t1"]); T20, P2m, _ = ph.linear_ephemeris(out["t2"])
            dm = (P2m / P1m) * (j - 1) / j - 1
            lam2_at = 2 * np.pi * (T0 - T20) / P2m
            n = np.arange(NE)
            nu = ph.ttv_frequency_per_epoch(j, dm)
            A = np.c_[np.ones(NE), n, np.sin(2 * np.pi * nu * n), np.cos(2 * np.pi * nu * n)]
            c = np.linalg.lstsq(A, out["t1"], rcond=None)[0]
            W_nb = c[2] + 1j * c[3]          # Im[W e^{i 2 pi nu n}] = Wr sin + Wi cos
            W_an = ph.ttv_complex_amplitude(P1m, j, dm, mu2, 0) * np.exp(1j * j * lam2_at)
            r = W_nb / W_an
            resid = out["t1"] - A @ c
            print(f" {j} {delta:+.3f} {m2:4d} | {abs(r):.3f} {np.angle(r):+.3f} | {np.std(resid)/abs(W_nb):.3f} | dm={dm:+.4f} amp={abs(W_nb)*1440:.2f}min")
