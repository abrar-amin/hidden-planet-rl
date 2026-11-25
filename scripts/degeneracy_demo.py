"""Mass-eccentricity degeneracy in the N-body truth (TTVFast): a perturber of mass m with circular
orbits and one of mass m/2 with a small free eccentricity chosen so that |f + 1.5 conj(Z)/Delta|
doubles give nearly the same transit-timing curve, while their RV amplitudes differ by 2x."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hidden_planet import physics as ph

P1, mu1, lam2, NE = 11.0, 8 * ph.ME, 1.3, 150
print(" Delta   m_A   m_B   e2_B    TTV amp A   rms(A-B)/rms(A)   rms(A - half-mass circular)/rms(A)   K2_A   K2_B [m/s]")
for delta in (-0.04, -0.02, 0.02, 0.04):
    P2 = 2 * P1 * (1 + delta)
    f, g = ph.coef_fg(2, delta)
    def ttv(m2, z2):
        mu2 = m2 * ph.ME
        f1, f2 = ph.forced_ecc(2, delta, mu1, mu2, P1, P2, 2 * lam2)
        # shift lambda2 so the conjunction phase is unchanged when the eccentricity term rotates nothing (Z real here)
        out = ph.ttvfast_simulate(P1, mu1, f1, P2, mu2, z2 + f2, 0.0, lam2, NE)
        return ph.linear_ephemeris(out["t1"])[2]
    mA, mB = 30.0, 15.0
    Z = np.conj(f * delta / 1.5)            # makes (-f - 1.5 conj(Z)/Delta) = -2f : same phase, double amplitude per unit mass
    z2B = Z / g
    A, B, C = ttv(mA, 0j), ttv(mB, z2B), ttv(mB, 0j)
    print(f" {delta:+.2f}  {mA:4.0f}  {mB:4.0f}  {abs(z2B):.4f}   {np.sqrt(2)*A.std()*1440:6.1f} min   {np.std(A-B)/A.std():10.3f}   {np.std(A-C)/A.std():24.3f}"
          f"            {ph.rv_semi_amplitude(mA*ph.ME,P2):5.2f}  {ph.rv_semi_amplitude(mB*ph.ME,P2):5.2f}")
