"""Cross-check the TTVFast backend against REBOUND IAS15 and against the Keplerian RV formula.
usage: validate_ttvfast.py [N=20]"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hidden_planet import physics as ph
from hidden_planet.tasks import sample_system, Simulator, NMAX

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rows = []
seeds = [s for s in range(400) if sample_system(s, 5, "val").has_planet][:N]
tt = tr = 0.0
for s in seeds:
    a = sample_system(s, 5, "val"); b = sample_system(s, 5, "val"); c = sample_system(s, 5, "val")
    t0 = time.perf_counter(); A = Simulator(a, "ttvfast"); tt += time.perf_counter() - t0
    t0 = time.perf_counter(); B = Simulator(b, "rebound"); tr += time.perf_counter() - t0
    C = Simulator(c, "analytic")
    tg = np.linspace(0, NMAX * a.P_cat, 500)
    ttv = A.t_transit - (a.T0 + np.arange(NMAX + 1) * a.P1)
    rows.append(dict(seed=s, m2=a.m2, delta=a.delta, sig_t_s=a.sig_t * 86400, sig_rv=a.sig_rv,
                     ttv_amp_min=float(np.sqrt(2) * ttv.std() * 1440),
                     dt_tf_rb_s=float(np.max(np.abs(A.t_transit - B.t_transit)) * 86400),
                     drv_tf_rb=float(np.max(np.abs(A.rv(tg) - B.rv(tg)))),
                     dP2=float(abs(a.P2 - b.P2)), dP2_vs_osc_pct=float(100 * abs(a.P2 / a.extra["P2_osc"] - 1)),
                     dt_tf_analytic_min=float(np.std(A.t_transit - C.t_transit) * 1440),
                     drv_tf_analytic=float(np.std(A.rv(tg) - C.rv(tg))), fallback=a.extra.get("backend_fallback")))
g = lambda k: np.array([r[k] for r in rows])
print(f"{N} level-5 (2:1, eccentric) systems with a perturber; wall time per system: TTVFast {tt/N*1e3:.1f} ms, REBOUND IAS15 {tr/N*1e3:.0f} ms")
print("TTVFast vs REBOUND   max |dt transit|: median %.3f s, worst %.3f s   (timing noise %.0f-%.0f s)" % (np.median(g("dt_tf_rb_s")), g("dt_tf_rb_s").max(), g("sig_t_s").min(), g("sig_t_s").max()))
print("TTVFast vs REBOUND   max |dRV|: median %.4f m/s, worst %.4f m/s   (RV noise %.1f-%.1f m/s); |dP2| worst %.2e d" % (np.median(g("drv_tf_rb")), g("drv_tf_rb").max(), g("sig_rv").min(), g("sig_rv").max(), g("dP2").max()))
print("TTVFast vs analytic model (same osculating elements; includes the mean-vs-osculating period shift): rms dt median %.2f min (TTV amp median %.1f min); rms dRV median %.2f m/s" % (np.median(g("dt_tf_analytic_min")), np.median(g("ttv_amp_min")), np.median(g("drv_tf_analytic"))))
print("mean P2 differs from osculating P2 by median %.3f%%, max %.3f%% (grading tolerance 0.5%%)" % (np.median(g("dP2_vs_osc_pct")), g("dP2_vs_osc_pct").max()))
print("fallbacks to analytic:", sum(r["fallback"] is not None for r in rows))
# RV sign/units: a system with no perturber is pure Kepler; inject a tiny perturber and compare
sp = sample_system(seeds[0], 5, "val"); sp.m2 = 1e-4
S = Simulator(sp, "ttvfast"); sp2 = sample_system(seeds[0], 5, "val"); sp2.m2 = 1e-4; K = Simulator(sp2, "analytic")
tg = np.linspace(0, NMAX * sp.P_cat, 400)
print("single-planet check (perturber 1e-4 Me): rms(TTVFast RV - Kepler RV) = %.4f m/s, K1 = %.2f m/s" % (np.std(S.rv(tg) - K.rv(tg)), ph.rv_semi_amplitude(sp.m1 * ph.ME, sp.P1)))
json.dump(rows, open("results/ttvfast_validation.json", "w"), indent=1)

# ---- integrator-level cross-check: same Jacobi elements fed to both codes ----
for mu1_scale, lab in ((1e-6, "inner planet ~massless (element conventions coincide)"), (1.0, "full inner mass (Jacobi-element conventions differ between the codes)")):
    dts, drv, amp = [], [], []
    for s_ in seeds:
        sp = sample_system(s_, 5, "val"); rv_t = np.linspace(1, 140 * sp.P1, 80)
        args = (sp.P1, sp.m1 * ph.ME * mu1_scale, sp.z1, sp.P2, sp.m2 * ph.ME, sp.z2, -0.6, sp.lam2_0, 150, rv_t)
        A, B = ph.ttvfast_simulate(*args), ph.nbody_simulate(*args)
        dts.append(np.max(np.abs(A["t1"] - B["t1"])) * 86400); drv.append(np.max(np.abs(A["rv"] - B["rv"])))
        amp.append(np.sqrt(2) * np.std(ph.linear_ephemeris(B["t1"])[2]) * 86400)
    print(f"\nSame Jacobi elements, {len(seeds)} systems x 150 orbits, {lab}:\n  TTVFast vs REBOUND IAS15 inner transit times: median max|dt| = {np.median(dts):.3f} s, "
          f"worst {np.max(dts):.3f} s (TTV amplitude median {np.median(amp):.0f} s)\n  RVs: median max|dRV| = {np.median(drv):.4f} m/s, worst {np.max(drv):.4f} m/s")
