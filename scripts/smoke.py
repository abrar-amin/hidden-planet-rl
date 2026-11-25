"""Smoke test of the simulator-level stack: reference identifiability, the EIG scheduler and the grader on 8 systems.
usage: smoke.py"""
"""Smoke test of the simulator-level stack: reference identifiability, the EIG scheduler and the grader on 8 systems.
usage: smoke.py"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hidden_planet.tasks import sample_system, Episode
from hidden_planet import policies as pol
from hidden_planet.reward import score
for seed in range(8):
    sp = sample_system(seed, 2, "train")
    t0 = time.time(); qref = pol.reference_identifiability(sp); t1 = time.time()
    ep = Episode(sp); rng = np.random.default_rng(seed)
    log = pol.eig_trajectory(ep, rng); t2 = time.time()
    e = log[-1]; ans = pol.bayes_answer(e)
    r = score(sp, ans, ep.spent, e["q_true"], qref)
    print(f"seed {seed} has={sp.has_planet} j={sp.j} d={sp.delta:+.3f} m2={sp.m2:.1f} P2={sp.P2:.3f} srv={sp.sig_rv:.1f} | qref={qref:.2f} ({t1-t0:.1f}s) | steps={len(log)} ({t2-t1:.1f}s) nT={e['n_tr']} nRV={e['n_rv']} "
          f"p0={e['p_none']:.2f} pP={e['p_period']:.2f} pB={e['p_both']:.2f} qtrue={e['q_true']:.2f} est=({e['period']:.3f},{e['mass']:.1f}) ans={ans['verdict']} R={r['reward']:.2f}")
    print("    traj:", " ".join(f"{x['spent']:.0f}:{max(x['p_none'],x['p_both']):.2f}/{x['rate']:.3f}" for x in log))
