"""What did the group-relative advantage actually reward in confirm_strict0? (offline)"""
import json, glob, collections, sys
import numpy as np
sys.path.insert(0, "diagnosis")
from a1_trends import load, parse, sig, RUN, mets
print("raw KL(sample||train) v1/v2, entropy, lr:")
for m in mets[:-1] if "optim/entropy" not in mets[-1] else mets:
    print(int(m["step"]), m.get("optim/kl_sample_train_v1"), m.get("optim/kl_sample_train_v2"), round(m.get("optim/entropy", 0), 4), m.get("optim/lr"), "trained_tokens", m.get("tokens/trained"))
rows = []
for it in range(10):
    for p in (parse(r) for r in load(it)):
        m = p["m"]
        fitted = any(nm == "fit_model" and isinstance(r, dict) and "best_perturber" in r for nm, a, r in p["calls"])
        committed = bool(m.get("verdict_none", 0) or m.get("verdict_planet", 0))
        conf = m.get("submit_conf", 0.0)
        failed = not (committed or m.get("verdict_cannot", 0))
        exp_score = (2 * conf - 1) if (committed and fitted) else m.get("score", -1.0)   # blind commits / failures keep realized score
        rows.append(dict(it=it, g=(it, p["g"]), R=p["R"], score=m.get("score", -1.0), spent=m.get("spent", 0.0), conf=conf, committed=committed, fitted=fitted,
                         failed=failed, cannot=bool(m.get("verdict_cannot", 0)), nrv=m.get("n_rvs", 0), nT=m.get("n_transits", 0), ac=p["ac"], turns=p["turns"],
                         exp_R=exp_score - 0.008 * m.get("spent", 0.0) + (p["R"] - (m.get("score", -1.0) - 0.008 * m.get("spent", 0.0))),
                         lowc=bool(m.get("lowconf_commit", 0)), has_planet=m.get("has_planet", 0), correct=m.get("correct", 0)))
A = lambda k, R=rows: np.array([r[k] for r in R], float)

def within(x, R):
    g = [r["g"] for r in R]; out = np.array(x, float).copy()
    for k in set(g):
        idx = [i for i, gg in enumerate(g) if gg == k]; out[idx] -= out[idx].mean()
    return out

for name, R in (("steps 0-9", rows), ("steps 1-9 (after the loop repair)", [r for r in rows if r["it"] >= 1]), ("steps 5-9", [r for r in rows if r["it"] >= 5])):
    n = len(R); ng = len({r["g"] for r in R}); f = 8 / 7      # unbiased within-group variance factor
    adv = within(A("R", R), R)
    print(f"\n== {name}: {n} rollouts, {ng} groups ==")
    print(f"  failed (no answer) {A('failed', R).mean():.3f}; committed {A('committed', R).mean():.3f}; abstain {A('cannot', R).mean():.3f}; blind commit {np.mean([r['committed'] and not r['fitted'] for r in R]):.3f}")
    print(f"  var(R) total {A('R', R).var():.3f}; within-group (unbiased) {f * adv.var():.3f}; between-group means {A('R', R).var() - adv.var():.3f}")
    vs, vc, ve = f * within(A('score', R), R).var(), f * within(0.008 * A('spent', R), R).var(), f * within(A('exp_R', R), R).var()
    print(f"  within-group var: score part {vs:.3f}, cost part {vc:.4f} (ratio cost/score {vc / vs:.3f}); with posterior-expected score {ve:.3f} -> luck share {1 - ve / (f * adv.var()):.2f}")
    print(f"  groups with zero reward variance: {sum(1 for k in {r['g'] for r in R} if np.var([r['R'] for r in R if r['g'] == k]) < 1e-12)}")
    # calibration of on-screen confidence for commits
    C = [r for r in R if r["committed"] and r["fitted"]]
    for lo, hi in ((0, .5), (.5, .7), (.7, .9), (.9, 1.01)):
        S = [r for r in C if lo <= r["conf"] < hi]
        if S: print(f"    commits with conf in [{lo},{hi}): n={len(S):3d}  mean conf {np.mean([r['conf'] for r in S]):.2f}  right {np.mean([r['score'] > 0 for r in S]):.2f}  n_groups {len({r['g'] for r in S})}")
    # what the advantage correlates with (token-weighted is what the IS loss sums)
    print(f"  corr(adv, action tokens) {np.corrcoef(adv, A('ac', R))[0, 1]:+.2f}; corr(adv, turns) {np.corrcoef(adv, A('turns', R))[0, 1]:+.2f}; mean ac tokens {A('ac', R).mean():.0f}")
    for feat in ("spent", "nrv", "nT", "cannot", "lowc", "failed", "committed"):
        x = within(A(feat, R), R)
        if x.var() < 1e-12: continue
        slope = (x * adv).sum() / (x * x).sum()
        # cluster-robust SE by group
        gs = [r["g"] for r in R]; res = adv - slope * x; num = 0.0
        for k in set(gs):
            idx = [i for i, gg in enumerate(gs) if gg == k]; num += (x[idx] * res[idx]).sum() ** 2
        se = np.sqrt(num) / (x * x).sum()
        print(f"    within-group slope of reward on {feat:9s}: {slope:+.4f} +- {se:.4f}  (t = {slope / se:+.1f})")
