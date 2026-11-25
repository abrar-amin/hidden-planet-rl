import json, sys, collections
import numpy as np
sys.path.insert(0, "diagnosis")
from a1_trends import load, parse, sig
cnt = collections.Counter(); ex = []
per_step = []
for it in range(10):
    ps = [parse(r) for r in load(it)]
    k = 0
    for p in ps:
        m = p["m"]
        if not m.get("lowconf_commit"): continue
        k += 1
        names = [c[0] for c in p["calls"]]
        fits = [r for nm, a, r in p["calls"] if nm == "fit_model" and isinstance(r, dict) and "best_perturber" in r]
        last_obs = max([i for i, n in enumerate(names) if n.startswith("observe")] or [-1]); last_fit = max([i for i, n in enumerate(names) if n == "fit_model"] or [-1])
        v = "planet" if m.get("verdict_planet") else "none"
        kind = ("no_fit" if not fits else "stale_fit(obs after last fit)" if last_obs > last_fit else "fresh_fit")
        f = fits[-1] if fits else None
        cnt[(v, kind)] += 1
        ex.append((it, p["g"], v, kind, sig(p), None if not f else (f["p_no_perturber"], f["best_perturber"]["p_both_right"]), m.get("score"), m.get("spent")))
    per_step.append(k)
print("lowconf commits per step:", per_step)
for k, v in cnt.most_common(): print(v, k)
for e in ex[:60]: print(e)
