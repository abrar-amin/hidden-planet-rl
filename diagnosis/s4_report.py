import json, sys, numpy as np
DEMO = 0.321   # deterministic demonstrator in the lattice family on dev (s2_pg.py)
for lr in ("0.2", "0.5"):
    res = json.load(open(f"diagnosis/sim_main_lr{lr}.json"))
    print(f"\n## step size {lr}: final dev reward (exact expectation, sampled policy), 24 RL seeds. demo(det.)={DEMO}")
    print(f"{'config':52s} start   final mean[sd]   vs start  vs demo   P(final>demo) P(>demo+.05)  spent  abstain lowconf")
    for nm in sorted(res):
        r = res[nm]; ts = sorted(map(int, next(iter(r.values())).keys()))
        f = np.array([r[s][str(ts[-1])] for s in r]); s0 = np.array([r[s][str(ts[0])][0] for s in r])
        print(f"{nm:52s} {s0.mean():+.3f}  {f[:, 0].mean():+.3f}[{f[:, 0].std():.3f}]  {f[:, 0].mean() - s0.mean():+.3f}   {f[:, 0].mean() - DEMO:+.3f}    {np.mean(f[:, 0] > DEMO):.2f}        {np.mean(f[:, 0] > DEMO + .05):.2f}       {f[:, 1].mean():.1f}  {f[:, 2].mean():.2f}   {f[:, 3].mean():.3f}")
