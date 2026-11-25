import sys, os, json, itertools, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import s2_pg as S

def job(a):
    name, seed, kw = a
    out = S.train(seed, **kw)
    return name, seed, {t: [float(x) for x in v] for t, v in out.items()}

def run(configs, n_seeds, tag):
    jobs = [(nm, sd, kw) for nm, kw in configs for sd in range(n_seeds)]
    res = {}
    with Pool(8) as p:
        for nm, sd, o in p.imap_unordered(job, jobs, chunksize=1):
            res.setdefault(nm, {})[sd] = o
    json.dump(res, open(f"diagnosis/sim_{tag}.json", "w"))
    print(f"\n### {tag}: dev (192 systems) expected sampled reward, mean over {n_seeds} RL seeds [sd across seeds]; gain = final - step 0; P(gain>x) = share of seeds")
    for nm, kw in configs:
        r = res[nm]; ts = sorted(next(iter(r.values())).keys())
        line = f"{nm:58s}"
        for t in ts:
            x = np.array([r[s][t][0] for s in r]); line += f" t={t}: {x.mean():+.3f}[{x.std():.3f}]"
        g = np.array([r[s][ts[-1]][0] - r[s][ts[0]][0] for s in r])
        lc = np.array([r[s][ts[-1]][3] for s in r]); sp = np.array([r[s][ts[-1]][1] for s in r]); ab = np.array([r[s][ts[-1]][2] for s in r])
        print(line + f" | gain {g.mean():+.3f} P(>0)={np.mean(g > 0):.2f} P(>.05)={np.mean(g > .05):.2f} P(>.13)={np.mean(g > .13):.2f} | final spent {sp.mean():.1f} abstain {ab.mean():.2f} lowconf {lc.mean():.3f}", flush=True)
    return res

if __name__ == "__main__":
    which = sys.argv[1]; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    T = dict(steps=30, n_sys=12, group=8, eval_at=[0, 10, 20, 30])
    if which == "lr":       # step-size sweep at the planned batch, three exploration levels
        cf = [(f"planned 12x8x30 real eps={e} lr={lr}", dict(T, lr=lr, eps=e)) for e in (0.02, 0.1, 0.3) for lr in (0.02, 0.05, 0.1, 0.2, 0.5)]
        run(cf, NS, "lr")
    if which == "main":
        lr = float(sys.argv[3]); cf = []
        for e in (0.02, 0.1, 0.3):
            cf += [(f"A confirm 5x8x10 real eps={e}", dict(steps=10, n_sys=5, group=8, lr=lr, eps=e, eval_at=[0, 10])),
                   (f"B planned 12x8x30 real eps={e}", dict(T, lr=lr, eps=e)),
                   (f"C planned 12x8x30 EXPECTED-score reward eps={e}", dict(T, lr=lr, eps=e, mode="exp")),
                   (f"D 12x8x100 real eps={e}", dict(steps=100, n_sys=12, group=8, lr=lr, eps=e, eval_at=[0, 30, 100])),
                   (f"E 12x8x300 real eps={e}", dict(steps=300, n_sys=12, group=8, lr=lr, eps=e, eval_at=[0, 100, 300])),
                   (f"F 48x8x30 real eps={e}", dict(steps=30, n_sys=48, group=8, lr=lr, eps=e, eval_at=[0, 10, 30])),
                   (f"G 48x8x30 EXPECTED eps={e}", dict(steps=30, n_sys=48, group=8, lr=lr, eps=e, mode="exp", eval_at=[0, 10, 30])),
                   (f"H 12x8x100 EXPECTED eps={e}", dict(steps=100, n_sys=12, group=8, lr=lr, eps=e, mode="exp", eval_at=[0, 30, 100])),
                   (f"I 12x8x30 real, no curriculum eps={e}", dict(T, lr=lr, eps=e, amb_frac=0.0)),
                   (f"J 24x4x30 real eps={e}", dict(steps=30, n_sys=24, group=4, lr=lr, eps=e, eval_at=[0, 30])),
                   (f"K 6x16x30 real eps={e}", dict(steps=30, n_sys=6, group=16, lr=lr, eps=e, eval_at=[0, 30]))]
        run(cf, NS, "main")
    if which == "opt":      # near-optimum of the family: expected reward, big batch, long
        cf = [(f"optimum proxy 96x8x400 EXPECTED eps=0.1 lr={lr}", dict(steps=400, n_sys=96, group=8, lr=lr, eps=0.1, mode="exp", eval_at=[0, 100, 200, 400])) for lr in (0.05, 0.1)]
        run(cf, 4, "opt")
