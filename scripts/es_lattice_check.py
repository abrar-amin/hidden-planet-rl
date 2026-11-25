"""Expected-score pre-check, part 2 (part 1: es_loophole_test.py): ES minus realised score on the diagnosis purchase lattice (diagnosis/lattice.pkl: 772 systems, real
TTVFast truth and real fit_model, 32 purchase nodes), by purchase node and by the probability on screen when committing.
A loophole would be a (node, verdict) or (verdict, displayed-p bin) whose ES exceeds the realised score AND whose ES beats abstaining (0)."""
import pickle, numpy as np
L = pickle.load(open("diagnosis/lattice.pkl", "rb"))["res"]
rows = []
for k, v in L.items():
    for node, x in v["nodes"].items():
        if x is None:
            continue
        pn, pb, scp, scn, sp = x
        rows.append((node, "planet", pb, 2 * pb - 1, scp)); rows.append((node, "none", pn, 2 * pn - 1, scn))
print("by purchase node (commit regardless of p): largest ES - realised")
g = {}
for node, vd, p, es, r in rows:
    g.setdefault((node, vd), []).append(es - r)
st = sorted(((np.mean(d), np.mean(d) - 1.96 * np.std(d, ddof=1) / np.sqrt(len(d)), k) for k, d in g.items()), reverse=True)
for m, lo, k in st[:5]:
    print(f"   {k}: {m:+.3f} (lower {lo:+.3f})")
print(f"   (node, verdict) pairs with lower bound > 0: {sum(s[1] > 0 for s in st)} of {len(st)}")
print("by displayed probability when committing: ES, realised, ES - realised")
bins = [0, .3, .5, .7, .9, .99, 1.01]
for vd in ("planet", "none"):
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [(es, r) for node, v, p, es, r in rows if v == vd and lo <= p < hi]
        if len(sel) > 20:
            es, r = np.array(sel).T; d = es - r
            flag = "  <-- ES > realised AND ES > 0 (exploitable)" if (d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d)) > 0 and es.mean() > 0) else ""
            print(f"   {vd:6s} p in [{lo:.2f},{hi:.2f}): ES {es.mean():+.3f} realised {r.mean():+.3f} diff {d.mean():+.3f} n={len(d)}{flag}")
