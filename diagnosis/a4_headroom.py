"""Where is the teacher - fixed gap on the dev set (192 systems), by what the agent sees at the first fit?"""
import json, numpy as np
def load(a):
    ep = {json.loads(l)["seed"]: json.loads(l) for l in open(f"results/dev192/refs_dev_{a}_episodes.jsonl")}
    tr = {json.loads(l)["seed"]: json.loads(l) for l in open(f"results/dev192/refs_dev_{a}_traces.jsonl")}
    return ep, tr
fe, ft = load("fixed"); te, tt = load("teacher")
rows = []
for sd in fe:
    f1 = json.loads(ft[sd]["steps"][0]["logs"]["tool_result_1"])
    rows.append(dict(pn=f1["p_no_perturber"], pb=f1["best_perturber"]["p_both_right"], d=te[sd]["reward"] - fe[sd]["reward"], ds=te[sd]["score"] - fe[sd]["score"],
                     dsp=te[sd]["spent"] - fe[sd]["spent"], rf=fe[sd]["reward"], rt=te[sd]["reward"]))
n = len(rows); print("total gap", np.mean([r["d"] for r in rows]))
for nm, cond in (("first fit p_none >= 0.9 (stop now)", lambda r: r["pn"] >= .9), ("0.3 <= p_none < 0.9 (more transits)", lambda r: .3 <= r["pn"] < .9),
                 ("p_none < 0.3, p_both >= 0.85 (commit now)", lambda r: r["pn"] < .3 and r["pb"] >= .85), ("p_none < 0.3, p_both < 0.85 (RVs, adaptively)", lambda r: r["pn"] < .3 and r["pb"] < .85)):
    S = [r for r in rows if cond(r)]
    if not S: print(nm, 0); continue
    print(f"{nm:46s} n={len(S):3d} ({len(S) / n:.2f})  fixed {np.mean([r['rf'] for r in S]):+.2f} teacher {np.mean([r['rt'] for r in S]):+.2f}  gap/system {np.mean([r['d'] for r in S]):+.3f} "
          f"(score {np.mean([r['ds'] for r in S]):+.3f}, cost {-0.008 * np.mean([r['dsp'] for r in S]):+.3f})  contribution to total {np.sum([r['d'] for r in S]) / n:+.3f}  sd of per-system diff {np.std([r['d'] for r in S]):.2f}")
