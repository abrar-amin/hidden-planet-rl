"""Per-step trends and within-group diversity in runs/confirm_strict0/rl (offline, read-only)."""
import json, glob, re, collections, sys
import numpy as np
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/confirm_strict0/rl"
mets = [json.loads(l) for l in open(f"{RUN}/metrics.jsonl")]
print("step  reward  entropy  kl_v1    spent  nT    nRV   rvrate none  plan  cann  inval lowc  turns frac_mixed")
for m in mets:
    g = lambda k: m.get(k, float('nan'))
    print(f"{int(g('step')):3d}  {g('env/all/reward/total'):+.3f}  {g('optim/entropy'):.4f}  {g('optim/kl_sample_train_v1'):.5f}  {g('env/all/spent'):5.1f} {g('env/all/n_transits'):5.1f} {g('env/all/n_rvs'):5.2f} {g('env/all/rv_buy_rate'):.2f}  {g('env/all/verdict_none'):.2f}  {g('env/all/verdict_planet'):.2f}  {g('env/all/verdict_cannot'):.2f}  {g('env/all/invalid'):.2f}  {g('env/all/lowconf_commit'):.2f}  {g('env/all/turns_per_episode'):.2f}  {g('env/all/by_group/frac_mixed'):.2f}")

def load(it):
    rows = [json.loads(l) for l in open(sorted(glob.glob(f"{RUN}/iteration_*/train_rollout_summaries.jsonl"))[it])]
    return rows

def parse(r):
    """-> dict: calls list [(name, args)], final metrics, total reward, action tokens."""
    calls, ac = [], 0
    for s in r["steps"]:
        ac += s["ac_len"]
        lg = s["logs"]
        i = 0
        while f"tool_call_{i}" in lg:
            c = lg[f"tool_call_{i}"]; nm = c.split("(", 1)[0]
            try: args = json.loads(c[len(nm) + 1:-1])
            except Exception: args = None
            try: res = json.loads(lg.get(f"tool_result_{i}", "null"))
            except Exception: res = None
            calls.append((nm, args, res)); i += 1
    fm = r["steps"][-1]["metrics"]
    return dict(calls=calls, m=fm, R=r["total_reward"], ac=ac, turns=len(r["steps"]), g=r["group_idx"])

def sig(p):
    """purchase signature: sequence of (kind, n_bought) ignoring free tools"""
    out = []
    for nm, args, res in p["calls"]:
        if nm == "observe_transits" and isinstance(res, dict): out.append(("T", len(res.get("epoch_and_oc_min", []))))
        elif nm == "observe_rvs" and isinstance(res, dict): out.append(("R", len(res.get("time_and_rv_ms", []))))
    return tuple(o for o in out if o[1] > 0)

if __name__ == "__main__":
    n_it = len(glob.glob(f"{RUN}/iteration_*/train_rollout_summaries.jsonl"))
    print("\nstep  n  distinct_sigs/grp  distinct_open/grp  distinct(nT,nRV)/grp  distinct_verdict/grp  sd_spent_within  frac_grps_all_same_purchase  blind_commit  var_within  var_between")
    allrows = []
    for it in range(n_it):
        ps = [parse(r) for r in load(it)]
        G = collections.defaultdict(list)
        for p in ps: G[p["g"]].append(p)
        ds, do, dn, dv, sds, same, vw = [], [], [], [], [], [], []
        for g, L in G.items():
            ds.append(len({sig(p) for p in L}))
            op = []
            for p in L:
                f = next((json.dumps(a, sort_keys=True) for nm, a, r in p["calls"] if nm.startswith("observe")), "NONE")
                op.append(f)
            do.append(len(set(op)))
            dn.append(len({(p["m"].get("n_transits"), p["m"].get("n_rvs")) for p in L}))
            verd = lambda p: "none" if p["m"].get("verdict_none") else "planet" if p["m"].get("verdict_planet") else "cannot" if p["m"].get("verdict_cannot") else "invalid"
            dv.append(len({verd(p) for p in L}))
            sds.append(np.std([p["m"].get("spent", 0) for p in L]))
            same.append(len({(p["m"].get("n_transits"), p["m"].get("n_rvs")) for p in L}) == 1)
            vw.append(np.var([p["R"] for p in L]))
        blind = np.mean([(p["m"].get("verdict_none", 0) + p["m"].get("verdict_planet", 0) > 0) and not any(nm == "fit_model" for nm, a, r in p["calls"]) for p in ps])
        vb = np.var([np.mean([p["R"] for p in L]) for L in G.values()])
        print(f"{it:3d} {len(ps):3d}   {np.mean(ds):.2f}   {np.mean(do):.2f}   {np.mean(dn):.2f}   {np.mean(dv):.2f}   {np.mean(sds):5.2f}   {np.mean(same):.2f}   {blind:.2f}   {np.mean(vw):.3f}  {vb:.3f}")
