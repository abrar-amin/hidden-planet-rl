"""Pre-registered comparisons of the experiment, from per-episode eval files of the model arms and of
the scripted references on the SAME systems.

usage: final_verdict.py DIR --set dev --dec sampled --init sft_compact --demo compact --arms rl_compact_s0 rl_compact_s1 --control control_compact
Files in DIR: final_<set>_<dec>_<arm>_episodes.jsonl / _traces.jsonl ; refs_<set>_<agent>_episodes.jsonl (eval_refs.py)

For every RL arm, paired on systems with a 95 % cluster bootstrap:
  D1 = RL - its own SFT init       D2 = RL - the demonstrator script       D3 = RL - the no-RL control
The PRIMARY metric is the posterior-expected score ES (hidden_planet/expected_score.py), recomputed here from the logged
per-episode fields for every file; the realised strict0 reward is reported alongside as a sanity check.
SUCCESS for an arm ("RL exceeds its demonstrator") iff the ES CIs of D1, D2 and D3 all lie entirely above 0, AND the realised-reward
point estimates of D2 and D3 are >= 0, AND every guard passes, AND the arm was not stopped by the futility rule.
Scripted references may hold several plays per system (eval_refs.py --policy-seeds); they are averaged per system.
The experiment succeeds iff EVERY listed RL arm (both seeds) succeeds.
--expect-arms N makes the both-seeds rule explicit: the experiment fails unless exactly N arms were listed and every one of
them was evaluated and succeeded (a missing, crashed, unevaluated or futility-stopped arm fails it). Pooled rule: the mean
over the listed arms of the realised D2 and D3 point estimates must each be >= half the mean of the corresponding ES estimates.
No arm is named for the sealed set unless the whole experiment succeeded. --n-expected makes the pairing guard absolute.
GUARDS (arm against its SFT init): paired on >= 90 % of the systems; no-answer rate <= 0.10; rate of commits whose exact posterior
probability (computed by the grader for every arm, never shown) is <= 0.5 not above the init's + 0.04; RV-buying rate >= half of the
init's; most common opening request used in <= 60 % of episodes (trace file required). Abstention counts are reported, not guarded."""
import os, sys, json, argparse
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--set", default="dev"); ap.add_argument("--dec", default="sampled")
ap.add_argument("--init", default="sft_compact"); ap.add_argument("--demo", default="compact"); ap.add_argument("--control", default="control_compact")
ap.add_argument("--arms", nargs="+", default=["rl_compact_s0", "rl_compact_s1"]); ap.add_argument("--boot", type=int, default=20000)
ap.add_argument("--expect-arms", type=int, default=0, help="number of pre-registered RL arms (2 for arm P); 0 = no check (exploratory arms)")
ap.add_argument("--n-expected", type=int, default=0, help="systems in the set (192 dev, 96 sealed); the pairing guard needs >= 90 %% of it")
a = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_quality as dq
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hidden_planet.expected_score import es_from_metrics


def load(path):
    by = {}
    for l in open(path):
        r = json.loads(l); r["es"] = es_from_metrics(r); r["es_minus_reward"] = r["es"] - float(r.get("reward", 0.0)); by.setdefault(r["seed"], []).append(r)
    return by


def paired(A, B, k="reward"):
    m = lambda by: {s: float(np.mean([r.get(k, 0.0) for r in rows])) for s, rows in by.items()}
    a_, b_ = m(A), m(B)
    common = sorted(set(a_) & set(b_))
    d = np.array([b_[s] - a_[s] for s in common])
    boot = d[np.random.default_rng(0).integers(0, len(d), (a.boot, len(d)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), len(common)


def single(A, k):
    """per-system mean of one quantity within one policy, with a 95 % cluster bootstrap over systems"""
    d = np.array([np.mean([r.get(k, 0.0) for r in rows]) for rows in A.values()])
    boot = d[np.random.default_rng(1).integers(0, len(d), (a.boot, len(d)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), len(d)


rate = lambda by, k: float(np.mean([r.get(k, 0.0) for rows in by.values() for r in rows]))
count = lambda by, k: (int(sum(r.get(k, 0.0) > 0.5 for rows in by.values() for r in rows)), int(sum(len(rows) for rows in by.values())))
f = lambda arm, kind: os.path.join(a.dir, f"final_{a.set}_{a.dec}_{arm}_{kind}.jsonl")
rf = lambda agent: os.path.join(a.dir, f"refs_{a.set}_{agent}_episodes.jsonl")
fmt = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
if not os.path.exists(f(a.init, "episodes")):
    sys.exit(f"missing {f(a.init, 'episodes')}")
init = load(f(a.init, "episodes"))
refs = {k: load(rf(k)) for k in ("fixed", "fixed_diverse", "compact", "teacher", "lite_teacher", "eig", "compact_smart", "random") if os.path.exists(rf(k))}
if a.demo not in refs:
    sys.exit(f"missing the demonstrator reference {rf(a.demo)}: D2 cannot be computed (run scripts/event.sh refs, or eval_refs.py)")
ctrl = load(f(a.control, "episodes")) if os.path.exists(f(a.control, "episodes")) else None
out = dict(set=a.set, decoding=a.dec, init=a.init, demo=a.demo, arms={})
print(f"=== {a.set} set, {a.dec} decoding: init {a.init}, demonstrator '{a.demo}', control {a.control if ctrl else '(missing)'} ===")
print(f"{a.init:18s} ES {rate(init, 'es'):+.3f} | reward {rate(init, 'reward'):+.3f} | fully correct {rate(init, 'correct'):.2f} | no answer {rate(init, 'invalid'):.2f} | abstains {rate(init, 'verdict_cannot'):.2f} | RV buying {rate(init, 'rv_buy_rate'):.2f} | spent {rate(init, 'spent'):.1f}")
for k, R in refs.items():
    d = paired(R, init); de = paired(R, init, "es")
    print(f"   reference {k:14s} ES {rate(R, 'es'):+.3f} reward {rate(R, 'reward'):+.3f} | init - {k}: ES {fmt(de)}, reward {fmt(d)}")
if ctrl:
    d = paired(init, ctrl); de = paired(init, ctrl, "es"); out["control_minus_init"] = dict(es=de[:3], reward=d[:3])
    print(f"{a.control:18s} ES {rate(ctrl, 'es'):+.3f} reward {rate(ctrl, 'reward'):+.3f} | control - init: ES {fmt(de)}, reward {fmt(d)}")
all_ok = True
for arm in a.arms:
    fut = os.path.join(a.dir, arm, "rl", "futility_stop.txt")
    if os.path.exists(fut):
        print(f"\n[{arm}] FUTILITY STOP (pre-registered; not evaluated on this set, cannot succeed): {open(fut).read().strip()}")
        out["arms"][arm] = dict(futility_stop=True, success=False); all_ok = False; continue
    if not os.path.exists(f(arm, "episodes")):
        print(f"\n[{arm}] no eval files (missing, crashed, no checkpoint or failed eval): cannot succeed")
        out["arms"][arm] = dict(missing=True, success=False); all_ok = False; continue
    B = load(f(arm, "episodes"))
    res = lambda d: '> 0' if d[1] > 0 else ('< 0' if d[2] < 0 else 'not resolved')
    D = {}
    for key in ("es", "reward"):
        D[key] = (paired(init, B, key), paired(refs[a.demo], B, key) if a.demo in refs else None, paired(ctrl, B, key) if ctrl else None)
    d1, d2, d3 = D["reward"]; e1, e2, e3 = D["es"]
    print(f"\n[{arm}] ES {rate(B, 'es'):+.3f} | reward {rate(B, 'reward'):+.3f} | fully correct {rate(B, 'correct'):.2f} | spent {rate(B, 'spent'):.1f}")
    print(f"   PRIMARY (expected score)                     | realised reward (sanity)")
    for nm, e, r in (("D1 - init:        ", e1, d1), ("D2 - demonstrator:", e2, d2), ("D3 - control:     ", e3, d3)):
        if e is not None:
            print(f"   {nm} ES {fmt(e)} {res(e):12s} | reward {fmt(r)} {res(r)}")
    gap = single(B, "es_minus_reward")
    print(f"   this arm's own ES - realised gap: {fmt(gap)} (scripted pairs on dev differ in this gap by up to about 0.05-0.09)")
    for k in ("fixed", "fixed_diverse", "teacher", "eig"):
        if k in refs and k != a.demo:
            print(f"      (secondary) {arm} - {k}: ES {fmt(paired(refs[k], B, 'es'))}, reward {fmt(paired(refs[k], B))}")
    ca, na = count(init, "verdict_cannot"); cb, nb = count(B, "verdict_cannot")
    print(f"   abstentions (reported, not a guard): {ca}/{na} -> {cb}/{nb}; commits made without a fit_model reading above 0.5 on screen: {rate(init, 'lowconf_commit'):.3f} -> {rate(B, 'lowconf_commit'):.3f}")
    have_tr = os.path.exists(f(arm, "traces"))
    dv = dq.diversity(dq.load_summaries(f(arm, "traces"))) if have_tr else None
    n_ref = a.n_expected or len(init)
    guards = [(f"paired on >= 90 % of the {'set' if a.n_expected else 'init'}'s systems", d1[3] >= 0.9 * n_ref and len(init) >= 0.9 * n_ref, f"n = {d1[3]} (init {len(init)}) of {n_ref}"),
              ("no-answer rate <= 0.10", rate(B, "invalid") <= 0.10, f"{rate(B, 'invalid'):.3f}"),
              ("commits with posterior <= 0.5 not above init + 0.04", rate(B, "post_lowconf_commit") <= rate(init, "post_lowconf_commit") + 0.04,
               f"{rate(init, 'post_lowconf_commit'):.3f} -> {rate(B, 'post_lowconf_commit'):.3f}"),
              ("RV buying >= half of init", rate(B, "rv_buy_rate") >= 0.5 * rate(init, "rv_buy_rate"), f"{rate(init, 'rv_buy_rate'):.3f} -> {rate(B, 'rv_buy_rate'):.3f}"),
              ("opening-request diversity (top share <= 0.60; trace file required)", bool(dv) and dv["top_first_request_share"] <= 0.60,
               f"{dv['distinct_first_requests']} distinct, top {dv['top_first_request_share']:.0%}" if dv else "TRACE FILE MISSING")]
    for name, ok, txt in guards:
        print(f"   {'PASS' if ok else 'FAIL'}  {name}: {txt}")
    g_ok = all(g[1] for g in guards)
    es_ok = bool(e1[1] > 0 and e2 is not None and e2[1] > 0 and e3 is not None and e3[1] > 0)
    sanity_ok = bool(d2 is not None and d2[0] >= 0 and d3 is not None and d3[0] >= 0)
    print(f"   {'PASS' if sanity_ok else 'FAIL'}  realised-reward point estimates of D2 and D3 >= 0")
    ok = bool(es_ok and sanity_ok and g_ok)
    all_ok &= ok
    print(f"   => {arm}: {'SUCCESS (expected score exceeds init, demonstrator and control; realised sanity and guards pass)' if ok else 'not met'}")
    out["arms"][arm] = dict(n_paired=d1[3], es=rate(B, "es"), reward=rate(B, "reward"),
                            es_d1=e1[:3], es_d2=e2[:3] if e2 else None, es_d3=e3[:3] if e3 else None,
                            d1=d1[:3], d2=d2[:3] if d2 else None, d3=d3[:3] if d3 else None,
                            realised_sanity=sanity_ok, es_minus_reward=gap[:3], guards={g[0]: bool(g[1]) for g in guards}, success=ok)
ev = [v for v in out["arms"].values() if v.get("es_d2") and v.get("es_d3")]
pooled_ok = False
if ev and len(ev) == len(a.arms):
    pe2, pe3 = np.mean([v["es_d2"][0] for v in ev]), np.mean([v["es_d3"][0] for v in ev])
    pr2, pr3 = np.mean([v["d2"][0] for v in ev]), np.mean([v["d3"][0] for v in ev])
    pooled_ok = bool(pr2 >= 0.5 * pe2 and pr3 >= 0.5 * pe3)
    out["pooled"] = dict(es_d2=pe2, es_d3=pe3, realised_d2=pr2, realised_d3=pr3, ok=pooled_ok)
    print(f"\n{'PASS' if pooled_ok else 'FAIL'}  pooled over the {len(ev)} arms: realised D2 {pr2:+.3f} >= half of ES D2 {pe2:+.3f}, and realised D3 {pr3:+.3f} >= half of ES D3 {pe3:+.3f}")
else:
    print(f"\nFAIL  pooled realised-vs-ES rule: needs every listed arm evaluated ({len(ev)} of {len(a.arms)})")
count_ok = (a.expect_arms == 0 or len(a.arms) == a.expect_arms)
if not count_ok:
    print(f"FAIL  {len(a.arms)} arm(s) listed, {a.expect_arms} pre-registered: the both-seeds rule cannot be met")
out["expect_arms"] = a.expect_arms
out["experiment_success"] = bool(all_ok and pooled_ok and count_ok and len(out["arms"]) == len(a.arms) and all(v["success"] for v in out["arms"].values()))
# pre-registered rule for the sealed set: the SUCCESSFUL arm with the highest expected score on this set and decoding; none -> the sealed set is not used
succ = {k: v["es"] for k, v in out["arms"].items() if v["success"]}
out["best_arm"] = max(succ, key=succ.get) if (succ and out["experiment_success"]) else None
print(f"arm for the sealed set by the pre-registered rule: {out['best_arm'] or 'none (the experiment did not succeed; the sealed set stays unused)'}")
print(f"\nPRE-REGISTERED RESULT ({a.set}, {a.dec}): " + ("SUCCESS: every RL arm exceeds its init, its demonstrator and the control" if out["experiment_success"] else "NOT MET"))
json.dump(out, open(os.path.join(a.dir, f"final_verdict_{a.set}_{a.dec}_{a.init}.json"), "w"), indent=1)
