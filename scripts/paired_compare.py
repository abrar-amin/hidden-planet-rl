"""Paired comparison of two evaluations on the same systems (per-episode JSONL from eval_zero_shot.py --episodes-out).
usage: paired_compare.py step0.jsonl step10.jsonl [--label-a "step 0" --label-b "step 10"]
Prints the mean difference B - A, the paired SE, a 95 % CI from a cluster bootstrap over systems, verdict
counts, rv_buy_rate and submit_conf, and the pre-registered verdict:
  S1 the 95 % CI of the reward difference lies entirely above 0
  S2 cannot rate of B >= 0.5 x that of A        S3 rv_buy_rate of B >= 0.5 x that of A"""
import sys, json, argparse
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
ap.add_argument("--boot", type=int, default=20000); ap.add_argument("--json", default=None)
args = ap.parse_args()


def load(p):
    R = [json.loads(l) for l in open(p)]
    by = {}
    for r in R:
        by.setdefault(r["seed"], []).append(r)
    return R, by


def counts(R):
    c = dict(planet_right=0, planet_wrong=0, none_right=0, none_wrong=0, cannot=0, other=0, invalid=0)
    for r in R:
        sc = r.get("score")
        if r.get("verdict_planet", 0) > 0.5: c["planet_right" if (sc or 0) > 0.99 else "planet_wrong"] += 1
        elif r.get("verdict_none", 0) > 0.5: c["none_right" if (sc or 0) > 0.99 else "none_wrong"] += 1
        elif r.get("verdict_cannot", 0) > 0.5: c["cannot"] += 1
        elif r.get("verdict_other", 0) > 0.5: c["other"] += 1
        else: c["invalid"] += 1
    return c


RA, A = load(args.a); RB, B = load(args.b)
seeds = sorted(set(A) & set(B))
if len(seeds) < len(A) or len(seeds) < len(B):
    print(f"WARNING: {len(A)} / {len(B)} systems in the two files, {len(seeds)} in common; using the common ones")
mean = lambda rows, k: float(np.mean([r.get(k, 0.0) for r in rows]))
a = np.array([mean(A[s], "reward") for s in seeds]); b = np.array([mean(B[s], "reward") for s in seeds])
d = b - a
rng = np.random.default_rng(0)
boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.boot)])
lo, hi = np.percentile(boot, [2.5, 97.5])
se = d.std(ddof=1) / np.sqrt(len(d))
fl = lambda R, k: mean(R, k)
print(f"{len(seeds)} systems, {len(RA)} / {len(RB)} episodes ({len(RA) / max(len(A), 1):.0f} samples per system)")
print(f"{'':22s}{args.label_a:>12s}{args.label_b:>12s}")
for name, k in (("reward", "reward"), ("score (no cost)", "score"), ("fully correct", "correct"), ("budget spent", "spent"), ("rv_buy_rate", "rv_buy_rate"),
                ("submit_conf", "submit_conf"), ("lowconf_commit", "lowconf_commit"), ("cannot rate", "verdict_cannot"), ("invalid", "invalid")):
    print(f"{name:22s}{fl(RA, k):12.3f}{fl(RB, k):12.3f}")
ca, cb = counts(RA), counts(RB)
print("verdicts".ljust(22) + "  " + "  ".join(f"{k} {ca[k]}->{cb[k]}" for k in ca))
print(f"\nreward difference {args.label_b} - {args.label_a}: {d.mean():+.3f}, paired SE {se:.3f}, 95 % CI (cluster bootstrap over systems) [{lo:+.3f}, {hi:+.3f}]")
print(f"unpaired SE for comparison: {np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)):.3f}; systems where the mean reward changed: {np.mean(np.abs(d) > 1e-9):.0%}")
sig = lambda rows: tuple((round(r.get("score", -9), 3), round(r.get("spent", -1), 1), r.get("verdict_planet", 0), r.get("verdict_none", 0), r.get("verdict_cannot", 0),
                         r.get("n_transits", -1), r.get("n_rvs", -1)) for r in sorted(rows, key=lambda r: r.get("sample", 0)))
changed = sum(sig(A[s]) != sig(B[s]) for s in seeds)
print(f"systems whose episode changed at all (verdict, score, purchases): {changed} of {len(seeds)}")
s1 = lo > 0
s2 = fl(RB, "verdict_cannot") >= 0.5 * fl(RA, "verdict_cannot")
s3 = fl(RB, "rv_buy_rate") >= 0.5 * fl(RA, "rv_buy_rate")
for name, ok, txt in (("S1 reward CI above 0", s1, f"[{lo:+.3f}, {hi:+.3f}]"), ("S2 abstention kept", s2, f"{fl(RA, 'verdict_cannot'):.3f} -> {fl(RB, 'verdict_cannot'):.3f}"),
                      ("S3 RV buying kept", s3, f"{fl(RA, 'rv_buy_rate'):.3f} -> {fl(RB, 'rv_buy_rate'):.3f}")):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {txt}")
verdict = "SUCCESS" if (s1 and s2 and s3) else ("HARM" if hi < 0 else "NOT MET (inconclusive if the CI straddles 0)")
print("OVERALL:", verdict)
if args.json:
    json.dump(dict(n_systems=len(seeds), n_changed=int(changed), cannot_a=fl(RA, 'verdict_cannot'), cannot_b=fl(RB, 'verdict_cannot'), rv_a=fl(RA, 'rv_buy_rate'), rv_b=fl(RB, 'rv_buy_rate'), diff=float(d.mean()), paired_se=float(se), ci=[float(lo), float(hi)], counts_a=ca, counts_b=cb, s1=bool(s1), s2=bool(s2), s3=bool(s3), verdict=verdict), open(args.json, "w"), indent=1)
