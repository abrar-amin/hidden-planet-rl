"""Pre-registered success criteria for an RL run, read from <run>/rl/metrics.csv (fixed-set evals).
usage: check_criteria.py runs/rehearsal_s0 [--cost 0.008]
  C1 eval reward at steps 10, 15, 20 all above step 0, and the last of them >= step 0 + 0.15
  C2 rv_buy_rate at the last eval >= 50 % of its step-0 value
  C3 eval fully-correct rate does not drop by more than 0.05 from step 0 to the last eval
  C4 invalid rate <= 0.10 at the last eval
Reward is eval_score - cost * eval_spent (the run's own reward setting)."""
import sys, csv, os, argparse
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--cost", type=float, default=0.008); ap.add_argument("--steps", type=int, nargs="+", default=[10, 15, 20])
ap.add_argument("--abstain", action="store_true", help="also check that abstention is kept and low-confidence commits do not rise (strict0 test)")
a = ap.parse_args()
path = os.path.join(a.run, "rl", "metrics.csv") if os.path.isdir(os.path.join(a.run, "rl")) else os.path.join(a.run, "metrics.csv")
rows = {}
for r in csv.DictReader(open(path)):
    if r.get("eval_score") not in (None, ""):
        f = lambda k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
        rows[int(float(r["step"]))] = dict(reward=f("eval_score") - a.cost * f("eval_spent"), correct=f("eval_correct"), invalid=f("eval_invalid"),
                                           rv=f("eval_rv_buy_rate"), conf=f("eval_submit_conf"), spent=f("eval_spent"), lowconf=f("eval_lowconf_commit") if r.get("eval_lowconf_commit") not in (None, "") else f("eval_lowconf_planet"),
                                           cannot=f("eval_verdict_cannot"))
print("step  reward  correct  invalid  rv_buy  submit_conf  lowconf_commit  cannot  spent")
for s in sorted(rows):
    v = rows[s]; print(f"{s:4d}  {v['reward']:+.3f}  {v['correct']:.2f}     {v['invalid']:.2f}     {v['rv']:.2f}    {v['conf']:.2f}         {v['lowconf']:.2f}            {v['cannot']:.2f}    {v['spent']:.1f}")
if 0 not in rows:
    sys.exit("no step-0 eval yet")
have = [s for s in a.steps if s in rows]
missing = [s for s in a.steps if s not in rows]
last = rows[max(rows)]; r0 = rows[0]
res = []
if missing:
    res.append(("C1 reward", None, f"evals missing at steps {missing}; so far: " + ", ".join(f"step {s} {rows[s]['reward'] - r0['reward']:+.3f}" for s in have)))
else:
    ok = all(rows[s]["reward"] > r0["reward"] for s in a.steps) and rows[a.steps[-1]]["reward"] >= r0["reward"] + 0.15
    res.append(("C1 reward", ok, "deltas vs step 0: " + ", ".join(f"step {s} {rows[s]['reward'] - r0['reward']:+.3f}" for s in a.steps) + " (need all > 0 and the last >= +0.15)"))
res.append(("C2 rv_buy_rate", last["rv"] >= 0.5 * r0["rv"], f"{r0['rv']:.2f} -> {last['rv']:.2f} (need >= {0.5 * r0['rv']:.2f})"))
res.append(("C3 correct", last["correct"] >= r0["correct"] - 0.05, f"{r0['correct']:.2f} -> {last['correct']:.2f} (need >= {r0['correct'] - 0.05:.2f})"))
res.append(("C4 invalid", last["invalid"] <= 0.10, f"{last['invalid']:.2f} (need <= 0.10)"))
if a.abstain:
    res.append(("C5 abstention kept", last["cannot"] >= 0.5 * r0["cannot"], f"cannot rate {r0['cannot']:.2f} -> {last['cannot']:.2f} (need >= {0.5 * r0['cannot']:.2f})"))
    res.append(("C6 low-confidence commits", not (last["lowconf"] > r0["lowconf"] + 0.04), f"{r0['lowconf']:.2f} -> {last['lowconf']:.2f} (must not rise by more than 0.04)"))
for name, ok, d in res:
    print(f"{'PASS' if ok else ('PENDING' if ok is None else 'FAIL')}  {name}: {d}")
print("OVERALL:", "PENDING" if any(o is None for _, o, _ in res) else ("SUCCESS" if all(o for _, o, _ in res) else "NOT MET"))
