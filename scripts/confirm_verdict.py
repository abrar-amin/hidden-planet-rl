"""Pre-registered verdict of the confirmation run (reads <run>/paired.json and <run>/decision_quality.json).
PRIMARY   paired decision accuracy B - A (decision_quality.py), 95 % cluster-bootstrap CI over systems
SECONDARY paired strict0 reward B - A (paired_compare.py), same CI
SUCCESS iff  (primary CI entirely > 0  OR  secondary CI entirely > 0)
        and the other metric shows no significant decline (its CI is not entirely < 0)
        and abstention neither collapses nor explodes: 0.5 x A <= cannot rate of B <= min(2 x A, 0.4)
        and RV buying is kept: rv_buy_rate of B >= 0.5 x that of A"""
import sys, json, os
run = sys.argv[1]
p = json.load(open(os.path.join(run, "paired.json"))); d = json.load(open(os.path.join(run, "decision_quality.json")))
da = d["paired"][0]
up_da, up_sc = da["ci"][0] > 0, p["ci"][0] > 0
down_da, down_sc = da["ci"][1] < 0, p["ci"][1] < 0
c_lo = p["cannot_b"] >= 0.5 * p["cannot_a"]; c_hi = p["cannot_b"] <= min(2 * p["cannot_a"], 0.4) + 1e-9
rv = p["rv_b"] >= 0.5 * p["rv_a"]
rows = [("PRIMARY decision accuracy", f"{da['diff']:+.3f}  CI [{da['ci'][0]:+.3f}, {da['ci'][1]:+.3f}]", "up" if up_da else ("DOWN" if down_da else "flat")),
        ("SECONDARY strict0 reward", f"{p['diff']:+.3f}  CI [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]", "up" if up_sc else ("DOWN" if down_sc else "flat")),
        ("abstention not collapsed", f"{p['cannot_a']:.3f} -> {p['cannot_b']:.3f} (need >= {0.5 * p['cannot_a']:.3f})", "PASS" if c_lo else "FAIL"),
        ("abstention not exploded", f"{p['cannot_a']:.3f} -> {p['cannot_b']:.3f} (need <= {min(2 * p['cannot_a'], 0.4):.3f})", "PASS" if c_hi else "FAIL"),
        ("RV buying kept", f"{p['rv_a']:.3f} -> {p['rv_b']:.3f} (need >= {0.5 * p['rv_a']:.3f})", "PASS" if rv else "FAIL"),
        ("episodes changed", f"{p.get('n_changed', '?')} of {p['n_systems']}", "")]
for n, v, s in rows:
    print(f"{n:28s} {v:48s} {s}")
ok = (up_da or up_sc) and not (down_da or down_sc) and c_lo and c_hi and rv
verdict = "SUCCESS" if ok else ("HARM" if (down_da or down_sc) else "NOT MET")
print("PRE-REGISTERED VERDICT:", verdict)
json.dump(dict(verdict=verdict, primary=da, secondary=dict(diff=p["diff"], ci=p["ci"])), open(os.path.join(run, "verdict.json"), "w"), indent=1)
