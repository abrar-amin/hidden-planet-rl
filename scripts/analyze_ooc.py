"""usage: analyze_ooc.py ooc.jsonl [ooc_detect.json]
With the detectability file, out-of-class systems whose extra physics is NOT detectable even by an
informed matched filter on a full-budget design are regraded as in-class (nobody could know)."""
import sys, json
import numpy as np
recs = [json.loads(l) for l in open(sys.argv[1])]
det = {}
if len(sys.argv) > 2:
    det = {(d["kind"], d["seed"]): d["detectable"] for d in json.load(open(sys.argv[2]))}
kinds = ["in_class", "two_planets", "activity", "drift"]
pols = list(recs[0]["res"])
LAM = 0.005
ci = lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))


def regrade(r, s):
    """Score of result s for record r under identifiability-aware OOC grading."""
    k = r["kind"]
    if k == "in_class" or det.get((k, r["seed"]), True):
        return s["score"]
    v = s["verdict"]
    if v == "other":
        return -0.5
    if k == "drift" or (k == "activity" and not r["has"]):
        return {"none": 1.0, "planet": -0.5, "cannot_determine": -0.3}[v]
    return s["score"]            # planet graded vs the 2:1 planet; none -0.5; cannot kept

for r in recs:
    r["detectable"] = det.get((r["kind"], r["seed"]), r["kind"] != "in_class")
    for p in pols:
        s = r["res"][p]; s["score2"] = regrade(r, s); s["reward2"] = s["score2"] - LAM * s["spent"]; s["wrong2"] = float(s["score2"] < 0)
    c = r["res"]["fixC_topup_structured"]; t = r["res"]["topup_only"]
    if r["kind"] != "in_class":
        r["res"]["INFORMED_template_script"] = dict(verdict="other" if r["detectable"] else t["verdict"], spent=c["spent"],
            score2=1.0 if r["detectable"] else t["score2"], reward2=(1.0 if r["detectable"] else t["score2"]) - LAM * c["spent"],
            wrong2=0.0 if r["detectable"] else t["wrong2"])
        r["res"]["CEILING_optimistic"] = dict(verdict="-", spent=r["res"]["eig"]["spent"], score2=1.0, reward2=1.0 - LAM * r["res"]["eig"]["spent"], wrong2=0.0)
    else:
        r["res"]["INFORMED_template_script"] = dict(t, verdict=t["verdict"])
        r["res"]["CEILING_optimistic"] = dict(verdict="-", spent=r["res"]["eig"]["spent"], score2=1.0, reward2=1.0 - LAM * r["res"]["eig"]["spent"], wrong2=0.0)
allp = pols + ["INFORMED_template_script", "CEILING_optimistic"]
print("n per kind:", {k: sum(r["kind"] == k for r in recs) for k in kinds}, "| informed-detectable frac:",
      {k: round(float(np.mean([r["detectable"] for r in recs if r["kind"] == k])), 2) for k in kinds[1:]})
print("\nreward mean ± 95% CI [confident-wrong | 'other' | cannot | cost]   columns: " + " | ".join(kinds) + " | POOLED OOC")
table = {}
for p in allp:
    row = f"{p:27s}"
    for k in kinds + ["OOC"]:
        R = [r for r in recs if (r["kind"] == k if k != "OOC" else r["kind"] != "in_class")]
        S = [r["res"][p] for r in R]
        x = np.array([s["reward2"] for s in S])
        table[(p, k)] = x
        row += (f" | {x.mean():+.2f}±{ci(x):.2f} [{np.mean([s['wrong2'] for s in S]):.2f}|{np.mean([s['verdict']=='other' for s in S]):.2f}|"
                f"{np.mean([s['verdict']=='cannot_determine' for s in S]):.2f}|{np.mean([s['spent'] for s in S]):2.0f}]")
    print(row)
d = lambda a, b, k="OOC": (table[(a, k)] - table[(b, k)])
pm = lambda x: f"{x.mean():+.3f} ± {ci(x):.3f}"
print("\nKEY NUMBERS (pooled out-of-class, paired, identifiability-aware grading)")
print(" (1) EIG in-class reward:", pm(table[("eig", "in_class")]), "| EIG out-of-class reward:", pm(table[("eig", "OOC")]),
      "| drop:", f"{table[('eig','in_class')].mean()-table[('eig','OOC')].mean():+.3f}")
for fx in ("fixA_chi2", "fixB_topup_chi2", "fixC_topup_structured"):
    print(f" (2) recovered by {fx:24s}: {pm(d(fx, 'eig'))}   (in-class cost of the fix: {pm(d(fx, 'eig', 'in_class'))})")
print(" (3) remaining gap, informed-template script - fixC:", pm(d("INFORMED_template_script", "fixC_topup_structured")))
print("     remaining gap, optimistic ceiling - fixC:      ", pm(d("CEILING_optimistic", "fixC_topup_structured")))
print("     optimistic ceiling - informed-template script: ", pm(d("CEILING_optimistic", "INFORMED_template_script")))
for k in kinds[1:]:
    print(f"     [{k:11s}] eig {table[('eig',k)].mean():+.2f} -> fixC {table[('fixC_topup_structured',k)].mean():+.2f} -> informed {table[('INFORMED_template_script',k)].mean():+.2f} -> ceiling {table[('CEILING_optimistic',k)].mean():+.2f}")
print("\nEIG stated confidence vs fully-correct (answered none/planet):")
for lab, sel in (("in_class", lambda r: r["kind"] == "in_class"), ("OOC detectable", lambda r: r["kind"] != "in_class" and r["detectable"]), ("OOC undetectable", lambda r: r["kind"] != "in_class" and not r["detectable"])):
    R = [r for r in recs if sel(r) and r["res"]["eig"]["verdict"] in ("none", "planet")]
    c = np.array([r["conf"] for r in R]); ok = np.array([r["res"]["eig"]["score2"] >= 0.99 for r in R])
    for lo, hi in ((0, .8), (.8, .95), (.95, 1.01)):
        m = (c >= lo) & (c < hi)
        if m.any(): print(f"  {lab:16s} conf [{lo:.2f},{hi:.2f}): n={m.sum():3d} mean conf {c[m].mean():.2f} accuracy {ok[m].mean():.2f}")
print("\nfalse-alarm ('other') rate in-class:", {p: round(float(np.mean([r['res'][p]['verdict']=='other' for r in recs if r['kind']=='in_class'])), 3) for p in pols})
print("flag rate on detectable OOC by kind (fixA / fixB / fixC):")
for k in kinds[1:]:
    R = [r for r in recs if r["kind"] == k and r["detectable"]]
    print(f"  {k:12s} n={len(R):3d} " + " / ".join(f"{np.mean([r['res'][p]['verdict']=='other' for r in R]):.2f}" for p in ("fixA_chi2", "fixB_topup_chi2", "fixC_topup_structured")))
