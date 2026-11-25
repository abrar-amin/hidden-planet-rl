"""Reference lines on the DEV set (test seeds 3000-3095), the same systems the event's model evals use.
Text-interface agents come from eval_refs.py; EIG and the hindsight ceiling are simulator-level (results/hard_dev.jsonl)
with the stop rule tuned on the val split only. Writes results/reference_dev_strict0.json (readable by plot.py --ref)."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze as an

COST = 0.008
an.set_reward("strict0", 1.0, COST)
S = json.load(open("results/dev192/refs_dev_summary.json"))
ref = {}
names = dict(fixed=("fixed", "fixed-design demonstrator (text)"), lite_teacher=("lite_teacher", "lite teacher (text, tool level lite)"), teacher=("fit", "fit-teacher script (text)"), compact=("compact", "compact demonstrator (text)"), compact_smart=("compact_smart", "compact + fit_model stopping (text)"),
             eig=("eig_text", "EIG through the text tools"), random=("random", "random policy"), always_none=("always_none", "always 'none'"), always_cannot=("always_cannot", "always 'cannot'"))
for k, (key, label) in names.items():
    if k in S:
        v = S[k]; ref[key] = dict(label=label, reward=v["reward"], ci=v["ci"], correct=v["correct"], spent=v["spent"], n=v["n"], f_cannot=v["cannot"], via="text interface")
val, dev = an.load("results/hard_val.jsonl"), an.load("results/hard_dev.jsonl")
_, cfg = an.tune(val)
for key, label, rows in (("eig", "Bayesian EIG scheduler (simulator)", [an.eig_score(r, **cfg) for r in dev]), ("hindsight", "hindsight ceiling (optimistic)", [an.hindsight(r, cfg["c_abst"]) for r in dev])):
    st = an.stats(rows); ref[key] = dict(label=label, reward=st["reward"], ci=st["reward_ci"], correct=st["acc"], spent=st["cost"], n=st["n"], f_cannot=st["f_cannot"], via="simulator")
ref["_meta"] = dict(level="hard", set="dev (test seeds 3000-3191)", reward_mode="strict0", wrong_penalty=1.0, cost=COST, eig_stop_rule=cfg, frac_no_planet=S["_meta"]["frac_no_planet"],
                    headroom_fixed_to_teacher=ref["fit"]["reward"] - ref["fixed"]["reward"], headroom_compact_to_teacher=ref["fit"]["reward"] - ref["compact"]["reward"])
json.dump(ref, open("results/reference_dev_strict0.json", "w"), indent=1, default=float)
for k, v in ref.items():
    if k != "_meta":
        print(f"{k:16s} {v['reward']:+.3f} ± {v['ci']:.3f}  correct {v['correct']:.2f}  spent {v['spent']:5.1f}  cannot {v['f_cannot']:.2f}  n={v['n']}  ({v['via']})")
print(ref["_meta"])
# paired: does the 'smart stopping' tweak beat the compact script on the same systems?
L = lambda a: {json.loads(l)["seed"]: json.loads(l)["reward"] for l in open(f"results/dev192/refs_dev_{a}_episodes.jsonl")}
for a_, b_ in (("fixed", "teacher"), ("fixed", "eig"), ("compact", "compact_smart"), ("compact", "teacher"), ("compact", "eig")):
    A, B = L(a_), L(b_); d = np.array([B[s] - A[s] for s in sorted(A)])
    boot = d[np.random.default_rng(0).integers(0, len(d), (20000, len(d)))].mean(1); lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"paired {b_} - {a_}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
