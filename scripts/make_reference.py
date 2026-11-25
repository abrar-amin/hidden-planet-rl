"""Build the reference lines for the headline figure under a given reward setting.
Sources: text-interface agents (dry_run.py JSON, re-scored per episode; the fit teacher is re-run because its
answers depend on the reward mode) and the simulator-level EIG scheduler + hindsight ceiling
(results/<level>_{val,test}.jsonl, stop rule re-tuned on val for the setting).
usage: make_reference.py easy|hard [--reward-mode strict --wrong-penalty 1.0 --cost 0.008]
writes results/reference_<level>.json (partial, 0.005) or results/reference_<level>_<mode>.json
"""
import os, sys, json, argparse, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze as an

ap = argparse.ArgumentParser()
ap.add_argument("level"); ap.add_argument("--reward-mode", default="partial"); ap.add_argument("--wrong-penalty", type=float, default=None)
ap.add_argument("--cost", type=float, default=0.005); ap.add_argument("--fit-n", type=int, default=200)
ap.add_argument("--tag", default="", help="'_rv' for the realistic-RV variant: reads results/<level>_rv_{val,test}.jsonl and text_*_<level>_rv*.json")
a = ap.parse_args()
level, R = a.level, "results"
an.set_reward(a.reward_mode, a.wrong_penalty, a.cost)
W = a.wrong_penalty or (1.0 if a.reward_mode == "strict0" else 0.5)
suffix = "" if (a.reward_mode == "partial" and a.cost == 0.005) else f"_{a.reward_mode}"
T = a.tag
if T:
    from hidden_planet.tasks import set_rv_mode
    set_rv_mode("realistic")
rvflag = ["--rv-availability", "realistic"] if T else []
ci = lambda x: float(1.96 * np.std(x, ddof=1) / np.sqrt(len(x)))
ref = {}


def text_rows(path):
    """Per-episode (score, spent, correct, cannot, rv) from a dry_run.py JSON, re-scored for the reward setting."""
    out = []
    for e in json.load(open(path))["episodes"]:
        sc = e.get("score", -1.0)
        if a.reward_mode == "strict0" and e.get("invalid", 0.0) < 0.5 and e.get("verdict_cannot", 0.0) > 0.5:
            sc = 0.0
        if a.reward_mode in ("strict", "strict0") and e.get("invalid", 0.0) < 0.5:
            if e.get("verdict_planet", 0.0) > 0.5:
                sc = 1.0 if e.get("correct", 0.0) >= 1.0 else -W
            elif e.get("verdict_none", 0.0) > 0.5:
                sc = 1.0 if sc >= 0.99 else -W
        out.append((sc, e.get("spent", 0.0), e.get("correct", 0.0), e.get("verdict_cannot", 0.0), float(e.get("n_rvs", 0.0) > 0)))
    return np.array(out)


fit_path = f"{R}/text_fit_{level}{T}{suffix}.json"
for ag in (("random", "compact") if T else ()):
    if not os.path.exists(f"{R}/text_{ag}_{level}{T}.json"):
        subprocess.run([sys.executable, "scripts/dry_run.py", "--agent", ag, "--level", level, "--n", "150", "--json", f"{R}/text_{ag}_{level}{T}.json"] + rvflag, capture_output=True)
if (suffix or T) and not os.path.exists(fit_path):
    subprocess.run([sys.executable, "scripts/dry_run.py", "--agent", "fit", "--level", level, "--n", str(a.fit_n if level == "hard" else 150), "--reward-mode", a.reward_mode,
                    "--wrong-penalty", str(W), "--cost-per-unit", str(a.cost), "--json", fit_path] + rvflag, capture_output=True)
for agent, label, path in (("fit", "text-only fit-following expert (SFT teacher)", fit_path), ("random", "random policy", f"{R}/text_random_{level}{T}.json"),
                           ("heuristic", "prototype-style script", f"{R}/text_heuristic_{level}{T}.json"), ("compact", "compact script", f"{R}/text_compact_{level}{T}.json")):
    if os.path.exists(path):
        t = text_rows(path); r = t[:, 0] - a.cost * t[:, 1]
        ref[agent] = dict(label=label, reward=float(r.mean()), ci=ci(r), correct=float(t[:, 2].mean()), spent=float(t[:, 1].mean()), n=len(t),
                          f_cannot=float(t[:, 3].mean()), rv_buy_rate=float(t[:, 4].mean()), via="text interface")
val, test = an.load(f"{R}/{level}{T}_val.jsonl"), an.load(f"{R}/{level}{T}_test.jsonl")
vbest, cfg = an.tune(val)
rows = dict(eig=[an.eig_score(r, **cfg) for r in test], hindsight=[an.hindsight(r, cfg["c_abst"]) for r in test])
for k in ("always_none", "always_cannot", "maxspend_cannot", "random_guess", "bayes_uniform_mixed", "proto_adaptive"):
    rows[k] = [an.rescore(r["pol"][k], r["has"]) for r in test]
labels = dict(eig="Bayesian EIG scheduler", hindsight="hindsight ceiling (optimistic)", always_none="always 'none'", always_cannot="always 'cannot'",
              maxspend_cannot="max spend then 'cannot'", random_guess="random answer, no data", bayes_uniform_mixed="fixed 30T+10RV + Bayes answer",
              proto_adaptive="prototype adaptive script")
for k, S in rows.items():
    st = an.stats(S)
    ref[k if k != "bayes_uniform_mixed" else "fixed_design_bayes"] = dict(label=labels[k], reward=st["reward"], ci=st["reward_ci"], correct=st["acc"], spent=st["cost"], n=st["n"],
                                                                          f_cannot=st["f_cannot"], abst_prec=st["abst_prec"], abst_rec=st["abst_rec"], via="simulator")
has = np.array([r["has"] for r in test]); det = np.array([max(r["q_ref"], r["eig_log"][-1]["q_true"]) >= an.TAU for r in test])
ref["_meta"] = dict(level=level, reward_mode=a.reward_mode, wrong_penalty=W, cost=a.cost, eig_stop_rule=cfg, n_val=len(val), n_test=len(test),
                    frac_no_planet=float(1 - has.mean()), frac_undeterminable=float((~det).mean()))
if T:
    rvok = np.array([bool((r["truth"].get("rv") or {}).get("available", True)) for r in test])
    ref["_meta"].update(rv_availability="realistic", frac_rv_unavailable=float(1 - rvok.mean()))
json.dump(ref, open(f"{R}/reference_{level}{suffix}{T}.json", "w"), indent=1, default=float)
print(f"reference lines: level {level}, reward {a.reward_mode} (wrong -{W}), cost {a.cost}/unit")
for k, v in ref.items():
    if k != "_meta":
        print(f"  {k:20s} {v['reward']:+.3f} ± {v['ci']:.3f}  correct {v['correct']:.2f}  cost {v['spent']:5.1f}  cannot {v.get('f_cannot', float('nan')):.2f}  n={v['n']}  ({v['via']})")
print(" ", ref["_meta"])
