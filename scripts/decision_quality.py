"""Offline decision-quality metric from rollout transcripts (free; works on finished runs).

Every assistant turn is judged using ONLY what the agent had seen up to then (latest fit_model output,
budget, RV availability) against the decision rule implied by the strict0 reward (+1 / -1 / 0, cost 0.008):

  no fit yet                 -> acquire is right; any submission is a 'blind' error
  pmax >= 0.9                -> submit the fit's top option; a further purchase is 'overspend'
  0.5 < pmax < 0.9           -> submit the top option or buy more: both acceptable; abstaining is an error
  pmax <= 0.5                -> buy RVs if available and affordable (transits if p_none >= 0.3 and affordable);
                                abstain only when nothing useful can be bought; committing is a 'low-confidence commit'
  (pmax = max(p_both_right, p_no_perturber); affordable RV = RVs possible, >= 6 units left, < 15 RVs bought)

Caveats (docs/AUDIT.md): the judge's rule is the scripted teacher's rule, so this measures adherence to that rule from
agent-visible information, not correctness against the truth (a truth check by confidence bin is printed alongside).
The primary number counts consecutive identical requests once, so that a loop of refused requests does not inflate
differences, and "answered" is reported separately.

For a policy cloned from a demonstrator that never calls fit_model (the compact arm), every answer is a 'blind commit' here,
so this number then measures adoption of the fit tool, not decision quality; use it for arms whose init calls fit_model.
Repeats are removed only when consecutive.

Reported per policy: overall decision accuracy (mean over episodes of the fraction of its turns judged right),
error rates per episode by type, verdict-matches-fit rate, overspend rate, Brier score and ECE of the fit
confidence at submission vs correctness; cluster-bootstrap 95 % CIs over systems; paired differences.

usage:
  decision_quality.py --run runs/rehearsal_s0/rl --iters 0 5 10 [--refs teacher eig --start 0 --n 48]
  decision_quality.py --traces A.jsonl B.jsonl --labels "step 0" "step 10" [--refs teacher eig --start 2000 --n 96]
"""
import os, sys, re, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")

ERRS = ("blind_commit", "lowconf_commit", "wrong_option", "premature_abstain", "abstain_when_decided", "overspend", "futile_acquire", "wasteful_purchase",
        "bulk_rv_request", "mixed_turn", "no_tool_call")


def parse_steps(rec):
    """cookbook rollout summary -> list of turns [(calls, results)], final metrics."""
    turns, final = [], {}
    for s in rec["steps"]:
        logs = s.get("logs") or {}
        calls, results = [], []
        i = 0
        while f"tool_call_{i}" in logs:
            m = re.match(r"^(\w+)\((.*)\)$", logs[f"tool_call_{i}"], re.S)
            try:
                calls.append((m.group(1), json.loads(m.group(2) or "{}")))
            except Exception:
                calls.append(("unparsed", {}))
            try:
                results.append(json.loads(logs.get(f"tool_result_{i}", "{}")))
            except Exception:
                results.append({})
            i += 1
        turns.append((calls, results))
        final.update({k: v for k, v in (s.get("metrics") or {}).items() if isinstance(v, (int, float))})
    return turns, final


def judge(turns, final):
    """Returns per-episode numbers. `acc` is the raw per-turn fraction (a loop of refused requests
    counts every repeat); `acc_dedup` counts consecutive identical requests once and is the PRIMARY number."""
    fit, remaining, n_rv, rv_ok = None, 60.0, 0, True
    errs = dict.fromkeys(ERRS, 0)
    out = dict(submit_conf=None, matches_fit=None, overspent=0, final_ok=None, first_request=None, rv_schedule=None)
    recs = []                                                  # (ok, signature)
    for calls, results in turns:
        sig = json.dumps([(n, a) for n, a in calls], sort_keys=True)
        if not calls:
            errs["no_tool_call"] += 1; recs.append((False, sig))
            continue
        pm = None if fit is None else max(fit["p_none"], fit["p_both"])
        afford_rv = rv_ok and remaining >= 6 and n_rv < 15
        useful_buy = afford_rv or (fit is not None and fit["p_none"] >= 0.3 and remaining >= 4)
        sub = next((c for c in calls if c[0] == "submit_answer"), None)
        acq = [c for c in calls if c[0] in ("observe_transits", "observe_rvs")]
        ok = None
        if out["first_request"] is None and acq:
            out["first_request"] = json.dumps(acq[0], sort_keys=True)
        rvc = next((c for c in acq if c[0] == "observe_rvs"), None)
        if out["rv_schedule"] is None and rvc is not None:
            out["rv_schedule"] = json.dumps(rvc[1], sort_keys=True)
        if sub is not None:                                    # a turn containing a submission is judged as the submission
            v = sub[1].get("verdict")
            if acq:
                errs["mixed_turn"] += 1
            if fit is None:
                errs["blind_commit"] += 1; ok = False
            elif v in ("planet", "none"):
                conf = fit["p_both"] if v == "planet" else fit["p_none"]
                top = "planet" if fit["p_both"] >= fit["p_none"] else "none"
                try:
                    same_numbers = v != "planet" or abs(float(sub[1].get("period_days") or 0) / fit["period"] - 1) < 0.005
                except (TypeError, ValueError):
                    same_numbers = False
                out["submit_conf"], out["matches_fit"] = conf, float(v == top and same_numbers)
                if conf <= 0.5:
                    errs["lowconf_commit"] += 1; ok = False
                elif v != top or not same_numbers:
                    errs["wrong_option"] += 1; ok = False
                else:
                    ok = not acq
            elif v == "cannot_determine":
                if pm > 0.5:
                    errs["abstain_when_decided"] += 1; ok = False
                elif useful_buy:
                    errs["premature_abstain"] += 1; ok = False
                else:
                    ok = not acq
            else:
                ok = False
            out["final_ok"] = float(bool(ok))
        elif acq:
            asked = sum(len(c[1].get("epochs", []) if isinstance(c[1].get("epochs"), list) else []) + len(c[1].get("times_days", []) if isinstance(c[1].get("times_days"), list) else []) for c in acq)
            bought = sum(len(r.get("epoch_and_oc_min", [])) + len(r.get("time_and_rv_ms", [])) for r in results if isinstance(r, dict))
            n_rv_asked = sum(len(c[1].get("times_days", [])) for c in acq if c[0] == "observe_rvs" and isinstance(c[1].get("times_days"), list))
            if fit is not None and pm >= 0.9:
                errs["overspend"] += 1; out["overspent"] = 1; ok = False
            elif bought == 0:
                errs["futile_acquire"] += 1; ok = False
            elif bought < 0.5 * max(asked, 1):                 # purchase quality: most of the request was refused (duplicates, over budget)
                errs["wasteful_purchase"] += 1; ok = False
            elif n_rv_asked >= 10:                             # one blind request that commits half the budget or more
                errs["bulk_rv_request"] += 1; ok = False
            else:
                ok = True
        if ok is not None:
            recs.append((bool(ok), sig))
        for (name, args), r in zip(calls, results):            # update what the agent has seen
            if not isinstance(r, dict):
                continue
            if name == "observe_rvs":
                n_rv += len(r.get("time_and_rv_ms", []))
                if any("not possible" in str(e) for e in r.get("errors", [])):
                    rv_ok = False
            if "remaining" in r:
                remaining = float(r["remaining"])
            if name == "fit_model" and "best_perturber" in r:
                b = r["best_perturber"]
                fit = dict(p_none=r["p_no_perturber"], p_both=b["p_both_right"], period=b["period_days"], mass=b["mass_earth"])
    dedup = [r for i, r in enumerate(recs) if i == 0 or r[1] != recs[i - 1][1]]
    correct = float(final.get("score", -1.0) > 0.99)
    committed = float(final.get("verdict_planet", 0) > 0.5 or final.get("verdict_none", 0) > 0.5)
    cannot = float(final.get("verdict_cannot", 0) > 0.5)
    return dict(acc=sum(r[0] for r in recs) / max(len(recs), 1), acc_dedup=sum(r[0] for r in dedup) / max(len(dedup), 1), n_dec=len(recs), n_repeats=len(recs) - len(dedup),
                correct=correct, committed=committed, cannot=cannot, answered=float(committed or cannot), no_error=float(not any(errs.values())),
                rv=float(n_rv > 0), score=final.get("score", -1.0), spent=final.get("spent", 60 - remaining), **{f"err_{k}": float(v > 0) for k, v in errs.items()}, **out)


def load_summaries(path):
    rows = []
    for l in open(path):
        rec = json.loads(l)
        j = judge(*parse_steps(rec)); j["system"] = rec["group_idx"]
        rows.append(j)
    return rows


def reference_rows(agent, start, n, mode="strict0", cost=0.008):
    """Teacher / EIG-through-text on the same test systems, traced through the same judge."""
    from hidden_planet.text_env import HiddenPlanetSession
    from hidden_planet.text_policies import fit_following_agent, eig_text_agent
    from hidden_planet.tasks import sample_system
    rows = []
    for i in range(n):
        s = HiddenPlanetSession(sample_system(start + i, 5, "test"), cost_per_unit=cost, reward_mode=mode, wrong_penalty=1.0)
        rng = np.random.default_rng(start + i + 17)
        gen = (fit_following_agent(s.user_prompt(), rng, max_turns=s.max_turns, mode=mode, wrong_penalty=1.0) if agent == "teacher"
               else eig_text_agent(s.user_prompt(), rng, max_turns=s.max_turns))
        turns = []
        step = next(gen)
        while True:
            res = [json.loads(s.call(nm, ar)[0]) for nm, ar in step[1]]
            turns.append((list(step[1]), res))
            if s.done:
                break
            step = gen.send(res)
        _, m = s.grade()
        j = judge(turns, m); j["system"] = i
        rows.append(j)
    return rows


def per_system(rows, key):
    by = {}
    for r in rows:
        if r.get(key) is not None:
            by.setdefault(r["system"], []).append(r[key])
    return {k: float(np.mean(v)) for k, v in by.items()}


def boot_ci(x, B=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def calibration(rows):
    c = np.array([r["submit_conf"] for r in rows if r["submit_conf"] is not None]); y = np.array([r["correct"] for r in rows if r["submit_conf"] is not None])
    if len(c) == 0:
        return float("nan"), float("nan"), 0
    ece = 0.0
    for lo, hi in ((0, .5), (.5, .7), (.7, .85), (.85, .95), (.95, 1.01)):
        m = (c >= lo) & (c < hi)
        if m.any():
            ece += m.mean() * abs(c[m].mean() - y[m].mean())
    return float(np.mean((c - y) ** 2)), float(ece), len(c)


def diversity(rows):
    from collections import Counter
    fr = Counter(r["first_request"] for r in rows if r.get("first_request")); rs = Counter(r["rv_schedule"] for r in rows if r.get("rv_schedule"))
    n1, n2 = sum(fr.values()), sum(rs.values())
    return dict(distinct_first_requests=len(fr), top_first_request_share=(max(fr.values()) / n1 if n1 else 0.0),
                distinct_rv_schedules=len(rs), n_rv_episodes=n2, top_rv_schedule_share=(max(rs.values()) / n2 if n2 else 0.0))


def report(label, rows):
    acc = per_system(rows, "acc_dedup"); a = np.array(list(acc.values())); lo, hi = boot_ci(a)
    raw = per_system(rows, "acc"); ar = np.array(list(raw.values()))
    brier, ece, nc = calibration(rows)
    mf = [r["matches_fit"] for r in rows if r["matches_fit"] is not None]
    fo = [r["final_ok"] for r in rows if r["final_ok"] is not None]
    e = lambda k: np.mean([r[f"err_{k}"] for r in rows])
    m = lambda k: float(np.mean([r[k] for r in rows]))
    dv = diversity(rows)
    print(f"{label:22s} n={len(rows):3d} ({len(acc)} systems) | decision accuracy, repeats counted once {a.mean():.3f} [{lo:.3f}, {hi:.3f}] (per-turn, every repeat counted {ar.mean():.3f}) | "
          f"answered {m('answered'):.2f} | final decision right given answered {np.mean(fo) if fo else float('nan'):.2f} (n={len(fo)}) | episodes with no judged error {m('no_error'):.2f}")
    print(f"{'':22s} fully correct {m('correct'):.2f} | cannot {m('cannot'):.2f} | bought RVs {m('rv'):.2f} | spent {m('spent'):.1f} | repeated identical requests per episode {m('n_repeats'):.2f}")
    print(f"{'':22s} errors per episode: low-confidence commit {e('lowconf_commit'):.2f}, blind commit {e('blind_commit'):.2f}, wrong option {e('wrong_option'):.2f}, "
          f"premature abstain {e('premature_abstain'):.2f}, abstain when decided {e('abstain_when_decided'):.2f}, overspend {e('overspend'):.2f}, futile purchase {e('futile_acquire'):.2f}, "
          f"wasteful purchase {e('wasteful_purchase'):.2f}, bulk RV request {e('bulk_rv_request'):.2f}, mixed turn {e('mixed_turn'):.2f}, no tool call {e('no_tool_call'):.2f}")
    bins = []
    for lo_, hi_ in ((0, .5), (.5, .8), (.8, 1.01)):
        sel = [r for r in rows if r["submit_conf"] is not None and lo_ <= r["submit_conf"] < hi_]
        bins.append(f"[{lo_:.1f},{min(hi_, 1):.1f}): {np.mean([r['correct'] for r in sel]) if sel else float('nan'):.2f} right (n={len(sel)})")
    print(f"{'':22s} truth check, commits by fit confidence: " + "; ".join(bins) + f" | Brier {brier:.3f}, ECE {ece:.3f} | answers matching the fit's top option {np.mean(mf) if mf else float('nan'):.2f}")
    print(f"{'':22s} request diversity: {dv['distinct_first_requests']} distinct first requests (most common used in {dv['top_first_request_share']:.0%} of episodes); "
          f"{dv['distinct_rv_schedules']} distinct RV schedules over {dv['n_rv_episodes']} RV-buying episodes (most common {dv['top_rv_schedule_share']:.0%})")
    return dict(label=label, n=len(rows), decision_accuracy=float(a.mean()), ci=[float(lo), float(hi)], decision_accuracy_per_turn=float(ar.mean()), answered=m("answered"),
                final_ok_given_answered=float(np.mean(fo)) if fo else None, no_error=m("no_error"), brier=brier, ece=ece, errors={k: float(e(k)) for k in ERRS},
                matches_fit=float(np.mean(mf)) if mf else None, diversity=dv, lowconf_commit=float(e("lowconf_commit")), cannot=m("cannot"), acc_by_system=acc, raw_by_system=raw)


def paired(la, ra, lb, rb):
    common = sorted(set(ra["acc_by_system"]) & set(rb["acc_by_system"]))
    d = np.array([rb["acc_by_system"][k] - ra["acc_by_system"][k] for k in common])
    lo, hi = boot_ci(d)
    dr = np.array([rb["raw_by_system"][k] - ra["raw_by_system"][k] for k in common]); lor, hir = boot_ci(dr)
    print(f"paired decision accuracy (repeats counted once) {lb} - {la}: {d.mean():+.3f}, paired SE {d.std(ddof=1) / np.sqrt(len(d)):.3f}, 95 % CI [{lo:+.3f}, {hi:+.3f}] over {len(common)} systems"
          f"   (per-turn, every repeat counted: {dr.mean():+.3f} [{lor:+.3f}, {hir:+.3f}])")
    return dict(a=la, b=lb, diff=float(d.mean()), ci=[float(lo), float(hi)], diff_per_turn=float(dr.mean()), ci_per_turn=[float(lor), float(hir)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None); ap.add_argument("--iters", type=int, nargs="+", default=[0, 5, 10])
    ap.add_argument("--traces", nargs="+", default=[]); ap.add_argument("--labels", nargs="+", default=[])
    ap.add_argument("--refs", nargs="*", default=[]); ap.add_argument("--start", type=int, default=0); ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    reps = []
    if a.run:
        for it in a.iters:
            p = os.path.join(a.run, f"iteration_{it:06d}", "eval_test_rollout_summaries.jsonl")
            if os.path.exists(p):
                reps.append(report(f"step {it}", load_summaries(p)))
    for i, p in enumerate(a.traces):
        reps.append(report(a.labels[i] if i < len(a.labels) else os.path.basename(p), load_summaries(p)))
    print()
    pairs = [paired(reps[0]["label"], reps[0], r["label"], r) for r in reps[1:]]
    refs = []
    if a.refs:
        print(f"\nreferences on the same systems (test seeds {a.start}..{a.start + a.n - 1}), same judge:")
        for agent in a.refs:
            refs.append(report(agent, reference_rows(agent, a.start, a.n)))
    if a.json:
        for r in reps + refs:
            r.pop("acc_by_system", None); r.pop("raw_by_system", None)
        json.dump(dict(policies=reps, paired=pairs, references=refs), open(a.json, "w"), indent=1)
