"""Aggregate baseline runs: tune the EIG stop rule on val, report everything on test.
usage: analyze.py VAL.jsonl TEST1.jsonl [TEST2.jsonl ...] [--json out.json]
"""
import os, sys, json, itertools
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hidden_planet import policies as pol
from hidden_planet.reward import score, TAU


def load(path):
    return [r for r in map(json.loads, open(path)) if "error" not in r]


def sp_of(r):
    return SimpleNamespace(has_planet=r["has"], P2=r["truth"]["P2"], m2=r["truth"]["m2"])


RW = dict(mode="partial", wrong_penalty=None, lam=0.005)      # reward setting used by every function below


def set_reward(mode="partial", wrong_penalty=None, lam=0.005):
    RW.update(mode=mode, wrong_penalty=wrong_penalty, lam=lam)


def _ans(e, c_abst):
    return pol.bayes_answer(e, c_abst, RW["mode"], RW["wrong_penalty"] or (1.0 if RW["mode"] == "strict0" else 0.5))


def rescore(s, has):
    """Re-score a stored result dict under the current reward setting."""
    v, W = s["verdict"], (RW["wrong_penalty"] or 0.5)
    if RW["mode"] == "strict0":
        W = RW["wrong_penalty"] or 1.0
        sc = {"none": (1.0 if not has else -W), "planet": (1.0 if s.get("correct", 0.0) >= 1.0 else -W), "cannot_determine": 0.0}.get(v, -1.0)
    elif RW["mode"] == "partial":
        sc = s["score"]
    elif v == "none":
        sc = 1.0 if not has else -W
    elif v == "planet":
        sc = 1.0 if s.get("correct", 0.0) >= 1.0 else -W
    else:
        sc = s["score"]
    out = dict(s); out["score"] = sc; out["reward"] = sc - RW["lam"] * s["spent"]
    return out


def eig_score(r, p_stop, rate_thr, min_spend, c_abst, abstain_aware=True):
    e = pol.replay_stop(r["eig_log"], pol.make_stop_rule(p_stop, rate_thr, min_spend))
    s = score(sp_of(r), _ans(e, c_abst), e["spent"], e["q_true"], r["q_ref"], abstain_aware, **RW)
    s["p_conf"] = max(e["p_none"], e["p_both"])
    return s


def hindsight(r, c_abst):
    sp = sp_of(r)
    cands = [score(sp, _ans(e, c_abst), e["spent"], e["q_true"], r["q_ref"], **RW) for e in r["eig_log"]]
    return max(cands, key=lambda s: s["reward"])


def tune(val):
    best = None
    for p_stop, rate_thr, min_spend, c_abst in itertools.product(
            (0.8, 0.9, 0.95, 0.98), (0.0, 0.01, 0.02, 0.03, 0.05, 0.08), (6, 12, 18), (-0.1, 0.0, 0.1, 0.2, 0.3)):
        m = np.mean([eig_score(r, p_stop, rate_thr, min_spend, c_abst)["reward"] for r in val])
        if best is None or m > best[0]:
            best = (m, dict(p_stop=p_stop, rate_thr=rate_thr, min_spend=min_spend, c_abst=c_abst))
    return best


def stats(S):
    """S: list of score dicts."""
    g = lambda k: np.array([s[k] for s in S], float)
    n = len(S)
    ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    verd = np.array([s["verdict"] for s in S]); det = np.array([s["determinable"] for s in S])
    ab = verd == "cannot_determine"; answered = (verd == "none") | (verd == "planet")
    return dict(n=n, reward=g("reward").mean(), reward_ci=ci(g("reward")), score=g("score").mean(),
                acc=g("correct").mean(), acc_ci=ci(g("correct")),
                acc_answered=g("correct")[answered].mean() if answered.any() else float("nan"),
                cost=g("spent").mean(), cost_ci=ci(g("spent")),
                f_none=(verd == "none").mean(), f_planet=(verd == "planet").mean(), f_cannot=ab.mean(),
                abst_prec=(~det[ab]).mean() if ab.any() else float("nan"),
                abst_rec=ab[~det].mean() if (~det).any() else float("nan"))


def fmt(name, st):
    return (f"{name:28s} {st['reward']:+.3f}±{st['reward_ci']:.3f}  acc {st['acc']:.3f}±{st['acc_ci']:.3f} "
            f"(ans'd {st['acc_answered']:.2f})  cost {st['cost']:5.1f}  none/planet/cannot {st['f_none']:.2f}/{st['f_planet']:.2f}/{st['f_cannot']:.2f}  "
            f"abst P/R {st['abst_prec']:.2f}/{st['abst_rec']:.2f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    val = load(args[0]); blocks = [load(p) for p in args[1:]]
    test = [r for b in blocks for r in b]
    vbest, cfg = tune(val)
    print(f"val n={len(val)}  test n={len(test)} in {len(blocks)} seed blocks")
    print(f"tuned EIG stop rule on val: {cfg}  (val reward {vbest:.3f})")
    det = np.array([max(r['q_ref'], r['eig_log'][-1]['q_true']) >= TAU for r in test]); has = np.array([r["has"] for r in test])
    print(f"test: frac no-planet {1-has.mean():.3f}; determinable (q_ref or full-budget EIG >= {TAU}): {det.mean():.3f}; "
          f"planet&determinable {(has&det).mean():.3f}; planet&undeterminable {(has&~det).mean():.3f}; none&undeterminable {(~has&~det).mean():.3f}")
    rows = {}
    for name in test[0]["pol"]:
        rows[name] = [r["pol"][name] for r in test]
    rows["eig_full_budget"] = [eig_score(r, 2.0, -1, 0, cfg["c_abst"]) for r in test]
    rows["eig_tuned_stop"] = [eig_score(r, **cfg) for r in test]
    rows["eig_untuned(p.9,r.02)"] = [eig_score(r, 0.9, 0.02, 12, 0.15) for r in test]
    rows["HINDSIGHT_ORACLE"] = [hindsight(r, cfg["c_abst"]) for r in test]
    out = {}
    print()
    for name, S in rows.items():
        st = stats(S); out[name] = st
        # seed-block means
        off = 0; bm = []
        for b in blocks:
            bm.append(np.mean([s["reward"] for s in S[off:off + len(b)]])); off += len(b)
        st["block_means"] = bm
        print(fmt(name, st), " blocks:", " ".join(f"{x:+.3f}" for x in bm))
    # paired headroom
    best_script = max((k for k in out if k not in ("HINDSIGHT_ORACLE",)), key=lambda k: out[k]["reward"])
    d = np.array([a["reward"] - b["reward"] for a, b in zip(rows["HINDSIGHT_ORACLE"], rows[best_script])])
    print(f"\nbest scripted = {best_script}; headroom to hindsight oracle (paired) = {d.mean():.3f} ± {1.96*d.std(ddof=1)/np.sqrt(len(d)):.3f}")
    for a, b in (("eig_tuned_stop", "proto_adaptive"), ("eig_tuned_stop", "bayes_uniform_mixed"), ("eig_tuned_stop", "bayes_rv_heavy")):
        d = np.array([x["reward"] - y["reward"] for x, y in zip(rows[a], rows[b])])
        print(f"paired {a} - {b} = {d.mean():+.3f} ± {1.96*d.std(ddof=1)/np.sqrt(len(d)):.3f}")
    # calibration of the grader posterior (EIG full budget)
    pc = np.array([s["p_conf"] for s in rows["eig_full_budget"]]); ok = np.array([s["correct"] for s in rows["eig_full_budget"]]); v = np.array([s["verdict"] for s in rows["eig_full_budget"]])
    print("\nposterior calibration (bayes_uniform_mixed + eig_full, answered only): conf bin -> empirical accuracy")
    pc2 = np.array([s["p_conf"] for s in rows["bayes_uniform_mixed"]]); ok2 = np.array([s["correct"] for s in rows["bayes_uniform_mixed"]]); v2 = np.array([s["verdict"] for s in rows["bayes_uniform_mixed"]])
    P = np.concatenate([pc[v != "cannot_determine"], pc2[v2 != "cannot_determine"]]); O = np.concatenate([ok[v != "cannot_determine"], ok2[v2 != "cannot_determine"]])
    for lo, hi in ((0, .6), (.6, .8), (.8, .95), (.95, 1.01)):
        m = (P >= lo) & (P < hi)
        if m.any(): print(f"   [{lo:.2f},{hi:.2f}): n={m.sum():4d} mean conf {P[m].mean():.3f}  acc {O[m].mean():.3f}")
    # by class
    print("\nreward by system class (eig_tuned_stop | bayes_uniform_mixed | proto_adaptive | hindsight):")
    for lab, m in (("no planet", ~has), ("planet, determinable", has & det), ("planet, undeterminable", has & ~det)):
        print(f"   {lab:24s} n={m.sum():4d} " + " | ".join(f"{np.mean([s['reward'] for s, mm in zip(rows[k], m) if mm]):+.3f}" for k in ("eig_tuned_stop", "bayes_uniform_mixed", "proto_adaptive", "HINDSIGHT_ORACLE")))
    for a in sys.argv[1:]:
        if a.startswith("--json="):
            json.dump(dict(cfg=cfg, table=out), open(a[7:], "w"), indent=1, default=float)
