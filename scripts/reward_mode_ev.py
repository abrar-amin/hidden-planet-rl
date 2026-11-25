"""Expected value of 'guess the alias after transits only' versus 'buy RVs first' (the text teacher),
under each reward mode. Hard level, systems where the first fit shows a perturber (p_no_perturber < 0.3).
The episodes are played once; every reward mode is a re-scoring of the same outcomes.
usage: reward_mode_ev.py [N=200] [cost=0.008]"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")
from hidden_planet.text_env import HiddenPlanetSession
from hidden_planet.text_policies import fit_following_agent
from hidden_planet.tasks import sample_system
from hidden_planet.reward import score

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
COST = float(sys.argv[2]) if len(sys.argv) > 2 else 0.008
RVMODE = sys.argv[3] if len(sys.argv) > 3 else "ideal"
from hidden_planet import tasks
tasks.set_rv_mode(RVMODE)
MODES = [("partial", None), ("strict", 0.5), ("strict", 1.0), ("strict0", 1.0)]


def play(seed, **kw):
    s = HiddenPlanetSession(sample_system(seed, 5, "train"))
    gen = fit_following_agent(s.user_prompt(), np.random.default_rng(seed + 17), max_turns=s.max_turns, **kw)
    step = next(gen); first = None
    while True:
        outs = [json.loads(s.call(n, a)[0]) for n, a in step[1]]
        if first is None and len(outs) > 1 and "best_perturber" in outs[1]:
            first = outs[1]
        if s.done:
            break
        step = gen.send(outs)
    return s, first


rows = []
for seed in range(N):
    g, first = play(seed, guess_early=True)
    if first is None or first["p_no_perturber"] >= 0.3:
        continue                                   # not an alias-decision episode
    rec = dict(seed=seed, p_both_first=first["best_perturber"]["p_both_right"], has=g.sp.has_planet, rv_ok=tasks.rv_available(g.sp))
    from hidden_planet.inference import Posterior, Prior
    qa = Posterior(Prior.from_system(g.sp), *g.ep.data()).prob_truth(g.sp)
    for mode, W in MODES:
        key = f"{mode}{'' if W is None else W}"
        rec[f"guess_{key}"] = score(g.sp, g.answer, g.ep.spent, 1, 1, lam=COST, mode=mode, wrong_penalty=W)["reward"]
        rec[f"abstain_{key}"] = score(g.sp, {"verdict": "cannot_determine"}, g.ep.spent, qa, g.q_ref(), lam=COST, mode=mode, wrong_penalty=W)["reward"]
        t, _ = play(seed, mode=mode, wrong_penalty=W or 0.5)   # W is 1.0 for strict0
        rec[f"teacher_verdict_{key}"] = t.answer["verdict"]
        qt = Posterior(Prior.from_system(t.sp), *t.ep.data()).prob_truth(t.sp)
        rec[f"teacher_{key}"] = score(t.sp, t.answer, t.ep.spent, qt, t.q_ref(), lam=COST, mode=mode, wrong_penalty=W)["reward"]
        rec[f"teacher_rv_{key}"] = float(len(t.ep.rvs) > 0); rec[f"teacher_spent_{key}"] = t.ep.spent
        rec[f"teacher_cannot_{key}"] = float(t.answer["verdict"] == "cannot_determine"); rec[f"det_{key}"] = float(max(qt, t.q_ref()) >= 0.6)
    rec["guess_spent"] = g.ep.spent
    rows.append(rec)
ci = lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))
g = lambda k, m=None: np.array([r[k] for r in rows if m is None or m(r)])
amb = lambda r: r["p_both_first"] < 0.6
print(f"hard level, RV availability {RVMODE}, cost {COST}/unit, {len(rows)} of {N} train systems where the first fit shows a perturber; {int(sum(map(amb, rows)))} of them ambiguous (p_both_right < 0.6 after transits only)")
print(f"{'reward mode':16s} | {'GUESS now':>16s} | {'ABSTAIN now':>16s} | {'TEACHER (buys RVs)':>20s} | teacher - guess | on AMBIGUOUS systems: guess / abstain / teacher")
out = {}
for mode, W in MODES:
    key = f"{mode}{'' if W is None else W}"
    a, b, c = g(f"guess_{key}"), g(f"abstain_{key}"), g(f"teacher_{key}")
    d = c - a
    out[key] = dict(guess=a.mean(), abstain=b.mean(), teacher=c.mean(), diff=d.mean(), diff_ci=ci(d), guess_amb=g(f"guess_{key}", amb).mean(), abstain_amb=g(f"abstain_{key}", amb).mean(),
                    teacher_amb=g(f"teacher_{key}", amb).mean(), teacher_rv=g(f"teacher_rv_{key}").mean(), teacher_spent=g(f"teacher_spent_{key}").mean())
    print(f"{mode + ('' if W is None else f' (wrong -{W})'):16s} | {a.mean():+.3f} ± {ci(a):.3f} | {b.mean():+.3f} ± {ci(b):.3f} | {c.mean():+.3f} ± {ci(c):.3f}     | {d.mean():+.3f} ± {ci(d):.3f} | "
          f"{out[key]['guess_amb']:+.2f} / {out[key]['abstain_amb']:+.2f} / {out[key]['teacher_amb']:+.2f}")
print(f"guess spends {g('guess_spent').mean():.1f} units; teacher spends {out['partial']['teacher_spent']:.1f} and buys RVs in {out['partial']['teacher_rv']:.0%} of these episodes")
key = "strict01.0"
for lab, m in (("all", lambda r: True), ("RVs available", lambda r: r["rv_ok"]), ("RVs NOT available", lambda r: not r["rv_ok"])):
    R = [r for r in rows if m(r)]
    if not R:
        continue
    c = np.array([r[f"teacher_cannot_{key}"] for r in R]); d = np.array([r[f"det_{key}"] for r in R])
    print(f"strict0 (wrong -1, cannot 0), {lab:18s} n={len(R):3d}: guess {np.mean([r[f'guess_{key}'] for r in R]):+.2f} | abstain now {np.mean([r[f'abstain_{key}'] for r in R]):+.2f} | "
          f"teacher {np.mean([r[f'teacher_{key}'] for r in R]):+.2f} | teacher abstains {c.mean():.2f} (precision {(1 - d[c > 0]).mean() if c.sum() else float('nan'):.2f}); undeterminable {1 - d.mean():.2f}; teacher buys RVs {np.mean([r[f'teacher_rv_{key}'] for r in R]):.2f}")
json.dump(dict(cost=COST, n=len(rows), rv_mode=RVMODE, modes=out), open(f"results/reward_mode_ev{'' if RVMODE == 'ideal' else '_rv'}.json", "w"), indent=1, default=float)
