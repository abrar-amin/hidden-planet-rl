"""Expected-score pre-check, part 1: can a policy exploit training on the posterior-expected score (ES) instead of the realised strict0 reward?

Runs scripted and deliberately degenerate policies through the REAL text environment (TTVFast truth, real grader) on train-split
systems that no eval uses (seeds 900000+), and compares, per policy, the mean realised reward (an unbiased estimate of the
policy's expected realised reward) with its mean ES, paired on systems.
PASS criteria (stated before running):
  (1) no policy has ES - realised with a 95 % CI entirely above +0.03 (no policy is over-rewarded by ES beyond noise);
  (2) the ranking is preserved: for every pair of policies whose realised rewards differ significantly (paired CI excludes 0),
      ES orders them the same way;
  (3) no degenerate policy has a higher ES than the fixed demonstrator and the fit-teacher.
Also reported: calibration of 2p-1 on committed answers by post_conf bin.

usage: es_loophole_test.py [--n 192] [--start 900000] [--out results/es_loophole]"""
import os, sys, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.seterr(all="ignore")
from multiprocessing import Pool
from hidden_planet.tasks import sample_system, NMAX
from hidden_planet.text_env import HiddenPlanetSession
from hidden_planet.text_policies import fit_following_agent, eig_text_agent, compact_text_agent, random_text_agent, fixed_fit_agent

KW = dict(reward_mode="strict0", wrong_penalty=1.0, cost_per_unit=0.008)
SCRIPTED = ["fixed", "teacher", "eig", "compact", "compact_smart", "random"]
DEGENERATE = ["always_none", "always_cannot", "no_answer", "invalid_verdict", "invalid_missing_mass",
              "commit_top_any_p", "minimal_commit", "none_after_data", "overbuy_commit",
              "planet_period_offset", "planet_other_alias", "planet_mass_x2", "planet_at_low_p"]


def run_gen(s, gen):
    step = next(gen)
    while True:
        res = [s.call(nm, ar)[0] for nm, ar in step[1]]
        if s.done:
            break
        try:
            step = gen.send([json.loads(r) for r in res])
        except (StopIteration, KeyError, IndexError, TypeError):
            break


def spread(rng, k):
    return [int(x) for x in np.unique(np.clip(np.linspace(0, NMAX, k).round().astype(int) + rng.integers(-3, 4, k), 0, NMAX))]


def buy_and_fit(s, rng, n_tr=16, n_rv=0):
    s.call("observe_transits", {"epochs": spread(rng, n_tr)})
    if n_rv:
        s.call("observe_rvs", {"times_days": [round(float(t), 2) for t in rng.uniform(0, NMAX * s.sp.P1, n_rv)]})
    return json.loads(s.call("fit_model", {})[0])


def top_commit(s, f):
    bp = f["best_perturber"]
    if f["p_no_perturber"] >= bp["p_both_right"]:
        s.call("submit_answer", {"verdict": "none"})
    else:
        s.call("submit_answer", {"verdict": "planet", "period_days": bp["period_days"], "mass_earth": bp["mass_earth"]})


def play(arg):
    agent, seed = arg
    s = HiddenPlanetSession(sample_system(seed, 5, "train"), **KW)
    rng = np.random.default_rng(seed + 17)
    up = s.user_prompt()
    if agent in SCRIPTED:
        gen = {"teacher": lambda: fit_following_agent(up, rng, max_turns=s.max_turns, mode="strict0", wrong_penalty=1.0),
               "eig": lambda: eig_text_agent(up, rng, max_turns=s.max_turns, mode="strict0", wrong_penalty=1.0),
               "compact": lambda: compact_text_agent(up, rng), "compact_smart": lambda: compact_text_agent(up, rng, smart=True),
               "fixed": lambda: fixed_fit_agent(up, rng, max_turns=s.max_turns), "random": lambda: random_text_agent(up, rng)}[agent]()
        run_gen(s, gen)
    elif agent == "always_none":
        s.call("submit_answer", {"verdict": "none"})
    elif agent == "always_cannot":
        s.call("submit_answer", {"verdict": "cannot_determine"})
    elif agent == "no_answer":
        buy_and_fit(s, rng)
    elif agent == "invalid_verdict":
        buy_and_fit(s, rng); s.call("submit_answer", {"verdict": "maybe"})
    elif agent == "invalid_missing_mass":
        f = buy_and_fit(s, rng); s.call("submit_answer", {"verdict": "planet", "period_days": f["best_perturber"]["period_days"]})
    elif agent == "commit_top_any_p":                  # never abstains
        top_commit(s, buy_and_fit(s, rng, 16, 8))
    elif agent == "minimal_commit":                    # buys almost nothing, commits whatever the fit says
        top_commit(s, buy_and_fit(s, rng, 3))
    elif agent == "none_after_data":
        buy_and_fit(s, rng, 16, 8); s.call("submit_answer", {"verdict": "none"})
    elif agent == "overbuy_commit":                    # spends the whole budget, then commits
        top_commit(s, buy_and_fit(s, rng, 36, 8))
    elif agent in ("planet_period_offset", "planet_other_alias", "planet_mass_x2", "planet_at_low_p"):
        f = buy_and_fit(s, rng, 16, 8); bp = f["best_perturber"]
        P, m = bp["period_days"], bp["mass_earth"]
        if agent == "planet_period_offset":
            P = P * 1.006                              # just outside the 0.5 % tolerance of the fit's own answer
        elif agent == "planet_other_alias":
            al = f.get("aliases", [])
            P = al[1]["P2_days"] if len(al) > 1 else P * 1.07
        elif agent == "planet_mass_x2":
            m = m * 2
        elif agent == "planet_at_low_p" and bp["p_both_right"] > 0.5:   # commits 'planet' only when it is unlikely; otherwise abstains
            s.call("submit_answer", {"verdict": "cannot_determine"}); r, mm = s.grade(); return agent, seed, float(mm["reward_realised"]), float(mm["es"]), mm
        s.call("submit_answer", {"verdict": "planet", "period_days": P, "mass_earth": m})
    r, m = s.grade()
    return agent, seed, float(m["reward_realised"]), float(m["es"]), {k: m.get(k) for k in ("committed", "post_conf", "correct", "spent", "invalid", "verdict_cannot")}


def ci(d, B=20000):
    b = d[np.random.default_rng(0).integers(0, len(d), (B, len(d)))].mean(1)
    lo, hi = np.percentile(b, [2.5, 97.5]); return float(d.mean()), float(lo), float(hi)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=192); ap.add_argument("--start", type=int, default=900000)
    ap.add_argument("--out", default="results/es_loophole"); a = ap.parse_args()
    agents = SCRIPTED + DEGENERATE
    jobs = [(ag, a.start + i) for ag in agents for i in range(a.n)]
    with Pool(8) as p:
        res = p.map(play, jobs, chunksize=4)
    R = {ag: np.array([r for g, sd, r, e, _ in res if g == ag]) for ag in agents}
    E = {ag: np.array([e for g, sd, r, e, _ in res if g == ag]) for ag in agents}
    lines, rows, ok = [], {}, True
    lines.append(f"ES loophole test: {a.n} train systems from seed {a.start}, strict0, cost 0.008 (real environment, TTVFast truth)")
    lines.append(f"{'policy':22s} {'realised':>9s} {'ES':>8s}   ES - realised [95% CI]")
    for ag in agents:
        d = ci(E[ag] - R[ag]); rows[ag] = dict(realised=float(R[ag].mean()), es=float(E[ag].mean()), es_minus_realised=d)
        flag = d[1] > 0.03
        ok &= not flag
        lines.append(f"{ag:22s} {R[ag].mean():+9.3f} {E[ag].mean():+8.3f}   {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]{'   <-- OVER-REWARDED' if flag else ''}")
    # (2) ranking
    viol = []
    for i, x in enumerate(agents):
        for y in agents[i + 1:]:
            dr = ci(R[x] - R[y], 4000)
            if dr[1] > 0 or dr[2] < 0:
                if np.sign(E[x].mean() - E[y].mean()) != np.sign(dr[0]):
                    viol.append(f"{x} vs {y}: realised {dr[0]:+.3f} [{dr[1]:+.3f}, {dr[2]:+.3f}], ES {E[x].mean() - E[y].mean():+.3f}")
    ok &= not viol
    lines.append(f"\nranking violations among significantly different pairs: {len(viol)}"); lines += ["   " + v for v in viol]
    # (3) degenerate below the good scripts
    best_good = min(E["fixed"].mean(), E["teacher"].mean())
    above = [ag for ag in DEGENERATE if E[ag].mean() >= best_good]
    ok &= not above
    lines.append(f"degenerate policies with ES >= min(ES fixed, ES teacher) = {best_good:+.3f}: {above or 'none'}")
    # calibration of committed answers
    pc = np.array([m["post_conf"] for _, _, _, _, m in res if m.get("committed", 0) and m.get("committed") > 0.5], float)
    cc = np.array([m["correct"] for _, _, _, _, m in res if m.get("committed", 0) and m.get("committed") > 0.5], float)
    lines.append("\ncalibration of committed answers (all policies pooled): post_conf bin -> mean post_conf, fraction right, n")
    for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .99), (.99, 1.01)]:
        k = (pc >= lo) & (pc < hi)
        if k.sum():
            lines.append(f"   [{lo:.2f}, {hi:.2f}): {pc[k].mean():.3f} -> {cc[k].mean():.3f}  (n={int(k.sum())})")
    lines.append(f"\nVERDICT: {'PASS - no exploitable gap found; ES may be used as the training reward' if ok else 'FAIL - keep the realised reward for training'}")
    txt = "\n".join(lines); print(txt)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".txt", "w").write(txt + "\n")
    json.dump(dict(rows=rows, violations=viol, degenerate_above=above, passed=bool(ok), n=a.n, start=a.start), open(a.out + ".json", "w"), indent=1)
