"""Free correctness gates before any live spend. One command, one PASS/FAIL line per gate.

  .venv/bin/python scripts/preflight.py            # all gates (G2/G6/G7 take a few minutes)
  .venv/bin/python scripts/preflight.py --only G1 G3 G4 G5
  .venv/bin/python scripts/preflight.py --data data/expert_fit_hard_strict0.jsonl

No network, no Tinker credits: G7 uses the fake service.
"""
import os, sys, json, argparse, asyncio, subprocess, math, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np
np.seterr(all="ignore")

MODEL, RENDERER = "Qwen/Qwen3.5-9B", "qwen3_5_disable_thinking"
RESULTS = []
CFG = dict(mode="partial", W=0.5, cost=0.005, rv="ideal", tag="")


def gate(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def load(paths):
    return [json.loads(l) for p in paths for l in open(p)]


def renderer():
    from tinker_cookbook import tokenizer_utils
    from tinker_cookbook.renderers import get_renderer
    tok = tokenizer_utils.get_tokenizer(MODEL)
    return tok, get_renderer(RENDERER, tok)


# ---------------------------------------------------------------- G1
def g1(recs):
    """Submitted period/mass reproducible, within the GRADING tolerance, from the episode's tool-output text only."""
    from sft_warmstart import _numbers
    n_pl = n_ok = 0
    for r in recs:
        ans = next(c["arguments"] for m in reversed(r["messages"]) for c in m.get("tool_calls", []) if c["name"] == "submit_answer")
        if ans.get("verdict") != "planet":
            continue
        n_pl += 1
        seen = [kv for m in r["messages"] if m["role"] == "tool" for kv in _numbers(json.loads(m["content"]))]   # re-parsed text
        per = any(("P2" in k or "period" in k) and v > 0 and abs(v / ans["period_days"] - 1) < 0.005 for k, v in seen)
        mass = any("mass" in k and v > 0 and abs(math.log(v / ans["mass_earth"])) < math.log(1.5) for k, v in seen)
        n_ok += per and mass
    gate("G1 information sufficiency", n_pl > 0 and n_ok / n_pl >= 0.995,
         f"{n_ok}/{n_pl} planet answers ({100 * n_ok / max(n_pl, 1):.1f} %) reproducible from that episode's tool outputs; {len(recs) - n_pl} non-planet answers need no numbers")


# ---------------------------------------------------------------- G2
def g2(n, level="hard"):
    """A policy that only reads tool-output text is competitive UNDER THE ACTIVE REWARD SETTING:
    teacher >= EIG - 0.2, teacher >= compact script, and its 95 % lower bound above every degenerate policy."""
    ref_path = os.path.join(ROOT, "results", f"reference_{level}{'' if CFG['mode'] == 'partial' and CFG['cost'] == 0.005 else '_' + CFG['mode']}{CFG['tag']}.json")
    if not os.path.exists(ref_path) or os.path.getmtime(ref_path) < os.path.getmtime(os.path.join(ROOT, "hidden_planet", "text_policies.py")):
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "make_reference.py"), level, "--reward-mode", CFG["mode"], "--wrong-penalty", str(CFG["W"]), "--cost", str(CFG["cost"]), "--fit-n", str(n)]
        fit_json = os.path.join(ROOT, "results", f"text_fit_{level}{CFG['tag']}{'' if CFG['mode'] == 'partial' and CFG['cost'] == 0.005 else '_' + CFG['mode']}.json")
        if os.path.exists(fit_json):
            os.remove(fit_json)                      # force a fresh teacher run with the current policy code
        subprocess.run(cmd + (["--tag", CFG["tag"]] if CFG["tag"] else []), cwd=ROOT, capture_output=True, text=True)
    r = json.load(open(ref_path)); m = r["_meta"]
    t, e, c = r["fit"], r["eig"], r["compact"]
    degenerate = max(r[k]["reward"] for k in ("random", "always_none", "always_cannot", "maxspend_cannot"))
    ok = (m.get("reward_mode", "partial") == CFG["mode"] and abs(m.get("cost", 0.005) - CFG["cost"]) < 1e-9 and t["reward"] >= e["reward"] - 0.2
          and t["reward"] >= c["reward"] and t["reward"] - t["ci"] > degenerate)
    gate("G2 text-only policy ceiling", ok,
         f"{level}, reward {CFG['mode']} (wrong -{CFG['W']}), cost {CFG['cost']}, RV {CFG['rv']}: fit-following teacher (reads only tool JSON, n={t['n']} held-out) "
         f"{t['reward']:+.3f} ± {t['ci']:.3f} (correct {t['correct']:.2f}, spent {t['spent']:.1f}, abstains {t.get('f_cannot', 0):.2f}); EIG {e['reward']:+.3f}, compact script {c['reward']:+.3f}, "
         f"best degenerate policy {degenerate:+.3f}. Criteria: teacher >= EIG - 0.2, >= compact, CI lower bound ({t['reward'] - t['ci']:+.3f}) > degenerate")


# ---------------------------------------------------------------- G3
def g3():
    from hidden_planet.tasks import sample_system, Episode
    from hidden_planet.reward import score, LAMBDA
    from hidden_planet.text_env import HiddenPlanetSession
    errs = []
    sp = next(s for s in (sample_system(k, 5, "test") for k in range(50)) if s.has_planet); Episode(sp)
    sn = next(s for s in (sample_system(k, 5, "test") for k in range(50)) if not s.has_planet)
    eq = lambda a, b, what: errs.append(what) if abs(a - b) > 1e-9 else None
    P = lambda p, m: {"verdict": "planet", "period_days": p, "mass_earth": m}
    eq(score(sp, P(sp.P2, sp.m2), 0, 0, 0)["reward"], 1.0, "perfect planet answer != +1")
    eq(score(sp, P(sp.P2 * 1.004, sp.m2 * 1.45), 0, 0, 0)["score"], 1.0, "inside tolerance")
    eq(score(sp, P(sp.P2 * 1.006, sp.m2), 0, 0, 0)["score"], 0.25, "wrong period should be -0.25 + 0.5")
    eq(score(sp, P(sp.P2, sp.m2 * 1.6), 0, 0, 0)["score"], 0.25, "wrong mass should be +0.5 - 0.25")
    eq(score(sp, P(sp.P2 * 1.1, sp.m2 * 3), 0, 0, 0)["score"], -0.5, "both wrong")
    eq(score(sp, {"verdict": "none"}, 0, 0, 0)["score"], -0.5, "none when planet")
    eq(score(sn, {"verdict": "none"}, 0, 0, 0)["score"], 1.0, "none when none")
    eq(score(sn, P(20.0, 10.0), 0, 0, 0)["score"], -0.5, "planet when none")
    eq(score(sp, {"verdict": "cannot_determine"}, 0, 0.1, 0.2)["score"], 0.4, "cannot when undeterminable")
    eq(score(sp, {"verdict": "cannot_determine"}, 0, 0.1, 0.7)["score"], -0.3, "cannot when determinable by reference")
    eq(score(sp, {"verdict": "cannot_determine"}, 0, 0.7, 0.1)["score"], -0.3, "cannot when determinable from own data")
    eq(score(sp, None, 10, 0, 0)["reward"], -1.0 - LAMBDA * 10, "no answer")
    # strict mode
    for Wp in (0.5, 1.0):
        k = dict(mode="strict", wrong_penalty=Wp)
        eq(score(sp, P(sp.P2, sp.m2), 0, 0, 0, **k)["score"], 1.0, "strict: both right")
        eq(score(sp, P(sp.P2 * 1.006, sp.m2), 0, 0, 0, **k)["score"], -Wp, "strict: period wrong")
        eq(score(sp, P(sp.P2, sp.m2 * 1.6), 0, 0, 0, **k)["score"], -Wp, "strict: mass wrong")
        eq(score(sn, P(20.0, 10.0), 0, 0, 0, **k)["score"], -Wp, "strict: planet when none")
        eq(score(sp, {"verdict": "none"}, 0, 0, 0, **k)["score"], -Wp, "strict: none when planet")
        eq(score(sn, {"verdict": "none"}, 0, 0, 0, **k)["score"], 1.0, "strict: none when none")
        eq(score(sp, {"verdict": "cannot_determine"}, 0, 0.1, 0.2, **k)["score"], 0.4, "strict: cannot undeterminable")
        eq(score(sp, {"verdict": "cannot_determine"}, 0, 0.1, 0.7, **k)["score"], -0.3, "strict: cannot determinable")
        st = HiddenPlanetSession(sample_system(sp.seed, 5, "test"), cost_per_unit=0.008, reward_mode="strict", wrong_penalty=Wp)
        st.call("observe_transits", {"epochs": list(range(0, 150, 10))})
        st.call("submit_answer", {"verdict": "planet", "period_days": st.sp.P2 * 1.03, "mass_earth": st.sp.m2})
        r, m = st.grade(); eq(r, -Wp - 0.008 * 15, "strict session reward"); eq(m["rv_buy_rate"], 0.0, "rv_buy_rate")
        if "only if BOTH" not in st.system_prompt():
            errs.append("strict prompt text missing")
    # strict0: cannot is always 0
    k0 = dict(mode="strict0")
    eq(score(sp, P(sp.P2, sp.m2), 0, 0, 0, **k0)["score"], 1.0, "strict0: both right")
    eq(score(sp, P(sp.P2 * 1.006, sp.m2), 0, 0, 0, **k0)["score"], -1.0, "strict0: period wrong")
    eq(score(sp, {"verdict": "none"}, 0, 0, 0, **k0)["score"], -1.0, "strict0: none when planet")
    eq(score(sn, {"verdict": "none"}, 0, 0, 0, **k0)["score"], 1.0, "strict0: none when none")
    for qa, qr in ((0.1, 0.2), (0.1, 0.9), (0.9, 0.1)):
        eq(score(sp, {"verdict": "cannot_determine"}, 10, qa, qr, lam=0.008, **k0)["reward"], -0.08, "strict0: cannot is 0 - cost regardless of the label")
    s0 = HiddenPlanetSession(sample_system(sp.seed, 5, "test"), cost_per_unit=0.008, reward_mode="strict0")
    s0.call("observe_transits", {"epochs": list(range(0, 150, 10))}); s0.call("submit_answer", {"verdict": "cannot_determine"})
    eq(s0.grade()[0], -0.008 * 15, "strict0 session: abstain"); 
    if "always earns 0" not in s0.system_prompt():
        errs.append("strict0 prompt text missing")
    for lam in (0.005, 0.012, 0.03):
        eq(score(sp, P(sp.P2, sp.m2), 37, 0, 0, lam=lam)["reward"], 1.0 - lam * 37, f"cost term lam={lam}")
        s = HiddenPlanetSession(sample_system(sp.seed, 5, "test"), cost_per_unit=lam)
        s.call("observe_transits", {"epochs": list(range(0, 150, 10))}); s.call("observe_rvs", {"times_days": [10.5, 20.5, 33.0]})
        s.call("submit_answer", {"verdict": "planet", "period_days": s.sp.P2, "mass_earth": s.sp.m2})
        r, m = s.grade()
        eq(m["spent"], 24.0, "spent accounting"); eq(r, 1.0 - lam * 24, f"session reward lam={lam}")
        eq(m["reward_std_cost"], m["score"] - 0.005 * m["spent"], "reward_std_cost recomputation")
        if f"{lam:.3f}" not in s.system_prompt():
            errs.append(f"prompt does not state cost {lam}")
    gate("G3 grader correctness", not errs, "53 checks, partial, strict (wrong -0.5 and -1.0) and strict0: max reward, each documented penalty, 'cannot' by the determinable label, cost term for 3 values of --cost-per-unit, "
         "reward_std_cost recomputation" + ("" if not errs else " | FAILED: " + "; ".join(errs)))


# ---------------------------------------------------------------- G4 + G5
def g4_g5(recs, n=50):
    from sft_warmstart import record_to_messages
    from tinker_cookbook.renderers import TrainOnWhat
    tok, r = renderer()
    rng = np.random.default_rng(0)
    pick = [recs[i] for i in rng.choice(len(recs), min(n, len(recs)), replace=False)]
    bad4, bad5, max_tok, asst_tok, n_calls = [], [], 0, [], 0
    for rec in pick:
        msgs = record_to_messages(rec, r)
        full, _ = r.build_supervised_example(msgs)
        text = tok.decode([int(t) for t in full.to_ints()])
        max_tok = max(max_tok, full.length)
        ep_asst = 0
        for i, m in enumerate(msgs):
            if m["role"] == "tool" and m["content"] not in text:
                bad4.append(f"seed {rec['seed']}: tool output not verbatim in rendered text")
            if m["role"] != "assistant":
                continue
            prompt = r.build_generation_prompt(msgs[:i]).to_ints()
            ex, w = r.build_supervised_example(msgs[:i + 1], train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
            toks = [int(t) for t in ex.to_ints()]
            parsed, term = r.parse_response(toks[len(prompt):])
            want = [(c.function.name, json.loads(c.function.arguments)) for c in m["tool_calls"]]
            got = [(c.function.name, json.loads(c.function.arguments)) for c in parsed.get("tool_calls", [])]
            n_calls += len(want)
            if not term.is_clean or got != want:
                bad4.append(f"seed {rec['seed']} turn {i}: tool calls differ after render -> tokenize -> parse")
            w = np.asarray(w, float)
            if toks[:len(prompt)] != list(prompt) or w[:len(prompt)].sum() != 0 or w[len(prompt):].sum() <= 0:
                bad5.append(f"seed {rec['seed']} turn {i}: loss mask is not exactly the assistant turn")
            ep_asst += int((w > 0).sum())
        asst_tok.append(ep_asst)
        # causality: the submitted numbers must not appear in any assistant turn before the tool output that shows them
        ans = msgs[-2]["tool_calls"][0]
        a = json.loads(ans.function.arguments)
        if a.get("verdict") == "planet":
            key = str(a["period_days"])
            first_tool = next((i for i, m in enumerate(msgs) if m["role"] == "tool" and key in m["content"]), None)
            first_asst = next((i for i, m in enumerate(msgs) if m["role"] == "assistant" and (key in str(m["content"]) or any(key in c.function.arguments for c in m["tool_calls"]))), None)
            if first_tool is None or first_asst is None or first_asst < first_tool:
                bad5.append(f"seed {rec['seed']}: answer {key} appears before the tool output that contains it")
    gate("G4 render round-trip", not bad4 and max_tok < 48_000,
         f"{len(pick)} episodes, {n_calls} tool calls: render -> tokenize ({RENDERER}) -> parse gives identical calls; every tool output appears verbatim; "
         f"longest episode {max_tok} tokens (env limit 48,000; model context 64K)" + ("" if not bad4 else " | " + bad4[0] + f" (+{len(bad4) - 1} more)"))
    gate("G5 SFT datum sanity", not bad5,
         f"{len(pick)} episodes: loss weights are zero on every prompt token and positive only on the assistant turn being trained; "
         f"assistant tokens per episode median {int(np.median(asst_tok))} (min {min(asst_tok)}, max {max(asst_tok)}); no answer precedes the tool output it is copied from"
         + ("" if not bad5 else " | " + bad5[0] + f" (+{len(bad5) - 1} more)"))


# ---------------------------------------------------------------- G6
async def g6(n_sys=30, noise=0.6):
    """Fraction of GRPO groups with non-zero reward variance, for an imperfect policy (the fit-following
    agent with randomised thresholds and occasional blunders), hard level, RL cost setting."""
    from hidden_planet.text_env import HiddenPlanetSession
    from hidden_planet.text_policies import fit_following_agent
    from hidden_planet.tasks import sample_system
    rows = []
    for seed in range(n_sys):
        R, S = [], []
        for g in range(8):
            s = HiddenPlanetSession(sample_system(seed, 5, "train"), cost_per_unit=CFG["cost"], reward_mode=CFG["mode"], wrong_penalty=CFG["W"])
            gen = fit_following_agent(s.user_prompt(), np.random.default_rng(seed * 100 + g), max_turns=s.max_turns, noise=noise, mode=CFG["mode"], wrong_penalty=CFG["W"])
            step = next(gen)
            while True:
                outs = [json.loads(s.call(nm, ar)[0]) for nm, ar in step[1]]
                if s.done:
                    break
                step = gen.send(outs)
            r, m = s.grade(); R.append(r); S.append(m["score"])
        rows.append((R, S))
    f = lambda k, idx: np.mean([np.std(r[idx][:k]) > 1e-9 for r in rows])
    mean_r = np.mean([np.mean(r[0]) for r in rows])
    ok = f(4, 0) >= 0.6 and f(8, 0) >= 0.6
    gate("G6 RL signal (offline simulation)", ok,
         f"{n_sys} hard systems, reward {CFG['mode']} (wrong -{CFG['W']}), cost {CFG['cost']}, RV {CFG['rv']}; noisy fit-following policy (mean reward {mean_r:+.2f}): groups with non-zero reward variance {f(4, 0):.2f} (size 4), {f(8, 0):.2f} (size 8); "
         f"with non-zero ANSWER-SCORE variance (ignoring cost) {f(4, 1):.2f} / {f(8, 1):.2f}; threshold 0.6. Simulated, not the live model: no per-episode live eval file exists.")


# ---------------------------------------------------------------- G7
def g7(data):
    run = os.path.join(ROOT, "runs", "preflight_dry")
    subprocess.run(["rm", "-rf", run])
    env = dict(os.environ, DRY="1", RUN=run, DATA=" ".join(data), SEALED="0", REWARD_MODE=CFG["mode"], WRONG=str(CFG["W"]), COST=str(CFG["cost"]), RV_AVAIL=CFG["rv"], SFT_STEPS="2", SFT_BATCH="8", RL_STEPS="3", RL_BATCH="3", RL_GROUP="4",
               EVAL_N="6", EVAL_EVERY="1", SAVE_EVERY="1", PILOT_STOP="0")
    t0 = time.time()
    p = subprocess.run(["bash", os.path.join(ROOT, "scripts", "run_arm.sh")], cwd=ROOT, env=env, capture_output=True, text=True)
    log = open(os.path.join(run, "run.log")).read() if os.path.exists(os.path.join(run, "run.log")) else ""
    need = ["eval_base_hard.json", "sft/sft.json", "eval_sft_hard.json", "rl/metrics.csv", "rl/checkpoints.jsonl", "eval_rl_easy.json", "eval_rl_hard.json", "learning_curve.png"]
    missing = [f for f in need if not os.path.exists(os.path.join(run, f))]
    ref = json.load(open(os.path.join(ROOT, "results", "reference_hard.json")))
    refs_ok = all(k in ref for k in ("random", "fit", "eig", "hindsight"))
    ok = p.returncode == 0 and not missing and "Traceback" not in log and "=== done" in log and refs_ok
    gate("G7 pipeline dry run (fake service)", ok,
         f"reduced sizes (2 SFT steps, 3 RL steps x 12 episodes, 6-episode evals), {time.time() - t0:.0f} s: "
         + ("all stages completed; learning_curve.png written with reference lines random / scripted expert / EIG / hindsight" if ok else
            f"missing {missing}; traceback in log: {'Traceback' in log}; exit {p.returncode}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=None); ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--reward-mode", default="strict0"); ap.add_argument("--wrong-penalty", type=float, default=1.0); ap.add_argument("--cost", type=float, default=0.008)
    ap.add_argument("--rv-availability", default="ideal", choices=["ideal", "realistic"])
    ap.add_argument("--tool-level", default="oracle", choices=["oracle", "lite"]); ap.add_argument("--demo", default="fit", help="demonstrator whose SFT data is checked: fit | fixed | compact | lite")
    ap.add_argument("--g2-n", type=int, default=200); ap.add_argument("--min-reward", type=float, default=0.2)
    a = ap.parse_args()
    from hidden_planet.tasks import set_rv_mode
    set_rv_mode(a.rv_availability)
    CFG.update(mode=a.reward_mode, W=a.wrong_penalty, cost=a.cost, rv=a.rv_availability, tag="_rv" if a.rv_availability == "realistic" else "")
    if a.demo == "lite":
        a.tool_level = "lite"
    import sft_warmstart as _sw
    _sw.SESSION_KW.update(reward_mode=a.reward_mode, wrong_penalty=a.wrong_penalty, cost_per_unit=a.cost, tool_level=a.tool_level)
    if a.data is None:
        a.data = [f"data/expert_{a.demo}_hard_{a.reward_mode}{CFG['tag']}.jsonl"]
    if a.demo != "fit" and a.only is None:
        a.only = ["G1", "G3", "G4", "G5"]              # G2 / G6 / G7 are specific to the fit-teacher pipeline; see the dev references for the others
    a.min_reward = -9.0                                  # every demonstrator is cloned unfiltered (event.sh)
    print(f"preflight setting: reward {a.reward_mode} (wrong -{a.wrong_penalty}), cost {a.cost}/unit, RV availability {a.rv_availability}, tool level {a.tool_level}, demonstrator {a.demo}")
    want = lambda g: a.only is None or g in a.only
    data = [os.path.join(ROOT, p) if not os.path.isabs(p) else p for p in a.data]
    from sft_warmstart import keep_episode
    recs = [r for r in load(data) if keep_episode(r, a.min_reward)]
    n_abst = sum(r["messages"][-2]["tool_calls"][0]["arguments"]["verdict"] == "cannot_determine" for r in recs)
    print(f"preflight: SFT data {a.data} ({len(recs)} kept episodes: reward >= {a.min_reward} or a correct abstention; {n_abst} abstentions)\n")
    if want("G1"): g1(recs)
    if want("G2"): g2(a.g2_n)
    if want("G3"): g3()
    if want("G4") or want("G5"): g4_g5(recs)
    if want("G6"): asyncio.run(g6())
    if want("G7"): g7(a.data)
    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{'ALL GATES PASS' if n_fail == 0 else f'{n_fail} GATE(S) FAILED'} ({len(RESULTS)} run)")
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)      # untracked output, so that the tracked results/preflight.json is never overwritten
    json.dump([dict(gate=n, ok=o, detail=d) for n, o, d in RESULTS], open(os.path.join(ROOT, "runs", f"preflight_{a.demo}.json"), "w"), indent=1)
    sys.exit(1 if n_fail else 0)
