"""Scripted agents that act ONLY through the text tool interface (same strings an LLM sees).

A policy is a generator: it yields (thought, [(tool_name, args), ...]) and receives the list of
parsed JSON tool results.
"""
from __future__ import annotations

import re

import numpy as np

from .inference import Posterior, Prior, label_eig
from .policies import bayes_answer, make_stop_rule
from .tasks import NMAX, COST, BUDGET


def parse_task(user_prompt: str) -> dict:
    """Recover the public task parameters from the prompt text (what an LLM would read)."""
    g = lambda pat: float(re.search(pat, user_prompt).group(1))
    return dict(P_cat=g(r"P1 = ([\d.]+) d"), T0_cat=g(r"T0 = (-?[\d.]+) d"), sig_t=g(r"precision: ([\d.]+) min") / 1440,
                sig_rv=(g(r"RV precision: ([\d.]+) m/s") if "RV precision: none" not in user_prompt else 1e3), t_max=g(r"from 0 to ([\d.]+) d"),
                rv_ok="RVs are NOT possible" not in user_prompt,
                sigma_e=(0.0 if "circular" in user_prompt.split("Transit-time")[1] else g(r"typical value ([\d.]+)\)")),
                mass_range=(g(r"Perturber masses here range from ([\d.]+) to"), g(r"range from [\d.]+ to ([\d.]+) Earth")))


def heuristic_text_agent(user_prompt: str, rng):
    """Staged heuristic: transits -> periodogram -> RVs on the aliases -> answer."""
    T = parse_task(user_prompt)
    ep0 = np.unique(np.clip(np.linspace(0, NMAX, 16).round().astype(int) + rng.integers(-3, 4, 16), 0, NMAX))
    used = set(int(x) for x in ep0)
    yield "Start with a sparse, jittered set of transits.", [("observe_transits", {"epochs": sorted(used)})]
    (fit,) = yield "Check for a timing signal.", [("fit_transit_timing", {})]
    if fit["peaks"][0]["delta_chi2"] < 25:
        more = [int(x) for x in rng.choice([n for n in range(NMAX + 1) if n not in used], 10, replace=False)]
        used |= set(more)
        yield "Weak signal; add transits.", [("observe_transits", {"epochs": sorted(more)})]
        (fit,) = yield "Refit.", [("fit_transit_timing", {})]
        if fit["peaks"][0]["delta_chi2"] < 25:
            yield "No timing signal: no perturber.", [("submit_answer", {"verdict": "none"})]
            return
    al = fit["peaks"][0]["aliases"]
    if not al:
        yield "No resonance alias fits.", [("submit_answer", {"verdict": "cannot_determine"})]
        return
    if not T["rv_ok"] or max(a["K2_if_circular_ms"] for a in al) < 0.5 * T["sig_rv"]:
        yield "RV signal hopeless at this precision.", [("submit_answer", {"verdict": "cannot_determine"})]
        return
    spent = len(used)
    n_rv = 0
    periods = [a["P2_days"] for a in al]
    best = None
    while spent + 3 * COST["rv"] <= BUDGET:
        k = 6 if n_rv == 0 else 3
        k = int(min(k, (BUDGET - spent) // COST["rv"]))
        times = [round(float(t), 2) for t in np.sort(rng.uniform(0, T["t_max"], k))]
        yield "Buy RVs to separate the aliases.", [("observe_rvs", {"times_days": times})]
        spent += 3 * k; n_rv += k
        (rf,) = yield "Fit each alias.", [("fit_rv", {"periods_days": periods})]
        fits = sorted(rf["fits"], key=lambda f: -f["delta_chi2"])
        best = fits[0]
        margin = best["delta_chi2"] - (fits[1]["delta_chi2"] if len(fits) > 1 else 0.0)
        if best["delta_chi2"] > 16 and margin > 9:
            yield "Alias resolved.", [("submit_answer", {"verdict": "planet", "period_days": best["P2_days"], "mass_earth": best["mass_earth"]})]
            return
    yield "Aliases not separable within budget.", [("submit_answer", {"verdict": "cannot_determine"})]


def eig_text_agent(user_prompt: str, rng, p_stop=0.9, rate_thr=0.02, min_spend=12.0, c_abst=0.15, batch=3, max_turns=None, mode="partial", wrong_penalty=0.5):
    """The Bayesian EIG scheduler, driven purely by tool results. Every observing reply also calls
    fit_model, and the stop decision and the submitted period/mass are read from that tool's output,
    so nothing the agent reports comes from hidden state. Only the CHOICE of what to observe next uses
    the agent's own posterior samples (rebuilt from the data the tools returned)."""
    T = parse_task(user_prompt)
    res = (2,)                                  # default task distribution is 2:1 only
    prior = Prior(T["P_cat"], T["T0_cat"], T["sig_t"], T["sig_rv"], T["sigma_e"], resonances=res, mass_range=T["mass_range"])
    tr, rv, spent = {}, [], 0.0
    stop = make_stop_rule(p_stop, rate_thr, min_spend)
    turns = 0
    if max_turns is not None:
        batch = 8 if max_turns <= 10 else 5        # keep episodes to about 5-8 replies
    summ = None
    while True:
        turns += 1
        n = np.array(sorted(tr), int)
        y = np.array([prior.T0_cat + k * prior.P_cat + tr[k] / 1440 for k in n])
        post = Posterior(prior, n, y, [r[0] for r in rv], [r[1] for r in rv])
        remaining = BUDGET - spent
        rate_t = rate_r = 0.0
        if remaining >= 1:
            S = post.sample(160, rng)
            free = np.array([k for k in range(NMAX + 1) if k not in tr])
            epochs = rng.choice(free, min(48, len(free)), replace=False)
            times = np.round(rng.uniform(0, T["t_max"], 48), 2)
            ptr, prv = post.predict(S, epochs, times)
            eig_t = label_eig(ptr, S["label"], prior.sig_t); rate_t = float(eig_t.max())
            if remaining >= 3 and T["rv_ok"]:
                eig_r = label_eig(prv, S["label"], prior.sig_rv) / 3; rate_r = float(eig_r.max())
        if summ is not None:
            e = dict(p_none=summ["p_no_perturber"], p_both=summ["best_perturber"]["p_both_right"], p_period=summ["best_perturber"]["p_period_right"],
                     p_mass=summ["best_perturber"]["p_mass_right"], period=summ["best_perturber"]["period_days"], mass=summ["best_perturber"]["mass_earth"],
                     spent=spent, rate=max(rate_t, rate_r))
            if remaining < 1 or stop(e) or (max_turns is not None and turns >= max_turns):
                a = bayes_answer(e, c_abst, mode, wrong_penalty)
                args = {"verdict": a["verdict"]}
                if a["verdict"] == "planet":
                    args.update(period_days=e["period"], mass_earth=e["mass"])
                why = (f"The fit gives p_no_perturber={e['p_none']:.2f} and p_both_right={e['p_both']:.2f}; "
                       + {"none": "no perturber.", "planet": ("the perturber is pinned down." if e["p_both"] >= 0.8 else "I report the fit's best estimate."),
                          "cannot_determine": "more data would not settle it."}[a["verdict"]])
                yield why, [("submit_answer", args)]
                return
        if rate_r > rate_t:
            idx = [int(i) for i in np.argsort(-eig_r)[:max(1, min(batch // 2, int(remaining // 3)))]]
            out = yield "RVs are the most informative next step.", [("observe_rvs", {"times_days": [float(times[i]) for i in idx]}), ("fit_model", {})]
            rv += [tuple(x) for x in out[0]["time_and_rv_ms"]]
        else:
            idx = [int(i) for i in np.argsort(-eig_t)[:int(min(batch, remaining))]]
            out = yield "More transit times are the most informative next step.", [("observe_transits", {"epochs": sorted(int(epochs[i]) for i in idx)}), ("fit_model", {})]
            for k, oc in out[0]["epoch_and_oc_min"]:
                tr[int(k)] = oc
        spent = out[0]["spent"]
        summ = out[1] if "best_perturber" in out[1] else summ


def compact_text_agent(user_prompt: str, rng, n_transits=14, n_rv=5, diverse=False, dup_demo=0.0, smart=False):
    """Simple, deliberately suboptimal demonstrator (<= 8 turns): a fixed batch of transits -> periodogram fit ->
    one or two RV batches -> pick the alias by fit_rv. It never consults fit_model, always buys the same amount,
    and decides by fixed chi-square thresholds.
    smart=True is the offline 'reachable headroom' tweak: same purchases, but the stop/answer decision uses fit_model
    (commit iff its probability > 0.5, with the fit's numbers; stop before RVs if p_no_perturber >= 0.9)."""
    T = parse_task(user_prompt)
    if diverse:
        n_transits = int(rng.integers(12, 17)); n_rv = int(rng.integers(4, 7))
    ep0 = np.unique(np.clip(np.linspace(0, NMAX, n_transits).round().astype(int) + rng.integers(-4, 5, n_transits), 0, NMAX))
    first = [("observe_transits", {"epochs": [int(x) for x in ep0]})]
    yield "Sample the transit times sparsely across the whole baseline.", first
    if smart:
        (fm,) = yield "Joint fit.", [("fit_model", {})]
        if "best_perturber" in fm and fm["p_no_perturber"] >= 0.9:
            yield f"p_no_perturber is {fm['p_no_perturber']:.2f}: no perturber.", [("submit_answer", {"verdict": "none"})]
            return
    (fit,) = yield "Is there a periodic timing signal?", [("fit_transit_timing", {})]
    pk = fit["peaks"][0]
    if pk["delta_chi2"] < 25 or not pk["aliases"]:
        yield "No significant timing variation, so no perturber.", [("submit_answer", {"verdict": "none"})]
        return
    al = pk["aliases"]
    if not T["rv_ok"] or max(a["K2_if_circular_ms"] for a in al) < 0.7 * T["sig_rv"]:
        yield "The implied RV signal is far below the noise; the alias cannot be resolved.", [("submit_answer", {"verdict": "cannot_determine"})]
        return
    last_times = None
    for attempt in range(2):
        times = [round(float(t), 2) for t in np.sort(rng.uniform(0, T["t_max"], n_rv if attempt == 0 else 4))]
        if attempt == 1 and dup_demo and rng.random() < dup_demo and last_times:
            (bad,) = yield "DUPLICATE_DEMO", [("observe_rvs", {"times_days": last_times})]
            times = bad.get("new_times_you_could_request") or times
            calls = [("observe_rvs", {"times_days": times})]
            yield "Those times were already observed, so the request was refused. I use new times instead.", calls
        else:
            yield ("RVs will tell the aliases apart and measure the mass." if attempt == 0 else "Not decisive yet: more RVs at new times."), [("observe_rvs", {"times_days": times})]
        last_times = times
        if smart:
            (fm,) = yield "Joint fit of everything.", [("fit_model", {})]
            b = fm.get("best_perturber")
            if b and (b["p_both_right"] > 0.85 or attempt == 1):
                if b["p_both_right"] > 0.5:
                    yield f"p_both_right is {b['p_both_right']:.2f}: report the fit.", [("submit_answer", {"verdict": "planet", "period_days": b["period_days"], "mass_earth": b["mass_earth"]})]
                elif fm["p_no_perturber"] > 0.5:
                    yield "Most likely no perturber.", [("submit_answer", {"verdict": "none"})]
                else:
                    yield "Neither option is more likely than not.", [("submit_answer", {"verdict": "cannot_determine"})]
                return
            continue
        (rf,) = yield "Compare the aliases.", [("fit_rv", {"periods_days": [a["P2_days"] for a in al]})]
        fits = sorted([f for f in rf.get("fits", []) if "delta_chi2" in f], key=lambda f: -f["delta_chi2"])
        if not fits:
            break
        margin = fits[0]["delta_chi2"] - (fits[1]["delta_chi2"] if len(fits) > 1 else 0.0)
        if fits[0]["delta_chi2"] > 12 and margin > 6:
            a = next(x for x in al if x["P2_days"] == fits[0]["P2_days"])
            mass = fits[0]["mass_earth"] if fits[0]["K2_ms"] > 5 * fits[0]["K2_sigma_ms"] else (
                a["mass_if_circular"] if T["sigma_e"] == 0 else fits[0]["mass_earth"])
            yield "One alias is clearly preferred.", [("submit_answer", {"verdict": "planet", "period_days": fits[0]["P2_days"], "mass_earth": mass})]
            return
    yield "The aliases cannot be separated.", [("submit_answer", {"verdict": "cannot_determine"})]


def random_text_agent(user_prompt: str, rng):
    """Uninformed baseline through the text interface: random purchases, random answer."""
    T = parse_task(user_prompt)
    k = int(rng.integers(0, 16))
    if k:
        yield "", [("observe_transits", {"epochs": [int(x) for x in rng.choice(NMAX + 1, k, replace=False)]})]
    k = int(rng.integers(0, 6))
    if k:
        yield "", [("observe_rvs", {"times_days": [round(float(t), 2) for t in rng.uniform(0, T["t_max"], k)]})]
    v = ["none", "cannot_determine", "planet"][int(rng.integers(3))]
    args = {"verdict": v}
    if v == "planet":
        args.update(period_days=round(T["P_cat"] * 2 * (1 + rng.choice([-1, 1]) * rng.uniform(0.01, 0.05)), 3),
                    mass_earth=round(float(np.exp(rng.uniform(np.log(4), np.log(80)))), 1))
    yield "", [("submit_answer", args)]


def fit_following_agent(user_prompt: str, rng, max_turns=None, noise=0.0, mode="partial", wrong_penalty=0.5, guess_early=False, dup_demo=0.0, diverse=False):
    """A policy that uses ONLY numbers printed by the tools (no inference of its own): observe, call
    fit_model, and branch on p_no_perturber / p_both_right. Every action is a simple function of the
    visible text, which makes it the cleanest SFT teacher and the text-only solvability proof.
    `noise` > 0 randomises thresholds and batch sizes (used to emulate an imperfect learner)."""
    T = parse_task(user_prompt)
    jit = lambda x, s: float(x + noise * s * rng.normal())
    hi_none, hi_pl = min(0.99, jit(0.90, 0.08)), min(0.99, jit(0.85, 0.1))
    n0 = int(np.clip(round(jit(14, 4)), 6, 20))
    if diverse:                                                 # SFT data: vary the opening so the student does not memorise one schedule
        n0 = int(rng.integers(12, 17))
    used = set(int(x) for x in np.unique(np.clip(np.linspace(0, NMAX, n0).round().astype(int) + rng.integers(-5 if diverse else -4, 6 if diverse else 5, n0), 0, NMAX)))
    out = yield "Start with transit times spread over the whole baseline, then fit.", [("observe_transits", {"epochs": sorted(used)}), ("fit_model", {})]
    turns, n_rv, no_rv_tries = 1, 0, 0
    last_req = ("observe_transits", {"epochs": sorted(used)})
    while True:
        if dup_demo and "unobserved_epochs_you_could_request" not in out[0] and "new_times_you_could_request" not in out[0] \
                and rng.random() < dup_demo and (max_turns is None or turns < max_turns - 3) and BUDGET - out[0].get("spent", 0.0) >= 6:
            # Recovery demonstration for SFT: a request that repeats old values is refused; the teacher then switches to the
            # alternatives the tool lists. The mistaken turn is flagged so that SFT does not train on it.
            bad = yield "DUPLICATE_DEMO", [last_req, ("fit_model", {})]
            turns += 1
            alt_e, alt_t = bad[0].get("unobserved_epochs_you_could_request"), bad[0].get("new_times_you_could_request")
            if alt_e:
                used |= set(alt_e)
                out = yield "Those epochs were already observed, so the request was refused. I use unobserved epochs instead.", [("observe_transits", {"epochs": alt_e}), ("fit_model", {})]
                turns += 1
            elif alt_t:
                n_rv += len(alt_t)
                out = yield "Those times were already observed, so the request was refused. I use new times instead.", [("observe_rvs", {"times_days": alt_t}), ("fit_model", {})]
                turns += 1
            else:
                out = [dict(out[0], spent=bad[0].get("spent", out[0].get("spent", 0.0))), bad[1] if "best_perturber" in bad[1] else out[1]]
        spent = out[0].get("spent", 0.0)
        s = out[1] if "best_perturber" in out[1] else None
        if s is None:
            yield "The fit failed; I cannot conclude.", [("submit_answer", {"verdict": "cannot_determine"})]
            return
        b = s["best_perturber"]; p0 = s["p_no_perturber"]
        if noise and rng.random() < 0.5 * noise:                # an occasional blunder
            yield "", [("submit_answer", {"verdict": ["none", "cannot_determine"][int(rng.integers(2))]})]
            return
        last = (max_turns is not None and turns >= max_turns - 1) or BUDGET - spent < 6
        if guess_early and p0 < 0.3:                            # a degenerate policy seen in live RL runs: submit the fit's best alias without RVs
            yield f"p_both_right is {b['p_both_right']:.2f}; report the fit's best estimate now.", [("submit_answer", {"verdict": "planet", "period_days": b["period_days"], "mass_earth": b["mass_earth"]})]
            return
        if p0 >= hi_none:
            yield f"p_no_perturber is {p0:.2f}: no perturber.", [("submit_answer", {"verdict": "none"})]
            return
        if b["p_both_right"] >= hi_pl:
            yield f"p_both_right is {b['p_both_right']:.2f}: report the fitted perturber.", [("submit_answer", {"verdict": "planet", "period_days": b["period_days"], "mass_earth": b["mass_earth"]})]
            return
        if last or (n_rv >= 9 and b["p_both_right"] < 0.5 and p0 < 0.5):
            e = dict(p_none=p0, p_both=b["p_both_right"], p_period=b["p_period_right"], p_mass=b["p_mass_right"], period=b["period_days"], mass=b["mass_earth"])
            a = bayes_answer(e, 0.15, mode, wrong_penalty)
            args = {"verdict": a["verdict"]}
            if a["verdict"] == "planet":
                args.update(period_days=b["period_days"], mass_earth=b["mass_earth"])
            yield (f"Out of useful observations (p_no_perturber {p0:.2f}, p_both_right {b['p_both_right']:.2f}); "
                   + {"none": "most likely no perturber.", "planet": "I report the fit's best estimate.", "cannot_determine": "this cannot be settled."}[a["verdict"]]), [("submit_answer", args)]
            return
        if p0 >= 0.3:                                           # maybe nothing there: more transits are the cheap test
            k = int(min(8, BUDGET - spent))
            free = [n for n in range(NMAX + 1) if n not in used]
            more = sorted(int(x) for x in rng.choice(free, k, replace=False)); used |= set(more)
            last_req = ("observe_transits", {"epochs": more})
            out = yield f"p_no_perturber is {p0:.2f}, not decisive: more transit times.", [("observe_transits", {"epochs": more}), ("fit_model", {})]
        elif not T["rv_ok"]:                                   # perturber likely but RVs impossible: one more batch of transits, then decide
            if no_rv_tries >= 1:
                e = dict(p_none=p0, p_both=b["p_both_right"], p_period=b["p_period_right"], p_mass=b["p_mass_right"], period=b["period_days"], mass=b["mass_earth"])
                a = bayes_answer(e, 0.15, mode, wrong_penalty)
                args = {"verdict": a["verdict"]}
                if a["verdict"] == "planet":
                    args.update(period_days=b["period_days"], mass_earth=b["mass_earth"])
                yield (f"RVs are not possible here and p_both_right is still {b['p_both_right']:.2f}; "
                       + {"none": "most likely no perturber.", "planet": "the fit is already decisive.", "cannot_determine": "the aliases cannot be separated: cannot determine."}[a["verdict"]]), [("submit_answer", args)]
                return
            no_rv_tries += 1
            k = int(min(8, BUDGET - spent))
            free = [n for n in range(NMAX + 1) if n not in used]
            more = sorted(int(x) for x in rng.choice(free, k, replace=False)); used |= set(more)
            out = yield f"A perturber is likely but RVs are not possible: try more transit times (p_both_right {b['p_both_right']:.2f}).", [("observe_transits", {"epochs": more}), ("fit_model", {})]
        else:                                                   # a perturber is there: RVs separate the aliases and give the mass
            k = int(min((int(rng.integers(4, 7)) if diverse else 5) if n_rv == 0 else (int(rng.integers(2, 5)) if diverse else 3), (BUDGET - spent) // 3))
            times = [round(float(t), 2) for t in np.sort(rng.uniform(0, T["t_max"], k))]
            n_rv += k
            last_req = ("observe_rvs", {"times_days": times})
            out = yield f"A perturber is likely (p_both_right {b['p_both_right']:.2f}): new RV times to separate the aliases and measure the mass.", [("observe_rvs", {"times_days": times}), ("fit_model", {})]
        turns += 1


def teacher_decide(user_prompt: str, turns, rng, max_turns=14, mode="strict0", wrong_penalty=1.0):
    """Stateless version of fit_following_agent for relabelling (DAgger): given the task text and the turns
    so far [(calls, results), ...] as seen in ANY policy's episode, return the teacher's next (thought, calls).
    Uses only tool outputs: epochs already bought, RVs bought, budget, and the latest fit_model summary."""
    T = parse_task(user_prompt)
    used, n_rv, spent, fit, n_obs = set(), 0, 0.0, None, 0
    for calls, results in turns:
        for (name, _), r in zip(calls, results):
            if not isinstance(r, dict):
                continue
            used |= {int(e[0]) for e in r.get("epoch_and_oc_min", [])}
            n_rv += len(r.get("time_and_rv_ms", []))
            if "spent" in r:
                spent = float(r["spent"])
            if name == "fit_model" and "best_perturber" in r:
                fit = r
    remaining = BUDGET - spent
    n_obs = len(used) + n_rv
    if turns:                                                   # the last request was refused as a duplicate: take the listed alternatives
        for (name, _), r in zip(*turns[-1]):
            if isinstance(r, dict) and r.get("unobserved_epochs_you_could_request") and remaining >= 1 and len(turns) < max_turns - 1:
                return ("Those epochs were already observed, so the request was refused. I use unobserved epochs instead.",
                        [("observe_transits", {"epochs": r["unobserved_epochs_you_could_request"]}), ("fit_model", {})])
            if isinstance(r, dict) and r.get("new_times_you_could_request") and remaining >= 3 and len(turns) < max_turns - 1:
                return ("Those times were already observed, so the request was refused. I use new times instead.",
                        [("observe_rvs", {"times_days": r["new_times_you_could_request"]}), ("fit_model", {})])
    if n_obs < 6:                                               # (almost) nothing bought yet: the opening move
        ep = np.unique(np.clip(np.linspace(0, NMAX, 14).round().astype(int) + rng.integers(-4, 5, 14), 0, NMAX))
        ep = sorted(int(x) for x in ep if int(x) not in used)[:int(max(remaining, 0))]
        if not ep:
            return "No budget for observations; this cannot be settled.", [("submit_answer", {"verdict": "cannot_determine"})]
        return "Start with transit times spread over the whole baseline, then fit.", [("observe_transits", {"epochs": ep}), ("fit_model", {})]
    if fit is None:
        return "Fit everything bought so far.", [("fit_model", {})]
    b = fit["best_perturber"]; p0 = fit["p_no_perturber"]
    last = len(turns) >= max_turns - 1 or remaining < 6

    def final(prefix):
        e = dict(p_none=p0, p_both=b["p_both_right"], p_period=b["p_period_right"], p_mass=b["p_mass_right"], period=b["period_days"], mass=b["mass_earth"])
        a = bayes_answer(e, 0.15, mode, wrong_penalty)
        args = {"verdict": a["verdict"]}
        if a["verdict"] == "planet":
            args.update(period_days=b["period_days"], mass_earth=b["mass_earth"])
        return (prefix + {"none": "most likely no perturber.", "planet": "I report the fit's best estimate.", "cannot_determine": "this cannot be settled."}[a["verdict"]]), [("submit_answer", args)]

    if p0 >= 0.90:
        return f"p_no_perturber is {p0:.2f}: no perturber.", [("submit_answer", {"verdict": "none"})]
    if b["p_both_right"] >= 0.85:
        return f"p_both_right is {b['p_both_right']:.2f}: report the fitted perturber.", [("submit_answer", {"verdict": "planet", "period_days": b["period_days"], "mass_earth": b["mass_earth"]})]
    if last or (n_rv >= 9 and b["p_both_right"] < 0.5 and p0 < 0.5):
        return final(f"Out of useful observations (p_no_perturber {p0:.2f}, p_both_right {b['p_both_right']:.2f}); ")
    free = [n for n in range(NMAX + 1) if n not in used]
    if p0 >= 0.3:
        k = int(min(8, remaining))
        more = sorted(int(x) for x in rng.choice(free, k, replace=False))
        return f"p_no_perturber is {p0:.2f}, not decisive: more transit times.", [("observe_transits", {"epochs": more}), ("fit_model", {})]
    if not T["rv_ok"]:
        if len(used) >= 22:
            return final(f"RVs are not possible here and p_both_right is still {b['p_both_right']:.2f}; ")
        more = sorted(int(x) for x in rng.choice(free, int(min(8, remaining)), replace=False))
        return f"A perturber is likely but RVs are not possible: try more transit times (p_both_right {b['p_both_right']:.2f}).", [("observe_transits", {"epochs": more}), ("fit_model", {})]
    k = int(min(5 if n_rv == 0 else 3, remaining // 3))
    times = [round(float(t), 2) for t in np.sort(rng.uniform(0, T["t_max"], k))]
    return f"A perturber is likely (p_both_right {b['p_both_right']:.2f}): RVs to separate the aliases and measure the mass.", [("observe_rvs", {"times_days": times}), ("fit_model", {})]


LITE_THRESHOLDS = dict(det=20.0, sep=14.0, ksnr=4.0, sep_low=4.0, ksnr_low=2.0, max_rv=13)   # tuned on 300 TRAIN seeds (scripts/lite_headroom.py)


def lite_teacher_agent(user_prompt: str, rng, max_turns=None, thr=None, diverse=False, dup_demo=0.0, **_):
    """Scripted teacher for TOOL_LEVEL=lite: fixed delta-chi2 thresholds on the lite fit_model output, no probabilities.
    detect  : best alias delta_chi2_vs_none >= det
    commit  : detected, >= 5 RVs, delta_chi2_between_aliases >= sep and K2 / K2_sigma >= ksnr -> 'planet' with the best alias' numbers
    none    : not detected after up to two transit batches
    abstain : detected but the aliases or the mass stay unresolved when RVs run out"""
    th = dict(LITE_THRESHOLDS, **(thr or {}))
    T = parse_task(user_prompt)
    n0 = int(rng.integers(12, 17)) if diverse else 14
    used = set(int(x) for x in np.unique(np.clip(np.linspace(0, NMAX, n0).round().astype(int) + rng.integers(-4, 5, n0), 0, NMAX)))
    out = yield "Start with transit times spread over the whole baseline, then fit.", [("observe_transits", {"epochs": sorted(used)}), ("fit_model", {})]
    turns, n_rv = 1, 0
    last_req = None
    while True:
        spent = out[0].get("spent", 0.0); remaining = BUDGET - spent
        f = out[1] if "aliases" in out[1] else None
        if f is None or not f["aliases"]:
            yield "The fit failed; I cannot conclude.", [("submit_answer", {"verdict": "cannot_determine"})]
            return
        a0 = f["aliases"][0]
        d1, d12 = a0["delta_chi2_vs_none"], f.get("delta_chi2_between_aliases", 0.0)
        snr = (a0["K2_ms"] / a0["K2_sigma_ms"]) if a0.get("K2_sigma_ms") else 0.0
        last = (max_turns is not None and turns >= max_turns - 1)
        if dup_demo and last_req is not None and rng.random() < dup_demo and not last and remaining >= 6:
            bad = yield "DUPLICATE_DEMO", [last_req, ("fit_model", {})]
            turns += 1; last_req = None
            alt_e, alt_t = bad[0].get("unobserved_epochs_you_could_request"), bad[0].get("new_times_you_could_request")
            if alt_e:
                used |= set(alt_e)
                out = yield "Those epochs were already observed, so the request was refused. I use unobserved epochs instead.", [("observe_transits", {"epochs": alt_e}), ("fit_model", {})]
                turns += 1; continue
            if alt_t:
                n_rv += len(alt_t)
                out = yield "Those times were already observed, so the request was refused. I use new times instead.", [("observe_rvs", {"times_days": alt_t}), ("fit_model", {})]
                turns += 1; continue
        if d1 < th["det"]:
            if len(used) < 22 and remaining >= 8 and not last:
                more = sorted(int(x) for x in rng.choice([n for n in range(NMAX + 1) if n not in used], 8, replace=False)); used |= set(more)
                last_req = ("observe_transits", {"epochs": more})
                out = yield f"delta_chi2_vs_none is only {d1:.0f}: more transit times before deciding.", [last_req, ("fit_model", {})]
                turns += 1; continue
            yield f"delta_chi2_vs_none is {d1:.0f}, below a clear detection: no perturber.", [("submit_answer", {"verdict": "none"})]
            return
        if n_rv >= 5 and d12 >= th["sep"] and snr >= th["ksnr"]:
            yield (f"Clear detection; the aliases differ by delta chi2 {d12:.0f} and K2 is {snr:.1f} sigma: report the best alias."), \
                [("submit_answer", {"verdict": "planet", "period_days": a0["period_days"], "mass_earth": a0["mass_earth"]})]
            return
        k = int(min(5 if n_rv == 0 else 3, remaining // 3))
        if T["rv_ok"] and k >= 2 and n_rv < th["max_rv"] and not last:
            times = [round(float(t), 2) for t in np.sort(rng.uniform(0, T["t_max"], k))]
            n_rv += k; last_req = ("observe_rvs", {"times_days": times})
            out = yield f"A perturber is detected (delta chi2 {d1:.0f}) but the aliases differ by only {d12:.0f}: new RV times.", [last_req, ("fit_model", {})]
            turns += 1; continue
        if n_rv >= 5 and d12 >= th["sep_low"] and snr >= th["ksnr_low"]:
            yield f"No more useful observations; the best alias is moderately preferred (delta chi2 {d12:.0f}, K2 {snr:.1f} sigma): report it.", \
                [("submit_answer", {"verdict": "planet", "period_days": a0["period_days"], "mass_earth": a0["mass_earth"]})]
            return
        yield f"A perturber is detected but the aliases (delta chi2 {d12:.0f}) or the mass (K2 {snr:.1f} sigma) stay unresolved: cannot determine.", [("submit_answer", {"verdict": "cannot_determine"})]
        return


def fixed_fit_agent(user_prompt: str, rng, max_turns=None, diverse=False, dup_demo=0.0, mode="strict0", wrong_penalty=1.0, n_transits=16, n_rv=8, **_):
    """Demonstrator for the primary arm: it USES fit_model for its final decision (commit the fit's top option iff its
    probability > 0.5, else abstain) but its purchases are crude and fixed: one batch of transits, then always one batch
    of RVs, whatever the first fit says. It never stops early on a clear 'no perturber', never buys more when the fit is
    ambiguous. The headroom above it is purely in what to buy and when to stop."""
    T = parse_task(user_prompt)
    if diverse:
        n_transits = int(rng.integers(14, 19)); n_rv = int(rng.integers(7, 10))
    ep0 = np.unique(np.clip(np.linspace(0, NMAX, n_transits).round().astype(int) + rng.integers(-4, 5, n_transits), 0, NMAX))
    out = yield "Transit times spread over the whole baseline, then the joint fit.", [("observe_transits", {"epochs": [int(x) for x in ep0]}), ("fit_model", {})]
    if T["rv_ok"]:
        times = [round(float(t), 2) for t in np.sort(rng.uniform(0, T["t_max"], n_rv))]
        out = yield "Now a set of RVs at new times, then the joint fit again.", [("observe_rvs", {"times_days": times}), ("fit_model", {})]
    f = out[1] if (len(out) > 1 and isinstance(out[1], dict) and "best_perturber" in out[1]) else None
    if f is None:
        yield "The fit failed; I cannot conclude.", [("submit_answer", {"verdict": "cannot_determine"})]
        return
    b, p0 = f["best_perturber"], f["p_no_perturber"]
    if b["p_both_right"] >= p0 and b["p_both_right"] > 0.5:
        yield f"p_both_right is {b['p_both_right']:.2f}: report the fitted perturber.", [("submit_answer", {"verdict": "planet", "period_days": b["period_days"], "mass_earth": b["mass_earth"]})]
    elif p0 > 0.5:
        yield f"p_no_perturber is {p0:.2f}: no perturber.", [("submit_answer", {"verdict": "none"})]
    else:
        yield f"Neither option is more likely than not (p_no_perturber {p0:.2f}, p_both_right {b['p_both_right']:.2f}): cannot determine.", [("submit_answer", {"verdict": "cannot_determine"})]
