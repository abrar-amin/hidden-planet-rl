"""Framework-free text interface: prompt, tool implementations (JSON in / JSON out), grading.

The tinker-cookbook adapter (tinker_env.py) and the dry-run harness both go through this class,
so scripted policies and LLMs see exactly the same strings.
"""
from __future__ import annotations

import json
import math

import numpy as np

from . import physics as ph
from .inference import Posterior, Prior
from .reward import score, INVALID, LAMBDA, VERDICTS
from .tasks import NMAX, COST, BUDGET, SIGMA_P, SIGMA_T0, LEVELS, Episode, SystemParams, sample_system, target_id

MAX_PER_CALL = 16

SYSTEM_PROMPT = """You are an astronomer running an observing campaign. A star has one known transiting planet. There may be a second, hidden, non-transiting planet (a "perturber") just outside it, close to a first-order mean-motion resonance with it. Your job is to decide what to observe, when to stop, and what to conclude.

Physics you can rely on:
- A perturber with period P2 near P1*j/(j-1) ({res_txt}; offset Delta = (P2/P1)*(j-1)/j - 1 with 0.01 <= |Delta| <= 0.05, either sign) makes the transit times oscillate sinusoidally around a linear ephemeris, with a super-period of (1+Delta)/((j-1)|Delta|) transit epochs. The same super-period fits several aliases ({alias_txt}), i.e. several different P2.
- {ecc_txt}
- Radial velocities (RVs) contain a sinusoid at the true P2 whose amplitude K2 gives the mass directly, but only if K2 is detectable above the RV noise. RVs also contain the known planet's signal and an unknown constant offset.
- Perturber masses range from {mlo:.0f} to {mhi:.0f} Earth masses. About one star in three has no perturber at all.

A sensible campaign: buy a spread of transit times, call fit_transit_timing (or fit_model), and only if there is a clear signal buy RVs and compare the aliases with fit_rv or fit_model. You may put an observing call and a fit call in the same reply. Report numbers that a tool has shown you; do not invent them.

Costs: each transit time costs 1 unit, each RV costs 3 units, and the total budget is {budget:.0f} units. Analysis tools are free. You have at most {max_turns} turns.

How to reply: every reply must contain exactly one tool call. Write at most two short sentences of reasoning before it; never write a long plan or an essay.{format_hint}

Finish by calling submit_answer exactly once with one of:
- verdict "none": there is no perturber.
- verdict "planet" with period_days and mass_earth: your estimate of the perturber.
- verdict "cannot_determine": the perturber's period and mass cannot be determined within the budget.

Scoring: {planet_txt} "none" earns +1 if right and -{wp} if wrong. {cannot_txt} Every unit spent costs {lam:.3f}. Not submitting a valid answer scores -1, and each reply without a valid tool call costs 0.05. Spend only what helps, and do not guess when the data cannot decide."""


def task_prompt(sp: SystemParams) -> str:
    ecc = ("Orbits are circular (no free eccentricity)." if sp.sigma_e == 0 else
           f"Orbital eccentricities are small but unknown (typical value {sp.sigma_e:.3f}).")
    rv = sp.extra.get("rv")
    if rv is None:
        rv_txt = f"RV precision: {sp.sig_rv:.1f} m/s per point."
    elif rv["available"]:
        rv_txt = (f"Host star: V = {rv['vmag']:.1f} mag, activity jitter about {rv['jitter']:.1f} m/s. RV precision: {sp.sig_rv:.1f} m/s per point.")
    else:
        rv_txt = (f"Host star: V = {rv['vmag']:.1f} mag, activity jitter about {rv['jitter']:.1f} m/s. RVs are NOT possible for this star (too faint): "
                  f"observe_rvs will be refused. RV precision: none.")
    return (
        f"Target {target_id(sp)}. Known transiting planet: catalogue period P1 = {sp.P_cat:.5f} d (+/- {SIGMA_P:.4f}), "
        f"transit epoch T0 = {sp.T0_cat:.4f} d (+/- {SIGMA_T0:.3f}). You may observe transit epochs 0..{NMAX} "
        f"(epoch n occurs near T0 + n*P1) and RVs at any time from 0 to {NMAX * sp.P_cat:.1f} d. "
        f"Transit-time precision: {sp.sig_t * 1440:.1f} min per transit. {rv_txt} {ecc} "
        f"For reference, a 10 Earth-mass planet at period 2*P1 would give K2 = {ph.rv_semi_amplitude(10 * ph.ME, 2 * sp.P_cat):.2f} m/s. "
        f"Perturber masses here range from {LEVELS[sp.level]['mass'][0]:.0f} to {LEVELS[sp.level]['mass'][1]:.0f} Earth masses. "
        f"Budget: {BUDGET:.0f} units.")


TOOL_SPECS = [
    dict(name="observe_transits",
         description=f"Measure mid-transit times at the given epochs (cost 1 each, max {MAX_PER_CALL} per call). Returns observed-minus-catalogue (O-C) timing residuals in minutes; a straight-line trend in O-C only means the catalogue period is slightly off.",
         parameters=dict(type="object", properties=dict(epochs=dict(type="array", items=dict(type="integer"), description=f"Transit epoch numbers, each 0..{NMAX}")), required=["epochs"])),
    dict(name="observe_rvs",
         description=f"Measure the star's radial velocity at the given times in days (cost 3 each, max {MAX_PER_CALL} per call). Returns RVs in m/s.",
         parameters=dict(type="object", properties=dict(times_days=dict(type="array", items=dict(type="number"), description="Observation times in days")), required=["times_days"])),
    dict(name="fit_transit_timing",
         description="Free. Fit the transit times bought so far with a linear ephemeris and with linear + one sinusoid. Returns the chi-square improvement, best super-period and amplitude, and the perturber periods P2 that each resonance alias would imply.",
         parameters=dict(type="object", properties={}, required=[])),
    dict(name="fit_rv",
         description="Free. Fit the RVs bought so far with offset + known planet + a sinusoid at each trial period (max 8). Returns K2, its uncertainty, the chi-square improvement over no perturber, and the implied mass.",
         parameters=dict(type="object", properties=dict(periods_days=dict(type="array", items=dict(type="number"), description="Trial perturber periods in days")), required=["periods_days"])),
    dict(name="fit_model",
         description="Free. Joint Bayesian fit of ALL data bought so far (transit times and RVs) with the single-perturber model. Returns the probability that there is no perturber, the most probable perturber period and mass with the probability that each is right, and the probability of each alias. Use its numbers for your answer.",
         parameters=dict(type="object", properties={}, required=[])),
    dict(name="get_data_summary",
         description="Free. List all observations bought so far and the remaining budget.",
         parameters=dict(type="object", properties={}, required=[])),
    dict(name="submit_answer",
         description="Submit the final answer and end the episode.",
         parameters=dict(type="object", properties=dict(
             verdict=dict(type="string", enum=list(VERDICTS)),
             period_days=dict(type=["number", "null"], description="Perturber period in days (verdict 'planet' only)"),
             mass_earth=dict(type=["number", "null"], description="Perturber mass in Earth masses (verdict 'planet' only)")),
             required=["verdict"])),
]


def _r(x, k=3):
    return round(float(x), k)


class HiddenPlanetSession:
    """One episode behind the text interface. call(name, args) -> (json_string, done)."""

    def __init__(self, sp: SystemParams, q_ref: float | None = None, backend: str | None = None,
                 abstain_aware: bool = True, max_turns: int | None = None, allow_other: bool = False,
                 cost_per_unit: float | None = None, reward_mode: str = "partial", wrong_penalty: float | None = None,
                 tool_level: str = "oracle", train_reward: str = "realised"):
        self.sp = sp
        if train_reward not in ("realised", "expected"):
            raise ValueError(f"train_reward must be 'realised' or 'expected', not {train_reward!r}")
        self.train_reward = train_reward      # 'expected': grade() returns the posterior-expected score (expected_score.py) instead of the realised one
        self.ep = Episode(sp, backend=backend)
        self.backend = backend
        self._q_ref = q_ref
        self.abstain_aware = abstain_aware
        self.max_turns = max_turns or LEVELS[sp.level].get("max_turns", 16)
        self.lam = LAMBDA if cost_per_unit is None else float(cost_per_unit)
        self.reward_mode, self.wrong_penalty = reward_mode, wrong_penalty
        self.tool_level = tool_level          # 'oracle': fit_model prints posterior probabilities; 'lite': chi2 / BIC only
        self._last_fit = None                 # last fit_model output (for the submit_conf metric)
        self.allow_other = allow_other        # optional 4th verdict for out-of-class evals (default off)
        self.answer = None
        self.done = False
        self.n_calls = 0
        self.n_errors = 0

    @classmethod
    def from_seed(cls, seed, level=5, split="train", ooc_kind=None, **kw):
        if ooc_kind:                                   # out-of-class truth (level 5 systems + hidden physics)
            from .ooc import sample_ooc_system
            return cls(sample_ooc_system(seed, ooc_kind, split), **kw)
        return cls(sample_system(seed, level, split), **kw)

    OTHER_TEXT = ('\n\nA fourth verdict is available: "other" means the data show a real signal that a single '
                  'near-resonant perturber cannot explain (for example a second perturber, stellar activity in the RVs, '
                  'or a steady drift of the transit times). It earns +1 if that is the case and -0.5 if it is not.')

    def tool_specs(self):
        lite = getattr(self, "tool_level", "oracle") == "lite"
        if not self.allow_other and not lite:
            return TOOL_SPECS
        import copy
        specs = copy.deepcopy(TOOL_SPECS)
        if self.allow_other:
            specs[-1]["parameters"]["properties"]["verdict"]["enum"] = list(VERDICTS) + ["other"]
        if lite:
            fm = next(t for t in specs if t["name"] == "fit_model")
            fm["description"] = ("Free. Joint maximum-likelihood fit of ALL data bought so far (transit times and RVs). Returns chi2, n_params and BIC for the no-perturber model "
                                 "and, for each resonance alias, the best-fit period and mass, chi2, delta_chi2_vs_none and BIC, plus delta_chi2_between_aliases. "
                                 "It gives NO probabilities: you must judge from the chi-square values whether to buy more, commit or abstain. Use its numbers for your answer.")
        return specs

    # ---------------- prompt ----------------
    def system_prompt(self, format_hint: str = ""):
        L = LEVELS[self.sp.level]
        res = L["res"]
        res_txt = "j = 2, the 2:1 resonance" if res == (2,) else "j = " + ", ".join(map(str, res))
        alias_txt = "the two signs of Delta" if res == (2,) else "each j and each sign of Delta"
        ecc_txt = ("Orbits here are circular, so for a given alias the oscillation amplitude gives the perturber mass directly." if L["sigma_e"] == 0 else
                   "The oscillation amplitude grows with the perturber mass but also depends on the unknown orbital eccentricities, so transit times alone do not pin down the mass.")
        wp = "0.5" if (self.reward_mode == "partial" or self.wrong_penalty is None) else f"{self.wrong_penalty:g}"
        planet_txt = ('"planet" earns +0.5 if the period is within 0.5% and +0.5 if the mass is within a factor 1.5 (each is -0.25 if wrong).' if self.reward_mode == "partial" else
                      f'"planet" earns +1 only if BOTH the period (within 0.5%) and the mass (within a factor 1.5) are right; otherwise it costs {wp}. '
                      'Guessing between two aliases that fit_model rates near 50/50 loses on average: buy RVs until p_both_right is high, or answer cannot_determine.')
        cannot_txt = '"cannot_determine" earns +0.4 if the answer truly could not be pinned down from affordable data, and -0.3 if it could.'
        if self.reward_mode == "strict0":
            wp = "1" if self.wrong_penalty is None else f"{self.wrong_penalty:g}"
            planet_txt = (f'"planet" earns +1 only if BOTH the period (within 0.5%) and the mass (within a factor 1.5) are right; otherwise it costs {wp}.')
            cannot_txt = ('"cannot_determine" always earns 0. So commit to "planet" or "none" only when fit_model puts its probability above one half '
                          '(p_both_right or p_no_perturber); if you can still afford observations that would raise it, buy them first; otherwise answer cannot_determine.')
        if getattr(self, "tool_level", "oracle") == "lite" and self.reward_mode == "strict0":
            cannot_txt = ('"cannot_determine" always earns 0. So commit to "planet" or "none" only when you judge it more likely right than wrong from the fit statistics: '
                          'fit_model gives chi2, delta_chi2_vs_none, delta_chi2_between_aliases and BIC, not probabilities. If you can still afford observations that would '
                          'separate the options, buy them first; otherwise answer cannot_determine.')
        return SYSTEM_PROMPT.format(budget=BUDGET, lam=self.lam, max_turns=self.max_turns, res_txt=res_txt, alias_txt=alias_txt,
                                    format_hint=(" A tool call looks exactly like this:\n" + format_hint) if format_hint else "",
                                    ecc_txt=ecc_txt, mlo=L["mass"][0], mhi=L["mass"][1], wp=wp, planet_txt=planet_txt, cannot_txt=cannot_txt) + (self.OTHER_TEXT if self.allow_other else "")

    def user_prompt(self):
        return task_prompt(self.sp)

    # ---------------- tools ----------------
    def call(self, name: str, args: dict):
        self.n_calls += 1
        if self.done:
            return json.dumps({"error": "episode already finished"}), True
        fn = getattr(self, "_tool_" + name, None)
        if fn is None:
            self.n_errors += 1
            return json.dumps({"error": f"unknown tool {name}"}), False
        try:
            if not isinstance(args, dict):
                raise TypeError("arguments must be a JSON object")
            out = fn(**args)
            if isinstance(out, dict) and "error" in out:
                self.n_errors += 1
            text = json.dumps(out, separators=(",", ":"), allow_nan=False)
        except TypeError as e:
            self.n_errors += 1
            text = json.dumps({"error": f"bad arguments: {str(e)[:200]}"})
        except Exception as e:                      # never let a tool crash a rollout
            self.n_errors += 1
            text = json.dumps({"error": f"{name} failed ({type(e).__name__}); check the arguments and the data bought so far"})
        return text, self.done

    def _budget(self):
        return {"spent": _r(self.ep.spent, 1), "remaining": _r(self.ep.remaining, 1)}

    def _tool_observe_transits(self, epochs):
        if not isinstance(epochs, (list, tuple)) or not epochs:
            self.n_errors += 1
            return {"error": "epochs must be a non-empty list of integers"}
        res, errs = [], []
        for n in list(epochs)[:MAX_PER_CALL]:
            try:
                ni = int(n)
                if ni != float(n):
                    raise ValueError
            except (TypeError, ValueError):
                errs.append(f"{n!r}: not an integer epoch"); continue
            y, err = self.ep.observe_transit(ni)
            if err:
                errs.append(f"{ni}: {err}")
            else:
                oc = (y - (self.sp.T0_cat + ni * self.sp.P_cat)) * 1440
                res.append([ni, _r(oc, 2)])
        out = {"epoch_and_oc_min": res, "sigma_min": _r(self.sp.sig_t * 1440, 2), **self._budget()}
        if len(epochs) > MAX_PER_CALL:
            errs.append(f"only the first {MAX_PER_CALL} epochs were used")
        if errs:
            out["errors"] = errs
            if any("already observed" in e for e in errs):
                out.update(self._alternatives("transit", bought=len(res)))
        return out

    def _tool_observe_rvs(self, times_days):
        if not isinstance(times_days, (list, tuple)) or not times_days:
            self.n_errors += 1
            return {"error": "times_days must be a non-empty list of numbers"}
        res, errs = [], []
        for t in list(times_days)[:MAX_PER_CALL]:
            try:
                tf = float(t)
                if not math.isfinite(tf):
                    raise ValueError
            except (TypeError, ValueError):
                errs.append(f"{t!r}: not a number"); continue
            v, err = self.ep.observe_rv(tf)
            if err:
                errs.append(f"{_r(tf, 2)}: {err}")
            else:
                res.append([_r(tf, 3), _r(v, 2)])
        out = {"time_and_rv_ms": res, "sigma_ms": _r(self.sp.sig_rv, 2), **self._budget()}
        if len(times_days) > MAX_PER_CALL:
            errs.append(f"only the first {MAX_PER_CALL} times were used")
        if errs:
            out["errors"] = errs
            if any("already observed" in e for e in errs):
                out.update(self._alternatives("rv", bought=len(res)))
        return out

    def _alternatives(self, kind, bought):
        """After a duplicate request: concrete unobserved alternatives, so that repeating the same call is never the only visible option."""
        out = {}
        if kind == "transit":
            pool = getattr(self.ep, "_pool", None)
            free = [n for n in range(NMAX + 1) if n not in self.ep.transits and (pool is None or n in pool)]
            k = int(min(8, len(free), max(self.ep.remaining, 0)))
            if k > 0:
                idx = np.linspace(0, len(free) - 1, k).round().astype(int)
                out["unobserved_epochs_you_could_request"] = sorted({int(free[i]) for i in idx})
        else:
            have = np.array([r[0] for r in self.ep.rvs] or [-1e9])
            k = int(min(5, max(self.ep.remaining, 0) // COST["rv"]))
            cand = [t for t in np.linspace(0.03, 0.97, 41) * self.ep.t_max if np.min(np.abs(have - t)) > 1.0]
            if k > 0 and cand:
                idx = np.linspace(0, len(cand) - 1, k).round().astype(int)
                out["new_times_you_could_request"] = sorted({_r(cand[i], 1) for i in idx})
        out["hint"] = (("Nothing was bought. " if bought == 0 else "") + "Do not repeat a request that was refused: it will be refused again. "
                       "Use values you have not observed yet (examples above), or call submit_answer.")
        return out

    def _tool_get_data_summary(self):
        n, y, t, v = self.ep.data()
        oc = (y - (self.sp.T0_cat + n * self.sp.P_cat)) * 1440
        return {"transits_epoch_oc_min": [[int(a), _r(b, 2)] for a, b in zip(n, oc)],
                "rvs_time_ms": [[_r(a, 3), _r(b, 2)] for a, b in zip(t, v)], **self._budget()}

    def _tool_fit_transit_timing(self):
        n, y, _, _ = self.ep.data()
        if len(n) < 6 or (n.max() - n.min()) < 20:
            return {"error": f"need at least 6 distinct transit epochs spanning at least 20 epochs; you have {len(n)} spanning {int(n.max() - n.min()) if len(n) else 0}"}
        sp = self.sp
        n = n.astype(float)
        r = (y - (sp.T0_cat + n * sp.P_cat)) / sp.sig_t
        A0 = np.c_[np.ones_like(n), n] / sp.sig_t
        chi0 = float(np.sum((r - A0 @ np.linalg.lstsq(A0, r, rcond=None)[0]) ** 2))
        freqs = np.linspace(1 / 400.0, 0.5, 4000)             # cycles per epoch
        a = 2 * np.pi * freqs[:, None] * n[None, :]
        X = np.stack([np.broadcast_to(A0[:, 0], a.shape), np.broadcast_to(A0[:, 1], a.shape),
                      np.sin(a) / sp.sig_t, np.cos(a) / sp.sig_t], -1)       # (F, N, 4)
        X[..., 1] *= 1.0 / NMAX                                # condition the slope column
        G = np.einsum("fni,fnj->fij", X, X)
        G = G + (1e-7 * np.trace(G, axis1=1, axis2=2))[:, None, None] * np.eye(4)     # relative ridge: aliased frequencies stay solvable
        b = np.einsum("fni,n->fi", X, r)
        c = np.linalg.solve(G, b[..., None])[..., 0]
        chi = float(r @ r) - np.einsum("fi,fi->f", b, c)
        order = np.argsort(chi)
        peaks = []
        for i in order:
            if all(abs(freqs[i] - freqs[k]) > 0.004 for k in peaks):
                peaks.append(i)
            if len(peaks) == 2:
                break
        out = {"n_transits": int(len(n)), "chi2_linear": _r(chi0, 1), "peaks": []}
        for i in peaks:
            ntt = 1 / freqs[i]
            amp = float(np.hypot(c[i, 2], c[i, 3]))
            cands = []
            for j in LEVELS[sp.level]["res"]:
                for sg in (1, -1):
                    den = (j - 1) * ntt - sg
                    if den <= 0:
                        continue
                    D = sg / den
                    if not (0.008 <= abs(D) <= 0.06):
                        continue
                    P2 = sp.P_cat * j / (j - 1) * (1 + D)
                    m0 = amp * math.pi * j ** (2 / 3) * (j - 1) ** (1 / 3) * abs(D) / (sp.P_cat * abs(ph.coef_fg(j, D)[0])) / ph.ME
                    cands.append({"j": j, "delta": _r(D, 4), "P2_days": _r(P2, 4), "mass_if_circular": _r(m0, 1),
                                  "K2_if_circular_ms": _r(ph.rv_semi_amplitude(m0 * ph.ME, P2), 2)})
            out["peaks"].append({"delta_chi2": _r(chi0 - chi[i], 1), "super_period_epochs": _r(ntt, 2),
                                 "amplitude_min": _r(amp * 1440, 2), "aliases": cands})
        out["note"] = "delta_chi2 above ~25 is a clear detection (3 extra parameters, many trial periods)."
        return out

    def _tool_fit_rv(self, periods_days):
        _, _, t, v = self.ep.data()
        if len(t) < 5:
            return {"error": f"need at least 5 RVs; you have {len(t)}"}
        if not isinstance(periods_days, (list, tuple)) or not periods_days:
            return {"error": "periods_days must be a non-empty list"}
        sp = self.sp
        tt = t - sp.T0_cat
        w = 1 / sp.sig_rv
        A0 = np.c_[np.ones_like(tt), -np.sin(2 * np.pi * tt / sp.P_cat)] * w
        c0 = np.linalg.lstsq(A0, v * w, rcond=None)[0]
        chi0 = float(np.sum((v * w - A0 @ c0) ** 2))
        fits = []
        for P in list(periods_days)[:8]:
            try:
                P = float(P)
                assert math.isfinite(P) and P > 0
            except (TypeError, ValueError, AssertionError):
                fits.append({"error": f"bad period {P!r}"}); continue
            A = np.c_[A0, np.sin(2 * np.pi * tt / P) * w, np.cos(2 * np.pi * tt / P) * w]
            G = A.T @ A
            G = G + 1e-7 * np.trace(G) * np.eye(4)
            c = np.linalg.solve(G, A.T @ (v * w))
            chi = float(np.sum((v * w - A @ c) ** 2))
            cov = np.linalg.inv(G)
            K = float(np.hypot(c[2], c[3]))
            sK = float(np.sqrt(0.5 * (cov[2, 2] + cov[3, 3])))
            if not all(map(math.isfinite, (K, sK, chi))):
                fits.append({"error": f"fit failed at period {P}"}); continue
            fits.append({"P2_days": _r(P, 4), "K2_ms": _r(K, 2), "K2_sigma_ms": _r(sK, 2), "delta_chi2": _r(chi0 - chi, 1),
                         "mass_earth": _r(ph.k_to_mass_earth(K, P), 1), "K1_ms": _r(c[1], 2)})
        return {"n_rvs": int(len(t)), "chi2_no_perturber": _r(chi0, 1), "fits": fits}

    def _tool_fit_model(self):
        n, y, t, v = self.ep.data()
        if getattr(self, "tool_level", "oracle") == "lite":
            from .lite import lite_fit
            return lite_fit(self.sp, n, y, t, v)
        if len(n) + len(t) < 3:
            return {"error": "buy at least 3 observations first"}
        post = Posterior(Prior.from_system(self.sp), n, y, t, v)
        su = post.summary()
        out = {"n_transits": int(len(n)), "n_rvs": int(len(t)), "p_no_perturber": _r(su["p_none"], 3),
               "best_perturber": {"period_days": _r(su["period"], 4), "mass_earth": _r(su["mass"], 1),
                                  "p_period_right": _r(su["p_period"], 3), "p_mass_right": _r(su["p_mass"], 3),
                                  "p_both_right": _r(su["p_both"], 3)}}
        wc = post.w_cm.sum(1)
        al = []
        for j in sorted(set(int(x) for x in post.js)):
            for sg in (1, -1):
                m = (post.js == j) & (np.sign(post.deltas) == sg)
                if m.any() and wc[m].sum() > 0.005:
                    al.append({"j": j, "delta_sign": "+" if sg > 0 else "-", "prob": _r(wc[m].sum(), 3),
                               "P2_days": _r(float(np.sum(wc[m] * post.P2[m]) / wc[m].sum()), 3)})
        out["aliases"] = sorted(al, key=lambda a: -a["prob"])[:4]
        self._last_fit = out
        out["note"] = "Probabilities assume the single-perturber model. Answer 'planet' only if p_both_right is high, 'none' only if p_no_perturber is high."
        return out

    def _tool_submit_answer(self, verdict=None, period_days=None, mass_earth=None, **_ignored):
        self.answer = {"verdict": verdict, "period_days": period_days, "mass_earth": mass_earth}
        self.done = True
        return {"status": "answer recorded"}

    # ---------------- grading ----------------
    def q_ref(self):
        if self._q_ref is None:
            from .policies import reference_identifiability
            self._q_ref = reference_identifiability(self.sp, self.backend)
        if callable(self._q_ref):                 # shared, memoised per group
            self._q_ref = self._q_ref()
        return self._q_ref

    def grade(self):
        """(reward, metrics). Unanswered / invalid episodes get INVALID - cost. Never raises."""
        try:
            return self._grade()
        except Exception:
            r = float(INVALID - self.lam * self.ep.spent)
            return r, {"grade_error": 1.0, "invalid": 1.0, "spent": float(self.ep.spent), "correct": 0.0, "es": r, "reward_realised": r}

    def _grade(self):
        # the identifiability statistics only matter for 'cannot_determine' -> compute lazily
        q_agent = q_ref = 1.0
        if (self.answer or {}).get("verdict") == "cannot_determine" and self.abstain_aware and self.reward_mode != "strict0":
            q_agent = Posterior(Prior.from_system(self.sp), *self.ep.data()).prob_truth(self.sp)
            q_ref = self.q_ref()
        if self.allow_other or self.sp.extra.get("ooc"):
            from .ooc import score_ooc
            v = (self.answer or {}).get("verdict")
            if v in VERDICTS + ("other",):
                s = dict(determinable=True, has_planet=self.sp.has_planet, period_ok=0.0, mass_ok=0.0, correct=0.0)
                s.update(score_ooc(self.sp, self.answer, self.ep.spent, q_agent, q_ref, lam=self.lam, mode=self.reward_mode, wrong_penalty=self.wrong_penalty))
            else:
                s = score(self.sp, self.answer, self.ep.spent, q_agent, q_ref, self.abstain_aware, lam=self.lam,
                          mode=self.reward_mode, wrong_penalty=self.wrong_penalty)
        else:
            s = score(self.sp, self.answer, self.ep.spent, q_agent, q_ref, self.abstain_aware, lam=self.lam,
                          mode=self.reward_mode, wrong_penalty=self.wrong_penalty)
        cannot = s["verdict"] == "cannot_determine"
        metrics = {"score": s["score"], "spent": s["spent"], "correct": s["correct"],
                   "reward_std_cost": s["score"] - LAMBDA * s["spent"],        # comparable across --cost-per-unit settings
                   "determinable": float(s["determinable"]) if cannot else 1.0, "has_planet": float(s["has_planet"]),
                   "verdict_none": float(s["verdict"] == "none"), "verdict_planet": float(s["verdict"] == "planet"),
                   "verdict_cannot": float(s["verdict"] == "cannot_determine"), "invalid": float(s["verdict"] == "invalid"),
                   "abstain_correct": float(s["verdict"] == "cannot_determine" and not s["determinable"]),
                   "n_transits": float(len(self.ep.transits)), "n_rvs": float(len(self.ep.rvs)),
                   "verdict_other": float(s["verdict"] == "other"), "tool_errors": float(self.n_errors)}
        lf = getattr(self, "_last_fit", None)
        pb = lf["best_perturber"]["p_both_right"] if lf else 0.0
        conf = pb if s["verdict"] == "planet" else (lf["p_no_perturber"] if (lf and s["verdict"] == "none") else (pb if lf else 0.0))
        # Posterior probability of the SUBMITTED answer given the data bought, computed here for every arm and tool level,
        # whether or not the agent ever called fit_model (never shown to the agent). post_lowconf_commit is the guard metric.
        post_conf = float("nan")
        if s["verdict"] in ("planet", "none"):
            try:
                pq = Posterior(Prior.from_system(self.sp), *self.ep.data())
                post_conf = float(pq.prob_answer(float(self.answer["period_days"]), float(self.answer["mass_earth"]))[1]) if s["verdict"] == "planet" else float(pq.p_none)
            except Exception:
                post_conf = 0.0
        committed = s["verdict"] in ("planet", "none")
        metrics.update(rv_buy_rate=float(len(self.ep.rvs) > 0), submit_conf=float(conf),
                       lowconf_planet=float(s["verdict"] == "planet" and pb < 0.6),
                       lowconf_commit=float(committed and conf < 0.5),            # committed without a >0.5 fit_model reading on screen (1 if fit_model was never called)
                       post_conf=float(post_conf if committed else 0.0), committed=float(committed),
                       post_lowconf_commit=float(committed and post_conf <= 0.5),  # committed although the exact posterior gave the answer <= 50 %
                       used_fit_model=float(lf is not None))
        from .expected_score import es_from_metrics
        es = es_from_metrics(metrics, cost=self.lam)
        metrics.update(es=float(es), reward_realised=float(s["reward"]))
        if getattr(self, "train_reward", "realised") == "expected":
            return float(es), metrics          # truth still graded and logged (score, correct, reward_realised); only the returned reward changes
        return float(s["reward"]), metrics
