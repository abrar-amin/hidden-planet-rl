"""Reward. Three modes, selected per session (`reward_mode`); cost per unit is configurable (`lam`).

answer = {"verdict": "none" | "planet" | "cannot_determine", "period_days": float, "mass_earth": float}
Tolerances: period within 0.5 %, mass within a factor 1.5.

  partial (default) : planet +0.5 per right component, -0.25 per wrong one; none +1 / -0.5;
                      cannot +0.4 if the truth was not determinable (label below), else -0.3.
  strict            : planet +1 only if period AND mass are right, else -W (W = wrong_penalty, default 0.5);
                      none +1 / -W; cannot as in partial.
  strict0           : planet +1 / -W and none +1 / -W with W default 1; cannot = 0 always.
                      No dependence on the determinable label: answer iff P(right) > W / (1 + W).
                      This is the setting of the pre-registered experiment.
  invalid or no answer: -1.          reward = score - lam * spent   (lam default 0.005; the experiment uses 0.008)

"Determinable" (partial / strict only): max(q_agent, q_ref) >= TAU, with q the exact in-class posterior probability of
the true answer given (a) the agent's data and (b) full-budget reference designs on the same system.
"""
from __future__ import annotations

import math

LAMBDA = 0.005
A_ABSTAIN = 0.4
B_ABSTAIN = 0.3
TAU = 0.6
INVALID = -1.0
PERIOD_TOL = 0.005
MASS_FACTOR = 1.5
VERDICTS = ("none", "planet", "cannot_determine")


def score(sp, answer, spent, q_agent, q_ref, abstain_aware=True, lam=LAMBDA, mode="partial", wrong_penalty=None):
    """Grade one answer; returns a dict with the score, the reward (score - lam * spent) and per-component flags.

    mode 'partial' (default): +0.5 / -0.25 per component.
    mode 'strict': a 'planet' answer scores +1 only if BOTH period and mass are right, else -wrong_penalty (default 0.5);
    a wrong 'none' also costs wrong_penalty. A coin flip between two aliases is still worth 0.5 * 1 - 0.5 * 0.5 = +0.25
    with the default penalty (measured expected values: scripts/reward_mode_ev.py).
    mode 'strict0': like strict with a default penalty of 1, and 'cannot_determine' scores exactly 0 whatever the system.
    There is no dependence on the determinable label, so the optimal rule is "answer iff P(correct) > W / (1 + W)"."""
    W = (1.0 if mode == "strict0" else 0.5) if wrong_penalty is None else float(wrong_penalty)
    v = (answer or {}).get("verdict")
    determinable = max(q_agent, q_ref) >= TAU
    out = dict(verdict=v, determinable=bool(determinable), q_agent=float(q_agent), q_ref=float(q_ref),
               spent=float(spent), correct=0.0, period_ok=0.0, mass_ok=0.0, has_planet=bool(sp.has_planet))
    if v == "none":
        s = 1.0 if not sp.has_planet else (-0.5 if mode == "partial" else -W)
        out["correct"] = float(not sp.has_planet)
    elif v == "planet":
        try:
            P, m = float(answer["period_days"]), float(answer["mass_earth"])
            ok = math.isfinite(P) and math.isfinite(m) and P > 0 and m > 0
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            s = INVALID; out["verdict"] = "invalid"
        else:
            pok = sp.has_planet and abs(P / sp.P2 - 1) < PERIOD_TOL
            mok = sp.has_planet and abs(math.log(m / sp.m2)) < math.log(MASS_FACTOR)
            if mode in ("strict", "strict0"):
                s = 1.0 if (pok and mok) else -W
            else:
                s = (0.5 if pok else -0.25) + (0.5 if mok else -0.25)
            out.update(period_ok=float(pok), mass_ok=float(mok), correct=float(pok and mok))
    elif v == "cannot_determine":
        if mode == "strict0":
            s = 0.0
        elif abstain_aware:
            s = -B_ABSTAIN if determinable else A_ABSTAIN
        else:
            s = 0.3     # flat credit for abstaining (ablation)
    else:
        s = INVALID; out["verdict"] = "invalid"
    out["score"] = s
    out["reward"] = s - lam * spent
    return out
