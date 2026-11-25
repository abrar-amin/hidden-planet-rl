"""Posterior-expected score (ES), the primary metric of the pre-registered experiment (rationale in diagnosis/DIAGNOSIS.md,
section 7).

For one episode, with c the observation cost per unit:
    committed answer ('planet' or 'none'):  ES = 2 * post_conf - 1 - c * spent
    'cannot_determine':                     ES = 0 - c * spent
    no valid answer (invalid / unanswered): ES = -1 - c * spent          (the same as the realised reward)

post_conf is computed by the grader (text_env._grade) for every arm and tool level, from the data the agent actually bought
and the same posterior that fit_model uses, and is never shown to the agent:
    'planet' -> posterior probability that the SUBMITTED period is within 0.5 % AND the SUBMITTED mass within a factor 1.5
                of the truth (inference.Posterior.prob_answer(P, m)[1]); the same tolerances the grader uses (reward.score).
                An answer that differs from the fit's top option gets the posterior mass of its own tolerance box.
    'none'   -> posterior probability of no perturber (Posterior.p_none).
Under strict0 (right +1, wrong -1) the expected realised score of a committed answer given the data is exactly 2 p - 1 when the
posterior is calibrated, so ES has the same expectation and the same optimum as the realised reward, without the outcome luck.
It inherits the posterior's model error: the posterior uses the analytic Lithwick-Xie-Wu model while the truth comes from TTVFast.
"""
import math

COST = 0.008


def es_from_metrics(m, cost=COST):
    """ES of one episode from its logged metrics (eval_zero_shot / eval_refs episode rows, or the grader's metrics dict)."""
    spent = float(m.get("spent", 0.0) or 0.0)
    if m.get("invalid", 0.0) > 0.5 or m.get("grade_error", 0.0) > 0.5:
        return -1.0 - cost * spent
    if m.get("verdict_cannot", 0.0) > 0.5:
        return -cost * spent
    if m.get("committed", 0.0) > 0.5:
        p = float(m.get("post_conf", 0.0))
        if not math.isfinite(p):
            p = 0.0
        return 2.0 * p - 1.0 - cost * spent
    return -1.0 - cost * spent          # anything else (e.g. no verdict logged) counts as no answer
