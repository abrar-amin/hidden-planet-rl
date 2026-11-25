"""Posterior-expected score (ES) metric, the expected training reward, and the futility rule."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from hidden_planet.expected_score import es_from_metrics
from hidden_planet.tasks import sample_system
from hidden_planet.text_env import HiddenPlanetSession
from hidden_planet.inference import Posterior, Prior

KW = dict(reward_mode="strict0", wrong_penalty=1.0, cost_per_unit=0.008)


def test_formula_cases():
    assert es_from_metrics(dict(invalid=1.0, spent=10)) == -1.0 - 0.08
    assert es_from_metrics(dict(grade_error=1.0, invalid=1.0, spent=0)) == -1.0
    assert es_from_metrics(dict(verdict_cannot=1.0, spent=20)) == -0.16
    assert abs(es_from_metrics(dict(committed=1.0, post_conf=0.9, spent=0)) - 0.8) < 1e-12
    assert es_from_metrics(dict(spent=5)) == -1.0 - 0.04            # nothing logged = no answer


def _played(seed, verdict, offset=1.0, train_reward="realised"):
    s = HiddenPlanetSession(sample_system(seed, 5, "test"), train_reward=train_reward, **KW)
    s.call("observe_transits", {"epochs": list(range(0, 150, 9))})
    f = json.loads(s.call("fit_model", {})[0])
    if verdict == "planet":
        bp = f["best_perturber"]
        s.call("submit_answer", {"verdict": "planet", "period_days": bp["period_days"] * offset, "mass_earth": bp["mass_earth"]})
    else:
        s.call("submit_answer", {"verdict": verdict})
    return s, f


def test_post_conf_is_probability_of_the_submitted_answer():
    for seed in (3000, 3001, 3002):
        for off in (1.0, 1.006):                                   # the fit's answer, and an answer just outside its tolerance box
            s, f = _played(seed, "planet", off)
            r, m = s.grade()
            post = Posterior(Prior.from_system(s.sp), *s.ep.data())
            exp = post.prob_answer(float(s.answer["period_days"]), float(s.answer["mass_earth"]))[1]
            assert abs(m["post_conf"] - exp) < 1e-12
            if off == 1.0:
                assert abs(m["post_conf"] - f["best_perturber"]["p_both_right"]) < 2e-3
            assert abs(m["es"] - (2 * exp - 1 - 0.008 * m["spent"])) < 1e-12
        s, f = _played(seed, "none"); r, m = s.grade()
        assert abs(m["post_conf"] - Posterior(Prior.from_system(s.sp), *s.ep.data()).p_none) < 1e-12


def test_expected_training_reward_returns_es_and_keeps_truth():
    for seed in (3000, 3003):
        s, _ = _played(seed, "planet", train_reward="expected"); r, m = s.grade()
        s2, _ = _played(seed, "planet"); r2, m2 = s2.grade()
        assert r == m["es"] and r2 == m2["reward_realised"] and m["reward_realised"] == r2 and m["correct"] == m2["correct"]
    s = HiddenPlanetSession(sample_system(3000, 5, "test"), train_reward="expected", **KW)
    r, m = s.grade()                                                # unanswered: same as realised
    assert r == -1.0 and m["invalid"] == 1.0


def test_bad_train_reward_rejected():
    try:
        HiddenPlanetSession(sample_system(3000, 5, "test"), train_reward="es", **KW)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_futility_rule():
    import train_rl
    assert train_rl.futility_verdict([(0, 0.2), (5, 0.3)], 15) is None
    assert train_rl.futility_verdict([(0, 0.2), (15, 0.2)], 15)["verdict"] == "STOP"           # equal is not above
    assert train_rl.futility_verdict([(0, 0.2), (15, 0.1)], 15)["verdict"] == "STOP"
    v = train_rl.futility_verdict([(0, 0.2), (10, 0.0), (15, 0.25), (20, 0.0)], 15)
    assert v["verdict"] == "CONTINUE" and v["step"] == 15
