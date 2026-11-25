import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(all="ignore")
from hidden_planet.tasks import sample_system, Episode, BUDGET, NMAX
from hidden_planet.reward import score, LAMBDA
from hidden_planet.text_env import HiddenPlanetSession, TOOL_SPECS
from hidden_planet import policies as pol
from hidden_planet.inference import Posterior, Prior


def test_deterministic_seeding_and_splits():
    a, b = sample_system(5, 2, "train"), sample_system(5, 2, "train")
    assert a.to_dict() == b.to_dict()
    assert sample_system(5, 2, "test").P_cat != a.P_cat
    e1, e2 = Episode(a), Episode(b)
    assert e1.observe_transit(10) == e2.observe_transit(10)
    assert e1.observe_rv(33.3) == e2.observe_rv(33.3)


def test_budget_and_validation():
    ep = Episode(sample_system(1, 2))
    assert ep.observe_transit(NMAX + 1)[0] is None and ep.spent == 0
    ep.observe_transit(3)
    assert ep.observe_transit(3)[0] is None and ep.spent == 1          # duplicates are free errors
    for i in range(30):
        ep.observe_rv(10.0 + i)
    assert ep.spent <= BUDGET and ep.observe_rv(5.0)[0] is None


def test_reward_bounds():
    sp = sample_system(2, 2)
    answers = [None, {}, {"verdict": "planet"}, {"verdict": "planet", "period_days": "x", "mass_earth": 1},
               {"verdict": "planet", "period_days": float("nan"), "mass_earth": 3}, {"verdict": "planet", "period_days": -1, "mass_earth": 0},
               {"verdict": "none"}, {"verdict": "cannot_determine"}, {"verdict": "planet", "period_days": sp.P2 or 10, "mass_earth": sp.m2 or 10},
               {"verdict": "PLANET"}, {"verdict": ["none"]}]
    for a in answers:
        for spent in (0, BUDGET):
            for q in (0.0, 1.0):
                r = score(sp, a, spent, q, q)["reward"]
                assert -1.0 - LAMBDA * BUDGET - 1e-9 <= r <= 1.0


def test_abstention_is_identifiability_aware():
    sp = sample_system(2, 2)
    lo = score(sp, pol.CANNOT, 0, 0.1, 0.2)["score"]; hi = score(sp, pol.CANNOT, 0, 0.1, 0.9)["score"]
    assert lo > 0 > hi
    assert score(sp, pol.CANNOT, 0, 0.9, 0.1)["score"] < 0      # determinable from the agent's own data


def test_posterior_recovers_strong_signal():
    for seed in range(200):
        sp = sample_system(seed, 5)                      # 2:1 level, N-body (TTVFast) truth, analytic inference
        if sp.has_planet and sp.m2 > 30 and sp.sig_rv < 2:
            break
    ep = Episode(sp); pol.run_design(ep, 30, 10, np.random.default_rng(0))
    post = pol.posterior_of(ep)
    assert post.p_none < 1e-3 and post.prob_truth(sp) > 0.8
    sp0 = next(s for s in (sample_system(k, 2) for k in range(50)) if not s.has_planet)
    ep = Episode(sp0); pol.run_design(ep, 60, 0, np.random.default_rng(0))
    assert pol.posterior_of(ep).p_none > 0.7


def test_text_session_roundtrip():
    from hidden_planet.tasks import LEVELS
    s = HiddenPlanetSession.from_seed(3, 2, "train", q_ref=0.5)
    out, done = s.call("observe_transits", {"epochs": [0, 10, 20, 30, 40, 50, 10, 999, "x"]})
    d = json.loads(out)
    assert len(d["epoch_and_oc_min"]) == 6 and len(d["errors"]) == 3 and d["spent"] == 6.0 and not done
    assert "peaks" in json.loads(s.call("fit_transit_timing", {})[0])
    assert "error" in json.loads(s.call("fit_rv", {"periods_days": [20.0]})[0])
    assert "error" in json.loads(s.call("no_such_tool", {})[0])
    assert "error" in json.loads(s.call("observe_rvs", {"wrong": 1})[0])
    r0, _ = s.grade()                                   # no answer yet -> invalid
    assert math.isclose(r0, -1.0 - LAMBDA * 6)
    out, done = s.call("submit_answer", {"verdict": "none"})
    assert done and s.call("observe_transits", {"epochs": [60]})[1] is True and s.ep.spent == 6
    r, m = s.grade()
    assert -1.3 <= r <= 1 and m["verdict_none"] == 1.0
    assert {t["name"] for t in TOOL_SPECS} == {"observe_transits", "observe_rvs", "fit_transit_timing", "fit_rv", "fit_model", "get_data_summary", "submit_answer"}


def test_fit_model_and_replay():
    import json as _json
    s = HiddenPlanetSession.from_seed(7, 5, "test")
    assert "error" in _json.loads(s.call("fit_model", {})[0])
    s.call("observe_transits", {"epochs": list(range(0, 150, 10))})
    d = _json.loads(s.call("fit_model", {})[0])
    assert 0 <= d["p_no_perturber"] <= 1 and d["best_perturber"]["period_days"] > 0
    from hidden_planet.replay import ReplaySession, TARGETS
    for koi in TARGETS:
        r = ReplaySession(koi, 0)
        out = _json.loads(r.call("observe_transits", {"epochs": list(range(0, 151, 5))})[0])
        assert len(out["epoch_and_oc_min"]) >= 10 and r.ep.spent == len(out["epoch_and_oc_min"])
        assert "errors" in _json.loads(r.call("observe_rvs", {"times_days": [5.0]})[0]) and r.ep.spent == len(out["epoch_and_oc_min"])
        r.call("submit_answer", {"verdict": "none"})
        rew, m = r.grade()
        assert -1.3 <= rew <= 1 and m["verdict_none"] == 1.0


def test_duplicate_request_lists_alternatives():
    """A refused duplicate must come back with concrete unobserved alternatives (a greedy policy loops on bare refusals)."""
    import json as _json
    s = HiddenPlanetSession.from_seed(11, 5, "test")
    first = list(range(0, 150, 12))
    s.call("observe_transits", {"epochs": first})
    d = _json.loads(s.call("observe_transits", {"epochs": first})[0])
    alt = d["unobserved_epochs_you_could_request"]
    assert d["epoch_and_oc_min"] == [] and alt and not set(alt) & set(first) and "Do not repeat" in d["hint"]
    d2 = _json.loads(s.call("observe_transits", {"epochs": alt})[0])
    assert len(d2["epoch_and_oc_min"]) == len(alt) and "errors" not in d2
    times = [100.5, 300.25, 700.0]
    s.call("observe_rvs", {"times_days": times})
    r = _json.loads(s.call("observe_rvs", {"times_days": times})[0])
    new = r["new_times_you_could_request"]
    assert r["time_and_rv_ms"] == [] and new and all(min(abs(t - x) for x in times) > 1.0 for t in new)
    assert len(_json.loads(s.call("observe_rvs", {"times_days": new})[0])["time_and_rv_ms"]) == len(new)
    # the stateless teacher takes the alternatives after a refusal
    from hidden_planet.text_policies import teacher_decide
    turns = [([("observe_transits", {"epochs": first}), ("fit_model", {})], [{"epoch_and_oc_min": [[e, 0.0] for e in first], "spent": 13.0}, {}]),
             ([("observe_transits", {"epochs": first})], [d])]
    th, calls = teacher_decide(s.user_prompt(), turns, np.random.default_rng(0))
    assert calls[0] == ("observe_transits", {"epochs": alt})


def test_seed_hidden_and_noise_keyed_by_split_and_time():
    a, b = sample_system(123, 5, "train"), sample_system(123, 5, "test")
    sa = HiddenPlanetSession(a); sb = HiddenPlanetSession(b)
    assert "0000123" not in sa.user_prompt() and "123." not in sa.user_prompt().split("Known")[0] and sa.user_prompt().startswith("Target T-")
    assert sa.user_prompt().split(".")[0] != sb.user_prompt().split(".")[0]          # different anonymous ids per split
    ea, eb = Episode(a), Episode(b)
    assert not np.allclose(ea._tnoise / a.sig_t, eb._tnoise / b.sig_t)                # train seed k and test seed k draw different noise
    e1, e2 = Episode(sample_system(7, 5, "test")), Episode(sample_system(7, 5, "test"))
    for t in (10.0, 55.5, 200.25):
        e1.observe_rv(t)
    for t in (200.25, 10.0, 55.5):                                                    # different purchase order, same values
        e2.observe_rv(t)
    assert dict(e1.rvs) == dict(e2.rvs)


def test_replay_builder_and_ooc_scorer_follow_reward_mode():
    import asyncio
    from tinker_cookbook import tokenizer_utils
    from tinker_cookbook.renderers import get_renderer
    from hidden_planet.tinker_env import ReplayEnvGroupBuilder
    from hidden_planet.ooc import sample_ooc_system, score_ooc
    envs = asyncio.run(ReplayEnvGroupBuilder("1.01", 0, "Qwen/Qwen3.5-9B", "qwen3_5_disable_thinking", False, 0.008, "strict0", 1.0).make_envs())
    assert len(envs) == 1 and envs[0].session.reward_mode == "strict0"
    sp = sample_ooc_system(3, "drift")
    assert score_ooc(sp, {"verdict": "cannot_determine"}, 10, lam=0.008, mode="strict0")["score"] == 0.0
    assert score_ooc(sp, {"verdict": "other"}, 0, mode="strict0")["score"] == 1.0
    assert score_ooc(sp, {"verdict": "none"}, 0, mode="strict0")["score"] == -1.0


def test_sealed_lock_and_ranges(tmp_path, monkeypatch):
    from hidden_planet import sealed
    monkeypatch.setattr(sealed, "LOCK", str(tmp_path / "lock.json"))
    assert sealed.SEALED_START == 20000 and sealed.overlaps(20050, 10) and not sealed.overlaps(3000, 192) and not sealed.overlaps(4000, 48)
    import pytest
    with pytest.raises(SystemExit):
        sealed.claim(None)
    assert sealed.claim("rl_fixed_s0")["status"] == "in progress"
    assert sealed.claim("rl_fixed_s0")["arm"] == "rl_fixed_s0"            # the same arm may resume
    with pytest.raises(SystemExit):
        sealed.claim("rl_fixed_s1")                                       # any other arm is refused
    sealed.finish("rl_fixed_s0")
    import json as _json
    assert _json.load(open(sealed.LOCK))["status"] == "complete"


def test_posterior_guard_metric_exists_without_fit_model():
    s = HiddenPlanetSession.from_seed(11, 5, "test", reward_mode="strict0")
    s.call("observe_transits", {"epochs": list(range(0, 150, 12))})
    s.call("submit_answer", {"verdict": "planet", "period_days": 12.345, "mass_earth": 9.0})     # never called fit_model, wild guess
    _, m = s.grade()
    assert m["used_fit_model"] == 0.0 and m["post_lowconf_commit"] == 1.0 and m["lowconf_commit"] == 1.0
