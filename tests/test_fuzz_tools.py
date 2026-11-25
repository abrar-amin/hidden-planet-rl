"""No tool call, however malformed, may raise: errors must come back as tool messages."""
import json, os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.seterr(all="ignore")
from hidden_planet.text_env import HiddenPlanetSession, TOOL_SPECS

JUNK = [None, True, 0, -1, 3, 150, 151, 1e9, -1e9, 0.5, float("nan"), float("inf"), "", "x", "12", "none", "planet", [], [[]], [None], ["a"],
        [0], [5, 5, 5], [0, 1], list(range(0, 151, 10)), list(range(200)), [1e12], [-3.2], [float("nan")], {}, {"a": 1}, [[1, 2]], [3, 3.5]]
NAMES = [t["name"] for t in TOOL_SPECS] + ["nonsense", "", "observe_transit"]
KEYS = ["epochs", "times_days", "periods_days", "verdict", "period_days", "mass_earth", "extra", "epoch"]


def test_fuzz_no_exception_escapes():
    rnd = random.Random(0)
    for seed in range(60):
        for kw in ({}, {"allow_other": True}, {"ooc_kind": "drift"}):
            s = HiddenPlanetSession.from_seed(seed, rnd.choice([0, 5]), "test", **kw)
            for _ in range(25):
                name = rnd.choice(NAMES)
                r = rnd.random()
                args = rnd.choice(JUNK) if r < 0.15 else {rnd.choice(KEYS): rnd.choice(JUNK) for _ in range(rnd.randint(0, 3))}
                if r > 0.85 and name in ("fit_transit_timing", "fit_rv", "get_data_summary"):
                    args = {"periods_days": [rnd.uniform(-5, 80) for _ in range(rnd.randint(0, 10))]} if name == "fit_rv" else {}
                out, done = s.call(name, args)
                json.loads(out)                        # always valid JSON
                if done:
                    break
            reward, metrics = s.grade()
            assert -1.31 <= reward <= 1.0 and all(isinstance(v, float) for v in metrics.values())


def test_degenerate_fits():
    for epochs in ([0, 1, 2, 3, 4, 5], [0, 30, 60, 90, 120, 150], [10, 20, 30, 40, 50, 60, 70], [0, 2, 4, 6, 8, 100], list(range(0, 150, 25)) + [1]):
        s = HiddenPlanetSession.from_seed(1, 0, "test")
        s.call("observe_transits", {"epochs": epochs})
        d = json.loads(s.call("fit_transit_timing", {})[0])
        assert "peaks" in d or "error" in d
    s = HiddenPlanetSession.from_seed(2, 0, "test")
    s.call("observe_rvs", {"times_days": [10, 10, 10, 10, 10, 10.0]})          # duplicates rejected
    assert s.ep.spent == 3
    s.call("observe_rvs", {"times_days": [20, 30, 40, 50]})
    d = json.loads(s.call("fit_rv", {"periods_days": [0, -1, 1e-9, 1e12, 14.5, "x", None]})[0])
    assert len(d["fits"]) == 7
