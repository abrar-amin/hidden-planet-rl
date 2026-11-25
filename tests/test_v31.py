"""Verdict and orchestration rules added after the audit in audit_v3/AUDIT_V3.md: the explicit
both-seeds rule, `final` waits for finished arms and ignores abandoned TAG arms, futility-stopped arms are never resumed,
caps are cumulative, and dry runs never generate sealed systems. Offline only; no sealed seed is generated."""
import json, os, shutil, subprocess, sys, time
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
REFS = os.path.join(ROOT, "results", "dev192_avg3")
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _copy(src_agent, dst):
    for kind in ("episodes", "traces"):
        shutil.copy(os.path.join(REFS, f"refs_dev_{src_agent}_{kind}.jsonl"), dst.format(kind=kind))


def _fixture(tmp_path, arms_as_teacher=("rl_fixed_s0",)):
    E = tmp_path / "ev"; E.mkdir()
    for f in os.listdir(REFS):
        if f.endswith("_episodes.jsonl"):
            shutil.copy(os.path.join(REFS, f), E / f)
    _copy("fixed", str(E / "final_dev_sampled_sft_fixed_{kind}.jsonl"))
    _copy("fixed", str(E / "final_dev_sampled_control_fixed_{kind}.jsonl"))
    for arm in arms_as_teacher:          # a clearly better policy: the fit-teacher script
        _copy("teacher", str(E / ("final_dev_sampled_" + arm + "_{kind}.jsonl")))
    return E


def _verdict(E, arms, extra=()):
    r = subprocess.run([PY, os.path.join(ROOT, "scripts", "final_verdict.py"), str(E), "--set", "dev", "--dec", "sampled", "--init", "sft_fixed",
                        "--demo", "fixed_diverse", "--control", "control_fixed", "--arms", *arms, "--n-expected", "192", "--boot", "2000", *extra],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return json.load(open(E / "final_verdict_dev_sampled_sft_fixed.json")), r.stdout


def test_two_good_arms_succeed(tmp_path):
    E = _fixture(tmp_path, ("rl_fixed_s0", "rl_fixed_s1"))
    v, out = _verdict(E, ["rl_fixed_s0", "rl_fixed_s1"], ["--expect-arms", "2"])
    assert v["experiment_success"] and v["best_arm"] in ("rl_fixed_s0", "rl_fixed_s1"), out
    assert v["pooled"]["ok"]


def test_one_arm_listed_fails_with_expect_arms(tmp_path):
    E = _fixture(tmp_path)
    v, out = _verdict(E, ["rl_fixed_s0"], ["--expect-arms", "2"])
    assert v["arms"]["rl_fixed_s0"]["success"]          # the arm itself is fine ...
    assert not v["experiment_success"] and v["best_arm"] is None, out    # ... but the both-seeds rule is not met


def test_missing_second_arm_fails(tmp_path):
    E = _fixture(tmp_path)
    v, out = _verdict(E, ["rl_fixed_s0", "rl_fixed_s1"], ["--expect-arms", "2"])
    assert v["arms"]["rl_fixed_s1"].get("missing") and not v["experiment_success"] and v["best_arm"] is None


def test_futility_stopped_arm_fails(tmp_path):
    E = _fixture(tmp_path, ("rl_fixed_s0", "rl_fixed_s1"))
    (E / "rl_fixed_s1" / "rl").mkdir(parents=True)
    (E / "rl_fixed_s1" / "rl" / "futility_stop.txt").write_text("test stop\n")
    v, out = _verdict(E, ["rl_fixed_s0", "rl_fixed_s1"], ["--expect-arms", "2"])
    assert v["arms"]["rl_fixed_s1"]["futility_stop"] and not v["experiment_success"] and v["best_arm"] is None


def test_pooled_rule_blocks_es_only_gain(tmp_path):
    # EIG has a higher ES than the fixed demonstrator but its realised gain over the fit-teacher-like control is small;
    # use EIG as the arm and the fit-teacher as control so D3's ES is positive while its realised value is not.
    E = _fixture(tmp_path, ())
    _copy("teacher", str(E / "final_dev_sampled_control_fixed_{kind}.jsonl"))
    for arm in ("rl_fixed_s0", "rl_fixed_s1"):
        _copy("eig", str(E / ("final_dev_sampled_" + arm + "_{kind}.jsonl")))
    v, out = _verdict(E, ["rl_fixed_s0", "rl_fixed_s1"], ["--expect-arms", "2"])
    p = v["pooled"]
    assert p["realised_d3"] < 0.5 * p["es_d3"] and not p["ok"] and not v["experiment_success"], out


def _event(E, *args, env_extra=None):
    env = dict(os.environ, DRY="1", QUICK="1", EVENT_DIR=str(E)); env.pop("TINKER_API_KEY", None); env.update(env_extra or {})
    return subprocess.run(["bash", os.path.join(ROOT, "scripts", "event.sh"), *args], capture_output=True, text=True, cwd=ROOT, env=env, timeout=600)


def test_final_refuses_until_arms_finished(tmp_path):
    E = tmp_path / "ev"; E.mkdir()
    r = _event(E, "final")
    assert r.returncode == 2 and "not finished yet" in r.stdout


def test_final_uses_explicit_list_and_ignores_abandoned_tag_arms(tmp_path):
    E = _fixture(tmp_path, ())
    for arm in ("rl_fixed_s0", "rl_fixed_s1"):                 # abandoned first attempt, with checkpoints that a glob would pick up
        (E / arm / "rl").mkdir(parents=True); (E / arm / "rl" / "checkpoints.jsonl").write_text('{"batch": 5, "sampler_path": "x"}\n')
        (E / arm / "finished.txt").write_text("exit 0\n")
    for arm in ("rl_fixed_s0b", "rl_fixed_s1b"):               # the TAG=b restart; both stopped for futility (so no eval is needed)
        (E / arm / "rl").mkdir(parents=True); (E / arm / "rl" / "futility_stop.txt").write_text("test\n")
    (E / "control_fixed").mkdir(); (E / "control_fixed" / "sft.json").write_text("{}")
    r = _event(E, "final", env_extra={"TAG": "b"})
    v = json.load(open(E / "final_verdict_dev_sampled_sft_fixed.json"))
    assert set(v["arms"]) == {"rl_fixed_s0b", "rl_fixed_s1b"}, r.stdout + r.stderr
    assert not v["experiment_success"] and v["best_arm"] is None


def test_rl_refuses_to_resume_futility_stopped_arm(tmp_path):
    E = tmp_path / "ev"; (E / "sft_fixed").mkdir(parents=True)
    (E / "sft_fixed" / "sft.json").write_text("{}"); (E / "gate_sft_fixed.txt").write_text("PASS\n")
    (E / "rl_fixed_s0" / "rl").mkdir(parents=True); (E / "rl_fixed_s0" / "rl" / "futility_stop.txt").write_text("stopped\n")
    r = _event(E, "rl", "fixed", "0", env_extra={"FORCE": "1"})
    assert r.returncode == 2 and "futility" in r.stdout and not (E / "rl_fixed_s0" / "finished.txt").exists()


def test_spend_counts_rows_dropped_on_resume_and_clock_is_cumulative(tmp_path):
    import train_rl as T
    row = lambda s: json.dumps({"step": s, "tokens/prefill_uncached": 1e6, "tokens/prefill_cached": 0, "tokens/sampled": 0, "tokens/trained": 0}) + "\n"
    d = tmp_path / "rl"; d.mkdir()
    (d / "metrics.jsonl").write_text("".join(row(s) for s in range(6)))
    full = T.spend_estimate(str(d), "Qwen/Qwen3.5-9B")
    # resume from checkpoint 3: the cookbook copies the file and drops steps >= 3, then re-runs 3 and 4
    shutil.copy(d / "metrics.jsonl", d / "metrics.jsonl.pre_resume_3")
    (d / "metrics.jsonl").write_text("".join(row(s) for s in range(5)))
    after = T.spend_estimate(str(d), "Qwen/Qwen3.5-9B")
    assert after == pytest.approx(full * 8 / 6)          # 6 original steps + 2 re-run steps (3 and 4 again; 5 was dropped)
    t = T.first_start(str(d)); time.sleep(0.01)
    assert T.first_start(str(d)) == t                    # kept across processes
    class A: pass
    a = A(); a.log = str(d); a.max_dollars = after * 0.5; a.max_minutes = 40
    assert "spend" in T.over_cap_at_start(a, "Qwen/Qwen3.5-9B")
    a.max_dollars = 1e9; json.dump(dict(first_start=time.time() - 41 * 60), open(d / "cap_clock.json", "w"))
    assert "min" in T.over_cap_at_start(a, "Qwen/Qwen3.5-9B")


def test_dry_run_never_generates_sealed_seeds(tmp_path):
    for args in (["--eval-set", "sealed", "--sealed-arm", "x"], ["--start", "20000", "--n", "4"]):
        r = subprocess.run([PY, os.path.join(ROOT, "scripts", "eval_zero_shot.py"), "--dry-run", *args, "--out", str(tmp_path / "x.json")],
                           capture_output=True, text=True, cwd=ROOT, timeout=300)
        assert r.returncode != 0 and "sealed" in (r.stdout + r.stderr)
    assert not os.path.exists(os.path.join(ROOT, "results", "SEALED_LOCK.json"))
