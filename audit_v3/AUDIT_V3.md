# Independent audit of the experiment plan, version 3 (tag `prereg-tuesday-v3`, commit 71009ed)

Auditor's scope: `PREREG_TUESDAY.md` v3, `hidden_planet/{expected_score,reward,text_env,tinker_env,inference,rollout_utils,sealed}.py`, `scripts/{event.sh,run_arm.sh,train_rl.py,final_verdict.py,eval_zero_shot.py,eval_refs.py,sft_warmstart.py,dry_event.sh,preflight.py,es_loophole_test.py}`, `results/es_loophole.txt`, `results/es_lattice_check.txt`, `results/dev192_avg3/`, the diff against `prereg-tuesday-v2`.

Everything was offline. No API key was read, no live call was made, no sealed seed (20000-20095) was generated, `results/SEALED_LOCK.json` does not exist and was not created. No tracked file was changed (`git status` shows only `audit_v3/`; `scripts/preflight.py` overwrites the tracked `results/preflight.json`, which I restored byte-identically from a copy). All audit scripts and outputs are in `audit_v3/`.

## Verdict

**Run after the listed fixes.** One fix is mandatory (B1, a few lines). S1-S4 are free and should be made tonight. Any change must be committed and re-tagged (for example `prereg-tuesday-v4`) before the live run, with this file named as the reason.

The core v3 machinery is sound: ES is computed correctly, RL does train on it, I could not construct an exploit, the futility rule is implemented as written and does not bias the dev analysis, and the rehearsal, preflights and tests pass.

## What was run

| Check | Result |
|---|---|
| `bash scripts/dry_event.sh` | exit 0, no traceback, `sealed lock untouched`, RL trained on ES, both futility checks written (`audit_v3/dry_event.log`, `runs/event_dry_123237`) |
| `preflight.py --reward-mode strict0 --demo fit` / `fixed` | 7 of 7 and 4 of 4 gates pass (`audit_v3/preflight.log`) |
| `pytest tests` | 28 passed |
| Real futility STOP path in the watchdog (never reached by the rehearsal, where the mock policy always improves) | cancels training, writes both files (`audit_v3/futility_stop_path.py`) |
| Exploit search: 21 purchase designs and commit rules, 800 train systems (930000+), real env | no policy has ES above the demonstrator without a realised gain (`audit_v3/exploit_800.txt`) |
| Static designs inside the SFT data's support, 600 train systems (940000+) | `audit_v3/static_design.txt` |
| Power, futility and ES-bias numbers recomputed from `results/dev192_avg3/` | `audit_v3/power.txt`, `audit_v3/power_sim.txt`, `audit_v3/pairwise_bias.txt` |
| Verdict script fed a single arm | reports SUCCESS (`audit_v3/fake_event/`) |
| Dev references regenerated from the tagged code (fixed, teacher, always_none; 3 plays) | byte-identical to `results/dev192_avg3/` (`audit_v3/refs_check/`) |

## BLOCKER

### B1. The "both seeds" rule is not enforced; one arm can produce "PRE-REGISTERED RESULT: SUCCESS" and unlock the sealed set

- `scripts/event.sh:118-122`: `final` builds the arm list from the glob `rl_${DEMO}_s*`. An arm with no `rl/checkpoints.jsonl` (crashed before step 5) is skipped by `continue`; an arm whose dev eval fails is dropped by `evals ... && ARMS=...`. Neither is passed to the verdict.
- `scripts/final_verdict.py:115`: `experiment_success = all_ok and len(out["arms"]) == len(a.arms)` only compares against the arms it was given.
- Evidence: `final_verdict.py audit_v3/fake_event --arms rl_fixed_s0` (one arm, files built from the dev references) prints `PRE-REGISTERED RESULT (dev, sampled): SUCCESS: every RL arm exceeds ...`, `experiment_success = True`, `best_arm = rl_fixed_s0`. `event.sh sealed` would then run.
- Same glob, opposite failure: the plan's own fallback (`TAG=b` after a slow start) leaves `rl_fixed_s0` and `rl_fixed_s1` next to `rl_fixed_s0b`, `rl_fixed_s1b`. If the abandoned arms reached step 5 they are evaluated on dev (about $4.6 and 10 minutes) and must also succeed, so the result is NOT MET whatever the real arms did.

Fix: `event.sh final` passes the explicit list `rl_${DEMO}_s0${TAG} rl_${DEMO}_s1${TAG}` (missing arms included, so the verdict prints "no eval files" and fails); `final_verdict.py` gets `--expect-arms 2` and sets `experiment_success = False` and `best_arm = None` unless exactly that many arms were listed for the primary demonstrator.

## SHOULD-FIX

### S1. ES differences between policies carry a bias of the same size as the effect being tested, and the realised "sanity" condition is weak

ES is correct as coded, but its gap to the realised reward differs by policy, by amounts comparable to the stated minimal detectable effect (0.05).

- Dev set, references averaged over 3 plays (`audit_v3/pairwise_bias.txt`): EIG - fit-teacher: **ES +0.050 [+0.014, +0.084], realised -0.009 [-0.067, +0.047]**. This pair would pass "ES CI above 0" with no realised effect.
- (ES - realised) differential between two policies has a CI excluding 0 in 3 of 10 scripted pairs on dev: fixed - compact -0.056 [-0.110, -0.003], teacher - EIG -0.059 [-0.107, -0.011], teacher - compact -0.090 [-0.159, -0.025].
- Train systems: (14 transits, 9 RVs) against the demonstrator's (16, 8): ES +0.022 [+0.004, +0.040], realised +0.002 (`static_design.txt`). Transits bunched at both ends of the baseline: ES - realised +0.061 [+0.017, +0.105] (`exploit_800.txt`).
- The loophole test could not have seen this: its criterion is "CI of ES - realised entirely above +0.03" with n = 192, where the CI half-width is about 0.08.
- Consequence, simulated with the dev noise levels (`power_sim.py`): if RL moves ES by +0.05 through this bias with zero true gain, the full success rule (both seeds, realised D2 and D3 point estimates >= 0, futility) passes in about 10-14 % of runs. Under a true null with no bias it passes in under 0.1 %.

Direction of the known miscalibration: the lattice check shows "planet" commits at posterior 0.5-0.9 under-rated by 0.13-0.18; the demonstrator makes 25 % of its dev episodes in that band. A policy that buys more until the posterior is above 0.9 is credited more in ES than it gains in truth, so the bias favours the RL arm in D1, D2 and D3 alike. The three references disagree on the sign of the band's error (dev: fixed +0.19 ES against +0.25 realised, EIG +0.19 against -0.03), so this is mostly noise at the available n, but it cannot be bounded below about 0.04.

Fixes (all free):
1. Change the claim wording from "with the realised score agreeing" to "the realised point estimates are non-negative"; state that an ES effect below about 0.05 is within the measured between-policy ES bias.
2. Strengthen the co-primary condition: pooled over both seeds, the realised D2 and D3 point estimates must be at least half of the ES estimates (cuts the bias-only pass rate to about 5 %; costs some power, so decide before the run).
3. Print each arm's own ES - realised gap with a CI in `final_verdict.py`.

Training on ES itself is acceptable: over 21 designs on 800 systems the largest attractive gap was 0.03, no policy beat the demonstrator in ES without a realised gain, and the loophole file's largest gap (overbuy +0.087) did not replicate (-0.010 [-0.056, +0.034] at n = 800).

### S2. The stated minimal detectable effect (about 0.05) is for one comparison, not for the success rule

- Recomputed paired ES SE on dev, single draw against single draw: 0.017-0.022 (teacher - fixed), 0.020-0.025 (EIG - fixed). The plan quotes 0.017-0.020, the lowest of the three draws. Same policy against itself (pure sampling noise): 0.019.
- One CI above 0 at 80 % power: about 0.04-0.05. Correct as stated.
- The success rule needs six CIs, two sanity conditions per seed and two futility passes. Simulated: true effect +0.05 gives **about 52 %**; 80 % needs about +0.065; if the LLM's sampling noise is 1.5 times the scripts', +0.05 gives about 9 % and 80 % needs about +0.09.

Fix: state these numbers in the Power section.

### S3. `event.sh final` is not safe to run early or twice

- `event.sh:67`: `selected_checkpoint.json` is written once. If `final` is started while an RL terminal is still training, the selection is frozen on a partial run.
- `event.sh:44` skips an eval whose output exists, while `event.sh:128` rewrites `control_fixed/eval_sampler.json` on every call. After a second `final` with a larger `MAXSTEP`, the dev verdict uses the old control checkpoint and `sealed` (`event.sh:160`) uses the new one.

Fix: `final` exits unless every expected arm has `futility_stop.txt` or a finished `run.log` (the `=== done` line); do not rewrite `eval_sampler.json` once it exists.

### S4. Restart behaviour weakens the caps

- On resume the cookbook drops `metrics.jsonl` rows at or after the checkpoint step (`tinker_cookbook/stores/training_store.py:274`; seen in `runs/event_dry_232248/rl_fixed_s0/rl/metrics.jsonl.pre_resume_000003`). The spend of up to 4 steps (about $2.5) leaves the estimate, and the final-step eval row can be lost so that checkpoint cannot be selected.
- `train_rl.py:195`: the 40-minute clock restarts with the process.
- Re-running `event.sh rl fixed N` on an arm that was stopped for futility resumes training to step 30 (the check is skipped because `futility_check.txt` exists). The verdict still records the stop, but about $9 is spent.

Fix: `event.sh rl` exits if `futility_stop.txt` exists; store the first start time in the run directory and count minutes from it.

## MINOR

1. **D2's reference is not the script that produced the SFT data.** The data use `diverse=True` (14-18 transits, 7-9 RVs: counts 136/159/123/148/134 and 224/236/240 in `data/expert_fixed_hard_strict0.jsonl`); `eval_refs.py:43` plays the non-diverse (16, 8) version. Diverse - (16, 8): ES +0.013 [-0.006, +0.032]; the best static design in the support is +0.022. Up to about 0.02 of D2 can be a fixed batch size, not adaptive purchasing or stopping. Add a `fixed_diverse` reference and report it.
2. **Control.** Token-matched on billed tokens, but each datum trains one assistant turn: 128 datums are about 43 episodes of loss-bearing turns against RL's 96. Its learning rate is RL's 2e-5, ten times below the SFT's own 2e-4 (`sft_warmstart.py:146`, `event.sh:112`). Say so where D3 is quoted. `sft.json` is written only at the end, so a crash loses the control (the verdict then fails closed). If the $20 guard fires, the last state is stored under the key of the planned final step.
3. **Selection asymmetry.** RL's checkpoint is the best of up to six on the selection set; the control is taken at a fixed step. Expected advantage under 0.01.
4. **Pairing guard is relative.** `final_verdict.py:95` compares with the init's own system count. Failed rollouts are skipped (`rollout_utils.py`), so an init eval that lost systems passes silently; the gate has the same property. Require n >= 0.9 x 192.
5. **Futility and collapse guards fail open** if the step-0 eval row is missing (`futility_verdict` returns `None` for the whole run).
6. **Sealed seeds in dry mode.** `eval_zero_shot.py --dry-run --eval-set sealed` (or `--start 20000`) generates the sealed systems without the lock (`and not a.dry_run`). `event.sh` avoids this path (seeds 9000+), but the entry point should refuse.
7. **ES ignores the 0.05 nudge penalty** that the realised eval reward includes. Negligible, and conservative for RL in the sanity check.
8. **Preflight does not cover version 3**: G6 uses the realised reward, G7 runs `run_arm.sh` defaults (AMBIG 0.6, realised reward, no futility). The rehearsal covers them. It also overwrites the tracked `results/preflight.json`; the committed file holds the `fit` result.
9. **No timeout or dollar cap on evals**; a hung call stalls `final`. If the gate fails at 4:12 the only path is `FORCE=1`. The fixed demonstrator's clone (15 steps x 32 datums, 0.23 epoch of 2,100 datums) has never been sampled live.
10. `selected_checkpoint.json` stores the ES under the key `selection_eval_reward`.

## Checked and found sound

- **post_conf and ES.** `text_env.py:445-457` computes the posterior from the data bought; "planet" uses the submitted period and mass (`Posterior.prob_answer`, same tolerances as `reward.score`), "none" uses `p_none`; "cannot" is 0; invalid, unanswered and grader errors equal the realised reward. `final_verdict.py` recomputes ES from logged fields for every file, references included.
- **Training reward.** In the rehearsal the training reward equals the logged ES (minus nudge penalties) and differs from `reward_realised`; in-run evals log both.
- **Exploits tried:** transit-only, RV-only, clustered and end-loaded epochs, overbuying, commit at any probability, thresholds 0.7 and 0.9, ES-greedy adaptive buying. None beats the demonstrator's ES.
- **Futility.** Implemented as written (strict inequality, first eval at or after step 15, selection set only). From dev noise, SE of the 48-system difference is about 0.039: continue probability 0.50 at no effect, 0.70 at +0.02, 0.90 at +0.05 (0.81 for both arms). The dev analysis is not biased by it or by checkpoint selection, since seeds 4000-4047 and 3000-3191 are disjoint.
- **Statistics.** Per-system means, paired on seed, cluster bootstrap over systems; the conjunction is conservative for multiple comparisons.
- **Seeds.** RL train 500000+ and 10500000+, curriculum pool 100000-101999, demonstrations 0-699, loophole test 900000+: disjoint.
- **Arm C and budget.** Commands contain no arm C. Recomputed: RL arm 30 x 96 x $0.006 + 7 x 48 x $0.004 = $18.6; control 30 x 128 x 2,448 tokens = 9.4 M tokens = $13.8; caps sum to about $88.
- **Sealed lock.** Not created by the rehearsal; `claim` refuses a second arm.
