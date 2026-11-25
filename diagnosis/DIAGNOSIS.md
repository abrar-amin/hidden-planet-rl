# Why RL has not improved strategy beyond imitation (Nov 16)

Everything here is offline: logs, code, and local simulation with the real environment code. No API calls, no key read, nothing tracked was modified. Scripts and raw outputs are in this directory. "Measured" means computed from logs or from the real environment; "simulated" means the tabular stand-in described in section 3; "inferred" is marked as such.

## Verdict

**"Too few steps" is true but is not the main problem, and more steps alone will not rescue the planned run.** Three things stack:

1. **The experience budget is 4 to 40 times below published practice** (400 episodes on Nov 15; about 2,900 in the planned run; comparable multi-turn runs use 10k to 60k). At this scale even an ideal learner gets a small fraction of the available gain.
2. **The available gain is smaller than the plan assumes and smaller than the evaluation can resolve.** The teacher minus demonstrator gap is +0.09 pooled over 836 systems (not +0.18), and the plan's minimal detectable effect is 0.13.
3. **The reward is dominated by outcome luck that the group baseline cannot remove**, because all 8 rollouts of a group share the same hidden truth. The effective sample size for "commit or abstain" and "buy more or stop" is the number of systems (12 per step), not the number of episodes (96).

Exploration, the advantage computation, truncation and the reward function are not the problem. The live LLM also drifted in a harmful direction (low-confidence commits) that the reward was pushing against, which the ideal learner does not do; that part is an LLM/optimisation effect that more steps at 40 episodes per step did not fix in nine tries.

**The planned run as tagged: probability of the pre-registered success is under about 5 % (my estimate).** The most likely reading is "nothing resolved".

## 1. What the Nov 15 run shows (measured, `a1_trends.py`, `a2_signal.py`, `a3_lowconf.py`)

Run: `runs/confirm_strict0/rl`, 10 steps, 5 systems x 8 rollouts, LR 4.7e-5, one Adam step per batch.

| Quantity | Value |
|---|---|
| Train reward, steps 1-9 | 0.47, 0.34, 0.51, 0.21, 0.19, 0.32, 0.08, -0.17, 0.56: no trend |
| Expected step-to-step noise of that curve | SE 0.27 per step (between-system variance 0.31 over 5 systems, within 0.4 over 40 episodes) |
| Sampled negative log-prob per action token | 0.26 at step 0, 0.36 to 0.51 afterwards: no entropy collapse at temperature 1 |
| Within-group reward variance (unbiased) | 0.43 all steps; 0.38 steps 1-9 |
| Share of it that is outcome luck | 43 % all steps, 51 % steps 1-9, 59 % steps 5-9 (main session said 39 %; same order) |
| Within-group variance from the cost term | 0.0077, which is 1.7 % of the score variance |
| No-answer episodes | 8 % in steps 0-4 (13 of the 16 are step 0), 0 % in steps 5-9 |
| Groups with zero reward variance | 0 of 50 |
| Truncated turns (512-token cap) | 1 of about 1,400 |
| Distinct purchase patterns per group of 8 | 5.6 to 7.6; no group where all 8 bought the same |
| Distinct opening requests per group | 8.0 of 8 |

Findings:

- **Exploration is not the bottleneck (measured).** Rollouts in a group differ in what they buy, not only in wording. This is for a clone of the randomised fit-teacher; the fixed-design clone for the planned run has never been run live, so its exploration is unknown.
- **The policy moves fast.** One update removed the loop (no-answer 33 % to 3 %). Earlier runs with other reward modes cut spending from 29 to 10 units in 10 steps (`runs/pilot`). Step size is not too small.
- **After step 0 the policy drifted against the reward.** Low-confidence commits went from 0 to 10-30 % of episodes, although the within-group slope of reward on a low-confidence commit is -0.70 ± 0.13 (pooled over 45 groups, t = -5.3). Per step, with 5 groups, that signal is about 1.7 sigma, so it was not reliably followed. Inferred cause: step 0's dominant signal was "end the episode" (slope on no-answer -1.7), applied in one large update, and it generalised to "commit".
- **Luck is shared within a group.** Low-confidence commits cluster in a few groups (6 of 8 rollouts in two groups). When most of a group commits and the truth happens to agree, the commit is reinforced; the baseline does not help because the truth is the same for all 8.
- **Calibration holds in the training rollouts.** Commits at on-screen confidence 0.5-0.7 were right 71 %, 0.7-0.9 right 94 %, above 0.9 right 99 %, below 0.5 right 25 %.

Bugs checked:

- Advantage: return minus group mean, no std normalisation, same scalar on every action token of every turn. Correct as designed (`tinker_cookbook/rl/data_processing.py`).
- Reward: recomputed components match; the earlier audit re-graded 192 episodes with 0 mismatches.
- Constant-reward groups are not filtered (`remove_constant_reward_groups=false`); none occurred.
- **`optim/kl_sample_train_v1` and `_v2` are exactly 0.0 in every live run** (confirm, pilot, rehearsal, mini). The metric is dead, so there is no measurement of how far each update moves the policy. I could not determine why offline.
- LR is constant; 4.7e-5 is `get_lr(...)/10`. The planned run uses 2e-5.
- A hallucinated fit followed by a zero-cost "none" scored +1 once (step 3). This is legitimate luck, not a loophole: a blind "none" has expected score about -0.3.

## 2. Headroom is about half of what the plan states (measured, `a4_headroom.py`, `a5_teacher_gap.py`, `a5b_teacher_gap_more.py`)

The plan quotes fit-teacher minus fixed demonstrator = +0.18 on the dev set. That is one draw of the scripts' own random numbers. Re-running both scripts through the real environment:

| Systems | n | Policy seeds | Teacher | Fixed | Gap (95 % CI) |
|---|---|---|---|---|---|
| Dev, test 3000-3191 | 192 | 3 | +0.460 | +0.314 | +0.146 ± 0.055 (single seeds: +0.180, +0.101, +0.158) |
| Test 2000-2095 and 4000-4047 | 144 | 2 | +0.318 | +0.236 | +0.083 ± 0.070 |
| Train 500000-500259 | 260 | 3 | +0.302 | +0.265 | +0.037 ± 0.038 |
| Train 500260-500499 | 240 | 2 | +0.353 | +0.247 | +0.107 ± 0.053 |
| **Pooled** | **836** | | | | **about +0.09** (roughly +0.04 to +0.14) |

The sealed set was not touched.

Where the dev gap comes from, by what the agent sees at the first fit:

| First fit | Share of systems | Gap per system | Contribution | Per-system SD |
|---|---|---|---|---|
| p_none ≥ 0.9: stop now | 16 % | +0.176, all cost saving | +0.028 | 0.05 |
| 0.3 ≤ p_none < 0.9: more transits | 24 % | +0.095 | +0.023 | 0.70 |
| p_none < 0.3, p_both < 0.85: adaptive RVs | 60 % | +0.215, score +0.235 | +0.129 | 0.69 |

- Every branch is perceivable: the teacher uses only printed numbers. Reachability is not the issue.
- 72 % of the gap is "keep buying RVs in small batches until p_both_right is high". The demonstrator never makes a third purchase, so the clone must discover a behaviour it has never seen, and its payoff has per-system SD 0.7.
- The only clean signal (stop after a decisive "none") is worth +0.03 in total. The 60 % ambiguous-first-fit curriculum under-samples exactly those systems.

## 3. Simulation: the same estimator on an ideal learner (`s1_precompute.py`, `s2_pg.py`, `s3_experiments.py`, `s4_report.py`)

Setup:

- Real environment outcomes (TTVFast truth, real `fit_model`, real strict0 grader) precomputed on a purchase lattice for 772 systems: opening of 10 or 16 transits, extra transit batches of 8, RV batches of 4.
- Tabular softmax policy over what the agent sees (stage, p_no_perturber bucket, p_both_right bucket, has RVs), 5 actions: commit, abstain, 8 transits, 4 RVs, 8 RVs.
- Initialised as a soft clone of the fixed demonstrator with exploration rate eps (share of non-demonstrator actions per decision).
- Same update as the cookbook: group-mean baseline, one Adam step per batch, 60 % curriculum. Evaluation is the exact expected reward on the 192 dev systems, 24 RL seeds per cell.

This is an optimistic stand-in: about 170 parameters that are exactly the decision-relevant ones, step size chosen by sweep, no language, no forgetting. It bounds what the signal allows; it does not predict the LLM.

Checks: the deterministic demonstrator scores +0.321 in the lattice (real script +0.31). The best policy in this family is about +0.40 (big-batch, 400-step run), so in-family headroom is +0.08. The lattice is coarser than the real teacher (+0.46 on dev), so absolute gains are understated by up to half; the ratios between rows are the result.

Final dev reward minus the demonstrator's +0.321 (step size 0.2 / 0.5; eps = 0.1):

| Training | Episodes | Gain over demonstrator | Share of seeds above demonstrator + 0.05 |
|---|---|---|---|
| Nov 15 shape: 5 x 8, 10 steps | 400 | -0.019 / -0.010 | 0 % / 0 % |
| **Planned run: 12 x 8, 30 steps** | 2,880 | **+0.019 / +0.038** | 4 % / 38 % |
| Planned run, reward = posterior-expected score | 2,880 | +0.038 / +0.050 | 25 % / 54 % |
| Planned run, no curriculum | 2,880 | +0.041 / +0.043 | 17 % / 42 % |
| 24 x 4, 30 steps | 2,880 | +0.032 / +0.038 | 21 % / 29 % |
| 6 x 16, 30 steps | 2,880 | +0.020 / +0.046 | 0 % / 33 % |
| 12 x 8, 100 steps | 9,600 | +0.046 / +0.046 | 38 % / 46 % |
| 12 x 8, 300 steps | 28,800 | +0.060 / +0.050 | 83 % / 54 % |
| 48 x 8, 30 steps | 11,520 | +0.046 / +0.053 | 29 % / 58 % |
| 48 x 8, 30 steps, expected score | 11,520 | +0.061 / +0.070 | 79 % / 88 % |
| 12 x 8, 100 steps, expected score | 9,600 | +0.065 / +0.056 | 83 % / 71 % |

What it says:

- **400 episodes cannot beat the demonstrator even for an ideal learner.** The Nov 15 null result is what the signal predicts.
- **The planned budget captures a quarter to a half of the in-family headroom**, +0.02 to +0.04. Most of it is spending less (40 down to 31-36 units), not answering better.
- **Three to four times more episodes gives +0.05; ten times gives +0.05 to +0.06.** Returns flatten because the remaining gain is the noisy adaptive-RV branch.
- **Replacing realised ±1 by the posterior-expected score is worth about as much as 3 times more episodes**, and stacks with a larger batch.
- With low exploration (eps = 0.02) the planned-run gain falls to +0.009 at step size 0.2; a faithful clone of a deterministic script explores little.
- "Gain over own start" is misleading: a noisy clone (eps = 0.3) starts at +0.19 and "gains" +0.16 just by sharpening back to the demonstrator. That is why D2 and the supervised control matter, and why D1 alone would prove nothing.
- The ideal learner never develops low-confidence commits (under 2 % in all cells). The live drift is therefore not forced by the reward; it is the LLM update (inferred).

## 4. Signal-to-noise arithmetic for the planned run

- **Training curve.** SE of a step's mean reward = sqrt(0.31/12 + 0.4/96) ≈ 0.17. A real +0.1 between the first and last ten steps is 1.3 sigma. The curve will not show learning; only the fixed-set evals can.
- **Evaluation.** Paired per-system SD between two policies is about 0.5-0.6, so 192 systems give SE 0.04-0.05 and a minimal detectable effect of about 0.10-0.13 (the plan says 0.13). True headroom is about 0.09 pooled, 0.15 on dev. To detect +0.1 with 80 % power needs about 280 paired systems; +0.05 needs about 1,100.
- **Credit assignment.** Episodes are 3 turns and about 330 action tokens; the delay is short. The issue is dilution: the decision is a handful of tokens (which tool, how many), while most tokens are epoch lists that receive the same advantage. Luck-driven advantages on those tokens push toward one memorised request list, consistent with the 96-of-96 identical openings seen at greedy decoding (inferred).
- **Cost term.** 0.008 per unit is 1.7 % of within-group variance, yet it is the signal that gets learned, live and simulated, because its sign is consistent within a group. It is small in value, not unlearnable: skipping 24 units is +0.19 per episode on 16 % of systems.

## 5. Comparison with the literature

Compiled by a web-research pass on Nov 16; numbers are as read from the cited pages, and episode totals marked "computed" are settings multiplied out.

| Our setting | Typical published setting | Assessment |
|---|---|---|
| Group-mean advantage, no std normalisation | DeepSeekMath GRPO divides by group std (https://arxiv.org/html/2402.03300v3). Dr. GRPO removes std and length normalisation (https://arxiv.org/html/2503.20783v2). Tinker docs: mean-centring only (https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/) | In line |
| Importance-sampling loss, one update per batch, no clipping, no KL, no entropy bonus | Cookbook defaults: `importance_sampling`, `kl_penalty_coef=0`, `num_substeps=1` (https://raw.githubusercontent.com/thinking-machines-lab/tinker-cookbook/main/tinker_cookbook/rl/train.py). SkyRL-Agent: KL and entropy off (https://arxiv.org/html/2511.16108v1). R1: KL 0.001, clip 10 (https://arxiv.org/html/2501.12948). DAPO and StarPO-S: clip 0.2/0.28. Tinker docs suggest KL 0.01-0.1 | Defensible for 30 steps. RAGEN reports collapse late, around 100-150 iterations (https://arxiv.org/html/2504.20073v2); add KL 0.01-0.05 for longer runs |
| Group size 8 | 8 (Dr. GRPO, SkyRL, cookbook multi-turn recipes); 16 (R1, DAPO, RAGEN); 64 (DeepSeekMath) | In line |
| **12 systems x 8 = 96 episodes per step** | RAGEN 8 x 16 = 128. Cookbook twenty questions 64 x 8 = 512; search tool 512 x 8; MATH 64 x 16. SkyRL 64 x 8 = 512. R1 512. DAPO 512 x 16 (https://arxiv.org/html/2503.14476v2). ART·E 12 x 4 = 48 (secondary source: https://www.zenml.io/llmops-database/building-art-e-reinforcement-learning-for-email-search-agent-development) | **Outlier: 4 to 40 times smaller**, and only 12 distinct tasks per step against 32-512 |
| **30 steps, about 2,900 episodes, 360 systems** (Nov 15: 400 episodes) | Cookbook twenty questions: reward 6 % to 16 % after 20 steps, about 10k episodes (computed). RAGEN 13k-26k trajectories. ART·E about 32k (computed). SkyRL about 64k (computed). Cookbook MATH/GSM8K results at 180-220 steps of 1-2k episodes | **Largest gap: 3 to 40 times below comparable runs.** "Gains within about 20 steps" in cookbook recipes come from steps 5-40 times larger |
| LoRA rank 32, LR 2e-5 (was 4.7e-5) | Cookbook RL recipes 2e-5 to 1e-4; Tinker docs start at 1e-5. `get_lr` gives about 4.7e-4 for LoRA supervised learning. "LoRA Without Regret": LoRA LR about 10x full fine-tuning; rank 1 suffices for RL (https://thinkingmachines.ai/blog/lora/) | In line, low side of the range. Rank is ample. Do not raise LR at this batch size |
| One scalar trajectory advantage on all turns | Same in StarPO-GRPO, SkyRL-Agent, ART, cookbook. Per-turn signals do better: ArCHer's utterance-level critic (https://arxiv.org/html/2402.19446); ΔBelief-RL's per-turn belief-gain reward, +8-9 points over outcome-only on twenty questions (https://arxiv.org/html/2602.12342) | Standard, but the published sample-efficiency gains on information-gathering tasks come from per-turn signals |
| Realised ±1 reward, half of its variance luck | Noise-corrected GRPO: reward flips shrink the gradient; debias (https://arxiv.org/html/2510.18924v3). SkyRL uses a leave-one-out baseline. DAPO resamples all-same groups; StarPO-S keeps the top 25 % of prompts by reward variance | Under-mitigated. Our noise is shared within the group, so baseline changes do not help; an expected-outcome reward does (standard control-variate argument, no specific paper verified) |
| SFT warm start, then RL | Standard (R1 pipeline, ΔBelief-RL). "SFT memorizes, RL generalizes": SFT first to stabilise format, RL for generalisation (https://arxiv.org/abs/2501.17161) | In line. No dedicated source found on warm-start causing exploration collapse beyond RAGEN's "echo trap" |
| RL with sparse reward as the only learning signal | On-policy distillation: RL teaches a fixed number of bits per episode, distillation O(N); 7-10x fewer steps (https://www.thinkingmachines.ai/blog/on-policy-distillation/). `scripts/dagger.py` already implements teacher relabelling | A dense alternative exists if the goal is a good agent rather than the claim "RL beats imitation" |

**Literature verdict.** The algorithm and hyperparameters are conventional; they are essentially the cookbook defaults and match Dr. GRPO and SkyRL-Agent. The scale is the outlier: batch and total experience are 4 to 40 times below what published multi-turn runs needed before gains appeared, on cleaner rewards than ours.

Not verified: DAPO total steps; RAGEN learning rate; ART·E primary source; the Tinker "scale LR with sqrt(batch)" note (search snippet only); verl multi-turn defaults.

## 6. Ranked causes

| # | Cause | Evidence | Confidence |
|---|---|---|---|
| 1 | Too little experience per step and in total for this noise level | Simulation: 400 episodes gives nothing, 2,880 gives +0.02-0.04; literature 10k-60k; per-step SE 0.17 | High |
| 2 | Headroom (about +0.09) is below the evaluation's resolution (0.10-0.13) | Four system sets, 836 systems; paired SD 0.5-0.6 | High |
| 3 | Outcome luck is shared within a group, so the baseline does not remove it | Luck share 43-59 %; low-confidence commits cluster by group; expected-score reward is worth 3x episodes in simulation | High (measured share), medium (size of benefit for the LLM) |
| 4 | The valuable behaviour (repeated small RV purchases) is absent from the demonstrator and has SD 0.7 per system | Gap decomposition; simulation gains are mostly cost saving | Medium-high |
| 5 | LLM-specific drift: one large early update generalised to "commit"; same advantage on all tokens | Low-confidence commits rose against a -0.70 slope; absent in the ideal learner | Medium (inferred mechanism) |
| 6 | Curriculum under-samples the one clean signal | Simulation: no-curriculum is equal or better at 30 steps | Low-medium (simulation only) |
| Not a cause | Exploration diversity, advantage or reward bugs, truncation, constant-reward groups, LR too small | Section 1 | High |

## 7. Fixes

Any change to the plan must be made, re-tagged and recorded as an amendment **before** the run. None of these needs a live call to implement.

Cheap, before the run:

1. **Add a posterior-expected-score metric to the verdict** (expected reward = committed x (2 x post_conf - 1) - 0.008 x spent). `post_conf` and `committed` are already logged per episode by the grader, so no environment change is needed. In the lattice this cuts the paired SE from 0.036 to 0.016 on the dev set (minimal detectable effect 0.10 to 0.046), equivalent to about 5 times more systems. Pre-register it as co-primary or as the named secondary. Caveat: it trusts the posterior's calibration (measured on 18,560 lattice nodes: 0.60 predicted, 0.66 right; 0.815, 0.833; 0.967, 0.968), and on the dev set it rates the demonstrator at +0.265 against a realised +0.321.
2. **Train on the expected score instead of realised ±1** (same formula, in `text_env._grade`, for training only). Same optimum under calibration, removes the shared-truth luck. Simulated value: about 3 times more episodes. This is an environment change, so it needs its own tag.
3. **More systems per step.** At fixed cost, 24 x 4 is slightly better than 12 x 8 in simulation and 6 x 16 is worse. If budget allows, drop an exploratory arm and run 32-48 systems per step.
4. **Lower or drop the 60 % curriculum** (to 0.3 or 0).
5. **State the measured headroom in the plan** (+0.09 pooled, +0.146 on dev averaged over script seeds) and average the scripted references over several script seeds, so D2 is not judged against one lucky or unlucky draw.
6. **Fix or drop the KL metric**, and keep the low-confidence-commit guard; it targets the drift seen on Nov 15.

Larger, after the run:

7. **Budget 10k+ episodes** for a real test (100 steps x 96 or 30 x 384), with KL 0.01-0.05 to the init.
8. **Per-turn shaping**: a small belief-gain term (the posterior is already computed), in the style of ΔBelief-RL with weight about 0.1.
9. **Raise the headroom**: a higher observation cost (for example 0.02) makes stopping decisions worth more against the ±1 noise; or start from a weaker demonstrator (arm A has +0.35 to the teacher, with the narrower claim the plan already states).
10. **If the goal is a good agent**, on-policy distillation or DAgger from the fit-teacher is the efficient route; it does not answer "does RL beat imitation".

## 8. Can the planned run detect improvement?

- Success needs D1, D2 and D3 all above zero in both seeds. D2 needs the RL policy to beat the script by roughly 0.10-0.13 on dev.
- An ideal learner at the planned budget gains +0.02 to +0.04 over the demonstrator in the lattice (the lattice understates absolute gains by up to half). The time cap will probably end arms at step 20-25.
- The full teacher is only +0.146 above the demonstrator on dev, +0.09 pooled.
- So a real effect, if any, will most likely be +0.02 to +0.06 and read as "not resolved". With fix 1 alone the same effect becomes detectable at about 0.05; with fixes 1-4 the odds of seeing a real D2 effect rise from under 5 % to perhaps 30-50 % (my judgement from the simulation, not a measurement).

## Limits of this diagnosis

- The tabular learner is not the LLM. It bounds the signal; it says nothing about LoRA optimisation, token dilution or forgetting.
- The lattice is coarser than the real action space and its headroom (+0.08) is about half the real teacher's on dev.
- The fixed-design clone has never been run live; its exploration rate is unknown, and the simulated result depends on it (eps 0.02 to 0.3 spans +0.009 to +0.03 at step size 0.2).
- The drift mechanism in section 1 is inferred from one run with one seed.
- The step-size mapping between the tabular policy and LoRA LR is unknown; I report two step sizes that bracket the best of a five-value sweep (`sim_lr.json`).
