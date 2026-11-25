# Claude models on the hidden-planet benchmark (Nov 15, 2026)

Hard level, strict0 reward (right +1, wrong -1, cannot 0), cost 0.008/unit, dev split test seeds 3000-3047.
Played through the same text interface as the trained model, via an isolated harness with hidden seeds
(players saw only opaque episode ids; told not to read any files). Graded against TTVFast truth.
References (teacher, EIG, compact) are the scripted policies on the SAME seeds (results/dev/).

| player (tool level) | n | score | minus teacher [95% CI] | minus EIG [95% CI] | minus compact [95% CI] |
|---|---|---|---|---|---|
| Fable (oracle) | 48 | +0.434 | -0.106 [-0.24, +0.00] | +0.014 [-0.19, +0.22] | +0.283 [+0.10, +0.47] |
| Fable (lite) | 24 | +0.428 | -0.114 [-0.23, -0.01] | -0.027 [-0.22, +0.20] | +0.168 [-0.02, +0.39] |
| Opus (oracle) | 24 | +0.353 | -0.189 [-0.41, -0.02] | -0.102 [-0.44, +0.24] | +0.093 [-0.19, +0.40] |
| Sonnet (oracle) | 24 | +0.412 | -0.130 [-0.32, -0.01] | -0.043 [-0.28, +0.24] | +0.152 [-0.09, +0.42] |
| Haiku (oracle) | 24 | +0.287 | -0.255 [-0.50, -0.08] | -0.168 [-0.44, +0.10] | +0.027 [-0.18, +0.27] |

Paired bootstrap over systems (20k resamples). Single run per model; run-to-run variance not measured.
Caveats: small n; each agent plays all episodes in one context (may adapt across episodes); prompt sensitivity untested.

## Our Qwen3.5-9B models on the same 48 systems (sampled, 2 per system, live Tinker, ~$0.37)
| player | score | minus teacher | minus EIG | minus compact |
|---|---|---|---|---|
| our RL model (confirm_strict0 step 10) | +0.160 | -0.380 [-0.57, -0.20] | -0.260 [-0.47, -0.04] | +0.009 [-0.17, +0.19] |
| our SFT model (confirm_strict0 init) | +0.078 | -0.462 [-0.64, -0.29] | -0.342 [-0.55, -0.13] | -0.073 [-0.28, +0.13] |
Our RL minus Fable: -0.274 [-0.47, -0.08] (significant). Our RL minus our SFT: +0.082 [-0.12, +0.29] (n.s.).
Note: an earlier 12-system snapshot suggested our RL model "ties" Fable; with 48 paired systems that does not hold.
| stock Qwen3.5-9B (no training; 1 sample) | -0.127 (correct 0.56, invalid 0.23, spent 37.9) | | | -0.278 [-0.56, -0.00] vs compact |

## Raw mode (no fit tools; players wrote their own Python), seeds 3000-3011, turn limit 20
| player | n | score | correct | spent | minus teacher | minus EIG | minus Fable-oracle |
|---|---|---|---|---|---|---|---|
| Fable raw+Python | 12 | +0.279 | 0.75 | 38.0 | -0.185 [-0.55,+0.04] | -0.091 [-0.43,+0.34] | +0.101 [-0.01,+0.28] |
| Opus raw+Python | 12 | +0.321 | 0.75 | 43.2 | -0.143 [-0.31,-0.02] | -0.050 [-0.37,+0.33] | +0.142 [-0.05,+0.41] |
