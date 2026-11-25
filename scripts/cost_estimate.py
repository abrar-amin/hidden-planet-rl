"""Budget table for an experiment: dollars and wall clock per RL step, steps that fit in $150 / 80 min.

Token counts are MEASURED from scripted episodes through the real env + Qwen3.5 tokenizer
(results/text_compact_<level>.json). Prices: tinker-docs models page, fetched 2026-11-15.
Latency inputs are from one small live test (8 parallel samples x 256 tokens in 3.5 s, i.e. ~590 tok/s
aggregate at concurrency 8) and are otherwise ASSUMED, hence two scenarios.
usage: cost_estimate.py [--level easy] [--reason-tokens 60] [--batch 16 --group 8]
"""
import json, argparse

PRICES = {  # USD per 1M tokens: prefill, cached prefill, sample, train
    "Qwen/Qwen3.5-9B": (0.30, 0.06, 0.807, 1.463),
    "Nemotron-3.5-Lightning-30B-A3B (50% promo)": (0.184, 0.0368, 0.495, 0.44),
    "Nemotron-3.5-Lightning-30B-A3B (list)": (0.368, 0.0736, 0.99, 0.88),
    "Qwen/Qwen3.6-35B-A3B": (0.45, 0.09, 1.335, 1.177),
    "Qwen/Qwen3.5-4B": (0.33, 0.066, 1.005, 0.737),
}

ap = argparse.ArgumentParser()
ap.add_argument("--level", default="easy"); ap.add_argument("--reason-tokens", type=float, default=60, help="extra sampled tokens per turn an LLM writes before the tool call")
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--group", type=int, default=8)
ap.add_argument("--eval-n", type=int, default=32); ap.add_argument("--eval-every", type=int, default=5)
ap.add_argument("--budget", type=float, default=150.0); ap.add_argument("--minutes", type=float, default=80.0)
a = ap.parse_args()
d = json.load(open(f"results/text_compact_{a.level}.json"))["summary"]
T = d["turns"]
sampled = d["sampled_tokens"] + a.reason_tokens * T
extra = a.reason_tokens * T * (T - 1) / 2
prefill = d["prefill_tokens"] + extra
cached = d["cached_prefill_tokens"] + extra
train = (d["final_context_tokens"] + a.reason_tokens * T) * 1.05
print(f"measured ({a.level}, scripted expert, n={d['n']}): {T:.1f} turns, prefill {d['prefill_tokens']:.0f} tok (of which {d['cached_prefill_tokens']:.0f} is a repeat of the previous turn), "
      f"sampled {d['sampled_tokens']:.0f}, final context {d['final_context_tokens']:.0f}")
print(f"assumed LLM episode (+{a.reason_tokens:.0f} tok/turn of prose): prefill {prefill:.0f}, sampled {sampled:.0f}, trained {train:.0f}\n")
eps = a.batch * a.group
print(f"{'model':46s} {'$/episode':>10s} {'(cache hits)':>12s} {'$/step':>8s} {'steps@$%d' % a.budget:>10s}   (step = {a.batch} systems x {a.group} rollouts = {eps} episodes, + {a.eval_n}-episode eval every {a.eval_every} steps)")
for m, (pp, pc, ps, pt) in PRICES.items():
    c_un = (prefill * pp + sampled * ps + train * pt) / 1e6
    c_ca = ((prefill - cached) * pp + cached * pc + sampled * ps + train * pt) / 1e6
    step = c_un * eps + (prefill * pp + sampled * ps) / 1e6 * a.eval_n / a.eval_every
    print(f"{m:46s} {c_un:10.5f} {c_ca:12.5f} {step:8.2f} {a.budget / step:10.0f}")
print("\nWall clock per step (UNMEASURED beyond the tiny live test; treat as a range):")
for name, tok_s, turn_floor, train_s, save_s in (("optimistic", 5000.0, 2.0, 15.0, 5.0), ("conservative", 590.0, 6.0, 90.0, 30.0)):
    sample_s = max(T * turn_floor, eps * sampled / tok_s)
    step_s = sample_s + train_s + save_s
    eval_s = max(T * turn_floor, a.eval_n * sampled / tok_s) / a.eval_every
    n = a.minutes * 60 / (step_s + eval_s)
    print(f"  {name:12s}: service throughput {tok_s:.0f} tok/s, >= {turn_floor:.0f} s/turn, train {train_s:.0f} s, save+new sampler {save_s:.0f} s "
          f"-> {step_s + eval_s:.0f} s/step -> {n:.0f} steps in {a.minutes:.0f} min ({n * eps:.0f} training episodes)")
