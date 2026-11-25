"""Convert the cookbook's metrics.jsonl into the flat CSV that plot.py reads."""
import csv, json, os

# cookbook key -> csv column. Env metrics are logged as env/all/<name>; held-out eval as test/env/all/<name>.
MAP = {"reward/total": "reward", "reward_std_cost": "reward_std_cost", "score": "score", "verdict_other": "verdict_other", "nudge": "nudge", "rv_buy_rate": "rv_buy_rate", "submit_conf": "submit_conf", "lowconf_planet": "lowconf_planet", "lowconf_commit": "lowconf_commit", "post_lowconf_commit": "post_lowconf_commit", "used_fit_model": "used_fit_model", "correct": "correct", "spent": "spent", "verdict_none": "verdict_none",
       "verdict_planet": "verdict_planet", "verdict_cannot": "verdict_cannot", "invalid": "invalid",
       "abstain_correct": "abstain_correct", "n_transits": "n_transits", "n_rvs": "n_rvs", "tool_errors": "tool_errors", "es": "es", "post_conf": "post_conf", "committed": "committed"}


def flatten(rec):
    row = {"step": rec.get("step", rec.get("progress/batch"))}
    for src, dst in (("cost/cumulative_usd", "cost_usd"), ("time/total", "step_seconds"), ("tokens/cumulative/sampled", "tokens_sampled"),
                     ("tokens/cumulative/trained", "tokens_trained")):
        if src in rec:
            row[dst] = rec[src]
    for k, v in rec.items():
        if not isinstance(v, (int, float)):
            continue
        for src, dst in MAP.items():
            if k == f"env/all/{src}":
                row[f"train_{dst}"] = v
            elif k.endswith(f"env/all/{src}") and k != f"env/all/{src}":
                row[f"eval_{dst}"] = v
    return row


def jsonl_to_csv(log_dir):
    src = os.path.join(log_dir, "metrics.jsonl")
    if not os.path.exists(src):
        return None
    rows = {}
    for line in open(src):
        try:
            r = flatten(json.loads(line))
        except json.JSONDecodeError:
            continue
        if r["step"] is None:
            continue
        rows.setdefault(int(r["step"]), {}).update(r)
    cols = ["step", "cost_usd", "step_seconds", "tokens_sampled", "tokens_trained"] + [f"{p}_{c}" for p in ("train", "eval") for c in MAP.values()]
    out = os.path.join(log_dir, "metrics.csv")
    with open(out + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for s in sorted(rows):
            w.writerow({c: rows[s].get(c, "") for c in cols})
    os.replace(out + ".tmp", out)
    return out
