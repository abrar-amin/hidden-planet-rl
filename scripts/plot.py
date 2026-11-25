"""Headline figure: reward vs RL step with reference lines (random, script, Bayesian EIG, ceiling).

usage: plot.py --run runs/qwen9b_easy [--run runs/other --label "SFT+RL"] --level easy --out figures/learning_curve.png
Reads <run>/metrics.csv (written by train_rl.py) or falls back to <run>/metrics.jsonl.
"""
import os, sys, json, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scripts_common import jsonl_to_csv

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3, fixed order
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#6b6a64", "#e4e3dd", "#fcfcfb"
REFS = [("hindsight", "hindsight ceiling"), ("fit", "fit-teacher script"), ("eig", "Bayesian EIG"), ("fixed", "fixed-design demonstrator"), ("random", "random policy")]


def load_run(path):
    jsonl_to_csv(path)
    rows = list(csv.DictReader(open(os.path.join(path, "metrics.csv"))))
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else np.nan
    return {k: np.array([f(r, k) for r in rows]) for k in rows[0]} if rows else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True); ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--level", default="easy"); ap.add_argument("--out", default="figures/learning_curve.png")
    ap.add_argument("--zero-shot", default=None, help="eval_zero_shot.py JSON to mark the base model")
    ap.add_argument("--sft", default=None, help="eval_zero_shot.py JSON to mark the SFT model")
    ap.add_argument("--title", default=None); ap.add_argument("--ref", default=None, help="reference JSON from make_reference.py (default results/reference_<level>.json)")
    a = ap.parse_args()
    rp = a.ref or f"results/reference_{a.level}.json"
    ref = json.load(open(rp)) if os.path.exists(rp) else {}
    C_REF = ref.get("_meta", {}).get("cost", 0.005)          # all rewards are shown at the reference cost
    MODE = ref.get("_meta", {}).get("reward_mode", "partial")
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(19.5, 4.8), gridspec_kw={"width_ratios": [1.5, 1, 1], "wspace": 0.95})
    xmax = 1
    for i, run in enumerate(a.run):
        d = load_run(run)
        if not d:
            continue
        lab = a.label[i] if i < len(a.label) else os.path.basename(run.rstrip("/"))
        c = SERIES[i % len(SERIES)]
        # reward at the STANDARD observation cost (0.005/unit), so runs trained with another
        # --cost-per-unit stay comparable with the reference lines
        has = lambda k: k in d and (~np.isnan(d[k])).any()
        tr = d["train_score"] - C_REF * d["train_spent"] if has("train_score") and has("train_spent") else d["train_reward"]
        ev = d["eval_score"] - C_REF * d["eval_spent"] if has("eval_score") and has("eval_spent") else d["eval_reward"]
        d["train_reward"], d["eval_reward"] = tr, ev
        m = ~np.isnan(d["train_reward"])
        ax.plot(d["step"][m], d["train_reward"][m], color=c, lw=1, alpha=0.45)
        e = ~np.isnan(d["eval_reward"])
        if "eval_spent" in d and (~np.isnan(d["eval_spent"])).any():
            k = ~np.isnan(d["eval_spent"])
            ax3.plot(d["step"][k], d["eval_spent"][k], color=c, lw=2, marker="o", ms=4, label=f"{lab}: budget spent")
        ax.plot(d["step"][e], d["eval_reward"][e], color=c, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, label=f"{lab} (held-out eval; thin line = training batches)")
        if e.any():
            ax.annotate(f"{d['eval_reward'][e][-1]:+.2f}", (d["step"][e][-1], d["eval_reward"][e][-1]), textcoords="offset points", xytext=(-4, -14), color=INK, fontsize=9, ha="right")
        xmax = max(xmax, np.nanmax(d["step"]))
        for key, ls, name in (("eval_correct", "-", "fully correct"), ("eval_invalid", ":", "no valid answer"), ("eval_rv_buy_rate", "--", "bought RVs"),
                              ("eval_submit_conf", "-.", "fit confidence at submission")):
            if key in d and (~np.isnan(d[key])).any():
                k = ~np.isnan(d[key])
                ax2.plot(d["step"][k], d[key][k], color=c, lw=2, ls=ls, marker="o", ms=4, label=f"{lab}: {name}")
    for path, mk, name in ((a.zero_shot, "D", "base model, zero-shot"), (a.sft, "s", "after SFT only")):
        if path and os.path.exists(path):
            z = json.load(open(path))
            ax.plot([0], [z["score"] - C_REF * z["spent"] if "score" in z else z["reward"]], marker=mk, ms=8, color=INK if mk == "D" else MUTED, ls="none", label=name)
    def ref_lines(A, items, fmt, minsep):
        """Dashed reference lines with labels in the right margin, nudged apart so they never overlap."""
        items = sorted(items, key=lambda kv: -kv[1])
        ys = []
        for name, y in items:
            A.axhline(y, color=MUTED, lw=1, ls="--", zorder=0)
            yl = y if not ys else min(y, ys[-1] - minsep)
            ys.append(yl)
            A.annotate(f"{name} {fmt.format(y)}", xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(1.03, yl),
                       textcoords=("axes fraction", "data"), color=INK, fontsize=8.5, va="center", ha="left",
                       arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6), annotation_clip=False)
    ref_lines(ax, [(name, ref[k]["reward"]) for k, name in REFS if k in ref], "{:+.2f}", 0.11)
    ref_lines(ax2, [(name, ref[k]["correct"]) for k, name in REFS if k in ("eig", "fit") and k in ref], "{:.2f}", 0.06)
    ref_lines(ax3, [(name, ref[k]["spent"]) for k, name in REFS if k in ("eig", "fit", "hindsight") and k in ref], "{:.0f}", 3.5)
    for A, yl, t in ((ax, "mean episode reward", a.title or f"Find the hidden planet ({a.level}, {MODE} reward, cost {C_REF:g}/unit): reward vs RL step"), (ax2, "fraction of episodes", "Answer quality (held-out)"),
                      (ax3, "units of telescope time (of 60)", "Observing cost (held-out)")):
        A.set_xlabel("RL step"); A.set_ylabel(yl); A.set_title(t, color=INK, fontsize=11, loc="left")
        A.grid(axis="y", color=GRID, lw=0.8); A.set_axisbelow(True)
        for s in ("top", "right"):
            A.spines[s].set_visible(False)
        A.set_xlim(-0.5, xmax + 0.5); A.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        A.legend(frameon=False, fontsize=8, loc="lower left" if A is not ax2 else "center left", labelcolor=INK)
    ax.set_ylim(-1.35, 1.05); ax2.set_ylim(-0.03, 1.03); ax3.set_ylim(0, 62)
    fig.subplots_adjust(left=0.04, right=0.875, top=0.9, bottom=0.13)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=160)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
