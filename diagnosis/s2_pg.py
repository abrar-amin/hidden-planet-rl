"""Group-relative policy gradient (same estimator as tinker-cookbook: advantage = return - mean of the 8 rollouts of the same
system; one Adam step per batch) on a tabular softmax policy over what the agent can SEE (stage, p_no_perturber bucket,
p_both_right bucket, has-RVs), acting in the real environment via the precomputed lattice (s1_precompute.py).
Init = soft clone of the fixed-design demonstrator (T16 -> R8 -> commit top option iff > 0.5 else abstain) with exploration eps.
This is an OPTIMISTIC stand-in for the LLM: ~170 parameters that are exactly the decision-relevant ones, no language, no forgetting,
step size tuned by sweep. If the signal is too weak here, it is too weak for the LoRA."""
import pickle, sys, os, json, itertools
import numpy as np
from functools import lru_cache
D = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lattice.pkl"), "rb"))
RES = D["res"]; LAM = 0.008
OPEN = (10, 16); ACT = ("COMMIT", "ABSTAIN", "T8", "R4", "R8"); NA = 5
PN = (0.1, 0.3, 0.5, 0.9); PB = (0.3, 0.5, 0.85)
NS = 4 * 5 * 4 * 2

def feat(node, stage, b):
    return ((min(stage, 3) * 5 + int(np.searchsorted(PN, node[0], side="right"))) * 4 + int(np.searchsorted(PB, node[1], side="right"))) * 2 + int(b > 0)

def valid(sysd, o, a, b):
    m = np.zeros(NA, bool); m[0] = m[1] = True
    n = sysd["nodes"]
    m[2] = n.get((o, a + 1, b)) is not None
    m[3] = n.get((o, a, b + 1)) is not None
    m[4] = n.get((o, a, b + 2)) is not None
    return m

def nxt(o, a, b, act):
    return (o, a + 1, b) if act == 2 else (o, a, b + 1) if act == 3 else (o, a, b + 2)

def term_reward(node, act, mode):
    """terminal reward. mode 'real': realized +-1; 'exp': calibrated expectation 2p-1 of the committed option (variance-reduced)."""
    pn, pb, sp_, sn_, spent = node
    if act == 1: return -LAM * spent
    top_planet = pb >= pn
    if mode == "real": sc = sp_ if top_planet else sn_
    else: sc = 2 * (pb if top_planet else pn) - 1
    return sc - LAM * spent

def init_theta(eps, shaped=True):
    """logits: th0 (opening, 2), th (NS x NA). Demonstrator: open 16; stage 0 -> R8; stage >= 1 -> COMMIT if top > .5 else ABSTAIN."""
    th0 = np.log(np.array([eps, 1 - eps])); th = np.zeros((NS, NA))
    hi = np.log((1 - eps) / (eps / (NA - 1)))
    for st in range(4):
        for i in range(5):
            for j in range(4):
                for r in range(2):
                    s = ((st * 5 + i) * 4 + j) * 2 + r
                    pref = 4 if st == 0 else (0 if (i >= 3 or j >= 2) else 1)
                    th[s, pref] = hi
    return th0, th

def probs(th_row, mask):
    z = np.where(mask, th_row, -1e9); z = z - z.max(); p = np.exp(z); return p / p.sum()

def rollout(sysd, th0, th, rng, mode):
    p0 = np.exp(th0 - th0.max()); p0 /= p0.sum()
    oi = rng.choice(2, p=p0); o = OPEN[oi]; a = b = 0; stage = 0; steps = []
    while True:
        node = sysd["nodes"][(o, a, b)]
        s = feat(node, stage, b); m = valid(sysd, o, a, b); p = probs(th[s], m)
        act = rng.choice(NA, p=p); steps.append((s, act, p))
        if act < 2:
            return term_reward(node, act, mode), (oi, p0), steps, node, act
        o, a, b = nxt(o, a, b, act); stage += 1

def evaluate(th0, th, seeds, split="test", greedy=False):
    """EXACT expected realized reward of the stochastic (or greedy) policy on the given systems; plus diagnostics."""
    p0 = np.exp(th0 - th0.max()); p0 /= p0.sum()
    if greedy: p0 = np.eye(2)[p0.argmax()]
    tot = np.zeros(5)   # reward, spent, P(abstain), P(commit with top<=.5), P(buy any RV)
    for sd in seeds:
        sysd = RES[(split, sd)]; memo = {}
        def V(o, a, b, stage):
            k = (o, a, b, min(stage, 3))
            if k in memo: return memo[k]
            node = sysd["nodes"][(o, a, b)]; m = valid(sysd, o, a, b); p = probs(th[feat(node, stage, b)], m)
            if greedy: p = np.eye(NA)[p.argmax()]
            v = np.zeros(5)
            for act in range(NA):
                if p[act] < 1e-12: continue
                if act < 2:
                    low = float(act == 0 and max(node[0], node[1]) <= 0.5)
                    v += p[act] * np.array([term_reward(node, act, "real"), node[4], float(act == 1), low, float(b > 0)])
                else:
                    v += p[act] * V(*nxt(o, a, b, act), stage + 1)
            memo[k] = v; return v
        for oi in range(2):
            if p0[oi] > 1e-12: tot += p0[oi] * V(OPEN[oi], 0, 0, 0)
    return tot / len(seeds)

def teacher_theta():
    """fit-teacher rule expressed in the family (deterministic): p0>=.9 or pb>=.85 commit; p0>=.3 -> T8; else R4; when nothing affordable -> rule."""
    th0 = np.array([-20.0, 0.0]); th = np.zeros((NS, NA))
    for st in range(4):
        for i in range(5):
            for j in range(4):
                for r in range(2):
                    s = ((st * 5 + i) * 4 + j) * 2 + r
                    fin = 0 if (i >= 3 or j >= 2) else 1
                    th[s, fin] = 20.0                                   # fallback when purchases are masked out
                    if i == 4 or j == 3: th[s, 0] = 60.0
                    elif i >= 2: th[s, 2] = 40.0
                    else: th[s, 3] = 40.0
    return th0, th

def train(seed, steps, n_sys, group, lr, eps, mode="real", amb_frac=0.6, eval_at=None, opt="adam", baseline="group"):
    rng = np.random.default_rng(seed)
    th0, th = init_theta(eps)
    m0 = np.zeros(2); v0 = np.zeros(2); m = np.zeros_like(th); v = np.zeros_like(th)
    b1, b2 = 0.9, 0.95
    eval_at = eval_at or [0, steps]
    out = {}
    for t in range(steps + 1):
        if t in eval_at:
            out[t] = evaluate(th0, th, D["dev"])
        if t == steps: break
        g0 = np.zeros(2); g = np.zeros_like(th); train_R = []
        for _ in range(n_sys):
            sd = int(rng.choice(D["amb"])) if rng.random() < amb_frac else int(rng.choice(D["fresh"]))
            sysd = RES[("train", sd)]
            ro = [rollout(sysd, th0, th, rng, mode) for _ in range(group)]
            R = np.array([r[0] for r in ro]); A = R - R.mean() if baseline == "group" else R
            train_R.extend(R)
            for (r, (oi, p0), stp, node, act), adv in zip(ro, A):
                e = -p0.copy(); e[oi] += 1; g0 += adv * e
                for (s, a_, p) in stp:
                    e = -p.copy(); e[a_] += 1; g[s] += adv * e
        N = n_sys * group; g0 /= N; g /= N
        if opt == "adam":
            m0 = b1 * m0 + (1 - b1) * g0; v0 = b2 * v0 + (1 - b2) * g0 ** 2; m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g ** 2
            c1, c2 = 1 - b1 ** (t + 1), 1 - b2 ** (t + 1)
            th0 = th0 + lr * (m0 / c1) / (np.sqrt(v0 / c2) + 1e-8); th = th + lr * (m / c1) / (np.sqrt(v / c2) + 1e-8)
        else:
            th0 = th0 + lr * g0; th = th + lr * g
    return out

if __name__ == "__main__":
    print("nodes failing fit:", sum(v is None for s in RES.values() for v in s["nodes"].values()))
    names = ("reward", "spent", "abstain", "lowconf_commit", "rv_rate")
    for nm, (a, b) in (("demonstrator eps=0", init_theta(1e-9)), ("teacher-in-family", teacher_theta())):
        print(f"{nm:22s} dev:", dict(zip(names, np.round(evaluate(a, b, D['dev']), 3))))
    for eps in (0.02, 0.1, 0.3):
        print(f"soft clone eps={eps:<5}  dev:", dict(zip(names, np.round(evaluate(*init_theta(eps), D['dev']), 3))))
    # calibration of printed probabilities on the lattice (train pool)
    P, S = [], []
    for (sp, sd), s in RES.items():
        if sp != "train": continue
        for n in s["nodes"].values():
            if n is None: continue
            tp = n[1] >= n[0]; P.append(max(n[0], n[1])); S.append((n[2] if tp else n[3]) > 0)
    P, S = np.array(P), np.array(S)
    for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)):
        k = (P >= lo) & (P < hi); print(f"  top-option prob [{lo},{hi}): n={k.sum():6d} mean p {P[k].mean():.3f} right {S[k].mean():.3f}")
