"""Scripted agents play full episodes through the real tinker-cookbook Env
(tokens in -> renderer.parse_response -> tools -> tokens out), with per-episode token counts.
Also the rollout pretty-printer (--show K) and the expert-trajectory dumper (--dump FILE).

usage: dry_run.py [--agent compact|heuristic|eig|random] [--level easy|hard] [--n 40] [--split test]
                  [--show 2] [--dump data/expert.jsonl] [--json out.json] [--model Qwen/Qwen3.5-9B]
"""
import os, sys, json, argparse, asyncio, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
    os.environ[k] = "1" if k != "TOKENIZERS_PARALLELISM" else "false"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import ToolCall
from hidden_planet.tinker_env import build_env
from hidden_planet.text_env import HiddenPlanetSession
from hidden_planet.text_policies import heuristic_text_agent, eig_text_agent, compact_text_agent, random_text_agent, fit_following_agent, lite_teacher_agent, fixed_fit_agent
from hidden_planet.tasks import LEVEL_NAMES
from hidden_planet.tasks import sample_system

AGENTS = dict(heuristic=heuristic_text_agent, eig=eig_text_agent, compact=compact_text_agent, random=random_text_agent, fit=fit_following_agent, lite=lite_teacher_agent, fixed=fixed_fit_agent)


def action_tokens(renderer, history, message):
    """Token ids the model would have to sample to produce `message` after `history`."""
    prompt = renderer.build_generation_prompt(history)
    full, _ = renderer.build_supervised_example(history + [message])
    p, f = prompt.to_ints(), [int(t) for t in full.to_ints()]
    assert f[:len(p)] == p, "generation prompt is not a prefix of the supervised rendering"
    return f[len(p):]


async def play(seed, level, split, renderer, agent, show=False, dump=None, skw=None, dup_demo=0.0, diverse=False):
    skw = skw or {}
    np.seterr(all="ignore")
    sp = sample_system(seed, level, split)
    session = HiddenPlanetSession(sp, **skw)
    env = build_env(session, renderer)
    ob, _stop = await env.initial_observation()
    menv = env.message_env
    n_prefix = len(menv.history) - 1          # system/tool-declaration messages before the user task
    gen = AGENTS[agent](session.user_prompt(), np.random.default_rng(seed + 17), **({"max_turns": session.max_turns} if agent in ("eig", "fit") else {}),
                        **({"mode": session.reward_mode, "wrong_penalty": session.wrong_penalty or (1.0 if session.reward_mode == "strict0" else 0.5)} if agent == "fit" else {}),
                        **({"dup_demo": dup_demo, "diverse": True} if agent in ("fit", "compact", "lite", "fixed") and (dup_demo or diverse) else {}),
                        **({"max_turns": session.max_turns} if agent == "lite" else {}),
                        **({"mode": session.reward_mode, "wrong_penalty": session.wrong_penalty or 1.0} if agent == "eig" and session.reward_mode != "partial" else {}))
    step = next(gen)
    prefill = cached = sampled = turns = 0
    prev_len = 0
    total_reward, metrics = 0.0, {}
    if show:
        for m in menv.history:
            print(f"--- {m['role']} ---\n{m['content']}\n")
    while True:
        thought, calls = step
        msg = {"role": "assistant", "content": thought, "tool_calls": [
            ToolCall(function=ToolCall.FunctionBody(name=n, arguments=json.dumps(a)), id=f"call_{turns}_{i}") for i, (n, a) in enumerate(calls)]}
        act = action_tokens(renderer, list(menv.history), msg)
        prefill += ob.length; cached += min(prev_len, ob.length); prev_len = ob.length + len(act)
        sampled += len(act); turns += 1
        n_hist = len(menv.history)
        res = await env.step(act)
        total_reward += res.reward; metrics.update(res.metrics)
        new = menv.history[n_hist:]
        if show:
            for m in new:
                c = m["content"] if isinstance(m["content"], str) else str(m["content"])
                tc = "".join(f"\n  -> {t.function.name}({t.function.arguments})" for t in (m.get("tool_calls") or []))
                print(f"--- {m['role']} ---\n{c[:700]}{tc}\n")
        if res.episode_done:
            break
        ob = res.next_observation
        results = [json.loads(m["content"]) for m in new if m["role"] == "tool"]
        step = gen.send(results)
    if dump is not None:          # renderer-agnostic record for SFT / distillation
        msgs = []
        for m in menv.history[n_prefix:]:
            c = m["content"] if isinstance(m["content"], str) else "".join(p.get("text", "") for p in m["content"])
            d = {"role": m["role"], "content": c.strip()}
            if m.get("tool_calls"):
                d["tool_calls"] = [{"name": t.function.name, "arguments": json.loads(t.function.arguments)} for t in m["tool_calls"]]
            if d["role"] == "assistant" and d["content"].startswith("DUPLICATE_DEMO"):
                d["content"] = "More observations."; d["no_train"] = True        # the deliberately mistaken turn: context only, never a training target
            if m["role"] == "tool":
                d["name"] = m.get("name", "")
            msgs.append(d)
        dump.write(json.dumps(dict(seed=seed, level=level, split=split, agent=agent, reward=total_reward, reward_mode=session.reward_mode,
                                   messages=msgs)) + "\n")
    return dict(seed=seed, reward=total_reward, turns=turns, prefill=prefill, cached=cached, sampled=sampled,
                final_context=prev_len, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B"); ap.add_argument("--renderer", default="qwen3_5_disable_thinking"); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--agent", default="compact"); ap.add_argument("--split", default="test")
    ap.add_argument("--level", default="easy"); ap.add_argument("--dump", default=None)
    ap.add_argument("--reward-mode", default="partial"); ap.add_argument("--wrong-penalty", type=float, default=None); ap.add_argument("--cost-per-unit", type=float, default=None)
    ap.add_argument("--tool-level", default="oracle", choices=["oracle", "lite"]); ap.add_argument("--diverse", action="store_true", help="vary batch sizes of the scripted demonstrator")
    ap.add_argument("--dup-demo", type=float, default=0.0, help="fit agent only: probability per acquisition of demonstrating recovery from a refused duplicate request"); ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--json", default=None)
    ap.add_argument("--rv-availability", default=None, choices=["ideal", "realistic"])
    a = ap.parse_args()
    if a.rv_availability:
        from hidden_planet.tasks import set_rv_mode
        set_rv_mode(a.rv_availability)
    tok = tokenizer_utils.get_tokenizer(a.model)
    renderer = get_renderer(a.renderer, tok)
    out = []
    t0 = time.time()
    level = LEVEL_NAMES[a.level] if a.level in LEVEL_NAMES else int(a.level)
    dump = open(a.dump, "w") if a.dump else None
    for i in range(a.n):
        r = await play(a.start + i, level, a.split, renderer, a.agent, show=i < a.show, dump=dump,
                       skw=dict(reward_mode=a.reward_mode, wrong_penalty=a.wrong_penalty, cost_per_unit=a.cost_per_unit, tool_level=a.tool_level), dup_demo=a.dup_demo, diverse=a.diverse)
        if i < a.show:
            print(f"==> episode reward {r['reward']:+.3f}\n" + "=" * 100)
        out.append(r)
    if dump:
        dump.close()
    g = lambda k: np.array([o.get(k, 0.0) for o in out], float)
    ci = lambda x: 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    summ = dict(model=a.model, agent=a.agent, n=a.n, level=a.level, split=a.split, reward_mode=a.reward_mode, cost_per_unit=a.cost_per_unit,
                rv_buy_rate=g("rv_buy_rate").mean(), submit_conf=g("submit_conf").mean(), lowconf_planet=g("lowconf_planet").mean(),
                reward=g("reward").mean(), reward_ci=ci(g("reward")), turns=g("turns").mean(), turns_max=g("turns").max(),
                prefill_tokens=g("prefill").mean(), cached_prefill_tokens=g("cached").mean(), sampled_tokens=g("sampled").mean(),
                final_context_tokens=g("final_context").mean(), final_context_max=g("final_context").max(),
                spent=g("spent").mean(), correct=g("correct").mean(), cannot=g("verdict_cannot").mean(), invalid=g("invalid").mean(),
                sec_per_episode=(time.time() - t0) / a.n)
    print(json.dumps(summ, indent=1))
    if a.json:
        json.dump(dict(summary=summ, episodes=out), open(a.json, "w"), indent=1, default=float)

if __name__ == "__main__":
    asyncio.run(main())
