"""The exact tool-call text shown to the model must parse with the same renderer, and prose-only /
truncated turns must be nudged rather than end the episode."""
import asyncio, json, os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from hidden_planet.tinker_env import build_env, tool_call_example, MAX_NUDGES, NUDGE_PENALTY
from hidden_planet.text_env import HiddenPlanetSession

CASES = [("Qwen/Qwen3.5-9B", "qwen3_5_disable_thinking"), ("Qwen/Qwen3.5-9B", "qwen3_5"),
         ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "nemotron3_ultra_disable_thinking")]


def _setup(model, rname):
    tok = tokenizer_utils.get_tokenizer(model)
    r = get_renderer(rname, tok)
    return tok, r


def _end(tok, r):
    stop = r.get_stop_sequences()[0]
    return [stop] if isinstance(stop, int) else tok.encode(stop, add_special_tokens=False)


@pytest.mark.parametrize("model,rname", CASES)
def test_handwritten_tool_call_parses(model, rname):
    tok, r = _setup(model, rname)
    ex = tool_call_example(r)
    assert "observe_transits" in ex
    # hand-edit the example into a different call, exactly as a model would type it
    text = "The timing fit shows nothing.\n\n" + ex.replace("observe_transits", "observe_rvs").replace("epochs", "times_days").replace("[0, 40, 80, 120]", "[12.5, 300, 801.25]")
    msg, term = r.parse_response(tok.encode(text, add_special_tokens=False) + _end(tok, r))
    assert term.is_clean and not msg.get("unparsed_tool_calls")
    (tc,) = msg["tool_calls"]
    assert tc.function.name == "observe_rvs" and json.loads(tc.function.arguments) == {"times_days": [12.5, 300, 801.25]}


@pytest.mark.parametrize("model,rname", CASES[:1])
def test_prose_turn_is_nudged_then_recovers(model, rname):
    tok, r = _setup(model, rname)

    async def go():
        s = HiddenPlanetSession.from_seed(4, 0, "test")
        env = build_env(s, r)
        ob, _ = await env.initial_observation()
        assert tool_call_example(r) in tok.decode(ob.to_ints())            # the syntax example is in the prompt
        prose = tok.encode("**Strategy:** 1. First I will think about super-periods at length", add_special_tokens=False)
        res = await env.step(prose, extra={"stop_reason": "length"})        # cut off by the token cap
        assert not res.episode_done and res.reward == -NUDGE_PENALTY
        assert "did not contain a valid tool call" in tok.decode(res.next_observation.to_ints())
        good = tok.encode(tool_call_example(r).replace("observe_transits", "submit_answer").replace("epochs", "verdict").replace("[0, 40, 80, 120]", "none"), add_special_tokens=False) + _end(tok, r)
        res = await env.step(good)
        assert res.episode_done and s.answer["verdict"] == "none" and res.reward > -0.6
        # too many prose turns -> graded as unanswered
        s2 = HiddenPlanetSession.from_seed(4, 0, "test"); env2 = build_env(s2, r); await env2.initial_observation()
        for k in range(MAX_NUDGES + 1):
            res = await env2.step(prose + _end(tok, r))
        assert res.episode_done and res.reward <= -1.0
    asyncio.run(go())
