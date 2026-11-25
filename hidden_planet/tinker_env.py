"""tinker-cookbook adapter: Tool objects, EnvGroupBuilder, RLDataset, RLDatasetBuilder.

Uses tinker_cookbook.tool_use.build_agent_tool_env, so the env is a standard
tinker_cookbook.rl.types.Env (initial_observation / step) and plugs into rl.train unchanged.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import chz
import tinker
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer, ToolCall, ToolSpec, get_text_content
from tinker_cookbook.rl import types as rl_types
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder
from tinker_cookbook.tool_use import build_agent_tool_env
from tinker_cookbook.tool_use.types import ToolInput, ToolResult

from .reward import INVALID
from .tasks import sample_system, EASY, HARD
from .text_env import TOOL_SPECS, HiddenPlanetSession


class SessionTool:
    """One tool of a HiddenPlanetSession, implementing the cookbook Tool protocol."""

    def __init__(self, session: HiddenPlanetSession, spec: dict[str, Any]):
        self._session, self._spec = session, spec

    @property
    def name(self) -> str:
        return self._spec["name"]

    @property
    def description(self) -> str:
        return self._spec["description"]

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._spec["parameters"]

    def to_spec(self) -> ToolSpec:
        return {"name": self.name, "description": self.description, "parameters": self.parameters_schema}

    async def run(self, input: ToolInput) -> ToolResult:
        # One lock per session keeps several calls from one reply in order; the numpy work runs in a
        # thread so 100+ concurrent rollouts do not block the event loop.
        lock = self._session.__dict__.setdefault("_lock", asyncio.Lock())
        async with lock:
            content, done = await asyncio.to_thread(self._session.call, self.name, input.arguments)
        return ToolResult(messages=[{"role": "tool", "content": content, "tool_call_id": input.call_id or "",
                                     "name": self.name}], should_stop=done)


def make_tools(session: HiddenPlanetSession) -> list[SessionTool]:
    return [SessionTool(session, s) for s in session.tool_specs()]


_HINT_CACHE: dict[int, str] = {}


def tool_call_example(renderer: Renderer) -> str:
    """The exact text this renderer expects for one tool call (rendered by the renderer itself)."""
    key = id(renderer)
    if key not in _HINT_CACHE:
        dummy: list[Message] = [{"role": "user", "content": "x"}]
        msg: Message = {"role": "assistant", "content": "", "tool_calls": [ToolCall(
            function=ToolCall.FunctionBody(name="observe_transits", arguments=json.dumps({"epochs": [0, 40, 80, 120]})), id="call_0")]}
        full, _ = renderer.build_supervised_example(dummy + [msg])
        n = len(renderer.build_generation_prompt(dummy).to_ints())
        text = renderer.tokenizer.decode([int(t) for t in full.to_ints()][n:])
        i, j = text.find("<"), text.rfind(">")           # keep the tool-call block, drop end-of-turn tokens
        for end in ("<|im_end|>", "<|eot_id|>", "<|end|>", "</s>", "<|endoftext|>"):
            text = text.replace(end, "")
        _HINT_CACHE[key] = text.strip()
    return _HINT_CACHE[key]


def initial_messages(session: HiddenPlanetSession, renderer: Renderer) -> list[Message]:
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[t.to_spec() for t in make_tools(session)], system_prompt=session.system_prompt(tool_call_example(renderer)))
    return prefix + [{"role": "user", "content": session.user_prompt()}]


MAX_NUDGES = 2
NUDGE_PENALTY = 0.05
NUDGE_TEXT = ("Your last reply did not contain a valid tool call. Reply with one short sentence and then exactly one tool call, "
              "in exactly this format:\n{example}")


class NudgingEnv(EnvFromMessageEnv):
    """EnvFromMessageEnv, except that a turn without a usable tool call (prose only, cut off by the
    token cap, or malformed call) is not fatal: the turn is kept as plain text, a short corrective
    user message is appended, and the turn costs NUDGE_PENALTY. After MAX_NUDGES such turns (or at
    the turn limit) the episode ends and is graded as unanswered (-1 - cost)."""

    def __init__(self, *args, session: HiddenPlanetSession, **kw):
        super().__init__(*args, **kw)
        self.session = session
        self.nudges = 0

    async def step(self, action, *, extra=None):
        stop_reason = (extra or {}).get("stop_reason", "stop")
        message, termination = self.renderer.parse_response(action)
        if message.get("tool_calls") and termination.is_clean and stop_reason != "length":
            return await super().step(action, extra=extra)
        menv = self.message_env
        self.nudges += 1
        menv._turn_count += 1
        text = (get_text_content(message) or "")[:400]
        menv.history.append({"role": "assistant", "content": text or "(no tool call)"})
        metrics = {"nudge": 1.0, "truncated_turn": float(stop_reason == "length")}
        if self.nudges > MAX_NUDGES or menv._turn_count >= menv.max_turns:
            reward, m = self.session.grade()
            metrics.update(m)
            metrics[f"{rl_types.STOP_METRIC_PREFIX}{rl_types.StopReason.PARSE_ERROR}"] = 1.0
            return rl_types.StepResult(reward=reward - NUDGE_PENALTY, episode_done=True, next_observation=tinker.ModelInput.empty(),
                                       next_stop_condition=self._base_stop_condition, metrics=metrics)
        menv.history.append({"role": "user", "content": NUDGE_TEXT.format(example=tool_call_example(self.renderer))})
        ob = await self._render_in_thread(menv.history)
        return rl_types.StepResult(reward=-NUDGE_PENALTY, episode_done=False, next_observation=ob,
                                   next_stop_condition=self._base_stop_condition, metrics=metrics)


def build_env(session: HiddenPlanetSession, renderer: Renderer, max_trajectory_tokens: int = 48_000,
              max_generation_tokens: int | None = 1024):
    async def reward_fn(history: list[Message]):
        return await asyncio.to_thread(session.grade)       # INVALID - cost if no valid submit_answer was made; the posterior check runs off the event loop

    base = build_agent_tool_env(
        renderer=renderer, tools=make_tools(session), initial_messages=initial_messages(session, renderer),
        reward_fn=reward_fn, max_turns=session.max_turns, failed_parse_reward=INVALID,
        terminate_on_parse_error=True, max_trajectory_tokens=max_trajectory_tokens,
        max_generation_tokens=max_generation_tokens, context_overflow_reward=INVALID)
    return NudgingEnv(renderer, base.message_env, session=session, failed_parse_reward=INVALID, terminate_on_parse_error=True,
                      max_trajectory_tokens=max_trajectory_tokens, max_generation_tokens=max_generation_tokens,
                      context_overflow_reward=INVALID)


class HiddenPlanetEnvGroupBuilder(EnvGroupBuilder):
    """group_size rollouts of the *same* system and noise realisation (variance reduction)."""

    def __init__(self, seed: int, level: int, split: str, group_size: int, model_name: str,
                 renderer_name: str | None = None, abstain_aware: bool = True, backend: str | None = None,
                 max_turns: int | None = None, allow_other: bool = False, ooc_kind: str | None = None,
                 cost_per_unit: float | None = None, reward_mode: str = "partial", wrong_penalty: float | None = None, tool_level: str = "oracle",
                 train_reward: str = "realised"):
        self.tool_level, self.train_reward = tool_level, train_reward
        self.allow_other, self.ooc_kind, self.cost_per_unit = allow_other, ooc_kind, cost_per_unit
        self.reward_mode, self.wrong_penalty = reward_mode, wrong_penalty
        self.seed, self.level, self.split, self.group_size = seed, level, split, group_size
        self.model_name, self.renderer_name = model_name, renderer_name
        self.abstain_aware, self.backend, self.max_turns = abstain_aware, backend, max_turns

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer = get_renderer(self.renderer_name or model_info.get_recommended_renderer_name(self.model_name), tokenizer)
        if self.ooc_kind:
            from .ooc import sample_ooc_system
            sp = sample_ooc_system(self.seed, self.ooc_kind, self.split)
        else:
            sp = sample_system(self.seed, self.level, self.split)
        from .policies import reference_identifiability
        box = {}

        def q_ref():                                             # at most once per group, only if someone abstains
            if "v" not in box:
                box["v"] = reference_identifiability(sp, self.backend)
            return box["v"]
        return [build_env(HiddenPlanetSession(sp, q_ref=q_ref, backend=self.backend,
                                              abstain_aware=self.abstain_aware, max_turns=self.max_turns,
                                              allow_other=self.allow_other, cost_per_unit=self.cost_per_unit,
                                              reward_mode=self.reward_mode, wrong_penalty=self.wrong_penalty, tool_level=self.tool_level,
                                              train_reward=self.train_reward), renderer)
                for _ in range(self.group_size)]

    def logging_tags(self) -> list[str]:
        return [f"hidden_planet_L{self.level}", self.split] + ([f"ooc_{self.ooc_kind}"] if self.ooc_kind else [])


_POOL: dict = {}


def _curriculum_pool():
    if "v" not in _POOL:
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "curriculum_hard.json")
        _POOL["v"] = json.load(open(path))["ambiguous"] if os.path.exists(path) else []
    return _POOL["v"]


def curriculum_level(batch_index: int, n_batches: int, schedule: str = "easy") -> int:
    """schedule: 'easy' | 'hard' | 'mix' (easy for the first 60 % of steps, then hard)."""
    if schedule == "easy":
        return EASY
    if schedule == "hard":
        return HARD
    return EASY if batch_index < 0.6 * n_batches else HARD


class HiddenPlanetDataset(RLDataset):
    def __init__(self, n_batches: int, batch_size: int, group_size: int, split: str, model_name: str,
                 renderer_name: str | None = None, seed: int = 0, schedule: str = "easy", eval_level: int = EASY, **env_kw):
        self.n_batches, self.batch_size, self.group_size, self.split = n_batches, batch_size, group_size, split
        self.model_name, self.renderer_name, self.seed, self.schedule, self.env_kw = model_name, renderer_name, seed, schedule, env_kw
        self.eval_level = eval_level

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        level = self.eval_level if self.split != "train" else curriculum_level(index, self.n_batches, self.schedule)
        base = 500_000 + self.seed * 10_000_000 + index * self.batch_size      # disjoint from the SFT demonstration seeds (train 0-699) for every RL seed
        if self.split != "train":
            base = self.env_kw.get("eval_start", 0)           # fixed held-out systems, independent of the training seed
        out = []
        for i in range(self.batch_size):
            kw = dict(self.env_kw)
            mix = kw.pop("ooc_mix", 0.0)
            amb = kw.pop("ambiguous_frac", 0.0); kw.pop("eval_start", None)
            sd = base + i
            if amb > 0 and self.split == "train":           # curriculum: a share of decision-relevant TRAIN systems
                pool = _curriculum_pool()
                u = ((base + i) * 40503 % 65521) / 65521.0
                if pool and u < amb:
                    sd = pool[(self.seed * 7919 + index * self.batch_size + i) % len(pool)]
            if mix > 0:                                    # deterministic out-of-class mixture (level-5 systems)
                u = ((base + i) * 2654435761 % 2**32) / 2**32
                if u < mix:
                    kw["ooc_kind"] = ("two_planets", "activity", "drift")[(base + i) % 3]
            out.append(HiddenPlanetEnvGroupBuilder(sd, level, self.split, self.group_size, self.model_name,
                                                   self.renderer_name, **kw))
        return out

    def __len__(self) -> int:
        return self.n_batches


@chz.chz
class HiddenPlanetDatasetBuilder(RLDatasetBuilder):
    model_name_for_tokenizer: str
    renderer_name: str | None = None
    batch_size: int = 16            # systems (groups) per training step
    group_size: int = 8             # rollouts per system
    n_batches: int = 30
    n_test_systems: int = 32
    seed: int = 0
    schedule: str = "easy"
    eval_level: int = EASY
    abstain_aware: bool = True
    backend: str | None = None
    cost_per_unit: float | None = None      # None = reward.LAMBDA (0.005)
    allow_other: bool = False               # 4th verdict 'other'
    ooc_mix: float = 0.0                    # fraction of out-of-class systems (train and eval)
    reward_mode: str = "partial"            # 'strict': planet = +1 only if period AND mass are right, else -wrong_penalty
    wrong_penalty: float | None = None
    ambiguous_frac: float = 0.0             # share of TRAIN systems drawn from data/curriculum_hard.json (ambiguous first fit)
    eval_start: int = 0                     # first test-split seed of the in-run eval set
    tool_level: str = "oracle"              # 'lite': fit_model prints chi2 / BIC, no posterior probabilities
    train_reward: str = "realised"          # 'expected': TRAINING episodes return the posterior-expected score; in-run evals always realised

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        kw = dict(abstain_aware=self.abstain_aware, backend=self.backend, cost_per_unit=self.cost_per_unit,
                  allow_other=self.allow_other, ooc_mix=self.ooc_mix, reward_mode=self.reward_mode, wrong_penalty=self.wrong_penalty, ambiguous_frac=self.ambiguous_frac, eval_start=self.eval_start, tool_level=self.tool_level)
        train = HiddenPlanetDataset(self.n_batches, self.batch_size, self.group_size, "train", self.model_name_for_tokenizer,
                                    self.renderer_name, self.seed, self.schedule, train_reward=self.train_reward, **kw)
        test = HiddenPlanetDataset(1, self.n_test_systems, 1, "test", self.model_name_for_tokenizer,          # realised reward; ES is logged as a metric
                                   self.renderer_name, 0, "easy", self.eval_level, **kw)
        return train, test


class ReplayEnvGroupBuilder(EnvGroupBuilder):
    """One archival-data episode (see hidden_planet/replay.py)."""

    def __init__(self, koi: str, seed: int, model_name: str, renderer_name: str | None = None, allow_other: bool = False,
                 cost_per_unit: float | None = None, reward_mode: str = "partial", wrong_penalty: float | None = None):
        self.koi, self.seed, self.model_name, self.renderer_name = koi, seed, model_name, renderer_name
        self.allow_other, self.cost_per_unit, self.reward_mode, self.wrong_penalty = allow_other, cost_per_unit, reward_mode, wrong_penalty

    async def make_envs(self) -> Sequence[Env]:
        from .replay import ReplaySession
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer = get_renderer(self.renderer_name or model_info.get_recommended_renderer_name(self.model_name), tokenizer)
        return [build_env(ReplaySession(self.koi, self.seed, allow_other=self.allow_other, cost_per_unit=self.cost_per_unit,
                                        reward_mode=self.reward_mode, wrong_penalty=self.wrong_penalty), renderer)]

    def logging_tags(self) -> list[str]:
        return [f"replay_{self.koi}"]
