"""A fake tinker.ServiceClient for dry runs: no network, no credits.

Only the remote calls are faked (sampling, forward_backward, optim_step, checkpoint saves). Everything
else in a dry run is the real code path: tinker_cookbook.rl.train.main, the real Env, tokenizer,
renderer, rollout runner, advantage/datum construction, evaluators, loggers.
Sampling is produced by hidden_planet.mock.MockCompleter, whose `skill` rises with the number of
fake optimiser steps so that the run shows a learning curve.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import tinker
from tinker.types.tensor_data import TensorData
import torch


class FakeFuture:
    """Awaitable that also supports .result() / .result_async(), like tinker.APIFuture."""

    def __init__(self, value):
        self._v = value

    def result(self, timeout=None):
        return self._v

    async def result_async(self, timeout=None):
        return self._v

    def coroutine(self):
        async def _c():
            return self._v
        return _c()

    def __await__(self):
        async def _c():
            await asyncio.sleep(0)
            return self._v
        return _c().__await__()


class _ClientFuture(FakeFuture):
    """create_sampling_client result: usable directly as a client and also via .result()/await."""

    def __getattr__(self, name):
        return getattr(self._v, name)


class FakeSamplingClient:
    def __init__(self, owner, skill):
        self._owner, self.skill = owner, skill
        self.calls = self.prompt_tokens = self.sampled_tokens = 0

    def _completer(self):
        from .mock import MockCompleter
        o = self._owner
        o.n_samplers += 1
        return MockCompleter(o.renderer, o.tokenizer, skill=self.skill, seed=o.seed * 100003 + o.n_samplers)

    def sample(self, prompt, num_samples=1, sampling_params=None, include_prompt_logprobs=False, **kw):
        if include_prompt_logprobs:                       # KL-reference scoring call
            self._owner.usage["prefill"] += prompt.length
            seq = SimpleNamespace(tokens=[0], logprobs=[-0.5], stop_reason="length", sequence_id=None)
            return FakeFuture(SimpleNamespace(sequences=[seq], prompt_cache_hit_tokens=0, prompt_logprobs=[None] + [-0.6] * (prompt.length - 1)))
        if not hasattr(self, "_mc"):
            self._mc = self._completer()
        seqs = []
        text = self._mc.tok.decode(prompt.to_ints())
        for i in range(num_samples):
            self._mc.calls += 1
            toks = self._mc._encode(*self._mc._decide(text))
            if sampling_params is not None and sampling_params.max_tokens:
                toks = toks[:sampling_params.max_tokens]
            seqs.append(SimpleNamespace(tokens=toks, logprobs=[-0.5] * len(toks), stop_reason="stop", sequence_id=None))
        self._owner.usage["prefill"] += prompt.length * num_samples
        self._owner.usage["sampled"] += sum(len(s.tokens) for s in seqs)
        self._owner.usage["sample_calls"] += num_samples
        return FakeFuture(SimpleNamespace(sequences=seqs, prompt_cache_hit_tokens=0))

    sample_async = sample

    def get_tokenizer(self):
        return self._owner.tokenizer


class FakeTrainingClient:
    def __init__(self, owner):
        self._owner = owner
        self.optim_steps = 0

    def _skill(self):
        o = self._owner
        return min(1.0, o.skill0 + (1 - o.skill0) * self.optim_steps / max(1, o.skill_steps))

    def forward_backward(self, data, loss_fn=None, loss_fn_config=None, **kw):
        outs = []
        for d in data:
            n = d.model_input.length
            self._owner.usage["trained"] += n
            outs.append({"logprobs": TensorData.from_torch(torch.full((n,), -0.5))})
        return FakeFuture(SimpleNamespace(loss_fn_outputs=outs, metrics={"loss:sum": 0.0}, loss_fn_output_type="fake"))

    forward_backward_async = forward_backward

    def optim_step(self, adam_params=None, **kw):
        self.optim_steps += 1
        return FakeFuture(SimpleNamespace(metrics={}))

    optim_step_async = optim_step

    def save_weights_and_get_sampling_client(self, name=None, **kw):
        return FakeFuture(FakeSamplingClient(self._owner, self._skill()))

    save_weights_and_get_sampling_client_async = save_weights_and_get_sampling_client

    def create_sampling_client(self, model_path=None, **kw):
        return FakeFuture(FakeSamplingClient(self._owner, self._skill()))

    def save_state(self, name, **kw):
        self._owner.checkpoints.append(name)
        return FakeFuture(SimpleNamespace(path=f"tinker://fake-run/state/{name}"))

    def save_weights_for_sampler(self, name, **kw):
        return FakeFuture(SimpleNamespace(path=f"tinker://fake-run/sampler/{name}"))

    save_state_async = save_state
    save_weights_for_sampler_async = save_weights_for_sampler

    def get_tokenizer(self):
        return self._owner.tokenizer

    def get_info(self):
        return SimpleNamespace(model_data=SimpleNamespace(model_name=self._owner.model_name, lora_rank=32), model_id="fake-run")


class FakeServiceClient:
    """Drop-in for tinker.ServiceClient in dry runs."""
    last = None

    def __init__(self, *a, **kw):
        cfg = FakeServiceClient.config
        from tinker_cookbook import tokenizer_utils
        from tinker_cookbook.renderers import get_renderer
        self.model_name = cfg["model_name"]
        self.tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        self.renderer = get_renderer(cfg["renderer_name"], self.tokenizer)
        self.seed, self.skill0, self.skill_steps = cfg.get("seed", 0), cfg.get("skill0", 0.25), cfg.get("skill_steps", 20)
        self.n_samplers = 0
        self.usage = dict(prefill=0, sampled=0, trained=0, sample_calls=0)
        self.checkpoints = []
        FakeServiceClient.last = self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def create_lora_training_client(self, base_model=None, rank=32, **kw):
        return FakeFuture(FakeTrainingClient(self))

    create_lora_training_client_async = create_lora_training_client

    def create_training_client_from_state(self, path, **kw):
        return FakeFuture(FakeTrainingClient(self))

    create_training_client_from_state_with_optimizer = create_training_client_from_state

    def create_sampling_client(self, base_model=None, model_path=None, **kw):
        return _ClientFuture(FakeSamplingClient(self, self.skill0))

    def create_rest_client(self):
        raise tinker.TinkerError("fake service: no REST client")

    def get_server_capabilities(self):
        return SimpleNamespace(supported_models=[])


def install(model_name, renderer_name, **cfg):
    """Monkeypatch tinker.ServiceClient. Returns a function that restores the original."""
    FakeServiceClient.config = dict(model_name=model_name, renderer_name=renderer_name, **cfg)
    orig = tinker.ServiceClient
    tinker.ServiceClient = FakeServiceClient

    def restore():
        tinker.ServiceClient = orig
    return restore
