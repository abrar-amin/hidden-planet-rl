"""Mock sampler: a tinker_cookbook TokenCompleter that needs no API and spends no credits.

It decodes the prompt, works out where the episode is from the tool outputs in the text, and emits
the next assistant turn as real tokens in the renderer's tool-call format. `skill` in [0, 1]
interpolates from a sloppy policy (malformed calls, overspending, random answers) to the compact
scripted expert, so dry runs produce a non-trivial, improvable reward signal.
"""
from __future__ import annotations

import json
import re

import numpy as np
import tinker
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.renderers.base import ToolCall

from .tasks import NMAX
from .text_policies import parse_task


class MockCompleter(TokenCompleter):
    def __init__(self, renderer, tokenizer, skill: float = 1.0, seed: int = 0):
        self.renderer, self.tok, self.skill = renderer, tokenizer, skill
        self.rng = np.random.default_rng(seed)
        self._dummy = [{"role": "user", "content": "x"}]
        self._plen = len(renderer.build_generation_prompt(self._dummy).to_ints())
        self.calls = 0

    def _encode(self, thought, calls):
        msg = {"role": "assistant", "content": thought, "tool_calls": [
            ToolCall(function=ToolCall.FunctionBody(name=n, arguments=json.dumps(a)), id=f"call_{self.calls}_{i}")
            for i, (n, a) in enumerate(calls)]}
        full, _ = self.renderer.build_supervised_example(self._dummy + [msg])
        return [int(t) for t in full.to_ints()][self._plen:]

    def _decide(self, text):
        rng = self.rng
        task = text[text.rindex("Target "):]
        T = parse_task(task)
        if rng.random() > self.skill:                       # sloppy move
            r = rng.random()
            if r < 0.25:
                return "", [("submit_answer", {"verdict": ["none", "cannot_determine"][int(rng.integers(2))]})]
            if r < 0.45:
                return "", [("observe_rvs", {"times_days": [round(float(t), 1) for t in rng.uniform(0, T["t_max"], 4)]})]
            if r < 0.6:
                return "I think the answer is none.", []      # forgets to call submit_answer
            if r < 0.7:
                return "", [("observe_transit", {"epoch": 3})]  # wrong tool name
        n_tr = task.count('"epoch_and_oc_min"'); n_rv = task.count('"time_and_rv_ms"')
        if n_tr == 0:
            ep = np.unique(np.clip(np.linspace(0, NMAX, 14).round().astype(int) + rng.integers(-4, 5, 14), 0, NMAX))
            return "Sample the transit times across the baseline.", [("observe_transits", {"epochs": [int(x) for x in ep]})]
        if '"peaks"' not in task:
            return "Look for a timing signal.", [("fit_transit_timing", {})]
        pk = task[task.rindex('"peaks"'):]
        dchi = float(re.search(r'"delta_chi2":(-?[\d.]+)', pk).group(1))
        al = re.findall(r'"P2_days":([\d.]+),"mass_if_circular":([\d.]+)', pk.split('"super_period_epochs"')[1])
        if dchi < 25 or not al:
            return "No timing signal.", [("submit_answer", {"verdict": "none"})]
        if "RVs are NOT possible" in task:
            return "RVs are impossible; the alias cannot be resolved.", [("submit_answer", {"verdict": "cannot_determine"})]
        if n_rv == 0:
            return "Get RVs.", [("observe_rvs", {"times_days": [round(float(t), 2) for t in np.sort(rng.uniform(0, T["t_max"], 5))]})]
        last_rv = task.rindex('"time_and_rv_ms"')
        if '"fits"' not in task[last_rv:]:
            return "Compare aliases.", [("fit_rv", {"periods_days": [float(a[0]) for a in al]})]
        fits = re.findall(r'"P2_days":([\d.]+),"K2_ms":([\d.]+),"K2_sigma_ms":([\d.]+),"delta_chi2":(-?[\d.]+),"mass_earth":([\d.]+)', task[task.rindex('"fits"'):])
        if not fits:
            return "The RV fit failed.", [("submit_answer", {"verdict": "cannot_determine"})]
        fits = sorted(fits, key=lambda f: -float(f[3]))
        margin = float(fits[0][3]) - (float(fits[1][3]) if len(fits) > 1 else 0.0)
        if (float(fits[0][3]) > 12 and margin > 6) or n_rv >= 2:
            if float(fits[0][3]) > 12 and margin > 6:
                return "Alias resolved.", [("submit_answer", {"verdict": "planet", "period_days": float(fits[0][0]), "mass_earth": float(fits[0][4])})]
            return "Cannot separate the aliases.", [("submit_answer", {"verdict": "cannot_determine"})]
        return "More RVs.", [("observe_rvs", {"times_days": [round(float(t), 2) for t in np.sort(rng.uniform(0, T["t_max"], 4))]})]

    async def __call__(self, model_input: tinker.ModelInput, stop: StopCondition, *, max_tokens: int | None = None) -> TokensWithLogprobs:
        self.calls += 1
        text = self.tok.decode(model_input.to_ints())
        thought, calls = self._decide(text)
        toks = self._encode(thought, calls)
        return TokensWithLogprobs(tokens=toks, maybe_logprobs=[-0.5] * len(toks), stop_reason="stop")
