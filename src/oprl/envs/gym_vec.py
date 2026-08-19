"""EnvAdapter implementations. Contract: tensors in and out, already on `self.device`.

Gymnasium autoreset is the biggest correctness trap here (DESIGN.md §4.2):
  - NEXT_STEP (the default since 1.0): the step **after** a termination is a dummy
    step -- its action is discarded and its reward is 0.
  - SAME_STEP (pre-1.0): the terminal step's obs is already the reset one, and the
    true final obs is tucked into `info`.

This module absorbs those differences so the `valid` mask seen by algorithms is
always correct.
"""

from __future__ import annotations

import numpy as np
import torch

from ..types import Masks, Obs


def _to_t(x, device, dtype=None) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x), device=device)
    return t.to(dtype) if dtype else t


class GymVecAdapter:
    """Wraps a `gymnasium.vector.VectorEnv`."""

    def __init__(self, venv, device: torch.device | str = "cpu"):
        self.venv = venv
        self.device = torch.device(device)
        self.num_envs = int(venv.num_envs)
        self.obs_space = venv.single_observation_space
        self.action_space = venv.single_action_space

        mode = str(venv.metadata.get("autoreset_mode", "NextStep"))
        self.next_step_autoreset = "Next" in mode or "next" in mode
        # Which envs finished last step -> this step is a dummy (NEXT_STEP mode only).
        self._prev_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ep_ret = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_len = np.zeros(self.num_envs, dtype=np.int64)
        self._discrete = type(self.action_space).__name__ in (
            "Discrete", "MultiDiscrete", "MultiBinary"
        )

    def _obs(self, obs) -> Obs:
        if isinstance(obs, dict):
            return {k: _to_t(v, self.device) for k, v in obs.items()}
        return _to_t(obs, self.device)

    def reset(self, seed: int | None = None) -> Obs:
        obs, _ = self.venv.reset(seed=seed)
        self._prev_done.zero_()
        self._ep_ret[:] = 0.0
        self._ep_len[:] = 0
        return self._obs(obs)

    def step(self, action: torch.Tensor):
        a = action.detach().cpu().numpy()
        if self._discrete:
            a = a.astype(np.int64)
        obs, reward, terminated, truncated, info = self.venv.step(a)

        rew = _to_t(reward, self.device, torch.float32)
        term = _to_t(terminated, self.device, torch.bool)
        trunc = _to_t(truncated, self.device, torch.bool)

        # Under NEXT_STEP, envs that finished last step produce a dummy step now, which
        # must be excluded from the loss. Failing to mask it means computing a policy
        # gradient for an action that was never executed: no crash, no divergence, just a
        # gradient biased in proportion to the termination rate.
        if self.next_step_autoreset:
            valid = ~self._prev_done
            self._prev_done = term | trunc
        else:
            valid = torch.ones_like(term)

        # Episode statistics must use the raw, un-normalized reward.
        v = valid.cpu().numpy()
        self._ep_ret += np.asarray(reward, dtype=np.float64) * v
        self._ep_len += v.astype(np.int64)
        done_np = (term | trunc).cpu().numpy()
        finished = []
        for i in np.nonzero(done_np)[0]:
            finished.append((float(self._ep_ret[i]), int(self._ep_len[i])))
            self._ep_ret[i] = 0.0
            self._ep_len[i] = 0
        info = dict(info) if isinstance(info, dict) else {}
        info["_finished_episodes"] = finished

        return self._obs(obs), rew, Masks(term, trunc, valid), info

    def close(self) -> None:
        self.venv.close()


class TensorEnvAdapter:
    """GPU-native environments (Isaac Lab / Brax / MJX) -- zero-copy passthrough.

    These already return device tensors; routing them through numpy would defeat the
    entire performance argument.
    """

    def __init__(self, env, device=None):
        self.env = env
        self.device = torch.device(device or getattr(env, "device", "cuda"))
        self.num_envs = int(env.num_envs)
        self.obs_space = env.single_observation_space
        self.action_space = env.single_action_space

    def reset(self, seed: int | None = None) -> Obs:
        out = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        return out[0] if isinstance(out, tuple) else out

    def step(self, action: torch.Tensor):
        obs, rew, term, trunc, info = self.env.step(action)
        valid = torch.ones_like(term, dtype=torch.bool)
        if isinstance(info, dict) and "_finished_episodes" not in info:
            info = {**info, "_finished_episodes": []}
        return obs, rew, Masks(term.bool(), trunc.bool(), valid), info

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()
