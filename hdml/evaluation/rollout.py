"""Shared closed-loop rollout utilities for the Unitree A1 MuJoCo env.

The benchmark, plotting, and recording scripts all drive
:class:`~hdml.evaluation.quadruped_dog_env.QuadrupedDogEnv` with the same
history/return-to-go bookkeeping. This module centralises that bookkeeping so
the scripts differ only in how they select an action.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import HDMLConfig
from hdml.utils.metrics import compute_action_smoothness

PolicyStep = Callable[
    [torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Any,
]


def torso_roll(obs: np.ndarray) -> float:
    """Torso roll angle in degrees from a Unitree A1 quaternion observation."""
    w, x, y, z = float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4])
    return float(np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))))


def leg_ik(x: float, z: float, l1: float = 0.20, l2: float = 0.20) -> tuple[float, float]:
    """Analytic 2-link inverse kinematics for the Unitree A1 thigh and calf."""
    r2 = x**2 + z**2
    r = math.sqrt(r2)
    cos_q2 = max(-1.0, min(1.0, (r2 - l1**2 - l2**2) / (2 * l1 * l2)))
    q2 = -math.acos(cos_q2)  # backward knee
    beta = math.atan2(-x, -z)
    cos_gamma = max(-1.0, min(1.0, r / (2 * l1)))
    gamma = math.acos(cos_gamma)
    q1 = beta + gamma
    return q1, q2


def load_hdml_checkpoint(
    path: str,
    cfg: HDMLConfig,
    device: torch.device,
) -> tuple[HDMLModel, np.ndarray, np.ndarray]:
    """Load an HDML checkpoint for deterministic closed-loop evaluation."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = HDMLModel.from_config(cfg.model).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    model.deterministic = True
    model.num_flow_steps = 4
    return model, np.asarray(ck["state_mean"], np.float32), np.asarray(ck["state_std"], np.float32)


def to_action(out: Any) -> np.ndarray:
    """Normalise the various baseline/HDML outputs to a clipped (action_dim,) array."""
    a = out[0] if isinstance(out, tuple) else out
    a = a[0, 0, :] if a.ndim == 3 else a[0, :]
    return np.clip(a.detach().cpu().numpy().astype(np.float32), -1.0, 1.0)


class UnitreeRolloutContext:
    """Maintains the observation/action/RTG history for one episode."""

    def __init__(
        self,
        cfg: HDMLConfig,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = device
        self.obs_history: list[np.ndarray] = []
        self.action_history: list[np.ndarray] = []
        self.rtg_history: list[float] = []
        self.time_history: list[int] = []
        self.remaining_return = float(cfg.env.target_return)
        self.actions: list[np.ndarray] = []

    def update(self, obs: np.ndarray, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Append the new observation/RTG/timestep and return context tensors."""
        norm_obs = (np.asarray(obs, np.float32) - self.state_mean) / self.state_std
        self.obs_history.append(norm_obs)
        self.rtg_history.append(self.remaining_return / self.cfg.env.scale_return)
        self.time_history.append(step)
        if not self.action_history:
            self.action_history.append(np.zeros(self.cfg.model.action_dim, np.float32))
        context_len = min(len(self.obs_history), self.cfg.training.context_length)
        states = torch.from_numpy(
            np.array(self.obs_history[-context_len:], np.float32)
        ).unsqueeze(0).to(self.device)
        actions = torch.from_numpy(
            np.array(self.action_history[-context_len:], np.float32)
        ).unsqueeze(0).to(self.device)
        rtgs = torch.from_numpy(
            np.array(self.rtg_history[-context_len:], np.float32).reshape(-1, 1)
        ).unsqueeze(0).to(self.device)
        timesteps = torch.from_numpy(
            np.array(self.time_history[-context_len:], np.int64)
        ).unsqueeze(0).to(self.device)
        return states, actions, rtgs, timesteps

    def record_action(self, action: np.ndarray) -> None:
        self.action_history.append(action)
        self.actions.append(action)

    def consume_reward(self, reward: float) -> None:
        self.remaining_return -= float(reward)


@dataclass
class RolloutResult:
    """Per-episode metrics from :func:`run_rollout`."""

    returns: list[float] = field(default_factory=list)
    jerks: list[float] = field(default_factory=list)
    completions: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)


def run_rollout(
    model: torch.nn.Module,
    cfg: HDMLConfig,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
    episodes: int,
    policy_step: PolicyStep,
    *,
    seed_base: int = 42,
    manual_torch_seed: bool = False,
    measure_latency: bool = False,
    max_steps: int | None = None,
) -> RolloutResult:
    """Run closed-loop episodes and aggregate return/jerk/completion metrics.

    Behaviour matches the historical per-script loops: history is truncated to
    ``cfg.training.context_length``, RTG starts at ``cfg.env.target_return`` and
    is decremented by the raw reward, and an episode stops on ``terminated`` or
    ``truncated``. ``policy_step`` receives ``(model, states, actions, rtgs,
    timesteps)`` and returns any tensor/tuple accepted by :func:`to_action`.
    """
    env = QuadrupedDogEnv(max_episode_steps=max_steps or cfg.env.max_episode_steps)
    result = RolloutResult()
    try:
        for ep in range(episodes):
            if manual_torch_seed:
                torch.manual_seed(seed_base + ep)
            obs, _ = env.reset(seed=seed_base + ep)
            context = UnitreeRolloutContext(cfg, state_mean, state_std, device)
            episode_return = 0.0
            steps = 0
            for t in range(max_steps or cfg.env.max_episode_steps):
                states, actions, rtgs, timesteps = context.update(obs, t)
                if measure_latency and device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = policy_step(model, states, actions, rtgs, timesteps)
                if measure_latency:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    result.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                action = to_action(out)
                context.record_action(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                context.consume_reward(float(reward))
                steps += 1
                if terminated or truncated:
                    break
            result.returns.append(episode_return)
            result.jerks.append(compute_action_smoothness(np.array(context.actions, np.float32)))
            result.completions.append(1.0 if steps >= (max_steps or cfg.env.max_episode_steps) else 0.0)
            result.actions.append(np.array(context.actions, np.float32))
    finally:
        env.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result
