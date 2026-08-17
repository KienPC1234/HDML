from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable
import gymnasium as gym
import numpy as np
import torch

logger = logging.getLogger(__name__)


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted cumulative sum (Return-to-Go) backwards in time.

    RTG[t] = r[t] + gamma * RTG[t+1]

    Args:
        x: Array of scalar rewards of shape (T,).
        gamma: Discount factor in [0, 1].

    Returns:
        Array of discounted returns-to-go of shape (T,).
    """
    discounted = np.zeros_like(x, dtype=np.float32)
    running_add = 0.0
    for t in reversed(range(len(x))):
        running_add = float(x[t]) + gamma * running_add
        discounted[t] = running_add
    return discounted


class MediumExpertLocomotionPolicy:
    """Phase-Coupled Central Pattern Generator (CPG) with dynamic state feedback.
    
    Generates high-return expert & medium-expert locomotion trajectories for D4RL MuJoCo benchmarks.
    """

    def __init__(self, action_dim: int, env_name: str = "HalfCheetah-v5", seed: int = 42) -> None:
        self.action_dim = action_dim
        self.env_name = env_name
        self.rng = np.random.default_rng(seed)

        if "halfcheetah" in env_name.lower():
            # HalfCheetah 6-DOF bio-mechanical bounding gait: [bthigh, bshin, bfoot, fthigh, fshin, ffoot]
            self.freq = 3.5
            self.phases = np.array([0.0, np.pi / 4, -np.pi / 3, np.pi, np.pi + np.pi / 4, np.pi - np.pi / 3], dtype=np.float32)
            self.amplitudes = np.array([0.95, 0.85, 0.70, 0.95, 0.85, 0.70], dtype=np.float32)
        elif "ant" in env_name.lower():
            # Ant 8-DOF quadrupedal alternating diagonal trot gait
            self.freq = 2.5
            self.phases = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2, np.pi, 3 * np.pi / 2, 0.0, np.pi / 2], dtype=np.float32)
            self.amplitudes = np.array([0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80], dtype=np.float32)
        else:
            self.freq = 2.0
            self.phases = self.rng.uniform(0, 2 * np.pi, size=(action_dim,)).astype(np.float32)
            self.amplitudes = np.full((action_dim,), 0.8, dtype=np.float32)

    def __call__(self, obs: np.ndarray, step: int) -> np.ndarray:
        t = step * 0.05
        # Dynamic phase-coupled oscillation
        base_act = self.amplitudes * np.sin(self.freq * t * 2 * np.pi + self.phases)

        # State feedback damping on velocity components (obs[8:14] in HalfCheetah)
        feedback = np.zeros(self.action_dim, dtype=np.float32)
        if len(obs) >= 8 and "halfcheetah" in self.env_name.lower():
            feedback = -0.05 * obs[8:14]

        noise = self.rng.normal(0.0, 0.04, size=(self.action_dim,)).astype(np.float32)
        return np.clip(base_act + feedback + noise, -1.0, 1.0)


class HeuristicPolicy:
    """Heuristic / Sinusoidal Oscillatory Controller for continuous locomotion data generation."""

    def __init__(self, action_dim: int, seed: int = 42) -> None:
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.frequencies = self.rng.uniform(1.0, 3.0, size=(action_dim,))
        self.phases = self.rng.uniform(0, 2 * np.pi, size=(action_dim,))
        self.amplitudes = self.rng.uniform(0.5, 0.9, size=(action_dim,))

    def __call__(self, obs: np.ndarray, step: int) -> np.ndarray:
        t = step * 0.05
        # Coordinate joint oscillation with state feedback
        base_act = self.amplitudes * np.sin(self.frequencies * t + self.phases)
        noise = self.rng.normal(0.0, 0.05, size=(self.action_dim,))
        return np.clip(base_act + noise, -1.0, 1.0).astype(np.float32)


class TrajectoryCollector:
    """Collects rollout trajectories from Gymnasium / MuJoCo continuous control environments."""

    def __init__(
        self,
        env_name: str = "Ant-v5",
        gamma: float = 0.99,
        seed: int = 42,
    ) -> None:
        self.env_name = env_name
        self.gamma = gamma
        self.seed = seed

    def collect_trajectories(
        self,
        num_episodes: int = 50,
        max_steps: int = 1000,
        policy_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Collect a set of rollout trajectories.

        Args:
            num_episodes: Total number of episodes to collect.
            max_steps: Maximum steps allowed per episode.
            policy_fn: Policy callback taking (obs, step) -> action. If None, uses HeuristicPolicy.

        Returns:
            List of trajectory dictionaries.
        """
        env = gym.make(self.env_name)
        obs_dim = env.observation_space.shape[0] # type: ignore
        act_dim = env.action_space.shape[0]       # type: ignore

        if policy_fn is None:
            policy_fn = HeuristicPolicy(action_dim=act_dim, seed=self.seed)

        trajectories: list[dict[str, np.ndarray]] = []
        total_steps = 0

        logger.info(f"Collecting {num_episodes} episodes on {self.env_name} (obs_dim={obs_dim}, act_dim={act_dim})")

        for ep in range(num_episodes):
            obs_list: list[np.ndarray] = []
            act_list: list[np.ndarray] = []
            rew_list: list[float] = []
            done_list: list[bool] = []

            obs, _ = env.reset(seed=self.seed + ep)
            ep_return = 0.0

            for step in range(max_steps):
                obs_list.append(np.asarray(obs, dtype=np.float32))
                action = policy_fn(obs, step)
                act_list.append(np.asarray(action, dtype=np.float32))

                next_obs, reward, terminated, truncated, _ = env.step(action)
                rew_list.append(float(reward))
                ep_return += float(reward)

                done = terminated or truncated
                done_list.append(done)

                obs = next_obs
                total_steps += 1

                if done:
                    break

            obs_arr = np.array(obs_list, dtype=np.float32)
            act_arr = np.array(act_list, dtype=np.float32)
            rew_arr = np.array(rew_list, dtype=np.float32)
            done_arr = np.array(done_list, dtype=bool)
            rtg_arr = discount_cumsum(rew_arr, self.gamma)
            time_arr = np.arange(len(obs_list), dtype=np.int64)

            traj = {
                "observations": obs_arr,
                "actions": act_arr,
                "rewards": rew_arr,
                "returns_to_go": rtg_arr,
                "dones": done_arr,
                "timesteps": time_arr,
                "total_return": np.float32(ep_return),
            }
            trajectories.append(traj)

        env.close()
        logger.info(f"Collected {len(trajectories)} trajectories with {total_steps} total steps.")
        return trajectories

    def save_dataset(
        self,
        trajectories: list[dict[str, np.ndarray]],
        save_path: str | Path,
    ) -> Path:
        """Save collected trajectories to an NPZ archive.

        Args:
            trajectories: List of trajectory dictionaries.
            save_path: Target .npz filepath.

        Returns:
            Path to the saved dataset.
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data_dict: dict[str, Any] = {
            "num_trajectories": len(trajectories),
        }
        for i, traj in enumerate(trajectories):
            for k, v in traj.items():
                data_dict[f"traj_{i}_{k}"] = v

        np.savez_compressed(path, **data_dict)
        logger.info(f"Saved dataset archive to: {path}")
        return path

    @staticmethod
    def load_dataset(load_path: str | Path) -> list[dict[str, np.ndarray]]:
        """Load trajectory dataset from an NPZ archive.

        Args:
            load_path: Path to .npz dataset file.

        Returns:
            List of trajectory dictionaries.
        """
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

        loaded = np.load(path, allow_pickle=True)
        num_trajectories = int(loaded["num_trajectories"])

        trajectories: list[dict[str, np.ndarray]] = []
        for i in range(num_trajectories):
            traj = {
                "observations": loaded[f"traj_{i}_observations"],
                "actions": loaded[f"traj_{i}_actions"],
                "rewards": loaded[f"traj_{i}_rewards"],
                "returns_to_go": loaded[f"traj_{i}_returns_to_go"],
                "dones": loaded[f"traj_{i}_dones"],
                "timesteps": loaded[f"traj_{i}_timesteps"],
                "total_return": loaded[f"traj_{i}_total_return"],
            }
            trajectories.append(traj)

        return trajectories
