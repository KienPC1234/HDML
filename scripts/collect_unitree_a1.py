#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import numpy as np
import torch

from hdml.data.collector import discount_cumsum
from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CollectUnitreeA1")


class UnitreeA1CPGPolicy:
    """12-DOF Dynamic Phase-Coupled Trot Gait with Balance Feedback for Unitree A1."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.freq = 2.0  # 2 Hz trot
        # Diagonal leg phases: FR & RL in phase 0, FL & RR in phase pi
        self.phases = np.array([0.0, np.pi, np.pi, 0.0], dtype=np.float32)

    def __call__(self, obs: np.ndarray, step: int, noise_level: float = 0.05) -> np.ndarray:
        t = step * 0.02
        # obs: [torso_z, quat_w, quat_x, quat_y, quat_z, ...]
        w, x_q, y_q, z_q = float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4])
        roll = float(np.arctan2(2 * (w * x_q + y_q * z_q), 1 - 2 * (x_q * x_q + y_q * y_q)))
        pitch = float(np.arcsin(np.clip(2 * (w * y_q - z_q * x_q), -1.0, 1.0)))
        yaw = float(np.arctan2(2 * (w * z_q + x_q * y_q), 1 - 2 * (y_q * y_q + z_q * z_q)))

        actions = np.zeros(12, dtype=np.float32)
        for i in range(4):
            phase = self.phases[i]
            hip_stab = -0.12 * roll - 0.05 * yaw
            yaw_diff = 0.08 * yaw if (i in [0, 2]) else -0.08 * yaw
            thigh_osc = 0.22 * np.sin(2.0 * t * 2 * np.pi + phase) + yaw_diff
            calf_osc = 0.22 * np.cos(2.0 * t * 2 * np.pi + phase)
            pitch_corr = -0.2 * pitch if (i in [0, 1]) else 0.2 * pitch

            actions[i * 3 + 0] = np.clip(hip_stab / 0.20, -1.0, 1.0)
            actions[i * 3 + 1] = np.clip((thigh_osc + pitch_corr) / 0.30, -1.0, 1.0)
            actions[i * 3 + 2] = np.clip(calf_osc / 0.30, -1.0, 1.0)

        noise = self.rng.normal(0.0, noise_level, size=actions.shape).astype(np.float32)
        return np.clip(actions + noise, -1.0, 1.0)


def collect_unitree_a1_dataset(
    output_path: str = "data/unitree_a1_trajectories.npz",
    num_episodes: int = 100,
    max_steps_per_episode: int = 500,
    gamma: float = 0.99,
) -> None:
    Path("data").mkdir(parents=True, exist_ok=True)
    env = QuadrupedDogEnv(max_episode_steps=max_steps_per_episode)
    policy = UnitreeA1CPGPolicy(seed=42)

    trajectories: list[dict[str, np.ndarray]] = []
    total_steps = 0
    returns: list[float] = []

    logger.info(f"Starting collection of {num_episodes} Unitree A1 episodes with kicks & disturbances...")

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        ep_obs: list[np.ndarray] = []
        ep_act: list[np.ndarray] = []
        ep_rew: list[float] = []
        ep_term: list[bool] = []
        ep_trunc: list[bool] = []

        noise_lvl = 0.03 if (ep % 2 == 0) else 0.08  # Expert and exploratory

        for step in range(max_steps_per_episode):
            # Apply random kick disturbances during collection (every 70 steps)
            if step % 70 == 0 and step > 0:
                kick_x = float(policy.rng.uniform(-8.0, 8.0))
                kick_y = float(policy.rng.uniform(-10.0, 10.0))
                env.apply_kick((kick_x, kick_y, 0.0))

            action = policy(obs, step, noise_level=noise_lvl)
            next_obs, reward, term, trunc, _ = env.step(action)

            ep_obs.append(obs)
            ep_act.append(action)
            ep_rew.append(reward)
            ep_term.append(term)
            ep_trunc.append(trunc)

            obs = next_obs
            total_steps += 1

            if trunc:
                break

        rewards_arr = np.array(ep_rew, dtype=np.float32)
        rtgs_arr = discount_cumsum(rewards_arr, gamma=gamma)
        ep_return = float(np.sum(rewards_arr))
        returns.append(ep_return)

        traj = {
            "observations": np.array(ep_obs, dtype=np.float32),
            "actions": np.array(ep_act, dtype=np.float32),
            "rewards": rewards_arr,
            "returns_to_go": rtgs_arr,
            "terminals": np.array(ep_term, dtype=bool),
            "timeouts": np.array(ep_trunc, dtype=bool),
        }
        trajectories.append(traj)

        if (ep + 1) % 25 == 0:
            logger.info(f"Collected [{ep+1}/{num_episodes}] episodes | Total frames: {total_steps} | Mean return: {np.mean(returns[-25:]):.2f}")

    env.close()

    # Save to compressed NPZ
    save_dict: dict[str, Any] = {"num_trajectories": len(trajectories)}
    for i, t in enumerate(trajectories):
        for k, v in t.items():
            save_dict[f"traj_{i}_{k}"] = v

    np.savez_compressed(output_path, **save_dict)
    logger.info(f"Dataset successfully saved to: {output_path} ({total_steps} frames, {len(trajectories)} trajectories)")


if __name__ == "__main__":
    collect_unitree_a1_dataset()
