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


def leg_ik(x: float, z: float, l1: float = 0.20, l2: float = 0.20) -> tuple[float, float]:
    """Analytic 2-Link Inverse Kinematics for Unitree A1 thigh and calf."""
    r2 = x**2 + z**2
    r = math.sqrt(r2)
    cos_q2 = max(-1.0, min(1.0, (r2 - l1**2 - l2**2) / (2 * l1 * l2)))
    q2 = -math.acos(cos_q2)  # backward knee
    beta = math.atan2(-x, -z)
    cos_gamma = max(-1.0, min(1.0, r / (2 * l1)))
    gamma = math.acos(cos_gamma)
    q1 = beta + gamma
    return q1, q2


class UnitreeA1CPGPolicy:
    """12-DOF Dynamic Phase-Coupled Raibert Trot Gait with Analytic IK and Balance Feedback for Unitree A1."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.freq = 2.2  # 2.2 Hz dynamic trot
        self.stride = 0.09
        self.height_nom = 0.26
        self.clearance = 0.045
        # Diagonal leg phases: FR (0), FL (0.5), RR (0.5), RL (0.0)
        self.phases = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)

    def __call__(self, obs: np.ndarray, step: int, noise_level: float = 0.05) -> np.ndarray:
        t = step * 0.02
        # obs: [torso_z, quat_w, quat_x, quat_y, quat_z, ...]
        w, x_q, y_q, z_q = float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4])
        roll = float(math.atan2(2 * (w * x_q + y_q * z_q), 1 - 2 * (x_q * x_q + y_q * y_q)))
        pitch = float(math.asin(max(-1.0, min(1.0, 2 * (w * y_q - z_q * x_q)))))
        yaw = float(math.atan2(2 * (w * z_q + x_q * y_q), 1 - 2 * (y_q * y_q + z_q * z_q)))

        actions = np.zeros(12, dtype=np.float32)
        for i in range(4):
            phi = (self.freq * t + self.phases[i]) % 1.0

            if phi < 0.5:
                # Stance phase: smooth linear rearward push on the ground
                s = phi / 0.5
                x_f = self.stride * (0.5 - s)
                z_f = -self.height_nom
            else:
                # Swing phase: smooth elliptical forward step with clearance
                s = (phi - 0.5) / 0.5
                x_f = -self.stride * math.cos(math.pi * s)
                z_f = -self.height_nom + self.clearance * math.sin(math.pi * s)

            q1, q2 = leg_ik(x_f, z_f)

            # Restorative balance control
            hip_stab = -0.10 * roll - 0.05 * yaw
            pitch_corr = -0.15 * pitch if (i in [0, 1]) else 0.15 * pitch

            # Map to normalized actions around base (q1 base 0.85, q2 base -1.80)
            actions[i * 3 + 0] = float(np.clip(hip_stab / 0.20, -1.0, 1.0))
            actions[i * 3 + 1] = float(np.clip(((q1 + pitch_corr) - 0.85) / 0.30, -1.0, 1.0))
            actions[i * 3 + 2] = float(np.clip((q2 - (-1.80)) / 0.30, -1.0, 1.0))

        noise = self.rng.normal(0.0, noise_level, size=actions.shape).astype(np.float32)
        return np.clip(actions + noise, -1.0, 1.0)


def collect_unitree_a1_dataset(
    output_path: str = "data/unitree_a1_trajectories.npz",
    num_episodes: int = 200,
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
