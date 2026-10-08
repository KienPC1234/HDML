#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import numpy as np

from hdml.data.collector import discount_cumsum
from hdml.evaluation.unitree_a1_maze_env import UnitreeA1MazeEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CollectUnitreeA1Maze")


def leg_ik(x: float, z: float, l1: float = 0.20, l2: float = 0.20) -> tuple[float, float]:
    """Analytic 2-Link Inverse Kinematics for Unitree A1 thigh and calf."""
    r2 = x**2 + z**2
    r = math.sqrt(r2)
    cos_q2 = max(-1.0, min(1.0, (r2 - l1**2 - l2**2) / (2 * l1 * l2)))
    q2 = -math.acos(cos_q2)
    beta = math.atan2(-x, -z)
    cos_gamma = max(-1.0, min(1.0, r / (2 * l1)))
    gamma = math.acos(cos_gamma)
    q1 = beta + gamma
    return q1, q2


class TopologicalMazeRouter:
    """Computes collision-free corridor waypoint path through the hard labyrinth graph."""

    CORRIDOR_NODES = {
        "SW": (-2.8, -2.8),
        "MW": (-2.8, 0.0),
        "NW": (-2.8, 2.8),
        "NC": (0.0, 2.8),
        "NE": (2.8, 2.8),
        "ME": (2.8, 0.0),
        "SE": (2.8, -2.8),
        "SC": (0.0, -2.8),
        "CC": (0.0, 0.0),
    }

    GRAPH = {
        "SW": ["MW", "SC"],
        "MW": ["SW", "NW"],
        "NW": ["MW", "NC"],
        "NC": ["NW", "NE"],
        "NE": ["NC", "ME"],
        "ME": ["NE", "SE"],
        "SE": ["ME", "SC"],
        "SC": ["SE", "SW"],
        "CC": ["NC", "SC"],
    }

    @classmethod
    def get_closest_node(cls, pos: tuple[float, float] | np.ndarray) -> str:
        best_node = "SW"
        best_dist = float("inf")
        for name, p in cls.CORRIDOR_NODES.items():
            d = (pos[0] - p[0])**2 + (pos[1] - p[1])**2
            if d < best_dist:
                best_dist = d
                best_node = name
        return best_node

    @classmethod
    def plan_path(cls, start: tuple[float, float] | np.ndarray, goal: tuple[float, float] | np.ndarray) -> list[tuple[float, float]]:
        start_node = cls.get_closest_node(start)
        goal_node = cls.get_closest_node(goal)

        if start_node == goal_node:
            return [tuple(goal)]

        queue = [[start_node]]
        visited = set()

        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == goal_node:
                waypoints = [cls.CORRIDOR_NODES[n] for n in path]
                waypoints.append((float(goal[0]), float(goal[1])))
                return waypoints

            if node not in visited:
                visited.add(node)
                for neighbor in cls.GRAPH.get(node, []):
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

        return [tuple(goal)]


class TopologicalQuadrupedPolicy:
    """12-DOF dynamic quadruped trot with corridor graph waypoint tracking and wall avoidance."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.freq = 2.6
        self.stride = 0.16
        self.height_nom = 0.26
        self.clearance = 0.052
        self.phases = np.array([0.0, 0.5, 0.5, 0.0], dtype=np.float32)

    def __call__(
        self,
        obs: np.ndarray,
        step: int,
        current_pos: np.ndarray,
        waypoints: list[tuple[float, float]],
        wp_idx: int,
        noise_level: float = 0.015,
    ) -> tuple[np.ndarray, int]:
        t = step * 0.02
        w, x_q, y_q, z_q = float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4])
        roll = float(math.atan2(2 * (w * x_q + y_q * z_q), 1 - 2 * (x_q * x_q + y_q * y_q)))
        pitch = float(math.asin(max(-1.0, min(1.0, 2 * (w * y_q - z_q * x_q)))))
        yaw = float(math.atan2(2 * (w * z_q + x_q * y_q), 1 - 2 * (y_q * y_q + z_q * z_q)))

        target_wp = waypoints[min(wp_idx, len(waypoints) - 1)]
        dist_to_wp = math.sqrt((target_wp[0] - current_pos[0])**2 + (target_wp[1] - current_pos[1])**2)
        if dist_to_wp < 0.80 and wp_idx < len(waypoints) - 1:
            wp_idx += 1
            target_wp = waypoints[wp_idx]

        dx_w = target_wp[0] - current_pos[0]
        dy_w = target_wp[1] - current_pos[1]
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        dx_r = dx_w * cos_y + dy_w * sin_y
        dy_r = -dx_w * sin_y + dy_w * cos_y
        target_heading = math.atan2(dy_r, dx_r)

        lidar = obs[35:51]
        front_dist = float(np.min(lidar[7:10]))
        left_dist = float(np.min(lidar[10:13]))
        right_dist = float(np.min(lidar[3:6]))

        # Wall centering
        wall_steer = 0.0
        if right_dist < 0.70:
            wall_steer += (0.70 - right_dist) * 0.8
        if left_dist < 0.70:
            wall_steer -= (0.70 - left_dist) * 0.8

        if front_dist < 0.70:
            # Turn towards active waypoint heading
            steer = 0.85 if target_heading > 0 else -0.85
        else:
            steer = float(np.clip(target_heading * 0.85 + wall_steer, -0.75, 0.75))

        stride_fwd = self.stride * max(0.35, 1.0 - 0.40 * abs(steer))

        actions = np.zeros(12, dtype=np.float32)
        for i in range(4):
            phi = (self.freq * t + self.phases[i]) % 1.0
            is_left = (i in [1, 3])
            leg_stride = stride_fwd * (1.0 - 0.45 * steer if is_left else 1.0 + 0.45 * steer)

            if phi < 0.5:
                s = phi / 0.5
                x_f = leg_stride * (0.5 - s)
                z_f = -self.height_nom
            else:
                s = (phi - 0.5) / 0.5
                x_f = -leg_stride * math.cos(math.pi * s)
                z_f = -self.height_nom + self.clearance * math.sin(math.pi * s)

            q1, q2 = leg_ik(x_f, z_f)

            hip_steer = 0.10 * steer if (i in [0, 2]) else -0.10 * steer
            hip_stab = -0.70 * roll + hip_steer
            pitch_corr = -0.20 * pitch if (i in [0, 1]) else 0.20 * pitch

            actions[i * 3 + 0] = float(np.clip(hip_stab / 0.20, -1.0, 1.0))
            actions[i * 3 + 1] = float(np.clip(((q1 + pitch_corr) - 0.85) / 0.30, -1.0, 1.0))
            actions[i * 3 + 2] = float(np.clip((q2 - (-1.80)) / 0.30, -1.0, 1.0))

        if noise_level > 0:
            actions += self.rng.normal(0.0, noise_level, size=12).astype(np.float32)

        return np.clip(actions, -1.0, 1.0), wp_idx


def collect_dynamic_goal_trajectories(
    num_episodes: int = 80,
    max_steps: int = 900,
    her_prob: float = 0.5,
    output_path: str = "data/unitree_a1_maze_trajectories.npz",
    seed: int = 42,
) -> None:
    """Collect diverse trajectories with dynamic random pillar relocation across all free corridor spaces."""
    rng = np.random.default_rng(seed)
    env = UnitreeA1MazeEnv(max_episode_steps=max_steps, sensor_noise=0.02)
    policy = TopologicalQuadrupedPolicy(seed=seed)

    trajectories: list[dict[str, np.ndarray]] = []
    total_steps = 0
    goals_reached = 0

    logger.info(f"Starting dynamic goal pillar data collection for {num_episodes} episodes...")

    # Comprehensive route distribution: All 8 critical long-horizon corridor paths + random goals
    long_horizon_pairs = [
        ((-2.8, -2.8), (2.8, 2.8)),  # SW -> NE (Full Labyrinth Solved)
        ((-2.8, 2.8), (2.8, -2.8)),  # NW -> SE (Full Labyrinth Solved)
        ((2.8, -2.8), (-2.8, 2.8)),  # SE -> NW (Full Labyrinth Solved)
        ((2.8, 2.8), (-2.8, -2.8)),  # NE -> SW (Full Labyrinth Solved)
        ((-2.8, -2.8), (-2.8, 2.8)), # SW -> NW (West Corridor)
        ((2.8, -2.8), (2.8, 2.8)),   # SE -> NE (East Corridor)
        ((-2.8, 2.8), (2.8, 2.8)),   # NW -> NE (North Corridor)
        ((-2.8, -2.8), (2.8, -2.8)), # SW -> SE (South Corridor)
    ]

    waypoints_pool = env.FREE_SPACE_WAYPOINTS

    for ep in range(num_episodes):
        if ep % 3 != 0:
            # 67% Primary long-horizon labyrinth corridor routes
            pair = long_horizon_pairs[ep % len(long_horizon_pairs)]
            start_pos = (float(pair[0][0] + rng.uniform(-0.1, 0.1)), float(pair[0][1] + rng.uniform(-0.1, 0.1)))
            goal_pos = (float(pair[1][0] + rng.uniform(-0.1, 0.1)), float(pair[1][1] + rng.uniform(-0.1, 0.1)))
        else:
            # 33% Dynamic random goal pillar relocation across all chambers
            start_idx = int(rng.integers(0, len(waypoints_pool)))
            goal_idx = int((start_idx + rng.integers(1, len(waypoints_pool))) % len(waypoints_pool))
            start_pt = waypoints_pool[start_idx]
            goal_pt = waypoints_pool[goal_idx]
            start_pos = (float(start_pt[0] + rng.uniform(-0.15, 0.15)), float(start_pt[1] + rng.uniform(-0.15, 0.15)))
            goal_pos = (float(goal_pt[0] + rng.uniform(-0.15, 0.15)), float(goal_pt[1] + rng.uniform(-0.15, 0.15)))

        env.start_pos = np.array(start_pos, dtype=np.float32)
        obs, info = env.reset(seed=seed + ep, options={"goal": goal_pos})

        waypoints = TopologicalMazeRouter.plan_path(start_pos, goal_pos)
        wp_idx = 0

        ep_obs: list[np.ndarray] = []
        ep_act: list[np.ndarray] = []
        ep_rew: list[float] = []
        ep_pos: list[np.ndarray] = []

        for step in range(max_steps):
            pos = np.array([env.data.qpos[0], env.data.qpos[1]], dtype=np.float32)
            act, wp_idx = policy(obs, step, pos, waypoints, wp_idx, noise_level=0.015)
            next_obs, rew, terminated, truncated, step_info = env.step(act)

            ep_obs.append(obs)
            ep_act.append(act)
            ep_rew.append(float(rew))
            ep_pos.append(pos.copy())

            obs = next_obs
            if terminated or truncated:
                if step_info.get("goal_reached", False):
                    goals_reached += 1
                break

        ep_len = len(ep_obs)
        total_steps += ep_len

        obs_arr = np.array(ep_obs, dtype=np.float32)
        act_arr = np.array(ep_act, dtype=np.float32)
        rew_arr = np.array(ep_rew, dtype=np.float32)
        timesteps = np.arange(ep_len, dtype=np.int64)

        sparse_rews = np.zeros(ep_len, dtype=np.float32)
        if step_info.get("goal_reached", False):
            sparse_rews[-1] = 1.0

        rtg = discount_cumsum(sparse_rews, gamma=0.99)
        # Target return normalized to 1.0 for solved trajectories
        if not step_info.get("goal_reached", False):
            # Dense distance progress fallback
            progress_rews = rew_arr / 100.0
            rtg = discount_cumsum(progress_rews, gamma=0.99)

        trajectories.append({
            "observations": obs_arr,
            "actions": act_arr,
            "rewards": rew_arr,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "terminals": np.zeros(ep_len, dtype=bool),
        })

    env.close()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_obs = np.concatenate([t["observations"] for t in trajectories], axis=0)
    state_mean = np.mean(all_obs, axis=0, dtype=np.float32)
    state_std = np.std(all_obs, axis=0, dtype=np.float32) + 1e-6

    # Zero-mean isotropic scaling for Relative Goal Vector to strictly preserve 2D vector angles
    state_mean[51:53] = 0.0
    state_std[51:53] = 4.0

    # Symmetric linear scaling for LiDAR rays [0.1m - 4.0m]
    state_mean[35:51] = 2.0
    state_std[35:51] = 2.0

    save_dict: dict[str, np.ndarray] = {
        "num_trajectories": np.array(len(trajectories), dtype=np.int64),
        "state_mean": state_mean,
        "state_std": state_std,
    }
    for i, t in enumerate(trajectories):
        for k, v in t.items():
            save_dict[f"traj_{i}_{k}"] = v

    np.savez_compressed(out, **save_dict)
    logger.info(
        f"Saved {len(trajectories)} trajectories ({total_steps} frames) to {out}. "
        f"Goals reached naturally: {goals_reached}/{num_episodes} ({(goals_reached/num_episodes)*100:.1f}%)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Dynamic Goal Hard Labyrinth trajectories.")
    parser.add_argument("--num-episodes", type=int, default=80, help="Number of episodes to collect")
    parser.add_argument("--max-steps", type=int, default=900, help="Max steps per episode")
    parser.add_argument("--her-prob", type=float, default=0.5, help="Hindsight Goal Relabeling probability")
    parser.add_argument("--output", type=str, default="data/unitree_a1_maze_trajectories.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    collect_dynamic_goal_trajectories(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        her_prob=args.her_prob,
        output_path=args.output,
        seed=args.seed,
    )
