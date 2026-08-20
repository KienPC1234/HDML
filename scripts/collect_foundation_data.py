"""Collect high-quality expert trajectory data from diverse MuJoCo environments.

Generates offline RL datasets for HDML-Foundation pre-training using
random exploration with reward filtering to retain only high-quality trajectories.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import gymnasium as gym

logger = logging.getLogger("CollectFoundationData")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(sh)


ENVS = {
    "ant": {
        "env_id": "Ant-v5",
        "num_episodes": 500,
        "max_steps": 1000,
        "keep_top_frac": 0.6,
    },
    "hopper": {
        "env_id": "Hopper-v5",
        "num_episodes": 500,
        "max_steps": 1000,
        "keep_top_frac": 0.6,
    },
    "walker2d": {
        "env_id": "Walker2d-v5",
        "num_episodes": 500,
        "max_steps": 1000,
        "keep_top_frac": 0.6,
    },
    "humanoid": {
        "env_id": "Humanoid-v5",
        "num_episodes": 400,
        "max_steps": 1000,
        "keep_top_frac": 0.6,
    },
    "swimmer": {
        "env_id": "Swimmer-v5",
        "num_episodes": 500,
        "max_steps": 1000,
        "keep_top_frac": 0.6,
    },
    "reacher": {
        "env_id": "Reacher-v5",
        "num_episodes": 500,
        "max_steps": 200,
        "keep_top_frac": 0.6,
    },
    "inv_double_pendulum": {
        "env_id": "InvertedDoublePendulum-v5",
        "num_episodes": 500,
        "max_steps": 1000,
        "keep_top_frac": 0.6,
    },
    "humanoid_standup": {
        "env_id": "HumanoidStandup-v5",
        "num_episodes": 400,
        "max_steps": 1000,
        "keep_top_frac": 0.6,
    },
}


def collect_env_data(
    env_id: str,
    num_episodes: int,
    max_steps: int,
    keep_top_frac: float,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Collect trajectory data from a single environment."""
    env = gym.make(env_id)
    trajectories: list[dict[str, list]] = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        traj: dict[str, list] = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminals": [],
        }
        ep_return = 0.0

        for step in range(max_steps):
            action = env.action_space.sample()
            traj["observations"].append(obs.astype(np.float32))
            traj["actions"].append(action.astype(np.float32))

            obs, reward, terminated, truncated, info = env.step(action)
            traj["rewards"].append(float(reward))
            traj["terminals"].append(terminated)
            ep_return += float(reward)

            if terminated or truncated:
                break

        traj["total_return"] = ep_return
        traj["length"] = len(traj["rewards"])
        trajectories.append(traj)

    env.close()

    # Sort by return and keep top fraction
    trajectories.sort(key=lambda t: t["total_return"], reverse=True)
    n_keep = max(1, int(len(trajectories) * keep_top_frac))
    kept = trajectories[:n_keep]

    # Build npz archive
    archive: dict[str, np.ndarray] = {"num_trajectories": np.array(len(kept))}
    total_steps = 0
    for i, traj in enumerate(kept):
        archive[f"traj_{i}_observations"] = np.array(traj["observations"], dtype=np.float32)
        archive[f"traj_{i}_actions"] = np.array(traj["actions"], dtype=np.float32)
        archive[f"traj_{i}_rewards"] = np.array(traj["rewards"], dtype=np.float32)
        archive[f"traj_{i}_terminals"] = np.array(traj["terminals"], dtype=bool)
        total_steps += traj["length"]

    returns = [t["total_return"] for t in kept]
    logger.info(
        f"  {env_id}: Kept {n_keep}/{num_episodes} trajs, "
        f"total_steps={total_steps:,}, "
        f"return={np.mean(returns):.1f}±{np.std(returns):.1f}"
    )
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Foundation Pre-training Data")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--envs", type=str, nargs="*", default=None,
                        help="Specific envs to collect (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = args.envs if args.envs else list(ENVS.keys())
    logger.info(f"Collecting data for {len(targets)} environments: {targets}")

    for name in targets:
        if name not in ENVS:
            logger.warning(f"Unknown environment '{name}', skipping.")
            continue

        cfg = ENVS[name]
        logger.info(f"Collecting {cfg['env_id']} ({cfg['num_episodes']} episodes)...")

        archive = collect_env_data(
            env_id=cfg["env_id"],
            num_episodes=cfg["num_episodes"],
            max_steps=cfg["max_steps"],
            keep_top_frac=cfg["keep_top_frac"],
            seed=args.seed,
        )

        out_path = out_dir / f"{name}_foundation.npz"
        np.savez_compressed(out_path, **archive)
        logger.info(f"  Saved to {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    logger.info("Data collection complete!")


if __name__ == "__main__":
    main()
