#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from hdml.data.collector import TrajectoryCollector, MediumExpertLocomotionPolicy, HeuristicPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect demonstration trajectories for HDML training.")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4", help="Gymnasium environment ID")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to collect")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--policy-type", type=str, default="medium_expert", choices=["medium_expert", "heuristic"])
    parser.add_argument("--output", type=str, default="data/halfcheetah_v4_trajectories.npz", help="Output .npz path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collector = TrajectoryCollector(env_name=args.env, seed=args.seed)

    if args.policy_type == "medium_expert":
        import gymnasium as gym
        temp_env = gym.make(args.env)
        act_dim = temp_env.action_space.shape[0]  # type: ignore
        temp_env.close()
        policy = MediumExpertLocomotionPolicy(action_dim=act_dim, env_name=args.env, seed=args.seed)
    else:
        policy = None

    trajectories = collector.collect_trajectories(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        policy_fn=policy,
    )
    out_path = collector.save_dataset(trajectories, args.output)
    logger.info(f"Dataset successfully created at: {out_path}")


if __name__ == "__main__":
    main()
