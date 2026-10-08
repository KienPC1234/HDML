#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from hdml.utils.config import HDMLConfig
from hdml.data.dataset import MinariDatasetAdapter
from hdml.data.collector import TrajectoryCollector
from hdml.models.hdml_model import HDMLModel
from hdml.utils.metrics import compute_action_smoothness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate unsupervised cognitive representation and spatial navigation.")
    parser.add_argument("--config", type=str, default="configs/pointmaze_umaze_unsupervised.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pointmaze_umaze/best_model.pt")
    parser.add_argument("--dataset", type=str, default="D4RL/pointmaze/umaze-v2")
    parser.add_argument("--num-eval-trajs", "--episodes", dest="num_eval_trajs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def evaluate_cognition(
    model: HDMLModel,
    trajectories: list[dict[str, np.ndarray]],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    context_length: int = 20,
    device: torch.device = torch.device("cuda"),
) -> dict[str, float]:
    """Evaluate self-awareness, forward world model accuracy, and action smoothness."""
    model.eval()
    dyn_errors: list[float] = []
    subgoal_dist_errors: list[float] = []
    all_jerks: list[float] = []
    goal_reaching_distances: list[float] = []

    for traj in trajectories:
        obs = traj["observations"]  # (T, 6) -> [x, y, vx, vy, gx, gy]
        actions = traj["actions"]   # (T, 2)
        if "next_states" in traj:
            next_states = traj["next_states"]
        elif "next_observations" in traj:
            next_states = traj["next_observations"]
        else:
            next_states = np.concatenate([obs[1:], obs[-1:]], axis=0)

        t_len = len(obs)
        if t_len < context_length:
            continue

        norm_obs = (obs - state_mean) / state_std
        norm_next = (next_states - state_mean) / state_std

        # Causal action input
        input_actions = np.concatenate([np.zeros((1, actions.shape[-1]), dtype=np.float32), actions[:-1]], axis=0)
        rtgs = np.zeros((t_len, 1), dtype=np.float32)
        timesteps = np.arange(t_len, dtype=np.int64)

        t_states = torch.from_numpy(norm_obs).unsqueeze(0).to(device)
        t_actions = torch.from_numpy(input_actions).unsqueeze(0).to(device)
        t_rtgs = torch.from_numpy(rtgs).unsqueeze(0).to(device)
        t_timesteps = torch.from_numpy(timesteps).unsqueeze(0).to(device)
        t_targets = torch.from_numpy(actions).unsqueeze(0).to(device)
        t_next = torch.from_numpy(norm_next).unsqueeze(0).to(device)

        with torch.inference_mode():
            actions_pred, subgoals_pred, _, next_states_pred, _ = model(
                states=t_states,
                rtgs=t_rtgs,
                actions=t_actions,
                timesteps=t_timesteps,
            )

            # 1. Self-Awareness Metric: Forward Dynamics Prediction Error
            mae_dyn = F.l1_loss(next_states_pred, t_next).item()
            dyn_errors.append(mae_dyn)

            # 2. Goal Navigation Alignment: Distance to target goal
            if obs.shape[-1] == 53:
                dist_to_goal = float(np.linalg.norm(obs[-1, 51:53]))
            elif obs.shape[-1] == 6:
                final_pos = obs[-1, :2]
                goal_pos = obs[-1, 4:6]
                dist_to_goal = float(np.linalg.norm(final_pos - goal_pos))
            else:
                dist_to_goal = float(np.linalg.norm(obs[-1, :2] - obs[-1, -2:]))
            goal_reaching_distances.append(dist_to_goal)

            # 3. Action Jerk & Smoothness
            acts_np = actions_pred.squeeze(0).cpu().numpy()
            jerk = compute_action_smoothness(acts_np)
            all_jerks.append(jerk)

    successes = sum(1 for d in goal_reaching_distances if d < 0.5)
    success_rate = (successes / max(1, len(goal_reaching_distances))) * 100.0

    return {
        "forward_dynamics_mae": float(np.mean(dyn_errors)),
        "forward_dynamics_std": float(np.std(dyn_errors)),
        "mean_final_distance_to_goal": float(np.mean(goal_reaching_distances)),
        "goal_reaching_success_rate": success_rate,
        "action_smoothness_jerk": float(np.mean(all_jerks)),
    }


def main() -> None:
    args = parse_args()
    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at {ckpt_path}. Run `python scripts/train_unsupervised_maze.py` first.")
        return

    logger.info(f"Loading unsupervised trained checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_mean = ckpt.get("state_mean", np.zeros(cfg.model.prop_dim, dtype=np.float32))
    state_std = ckpt.get("state_std", np.ones(cfg.model.prop_dim, dtype=np.float32))

    model = HDMLModel.from_config(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Load evaluation trajectories
    if Path(args.dataset).exists():
        eval_trajs = TrajectoryCollector.load_dataset(args.dataset)
        if args.num_eval_trajs is not None and len(eval_trajs) > args.num_eval_trajs:
            eval_trajs = eval_trajs[:args.num_eval_trajs]
    else:
        eval_trajs = MinariDatasetAdapter.load_minari_dataset(
            dataset_name=args.dataset,
            max_episodes=args.num_eval_trajs,
            her_probability=0.0,  # test on actual task goals
            seed=999,
        )

    results = evaluate_cognition(
        model=model,
        trajectories=eval_trajs,
        state_mean=state_mean,
        state_std=state_std,
        context_length=cfg.training.context_length,
        device=device,
    )

    print("\n" + "=" * 100)
    print("UNSUPERVISED SPATIAL COGNITION & FORWARD WORLD MODEL EVALUATION (PointMaze)")
    print("=" * 100)
    print(f"Self-Awareness Forward Dynamics MAE : {results['forward_dynamics_mae']:.4f} +/- {results['forward_dynamics_std']:.4f}")
    print(f"Goal Reaching Success Rate (<0.5m)  : {results['goal_reaching_success_rate']:.1f}%")
    print(f"Mean Final Distance to Target Goal  : {results['mean_final_distance_to_goal']:.3f} m")
    print(f"Liquid Continuous Actuation Jerk    : {results['action_smoothness_jerk']:.4f}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
