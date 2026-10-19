#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any
import numpy as np
import torch

from hdml.utils.config import HDMLConfig
from hdml.models.hdml_model import HDMLModel
from hdml.data.collector import TrajectoryCollector
from hdml.data.dataset import TrajectoryDataset, FastTensorTrajectoryDataset, MinariDatasetAdapter
from hdml.training.trainer import HDMLTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HDML Offline Decision Mamba policy.")
    parser.add_argument("--config", type=str, default="configs/halfcheetah_v5_default.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="data/halfcheetah_v5_expert.npz", help="Path to dataset NPZ or Minari dataset name (e.g. D4RL/pointmaze/umaze-v2)")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum number of dataset episodes to load")
    parser.add_argument("--her-prob", type=float, default=0.0, help="Hindsight goal relabeling probability for goal-conditioned / maze learning")
    # Tri-state switches: None keeps the YAML value, the flag overrides it.
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="Override training.use_amp")
    parser.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers (non-fast datasets only)")
    parser.add_argument("--stride", type=int, default=1, help="Subsampling stride along trajectories for high-speed training")
    parser.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint .pt (restores optimizer, scaler, RNG, epoch and LR schedule)")
    parser.add_argument("--fast-data", action=argparse.BooleanOptionalAction, default=None, help="Override training.use_fast_dataset")
    parser.add_argument("--advantage-weighting", action=argparse.BooleanOptionalAction, default=None, help="Override training.use_advantage_weighting")
    parser.add_argument("--best-metric", type=str, default=None, help="Override training.best_metric (e.g. val_action_loss, val_loss)")
    parser.add_argument("--compile-cfc", action=argparse.BooleanOptionalAction, default=None, help="Override training.compile_cfc")
    parser.add_argument("--cfc", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable CfC action filter (e.g. --no-cfc for lightweight NPC)")
    parser.add_argument("--critic", action=argparse.BooleanOptionalAction, default=None, help="Enable or disable HiQC critic (e.g. --no-critic for pure imitation)")
    parser.add_argument("--policy", type=str, default=None, choices=["flow", "gaussian"], help="Action policy type ('flow' or 'gaussian')")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None, help="Override training.use_wandb")
    parser.add_argument("--wandb-project", type=str, default=None, help="Override training.wandb_project")
    parser.add_argument("--wandb-name", type=str, default=None, help="WandB experiment run name")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=None, help="Override training.use_tensorboard")
    return parser.parse_args()


def _apply_overrides(cfg: HDMLConfig, args: argparse.Namespace) -> None:
    """Apply explicitly-passed CLI values on top of the YAML training config."""
    training_overrides: dict[str, Any] = {
        "seed": args.seed,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "use_amp": args.amp,
        "num_workers": args.num_workers,
        "use_fast_dataset": args.fast_data,
        "use_advantage_weighting": args.advantage_weighting,
        "best_metric": args.best_metric,
        "compile_cfc": args.compile_cfc,
        "use_wandb": args.wandb,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_name,
        "use_tensorboard": args.tensorboard,
    }
    for key, value in training_overrides.items():
        if value is not None:
            setattr(cfg.training, key, value)

    # Model architecture overrides for fast user experimentation / NPC deployment
    if args.cfc is not None:
        cfg.model.use_cfc = args.cfc
    if args.critic is not None:
        cfg.model.use_critic = args.critic
    if args.policy is not None:
        cfg.model.action_policy = args.policy
    if args.output_dir is not None:
        cfg.training.output_dir = args.output_dir
        cfg.training.log_dir = str(Path(args.output_dir) / "logs")


def main() -> None:
    args = parse_args()

    # Load configuration
    cfg = HDMLConfig.from_yaml(args.config)
    _apply_overrides(cfg, args)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Target execution hardware: {device}")

    # Set seeds
    torch.manual_seed(cfg.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.training.seed)

    # Load dataset: support Minari datasets and local NPZ
    is_minari = args.dataset.startswith("minari:") or ("/" in args.dataset and not Path(args.dataset).exists())
    if is_minari:
        dataset_name = args.dataset.removeprefix("minari:")
        logger.info(f"Loading official Minari dataset: {dataset_name} (HER={args.her_prob})")
        trajectories = MinariDatasetAdapter.load_minari_dataset(
            dataset_name=dataset_name,
            gamma=cfg.training.gamma,
            max_episodes=args.max_episodes,
            her_probability=args.her_prob,
            seed=cfg.training.seed,
        )
    else:
        data_path = Path(args.dataset)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset missing: {data_path}. Supply provenance-checked data explicitly.")

        trajectories = TrajectoryCollector.load_dataset(data_path)
        if args.max_episodes is not None and len(trajectories) > args.max_episodes:
            trajectories = trajectories[: args.max_episodes]

    # Split train/val deterministically to avoid collection-order bias
    rng = np.random.default_rng(cfg.training.seed)
    perm = rng.permutation(len(trajectories))
    split_idx = max(1, int(len(trajectories) * 0.85))
    train_trajs = [trajectories[i] for i in perm[:split_idx]]
    val_trajs = [trajectories[i] for i in perm[split_idx:]] if split_idx < len(trajectories) else None

    dataset_cls = FastTensorTrajectoryDataset if cfg.training.use_fast_dataset else TrajectoryDataset

    train_dataset = dataset_cls(
        trajectories=train_trajs,
        context_length=cfg.training.context_length,
        chunk_size=cfg.model.chunk_size,
        scale_return=cfg.env.scale_return,
        gamma=cfg.training.gamma,
        stride=args.stride,
    )

    val_dataset = dataset_cls(
        trajectories=val_trajs,
        context_length=cfg.training.context_length,
        chunk_size=cfg.model.chunk_size,
        scale_return=cfg.env.scale_return,
        state_mean=train_dataset.state_mean,
        state_std=train_dataset.state_std,
        gamma=cfg.training.gamma,
        stride=args.stride,
    ) if val_trajs else None

    # Update model dimensions from dataset
    cfg.model.prop_dim = train_dataset.prop_dim
    cfg.model.action_dim = train_dataset.action_dim

    # Persist effective run configuration to output directory
    out_dir = Path(cfg.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "effective_config.yaml")

    # Instantiate Model
    model = HDMLModel.from_config(cfg.model)
    logger.info(
        f"HDML Model instantiated with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters."
    )

    # Instantiate Trainer and Train
    trainer = HDMLTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cfg.training,
        device=device,
    )

    # Optional resume from a checkpoint
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    trainer.train()
    logger.info(f"Training finished. Best Loss ({cfg.training.best_metric}): {trainer.best_loss:.4f}")


if __name__ == "__main__":
    main()

