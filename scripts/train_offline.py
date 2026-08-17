#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import torch

from hdml.utils.config import HDMLConfig
from hdml.models.hdml_model import HDMLModel
from hdml.data.collector import TrajectoryCollector
from hdml.data.dataset import TrajectoryDataset, FastTensorTrajectoryDataset
from hdml.training.trainer import HDMLTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HDML Offline Decision Mamba policy.")
    parser.add_argument("--config", type=str, default="configs/ant_v4_default.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="data/ant_v4_trajectories.npz", help="Path to dataset NPZ")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision (AMP)")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--fast-data", action="store_true", default=True, help="Use pre-vectorized contiguous tensor dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    cfg = HDMLConfig.from_yaml(args.config)
    if args.epochs is not None:
        cfg.training.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    cfg.training.use_amp = args.amp
    cfg.training.num_workers = args.num_workers

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Target execution hardware: {device}")

    # Set seeds
    torch.manual_seed(cfg.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.training.seed)

    # Check if dataset exists, if not collect a default demo set
    data_path = Path(args.dataset)
    if not data_path.exists():
        logger.warning(f"Dataset not found at {data_path}. Generating demonstration dataset...")
        collector = TrajectoryCollector(env_name=cfg.env.env_name, seed=cfg.training.seed)
        trajs = collector.collect_trajectories(num_episodes=30, max_steps=cfg.env.max_episode_steps)
        collector.save_dataset(trajs, data_path)

    # Load trajectories
    trajectories = TrajectoryCollector.load_dataset(data_path)

    # Split train/val
    split_idx = max(1, int(len(trajectories) * 0.85))
    train_trajs = trajectories[:split_idx]
    val_trajs = trajectories[split_idx:] if split_idx < len(trajectories) else None

    dataset_cls = FastTensorTrajectoryDataset if args.fast_data else TrajectoryDataset

    train_dataset = dataset_cls(
        trajectories=train_trajs,
        context_length=cfg.training.context_length,
        scale_return=cfg.env.scale_return,
    )

    val_dataset = dataset_cls(
        trajectories=val_trajs,
        context_length=cfg.training.context_length,
        scale_return=cfg.env.scale_return,
        state_mean=train_dataset.state_mean,
        state_std=train_dataset.state_std,
    ) if val_trajs else None

    # Update model dimensions from dataset
    cfg.model.prop_dim = train_dataset.prop_dim
    cfg.model.action_dim = train_dataset.action_dim

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

    history = trainer.train()
    logger.info(f"Training finished. Best Loss: {trainer.best_loss:.4f}")


if __name__ == "__main__":
    main()
