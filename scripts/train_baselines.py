#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import torch

from hdml.utils.config import HDMLConfig
from hdml.data.collector import TrajectoryCollector
from hdml.data.dataset import FastTensorTrajectoryDataset, TrajectoryDataset
from hdml.models import (
    DecisionTransformerBaseline,
    DecisionRNNBaseline,
    DiffusionPolicyBaseline,
    IQLBaseline,
    MLPBCBaseline,
    MambaMLPHeadAblation,
    TransformerLiquidHeadAblation,
)
from hdml.training.baseline_trainer import BaselineTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_FACTORY: dict[str, str] = {
    "dt": "DecisionTransformerBaseline",
    "rnn": "DecisionRNNBaseline",
    "mlp": "MLPBCBaseline",
    "diffusion": "DiffusionPolicyBaseline",
    "iql": "IQLBaseline",
    "mamba_mlp": "MambaMLPHeadAblation",
    "transformer_liquid": "TransformerLiquidHeadAblation",
}

BASELINE_DIR = "baselines"


def build_model(model_type: str, cfg: HDMLConfig) -> torch.nn.Module:
    """Instantiate a baseline / ablation model from configuration."""
    m = cfg.model
    if model_type == "dt":
        return DecisionTransformerBaseline(
            prop_dim=m.prop_dim, action_dim=m.action_dim, d_model=m.d_model, nhead=4, num_layers=m.num_mamba_layers
        )
    if model_type == "rnn":
        return DecisionRNNBaseline(
            prop_dim=m.prop_dim, action_dim=m.action_dim, d_model=m.d_model, num_layers=m.num_mamba_layers
        )
    if model_type == "mlp":
        return MLPBCBaseline(prop_dim=m.prop_dim, action_dim=m.action_dim, hidden_dim=256)
    if model_type == "diffusion":
        return DiffusionPolicyBaseline(
            prop_dim=m.prop_dim, action_dim=m.action_dim, d_model=m.d_model, denoising_steps=10
        )
    if model_type == "iql":
        return IQLBaseline(prop_dim=m.prop_dim, action_dim=m.action_dim, hidden_dim=256)
    if model_type == "mamba_mlp":
        return MambaMLPHeadAblation(
            prop_dim=m.prop_dim,
            action_dim=m.action_dim,
            d_model=m.d_model,
            num_mamba_layers=m.num_mamba_layers,
        )
    if model_type == "transformer_liquid":
        return TransformerLiquidHeadAblation(
            prop_dim=m.prop_dim,
            action_dim=m.action_dim,
            d_model=m.d_model,
            nhead=4,
            num_layers=m.num_mamba_layers,
            d_subgoal=m.d_subgoal,
            cfc_units=m.cfc_units,
        )
    raise ValueError(f"Unknown model type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SOTA baseline architectures for fair benchmarking.")
    parser.add_argument("--config", type=str, default="configs/halfcheetah_v5_default.yaml", help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default="data/halfcheetah_v5_trajectories.npz", help="Path to dataset NPZ")
    parser.add_argument("--model", type=str, default="all", choices=["all", *MODEL_FACTORY], help="Baseline model to train")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable AMP")
    args = parser.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    if args.epochs is not None:
        cfg.training.max_epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    cfg.training.use_amp = args.amp
    cfg.training.num_workers = 0
    cfg.training.pin_memory = False

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Target execution hardware: {device}")
    torch.manual_seed(cfg.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.training.seed)

    data_path = Path(args.dataset)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run `python scripts/collect_data.py` first."
        )

    trajectories = TrajectoryCollector.load_dataset(data_path)
    split_idx = max(1, int(len(trajectories) * 0.85))
    train_trajs = trajectories[:split_idx]
    val_trajs = trajectories[split_idx:] if split_idx < len(trajectories) else None

    dataset_cls = FastTensorTrajectoryDataset
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

    cfg.model.prop_dim = train_dataset.prop_dim
    cfg.model.action_dim = train_dataset.action_dim

    default_out_dir = Path(cfg.training.output_dir)
    ckpt_dir = Path(args.output_dir) if args.output_dir else default_out_dir / BASELINE_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    models_to_train = list(MODEL_FACTORY) if args.model == "all" else [args.model]

    for model_type in models_to_train:
        cfg.training.output_dir = str(ckpt_dir / model_type)
        cfg.training.log_dir = str(ckpt_dir / model_type / "logs")
        cfg.training.seed = cfg.training.seed + sum(ord(c) for c in model_type) % 100
        torch.manual_seed(cfg.training.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.training.seed)

        model = build_model(model_type, cfg).to(device)
        logger.info(
            f"Training baseline '{model_type}' with "
            f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} params."
        )
        trainer = BaselineTrainer(
            model=model,
            model_type=model_type,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=cfg.training,
            device=device,
        )
        trainer.train()
        trainer.save_checkpoint(ckpt_dir / f"{model_type}_best.pt")
        logger.info(f"Baseline '{model_type}' training completed. Best Val Loss: {trainer.best_loss:.4f}")


if __name__ == "__main__":
    main()
