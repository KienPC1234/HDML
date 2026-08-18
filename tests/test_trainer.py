from __future__ import annotations

import pytest
import numpy as np
import torch
from pathlib import Path
from hdml.models import HDMLModel
from hdml.data.dataset import FastTensorTrajectoryDataset
from hdml.training.trainer import HDMLTrainer
from hdml.utils.config import HDMLConfig


def test_hdml_v2_trainer_fast(tmp_path: Path) -> None:
    cfg_path = "configs/halfcheetah_v5_default.yaml"
    if not Path(cfg_path).exists():
        cfg_path = "configs/halfcheetah_v4_default.yaml"
    cfg = HDMLConfig.from_yaml(cfg_path)
    cfg.training.max_epochs = 2
    cfg.training.batch_size = 4
    cfg.training.use_amp = False  # Keep simple for testing on both CPU/GPU
    cfg.training.output_dir = str(tmp_path / "checkpoints")
    cfg.training.log_dir = str(tmp_path / "logs")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create dummy trajectories for testing
    trajectories = []
    for _ in range(3):
        trajectories.append({
            "observations": np.random.randn(30, cfg.model.prop_dim).astype(np.float32),
            "actions": np.random.randn(30, cfg.model.action_dim).astype(np.float32),
            "returns_to_go": np.random.randn(30).astype(np.float32),
            "timesteps": np.arange(30).astype(np.int32),
        })
    
    train_ds = FastTensorTrajectoryDataset(
        trajectories=trajectories[:2],
        context_length=cfg.training.context_length,
        scale_return=cfg.env.scale_return,
    )
    val_ds = FastTensorTrajectoryDataset(
        trajectories=trajectories[2:3],
        context_length=cfg.training.context_length,
        scale_return=cfg.env.scale_return,
        state_mean=train_ds.state_mean,
        state_std=train_ds.state_std,
    )
    
    model = HDMLModel.from_config(cfg.model).to(device)
    trainer = HDMLTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=cfg.training,
        device=device,
    )
    
    history = trainer.train()
    assert len(history) == cfg.training.max_epochs
    assert (Path(cfg.training.output_dir) / "best_model.pt").exists()
