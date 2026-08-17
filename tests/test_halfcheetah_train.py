from __future__ import annotations

import pytest
import numpy as np
import torch
from pathlib import Path
from hdml.models import HDMLModel
from hdml.data.collector import TrajectoryCollector
from hdml.data.dataset import FastTensorTrajectoryDataset
from hdml.training.trainer import HDMLTrainer
from hdml.utils.config import HDMLConfig


def test_train_halfcheetah_fast() -> None:
    cfg = HDMLConfig.from_yaml("configs/halfcheetah_v4_default.yaml")
    cfg.training.max_epochs = 2
    cfg.training.batch_size = 32
    cfg.training.use_amp = True
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trajectories = TrajectoryCollector.load_dataset("data/halfcheetah_v4_trajectories.npz")
    
    train_ds = FastTensorTrajectoryDataset(
        trajectories=trajectories[:4],
        context_length=cfg.training.context_length,
        scale_return=cfg.env.scale_return,
    )
    val_ds = FastTensorTrajectoryDataset(
        trajectories=trajectories[4:6],
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
    assert history[-1]["val_loss"] < history[0]["train_loss"]
    assert (trainer.output_dir / "best_model.pt").exists()
