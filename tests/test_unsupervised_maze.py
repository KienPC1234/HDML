from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from hdml.data.dataset import FastTensorTrajectoryDataset, MinariDatasetAdapter
from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import ModelConfig, TrainingConfig
from hdml.training.trainer import HDMLTrainer


def test_minari_maze_adapter_format():
    """Verify that MinariMazeAdapter extracts observations, goals, and next states correctly."""
    # Test loading small slice of PointMaze dataset
    try:
        trajectories = MinariDatasetAdapter.load_minari_dataset(
            dataset_name="D4RL/pointmaze/umaze-v2",
            max_episodes=5,
            her_probability=0.5,
            seed=42,
        )
    except Exception as e:
        pytest.skip(f"Minari dataset download unavailable in test environment: {e}")

    assert len(trajectories) == 5
    for traj in trajectories:
        assert "observations" in traj
        assert "actions" in traj
        assert "next_states" in traj
        assert "rewards" in traj
        assert "returns_to_go" in traj
        assert traj["observations"].shape[-1] == 6  # 4D prop + 2D goal
        assert traj["actions"].shape[-1] == 2       # 2D continuous control


def test_unsupervised_trainer_fast_step():
    """Verify forward-backward pass and composite loss calculations in HDMLTrainer."""
    cfg = ModelConfig(
        prop_dim=6,
        action_dim=2,
        d_model=32,
        d_state=8,
        d_conv=2,
        expand=2,
        num_mamba_layers=1,
        d_subgoal=16,
        cfc_units=16,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HDMLModel.from_config(cfg).to(device)

    # Synthetic trajectories
    dummy_trajs = [
        {
            "observations": np.random.randn(30, 6).astype(np.float32),
            "actions": np.random.randn(30, 2).astype(np.float32),
            "rewards": np.zeros(30, dtype=np.float32),
            "returns_to_go": np.zeros(30, dtype=np.float32),
            "timesteps": np.arange(30, dtype=np.int64),
            "next_states": np.random.randn(30, 6).astype(np.float32),
            "dones": np.zeros(30, dtype=bool),
            "total_return": np.float32(0.0),
        }
        for _ in range(4)
    ]

    dataset = FastTensorTrajectoryDataset(dummy_trajs, context_length=10, scale_return=1.0)

    train_cfg = TrainingConfig(
        max_epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        dynamics_weight=1.0,
        subgoal_loss_weight=0.5,
        use_amp=False,
    )

    trainer = HDMLTrainer(
        model=model,
        train_dataset=dataset,
        config=train_cfg,
        device=device,
    )

    metrics = trainer.train_epoch(epoch=1)
    assert "train_loss" in metrics
    assert "train_dynamics_loss" in metrics
    assert "train_action_loss" in metrics
    assert np.isfinite(metrics["train_loss"])
