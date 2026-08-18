from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from hdml.data.collector import discount_cumsum, TrajectoryCollector
from hdml.data.dataset import TrajectoryDataset, FastTensorTrajectoryDataset


def test_discount_cumsum() -> None:
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    gamma = 0.9
    rtg = discount_cumsum(rewards, gamma)

    # rtg[2] = 1.0
    # rtg[1] = 1.0 + 0.9 * 1.0 = 1.9
    # rtg[0] = 1.0 + 0.9 * 1.9 = 2.71
    expected = np.array([2.71, 1.9, 1.0], dtype=np.float32)
    np.testing.assert_allclose(rtg, expected, rtol=1e-5)


def test_trajectory_collector_save_load(tmp_path: Path) -> None:
    collector = TrajectoryCollector(env_name="Ant-v5", seed=42)
    # Collect small trajectory set
    trajs = collector.collect_trajectories(num_episodes=2, max_steps=10)
    assert len(trajs) == 2

    save_path = tmp_path / "test_dataset.npz"
    collector.save_dataset(trajs, save_path)
    assert save_path.exists()

    loaded_trajs = TrajectoryCollector.load_dataset(save_path)
    assert len(loaded_trajs) == 2
    assert loaded_trajs[0]["observations"].shape == trajs[0]["observations"].shape


def test_trajectory_dataset_batching() -> None:
    dummy_trajs = [
        {
            "observations": np.random.randn(15, 8).astype(np.float32),
            "actions": np.random.randn(15, 2).astype(np.float32),
            "rewards": np.ones(15, dtype=np.float32),
            "returns_to_go": np.ones(15, dtype=np.float32),
            "dones": np.zeros(15, dtype=bool),
            "timesteps": np.arange(15, dtype=np.int64),
            "total_return": np.float32(15.0),
        },
        {
            "observations": np.random.randn(25, 8).astype(np.float32),
            "actions": np.random.randn(25, 2).astype(np.float32),
            "rewards": np.ones(25, dtype=np.float32),
            "returns_to_go": np.ones(25, dtype=np.float32),
            "dones": np.zeros(25, dtype=bool),
            "timesteps": np.arange(25, dtype=np.int64),
            "total_return": np.float32(25.0),
        },
    ]

    dataset = TrajectoryDataset(dummy_trajs, context_length=10, scale_return=10.0)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    assert batch["states"].shape == (4, 10, 8)
    assert batch["actions"].shape == (4, 10, 2)
    assert batch["rtgs"].shape == (4, 10, 1)
    assert batch["timesteps"].shape == (4, 10)
    assert batch["mask"].shape == (4, 10)
    assert (batch["mask"] >= 0.0).all() and (batch["mask"] <= 1.0).all()


def test_no_leakage_action_input_shift() -> None:
    """The model input action at position j must be a_{j-1} (causal, no leakage).

    The prediction target at a valid position j is a_j. Therefore for any two
    consecutive valid positions (j, j+1), the input action at j+1 must equal the
    target action at j. This locks in the standard Decision-Transformer formulation.
    """
    traj = {
        "observations": np.random.randn(30, 8).astype(np.float32),
        "actions": np.random.randn(30, 2).astype(np.float32),
        "rewards": np.ones(30, dtype=np.float32),
        "returns_to_go": np.ones(30, dtype=np.float32),
        "dones": np.zeros(30, dtype=bool),
        "timesteps": np.arange(30, dtype=np.int64),
        "total_return": np.float32(30.0),
    }

    for dataset_cls in (FastTensorTrajectoryDataset, TrajectoryDataset):
        dataset = dataset_cls([traj], context_length=8, scale_return=10.0)
        for i in range(len(dataset)):
            sample = dataset[i]
            mask = sample["mask"].numpy()
            a_in = sample["actions"].numpy()
            a_tgt = sample["target_actions"].numpy()
            k = a_in.shape[0]
            for j in range(k - 1):
                if mask[j] > 0.5 and mask[j + 1] > 0.5:
                    np.testing.assert_allclose(a_in[j + 1], a_tgt[j], atol=1e-6,
                                               err_msg=f"{dataset_cls.__name__} leak at window {i}, pos {j}")
