from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import torch

from hdml.data.dataset import FastTensorTrajectoryDataset
from hdml.models.hdml_model import HDMLModel
from hdml.training.trainer import HDMLTrainer, TensorBatchLoader
from hdml.utils.config import ModelConfig, TrainingConfig


def _create_mock_trajectories(num_trajs: int = 5, traj_len: int = 30, prop_dim: int = 12, action_dim: int = 4):
    trajs = []
    for _ in range(num_trajs):
        obs = np.random.randn(traj_len, prop_dim).astype(np.float32)
        acts = np.random.randn(traj_len, action_dim).astype(np.float32)
        rews = np.random.randn(traj_len).astype(np.float32)
        rtgs = np.cumsum(rews[::-1])[::-1].astype(np.float32)
        trajs.append({
            "observations": obs,
            "actions": acts,
            "rewards": rews,
            "returns_to_go": rtgs,
            "timesteps": np.arange(traj_len, dtype=np.int64),
        })
    return trajs


def test_tensor_batch_loader():
    num_samples = 50
    tensors = {
        "states": torch.randn(num_samples, 20, 12),
        "actions": torch.randn(num_samples, 20, 4),
    }
    gen = torch.Generator().manual_seed(42)
    loader = TensorBatchLoader(
        tensors=tensors,
        batch_size=16,
        shuffle=True,
        device=torch.device("cpu"),
        generator=gen,
    )
    assert len(loader) == 4
    batches = list(loader)
    assert len(batches) == 4
    assert batches[0]["states"].shape == (16, 20, 12)
    assert batches[3]["states"].shape == (2, 20, 12)


def test_trainer_step_and_advantage_weighting():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajs = _create_mock_trajectories()
    dataset = FastTensorTrajectoryDataset(
        trajectories=trajs,
        context_length=10,
        chunk_size=4,
    )

    model_cfg = ModelConfig(
        prop_dim=12,
        action_dim=4,
        d_model=64,
        d_state=16,
        expand=2,
        num_mamba_layers=1,
        d_subgoal=12,
        cfc_units=16,
        cfc_backbone_units=32,
        use_native_mamba3=torch.cuda.is_available(),
    )
    model = HDMLModel.from_config(model_cfg)

    # Test with advantage weighting enabled
    train_cfg = TrainingConfig(
        max_epochs=1,
        batch_size=16,
        context_length=10,
        use_advantage_weighting=True,
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
    assert "train_action_loss" in metrics
    assert "train_adv_weight_max" in metrics
    assert metrics["train_adv_weight_max"] > 0.0


def test_trainer_checkpoint_save_and_exact_resume():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trajs = _create_mock_trajectories()
    dataset = FastTensorTrajectoryDataset(
        trajectories=trajs,
        context_length=10,
        chunk_size=4,
    )

    model_cfg = ModelConfig(
        prop_dim=12,
        action_dim=4,
        d_model=64,
        d_state=16,
        expand=2,
        num_mamba_layers=1,
        d_subgoal=12,
        cfc_units=16,
        cfc_backbone_units=32,
        use_native_mamba3=torch.cuda.is_available(),
    )
    model = HDMLModel.from_config(model_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        train_cfg = TrainingConfig(
            max_epochs=2,
            batch_size=16,
            context_length=10,
            output_dir=tmpdir,
            save_interval=1,
            eval_interval=1,
            use_amp=False,
        )

        trainer1 = HDMLTrainer(
            model=model,
            train_dataset=dataset,
            config=train_cfg,
            device=device,
        )

        trainer1.train_epoch(epoch=1)
        chkpt_path = trainer1.save_checkpoint(Path(tmpdir) / "checkpoint_ep001.pt", epoch=1)

        # Resume in trainer2
        model2 = HDMLModel.from_config(model_cfg)
        trainer2 = HDMLTrainer(
            model=model2,
            train_dataset=dataset,
            config=train_cfg,
            device=device,
        )

        trainer2.load_checkpoint(chkpt_path)
        assert trainer2.start_epoch == 1
        assert trainer2.current_step == trainer1.current_step

        # Weights must match exactly
        for p1, p2 in zip(trainer1.model.parameters(), trainer2.model.parameters()):
            assert torch.allclose(p1, p2)

