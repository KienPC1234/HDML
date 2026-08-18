from __future__ import annotations

import numpy as np
import torch
from hdml.data.dataset import FastTensorTrajectoryDataset
from hdml.training.baseline_trainer import BaselineTrainer
from hdml.utils.config import TrainingConfig


def _dummy_trajectories(n: int = 6, length: int = 30, prop_dim: int = 17, act_dim: int = 6) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    trajs: list[dict[str, np.ndarray]] = []
    for _ in range(n):
        obs = rng.normal(size=(length, prop_dim)).astype(np.float32)
        act = np.clip(rng.normal(size=(length, act_dim)), -1.0, 1.0).astype(np.float32)
        rew = np.ones(length, dtype=np.float32)
        trajs.append(
            {
                "observations": obs,
                "actions": act,
                "rewards": rew,
                "returns_to_go": np.linspace(5.0, 1.0, length).astype(np.float32),
                "dones": np.zeros(length, dtype=bool),
                "timesteps": np.arange(length, dtype=np.int64),
                "total_return": np.float32(length),
            }
        )
    return trajs


def test_baseline_trainer_dt_gradients_and_checkpoint(tmp_path) -> None:
    """BaselineTrainer must train a sequence BC model with gradients and save a checkpoint."""
    from hdml.models import DecisionTransformerBaseline

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = FastTensorTrajectoryDataset(_dummy_trajectories(), context_length=8, scale_return=10.0)
    cfg = TrainingConfig(
        max_epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        warmup_steps=5,
        use_amp=torch.cuda.is_available(),
        output_dir=str(tmp_path),
    )
    model = DecisionTransformerBaseline(prop_dim=17, action_dim=6, d_model=64).to(device)
    trainer = BaselineTrainer(model=model, model_type="dt", train_dataset=ds, config=cfg, device=device)

    # Single train step to verify gradients flow
    batch = next(iter(trainer.train_loader))
    loss, loss_dict = trainer._compute_loss(batch)
    assert torch.isfinite(loss).all()
    loss.backward()
    has_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert has_grad, "No gradient flowed through the DT baseline"

    ckpt = trainer.save_checkpoint(tmp_path / "dt_best.pt")
    assert ckpt.exists()
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert loaded["model_type"] == "dt"
    assert loaded["state_mean"] is not None
    assert loaded["state_std"] is not None


def test_baseline_trainer_mlp_markovian() -> None:
    """MLP baseline trains on the last valid position of each window."""
    from hdml.models import MLPBCBaseline

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = FastTensorTrajectoryDataset(_dummy_trajectories(), context_length=8, scale_return=10.0)
    cfg = TrainingConfig(max_epochs=1, batch_size=8, learning_rate=1e-3, warmup_steps=5, output_dir="/tmp/hdml_tests_mlp")
    model = MLPBCBaseline(prop_dim=17, action_dim=6, hidden_dim=64).to(device)
    trainer = BaselineTrainer(model=model, model_type="mlp", train_dataset=ds, config=cfg, device=device)
    batch = next(iter(trainer.train_loader))
    loss, loss_dict = trainer._compute_loss(batch)
    assert torch.isfinite(loss).all()
    assert "action_loss" in loss_dict
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_baseline_trainer_diffusion_ddpm() -> None:
    """Diffusion baseline trains via DDPM epsilon-prediction."""
    from hdml.models import DiffusionPolicyBaseline

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = FastTensorTrajectoryDataset(_dummy_trajectories(), context_length=8, scale_return=10.0)
    cfg = TrainingConfig(max_epochs=1, batch_size=8, learning_rate=1e-3, warmup_steps=5, output_dir="/tmp/hdml_tests_diff")
    model = DiffusionPolicyBaseline(prop_dim=17, action_dim=6, d_model=32, denoising_steps=3).to(device)
    trainer = BaselineTrainer(model=model, model_type="diffusion", train_dataset=ds, config=cfg, device=device)
    batch = next(iter(trainer.train_loader))
    loss, loss_dict = trainer._compute_loss(batch)
    assert torch.isfinite(loss).all()
    assert "diffusion_loss" in loss_dict
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_baseline_trainer_iql_expectile() -> None:
    """IQL baseline trains value (expectile) and policy (advantage-weighted) heads."""
    from hdml.models import IQLBaseline

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = FastTensorTrajectoryDataset(_dummy_trajectories(), context_length=8, scale_return=10.0)
    cfg = TrainingConfig(max_epochs=1, batch_size=8, learning_rate=1e-3, warmup_steps=5, output_dir="/tmp/hdml_tests_iql")
    model = IQLBaseline(prop_dim=17, action_dim=6, hidden_dim=64).to(device)
    trainer = BaselineTrainer(model=model, model_type="iql", train_dataset=ds, config=cfg, device=device)
    batch = next(iter(trainer.train_loader))
    loss, loss_dict = trainer._compute_loss(batch)
    assert torch.isfinite(loss).all()
    assert "policy_loss" in loss_dict and "value_loss" in loss_dict
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())