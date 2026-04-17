from __future__ import annotations

import pytest
import torch
from hdml.models.hdml_model import HDMLModel
from hdml.evaluation.evaluator import HDMLEvaluator
from hdml.evaluation.perturbations import SensorNoisePerturbation, ForceImpulsePerturbation
from hdml.utils.config import ModelConfig


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_evaluator_closed_loop_ant(device: torch.device) -> None:
    # Ant-v4 has prop_dim=27, action_dim=8
    cfg = ModelConfig(
        prop_dim=27,
        action_dim=8,
        d_model=64,
        d_subgoal=32,
        cfc_units=16,
        num_mamba_layers=1,
    )
    model = HDMLModel.from_config(cfg).to(device)

    evaluator = HDMLEvaluator(
        model=model,
        env_name="Ant-v4",
        context_length=10,
        target_return=1000.0,
        scale_return=1000.0,
        device=device,
    )

    results = evaluator.evaluate_benchmark(num_episodes=2, with_perturbations=False)

    assert "mean_return" in results
    assert "mean_smoothness" in results
    assert "mean_length" in results
    assert results["mean_length"] > 0


def test_evaluator_with_perturbations(device: torch.device) -> None:
    cfg = ModelConfig(
        prop_dim=27,
        action_dim=8,
        d_model=64,
        d_subgoal=32,
        cfc_units=16,
        num_mamba_layers=1,
    )
    model = HDMLModel.from_config(cfg).to(device)

    evaluator = HDMLEvaluator(
        model=model,
        env_name="Ant-v4",
        context_length=10,
        target_return=1000.0,
        scale_return=1000.0,
        device=device,
    )

    results = evaluator.evaluate_benchmark(num_episodes=2, with_perturbations=True)
    assert results["mean_length"] > 0
    assert results["mean_smoothness"] >= 0.0


def test_evaluator_hierarchical_macro_interval(device: torch.device) -> None:
    cfg = ModelConfig(
        prop_dim=27,
        action_dim=8,
        d_model=64,
        d_subgoal=32,
        cfc_units=16,
        num_mamba_layers=1,
    )
    model = HDMLModel.from_config(cfg).to(device)

    evaluator = HDMLEvaluator(
        model=model,
        env_name="Ant-v4",
        context_length=10,
        target_return=1000.0,
        scale_return=1000.0,
        device=device,
    )

    results = evaluator.evaluate_benchmark(num_episodes=2, with_perturbations=False, macro_interval=5)
    assert results["mean_length"] > 0
    assert results["mean_smoothness"] >= 0.0

