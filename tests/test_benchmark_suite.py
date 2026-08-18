from __future__ import annotations

from pathlib import Path
import pytest
import torch
from hdml.models import (
    HDMLModel,
    DecisionTransformerBaseline,
    DecisionRNNBaseline,
    DiffusionPolicyBaseline,
    IQLBaseline,
    MLPBCBaseline,
)
from hdml.utils.config import HDMLConfig
from scripts.benchmark_baselines import evaluate_policy, get_d4rl_normalized_score


def test_d4rl_score_calculation() -> None:
    score = get_d4rl_normalized_score("HalfCheetah-v5", 12135.0)
    assert abs(score - 100.0) < 1e-4

    # Official D4RL reference bounds for HalfCheetah: random = -280.178739, expert = 12135.0
    rand_score = get_d4rl_normalized_score("HalfCheetah-v5", -280.178739)
    assert abs(rand_score - 0.0) < 1e-4


def test_benchmark_models_execution() -> None:
    cfg_path = "configs/halfcheetah_v5_default.yaml"
    if not Path(cfg_path).exists():
        cfg_path = "configs/halfcheetah_v4_default.yaml"
    cfg = HDMLConfig.from_yaml(cfg_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hdml = HDMLModel.from_config(cfg.model).to(device)
    dt = DecisionTransformerBaseline(prop_dim=17, action_dim=6, d_model=64).to(device)
    diff = DiffusionPolicyBaseline(prop_dim=17, action_dim=6, d_model=64, denoising_steps=3).to(device)
    iql = IQLBaseline(prop_dim=17, action_dim=6, hidden_dim=64).to(device)

    # Fast 1-episode evaluation test
    res_hdml = evaluate_policy(hdml, "hdml", env_name="HalfCheetah-v5", num_episodes=1, device=device)
    assert "mean_return" in res_hdml
    assert "d4rl_normalized_score" in res_hdml
    assert "frequency_hz" in res_hdml
    assert res_hdml["frequency_hz"] > 10.0

    res_diff = evaluate_policy(diff, "diffusion", env_name="HalfCheetah-v5", num_episodes=1, device=device)
    assert "mean_latency_ms" in res_diff

    res_iql = evaluate_policy(iql, "iql", env_name="HalfCheetah-v5", num_episodes=1, device=device)
    assert "frequency_hz" in res_iql
