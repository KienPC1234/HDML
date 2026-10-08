"""Tests for Baseline Architectures and Ablations."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hdml.models.baselines import (
    DecisionTransformerBaseline,
    DecisionRNNBaseline,
    DiffusionPolicyBaseline,
    IQLBaseline,
    MLPBCBaseline,
)
from hdml.models.ablations import MambaMLPHeadAblation, TransformerLiquidHeadAblation


def test_baseline_forward_passes() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = torch.randn(2, 10, 17, device=device)
    actions = torch.randn(2, 10, 6, device=device)
    rtgs = torch.randn(2, 10, 1, device=device)
    timesteps = torch.arange(10, device=device).unsqueeze(0).repeat(2, 1)

    # 1. Decision Transformer
    dt = DecisionTransformerBaseline(prop_dim=17, action_dim=6, d_model=128, nhead=4, num_layers=2).to(device)
    dt_out = dt(states, rtgs, actions, timesteps)
    assert dt_out.shape == (2, 10, 6)

    # 2. Decision RNN
    rnn = DecisionRNNBaseline(prop_dim=17, action_dim=6, d_model=128, num_layers=2).to(device)
    rnn_out, _ = rnn(states, rtgs, actions, timesteps)
    assert rnn_out.shape == (2, 10, 6)

    # 3. Diffusion Policy
    diff = DiffusionPolicyBaseline(prop_dim=17, action_dim=6, d_model=128, denoising_steps=5).to(device)
    diff_action = diff.get_action(states, rtgs)
    assert diff_action.shape == (2, 6)

    # 4. IQL Baseline
    iql = IQLBaseline(prop_dim=17, action_dim=6, hidden_dim=128).to(device)
    iql_out = iql(states[:, -1, :])
    assert iql_out.shape == (2, 6)

    # 5. MLP BC
    mlp = MLPBCBaseline(prop_dim=17, action_dim=6, hidden_dim=128).to(device)
    mlp_out = mlp(states, rtgs)
    assert mlp_out.shape == (2, 10, 6)


def test_ablations_forward_passes() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states = torch.randn(2, 10, 17, device=device)
    actions = torch.randn(2, 10, 6, device=device)
    rtgs = torch.randn(2, 10, 1, device=device)
    timesteps = torch.arange(10, device=device).unsqueeze(0).repeat(2, 1)

    # 1. Mamba + MLP (No Liquid Head)
    mamba_mlp = MambaMLPHeadAblation(prop_dim=17, action_dim=6, d_model=128, num_mamba_layers=2).to(device)
    acts1, val1 = mamba_mlp(states, rtgs, actions, timesteps)
    assert acts1.shape == (2, 10, 6)

    # 2. Transformer + Liquid Head
    trans_liquid = TransformerLiquidHeadAblation(prop_dim=17, action_dim=6, d_model=128, nhead=4, num_layers=2).to(device)
    acts2, sub2, val2, _ = trans_liquid(states, rtgs, actions, timesteps)
    assert acts2.shape == (2, 10, 6)
