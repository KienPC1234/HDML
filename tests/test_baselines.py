from __future__ import annotations

import pytest
import torch
from hdml.models import (
    DecisionTransformerBaseline,
    DecisionRNNBaseline,
    DiffusionPolicyBaseline,
    IQLBaseline,
    MLPBCBaseline,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_decision_transformer_forward(device: torch.device) -> None:
    model = DecisionTransformerBaseline(prop_dim=27, action_dim=8, d_model=64, num_layers=2).to(device)
    states = torch.randn(2, 10, 27, device=device)
    rtgs = torch.randn(2, 10, 1, device=device)
    actions = torch.randn(2, 10, 8, device=device)
    timesteps = torch.arange(10, device=device).unsqueeze(0).expand(2, 10)

    out = model(states, rtgs, actions, timesteps)
    assert out.shape == (2, 10, 8)

    step_act = model.get_action(states, rtgs, actions, timesteps)
    assert step_act.shape == (2, 8)


def test_decision_rnn_forward(device: torch.device) -> None:
    model = DecisionRNNBaseline(prop_dim=27, action_dim=8, d_model=64, num_layers=2).to(device)
    states = torch.randn(2, 10, 27, device=device)
    rtgs = torch.randn(2, 10, 1, device=device)
    actions = torch.randn(2, 10, 8, device=device)
    timesteps = torch.arange(10, device=device).unsqueeze(0).expand(2, 10)

    out, hx = model(states, rtgs, actions, timesteps)
    assert out.shape == (2, 10, 8)
    assert hx[0].shape == (2, 2, 64)

    step_act, next_hx = model.get_action(states, rtgs, actions, timesteps, hx)
    assert step_act.shape == (2, 8)


def test_diffusion_policy_forward(device: torch.device) -> None:
    model = DiffusionPolicyBaseline(prop_dim=27, action_dim=8, d_model=64, denoising_steps=5).to(device)
    noisy_actions = torch.randn(2, 8, device=device)
    k = torch.tensor([3, 1], device=device)
    states = torch.randn(2, 27, device=device)
    rtgs = torch.randn(2, 1, device=device)

    eps_pred = model(noisy_actions, k, states, rtgs)
    assert eps_pred.shape == (2, 8)

    # Test multi-step reverse diffusion rollout
    seq_states = torch.randn(2, 10, 27, device=device)
    seq_rtgs = torch.randn(2, 10, 1, device=device)
    act = model.get_action(seq_states, seq_rtgs)
    assert act.shape == (2, 8)


def test_iql_baseline_forward(device: torch.device) -> None:
    model = IQLBaseline(prop_dim=27, action_dim=8, hidden_dim=64).to(device)
    states = torch.randn(2, 27, device=device)
    actions = torch.randn(2, 8, device=device)

    q1 = model.q1(torch.cat([states, actions], dim=-1))
    v = model.v(states)
    act_pred = model(states)
    assert q1.shape == (2, 1)
    assert v.shape == (2, 1)
    assert act_pred.shape == (2, 8)

    step_act = model.get_action(states)
    assert step_act.shape == (2, 8)


def test_mlp_bc_forward(device: torch.device) -> None:
    model = MLPBCBaseline(prop_dim=27, action_dim=8, hidden_dim=64).to(device)
    states = torch.randn(2, 27, device=device)
    rtgs = torch.randn(2, 1, device=device)

    out = model(states, rtgs)
    assert out.shape == (2, 8)

    step_act = model.get_action(states[0], rtgs[0])
    assert step_act.shape == (1, 8)
