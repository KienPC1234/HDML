from __future__ import annotations

import pytest
import torch
from hdml.models import MambaMLPHeadAblation, TransformerLiquidHeadAblation


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_mamba_mlp_head_ablation_forward_and_grad(device: torch.device) -> None:
    model = MambaMLPHeadAblation(
        prop_dim=27,
        action_dim=8,
        d_model=64,
        num_mamba_layers=2,
    ).to(device)

    states = torch.randn(2, 10, 27, device=device)
    rtgs = torch.randn(2, 10, 1, device=device)
    actions = torch.randn(2, 10, 8, device=device)
    timesteps = torch.arange(10, device=device).unsqueeze(0).expand(2, 10)

    act_pred, val_pred = model(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)
    assert act_pred.shape == (2, 10, 8)
    assert val_pred.shape == (2, 10, 1)

    loss = act_pred.sum() + val_pred.sum()
    loss.backward()
    assert model.mlp_head[0].weight.grad is not None

    step_act = model.get_action(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)
    assert step_act.shape == (2, 8)


def test_transformer_liquid_head_ablation_forward_and_grad(device: torch.device) -> None:
    model = TransformerLiquidHeadAblation(
        prop_dim=27,
        action_dim=8,
        d_model=64,
        nhead=2,
        num_layers=2,
        d_subgoal=32,
        cfc_units=16,
    ).to(device)

    states = torch.randn(2, 10, 27, device=device)
    rtgs = torch.randn(2, 10, 1, device=device)
    actions = torch.randn(2, 10, 8, device=device)
    timesteps = torch.arange(10, device=device).unsqueeze(0).expand(2, 10)

    act_pred, subgoals, val_pred, next_hx = model(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)
    assert act_pred.shape == (2, 10, 8)
    assert subgoals.shape == (2, 10, 32)
    assert val_pred.shape == (2, 10, 1)

    loss = act_pred.sum() + subgoals.sum()
    loss.backward()
    assert model.subgoal_head[0].weight.grad is not None

    step_act, next_h, step_sub = model.get_action(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)
    assert step_act.shape == (2, 8)
    assert step_sub.shape == (2, 32)
