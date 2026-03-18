from __future__ import annotations

import pytest
import torch
from hdml.models.liquid_head import LiquidReactiveControlHead


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_liquid_head_sequence_mode(device: torch.device) -> None:
    batch_size = 4
    seq_len = 12
    d_subgoal = 64
    prop_dim = 27
    action_dim = 8
    units = 32

    head = LiquidReactiveControlHead(
        d_subgoal=d_subgoal,
        prop_dim=prop_dim,
        action_dim=action_dim,
        units=units,
    ).to(device)

    subgoals = torch.randn(batch_size, seq_len, d_subgoal, device=device)
    prop = torch.randn(batch_size, seq_len, prop_dim, device=device)

    actions, next_hx = head(subgoals, prop)

    assert actions.shape == (batch_size, seq_len, action_dim)
    assert (actions >= -1.0).all() and (actions <= 1.0).all(), "Actions violated [-1, 1] bounds"
    assert torch.isfinite(actions).all()
    assert next_hx is not None


def test_liquid_head_step_mode(device: torch.device) -> None:
    batch_size = 2
    d_subgoal = 32
    prop_dim = 16
    action_dim = 4

    head = LiquidReactiveControlHead(
        d_subgoal=d_subgoal,
        prop_dim=prop_dim,
        action_dim=action_dim,
        units=16,
    ).to(device)

    subgoal_step = torch.randn(batch_size, d_subgoal, device=device)
    prop_step = torch.randn(batch_size, prop_dim, device=device)

    # First step (hx is None)
    act_1, hx_1 = head(subgoal_step, prop_step, hx=None)
    assert act_1.shape == (batch_size, action_dim)

    # Second step (pass hx_1)
    act_2, hx_2 = head(subgoal_step, prop_step, hx=hx_1)
    assert act_2.shape == (batch_size, action_dim)
    assert torch.isfinite(act_2).all()


def test_liquid_head_gradients(device: torch.device) -> None:
    head = LiquidReactiveControlHead(
        d_subgoal=32,
        prop_dim=16,
        action_dim=4,
        units=16,
    ).to(device)

    subgoals = torch.randn(2, 6, 32, device=device, requires_grad=True)
    prop = torch.randn(2, 6, 16, device=device, requires_grad=True)

    actions, _ = head(subgoals, prop)
    loss = actions.sum()
    loss.backward()

    assert subgoals.grad is not None
    assert prop.grad is not None
    assert torch.isfinite(subgoals.grad).all()
