from __future__ import annotations

import pytest
import torch
from hdml.models.hdml_model import HDMLModel
from hdml.training.losses import HDMLLoss
from hdml.utils.config import ModelConfig


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_hdml_full_forward_backward(device: torch.device) -> None:
    cfg = ModelConfig(
        prop_dim=27,
        action_dim=8,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        num_mamba_layers=2,
        d_subgoal=32,
        cfc_units=16,
    )
    model = HDMLModel.from_config(cfg).to(device)

    batch_size = 3
    seq_len = 10

    states = torch.randn(batch_size, seq_len, cfg.prop_dim, device=device)
    rtgs = torch.randn(batch_size, seq_len, 1, device=device)
    actions = torch.randn(batch_size, seq_len, cfg.action_dim, device=device)
    timesteps = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

    target_actions = torch.randn(batch_size, seq_len, cfg.action_dim, device=device)
    target_rtgs = torch.randn(batch_size, seq_len, 1, device=device)
    mask = torch.ones(batch_size, seq_len, device=device)

    # 1. Forward Pass
    actions_pred, subgoals_pred, values_pred, next_hx = model(
        states=states,
        rtgs=rtgs,
        actions=actions,
        timesteps=timesteps,
    )

    assert actions_pred.shape == (batch_size, seq_len, cfg.action_dim)
    assert subgoals_pred.shape == (batch_size, seq_len, cfg.d_subgoal)
    assert values_pred.shape == (batch_size, seq_len, 1)

    # 2. Loss Computation
    criterion = HDMLLoss()
    loss, loss_dict = criterion(
        actions_pred=actions_pred,
        target_actions=target_actions,
        subgoals=subgoals_pred,
        values_pred=values_pred,
        target_rtgs=target_rtgs,
        mask=mask,
    )

    assert torch.isfinite(loss).all()
    assert "total_loss" in loss_dict
    assert "action_loss" in loss_dict

    # 3. Backward Pass & Gradient Flow Check
    model.zero_grad()
    loss.backward()

    # Verify all layers have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for parameter: {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for: {name}"


def test_hdml_get_action_rollout(device: torch.device) -> None:
    cfg = ModelConfig(prop_dim=16, action_dim=4, d_model=32, d_subgoal=16, cfc_units=16, num_mamba_layers=1)
    model = HDMLModel.from_config(cfg).to(device)
    model.eval()

    context_len = 5
    states = torch.randn(1, context_len, 16, device=device)
    rtgs = torch.randn(1, context_len, 1, device=device)
    actions = torch.randn(1, context_len, 4, device=device)

    with torch.inference_mode():
        action, hx, subgoal = model.get_action(
            states=states,
            rtgs=rtgs,
            actions=actions,
        )

    assert action.shape == (1, 4)
    assert subgoal.shape == (1, 16)
    assert (action >= -1.0).all() and (action <= 1.0).all()
