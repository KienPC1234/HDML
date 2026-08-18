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

    target_velocity = torch.randn(batch_size, seq_len, 4, cfg.action_dim, device=device)
    pred_velocity = torch.randn(batch_size, seq_len, 4, cfg.action_dim, device=device, requires_grad=True)
    noise = torch.randn(batch_size, seq_len, 4, cfg.action_dim, device=device)
    q1_pred = torch.randn(batch_size, seq_len, 1, device=device, requires_grad=True)
    q2_pred = torch.randn(batch_size, seq_len, 1, device=device, requires_grad=True)
    q_target = torch.randn(batch_size, seq_len, 1, device=device)
    action_chunk = torch.randn(batch_size, seq_len, 4, cfg.action_dim, device=device, requires_grad=True)
    target_states = torch.randn(batch_size, seq_len, cfg.prop_dim, device=device)
    mask = torch.ones(batch_size, seq_len, device=device)

    # 1. Forward Pass (Mamba backbone + direct action head)
    actions_pred, subgoals_pred, values_pred, next_states_pred, next_hx = model(
        states=states, rtgs=rtgs, actions=actions
    )

    assert actions_pred.shape == (batch_size, seq_len, cfg.action_dim)
    assert subgoals_pred.shape == (batch_size, seq_len, cfg.d_subgoal)
    assert next_states_pred.shape == (batch_size, seq_len, cfg.prop_dim)

    # 2. Loss Computation
    criterion = HDMLLoss()
    context = torch.cat([subgoals_pred, states], dim=-1)
    
    loss, loss_dict = criterion(
        target_velocity=target_velocity,
        pred_velocity=pred_velocity,
        noise=noise,
        q1_pred=q1_pred,
        q2_pred=q2_pred,
        q_target=q_target,
        values_pred=values_pred,
        state_repr=context,
        action_chunk=action_chunk,
        critic=model.hiqc_critic,
        next_states_pred=next_states_pred,
        target_states=target_states,
        mask=mask,
    )

    assert torch.isfinite(loss).all()
    assert "total_loss" in loss_dict
    assert "flow_loss" in loss_dict
    assert "pave_loss" in loss_dict
    assert "value_loss" in loss_dict

    # 3. Backward Pass & Gradient Flow Check
    model.zero_grad()
    loss.backward()

    # Verify parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad and "flow_policy" not in name:  # We mocked flow policy input here
            if param.grad is not None:
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
        action, hx, info = model.get_action(
            states=states,
            rtgs=rtgs,
            actions=actions,
        )

    assert action.shape == (1, 4)
    assert info["subgoal"].shape == (1, 16)
    assert info["action"].shape == (1, 4)
