from __future__ import annotations

import pytest
import torch
from hdml.models.fusion import CrossModalFusion, VisualPatchEncoder


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_cross_modal_fusion_forward_shapes(device: torch.device) -> None:
    batch_size = 4
    seq_len = 10
    prop_dim = 27
    action_dim = 8
    d_model = 128

    fusion = CrossModalFusion(
        prop_dim=prop_dim,
        action_dim=action_dim,
        d_model=d_model,
        use_visual=False,
    ).to(device)

    states = torch.randn(batch_size, seq_len, prop_dim, device=device)
    rtgs = torch.randn(batch_size, seq_len, 1, device=device)
    actions = torch.randn(batch_size, seq_len, action_dim, device=device)
    timesteps = torch.randint(0, 100, (batch_size, seq_len), device=device)

    u_t = fusion(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)

    assert u_t.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {u_t.shape}"
    assert torch.isfinite(u_t).all(), "Output contains NaN or Inf values."


def test_cross_modal_fusion_defaults(device: torch.device) -> None:
    batch_size = 2
    seq_len = 5
    prop_dim = 16
    action_dim = 4
    d_model = 64

    fusion = CrossModalFusion(
        prop_dim=prop_dim,
        action_dim=action_dim,
        d_model=d_model,
    ).to(device)

    states = torch.randn(batch_size, seq_len, prop_dim, device=device)
    rtgs = torch.randn(batch_size, seq_len, device=device)  # 2D rtg test

    # actions and timesteps are None by default
    u_t = fusion(states=states, rtgs=rtgs)

    assert u_t.shape == (batch_size, seq_len, d_model)
    assert torch.isfinite(u_t).all()


def test_cross_modal_fusion_gradients(device: torch.device) -> None:
    fusion = CrossModalFusion(prop_dim=27, action_dim=8, d_model=64).to(device)

    states = torch.randn(2, 5, 27, device=device, requires_grad=True)
    rtgs = torch.randn(2, 5, 1, device=device, requires_grad=True)
    actions = torch.randn(2, 5, 8, device=device, requires_grad=True)

    u_t = fusion(states=states, rtgs=rtgs, actions=actions)
    loss = u_t.sum()
    loss.backward()

    assert states.grad is not None, "Gradients did not backprop to states"
    assert rtgs.grad is not None, "Gradients did not backprop to rtgs"
    assert actions.grad is not None, "Gradients did not backprop to actions"
    assert torch.isfinite(states.grad).all()


def test_visual_patch_encoder(device: torch.device) -> None:
    encoder = VisualPatchEncoder(in_channels=1, image_size=64, d_model=128).to(device)

    # 4D input (B, C, H, W)
    x4d = torch.randn(2, 1, 64, 64, device=device)
    out4d = encoder(x4d)
    assert out4d.shape == (2, 128)

    # 5D input (B, T, C, H, W)
    x5d = torch.randn(2, 4, 1, 64, 64, device=device)
    out5d = encoder(x5d)
    assert out5d.shape == (2, 4, 128)
