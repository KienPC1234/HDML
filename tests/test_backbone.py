from __future__ import annotations

import pytest
import torch
from hdml.models.backbone import MambaCognitiveBackbone


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_mamba_backbone_forward_shapes(device: torch.device) -> None:
    batch_size = 4
    seq_len = 16
    d_model = 128
    d_subgoal = 64

    backbone = MambaCognitiveBackbone(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=3,
        d_subgoal=d_subgoal,
    ).to(device)

    u_t = torch.randn(batch_size, seq_len, d_model, device=device)
    subgoals, values, latent = backbone(u_t)

    assert subgoals.shape == (batch_size, seq_len, d_subgoal)
    assert values.shape == (batch_size, seq_len, 1)
    assert latent.shape == (batch_size, seq_len, d_model)
    assert torch.isfinite(subgoals).all()
    assert torch.isfinite(values).all()


def test_mamba_backbone_gradients(device: torch.device) -> None:
    backbone = MambaCognitiveBackbone(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=2,
        d_subgoal=32,
    ).to(device)

    u_t = torch.randn(2, 8, 64, device=device, requires_grad=True)
    subgoals, values, _ = backbone(u_t)

    loss = subgoals.sum() + values.sum()
    loss.backward()

    assert u_t.grad is not None
    assert torch.isfinite(u_t.grad).all()

    # Check that model parameters received gradients
    for name, param in backbone.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None grad"
