"""Tests for HDML-Foundation Multi-Embodiment Architecture and Pipelines."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import numpy as np

from hdml.models.foundation import HDMLFoundationModel, UniversalEmbodimentAdapter
from hdml.data.multi_embodiment_dataset import MultiEmbodimentDataset, collate_multi_embodiment


def test_universal_adapter_shapes_and_gradients() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = UniversalEmbodimentAdapter(prop_dim=53, action_dim=12, d_model=384).to(device)

    states = torch.randn(4, 30, 53, device=device)
    actions = torch.randn(4, 30, 12, device=device)
    latents = torch.randn(4, 30, 384, device=device)

    st_feats = adapter.project_state(states)
    act_feats = adapter.project_action(actions)
    decoded_acts = adapter.decode_action(latents)

    assert st_feats.shape == (4, 30, 384)
    assert act_feats.shape == (4, 30, 384)
    assert decoded_acts.shape == (4, 30, 12)
    assert (decoded_acts >= -1.0).all() and (decoded_acts <= 1.0).all()


def test_hdml_foundation_multi_embodiment_forward_backward() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HDMLFoundationModel(
        d_model=128,
        num_mamba_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        cfc_units=32,
        cfc_backbone_units=64,
        device=device,
    )

    # Register 3 distinct robot morphologies
    model.register_embodiment("cheetah", prop_dim=17, action_dim=6)
    model.register_embodiment("ant", prop_dim=27, action_dim=8)
    model.register_embodiment("a1", prop_dim=53, action_dim=12)

    # 1. Forward pass for Cheetah
    c_states = torch.randn(2, 20, 17, device=device)
    c_actions = torch.randn(2, 20, 6, device=device)
    c_rtgs = torch.randn(2, 20, 1, device=device)
    c_times = torch.arange(20, device=device).unsqueeze(0).repeat(2, 1)

    c_acts, c_int, c_val, c_rtg_pred, _ = model(
        states=c_states,
        rtgs=c_rtgs,
        actions=c_actions,
        timesteps=c_times,
        embodiment_name="cheetah",
        embodiment_idx=0,
    )
    assert c_acts.shape == (2, 20, 6)
    assert c_int.shape == (2, 20, 64)

    # 2. Forward pass for Unitree A1
    a_states = torch.randn(2, 20, 53, device=device)
    a_actions = torch.randn(2, 20, 12, device=device)
    a_rtgs = torch.randn(2, 20, 1, device=device)
    a_times = torch.arange(20, device=device).unsqueeze(0).repeat(2, 1)

    a_acts, a_int, a_val, a_rtg_pred, _ = model(
        states=a_states,
        rtgs=a_rtgs,
        actions=a_actions,
        timesteps=a_times,
        embodiment_name="a1",
        embodiment_idx=2,
    )
    assert a_acts.shape == (2, 20, 12)

    # 3. Test Backward Gradient Flow
    loss = (
        nn.functional.smooth_l1_loss(c_acts, torch.zeros_like(c_acts))
        + nn.functional.smooth_l1_loss(a_acts, torch.zeros_like(a_acts))
        + c_int.sum()
        + c_rtg_pred.sum()
    )
    loss.backward()

    assert model.fusion_linear.weight.grad is not None
    assert model.fusion_norm.weight.grad is not None
    for layer in model.mamba_backbone.layers:
        for p in layer.parameters():
            if p.requires_grad:
                assert p.grad is not None


def test_hdml_foundation_backbone_freezing() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HDMLFoundationModel(d_model=64, num_mamba_layers=2, device=device)
    model.register_embodiment("robot_a", prop_dim=10, action_dim=2)

    # Freeze core
    model.freeze_backbone()

    # Verify backbone parameters do not require grad
    for p in model.mamba_backbone.parameters():
        assert not p.requires_grad
    for p in model.cfc_filter.parameters():
        assert not p.requires_grad
    for p in model.fusion_linear.parameters():
        assert not p.requires_grad

    # Register new target embodiment after freeze
    adapter_b = model.register_embodiment("robot_b", prop_dim=20, action_dim=4)
    for p in adapter_b.parameters():
        assert p.requires_grad


def test_multi_embodiment_dataset_collation(tmp_path) -> None:
    # Create synthetic trajectory files for two embodiments
    cheetah_path = tmp_path / "cheetah.npz"
    ant_path = tmp_path / "ant.npz"

    np.savez(
        cheetah_path,
        observations=np.random.randn(100, 17).astype(np.float32),
        actions=np.random.uniform(-1, 1, size=(100, 6)).astype(np.float32),
        rewards=np.random.randn(100).astype(np.float32),
        terminals=np.zeros(100, dtype=bool),
    )
    np.savez(
        ant_path,
        observations=np.random.randn(80, 27).astype(np.float32),
        actions=np.random.uniform(-1, 1, size=(80, 8)).astype(np.float32),
        rewards=np.random.randn(80).astype(np.float32),
        terminals=np.zeros(80, dtype=bool),
    )

    dataset = MultiEmbodimentDataset(
        embodiment_paths={"cheetah": cheetah_path, "ant": ant_path},
        context_length=15,
    )
    assert len(dataset) > 0

    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_multi_embodiment)
    batch_dict = next(iter(loader))

    assert len(batch_dict) > 0
    for name, batch in batch_dict.items():
        assert "states" in batch
        assert "actions_in" in batch
        assert "actions_target" in batch
        assert "rtgs" in batch
        assert batch["states"].shape[1] == 15
