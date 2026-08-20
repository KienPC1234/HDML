"""Core Architecture and Component Tests for HDML."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hdml.models.mamba3_backbone import Mamba3CognitiveBackbone, Mamba3Block, apply_rope
from hdml.models.fusion import CrossModalFusion, VisualPatchEncoder
from hdml.models.liquid_head import CfCActionFilter
from hdml.models.flow_policy import FlowPolicy
from hdml.models.hiqc_critic import HiQCCritic
from hdml.models.hdml_model import HDMLModel
from hdml.training.losses import HDMLLoss


def test_rope_and_mamba3_backbone() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 10, 128, device=device)
    rope_x = apply_rope(x, dim=-1)
    assert rope_x.shape == x.shape

    block = Mamba3Block(d_model=128, d_state=16, d_conv=4, expand=2).to(device)
    out = block(x)
    assert out.shape == x.shape

    backbone = Mamba3CognitiveBackbone(
        d_model=128,
        num_layers=3,
        d_state=16,
        d_conv=4,
        expand=2,
        d_subgoal=64,
        prop_dim=17,
    ).to(device)
    subgoals, latent_features, values, s_pred = backbone(x)
    assert subgoals.shape == (2, 10, 64)
    assert latent_features.shape == (2, 10, 128)


def test_cross_modal_fusion() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion = CrossModalFusion(prop_dim=17, action_dim=6, d_model=128).to(device)
    states = torch.randn(2, 10, 17, device=device)
    actions = torch.randn(2, 10, 6, device=device)
    rtgs = torch.randn(2, 10, 1, device=device)
    timesteps = torch.arange(10, device=device).unsqueeze(0).repeat(2, 1)

    u_t = fusion(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)
    assert u_t.shape == (2, 10, 128)


def test_cfc_and_flow_policy() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfc = CfCActionFilter(action_dim=6, chunk_size=4, state_dim=17, units=32).to(device)
    act_chunk = torch.randn(2, 10, 4, 6, device=device)
    state = torch.randn(2, 10, 17, device=device)
    a_out, _ = cfc(act_chunk, state)
    assert a_out.shape == (2, 10, 4, 6)

    flow = FlowPolicy(action_dim=6, chunk_size=4, context_dim=81).to(device)
    context = torch.randn(2, 81, device=device)
    act_chunk_single = torch.randn(2, 4, 6, device=device)
    target_v, pred_v, _ = flow.forward_train(act_chunk_single, context)
    assert target_v.shape == pred_v.shape


def test_hdml_full_model_and_loss() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HDMLModel(
        prop_dim=17,
        action_dim=6,
        d_model=128,
        num_mamba_layers=3,
        d_state=16,
        d_conv=4,
        expand=2,
        cfc_units=32,
    ).to(device)
    states = torch.randn(2, 10, 17, device=device)
    actions = torch.randn(2, 10, 6, device=device)
    rtgs = torch.randn(2, 10, 1, device=device)
    timesteps = torch.arange(10, device=device).unsqueeze(0).repeat(2, 1)

    acts_pred, subgoals_pred, values_pred, rtg_pred, _ = model(
        states=states,
        rtgs=rtgs,
        actions=actions,
        timesteps=timesteps,
    )
    assert acts_pred.shape == (2, 10, 6)
    assert subgoals_pred.shape == (2, 10, 64)

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(acts_pred, actions)
    assert loss.item() > 0.0
    loss.backward()
    assert model.fusion.prop_encoder[0].weight.grad is not None
