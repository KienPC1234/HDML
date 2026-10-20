"""Core Architecture and Component Tests for HDML."""

from __future__ import annotations

import torch
import torch.nn as nn

from hdml.models.mamba3_native import Mamba3NativeBackbone, Mamba3NativeBlock
from hdml.models.mamba3_portable import PortableMamba3Backbone, PortableMamba3Block
from hdml.models.fusion import CrossModalFusion
from hdml.models.liquid_head import CfCActionFilter
from hdml.models.flow_policy import FlowPolicy
from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import ModelConfig


def test_mamba3_native_and_portable_backbone() -> None:
    # 1. Portable Mamba-3 (CPU/CUDA)
    x = torch.randn(2, 10, 128)
    p_block = PortableMamba3Block(d_model=128, d_state=16, expand=2, headdim=64)
    p_out = p_block(x)
    assert p_out.shape == x.shape

    p_backbone = PortableMamba3Backbone(
        d_model=128,
        num_layers=2,
        d_state=16,
        expand=2,
        d_subgoal=64,
        prop_dim=17,
        headdim=64,
    )
    subgoals, latent_features, values, s_pred = p_backbone(x)
    assert subgoals.shape == (2, 10, 64)
    assert latent_features.shape == (2, 10, 128)
    assert values.shape == (2, 10, 1)
    assert s_pred.shape == (2, 10, 17)

    # 2. Native Mamba-3 (CUDA)
    if torch.cuda.is_available():
        x_cuda = x.cuda()
        n_block = Mamba3NativeBlock(d_model=128, d_state=16, expand=2, headdim=64).cuda()
        n_out = n_block(x_cuda)
        assert n_out.shape == x_cuda.shape

        n_backbone = Mamba3NativeBackbone(
            d_model=128,
            num_layers=2,
            d_state=16,
            expand=2,
            d_subgoal=64,
            prop_dim=17,
            headdim=64,
        ).cuda()
        subgoals, latent_features, values, s_pred = n_backbone(x_cuda)
        assert subgoals.shape == (2, 10, 64)
        assert latent_features.shape == (2, 10, 128)
        assert values.shape == (2, 10, 1)
        assert s_pred.shape == (2, 10, 17)


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
    model = HDMLModel.from_config(ModelConfig(prop_dim=17, action_dim=6, d_model=128,
                                                num_mamba_layers=3, cfc_units=32)).to(device)
    assert model.cfc_filter is not None
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



def test_configured_cfc_receives_gradients() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HDMLModel.from_config(ModelConfig(prop_dim=17, action_dim=6)).to(device)
    out = model(torch.randn(2, 5, 17, device=device), torch.zeros(2, 5, 1, device=device))[0]
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.cfc_filter.parameters())


def test_hierarchical_micro_tier_shapes_and_grad() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HDMLModel.from_config(
        ModelConfig(prop_dim=17, action_dim=6, d_model=64, num_mamba_layers=2, d_subgoal=64)
    ).to(device)
    action, _ = model.act_from_subgoal(
        torch.zeros(1, model.d_subgoal, device=device),
        torch.zeros(1, 17, device=device),
        torch.zeros(1, 1, device=device),
    )
    assert action.shape == (1, 6)
    assert torch.isfinite(action).all()


def test_cfc_full_sequence_matches_stepwise_carried_state() -> None:
    """Training (full sequence) and rollout (token-by-token hx) must agree."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfc = CfCActionFilter(
        action_dim=6, chunk_size=1, state_dim=16, units=24, residual=0.5
    ).to(device)
    actions = torch.randn(3, 7, 6, device=device)
    latent = torch.randn(3, 7, 16, device=device)

    full, full_hx = cfc(actions, latent, hx=None)
    stepwise: list[torch.Tensor] = []
    hx: torch.Tensor | None = None
    for t in range(actions.shape[1]):
        out_t, hx = cfc(actions[:, t], latent[:, t], hx=hx)
        stepwise.append(out_t)
    stepwise_tensor = torch.stack(stepwise, dim=1)

    torch.testing.assert_close(stepwise_tensor, full, atol=1e-6, rtol=1e-5)
    assert hx is not None
    torch.testing.assert_close(hx, full_hx, atol=1e-6, rtol=1e-5)


def test_cfc_reset_starts_from_zero_state() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfc = CfCActionFilter(
        action_dim=5, chunk_size=1, state_dim=12, units=16, residual=0.5
    ).to(device)
    actions = torch.randn(2, 5, 5, device=device)
    latent = torch.randn(2, 5, 12, device=device)

    full, _ = cfc(actions, latent, hx=None)
    _, hx0 = cfc(actions[:, 0], latent[:, 0], hx=None)
    carried, _ = cfc(actions[:, 1], latent[:, 1], hx=hx0)
    reset, _ = cfc(actions[:, 1], latent[:, 1], hx=None)

    torch.testing.assert_close(carried, full[:, 1], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reset, cfc(actions[:, 1], latent[:, 1], hx=None)[0])
    assert not torch.allclose(carried, reset, atol=1e-7, rtol=1e-7)


def test_get_action_carries_cfc_state_across_long_context() -> None:
    """The old guard rejected hx with T>1; stateful rollout must accept it."""
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HDMLModel(
        prop_dim=9,
        action_dim=4,
        d_model=32,
        num_mamba_layers=1,
        d_subgoal=9,
        chunk_size=3,
        use_native_mamba3=False,
        use_cfc=True,
        cfc_units=8,
        cfc_residual=0.5,
        use_critic=False,
        num_flow_steps=2,
    ).to(device).eval()
    model.deterministic = True
    states = torch.randn(1, 5, 9, device=device)
    actions = torch.randn(1, 5, 4, device=device)
    rtgs = torch.zeros(1, 5, 1, device=device)
    timesteps = torch.arange(5, device=device).unsqueeze(0)

    action_reset, hx_reset, _ = model.get_action(
        states=states, rtgs=rtgs, actions=actions, timesteps=timesteps, hx=None
    )
    assert hx_reset is not None
    action_carried, hx_carried, _ = model.get_action(
        states=states, rtgs=rtgs, actions=actions, timesteps=timesteps, hx=hx_reset
    )
    assert hx_carried is not None and torch.isfinite(hx_carried).all()
    assert action_reset.shape == action_carried.shape == (1, 4)
    assert not torch.allclose(action_reset, action_carried, atol=1e-7, rtol=1e-7)
