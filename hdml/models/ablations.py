from __future__ import annotations

import torch
import torch.nn as nn
from ncps.torch import CfC

from hdml.models.fusion import CrossModalFusion
from hdml.models.mamba3_backbone import Mamba3CognitiveBackbone


class _CfCLiquidHead(nn.Module):
    """Closed-Form Continuous-Time (CfC) single-step action head.

    Maps a latent subgoal plus instantaneous proprioception to a bounded continuous
    action via an ODE-based CfC cell. Used by the ``TransformerLiquidHeadAblation``
    to isolate the contribution of the Mamba backbone while retaining the liquid head.

    Shape Contract:
        Input:
            subgoals:     (B, T, d_subgoal) or (B, d_subgoal)
            current_prop: (B, T, prop_dim)  or (B, prop_dim)
        Output:
            actions:  (B, T, action_dim) or (B, action_dim), bounded in [-1, 1]
            next_hx:  CfC hidden state
    """

    def __init__(
        self,
        d_subgoal: int,
        prop_dim: int,
        action_dim: int,
        units: int = 32,
        backbone_units: int = 64,
        backbone_layers: int = 1,
    ) -> None:
        super().__init__()
        self.d_subgoal = d_subgoal
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.units = units
        self.input_dim = d_subgoal + prop_dim

        self.cfc = CfC(
            input_size=self.input_dim,
            units=units,
            proj_size=None,
            return_sequences=True,
            batch_first=True,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=0.0,
            mode="default",
        )

        self.action_out = nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        subgoals: torch.Tensor,
        current_prop: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_2d = subgoals.ndim == 2

        if is_2d:
            assert current_prop.ndim == 2, (
                f"Expected current_prop to be 2D when subgoals is 2D, got {current_prop.shape}"
            )
            subgoals_seq = subgoals.unsqueeze(1)
            prop_seq = current_prop.unsqueeze(1)
        else:
            assert subgoals.ndim == 3, f"Expected subgoals 3D, got {subgoals.shape}"
            assert current_prop.ndim == 3, f"Expected current_prop 3D, got {current_prop.shape}"
            subgoals_seq = subgoals
            prop_seq = current_prop

        assert subgoals_seq.shape[-1] == self.d_subgoal, (
            f"Expected subgoal dim {self.d_subgoal}, got {subgoals_seq.shape[-1]}"
        )
        assert prop_seq.shape[-1] == self.prop_dim, (
            f"Expected prop dim {self.prop_dim}, got {prop_seq.shape[-1]}"
        )

        cfc_input = torch.cat([subgoals_seq, prop_seq], dim=-1)
        cfc_out, next_hx = self.cfc(cfc_input, hx)
        actions = self.action_out(cfc_out)

        if is_2d:
            actions = actions.squeeze(1)

        return actions, next_hx


class MambaMLPHeadAblation(nn.Module):
    """Ablation Variant A: Mamba-3 Backbone + Standard MLP Action Head.

    Isolates the Mamba backbone without the Liquid/CfC filter. Used in ablation studies
    to show that mechanical jerk reduction is driven by the liquid ODE filter, not the
    Mamba backbone alone.
    """

    def __init__(
        self,
        prop_dim: int = 27,
        action_dim: int = 8,
        d_model: int = 128,
        num_mamba_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.d_model = d_model

        self.fusion = CrossModalFusion(
            prop_dim=prop_dim,
            action_dim=action_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.backbone = Mamba3CognitiveBackbone(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_mamba_layers,
            d_subgoal=d_model,
            prop_dim=prop_dim,
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        visual_frames: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_t = self.fusion(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps, visual_frames=visual_frames)
        _, latent_features, values_pred, _ = self.backbone(u_t)
        actions_pred = self.mlp_head(latent_features)
        return actions_pred, values_pred

    @torch.inference_mode()
    def get_action(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions_pred, _ = self.forward(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)
        return actions_pred[:, -1, :]


class TransformerLiquidHeadAblation(nn.Module):
    """Ablation Variant B: Causal Transformer + Closed-Form Liquid (CfC) Head.

    Replaces the Mamba-3 backbone with a standard Causal Transformer Encoder while
    retaining the liquid CfC head. Isolates the contribution of the O(N)/O(1) Mamba
    backbone to control frequency and long-horizon state memory.
    """

    def __init__(
        self,
        prop_dim: int = 27,
        action_dim: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        d_subgoal: int = 64,
        cfc_units: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.d_subgoal = d_subgoal

        self.fusion = CrossModalFusion(
            prop_dim=prop_dim,
            action_dim=action_dim,
            d_model=d_model,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.subgoal_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_subgoal),
            nn.LayerNorm(d_subgoal),
        )

        self.liquid_head = _CfCLiquidHead(
            d_subgoal=d_subgoal,
            prop_dim=prop_dim,
            action_dim=action_dim,
            units=cfc_units,
        )

        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        visual_frames: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = states.shape
        u_t = self.fusion(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps, visual_frames=visual_frames)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=states.device)
        h = self.transformer(u_t, mask=causal_mask, is_causal=True)

        subgoals = self.subgoal_head(h)
        values_pred = self.value_head(h)

        actions_pred, next_hx = self.liquid_head(subgoals=subgoals, current_prop=states, hx=hx)
        return actions_pred, subgoals, values_pred, next_hx

    @torch.inference_mode()
    def get_action(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions_pred, subgoals, _, next_hx = self.forward(
            states=states, rtgs=rtgs, actions=actions, timesteps=timesteps, hx=hx
        )
        return actions_pred[:, -1, :], next_hx, subgoals[:, -1, :]
