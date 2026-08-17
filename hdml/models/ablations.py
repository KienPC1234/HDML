from __future__ import annotations

import torch
import torch.nn as nn
from hdml.models.fusion import CrossModalFusion
from hdml.models.backbone import MambaCognitiveBackbone
from hdml.models.liquid_head import LiquidReactiveControlHead


class MambaMLPHeadAblation(nn.Module):
    """Ablation Variant A: Decision Mamba Backbone + Standard MLP Action Head.
    
    This variant isolates the Mamba backbone without the Liquid Neural Network head.
    Used in scientific ablation studies to prove that mechanical jerk reduction is
    specifically driven by the Closed-Form Liquid ODE head, not merely the Mamba backbone.
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

        self.backbone = MambaCognitiveBackbone(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_mamba_layers,
            d_subgoal=d_model,
        )

        # Standard Multi-Layer Perceptron Action Head (Discrete Step Prediction)
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
        _, values_pred, latent_features = self.backbone(u_t)
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
    """Ablation Variant B: Causal Multi-Head Transformer + Closed-Form Liquid Head (CfC).
    
    This variant replaces the Mamba S6 backbone with a standard Causal Transformer Encoder,
    while retaining the MIT Closed-Form Continuous-Time Liquid Head.
    Used in scientific ablation studies to prove that O(1) state memory and high control
    frequency (> 340 Hz) are uniquely enabled by the Mamba S6 backbone.
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

        self.liquid_head = LiquidReactiveControlHead(
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
