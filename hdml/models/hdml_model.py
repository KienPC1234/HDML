from __future__ import annotations

import torch
import torch.nn as nn
from hdml.models.fusion import CrossModalFusion
from hdml.models.backbone import MambaCognitiveBackbone
from hdml.models.liquid_head import LiquidReactiveControlHead
from hdml.utils.config import ModelConfig


class HDMLModel(nn.Module):
    """Toàn bộ mô hình Hierarchical Decision Mamba-Liquid (HDML).

    Integrates:
      1. CrossModalFusion: Encodes and fuses proprioceptive kinematics, Return-to-Go,
         actions, and temporal embeddings into U_t.
      2. MambaCognitiveBackbone: Selective State Space model (S6) for O(N) long-horizon
         latent subgoal planning c_t.
      3. LiquidReactiveControlHead: MIT CfC continuous-time ordinary differential
         equation policy head for high-frequency reactive motor torque control.

    Shape Contract:
        Input (Sequence Training):
            states:        (Batch, Seq_Len, prop_dim)
            rtgs:          (Batch, Seq_Len, 1)
            actions:       (Batch, Seq_Len, action_dim) | None
            timesteps:     (Batch, Seq_Len) | None
            visual_frames: (Batch, Seq_Len, C, H, W) | None
            hx:            Previous Liquid hidden state | None
        Output:
            actions_pred:  (Batch, Seq_Len, action_dim)
            subgoals_pred: (Batch, Seq_Len, d_subgoal)
            values_pred:   (Batch, Seq_Len, 1)
            next_hx:       Updated Liquid hidden state
    """

    def __init__(
        self,
        prop_dim: int = 27,
        action_dim: int = 8,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_mamba_layers: int = 3,
        d_subgoal: int = 64,
        cfc_units: int = 32,
        cfc_backbone_units: int = 64,
        cfc_backbone_layers: int = 1,
        use_visual: bool = False,
        visual_channels: int = 1,
        visual_image_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.d_subgoal = d_subgoal

        # 1. Multi-modal Tokenizer and Fusion Layer
        self.fusion = CrossModalFusion(
            prop_dim=prop_dim,
            action_dim=action_dim,
            d_model=d_model,
            use_visual=use_visual,
            visual_channels=visual_channels,
            visual_image_size=visual_image_size,
            dropout=dropout,
        )

        # 2. Mamba Cognitive Planning Backbone
        self.mamba_backbone = MambaCognitiveBackbone(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_mamba_layers,
            d_subgoal=d_subgoal,
        )

        # 3. Liquid Reactive Control Head
        self.liquid_head = LiquidReactiveControlHead(
            d_subgoal=d_subgoal,
            prop_dim=prop_dim,
            action_dim=action_dim,
            units=cfc_units,
            backbone_units=cfc_backbone_units,
            backbone_layers=cfc_backbone_layers,
        )

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> HDMLModel:
        """Create an HDMLModel instance from a ModelConfig."""
        return cls(
            prop_dim=cfg.prop_dim,
            action_dim=cfg.action_dim,
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            num_mamba_layers=cfg.num_mamba_layers,
            d_subgoal=cfg.d_subgoal,
            cfc_units=cfg.cfc_units,
            cfc_backbone_units=cfg.cfc_backbone_units,
            cfc_backbone_layers=cfg.cfc_backbone_layers,
            use_visual=cfg.use_visual,
            visual_channels=cfg.visual_channels,
            visual_image_size=cfg.visual_image_size,
            dropout=cfg.dropout,
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
        """Sequence-level forward pass for training.

        Args:
            states: (B, T, prop_dim)
            rtgs: (B, T, 1) or (B, T)
            actions: (B, T, action_dim) | None
            timesteps: (B, T) | None
            visual_frames: (B, T, C, H, W) | None
            hx: Hidden state tensor for CfC | None

        Returns:
            actions_pred: (B, T, action_dim)
            subgoals_pred: (B, T, d_subgoal)
            values_pred: (B, T, 1)
            next_hx: Updated CfC hidden state
        """
        # 1. Cross-Modal Fusion
        u_t = self.fusion(
            states=states,
            rtgs=rtgs,
            actions=actions,
            timesteps=timesteps,
            visual_frames=visual_frames,
        )

        # 2. Mamba Cognitive Planning
        subgoals_pred, values_pred, _ = self.mamba_backbone(u_t)

        # 3. Liquid Reactive Actuation
        actions_pred, next_hx = self.liquid_head(
            subgoals=subgoals_pred,
            current_prop=states,
            hx=hx,
        )

        return actions_pred, subgoals_pred, values_pred, next_hx

    @torch.inference_mode()
    def get_action(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        visual_frames: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform step inference for closed-loop online rollouts.

        Extracts the action at the final timestep index.

        Args:
            states: Context trajectory states, shape (B, T, prop_dim)
            rtgs: Context trajectory RTGs, shape (B, T, 1) or (B, T)
            actions: Context trajectory actions, shape (B, T, action_dim) or None
            timesteps: Context trajectory timesteps, shape (B, T) or None
            visual_frames: Context visual frames or None
            hx: Liquid hidden state or None

        Returns:
            last_action: Action for the current step, shape (B, action_dim)
            next_hx: Updated Liquid hidden state
            last_subgoal: Planned subgoal for the current step, shape (B, d_subgoal)
        """
        actions_pred, subgoals_pred, _, next_hx = self.forward(
            states=states,
            rtgs=rtgs,
            actions=actions,
            timesteps=timesteps,
            visual_frames=visual_frames,
            hx=hx,
        )

        last_action = actions_pred[:, -1, :]
        last_subgoal = subgoals_pred[:, -1, :]

        return last_action, next_hx, last_subgoal
