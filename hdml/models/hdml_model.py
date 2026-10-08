from __future__ import annotations

import torch
import torch.nn as nn
from hdml.models.fusion import CrossModalFusion
from hdml.models.mamba3_backbone import Mamba3CognitiveBackbone
from hdml.models.liquid_head import CfCActionFilter
from hdml.models.flow_policy import FlowPolicy, GaussianActionPolicy
from hdml.models.hiqc_critic import HiQCCritic
from hdml.utils.config import ModelConfig


class HDMLModel(nn.Module):
    """Windowed Mamba-1/RoPE policy with a direct MLP action head and optional CfC.

    ``from_config`` enables CfC. Legacy flow/critic modules remain in the state
    dictionary for loading old checkpoints, but are inactive in the supported
    trainer and are not a validated dual-rate controller.
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
        self.d_subgoal = prop_dim if d_subgoal is None else d_subgoal

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

        # 2. Windowed Mamba-1/RoPE backbone
        self.mamba_backbone = Mamba3CognitiveBackbone(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_mamba_layers,
            d_subgoal=self.d_subgoal,
            prop_dim=prop_dim,
        )

        # 3. Direct Action Generation Head (High-capacity Mamba sequence policy)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, action_dim),
            nn.Tanh(),
        )

        self.chunk_size = 4
        self.flow_policy = None
        self.hiqc_critic = None
        self.cfc_filter = None

        # Legacy inactive value network, retained only for checkpoint compatibility.
        self.value_net = nn.Sequential(
            nn.Linear(prop_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> HDMLModel:
        """Create an HDMLModel instance from a ModelConfig."""
        model = cls(
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
        model.chunk_size = cfg.chunk_size
        flow_context_dim = cfg.d_subgoal + cfg.prop_dim + 1
        critic_context_dim = cfg.d_subgoal + cfg.prop_dim

        if cfg.action_policy == "flow":
            model.flow_policy = FlowPolicy(
                action_dim=cfg.action_dim,
                chunk_size=model.chunk_size,
                context_dim=flow_context_dim,
                hidden_dim=256,
            )
        else:
            model.flow_policy = GaussianActionPolicy(
                action_dim=cfg.action_dim,
                chunk_size=model.chunk_size,
                context_dim=flow_context_dim,
                hidden_dim=256,
            )
        model.hiqc_critic = HiQCCritic(
            state_dim=critic_context_dim,
            action_dim=cfg.action_dim,
            chunk_size=model.chunk_size,
            hidden_dim=256
        )
        model.cfc_filter = CfCActionFilter(
            action_dim=cfg.action_dim,
            chunk_size=1,
            state_dim=cfg.d_model,
            units=cfg.cfc_units,
            backbone_units=cfg.cfc_backbone_units,
            backbone_layers=cfg.cfc_backbone_layers,
            residual=cfg.cfc_residual,
        )
            
        return model

    def forward(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        visual_frames: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sequence-level forward pass for training and inference.

        Args:
            states: (B, T, prop_dim)
            rtgs: (B, T, 1) or (B, T)
            actions: (B, T, action_dim) | None
            timesteps: (B, T) | None
            visual_frames: (B, T, C, H, W) | None
            hx: Hidden state tensor for CfC | None

        Returns:
            actions_pred: Predicted actions (B, T, action_dim)
            subgoals_pred: Predicted subgoals / future state representations (B, T, d_subgoal)
            values_pred: Value predictions (B, T, 1)
            next_states_pred: Forward dynamics predictions (B, T, prop_dim)
            next_hx: Updated CfC hidden state | None
        """
        # 1. Cross-Modal Fusion
        u_t = self.fusion(
            states=states,
            rtgs=rtgs,
            actions=actions,
            timesteps=timesteps,
            visual_frames=visual_frames,
        )

        # 2. Windowed sequence features
        subgoals_pred, latent_features, values_pred, next_states_pred = self.mamba_backbone(u_t)

        # 3. Direct Action Prediction + Liquid CfC ODE Filter
        raw_actions = self.action_head(latent_features)
        if self.cfc_filter is not None:
            actions_pred, next_hx = self.cfc_filter(raw_actions, latent_features, hx=hx)
        else:
            actions_pred = raw_actions
            next_hx = None

        return actions_pred, subgoals_pred, values_pred, next_states_pred, next_hx

    @torch.inference_mode()
    def get_action(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        visual_frames: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
            extras: Dictionary containing planned subgoals and predicted action
        """
        if states.shape[1] > 1 and hx is not None:
            raise ValueError("Overlapping context must use hx=None; otherwise history is replayed twice")
        actions_pred, subgoals_pred, values_pred, next_states_pred, next_hx = self.forward(
            states=states,
            rtgs=rtgs,
            actions=actions,
            timesteps=timesteps,
            visual_frames=visual_frames,
            hx=hx,
        )

        last_action = actions_pred[:, -1, :]
        last_subgoal = subgoals_pred[:, -1, :]

        return last_action, next_hx, {"subgoal": last_subgoal, "action": last_action}

    @torch.inference_mode()
    def act_from_subgoal(
        self,
        subgoal: torch.Tensor,
        current_prop: torch.Tensor,
        rtg: torch.Tensor,
        hx: torch.Tensor | None = None,
        num_flow_steps: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reject the untrained legacy dual-rate path explicitly."""
        raise ValueError(
            "Dual-rate act_from_subgoal is unsupported: the active trainer does not "
            "train the flow/critic path. Use windowed get_action with macro_interval=1."
        )
