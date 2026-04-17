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
    """Toàn bộ mô hình Hierarchical Decision Mamba-Liquid (HDML).

    Integrates:
      1. CrossModalFusion: Encodes and fuses proprioceptive kinematics, Return-to-Go,
         actions, and temporal embeddings into U_t.
      2. Mamba3CognitiveBackbone: Selective State Space model (Mamba-3 emulation) for
         O(N) long-horizon latent subgoal planning c_t.
      3. FlowPolicy: generative flow-matching action-chunk generator (replaces the
         unimodal Gaussian policy head).
      4. HiQCCritic: chunk-level value function for Q-guided flow training.
      5. CfCActionFilter: MIT CfC continuous-time ODE filter for high-frequency
         reactive motor torque smoothing.

    Shape Contract:
        Input (Sequence Training):
            states:        (Batch, Seq_Len, prop_dim)
            rtgs:          (Batch, Seq_Len, 1)
            actions:       (Batch, Seq_Len, action_dim) | None
            timesteps:     (Batch, Seq_Len) | None
            visual_frames: (Batch, Seq_Len, C, H, W) | None
            hx:            Previous Liquid hidden state | None
        Output:
            actions_pred:  None (flow-matching head handled separately)
            subgoals_pred: (Batch, Seq_Len, d_subgoal)
            values_pred:   (Batch, Seq_Len, 1)
            next_states_pred: (Batch, Seq_Len, prop_dim)
            next_hx:       None (CfC hidden state handled in get_action)
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

        # 2. Mamba-3 Cognitive Planning Backbone (HDML)
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

        # Independent IQL value network V(s): a state-only MLP used for the
        # k-step Bellman backup and the advantage weighting (Q - V).
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

        # 2. Mamba-3 Cognitive Planning
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
        """Fast micro-actuation without re-running the Mamba backbone.

        Reuses the latest latent subgoal from the macro-planner to generate the next
        action chunk via flow sampling + CfC filtering. This is the low-level
        high-frequency path used between macro-planning invocations.

        Args:
            subgoal: Latent subgoal, shape (B, d_subgoal).
            current_prop: Current proprioceptive state, shape (B, prop_dim).
            rtg: Current scaled return-to-go, shape (B, 1).
            hx: CfC hidden state from the previous step, or None.
            num_flow_steps: Euler integration steps for flow sampling.

        Returns:
            action: Next action, shape (B, action_dim).
            next_hx: Updated CfC hidden state.
        """
        flow_context = torch.cat([subgoal, current_prop, rtg], dim=-1)
        critic_context = torch.cat([subgoal, current_prop], dim=-1)
        raw_chunk = self.flow_policy.sample(context=flow_context, num_steps=num_flow_steps)

        if self.cfc_filter is not None:
            action_chunk, next_hx = self.cfc_filter(raw_chunk, critic_context, hx)
        else:
            action_chunk, next_hx = raw_chunk, None

        return action_chunk[:, 0, :], next_hx
