from __future__ import annotations

import torch
import torch.nn as nn
from hdml.models.fusion import CrossModalFusion
from hdml.models.mamba3_backbone import Mamba3CognitiveBackbone
from hdml.models.mamba3_native import Mamba3NativeBackbone
from hdml.models.liquid_head import CfCActionFilter
from hdml.models.flow_policy import FlowPolicy, GaussianActionPolicy
from hdml.models.hiqc_critic import HiQCCritic
from hdml.utils.config import ModelConfig


class HDMLModel(nn.Module):
    """Hierarchical Decision Mamba-Liquid policy.

    Backbone: a native Mamba-3 selective SSM (built-in rotary state-space
    embedding, trapezoidal discretisation) when ``use_native_mamba3`` is true;
    the legacy Mamba-1 + input-RoPE block is kept for old checkpoints.

    Policy: Flow Matching over an action chunk (``flow_policy``). At every step
    the flow field is integrated from a deterministic prior to a local action
    chunk; the first action of the chunk is refined by a continuous-time CfC
    filter before it is sent to the actuator.
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
        cfc_residual: float = 0.05,
        use_visual: bool = False,
        visual_channels: int = 1,
        visual_image_size: int = 64,
        dropout: float = 0.1,
        chunk_size: int = 5,
        action_policy: str = "flow",
        use_native_mamba3: bool = True,
        mamba3_headdim: int = 64,
        mamba3_ngroups: int = 1,
        mamba3_rope_fraction: float = 0.5,
        mamba3_outproj_norm: bool = False,
        mamba3_chunk: int = 64,
    ) -> None:
        super().__init__()
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.d_subgoal = prop_dim if d_subgoal is None else d_subgoal
        self.use_flow = True
        self.num_flow_steps = 4
        self.chunk_size = chunk_size
        # Deterministic flow solve (conditional-mean estimate) is the default for
        # evaluation: it scores much higher than stochastic sampling and is the
        # standard way to evaluate a flow-matching policy. Set False to sample.
        self.deterministic = True

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

        # 2. Selective SSM backbone (native Mamba-3 by default).
        #    Native Mamba-3 uses fused CUDA/Triton kernels; fall back to the
        #    legacy Mamba-1+RoPE block on CPU-only hosts.
        if use_native_mamba3 and not torch.cuda.is_available():
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "CUDA unavailable: falling back to legacy Mamba-1 backbone."
            )
            use_native_mamba3 = False
        self.use_native_mamba3 = use_native_mamba3
        if use_native_mamba3:
            self.mamba_backbone: nn.Module = Mamba3NativeBackbone(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                num_layers=num_mamba_layers,
                d_subgoal=self.d_subgoal,
                prop_dim=prop_dim,
                headdim=mamba3_headdim,
                ngroups=mamba3_ngroups,
                rope_fraction=mamba3_rope_fraction,
                is_outproj_norm=mamba3_outproj_norm,
                chunk_size=mamba3_chunk,
            )
        else:
            self.mamba_backbone = Mamba3CognitiveBackbone(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                num_layers=num_mamba_layers,
                d_subgoal=self.d_subgoal,
                prop_dim=prop_dim,
            )

        # 3. Generative Flow-Matching action-chunk policy and HiQC critic.
        flow_context_dim = self.d_subgoal + prop_dim + 1
        critic_context_dim = self.d_subgoal + prop_dim
        self.flow_context_dim = flow_context_dim
        self.critic_context_dim = critic_context_dim
        if action_policy == "flow":
            self.flow_policy: nn.Module = FlowPolicy(
                action_dim=action_dim, chunk_size=self.chunk_size,
                context_dim=flow_context_dim, hidden_dim=256,
            )
        else:
            self.flow_policy = GaussianActionPolicy(
                action_dim=action_dim, chunk_size=self.chunk_size,
                context_dim=flow_context_dim, hidden_dim=256,
            )
        self.hiqc_critic: nn.Module = HiQCCritic(
            state_dim=critic_context_dim, action_dim=action_dim,
            chunk_size=self.chunk_size, hidden_dim=256,
        )
        # CfC filters the single executed action (per the two-tier control diagram).
        self.cfc_filter: nn.Module = CfCActionFilter(
            action_dim=action_dim,
            chunk_size=1,
            state_dim=d_model,
            units=cfc_units,
            backbone_units=cfc_backbone_units,
            backbone_layers=cfc_backbone_layers,
            residual=cfc_residual,
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
            cfc_residual=cfg.cfc_residual,
            use_visual=cfg.use_visual,
            visual_channels=cfg.visual_channels,
            visual_image_size=cfg.visual_image_size,
            dropout=cfg.dropout,
            chunk_size=cfg.chunk_size,
            action_policy=cfg.action_policy,
            use_native_mamba3=cfg.use_native_mamba3,
            mamba3_headdim=cfg.mamba3_headdim,
            mamba3_ngroups=cfg.mamba3_ngroups,
            mamba3_rope_fraction=cfg.mamba3_rope_fraction,
            mamba3_outproj_norm=cfg.mamba3_outproj_norm,
            mamba3_chunk=cfg.mamba3_chunk,
        )
        return model

    # ------------------------------------------------------------------
    # Shared encoding used by training, inference and the loss.
    # ------------------------------------------------------------------
    def encode(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        visual_frames: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
        with_action: bool = True,
        action_last_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Shared policy computation for training and inference.

        Runs the multi-modal fusion, the Mamba-3 backbone, the flow-matching
        action-chunk generator and the CfC output filter in one pass.

        Args:
            with_action: when False, skip the flow/CfC action branch (used for
                auxiliary passes).
            action_last_only: when True, run the flow/CfC branch only on the last
                context step (used at inference, where only the last action is
                executed). This avoids recomputing flow+Ode over the whole window.

        Returns:
            subgoals:        (B, T, d_subgoal)
            latent:          (B, T, d_model)
            values:          (B, T, 1)
            next_states:     (B, T, prop_dim)
            flow_context:    (B, T, d_subgoal + prop_dim + 1)
            executed_action: (B, T, action_dim) or (B, 1, action_dim) or None
        """
        u_t = self.fusion(
            states=states,
            rtgs=rtgs,
            actions=actions,
            timesteps=timesteps,
            visual_frames=visual_frames,
        )
        subgoals, latent, values, next_states = self.mamba_backbone(u_t)
        if rtgs.ndim == 2:
            rtgs = rtgs.unsqueeze(-1)
        flow_context = torch.cat([subgoals, states, rtgs], dim=-1)

        if not with_action:
            return subgoals, latent, values, next_states, flow_context, None

        if action_last_only:
            fc = flow_context[:, -1:, :]
            lat = latent[:, -1:, :]
        else:
            fc = flow_context
            lat = latent
        chunk = self._sample_chunk(fc)                    # (B, t, chunk, action_dim)
        nominal = chunk[:, :, 0, :]                       # first step of the chunk
        if self.cfc_filter is not None:
            executed_action, _ = self.cfc_filter(nominal, lat, hx=hx)
        else:
            executed_action = nominal
        return subgoals, latent, values, next_states, flow_context, executed_action

    def _sample_chunk(self, flow_context: torch.Tensor) -> torch.Tensor:
        """Differentiable action-chunk sample -> (B, T, chunk, action_dim).

        Uses the stochastic flow sampler (trained with a Gaussian prior). A fixed
        evaluation generator makes closed-loop rollouts reproducible.
        """
        assert self.flow_policy is not None, "flow_policy is not initialised"
        B, T, _ = flow_context.shape
        flat = flow_context.reshape(B * T, -1)
        if isinstance(self.flow_policy, FlowPolicy):
            if self.deterministic:
                chunk = self.flow_policy.sample_deterministic(flat, num_steps=self.num_flow_steps)
            else:
                chunk = self.flow_policy.sample(flat, num_steps=self.num_flow_steps)
        else:  # GaussianActionPolicy
            chunk = self.flow_policy.sample(flat)  # type: ignore[operator]
        return chunk.view(B, T, self.chunk_size, self.action_dim)

    def forward(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        visual_frames: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Sequence-level forward pass.

        Args:
            states: (B, T, prop_dim)
            rtgs: (B, T, 1) or (B, T)
            actions: (B, T, action_dim) | None
            timesteps: (B, T) | None
            visual_frames: (B, T, C, H, W) | None
            hx: Hidden state tensor for the CfC filter | None

        Returns:
            actions_pred: (B, T, action_dim)  first action of the refined chunk
            subgoals_pred: (B, T, d_subgoal)
            values_pred: (B, T, 1)
            next_states_pred: (B, T, prop_dim)
            next_hx: Updated CfC hidden state | None
        """
        subgoals_pred, latent_features, values_pred, next_states_pred, _, actions_pred = self.encode(
            states=states, rtgs=rtgs, actions=actions, timesteps=timesteps,
            visual_frames=visual_frames, hx=hx,
        )
        return actions_pred, subgoals_pred, values_pred, next_states_pred, None

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
        """Step inference for closed-loop rollouts; returns the last-step action."""
        if states.shape[1] > 1 and hx is not None:
            raise ValueError("Overlapping context must use hx=None; otherwise history is replayed twice")
        subgoals_pred, latent_features, values_pred, next_states_pred, _, actions_pred = self.encode(
            states=states, rtgs=rtgs, actions=actions, timesteps=timesteps,
            visual_frames=visual_frames, hx=hx, action_last_only=True,
        )
        last_action = actions_pred[:, -1, :]
        last_subgoal = subgoals_pred[:, -1, :]
        return last_action, None, {
            "subgoal": last_subgoal,
            "action": last_action,
            "next_state": next_states_pred[:, -1, :],
        }

    @torch.inference_mode()
    def act_from_subgoal(
        self,
        subgoal: torch.Tensor,
        current_prop: torch.Tensor,
        rtg: torch.Tensor,
        hx: torch.Tensor | None = None,
        num_flow_steps: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Micro-actuation tier: regenerate an action chunk from a frozen subgoal.

        Implements the hierarchical decoupling of the two-tier controller: the
        macroscopic Mamba tier plans a subgoal, then this micro tier samples a
        fresh flow chunk from that subgoal without recomputing the backbone.

        Args:
            subgoal: (B, d_subgoal) frozen macro-plan from the Mamba backbone.
            current_prop: (B, prop_dim) current proprioceptive state.
            rtg: (B, 1) current return-to-go.
            hx: CfC hidden state or None.
            num_flow_steps: Euler steps for the flow ODE.

        Returns:
            action: (B, action_dim)
            next_hx: updated CfC hidden state
        """
        if subgoal.ndim != 2 or current_prop.ndim != 2:
            raise ValueError("act_from_subgoal expects (B, d) tensors")
        if rtg.ndim == 1:
            rtg = rtg.unsqueeze(-1)
        flow_context = torch.cat([subgoal, current_prop, rtg], dim=-1)  # (B, context_dim)
        sampler = getattr(self.flow_policy, "sample_deterministic", None)
        chunk = (
            sampler(flow_context, num_steps=num_flow_steps)
            if sampler is not None
            else self.flow_policy.sample(flow_context)  # type: ignore[union-attr]
        )
        nominal = chunk[:, 0, :]  # (B, action_dim)
        # CfC needs a 2D state representation; use a zero latent for the micro step.
        latent = torch.zeros(current_prop.shape[0], self.d_model, device=current_prop.device, dtype=current_prop.dtype)
        if self.cfc_filter is not None:
            action, next_hx = self.cfc_filter(nominal, latent, hx=hx)
        else:
            action, next_hx = nominal, None
        return action, next_hx
