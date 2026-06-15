"""HDML-Foundation: Universal Multi-Embodiment Sequence-to-Action Backbone.

Combines scalable selective state-spaces (Mamba-3) with continuous-time
Liquid Neural Networks (CfC ODE) for multi-robot pre-training and fast adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from hdml.models.mamba3_backbone import Mamba3CognitiveBackbone
from ncps.torch import CfC


@dataclass
class EmbodimentSpec:
    """Specification of an individual robotic embodiment."""
    name: str
    prop_dim: int
    action_dim: int
    max_action: float = 1.0
    control_freq_hz: float = 50.0


class UniversalEmbodimentAdapter(nn.Module):
    """Dynamic adapter mapping arbitrary embodiment dimensions to a shared latent space."""

    def __init__(
        self,
        prop_dim: int,
        action_dim: int,
        d_model: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.d_model = d_model

        # State & Return-to-go projector
        self.state_proj = nn.Sequential(
            nn.Linear(prop_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Action history projector
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Output Action Head
        self.action_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim),
        )

    def project_state(self, states: torch.Tensor) -> torch.Tensor:
        """Projects proprioceptive states (B, T, prop_dim) -> (B, T, d_model)."""
        return self.state_proj(states)

    def project_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Projects past actions (B, T, action_dim) -> (B, T, d_model)."""
        return self.action_proj(actions)

    def decode_action(self, latents: torch.Tensor) -> torch.Tensor:
        """Decodes latent features (B, T, d_model) -> actions (B, T, action_dim)."""
        return torch.tanh(self.action_out(latents))


class HDMLFoundationModel(nn.Module):
    """Hierarchical Decision Mamba-Liquid Foundation Model.

    A generalist robot sequence backbone capable of pre-training across diverse
    embodiments and fine-tuning quickly on new robotics hardware.
    """

    def __init__(
        self,
        d_model: int = 384,
        num_mamba_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        cfc_units: int = 96,
        cfc_backbone_units: int = 192,
        cfc_residual: float = 0.05,
        max_timesteps: int = 4096,
        max_embodiments: int = 32,
        dropout: float = 0.1,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.cfc_residual = cfc_residual

        # 1. Universal Embeddings
        self.embodiment_embedding = nn.Embedding(max_embodiments, d_model)
        self.timestep_embedding = nn.Embedding(max_timesteps, d_model)
        self.rtg_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Multi-modal Token Fusion Layer
        self.fusion_norm = nn.LayerNorm(d_model)
        self.fusion_linear = nn.Linear(d_model * 4, d_model)

        # 2. Scaled Mamba Sequence Backbone
        self.mamba_backbone = Mamba3CognitiveBackbone(
            d_model=d_model,
            num_layers=num_mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            d_subgoal=64,
        )

        # 3. Universal Continuous-Time Liquid CfC ODE Filter
        self.cfc_filter = CfC(
            input_size=d_model,
            units=cfc_units,
            proj_size=d_model,
            return_sequences=True,
            batch_first=True,
            backbone_units=cfc_backbone_units,
            backbone_layers=1,
            mode="default",
        )

        # 4. Universal Predictive Auxiliary Heads
        self.intent_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 64),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.rtg_pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # 5. Dynamic Embodiment Adapters Registry
        self.adapters = nn.ModuleDict()
        self.to(self.device)

    def register_embodiment(self, name: str, prop_dim: int, action_dim: int) -> UniversalEmbodimentAdapter:
        """Registers a new embodiment adapter."""
        adapter = UniversalEmbodimentAdapter(
            prop_dim=prop_dim,
            action_dim=action_dim,
            d_model=self.d_model,
        ).to(self.device)
        self.adapters[name] = adapter
        return adapter

    def get_adapter(self, name: str) -> UniversalEmbodimentAdapter:
        """Retrieves registered embodiment adapter."""
        if name not in self.adapters:
            raise KeyError(f"Embodiment '{name}' not registered in Foundation Model. Available: {list(self.adapters.keys())}")
        return self.adapters[name]

    def forward(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        embodiment_name: str,
        embodiment_idx: int = 0,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Multi-embodiment forward pass.

        Args:
            states: (B, T, prop_dim)
            rtgs: (B, T, 1) or (B, T)
            actions: (B, T, action_dim) past actions
            timesteps: (B, T) timesteps
            embodiment_name: Registered embodiment key
            embodiment_idx: Integer embodiment ID for embedding
            hx: Hidden state for Liquid CfC

        Returns:
            actions_pred: (B, T, action_dim)
            intents_pred: (B, T, 64)
            values_pred: (B, T, 1)
            rtg_pred: (B, T, 1)
            next_hx: Updated CfC hidden state
        """
        adapter = self.get_adapter(embodiment_name)
        B, T, _ = states.shape

        if rtgs.dim() == 2:
            rtgs = rtgs.unsqueeze(-1)

        # 1. Project through embodiment-specific adapters
        state_feats = adapter.project_state(states)
        action_feats = adapter.project_action(actions)
        rtg_feats = self.rtg_proj(rtgs)

        # 2. Add Positional & Embodiment Embeddings
        t_emb = self.timestep_embedding(timesteps.clamp(0, self.timestep_embedding.num_embeddings - 1))
        emb_id_tensor = torch.full((B, T), embodiment_idx, dtype=torch.int64, device=states.device)
        emb_token = self.embodiment_embedding(emb_id_tensor)

        state_token = state_feats + t_emb + emb_token
        rtg_token = rtg_feats + t_emb
        action_token = action_feats + t_emb
        context_token = emb_token + t_emb

        # 3. Fuse Modalities
        fused_tokens = torch.cat([state_token, rtg_token, action_token, context_token], dim=-1)
        u_t = self.fusion_norm(self.fusion_linear(fused_tokens))

        # 4. Phase-Gated Mamba-3 Core
        _, latent_features, values_pred, _ = self.mamba_backbone(u_t)

        # 5. Liquid CfC Dynamic Refinement
        if self.cfc_filter is not None:
            cfc_out, next_hx = self.cfc_filter(latent_features, hx=hx)
            liquid_latents = latent_features + self.cfc_residual * cfc_out
        else:
            liquid_latents = latent_features
            next_hx = None

        # 6. Decode into Embodiment-Specific Action Space
        actions_pred = adapter.decode_action(liquid_latents)
        intents_pred = self.intent_head(latent_features)
        rtg_pred = self.rtg_pred_head(latent_features)

        return actions_pred, intents_pred, values_pred, rtg_pred, next_hx

    @torch.inference_mode()
    def get_action(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        embodiment_name: str,
        embodiment_idx: int = 0,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Step-level action inference for closed-loop execution."""
        actions_pred, _, _, _, next_hx = self.forward(
            states=states,
            rtgs=rtgs,
            actions=actions,
            timesteps=timesteps,
            embodiment_name=embodiment_name,
            embodiment_idx=embodiment_idx,
            hx=hx,
        )
        return actions_pred[:, -1], next_hx

    def freeze_backbone(self) -> None:
        """Freezes core Mamba-Liquid weights for fast few-shot adaptation."""
        for param in self.mamba_backbone.parameters():
            param.requires_grad = False
        for param in self.cfc_filter.parameters():
            param.requires_grad = False
        for param in self.fusion_linear.parameters():
            param.requires_grad = False
        for param in self.fusion_norm.parameters():
            param.requires_grad = False
        self.embodiment_embedding.weight.requires_grad = False
        self.timestep_embedding.weight.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreezes all parameters for full foundation pre-training."""
        for param in self.parameters():
            param.requires_grad = True
