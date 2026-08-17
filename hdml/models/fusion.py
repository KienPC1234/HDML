from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class VisualPatchEncoder(nn.Module):
    """Encodes 2D visual/depth frames into spatial visual tokens.

    Input:  (B, C, H, W) or (B, T, C, H, W)
    Output: (B, d_model) or (B, T, d_model)
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_size: int = 64,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.d_model = d_model

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # (B, 32, H/2, W/2)
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # (B, 64, H/4, W/4)
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # (B, 128, H/8, W/8)
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),                                    # (B, 128, 2, 2)
            nn.Flatten(),                                                    # (B, 512)
        )
        self.proj = nn.Linear(512, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, visual_frames: torch.Tensor) -> torch.Tensor:
        """Forward pass for visual frame encoder.

        Args:
            visual_frames: Tensor of shape (B, C, H, W) or (B, T, C, H, W)

        Returns:
            Encoded visual tokens of shape (B, d_model) or (B, T, d_model)
        """
        assert visual_frames.ndim in (4, 5), (
            f"Expected visual_frames to have 4 or 5 dims, got shape {visual_frames.shape}"
        )

        if visual_frames.ndim == 5:
            b, t, c, h, w = visual_frames.shape
            flat_frames = visual_frames.view(b * t, c, h, w)
            feats = self.conv_net(flat_frames)
            tokens = self.layer_norm(self.proj(feats))
            return tokens.view(b, t, self.d_model)

        feats = self.conv_net(visual_frames)
        tokens = self.layer_norm(self.proj(feats))
        return tokens


class CrossModalFusion(nn.Module):
    """Khối 1: Tầng Dung hợp Nhận thức Thể thức Giao thoa (Cross-Modal Fusion Layer).

    Merges proprioceptive kinematics, Return-to-Go (RTG), previous action tokens,
    temporal timestep embeddings, and optional visual representations into a unified
    latent sequence representation U_t for the Mamba Cognitive Backbone.

    Shape Contract:
        Input:
            states:    (Batch, Seq_Len, prop_dim)
            rtgs:      (Batch, Seq_Len, 1)
            actions:   (Batch, Seq_Len, action_dim) | None
            timesteps: (Batch, Seq_Len) | None
            visual:    (Batch, Seq_Len, C, H, W) | None
        Output:
            u_t:       (Batch, Seq_Len, d_model)
    """

    def __init__(
        self,
        prop_dim: int = 27,
        action_dim: int = 8,
        d_model: int = 128,
        max_timesteps: int = 4096,
        use_visual: bool = False,
        visual_channels: int = 1,
        visual_image_size: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.use_visual = use_visual

        # 1. Proprioceptive State Kinematics Encoder: MLP
        self.prop_encoder = nn.Sequential(
            nn.Linear(prop_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # 2. Return-to-Go (RTG) Target Reward Encoder
        self.rtg_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # 3. Action Embedding Encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # 4. Temporal Timestep Positional Embedding
        self.timestep_embed = nn.Embedding(max_timesteps, d_model)

        # 5. Optional Visual Encoder
        if self.use_visual:
            self.visual_encoder = VisualPatchEncoder(
                in_channels=visual_channels,
                image_size=visual_image_size,
                d_model=d_model,
            )
            fusion_in_dim = d_model * 4
        else:
            self.visual_encoder = None
            fusion_in_dim = d_model * 3

        # Cross-modal projection & normalization
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        visual_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed and fuse multimodal inputs into a sequence of tokens.

        Args:
            states: Proprioceptive joint states, shape (B, T, prop_dim).
            rtgs: Target Return-to-Go, shape (B, T, 1) or (B, T).
            actions: Action tokens, shape (B, T, action_dim). If None, initialized to zeros.
            timesteps: Integer timestep indices, shape (B, T). If None, uses default 0..T-1.
            visual_frames: Visual image frames, shape (B, T, C, H, W) or None.

        Returns:
            Fused sequence tensor U_t, shape (B, T, d_model).
        """
        assert states.ndim == 3, f"Expected states (B, T, prop_dim), got {states.shape}"
        assert states.shape[-1] == self.prop_dim, (
            f"Expected state dim {self.prop_dim}, got {states.shape[-1]}"
        )

        batch_size, seq_len, _ = states.shape
        device = states.device

        # Reshape RTG if needed
        if rtgs.ndim == 2:
            rtgs = rtgs.unsqueeze(-1)
        assert rtgs.shape == (batch_size, seq_len, 1), (
            f"Expected rtgs shape {(batch_size, seq_len, 1)}, got {rtgs.shape}"
        )

        # Handle actions
        if actions is None:
            actions = torch.zeros(batch_size, seq_len, self.action_dim, device=device)
        assert actions.shape == (batch_size, seq_len, self.action_dim), (
            f"Expected actions shape {(batch_size, seq_len, self.action_dim)}, got {actions.shape}"
        )

        # Handle timesteps
        if timesteps is None:
            timesteps = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        assert timesteps.shape == (batch_size, seq_len), (
            f"Expected timesteps shape {(batch_size, seq_len)}, got {timesteps.shape}"
        )

        # Encode individual modalities
        z_prop = self.prop_encoder(states)                 # (B, T, d_model)
        z_rtg = self.rtg_encoder(rtgs)                     # (B, T, d_model)
        z_act = self.action_encoder(actions)               # (B, T, d_model)
        z_time = self.timestep_embed(timesteps.clamp(0, self.timestep_embed.num_embeddings - 1)) # (B, T, d_model)

        # Add positional temporal information to state & rtg
        z_prop = z_prop + z_time
        z_rtg = z_rtg + z_time
        z_act = z_act + z_time

        modalities = [z_prop, z_rtg, z_act]

        if self.use_visual:
            assert self.visual_encoder is not None, "Visual encoder not initialized."
            if visual_frames is None:
                z_vis = torch.zeros(batch_size, seq_len, self.d_model, device=device)
            else:
                z_vis = self.visual_encoder(visual_frames)
            z_vis = z_vis + z_time
            modalities.append(z_vis)

        # Concatenate along channel dimension and project
        fused_raw = torch.cat(modalities, dim=-1)           # (B, T, fusion_in_dim)
        u_t = self.fusion_proj(fused_raw)                   # (B, T, d_model)

        assert u_t.shape == (batch_size, seq_len, self.d_model), (
            f"Expected fused output shape {(batch_size, seq_len, self.d_model)}, got {u_t.shape}"
        )
        return u_t
