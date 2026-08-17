from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    """Residual Mamba Block with Pre-LayerNorm.

    Input:  (Batch, Seq_Len, d_model)
    Output: (Batch, Seq_Len, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return x + self.mamba(self.norm(x))


class MambaCognitiveBackbone(nn.Module):
    """Khối 2: Xương sống Lập kế hoạch Chiến lược (Mamba Cognitive Backbone).

    Uses multi-layer Selective State Space Models (Mamba S6) to model long-horizon
    causal trajectory dynamics with O(N) linear time complexity and O(1) state memory.
    Predicts a sequence of latent subgoals c_t in R^{d_subgoal}.

    Shape Contract:
        Input:
            u_t: (Batch, Seq_Len, d_model)
        Output:
            subgoals: (Batch, Seq_Len, d_subgoal)
            latent_features: (Batch, Seq_Len, d_model)
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 3,
        d_subgoal: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.num_layers = num_layers
        self.d_subgoal = d_subgoal

        # Stack of Mamba SSM Layers
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Latent Subgoal Planning Head
        self.subgoal_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_subgoal),
            nn.LayerNorm(d_subgoal),
        )

        # Auxiliary Return / Value Head (for Algorithm Distillation & auxiliary loss)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        u_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence forward pass over historical context.

        Args:
            u_t: Fused token sequence from CrossModalFusion, shape (B, T, d_model).

        Returns:
            subgoals: Latent subgoals, shape (B, T, d_subgoal).
            values: Predicted state-values/returns, shape (B, T, 1).
            latent: Hidden states, shape (B, T, d_model).
        """
        assert u_t.ndim == 3, f"Expected u_t (B, T, d_model), got shape {u_t.shape}"
        assert u_t.shape[-1] == self.d_model, (
            f"Expected feature dim {self.d_model}, got {u_t.shape[-1]}"
        )

        x = u_t
        for layer in self.layers:
            x = layer(x)

        latent = self.final_norm(x)                         # (B, T, d_model)
        subgoals = self.subgoal_head(latent)               # (B, T, d_subgoal)
        values = self.value_head(latent)                   # (B, T, 1)

        return subgoals, values, latent
