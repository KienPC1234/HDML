"""Native Mamba-3 selective state-space backbone for HDML.

This module implements the macroscopic sequence backbone described in the
project report using the *native* ``mamba_ssm.modules.mamba3.Mamba3`` layer,
which provides, inside the fused Triton/CUTE kernel:

  * selective state-space recurrences,
  * rotary state-space embeddings (RoPE) applied to the B/C projections via the
    ``rope_fraction`` / ``angles`` mechanism (no external Givens rotation needed),
  * trapezoidal discretisation of the state transition (the learned ``trap``
    gate),
  * exponential-trapezoidal ``dd_A`` parameterisation.

Shape contract
--------------
    Input:  (Batch, Seq_Len, d_model)
    Output: (Batch, Seq_Len, d_model)

The block keeps the same public attributes used by older HDML checkpoints
(``norm``, ``mamba``) so that the wrapper can be swapped in ``HDMLModel``
without touching the training/evaluation code.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from mamba_ssm import Mamba3

logger = logging.getLogger(__name__)


class Mamba3NativeBlock(nn.Module):
    """One pre-norm residual block built on the native Mamba-3 SSM layer.

    ``d_inner = expand * d_model`` must be divisible by ``headdim``. The RoPE
    split size ``d_state * rope_fraction`` must be even and yield at least one
    angle pair.

    Args:
        d_model: Hidden width of the block.
        d_state: SSM state size per head.
        expand: Expansion factor for the inner dimension.
        headdim: Per-head width of the inner dimension.
        ngroups: Number of B/C groups (shared across heads).
        rope_fraction: Fraction of the state used for rotary embeddings
            (0.5 or 1.0 as required by the kernel).
        chunk_size: Fused kernel sequence chunk size.
    """

    def __init__(
        self,
        d_model: int = 384,
        d_state: int = 16,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        is_outproj_norm: bool = False,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        d_inner = int(expand * d_model)
        if d_inner % headdim != 0:
            raise ValueError(
                f"expand*d_model ({d_inner}) must be divisible by headdim ({headdim})."
            )
        split = int(d_state * rope_fraction)
        if split % 2 != 0:
            split -= 1
        if split < 2:
            raise ValueError(
                f"d_state*rope_fraction must yield at least one angle pair, got {split}."
            )

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.rope_fraction = rope_fraction

        self.norm = nn.LayerNorm(d_model)
        # Native Mamba-3 with built-in rotary state-space embedding.
        self.mamba = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            rope_fraction=rope_fraction,
            is_outproj_norm=is_outproj_norm,
            chunk_size=chunk_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (Batch, Seq_Len, d_model).

        Returns:
            Tensor of shape (Batch, Seq_Len, d_model).
        """
        assert x.ndim == 3, f"Expected (B, L, D), got {tuple(x.shape)}"
        assert x.shape[-1] == self.d_model, (
            f"Expected last dim {self.d_model}, got {x.shape[-1]}"
        )
        return x + self.mamba(self.norm(x))


class Mamba3NativeBackbone(nn.Module):
    """Stack of native Mamba-3 blocks with the HDML auxiliary heads.

    Mirrors ``Mamba3CognitiveBackbone`` so it is a drop-in replacement in
    ``HDMLModel``.

    Shape contract:
        u_t: (B, L, d_model)
        -> subgoals:    (B, L, d_subgoal)
        -> latent:      (B, L, d_model)
        -> values:      (B, L, 1)
        -> next_states: (B, L, prop_dim)
    """

    def __init__(
        self,
        d_model: int = 384,
        d_state: int = 16,
        expand: int = 2,
        num_layers: int = 8,
        d_subgoal: int = 128,
        prop_dim: int = 27,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        is_outproj_norm: bool = False,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.num_layers = num_layers
        self.prop_dim = prop_dim
        self.d_subgoal = d_subgoal

        self.layers = nn.ModuleList(
            [
                Mamba3NativeBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    headdim=headdim,
                    ngroups=ngroups,
                    rope_fraction=rope_fraction,
                    is_outproj_norm=is_outproj_norm,
                    chunk_size=chunk_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

        self.subgoal_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_subgoal),
            nn.LayerNorm(d_subgoal),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.forward_dynamics_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, prop_dim),
        )

    def forward(
        self, u_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = u_t
        for layer in self.layers:
            x = layer(x)
        latent_features = self.final_norm(x)
        return (
            self.subgoal_head(latent_features),
            latent_features,
            self.value_head(latent_features),
            self.forward_dynamics_head(latent_features),
        )
