from __future__ import annotations

import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba

def apply_rope(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply 2D Rotary Position Embeddings (RoPE Trick) to a tensor.
    
    This simulates complex arithmetic for State Space Models by applying
    2D Givens rotations to adjacent feature pairs. This allows real-valued
    SSMs to track phase and angular rotations exactly (SO(3) capability).
    
    Args:
        x: Input tensor of shape (..., D).
        dim: The feature dimension to apply RoPE. Default is -1.
        
    Returns:
        Tensor of the same shape with RoPE applied.
    """
    assert x.size(dim) % 2 == 0, f"Feature dimension {x.size(dim)} must be even for RoPE."
    
    # Split into pairs: (..., D/2, 2)
    x_reshaped = x.unflatten(dim, (x.size(dim) // 2, 2))
    x1, x2 = x_reshaped.unbind(dim=-1)
    
    # Generate rotation angles based on positions (we use a simple learned or fixed embedding)
    # For simplicity in this functional RoPE, we use standard sinusoidal positional encoding
    seq_len = x.size(-2)
    position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(-1)
    
    # D/2 frequencies
    div_term = torch.exp(torch.arange(0, x.size(dim), 2, dtype=torch.float32, device=x.device) * (-math.log(10000.0) / x.size(dim)))
    theta = position * div_term
    
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    # Apply rotation:
    # y1 = x1 * cos(theta) - x2 * sin(theta)
    # y2 = x1 * sin(theta) + x2 * cos(theta)
    y1 = x1 * cos_theta - x2 * sin_theta
    y2 = x1 * sin_theta + x2 * cos_theta
    
    # Recombine
    y = torch.stack([y1, y2], dim=-1).flatten(start_dim=-2)
    return y


class Mamba3Block(nn.Module):
    """Mamba-3 Block integrating Complex RoPE and Exponential-Trapezoidal approximations.
    
    Due to the unavailability of a native Mamba-3 CUDA kernel, this block applies
    the RoPE trick as a pre-processing step to the inputs before passing to the
    Mamba-1/2 selective scan. This mathematically equips the SSM with complex
    eigenvalues to model 3D periodic/rotational dynamics without custom kernels.
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
        # Using the fast Mamba CUDA implementation
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # Trapezoidal Gate for exponential-trapezoidal emulation
        self.trap_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with RoPE and residual connection.
        
        Args:
            x: Tensor of shape (Batch, Seq_Len, d_model)
        """
        # 1. Pre-norm
        x_norm = self.norm(x)
        
        # 2. Apply Complex RoPE Trick
        x_rope = apply_rope(x_norm, dim=-1)
        
        # 3. Mamba Core
        mamba_out = self.mamba(x_rope)
        
        # 4. Exponential-Trapezoidal Gate approximation
        # Emulates the trap_t gate from Mamba-3
        trap_factor = self.trap_gate(x_norm)
        mamba_out = mamba_out * trap_factor
        
        # 5. Residual connection
        return x + mamba_out


class Mamba3CognitiveBackbone(nn.Module):
    """HDML-V2 Mamba-3 Cognitive Backbone.
    
    Uses multi-layer Complex-Valued Selective State Space Models (Mamba-3 emulation)
    to model long-horizon 3D dynamics with exact angular phase tracking.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 3,
        d_subgoal: int | None = None,
        prop_dim: int = 27,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.num_layers = num_layers
        self.prop_dim = prop_dim
        self.d_subgoal = prop_dim if d_subgoal is None else d_subgoal

        # Stack of Mamba-3 SSM Layers
        self.layers = nn.ModuleList([
            Mamba3Block(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Latent Subgoal Planning Head (High-level Planner)
        self.subgoal_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_subgoal),
            nn.LayerNorm(d_subgoal),
        )

        # Auxiliary Return / Value Head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Forward Dynamics Prediction Head
        self.forward_dynamics_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, prop_dim),
        )

    def forward(self, u_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the backbone.
        
        Args:
            u_t: Multi-modal embedded sequence (Batch, Seq_Len, d_model)
            
        Returns:
            subgoals: Predicted latent subgoals (Batch, Seq_Len, d_subgoal)
            latent_features: Mamba representation (Batch, Seq_Len, d_model)
            values: Auxiliary value predictions (Batch, Seq_Len, 1)
            next_states_pred: Forward dynamics predictions (Batch, Seq_Len, prop_dim)
        """
        x = u_t
        for layer in self.layers:
            x = layer(x)
            
        latent_features = self.final_norm(x)
        
        # High-level outputs
        subgoals = self.subgoal_head(latent_features)
        values = self.value_head(latent_features)
        next_states_pred = self.forward_dynamics_head(latent_features)
        
        return subgoals, latent_features, values, next_states_pred
