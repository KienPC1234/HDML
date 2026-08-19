from __future__ import annotations

import torch
import torch.nn as nn
from ncps.torch import CfC



class CfCActionFilter(nn.Module):
    """
    Closed-Form Continuous-Time (CfC) Action Filter for HDML.
    Maps predicted action chunks and high-frequency latent state to smooth continuous-time physical actions.
    """
    def __init__(
        self,
        action_dim: int,
        chunk_size: int,
        state_dim: int,
        units: int = 32,
        residual: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.residual = residual
        
        # We process the flattened action chunk + state
        self.input_dim = (action_dim * chunk_size) + state_dim
        
        self.cfc = CfC(
            input_size=self.input_dim,
            units=units,
            proj_size=None,
            return_sequences=True,
            batch_first=True,
            backbone_units=64,
            backbone_layers=1,
            mode="default",
        )
        
        self.out_proj = nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, action_dim * chunk_size),
            nn.Tanh()
        )
        # Initialize final projection with small weights so initial filtering is near-identity
        nn.init.uniform_(self.out_proj[2].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.out_proj[2].bias)
        
    def forward(
        self, 
        actions: torch.Tensor, 
        state_repr: torch.Tensor,
        hx: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Filter actions dynamically via continuous-time ODE dynamics.
        Args:
            actions: (B, T, action_dim), (B, action_dim), or (B, T, chunk, action_dim)
            state_repr: (B, T, state_dim) or (B, state_dim)
            hx: Hidden state
        Returns:
            filtered_action: Same shape as actions
            next_hx: Updated hidden state
        """
        orig_shape = actions.shape
        if actions.ndim == 2:
            flat_actions = actions.unsqueeze(1)
            s_repr = state_repr.unsqueeze(1) if state_repr.ndim == 2 else state_repr
        elif actions.ndim == 3:
            if state_repr.ndim == 2:
                B, C, D = actions.shape
                flat_actions = actions.view(B, 1, C * D)
                s_repr = state_repr.unsqueeze(1)
            else:
                flat_actions = actions
                s_repr = state_repr
        elif actions.ndim == 4:
            B, T, C, D = actions.shape
            flat_actions = actions.view(B, T, C * D)
            s_repr = state_repr
        else:
            raise ValueError(f"Unsupported action tensor shape: {actions.shape}")

        cfc_input = torch.cat([flat_actions, s_repr], dim=-1)
        cfc_out, next_hx = self.cfc(cfc_input, hx)
        filtered = self.out_proj(cfc_out)
        
        filtered = filtered.view(orig_shape)
        filtered_action = actions + self.residual * filtered
            
        return filtered_action, next_hx
