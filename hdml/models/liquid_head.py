from __future__ import annotations

import torch
import torch.nn as nn
from ncps.torch import CfC
from typing import cast



class CfCActionFilter(nn.Module):
    """Closed-Form Continuous-Time (CfC) action filter for HDML.

    The filter input is ``[first action of the flow chunk, Mamba latent]``.
    ``state_repr`` is the backbone latent (``d_model``), **not** the raw sensor
    vector: the report's conceptual equation is implemented here in latent
    space. ``ncps.torch.CfC`` advances one normalized time unit per token when
    no ``timespans`` tensor is supplied (``ts = 1.0``). That is a normalized
    step, not a physical 20 ms. Changing it to ``0.02`` changes the ODE
    dynamics and is not done implicitly for existing checkpoints.
    """
    def __init__(
        self,
        action_dim: int,
        chunk_size: int,
        state_dim: int,
        units: int = 32,
        residual: float = 0.1,
        backbone_units: int = 64,
        backbone_layers: int = 1,
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
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            mode="default",
        )

        self.out_proj = nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, action_dim * chunk_size),
            nn.Tanh()
        )
        # Initialize final projection with small weights so initial filtering is near-identity
        final_layer = cast(nn.Linear, self.out_proj[2])
        nn.init.uniform_(final_layer.weight, -1e-3, 1e-3)
        nn.init.zeros_(final_layer.bias)

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
            state_repr: Mamba latent, (B, T, state_dim) or (B, state_dim).
                This is not the raw proprioceptive/sensor vector.
            hx: Previous CfC hidden state, or None to start from zeros.
        Returns:
            filtered_action: Same shape as actions
            next_hx: Updated hidden state. Passing it to the next call advances
                the same recurrence one token at a time.
        """
        orig_shape = actions.shape
        if actions.ndim == 2:
            flat_actions = actions.unsqueeze(1)
            s_repr = state_repr.unsqueeze(1) if state_repr.ndim == 2 else state_repr
        elif actions.ndim == 3:
            if state_repr.ndim == 2:
                # (B, T, D) actions + (B, Ds) state: broadcast the state
                # across time. The time axis is preserved (no collapsing to
                # seq-len 1); behaviour on the working 3D+3D path is unchanged.
                B, T, _ = actions.shape
                flat_actions = actions
                s_repr = state_repr.unsqueeze(1).expand(B, T, -1)
            else:
                flat_actions = actions
                s_repr = state_repr
        elif actions.ndim == 4:
            B, T, C, D = actions.shape
            flat_actions = actions.view(B, T, C * D)
            s_repr = state_repr
        else:
            raise ValueError(f"Unsupported action tensor shape: {actions.shape}")

        # Align floating dtypes without changing the all-float32 working path.
        if flat_actions.is_floating_point() and s_repr.is_floating_point() and flat_actions.dtype != s_repr.dtype:
            s_repr = s_repr.to(dtype=flat_actions.dtype)
        cfc_input = torch.cat([flat_actions, s_repr], dim=-1)
        if not torch.onnx.is_in_onnx_export() and cfc_input.shape[-1] != self.input_dim:
            raise ValueError(f"CfC input dimension {cfc_input.shape[-1]} != {self.input_dim}")
        cfc_out, next_hx = self.cfc(cfc_input, hx)
        filtered = self.out_proj(cfc_out)

        filtered = filtered.view(orig_shape)
        filtered_action = actions + self.residual * filtered

        return filtered_action, next_hx

