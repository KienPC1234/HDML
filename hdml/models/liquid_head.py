from __future__ import annotations

import torch
import torch.nn as nn
from ncps.torch import CfC


class LiquidReactiveControlHead(nn.Module):
    """Khối 3: Đầu Điều khiển Phản xạ Lỏng (Liquid Reactive Control Head).

    Uses MIT Closed-form Continuous-Time Neural Networks (CfC) to model continuous
    dynamic ordinary differential equations (ODEs). Adapts neural time-constants
    tau_i dynamically to reject high-frequency physical perturbations and output
    smooth continuous joint torques / action commands in [-1, 1].

    Shape Contract:
        Input:
            subgoals:     (Batch, Seq_Len, d_subgoal) or (Batch, d_subgoal)
            current_prop: (Batch, Seq_Len, prop_dim) or (Batch, prop_dim)
            hx:           Hidden state tensor from previous step, or None
        Output:
            actions:      (Batch, Seq_Len, action_dim) or (Batch, action_dim)
            next_hx:      Updated hidden state tensor
    """

    def __init__(
        self,
        d_subgoal: int = 64,
        prop_dim: int = 27,
        action_dim: int = 8,
        units: int = 32,
        backbone_units: int = 64,
        backbone_layers: int = 1,
        mode: str = "default",
    ) -> None:
        super().__init__()
        self.d_subgoal = d_subgoal
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.units = units
        self.input_dim = d_subgoal + prop_dim

        # Closed-form Continuous-time Neural Network (CfC) from ncps
        self.cfc = CfC(
            input_size=self.input_dim,
            units=units,
            proj_size=None,
            return_sequences=True,
            batch_first=True,
            backbone_units=backbone_units,
            backbone_layers=backbone_layers,
            backbone_dropout=0.0,
            mode=mode,
        )

        # High-Frequency Action Output Projection
        self.action_out = nn.Sequential(
            nn.Linear(units, units),
            nn.SiLU(),
            nn.Linear(units, action_dim),
            nn.Tanh(),  # Bounded continuous action space [-1.0, 1.0]
        )

    def forward(
        self,
        subgoals: torch.Tensor,
        current_prop: torch.Tensor,
        hx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through Liquid Control Head.

        Handles both 3D sequence tensors (B, T, D) for training and
        2D step tensors (B, D) for real-time online rollout.

        Args:
            subgoals: Latent subgoals, shape (B, T, d_subgoal) or (B, d_subgoal).
            current_prop: Kinematic proprioception, shape (B, T, prop_dim) or (B, prop_dim).
            hx: Previous hidden state tensor, or None.

        Returns:
            actions: Continuous action commands in [-1, 1], shape (B, T, action_dim) or (B, action_dim).
            next_hx: Updated hidden state tensor.
        """
        is_2d = subgoals.ndim == 2

        if is_2d:
            assert current_prop.ndim == 2, (
                f"Expected current_prop to be 2D when subgoals is 2D, got {current_prop.shape}"
            )
            subgoals_seq = subgoals.unsqueeze(1)
            prop_seq = current_prop.unsqueeze(1)
        else:
            assert subgoals.ndim == 3, f"Expected subgoals 3D, got {subgoals.shape}"
            assert current_prop.ndim == 3, f"Expected current_prop 3D, got {current_prop.shape}"
            subgoals_seq = subgoals
            prop_seq = current_prop

        assert subgoals_seq.shape[-1] == self.d_subgoal, (
            f"Expected subgoal dim {self.d_subgoal}, got {subgoals_seq.shape[-1]}"
        )
        assert prop_seq.shape[-1] == self.prop_dim, (
            f"Expected prop dim {self.prop_dim}, got {prop_seq.shape[-1]}"
        )

        # Concatenate Subgoal intent with instantaneous proprioceptive feedback
        cfc_input = torch.cat([subgoals_seq, prop_seq], dim=-1)  # (B, T, d_subgoal + prop_dim)

        cfc_out, next_hx = self.cfc(cfc_input, hx)                # (B, T, units)
        actions = self.action_out(cfc_out)                        # (B, T, action_dim)

        if is_2d:
            actions = actions.squeeze(1)

        return actions, next_hx
