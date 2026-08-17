from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HDMLLoss(nn.Module):
    """Composite Loss Function for Hierarchical Decision Mamba-Liquid.

    Combines:
      1. Action Loss: Huber / Smooth L1 loss between predicted and ground-truth actions,
         with optional Self-Evolving Progressive Advantage Weighting (Section 2.1).
      2. Subgoal Temporal & Norm Regularization: Prevents latent representation drift.
      3. Value / Return Prediction Loss: Auxiliary loss for long-horizon credit assignment.

    Shape Contract:
        actions_pred:   (B, T, action_dim)
        target_actions: (B, T, action_dim)
        subgoals:       (B, T, d_subgoal)
        values_pred:    (B, T, 1)
        target_rtgs:    (B, T, 1)
        mask:           (B, T)
    """

    def __init__(
        self,
        action_weight: float = 1.0,
        subgoal_weight: float = 0.1,
        value_weight: float = 0.05,
        use_advantage_weighting: bool = False,
        advantage_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.action_weight = action_weight
        self.subgoal_weight = subgoal_weight
        self.value_weight = value_weight
        self.use_advantage_weighting = use_advantage_weighting
        self.advantage_temperature = advantage_temperature

    def forward(
        self,
        actions_pred: torch.Tensor,
        target_actions: torch.Tensor,
        subgoals: torch.Tensor,
        values_pred: torch.Tensor,
        target_rtgs: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute masked composite loss.

        Args:
            actions_pred: (B, T, action_dim)
            target_actions: (B, T, action_dim)
            subgoals: (B, T, d_subgoal)
            values_pred: (B, T, 1)
            target_rtgs: (B, T, 1)
            mask: (B, T) float mask (1.0 for valid, 0.0 for padding)

        Returns:
            total_loss: Scalar torch.Tensor with backward gradient.
            loss_dict: Dictionary with scalar loss breakdowns.
        """
        assert actions_pred.shape == target_actions.shape, (
            f"Action shape mismatch: {actions_pred.shape} vs {target_actions.shape}"
        )
        assert mask.ndim == 2, f"Expected 2D mask, got {mask.shape}"

        mask_expanded_act = mask.unsqueeze(-1)  # (B, T, 1)
        valid_tokens = torch.clamp(mask.sum(), min=1.0)

        # 1. Progressive Advantage Weighting (Self-Evolving Policy Filter)
        if self.use_advantage_weighting:
            with torch.no_grad():
                advantage = target_rtgs - values_pred.detach()
                adv_weights = torch.clamp(
                    torch.exp(advantage / self.advantage_temperature),
                    min=0.1,
                    max=10.0,
                )  # (B, T, 1)
        else:
            adv_weights = torch.ones_like(mask_expanded_act)

        # 2. Action Huber Loss with Advantage Weighting
        raw_act_loss = F.smooth_l1_loss(actions_pred, target_actions, reduction="none")  # (B, T, action_dim)
        weighted_act_loss = raw_act_loss * mask_expanded_act * adv_weights
        act_loss = weighted_act_loss.sum() / (valid_tokens * actions_pred.shape[-1])

        # 3. Subgoal Regularization: L2 magnitude + Smooth temporal transition
        subgoal_l2 = (subgoals ** 2).sum(dim=-1)  # (B, T)
        if subgoals.shape[1] > 1:
            subgoal_diff = ((subgoals[:, 1:] - subgoals[:, :-1]) ** 2).sum(dim=-1)  # (B, T-1)
            subgoal_smooth = (subgoal_diff * mask[:, 1:]).sum() / torch.clamp(mask[:, 1:].sum(), min=1.0)
        else:
            subgoal_smooth = torch.tensor(0.0, device=subgoals.device)

        subgoal_norm = (subgoal_l2 * mask).sum() / valid_tokens
        subgoal_loss = subgoal_norm + 0.5 * subgoal_smooth

        # 4. Value / Return Prediction Loss
        raw_val_loss = F.mse_loss(values_pred, target_rtgs, reduction="none")  # (B, T, 1)
        val_loss = (raw_val_loss * mask_expanded_act).sum() / valid_tokens

        # Composite total loss
        total_loss = (
            self.action_weight * act_loss
            + self.subgoal_weight * subgoal_loss
            + self.value_weight * val_loss
        )

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "action_loss": float(act_loss.item()),
            "subgoal_loss": float(subgoal_loss.item()),
            "value_loss": float(val_loss.item()),
        }

        return total_loss, loss_dict
