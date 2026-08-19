from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HDMLLoss(nn.Module):
    """
    HDML Comprehensive Loss Function.

    Includes:
    1. Flow Matching Loss (Action Generation)
    2. Q-Function Chunk Loss (HiQC Critic) via expectile regression (IQL-style)
    3. Value Function Loss (backbone value head) via conservative expectile regression
    4. PAVE (Policy-Aware Value-field Equalization) via Hutchinson Trace
    5. Grad-CAPS (Temporal Action Regularization on the predicted action chunk)
    6. Dynamics / World Model Loss
    """

    def __init__(
        self,
        flow_weight: float = 1.0,
        q_weight: float = 1.0,
        value_weight: float = 1.0,
        pave_weight: float = 0.01,
        grad_caps_weight: float = 0.01,
        dynamics_weight: float = 0.1,
        expectile: float = 0.9,
        value_expectile: float = 0.7,
        use_advantage_weighting: bool = False,
        advantage_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.flow_weight = flow_weight
        self.q_weight = q_weight
        self.value_weight = value_weight
        self.pave_weight = pave_weight
        self.grad_caps_weight = grad_caps_weight
        self.dynamics_weight = dynamics_weight
        self.expectile = expectile
        self.value_expectile = value_expectile
        self.use_advantage_weighting = use_advantage_weighting
        self.advantage_temperature = advantage_temperature

    @staticmethod
    def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        """Asymmetric squared loss L2^kappa(x) = |kappa - 1[x<0]| * x^2.

        With expectile > 0.5, positive residuals (under-estimation) are weighted more
        heavily, biasing the value function toward the upper tail of the return
        distribution (the IQL optimism mechanism).
        """
        weight = torch.where(diff > 0, expectile, 1.0 - expectile)
        return weight * (diff ** 2)

    def _advantage_weights(
        self,
        q1_pred: torch.Tensor,
        q2_pred: torch.Tensor,
        values_pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """DIVO-style binary advantage weights for the flow-matching objective.

        w(s, a) = 1[Q(s, a) > V(s)] * exp((Q(s, a) - V(s)) / beta)

        Args:
            q1_pred/q2_pred: HiQC critic outputs, shape (B, T, 1).
            values_pred: Backbone value head output, shape (B, T, 1).
            mask: Validity mask, shape (B, T).

        Returns:
            Advantage weights normalized to unit mean over valid tokens, shape (B, T, 1).
        """
        with torch.no_grad():
            q = torch.minimum(q1_pred, q2_pred)  # conservative Q estimate
            adv = q - values_pred
            # Normalize the advantage to zero-mean unit-variance over valid tokens so
            # the exponential weighting is invariant to the reward scale.
            valid = mask.unsqueeze(-1)
            num_valid = valid.sum().clamp(min=1.0)
            adv_mean = (adv * valid).sum() / num_valid
            adv_var = (((adv - adv_mean) ** 2) * valid).sum() / num_valid
            adv_std = torch.sqrt(adv_var.clamp(min=1e-8))
            adv_norm = (adv - adv_mean) / adv_std
            # Bounded exponential weighting (AWR/IQL formulation) - do NOT zero out half the dataset
            weight = torch.clamp(torch.exp(adv_norm / self.advantage_temperature), min=0.1, max=10.0)
            weight = weight * valid
            # Normalize so the weights have unit mean over valid tokens (stable loss scale).
            weight_sum = weight.sum().clamp(min=1.0)
            weight = weight * (num_valid / weight_sum)
        return weight

    def forward(
        self,
        target_velocity: torch.Tensor | None,
        pred_velocity: torch.Tensor | None,
        noise: torch.Tensor | None,
        q1_pred: torch.Tensor | None,
        q2_pred: torch.Tensor | None,
        q_target: torch.Tensor | None,
        values_pred: torch.Tensor | None,
        state_repr: torch.Tensor,
        action_chunk: torch.Tensor,
        critic: nn.Module,
        next_states_pred: torch.Tensor | None,
        target_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute HDML composite loss.

        Args:
            target_velocity: Flow-matching target velocity (a_1 - a_0).
            pred_velocity: Predicted velocity field v_theta.
            noise: The Gaussian noise a_0 sampled for flow interpolation.
            q1_pred/q2_pred: HiQC critic outputs, shape (B, T, 1).
            q_target: Target value (scaled return-to-go), shape (B, T, 1).
            values_pred: Backbone value head output, shape (B, T, 1).
            state_repr: Context representation, shape (B, T, context_dim).
            action_chunk: Ground-truth action chunk for PAVE, shape (B, T, c, d).
            critic: HiQC critic module for PAVE regularization.
            next_states_pred: Forward-dynamics prediction, shape (B, T, prop_dim).
            target_states: Normalized states, shape (B, T, prop_dim).
            mask: Validity mask, shape (B, T).
        """
        # --- 1. Flow Matching Loss ---
        if target_velocity is not None and pred_velocity is not None:
            flow_loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1) if pred_velocity.ndim == 4 else mask.unsqueeze(-1)
            # Normalize by the total number of valid scalar elements (B*T*C*D) so the
            # flow loss is a proper per-element mean (previously inflated by C*D).
            num_elements = mask.sum() * (pred_velocity.shape[-1] * pred_velocity.shape[-2] if pred_velocity.ndim == 4 else pred_velocity.shape[-1])
            if (
                self.use_advantage_weighting
                and q1_pred is not None
                and q2_pred is not None
                and values_pred is not None
            ):
                adv_weights = self._advantage_weights(q1_pred, q2_pred, values_pred, mask)
                adv_expanded = adv_weights.unsqueeze(-1) if pred_velocity.ndim == 4 else adv_weights
                flow_loss = (flow_loss * mask_expanded * adv_expanded).sum() / torch.clamp(num_elements, min=1.0)
            else:
                flow_loss = (flow_loss * mask_expanded).sum() / torch.clamp(num_elements, min=1.0)
        else:
            flow_loss = torch.tensor(0.0, device=mask.device)

        # --- 2. Q-Function Chunk Loss (k-step Bellman MSE) ---
        if q1_pred is not None and q2_pred is not None and q_target is not None:
            mask_expanded_q = mask.unsqueeze(-1)
            q1_loss = F.mse_loss(q1_pred, q_target, reduction="none") * mask_expanded_q
            q2_loss = F.mse_loss(q2_pred, q_target, reduction="none") * mask_expanded_q
            q_loss = (q1_loss.sum() + q2_loss.sum()) / torch.clamp(mask_expanded_q.sum() * 2, min=1.0)
        else:
            q_loss = torch.tensor(0.0, device=mask.device)

        # --- 3. Value Function Loss (IQL expectile regression toward Q) ---
        # V(s) is regressed toward the lower-bound Q(s, a) via asymmetric expectile
        # loss (tau < 0.5 penalizes V > Q more, keeping V conservative and below the
        # upper tail of Q). This yields a meaningful advantage Q - V for DIVO.
        if values_pred is not None and q1_pred is not None and q2_pred is not None:
            with torch.no_grad():
                q_min = torch.minimum(q1_pred, q2_pred).detach()
            mask_expanded_v = mask.unsqueeze(-1)
            v_loss = self.expectile_loss(q_min - values_pred, self.value_expectile) * mask_expanded_v
            v_loss = v_loss.sum() / torch.clamp(mask_expanded_v.sum(), min=1.0)
        else:
            v_loss = torch.tensor(0.0, device=mask.device)

        # --- 4. PAVE Loss (Hutchinson Trace Estimator) ---
        if self.pave_weight > 0.0 and state_repr is not None and action_chunk is not None and torch.is_grad_enabled():
            pave_loss = self.compute_pave_loss(critic, state_repr, action_chunk, mask)
        else:
            pave_loss = torch.tensor(0.0, device=mask.device)

        # --- 5. Grad-CAPS Loss (Action Smoothness on predicted chunk) ---
        if self.grad_caps_weight > 0.0 and pred_velocity is not None:
            # Reconstructed predicted action chunk (noise is None for Gaussian head).
            pred_chunk = pred_velocity if noise is None else pred_velocity + noise
            if pred_chunk.ndim == 4 and pred_chunk.shape[1] > 2:
                accel = pred_chunk[:, 2:] - 2 * pred_chunk[:, 1:-1] + pred_chunk[:, :-2]  # (B, T-2, c, d)
                accel_norm = (accel ** 2).sum(dim=-1).sum(dim=-1)  # (B, T-2)
                grad_caps_loss = (accel_norm * mask[:, 2:]).sum() / torch.clamp(mask[:, 2:].sum(), min=1.0)
            else:
                grad_caps_loss = torch.tensor(0.0, device=mask.device)
        else:
            grad_caps_loss = torch.tensor(0.0, device=mask.device)

        # --- 6. Forward Dynamics Predictor Loss ---
        if self.dynamics_weight > 0.0 and next_states_pred is not None and next_states_pred.shape[1] > 1:
            raw_dyn_loss = F.mse_loss(next_states_pred[:, :-1], target_states[:, 1:], reduction="none")
            dyn_mask = mask[:, 1:].unsqueeze(-1)
            dyn_loss = (raw_dyn_loss * dyn_mask).sum() / torch.clamp(dyn_mask.sum() * next_states_pred.shape[-1], min=1.0)
        else:
            dyn_loss = torch.tensor(0.0, device=mask.device)

        # Total Loss
        total_loss = (
            self.flow_weight * flow_loss
            + self.q_weight * q_loss
            + self.value_weight * v_loss
            + self.pave_weight * pave_loss
            + self.grad_caps_weight * grad_caps_loss
            + self.dynamics_weight * dyn_loss
        )

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "flow_loss": float(flow_loss.item()),
            "q_loss": float(q_loss.item()),
            "value_loss": float(v_loss.item()),
            "pave_loss": float(pave_loss.item()),
            "grad_caps_loss": float(grad_caps_loss.item()),
            "dynamics_loss": float(dyn_loss.item()),
        }

        return total_loss, loss_dict

    def compute_pave_loss(
        self, critic: nn.Module, state_repr: torch.Tensor, action_chunk: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the Policy-Aware Value-field Equalization (PAVE) regularization loss.
        Estimates the trace of the mixed Hessian \nabla_s (\nabla_a Q^T v) using Hutchinson trace.
        """
        # Detach states and actions to avoid backpropping through the policy/backbone unnecessarily
        # during the Hessian computation, as we only want to regularize the critic's curvature.
        s = state_repr.detach().requires_grad_(True)
        a = action_chunk.detach().requires_grad_(True)

        q1, _ = critic(s, a)
        q = q1.squeeze(-1)  # (B, T)

        # We only want to compute this over valid tokens
        q_sum = (q * mask).sum()

        # First derivative: \nabla_a Q
        grad_a = torch.autograd.grad(
            outputs=q_sum,
            inputs=a,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Random vector v ~ N(0, I)
        v = torch.randn_like(a)

        # Project gradient with v
        grad_a_v_sum = (grad_a * v).sum()

        # Second derivative: \nabla_s (\nabla_a Q^T v)
        grad_s = torch.autograd.grad(
            outputs=grad_a_v_sum,
            inputs=s,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # PAVE penalty: || \nabla_s (\nabla_a Q^T v) ||_2^2
        pave_penalty = (grad_s ** 2).sum(dim=-1)  # (B, T)

        # Mask and average
        pave_loss = (pave_penalty * mask).sum() / torch.clamp(mask.sum(), min=1.0)

        return pave_loss
