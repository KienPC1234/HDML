"""
Generative Q-Weighted Flow Matching Policy for Action Chunk Generation.
Replaces the Gaussian Unimodal Policy with an Optimal Transport Flow Matching process.
"""
from __future__ import annotations

import torch
import torch.nn as nn

class FlowVelocityField(nn.Module):
    """
    Velocity field v_theta for Flow Matching.
    Predicts the derivative da/d_tau given the current action chunk a_tau, time tau, and context.
    """
    def __init__(self, action_dim: int, chunk_size: int, context_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.chunk_dim = action_dim * chunk_size
        
        # We embed the continuous time scalar tau into a higher dimensional vector
        self.tau_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.net = nn.Sequential(
            nn.Linear(self.chunk_dim + context_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.chunk_dim)
        )
        
    def forward(self, a_tau: torch.Tensor, tau: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a_tau: Tensor of shape (..., chunk_size, action_dim) or (..., chunk_dim)
            tau: Tensor of shape (..., 1)
            context: Tensor of shape (..., context_dim)
            
        Returns:
            v_theta: Velocity tensor of the same shape as a_tau.
        """
        orig_shape = a_tau.shape
        is_4d = a_tau.ndim == 4
        
        if is_4d:
            B, T, C, D = a_tau.shape
            a_tau_flat = a_tau.view(B, T, C * D)
        elif a_tau.ndim == 3:
            B, C, D = a_tau.shape
            a_tau_flat = a_tau.view(B, C * D)
        else:
            a_tau_flat = a_tau
            
        t_emb = self.tau_embed(tau)
        
        # Concatenate inputs
        x = torch.cat([a_tau_flat, t_emb, context], dim=-1)
        
        v_flat = self.net(x)
        
        if is_4d:
            return v_flat.view(B, T, C, D)
        elif len(orig_shape) >= 2 and orig_shape[-2] == self.chunk_size and orig_shape[-1] == self.action_dim:
            return v_flat.view(*orig_shape)
            
        return v_flat


class FlowPolicy(nn.Module):
    """
    1-Step to N-Step Flow Matching Policy.
    """
    def __init__(self, action_dim: int, chunk_size: int, context_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.v_field = FlowVelocityField(action_dim, chunk_size, context_dim, hidden_dim)
        
    def forward_train(self, a_1: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the target velocity and predicted velocity for the flow matching loss.
        
        Args:
            a_1: Ground truth action chunk, shape (B, T, chunk_size, action_dim)
            context: Context tensor, shape (B, T, context_dim)
            
        Returns:
            target_velocity: (a_1 - a_0)
            pred_velocity: v_theta(a_tau, tau, context)
            a_0: The sampled Gaussian noise (allows reconstructing the predicted
                 action chunk as `pred_velocity + a_0` for Grad-CAPS regularization).
        """
        a_0 = torch.randn_like(a_1)
        
        # Sample random tau ~ U(0, 1)
        if a_1.ndim == 4:
            B, T, C, D = a_1.shape
            tau = torch.rand((B, T, 1), device=a_1.device)
        else:
            B = a_1.shape[0]
            tau = torch.rand((B, 1), device=a_1.device)
            
        # Interpolate a_tau
        # Expand tau to match a_1 dims
        tau_expanded = tau.unsqueeze(-1) if a_1.ndim == 4 else tau.unsqueeze(-1)
        
        a_tau = tau_expanded * a_1 + (1.0 - tau_expanded) * a_0
        target_velocity = a_1 - a_0
        
        pred_velocity = self.v_field(a_tau, tau, context)
        
        return target_velocity, pred_velocity, a_0
        
    @torch.inference_mode()
    def sample(self, context: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        """
        Sample action chunks using Euler integration.
        
        Args:
            context: (B, context_dim)
            num_steps: Number of integration steps (1 to 4 is recommended)
            
        Returns:
            a_1: Sampled action chunk (B, chunk_size, action_dim)
        """
        B = context.shape[0]
        a_tau = torch.randn((B, self.chunk_size, self.action_dim), device=context.device)
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            tau_val = i * dt
            tau = torch.full((B, 1), tau_val, device=context.device)
            v = self.v_field(a_tau, tau, context)
            a_tau = a_tau + v * dt
            
        return a_tau


class GaussianActionPolicy(nn.Module):
    """Deterministic/Gaussian action-chunk policy head (BC-style regressor).

    Replaces the flow-matching velocity field with a direct state-conditioned
    action-chunk regressor. Reconstructs actions far more accurately than flow
    matching (the flow objective has an irreducible noise floor from the Gaussian
    prior), at the cost of unimodal (non-generative) samples.

    Interface is identical to FlowPolicy so the trainer/evaluator are unchanged:
      - forward_train(a_1, context) -> (a_1, mu, None)
      - sample(context) -> mu (deterministic) or mu + sigma * eps
    """

    def __init__(
        self,
        action_dim: int,
        chunk_size: int,
        context_dim: int,
        hidden_dim: int = 256,
        log_std: float = -2.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.chunk_dim = action_dim * chunk_size
        self.context_dim = context_dim

        self.mu_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.chunk_dim),
        )
        # Fixed (learnable) log-std for optional stochastic sampling.
        self.log_std = nn.Parameter(torch.full((self.chunk_dim,), float(log_std)))

    def forward_train(
        self, a_1: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Regress the action chunk mean against the ground truth chunk.

        Args:
            a_1: Ground truth action chunk, (..., chunk_size, action_dim).
            context: Conditioning context, (..., context_dim).

        Returns:
            (a_1, mu, None): target actions, predicted mean, and no noise term.
        """
        mu_flat = self.mu_net(context)
        if a_1.ndim == 4:
            B, T, C, D = a_1.shape
            mu = mu_flat.view(B, T, C, D)
        elif a_1.ndim == 3:
            B, C, D = a_1.shape
            mu = mu_flat.view(B, C, D)
        else:
            mu = mu_flat
        return a_1, mu, None

    @torch.inference_mode()
    def sample(
        self, context: torch.Tensor, num_steps: int = 1, stochastic: bool = False
    ) -> torch.Tensor:
        """Return the deterministic action chunk (or a low-variance sample).

        Args:
            context: (B, context_dim).
            num_steps: Ignored (kept for FlowPolicy interface compatibility).
            stochastic: If True, adds Gaussian noise scaled by the learned std.

        Returns:
            action chunk (B, chunk_size, action_dim).
        """
        B = context.shape[0]
        mu = self.mu_net(context).view(B, self.chunk_size, self.action_dim)
        if not stochastic:
            return mu
        std = torch.exp(self.log_std).view(1, self.chunk_size, self.action_dim)
        return mu + std * torch.randn_like(mu)
