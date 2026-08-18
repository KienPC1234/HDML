from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hdml.models.fusion import CrossModalFusion


class DecisionTransformerBaseline(nn.Module):
    """Decision Transformer (DT) Baseline.
    
    Standard causal multi-head self-attention sequence model for offline RL.
    Quadratic time and memory complexity O(N^2).
    """

    def __init__(
        self,
        prop_dim: int = 27,
        action_dim: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.d_model = d_model

        self.fusion = CrossModalFusion(
            prop_dim=prop_dim,
            action_dim=action_dim,
            d_model=d_model,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Standard Linear / MLP Action Head (Discrete Step Prediction)
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        u_t = self.fusion(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=states.device)
        h = self.transformer(u_t, mask=causal_mask, is_causal=True)
        actions_pred = self.action_head(h)
        return actions_pred

    @torch.inference_mode()
    def get_action(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions_pred = self.forward(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)
        return actions_pred[:, -1, :]


class DecisionRNNBaseline(nn.Module):
    """Decision RNN / LSTM Baseline.
    
    Recurrent sequence model with constant step memory but no continuous ODE adaptation.
    """

    def __init__(
        self,
        prop_dim: int = 27,
        action_dim: int = 8,
        d_model: int = 128,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.d_model = d_model

        self.fusion = CrossModalFusion(
            prop_dim=prop_dim,
            action_dim=action_dim,
            d_model=d_model,
        )

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
        )

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        u_t = self.fusion(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)
        out, next_hx = self.lstm(u_t, hx)
        actions_pred = self.action_head(out)
        return actions_pred, next_hx

    @torch.inference_mode()
    def get_action(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        actions_pred, next_hx = self.forward(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps, hx=hx)
        return actions_pred[:, -1, :], next_hx


class DiffusionPolicyBaseline(nn.Module):
    """Diffusion Policy (DDPM Continuous Control Baseline).
    
    Generative action model using iterative score-based / DDPM denoising diffusion.
    High expressivity for multi-modal trajectories, but computationally slow (K >= 10 steps).
    """

    def __init__(
        self,
        prop_dim: int = 27,
        action_dim: int = 8,
        d_model: int = 128,
        denoising_steps: int = 10,
    ) -> None:
        super().__init__()
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.denoising_steps = denoising_steps

        # State conditioning network
        self.cond_encoder = nn.Sequential(
            nn.Linear(prop_dim + 1, d_model),
            nn.Mish(),
            nn.Linear(d_model, d_model),
            nn.Mish(),
        )

        # Diffusion noise prediction network
        self.noise_net = nn.Sequential(
            nn.Linear(action_dim + d_model + d_model, d_model * 2),
            nn.Mish(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.Mish(),
            nn.Linear(d_model * 2, action_dim),
        )

        # Precompute DDPM beta schedule (linear variance schedule)
        betas = torch.linspace(1e-4, 0.02, denoising_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def _timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal diffusion timestep embedding."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
        emb = timesteps.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self,
        noisy_actions: torch.Tensor,
        diffusion_k: torch.Tensor,
        states: torch.Tensor,
        rtgs: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise epsilon from noisy action, diffusion timestep k, and state conditioning."""
        if states.ndim == 3:
            states = states[:, -1, :]
        if rtgs.ndim == 3:
            rtgs = rtgs[:, -1, :]
        elif rtgs.ndim == 1:
            rtgs = rtgs.unsqueeze(-1)

        cond = self.cond_encoder(torch.cat([states, rtgs], dim=-1))
        t_emb = self._timestep_embedding(diffusion_k, self.d_model)

        inp = torch.cat([noisy_actions, t_emb, cond], dim=-1)
        return self.noise_net(inp)

    @torch.inference_mode()
    def get_action(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Iterative reverse diffusion sampling (K steps from Gaussian noise)."""
        if states.ndim == 3:
            cur_state = states[:, -1, :]
        else:
            cur_state = states
        if rtgs.ndim == 3:
            cur_rtg = rtgs[:, -1, :]
        elif rtgs.ndim == 1:
            cur_rtg = rtgs.unsqueeze(-1)
        else:
            cur_rtg = rtgs

        batch_size = cur_state.shape[0]
        dev = cur_state.device

        # Start from standard normal noise
        act_k = torch.randn((batch_size, self.action_dim), device=dev)

        # Iterative denoising loop (K steps)
        for k in reversed(range(self.denoising_steps)):
            k_tensor = torch.full((batch_size,), k, device=dev, dtype=torch.long)
            eps_pred = self.forward(act_k, k_tensor, cur_state, cur_rtg)

            alpha = self.alphas[k]
            alpha_cumprod = self.alphas_cumprod[k]
            beta = self.betas[k]

            # DDPM mean update
            act_k = (1.0 / torch.sqrt(alpha)) * (
                act_k - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_cumprod)) * eps_pred
            )

            if k > 0:
                noise = torch.randn_like(act_k)
                act_k = act_k + torch.sqrt(beta) * noise

        return torch.clamp(act_k, -1.0, 1.0)


class IQLBaseline(nn.Module):
    """Implicit Q-Learning (IQL / Pure Offline Q-Learning Baseline).
    
    Fast feedforward policy using expectile value regression without OOD query.
    Fast execution (> 100 Hz), but myopic 1-step Bellman updates without macro sequence context.
    """

    def __init__(
        self,
        prop_dim: int = 27,
        action_dim: int = 8,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.prop_dim = prop_dim
        self.action_dim = action_dim

        # Twin Q-Networks
        self.q1 = nn.Sequential(
            nn.Linear(prop_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(prop_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Value Network (Expectile Regression)
        self.v = nn.Sequential(
            nn.Linear(prop_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Actor Policy Network
        self.actor = nn.Sequential(
            nn.Linear(prop_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 3:
            states = states[:, -1, :]
        return self.actor(states)

    @torch.inference_mode()
    def get_action(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if states.ndim == 3:
            states = states[:, -1, :]
        elif states.ndim == 1:
            states = states.unsqueeze(0)
        return self.actor(states)


class MLPBCBaseline(nn.Module):
    """Standard Reactive MLP Behavior Cloning Baseline.
    
    Markovian feed-forward network without sequence context or continuous dynamics.
    """

    def __init__(
        self,
        prop_dim: int = 27,
        action_dim: int = 8,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prop_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor, rtgs: torch.Tensor) -> torch.Tensor:
        if states.ndim == 2:
            if rtgs.ndim == 1:
                rtgs = rtgs.unsqueeze(-1)
            elif rtgs.ndim == 3:
                rtgs = rtgs.squeeze(-1)
        elif states.ndim == 3:
            if rtgs.ndim == 2:
                rtgs = rtgs.unsqueeze(-1)
        inp = torch.cat([states, rtgs], dim=-1)
        return self.net(inp)

    @torch.inference_mode()
    def get_action(self, state: torch.Tensor, rtg: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if rtg.ndim == 0:
            rtg = rtg.unsqueeze(0).unsqueeze(-1)
        elif rtg.ndim == 1:
            if rtg.shape[0] == state.shape[0]:
                rtg = rtg.unsqueeze(-1)
            else:
                rtg = rtg.unsqueeze(0)
        inp = torch.cat([state, rtg], dim=-1)
        return self.net(inp)
