"""
HiQC (Hierarchical Implicit Q-Chunking) Critic.
Evaluates Q-values for entire action chunks rather than single-step actions.
"""
from __future__ import annotations

import torch
import torch.nn as nn

class HiQCCritic(nn.Module):
    """
    Critic for Action Chunks.
    Estimates Q(s, a_{t:t+k}) for a given state representation and action chunk.
    """
    def __init__(self, state_dim: int, action_dim: int, chunk_size: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        
        # Flattened action chunk dimension
        self.chunk_dim = chunk_size * action_dim
        
        # Double Q-learning architecture (Twin Delayed DDPG / SAC style)
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + self.chunk_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + self.chunk_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_repr: torch.Tensor, action_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q1 and Q2 values for the given state and action chunk.
        
        Args:
            state_repr: Tensor of shape (B, T, state_dim) or (B, state_dim)
            action_chunk: Tensor of shape (B, T, chunk_size, action_dim) or (B, chunk_size, action_dim)
            
        Returns:
            q1: Tensor of shape (B, T, 1) or (B, 1)
            q2: Tensor of shape (B, T, 1) or (B, 1)
        """
        # Flatten action chunk
        if action_chunk.ndim == 4:  # (B, T, chunk_size, action_dim)
            B, T, C, D = action_chunk.shape
            assert C == self.chunk_size and D == self.action_dim
            flat_actions = action_chunk.view(B, T, C * D)
        elif action_chunk.ndim == 3:  # (B, chunk_size, action_dim)
            B, C, D = action_chunk.shape
            assert C == self.chunk_size and D == self.action_dim
            flat_actions = action_chunk.view(B, C * D)
        else:
            raise ValueError(f"Unexpected action chunk shape: {action_chunk.shape}")
            
        assert state_repr.shape[:-1] == flat_actions.shape[:-1], \
            f"Batch/Time dimensions mismatch: {state_repr.shape} vs {flat_actions.shape}"
            
        xu = torch.cat([state_repr, flat_actions], dim=-1)
        
        q1 = self.q1_net(xu)
        q2 = self.q2_net(xu)
        return q1, q2
