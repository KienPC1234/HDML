from __future__ import annotations

"""Portable (pure PyTorch) Mamba-3 SISO forward, for CPU inference and ONNX export.

The native ``mamba_ssm.Mamba3`` layer runs a fused Triton/CUTE kernel that cannot
be traced by ``torch.onnx`` nor executed on CPU. This module re-implements the
same SISO computation with plain tensor ops so the trained weights can be
exported and run on CPU.

It keeps the same parameter names and shapes as ``mamba_ssm.Mamba3`` (SISO,
``ngroups=1``, ``is_mimo=False``), so ``load_state_dict`` from a native module
works directly, and the two forward passes agree numerically (see parity test).

Shape contract:
    Input:  (Batch, Seq_Len, d_model)
    Output: (Batch, Seq_Len, d_model)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RMSNormNoGate(nn.Module):
    """RMS normalisation matching ``mamba_ssm`` RMSNormGated without gating."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class PortableMamba3(nn.Module):
    """SISO Mamba-3 implemented with dense O(T^2) tensor math (one chunk).

    Args:
        d_model: input/output width.
        d_state: SSM state size (also the Q/K head width).
        expand: inner expansion factor.
        headdim: per-head V width.
        rope_fraction: fraction of the state used for rotary embedding.
        A_floor: lower clamp on the decay rate magnitude.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        rope_fraction: float = 0.5,
        A_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.A_floor = A_floor
        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim
        self.num_bc_heads = 1
        self.mimo_rank = 1

        split = int(d_state * rope_fraction)
        if split % 2 != 0:
            split -= 1
        self.split_tensor_size = split
        self.num_rope_angles = split // 2

        d_in_proj = (
            2 * self.d_inner
            + 2 * self.d_state * self.num_bc_heads * self.mimo_rank
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, self.mimo_rank, self.d_state))
        self.B_norm = _RMSNormNoGate(self.d_state)
        self.C_norm = _RMSNormNoGate(self.d_state)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    @staticmethod
    def _apply_rope_adjacent(x: torch.Tensor, angles: torch.Tensor, n_angle_dims: int) -> torch.Tensor:
        """Rotate adjacent pairs of the first ``2*n_angle_dims`` features.

        x: (..., D); angles: (..., n_angle_dims) broadcastable to (..., n_angle_dims).
        """
        xr = x.clone()
        half = n_angle_dims  # number of pairs
        if half == 0:
            return xr
        a = xr[..., : 2 * half].reshape(*xr.shape[:-1], half, 2)
        x0, x1 = a[..., 0], a[..., 1]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        xr[..., : 2 * half] = torch.stack([y0, y1], dim=-1).reshape(*xr.shape[:-1], 2 * half)
        return xr

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, l, _ = u.shape
        zx = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap_raw, angle_raw = torch.split(
            zx,
            [
                self.d_inner,
                self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads,
                self.nheads,
                self.nheads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        # (b, l, nheads, headdim) / (b, l, d_state)
        z = z.view(b, l, self.nheads, self.headdim)
        x = x.view(b, l, self.nheads, self.headdim)
        B = B.view(b, l, self.num_bc_heads, self.d_state)
        C = C.view(b, l, self.num_bc_heads, self.d_state)
        dd_dt = dd_dt.view(b, l, self.nheads)
        dd_A = dd_A.view(b, l, self.nheads)
        trap = torch.sigmoid(trap_raw.view(b, l, self.nheads))

        # dt / A / adt
        dt = F.softplus(dd_dt + self.dt_bias)  # (b, l, nheads)
        A = -F.softplus(dd_A)
        A = torch.clamp(A, max=-self.A_floor)
        adt = (A * dt).transpose(1, 2)  # (b, nheads, l)
        dt_t = dt.transpose(1, 2)       # (b, nheads, l)
        trap_t = trap.transpose(1, 2)   # (b, nheads, l)

        # trapezoidal scale: dt_{t+1}(1 - trap_{t+1}) + dt_t trap_t
        dt_shift = torch.zeros_like(dt_t)
        dt_shift[:, :, :-1] = dt_t[:, :, 1:]
        trap_shift = torch.zeros_like(trap_t)
        trap_shift[:, :, :-1] = trap_t[:, :, 1:]
        scale = (dt_shift * (1 - trap_shift) + dt_t * trap_t).transpose(1, 2)  # (b,l,nheads)
        gamma = (dt_t * trap_t).transpose(1, 2)                              # (b,l,nheads)

        # normalise B, C then add bias
        Bn = self.B_norm(B).expand(b, l, self.nheads, self.d_state)
        Cn = self.C_norm(C).expand(b, l, self.nheads, self.d_state)
        k = Cn * 0 + Bn + self.B_bias.view(1, 1, self.nheads, self.d_state)
        q = Cn + self.C_bias.view(1, 1, self.nheads, self.d_state)

        # qk dot (pre-rope) * gamma: extra skip term
        qk_dot = (q * k).sum(dim=-1) * gamma  # (b,l,nheads)

        # rotary state-space embedding: cumsum(tanh(angle)*pi * dt) mod 2pi per head
        angle_vals = torch.tanh(angle_raw) * math.pi           # (b,l,num_angle)
        angle_dt = angle_vals.unsqueeze(2) * dt.unsqueeze(-1)  # (b,l,nheads,num_angle)
        angle_cs = torch.cumsum(angle_dt, dim=1)
        angle_cs = torch.remainder(angle_cs, 2 * math.pi)      # (b,l,nheads,num_angle)

        k = self._apply_rope_adjacent(k, angle_cs, self.num_rope_angles)
        q = self._apply_rope_adjacent(q, angle_cs, self.num_rope_angles)
        k_scaled = k * scale.unsqueeze(-1)                     # (b,l,nheads,d_state)

        # dense causal attention form of the SISO recurrence (single chunk)
        da_cs = torch.cumsum(adt, dim=-1).transpose(1, 2)      # (b,l,nheads)
        # s[b,i,j,h] = (q_i · k_scaled_j) * exp(da_cs_i - da_cs_j) for j < i
        qk = torch.einsum("blhd,bmhd->blmh", q, k_scaled)      # (b,l,l,nheads)
        decay = da_cs.unsqueeze(2) - da_cs.unsqueeze(1)        # (b,l,l,nheads)
        # Clamp the exponent: adt>0 or long L otherwise gives exp(+large)=Inf/NaN.
        decay = torch.clamp(decay, min=-50.0, max=10.0)
        causal = torch.tril(torch.ones(l, l, device=u.device, dtype=torch.bool), diagonal=-1)
        s = qk * torch.exp(decay) * causal.unsqueeze(0).unsqueeze(-1)
        o = torch.einsum("blmh,bmhd->blhd", s, x)             # (b,l,nheads,headdim)
        o = o + (self.D.view(1, 1, self.nheads, 1) + qk_dot.unsqueeze(-1)) * x

        # z-gating and output projection
        o = o * F.silu(z)
        y = o.reshape(b, l, self.d_inner)
        return self.out_proj(y)
