#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from hdml.models import (
    HDMLModel,
    DecisionTransformerBaseline,
    DecisionRNNBaseline,
    DiffusionPolicyBaseline,
    IQLBaseline,
    MLPBCBaseline,
    MambaMLPHeadAblation,
    TransformerLiquidHeadAblation,
)
from hdml.utils.config import ModelConfig


def count_parameters(module: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def compute_mamba_layer_flops(d_model: int, d_inner: int, d_state: int, d_conv: int, seq_len: int) -> float:
    """Analytical FLOPs for single Mamba S6 block."""
    # 1. in_proj: (B, L, d_model) -> (B, L, 2 * d_inner)
    in_proj_flops = 2.0 * seq_len * d_model * (2 * d_inner)
    # 2. conv1d: (B, 2*d_inner, L) with kernel d_conv
    conv_flops = 2.0 * seq_len * (2 * d_inner) * d_conv
    # 3. x_proj: (B, L, d_inner) -> (B, L, dt_rank + 2*d_state)
    dt_rank = max(1, d_model // 16)
    x_proj_flops = 2.0 * seq_len * d_inner * (dt_rank + 2 * d_state)
    # 4. dt_proj: (B, L, dt_rank) -> (B, L, d_inner)
    dt_proj_flops = 2.0 * seq_len * dt_rank * d_inner
    # 5. selective_scan SSM recurrence:
    # h_t = A * h_{t-1} + B * x_t  => 2 * d_inner * d_state
    # y_t = C * h_t                => 2 * d_inner * d_state
    ssm_flops = 4.0 * seq_len * d_inner * d_state
    # 6. out_proj: (B, L, d_inner) -> (B, L, d_model)
    out_proj_flops = 2.0 * seq_len * d_inner * d_model
    # 7. gating & silu activations
    act_flops = 3.0 * seq_len * d_inner

    return in_proj_flops + conv_flops + x_proj_flops + dt_proj_flops + ssm_flops + out_proj_flops + act_flops


def compute_transformer_layer_flops(d_model: int, nhead: int, seq_len: int) -> float:
    """Analytical FLOPs for standard Causal Transformer layer."""
    # Total tokens in DT is 3 * seq_len (R, s, a)
    T = 3 * seq_len
    # 1. Q, K, V projections
    qkv_flops = 3.0 * 2.0 * T * d_model * d_model
    # 2. Q @ K^T attention scores: (B, nhead, T, d_k) @ (B, nhead, d_k, T) -> (B, nhead, T, T)
    attn_score_flops = 2.0 * nhead * T * T * (d_model // nhead)
    # 3. Softmax & Dropout
    softmax_flops = 3.0 * nhead * T * T
    # 4. Attn @ V context: (B, nhead, T, T) @ (B, nhead, T, d_k) -> (B, nhead, T, d_k)
    attn_val_flops = 2.0 * nhead * T * T * (d_model // nhead)
    # 5. Out projection
    out_proj_flops = 2.0 * T * d_model * d_model
    # 6. MLP Feedforward (d_model -> 4*d_model -> d_model)
    ffn_flops = 2.0 * 2.0 * T * d_model * (4 * d_model)
    # 7. LayerNorms (2 per block)
    ln_flops = 4.0 * T * d_model

    return qkv_flops + attn_score_flops + softmax_flops + attn_val_flops + out_proj_flops + ffn_flops + ln_flops


def compute_liquid_head_flops(prop_dim: int, d_subgoal: int, cfc_units: int, action_dim: int, seq_len: int = 1) -> float:
    """Analytical FLOPs for Closed-Form Continuous-time (CfC) Motor Head."""
    in_dim = prop_dim + d_subgoal
    # Backbone layer (Linear + GELU)
    bb_flops = 2.0 * seq_len * in_dim * 64 + 2.0 * seq_len * 64 * cfc_units
    # CfC Gating projections (3 gates: f, g, h for time-constant tau ODE)
    gate_flops = 3.0 * 2.0 * seq_len * (in_dim + cfc_units) * cfc_units
    # Closed-form ODE solution calculation
    ode_solve_flops = 8.0 * seq_len * cfc_units
    # Action Head Linear projection
    act_proj_flops = 2.0 * seq_len * cfc_units * action_dim

    return bb_flops + gate_flops + ode_solve_flops + act_proj_flops


def compute_diffusion_flops(prop_dim: int, action_dim: int, d_model: int, denoising_steps: int = 10) -> float:
    """Analytical FLOPs for Diffusion Policy DDPM (K denoising steps)."""
    # Each step passes through 3-layer ResNet MLP: (action_dim + prop_dim + 1) -> d_model -> d_model -> action_dim
    in_dim = action_dim + prop_dim + d_model
    per_step_mlp = (
        2.0 * in_dim * d_model
        + 2.0 * d_model * d_model
        + 2.0 * d_model * d_model
        + 2.0 * d_model * action_dim
    )
    return float(denoising_steps * per_step_mlp)


def profile_all_architectures(
    prop_dim: int = 17,
    action_dim: int = 6,
    d_model: int = 128,
    num_layers: int = 3,
    context_length: int = 20,
) -> dict[str, dict[str, float]]:
    """Profile theoretical parameters and FLOPs across context horizons."""
    cfg = ModelConfig(
        prop_dim=prop_dim,
        action_dim=action_dim,
        d_model=d_model,
        num_mamba_layers=num_layers,
        d_subgoal=64,
        cfc_units=32,
    )

    # 1. Instantiate models
    hdml = HDMLModel.from_config(cfg)
    mamba_mlp = MambaMLPHeadAblation(prop_dim, action_dim, d_model, num_mamba_layers=num_layers)
    tf_liquid = TransformerLiquidHeadAblation(prop_dim, action_dim, d_model, num_layers=num_layers)
    dt = DecisionTransformerBaseline(prop_dim, action_dim, d_model, num_layers=num_layers)
    diff = DiffusionPolicyBaseline(prop_dim, action_dim, d_model, denoising_steps=10)
    rnn = DecisionRNNBaseline(prop_dim, action_dim, d_model, num_layers=num_layers)
    iql = IQLBaseline(prop_dim, action_dim, hidden_dim=256)
    mlp_bc = MLPBCBaseline(prop_dim, action_dim, hidden_dim=128)

    models_dict = {
        "HDML (Mamba + Liquid CfC - Ours)": hdml,
        "Ablation: Mamba + MLP Head": mamba_mlp,
        "Ablation: Transformer + Liquid Head": tf_liquid,
        "Decision Transformer (DT)": dt,
        "Diffusion Policy (DDPM K=10)": diff,
        "Decision RNN (LSTM)": rnn,
        "Implicit Q-Learning (IQL)": iql,
        "MLP-BC (Behavior Cloning)": mlp_bc,
    }

    # 2. Compute Parameters Breakdown
    param_counts = {name: count_parameters(m) for name, m in models_dict.items()}

    # 3. Analytical FLOPs for Single Step and Context Window T=20, 100, 500, 1000
    d_inner = d_model * 2
    d_state = 16
    d_conv = 4

    results = {}
    for name in models_dict:
        params = param_counts[name]

        if "HDML" in name:
            # Macro Mamba (L tokens) + Micro Liquid Head (1 step)
            flops_seq_20 = num_layers * compute_mamba_layer_flops(d_model, d_inner, d_state, d_conv, 20) + compute_liquid_head_flops(prop_dim, 64, 32, action_dim, 20)
            flops_seq_100 = num_layers * compute_mamba_layer_flops(d_model, d_inner, d_state, d_conv, 100) + compute_liquid_head_flops(prop_dim, 64, 32, action_dim, 100)
            flops_seq_500 = num_layers * compute_mamba_layer_flops(d_model, d_inner, d_state, d_conv, 500) + compute_liquid_head_flops(prop_dim, 64, 32, action_dim, 500)
            flops_seq_1000 = num_layers * compute_mamba_layer_flops(d_model, d_inner, d_state, d_conv, 1000) + compute_liquid_head_flops(prop_dim, 64, 32, action_dim, 1000)
            single_step_flops = compute_liquid_head_flops(prop_dim, 64, 32, action_dim, 1) + (num_layers * compute_mamba_layer_flops(d_model, d_inner, d_state, d_conv, 1) / 5.0)
            complexity_order = r"$\mathcal{O}(N) / \mathcal{O}(1)$"
        elif "Mamba + MLP" in name:
            flops_seq_20 = num_layers * compute_mamba_layer_flops(d_model, d_inner, d_state, d_conv, 20) + 2.0 * 20 * d_model * action_dim
            flops_seq_100 = num_layers * compute_mamba_layer_flops(d_model, d_inner, d_state, d_conv, 100) + 2.0 * 100 * d_model * action_dim
            flops_seq_500 = num_layers * compute_mamba_layer_flops(d_model, d_inner, d_state, d_conv, 500) + 2.0 * 500 * d_model * action_dim
            flops_seq_1000 = num_layers * compute_mamba_layer_flops(d_model, d_inner, d_state, d_conv, 1000) + 2.0 * 1000 * d_model * action_dim
            single_step_flops = 2.0 * d_model * action_dim + (num_layers * compute_mamba_layer_flops(d_model, d_inner, d_state, d_conv, 1) / 5.0)
            complexity_order = r"$\mathcal{O}(N) / \mathcal{O}(1)$"
        elif "Transformer" in name or "Decision Transformer" in name:
            flops_seq_20 = num_layers * compute_transformer_layer_flops(d_model, 4, 20)
            flops_seq_100 = num_layers * compute_transformer_layer_flops(d_model, 4, 100)
            flops_seq_500 = num_layers * compute_transformer_layer_flops(d_model, 4, 500)
            flops_seq_1000 = num_layers * compute_transformer_layer_flops(d_model, 4, 1000)
            single_step_flops = num_layers * compute_transformer_layer_flops(d_model, 4, 20) / 20.0
            complexity_order = r"$\mathcal{O}(N^2) / \mathcal{O}(N)$"
        elif "Diffusion" in name:
            single_step_flops = compute_diffusion_flops(prop_dim, action_dim, d_model, 10)
            flops_seq_20 = single_step_flops * 20
            flops_seq_100 = single_step_flops * 100
            flops_seq_500 = single_step_flops * 500
            flops_seq_1000 = single_step_flops * 1000
            complexity_order = r"$\mathcal{O}(K \cdot N) / \mathcal{O}(N)$"
        elif "RNN" in name:
            rnn_cell_flops = 2.0 * 4.0 * (d_model + prop_dim) * d_model * num_layers
            single_step_flops = rnn_cell_flops + 2.0 * d_model * action_dim
            flops_seq_20 = single_step_flops * 20
            flops_seq_100 = single_step_flops * 100
            flops_seq_500 = single_step_flops * 500
            flops_seq_1000 = single_step_flops * 1000
            complexity_order = r"$\mathcal{O}(N) / \mathcal{O}(1)$"
        else:  # IQL / MLP
            mlp_flops = 2.0 * (prop_dim * 256 + 256 * 256 + 256 * action_dim)
            single_step_flops = mlp_flops
            flops_seq_20 = mlp_flops * 20
            flops_seq_100 = mlp_flops * 100
            flops_seq_500 = mlp_flops * 500
            flops_seq_1000 = mlp_flops * 1000
            complexity_order = r"$\mathcal{O}(1) / \mathcal{O}(1)$"

        results[name] = {
            "params": params,
            "complexity": complexity_order,
            "single_step_mflops": single_step_flops / 1e6,
            "seq_20_mflops": flops_seq_20 / 1e6,
            "seq_100_mflops": flops_seq_100 / 1e6,
            "seq_500_mflops": flops_seq_500 / 1e6,
            "seq_1000_mflops": flops_seq_1000 / 1e6,
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile computational FLOPs and parameter complexity.")
    parser.add_argument("--prop-dim", type=int, default=17)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args()

    results = profile_all_architectures(
        prop_dim=args.prop_dim,
        action_dim=args.action_dim,
        d_model=args.d_model,
        num_layers=args.layers,
    )

    print("\n" + "=" * 125)
    print(f"THEORETICAL HARDWARE-INDEPENDENT COMPUTATIONAL COMPLEXITY & FLOPS ANALYSIS")
    print("=" * 125)
    print(
        f"{'Architecture / Model Variant':<42} | {'Params':<9} | {'Complexity':<12} | {'Step (MFLOPs)':<14} | {'T=20 (M)':<10} | {'T=100 (M)':<10} | {'T=1000 (M)':<10}"
    )
    print("-" * 125)
    for name, r in results.items():
        print(
            f"{name:<42} | {r['params']:<9,d} | {r['complexity']:<12} | {r['single_step_mflops']:<14.4f} | {r['seq_20_mflops']:<10.2f} | {r['seq_100_mflops']:<10.2f} | {r['seq_1000_mflops']:<10.2f}"
        )
    print("=" * 125 + "\n")

    # Generate Publication-Ready LaTeX Table
    print("\n% LaTeX Table 4: Theoretical Computational Complexity & FLOPs Breakdown")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Theoretical Hardware-Independent Computational Complexity, Parameter Footprint, and FLOPs Scaling across Temporal Horizons ($T$).}")
    print(r"\label{tab:complexity_flops}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{lccccccr}")
    print(r"\toprule")
    print(r"\textbf{Architecture / Model Variant} & \textbf{Trainable Params} & \textbf{Complexity (Time/Mem)} & \textbf{Per-Step (MFLOPs) $\downarrow$} & \textbf{$T=20$ (MFLOPs)} & \textbf{$T=100$ (MFLOPs)} & \textbf{$T=1000$ (MFLOPs)} \\")
    print(r"\midrule")

    for name, r in results.items():
        is_bold = "Ours" in name
        prefix = r"\textbf{" if is_bold else ""
        suffix = r"}" if is_bold else ""

        print(
            f"{prefix}{name}{suffix} & {r['params']:,d} & {r['complexity']} & "
            f"{r['single_step_mflops']:.4f} & {r['seq_20_mflops']:.2f} & {r['seq_100_mflops']:.2f} & {r['seq_1000_mflops']:.2f} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table*}")


if __name__ == "__main__":
    main()
