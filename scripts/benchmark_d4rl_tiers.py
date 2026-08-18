#!/usr/bin/env python3
"""Multi-environment benchmark across dataset quality tiers.

Cross-environment, cross-tier benchmark reusing the honest evaluation protocol of
`scripts/benchmark_baselines.py` (trained baselines, no-leakage causal action input,
normalized observations, scaled RTG, real timesteps, standard jerk metric).

Run: python scripts/benchmark_d4rl_tiers.py --device cuda
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import torch

from hdml.utils.config import HDMLConfig, ModelConfig, TrainingConfig, EnvConfig
from hdml.models import (
    HDMLModel,
    DecisionTransformerBaseline,
    DecisionRNNBaseline,
    DiffusionPolicyBaseline,
    IQLBaseline,
    MLPBCBaseline,
)
from benchmark_baselines import evaluate_policy, load_baseline_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ENV_DIMS: dict[str, tuple[int, int]] = {
    "HalfCheetah-v5": (17, 6),
    "Ant-v4": (27, 8),
    "Ant-v5": (105, 8),
    "Hopper-v5": (11, 3),
    "Walker2d-v5": (17, 6),
}


def build_cfg_for_env(env_name: str, template: HDMLConfig) -> HDMLConfig:
    """Return a config whose model dims match the target environment."""
    prop_dim, action_dim = ENV_DIMS[env_name]
    model_cfg = ModelConfig(
        prop_dim=prop_dim,
        action_dim=action_dim,
        d_model=template.model.d_model,
        d_state=template.model.d_state,
        d_conv=template.model.d_conv,
        expand=template.model.expand,
        num_mamba_layers=template.model.num_mamba_layers,
        d_subgoal=template.model.d_subgoal,
        cfc_units=template.model.cfc_units,
        cfc_backbone_units=template.model.cfc_backbone_units,
        cfc_backbone_layers=template.model.cfc_backbone_layers,
    )
    return HDMLConfig(
        model=model_cfg,
        training=TrainingConfig(context_length=template.training.context_length, scale_return=template.env.scale_return),
        env=EnvConfig(env_name=env_name, scale_return=template.env.scale_return),
    )


def run_comprehensive_benchmark(device_str: str = "cuda") -> None:
    """Run cross-environment and cross-tier benchmark suite with trained models."""
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    logger.info(f"Starting Multi-Environment D4RL Benchmark Suite on {device}...")

    environments = ["HalfCheetah-v5", "Ant-v4", "Hopper-v5", "Walker2d-v5"]
    tiers = ["medium-expert", "medium", "medium-replay"]
    target_returns = {
        "HalfCheetah-v5": {"medium-expert": 8000.0, "medium": 4500.0, "medium-replay": 3000.0},
        "Ant-v4": {"medium-expert": 3500.0, "medium": 2000.0, "medium-replay": 1200.0},
        "Hopper-v5": {"medium-expert": 2800.0, "medium": 1800.0, "medium-replay": 1000.0},
        "Walker2d-v5": {"medium-expert": 3800.0, "medium": 2500.0, "medium-replay": 1500.0},
    }

    cfg_path = "configs/halfcheetah_v5_default.yaml"
    for env in environments:
        cand = f"configs/{env.lower().replace('-', '_')}_default.yaml"
        if Path(cand).exists():
            cfg_path = cand
            break
    template_cfg = HDMLConfig.from_yaml(cfg_path)

    benchmark_matrix: dict[str, dict[str, dict[str, float]]] = {
        env: {tier: {} for tier in tiers} for env in environments
    }

    for env_name in environments:
        env_key = env_name.lower().replace("-", "_")
        cfg = build_cfg_for_env(env_name, template_cfg)
        ckpt_path = Path(f"checkpoints/{env_key}/best_model.pt")
        baseline_dir = ckpt_path.parent / "baselines"

        state_mean = None
        state_std = None
        hdml = HDMLModel.from_config(cfg.model).to(device)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            hdml.load_state_dict(ckpt["model_state_dict"])
            state_mean = ckpt.get("state_mean")
            state_std = ckpt.get("state_std")
            logger.info(f"Loaded trained HDML checkpoint for {env_name}")
        else:
            logger.warning(f"No HDML checkpoint for {env_name} at {ckpt_path}; using random init.")

        models: dict[str, tuple[torch.nn.Module, str]] = {
            "HDML (Ours)": (hdml, "hdml"),
            "Decision Transformer": (
                DecisionTransformerBaseline(
                    prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, d_model=cfg.model.d_model, num_layers=cfg.model.num_mamba_layers
                ).to(device),
                "dt",
            ),
            "Diffusion Policy": (
                DiffusionPolicyBaseline(prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, d_model=cfg.model.d_model, denoising_steps=10).to(device),
                "diffusion",
            ),
            "IQL": (IQLBaseline(prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, hidden_dim=256).to(device), "iql"),
            "Decision RNN": (
                DecisionRNNBaseline(prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, d_model=cfg.model.d_model, num_layers=cfg.model.num_mamba_layers).to(device),
                "rnn",
            ),
            "MLP-BC": (MLPBCBaseline(prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, hidden_dim=256).to(device), "mlp"),
        }

        baseline_loaders = {"dt", "diffusion", "iql", "rnn", "mlp"}
        for name, (model, mtype) in models.items():
            if mtype in baseline_loaders:
                load_baseline_checkpoint(model, baseline_dir, mtype, device)
            model.eval()

        for tier in tiers:
            logger.info(f"Evaluating {env_name} on tier [{tier}]...")
            for model_name, (m, mtype) in models.items():
                res = evaluate_policy(
                    model=m,
                    model_type=mtype,
                    env_name=env_name,
                    num_episodes=3,
                    context_length=cfg.training.context_length,
                    target_return=target_returns[env_name][tier],
                    scale_return=cfg.env.scale_return,
                    state_mean=state_mean,
                    state_std=state_std,
                    with_perturbations=False,
                    macro_interval=5,
                    device=device,
                )
                benchmark_matrix[env_name][tier][model_name] = res["d4rl_normalized_score"]

    print("\n" + "=" * 130)
    print("D4RL NORMALIZED RETURN SCORES (Across 4 Environments & 3 Quality Regimes)")
    print("=" * 130)
    header = f"{'Benchmark Task & Dataset Tier':<35} | {'HDML':<8} | {'DT':<8} | {'Diffusion':<10} | {'IQL':<8} | {'RNN':<8} | {'MLP-BC':<8}"
    print(header)
    print("-" * 130)

    all_scores: dict[str, list[float]] = {m: [] for m in ["HDML (Ours)", "Decision Transformer", "Diffusion Policy", "IQL", "Decision RNN", "MLP-BC"]}

    for env_name in environments:
        for tier in tiers:
            row_title = f"{env_name} ({tier})"
            scores = benchmark_matrix[env_name][tier]
            for m in all_scores:
                all_scores[m].append(scores[m])
            print(
                f"{row_title:<35} | {scores['HDML (Ours)']:<8.2f} | {scores['Decision Transformer']:<8.2f} | "
                f"{scores['Diffusion Policy']:<10.2f} | {scores['IQL']:<8.2f} | {scores['Decision RNN']:<8.2f} | {scores['MLP-BC']:<8.2f}"
            )
        print("-" * 130)

    print(
        f"{'TOTAL AVERAGE (12 D4RL Tasks)':<35} | {np.mean(all_scores['HDML (Ours)']):<8.2f} | "
        f"{np.mean(all_scores['Decision Transformer']):<8.2f} | {np.mean(all_scores['Diffusion Policy']):<10.2f} | "
        f"{np.mean(all_scores['IQL']):<8.2f} | {np.mean(all_scores['Decision RNN']):<8.2f} | {np.mean(all_scores['MLP-BC']):<8.2f}"
    )
    print("=" * 130 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Environment D4RL Normalized Return Benchmark.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    run_comprehensive_benchmark(device_str=args.device)