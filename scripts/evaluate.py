#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import torch

from hdml.utils.config import HDMLConfig
from hdml.models.hdml_model import HDMLModel
from hdml.evaluation.evaluator import HDMLEvaluator
from hdml.utils.metrics import benchmark_inference_latency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HDML policy in MuJoCo simulation.")
    parser.add_argument("--config", type=str, default="configs/ant_v4_default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pt")
    parser.add_argument("--episodes", type=int, default=5, help="Number of benchmark episodes")
    parser.add_argument("--macro-interval", type=int, default=5, help="Macro-planning interval (e.g. 5 or 10 steps per Mamba call)")
    parser.add_argument("--device", type=str, default="cuda", help="Evaluation device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    model = HDMLModel.from_config(cfg.model).to(device)
    state_mean = None
    state_std = None

    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        state_mean = ckpt.get("state_mean")
        state_std = ckpt.get("state_std")
    else:
        logger.warning("No checkpoint provided or checkpoint file not found. Evaluating model with initialized weights.")

    evaluator = HDMLEvaluator(
        model=model,
        env_name=cfg.env.env_name,
        context_length=cfg.training.context_length,
        target_return=cfg.env.target_return,
        scale_return=cfg.env.scale_return,
        state_mean=state_mean,
        state_std=state_std,
        device=device,
    )

    # 1. Standard Benchmark Evaluation with Macro Interval Decoupling
    logger.info(f"=== 1. Standard Evaluation (Macro Interval={args.macro_interval}, Perturbations=OFF) ===")
    std_results = evaluator.evaluate_benchmark(num_episodes=args.episodes, with_perturbations=False, macro_interval=args.macro_interval)
    print(f"Standard Results: Mean Return = {std_results['mean_return']:.2f} +/- {std_results['std_return']:.2f} | Jerk = {std_results['mean_smoothness']:.4f}")

    # 2. Perturbation Robustness Evaluation
    logger.info(f"=== 2. Robustness Evaluation (Macro Interval={args.macro_interval}, Perturbations=ON) ===")
    rob_results = evaluator.evaluate_benchmark(num_episodes=args.episodes, with_perturbations=True, macro_interval=args.macro_interval)
    print(f"Robustness Results: Mean Return = {rob_results['mean_return']:.2f} +/- {rob_results['std_return']:.2f} | Jerk = {rob_results['mean_smoothness']:.4f}")

    # 3. Latency & Throughput Benchmark
    logger.info("=== 3. Hardware Latency & Throughput Benchmark ===")
    sample_states = torch.randn(1, cfg.training.context_length, cfg.model.prop_dim, device=device)
    sample_rtgs = torch.randn(1, cfg.training.context_length, 1, device=device)
    sample_actions = torch.randn(1, cfg.training.context_length, cfg.model.action_dim, device=device)
    sample_timesteps = torch.arange(cfg.training.context_length, device=device).unsqueeze(0)

    latency_stats = benchmark_inference_latency(
        model_fn=model.get_action,
        sample_inputs=(sample_states, sample_rtgs, sample_actions, sample_timesteps),
        num_warmup=20,
        num_iterations=100,
        device=device,
    )
    print(f"Inference Latency: Mean = {latency_stats['mean_latency_ms']:.3f} ms | Min = {latency_stats['min_latency_ms']:.3f} ms | Throughput = {latency_stats['throughput_hz']:.1f} Hz")


if __name__ == "__main__":
    main()
