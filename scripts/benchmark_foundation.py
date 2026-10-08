"""Comprehensive Benchmark Suite for HDML-Foundation Model.

Evaluates:
1. Multi-Embodiment Few-Shot Transfer (Frozen Mamba-Liquid Backbone, Trainable Adapter only).
2. Action Prediction Error & Smoothness (Jerk |Δ²a|).
3. Hardware Inference Throughput & Latency (Hz / FPS) on active GPU.
4. Parameter Efficiency (% frozen vs % trained).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn

from hdml.models.foundation import HDMLFoundationModel
from hdml.data.multi_embodiment_dataset import FastEmbodimentBuffer

logger = logging.getLogger("BenchmarkFoundation")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(sh)


BENCHMARK_TARGETS = {
    "Unitree A1 (Quadruped)": {
        "embodiment_name": "unitree_a1_maze",
        "dataset_path": "data/unitree_a1_maze_trajectories.npz",
        "prop_dim": 53,
        "action_dim": 12,
        "morphology": "12-DoF Quadruped",
    },
    "Ant (Insect / 4-Legged)": {
        "embodiment_name": "ant",
        "dataset_path": "data/ant_foundation.npz",
        "prop_dim": 105,
        "action_dim": 8,
        "morphology": "8-DoF Quadruped",
    },
    "Walker2d (Bipedal)": {
        "embodiment_name": "walker2d",
        "dataset_path": "data/walker2d_foundation.npz",
        "prop_dim": 17,
        "action_dim": 6,
        "morphology": "6-DoF Bipedal",
    },
    "Hopper (1-Legged Hop)": {
        "embodiment_name": "hopper",
        "dataset_path": "data/hopper_foundation.npz",
        "prop_dim": 11,
        "action_dim": 3,
        "morphology": "3-DoF Monopod",
    },
    "Humanoid (Bipedal 3D)": {
        "embodiment_name": "humanoid",
        "dataset_path": "data/humanoid_foundation.npz",
        "prop_dim": 348,
        "action_dim": 17,
        "morphology": "17-DoF Humanoid",
    },
    "Swimmer (Fluid / Snake)": {
        "embodiment_name": "swimmer",
        "dataset_path": "data/swimmer_foundation.npz",
        "prop_dim": 8,
        "action_dim": 2,
        "morphology": "2-DoF Serpentine",
    },
}


def benchmark_throughput_and_latency(
    model: HDMLFoundationModel,
    embodiment_name: str,
    prop_dim: int,
    action_dim: int,
    device: torch.device,
    num_warmup: int = 100,
    num_steps: int = 1000,
) -> dict[str, float]:
    """Measure device-synchronized inference latency and control frequency (Hz)."""
    model.eval()
    states = torch.randn(1, 30, prop_dim, device=device)
    actions = torch.randn(1, 30, action_dim, device=device)
    rtgs = torch.randn(1, 30, 1, device=device)
    timesteps = torch.arange(30, device=device).unsqueeze(0)

    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(
                states=states,
                rtgs=rtgs,
                actions=actions,
                timesteps=timesteps,
                embodiment_name=embodiment_name,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Benchmark timing
    latencies: list[float] = []
    with torch.inference_mode():
        for _ in range(num_steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(
                states=states,
                rtgs=rtgs,
                actions=actions,
                timesteps=timesteps,
                embodiment_name=embodiment_name,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_lat, 3),
        "std_latency_ms": round(std_lat, 3),
        "control_frequency_hz": round(fps, 1),
    }


def compute_action_jerk(actions: torch.Tensor) -> float:
    """Compute mean second difference |Δ²a| as the action jerk metric."""
    if actions.shape[1] < 3:
        return 0.0
    delta1 = actions[:, 1:] - actions[:, :-1]
    delta2 = delta1[:, 1:] - delta1[:, :-1]
    return float(delta2.abs().mean().item())


def run_comprehensive_benchmark(
    checkpoint_path: str,
    output_json: str = "results/corrected/benchmark_foundation_results.json",
    device_str: str = "cuda",
    allow_pretraining_overlap: bool = False,
    seed: int = 42,
) -> None:
    """Evaluate each adapter on held-out target episodes with explicit exposure labels."""
    from scripts.evaluate_foundation_transfer import evaluate_transfer
    missing = [info["dataset_path"] for info in BENCHMARK_TARGETS.values()
               if not Path(info["dataset_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Required target datasets are missing: {missing}")
    results = []
    for info in BENCHMARK_TARGETS.values():
        results.append(evaluate_transfer(
            checkpoint_path, info["embodiment_name"], info["dataset_path"],
            info["prop_dim"], info["action_dim"], device=device_str, seed=seed,
            allow_pretraining_overlap=allow_pretraining_overlap))
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"protocol_version": 2, "results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDML-Foundation Benchmark Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hdml_foundation/hdml_foundation_best.pt")
    parser.add_argument("--output", type=str, default="logs/benchmark_foundation_results.json")
    parser.add_argument("--allow-pretraining-overlap", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_comprehensive_benchmark(checkpoint_path=args.checkpoint, output_json=args.output,
                                allow_pretraining_overlap=args.allow_pretraining_overlap, seed=args.seed)

