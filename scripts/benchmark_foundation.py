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
        "morphology": "8-DoF Hexapod",
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
            t0 = time.perf_counter()
            _ = model(
                states=states,
                rtgs=rtgs,
                actions=actions,
                timesteps=timesteps,
                embodiment_name=embodiment_name,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
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
    output_json: str = "logs/benchmark_foundation_results.json",
    device_str: str = "cuda",
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting HDML-Foundation Comprehensive Benchmark on {device}...")
    logger.info(f"Loading pre-trained foundation weights from {checkpoint_path}...")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": checkpoint_path,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "foundation_architecture": {
            "d_model": cfg.get("model", {}).get("d_model", 384),
            "num_mamba_layers": cfg.get("model", {}).get("num_mamba_layers", 8),
            "cfc_units": cfg.get("model", {}).get("cfc_units", 96),
        },
        "embodiment_benchmarks": [],
    }

    # Iterate through all benchmark targets
    for target_title, target_info in BENCHMARK_TARGETS.items():
        emb_name = target_info["embodiment_name"]
        ds_path = target_info["dataset_path"]
        prop_dim = target_info["prop_dim"]
        act_dim = target_info["action_dim"]
        morph = target_info["morphology"]

        if not Path(ds_path).exists():
            logger.warning(f"Dataset {ds_path} not found for {target_title}, skipping.")
            continue

        logger.info(f"\n=======================================================")
        logger.info(f"Benchmarking Target: {target_title} ({morph})")
        logger.info(f"State Dim: {prop_dim} | Action Dim: {act_dim}")
        logger.info(f"=======================================================")

        # Instantiate clean foundation model
        model = HDMLFoundationModel(
            d_model=cfg.get("model", {}).get("d_model", 384),
            num_mamba_layers=cfg.get("model", {}).get("num_mamba_layers", 8),
            cfc_units=cfg.get("model", {}).get("cfc_units", 96),
            cfc_backbone_units=cfg.get("model", {}).get("cfc_backbone_units", 192),
            device=device,
        )
        base_state_dict = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("adapters.")}
        model.load_state_dict(base_state_dict, strict=False)

        # 1. Register and Freeze
        adapter = model.register_embodiment(emb_name, prop_dim=prop_dim, action_dim=act_dim)
        model.freeze_backbone()

        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = frozen_params + trainable_params
        frozen_pct = (frozen_params / total_params) * 100.0

        logger.info(f"Parameters: Total={total_params:,} | Frozen={frozen_params:,} ({frozen_pct:.1f}%) | Adapter Trainable={trainable_params:,}")

        # 2. Measure Hardware Latency & Frequency
        timing_stats = benchmark_throughput_and_latency(
            model=model,
            embodiment_name=emb_name,
            prop_dim=prop_dim,
            action_dim=act_dim,
            device=device,
        )
        logger.info(f"Hardware Speed: Latency={timing_stats['mean_latency_ms']} ± {timing_stats['std_latency_ms']} ms | Frequency={timing_stats['control_frequency_hz']} Hz")

        # 3. Few-Shot Adaptation Evaluation (3 epochs, B=128)
        buffer = FastEmbodimentBuffer(name=emb_name, path=ds_path, context_length=30)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001, weight_decay=1e-4)

        t_start = time.perf_counter()
        epoch_losses: list[float] = []
        epoch_jerks: list[float] = []

        model.train()
        for ep in range(1, 4):
            ep_l = []
            ep_j = []
            for _ in range(100):
                batch = buffer.sample_batch(batch_size=128, device=device)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    acts_pred, _, _, _, _ = model(
                        states=batch["states"],
                        rtgs=batch["rtgs"],
                        actions=batch["actions_in"],
                        timesteps=batch["timesteps"],
                        embodiment_name=emb_name,
                        embodiment_idx=batch["embodiment_idx"],
                    )
                    loss = nn.functional.smooth_l1_loss(acts_pred, batch["actions_target"])

                loss.backward()
                optimizer.step()
                ep_l.append(loss.item())
                ep_j.append(compute_action_jerk(acts_pred))

            epoch_losses.append(float(np.mean(ep_l)))
            epoch_jerks.append(float(np.mean(ep_j)))
            logger.info(f"  Epoch {ep}/3 -> Action Loss: {epoch_losses[-1]:.5f} | Action Jerk |Δ²a|: {epoch_jerks[-1]:.4f}")

        t_end = time.perf_counter()
        adaptation_time = round(t_end - t_start, 2)
        logger.info(f"  Adaptation Time: {adaptation_time}s | Final Action Error: {epoch_losses[-1]:.5f}")

        results["embodiment_benchmarks"].append({
            "target": target_title,
            "morphology": morph,
            "prop_dim": prop_dim,
            "action_dim": act_dim,
            "frozen_params": frozen_params,
            "trainable_params": trainable_params,
            "frozen_percentage": round(frozen_pct, 1),
            "adaptation_time_s": adaptation_time,
            "final_action_loss": round(epoch_losses[-1], 5),
            "final_action_jerk": round(epoch_jerks[-1], 4),
            "latency_ms": timing_stats["mean_latency_ms"],
            "control_frequency_hz": timing_stats["control_frequency_hz"],
        })

    # Save results JSON
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nAll benchmark results successfully saved to {output_json}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HDML-Foundation Benchmark Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hdml_foundation/hdml_foundation_best.pt")
    parser.add_argument("--output", type=str, default="logs/benchmark_foundation_results.json")
    args = parser.parse_args()

    run_comprehensive_benchmark(checkpoint_path=args.checkpoint, output_json=args.output)
