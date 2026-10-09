#!/usr/bin/env python3
"""Measure pure-inference latency and closed-loop control cycle time for HDML.

The project report distinguishes two numbers that were previously conflated:
  * pure model inference latency (backbone + flow + CfC on a fixed context);
  * end-to-end closed-loop cycle = sensor read + preprocess + model inference +
    safety monitor + physics step.

This script measures both with explicit CUDA synchronisation and writes a JSON
record so every reported latency is traceable to a run.

Example
-------
    python scripts/measure_closed_loop.py \
        --config configs/halfcheetah_v5_default.yaml \
        --checkpoint checkpoints/rebuild/halfcheetah_v5/best_model.pt \
        --episodes 3 --output results/rebuild_20261009/latency.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import HDMLConfig
from hdml.utils.metrics import benchmark_inference_latency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("measure_closed_loop")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/halfcheetah_v5_default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/rebuild_20261009/latency.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    model = HDMLModel.from_config(cfg.model).to(device)
    state_mean = state_std = None
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        state_mean = ckpt.get("state_mean")
        state_std = ckpt.get("state_std")
    model.eval()

    # --- 1. Pure model inference latency on a fixed context window. ---
    ctx = cfg.training.context_length
    sample_states = torch.randn(1, ctx, cfg.model.prop_dim, device=device)
    sample_rtgs = torch.randn(1, ctx, 1, device=device)
    sample_actions = torch.randn(1, ctx, cfg.model.action_dim, device=device)
    sample_t = torch.arange(ctx, device=device).unsqueeze(0)
    latency = benchmark_inference_latency(
        model_fn=model.get_action,
        sample_inputs=(sample_states, sample_rtgs, sample_actions, sample_t),
        num_warmup=30,
        num_iterations=200,
        device=device,
    )
    logger.info(
        "Pure inference: %.3f ms (%.1f Hz)", latency["mean_latency_ms"], latency["throughput_hz"]
    )

    # --- 2. Closed-loop cycle = full control step inside the MuJoCo loop. ---
    env = gym.make(cfg.env.env_name)
    obs_dim = env.observation_space.shape[0]  # type: ignore
    act_dim = env.action_space.shape[0]  # type: ignore
    st_mean = state_mean if state_mean is not None else np.zeros(obs_dim, dtype=np.float32)
    st_std = state_std if state_std is not None else np.ones(obs_dim, dtype=np.float32)

    cycle_ms: list[float] = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=cfg.env.seed + ep)
        hs, ha, hr, ht = [], [], [], []
        for t in range(cfg.env.max_episode_steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = __import__("time").perf_counter()
            raw = np.asarray(obs, dtype=np.float32)
            norm = (raw - st_mean) / st_std
            hs.append(norm)
            hr.append(0.0)
            ht.append(t)
            if not ha:
                ha.append(np.zeros(act_dim, dtype=np.float32))
            cl = min(len(hs), ctx)
            ts = torch.from_numpy(np.array(hs[-cl:], dtype=np.float32)).unsqueeze(0).to(device)
            ta = torch.from_numpy(np.array(ha[-cl:], dtype=np.float32)).unsqueeze(0).to(device)
            tr = torch.from_numpy(np.array(hr[-cl:], dtype=np.float32).reshape(-1, 1)).unsqueeze(0).to(device)
            tt = torch.from_numpy(np.array(ht[-cl:], dtype=np.int64)).unsqueeze(0).to(device)
            with torch.inference_mode():
                action_t, _, _ = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)
            action = np.clip(action_t[0].cpu().numpy().astype(np.float32), -1.0, 1.0)
            ha.append(action)
            obs, _, term, trunc, _ = env.step(action)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = __import__("time").perf_counter()
            cycle_ms.append((t1 - t0) * 1000.0)
            if term or trunc:
                break
    env.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mean_cycle = float(statistics.fmean(cycle_ms))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- 3. Portable CPU deployment path (Mamba-1 + RoPE + flow + CfC). ---
    cpu_inference_ms = None
    try:
        from hdml.deployment.onnx_exporter import HDMLDeploymentWrapper
        portable = HDMLModel(
            prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim,
            d_model=cfg.model.d_model, num_mamba_layers=cfg.model.num_mamba_layers,
            d_subgoal=cfg.model.d_subgoal, chunk_size=cfg.model.chunk_size,
            use_native_mamba3=False,
        ).cpu().eval()
        portable.deterministic = True
        portable.num_flow_steps = 4
        wrapper = HDMLDeploymentWrapper(portable).eval()
        ps = torch.randn(1, ctx, cfg.model.prop_dim)
        pr = torch.randn(1, ctx, 1)
        pa = torch.randn(1, ctx, cfg.model.action_dim)
        pt = torch.arange(ctx).unsqueeze(0)
        import time as _time
        with torch.inference_mode():
            for _ in range(10):
                wrapper(ps, pr, pa, pt)
            t0 = _time.perf_counter()
            for _ in range(100):
                wrapper(ps, pr, pa, pt)
            cpu_inference_ms = (_time.perf_counter() - t0) / 100 * 1000.0
        logger.info("CPU portable inference: %.2f ms (%.1f Hz)", cpu_inference_ms, 1000.0 / cpu_inference_ms)
    except Exception as exc:  # noqa: BLE001 - CPU path is optional
        logger.warning("CPU portable measurement skipped: %s", exc)

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "env": cfg.env.env_name,
        "device": str(device),
        "context_length": ctx,
        "pure_inference_ms": latency["mean_latency_ms"],
        "pure_inference_hz": latency["throughput_hz"],
        "closed_loop_cycle_ms": mean_cycle,
        "closed_loop_hz": 1000.0 / mean_cycle if mean_cycle > 0 else 0.0,
        "cpu_portable_inference_ms": cpu_inference_ms,
        "cpu_portable_inference_hz": (1000.0 / cpu_inference_ms) if cpu_inference_ms else None,
        "cycle_samples": len(cycle_ms),
        "note": "closed-loop includes sensor read, normalisation, model inference and env physics step; "
                "cpu path is the portable Mamba-1 deployment wrapper",
    }
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info(
        "Closed-loop: %.3f ms (%.1f Hz). Written to %s",
        mean_cycle, record["closed_loop_hz"], out,
    )


if __name__ == "__main__":
    main()
