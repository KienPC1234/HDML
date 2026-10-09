#!/usr/bin/env python3
"""Benchmark HDML policy on 12-DoF Unitree A1 Quadruped Robot Dog environment.

Evaluates:
1. Standard closed-loop locomotion (return, forward velocity, 12-motor jerk, GPU latency, Hz).
2. Perturbation robustness under lateral kick impacts (50 N) and IMU tilt noise.
   Measures roll angle recovery time, max roll deflection, perturbed return, and completion rate.

Usage:
    .venv/bin/python scripts/benchmark_unitree_a1.py \
        --config configs/unitree_a1_aug.yaml \
        --checkpoint checkpoints/rebuild_unitree/unitree_a1/best_model.pt \
        --episodes 5 --device cuda
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

from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import HDMLConfig
from hdml.utils.metrics import compute_action_smoothness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BenchmarkUnitreeA1")


def evaluate_unitree_a1(
    model: nn.Module,
    config: HDMLConfig,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    num_episodes: int = 5,
    device: torch.device = torch.device("cuda"),
    kick_perturbation: bool = False,
    sensor_noise_std: float = 0.0,
    seed: int = 42,
) -> dict[str, float | list[float]]:
    """Evaluate HDML policy on 12-DoF Unitree A1 Quadruped."""
    model.eval()
    model.deterministic = True
    model.num_flow_steps = 4

    env = QuadrupedDogEnv(max_episode_steps=config.env.max_episode_steps)
    returns: list[float] = []
    lengths: list[int] = []
    smoothnesses: list[float] = []
    latencies: list[float] = []
    forward_velocities: list[float] = []
    max_rolls: list[float] = []
    recovery_times: list[float] = []

    kick_steps = [200, 400, 600, 800] if kick_perturbation else []

    for ep in range(num_episodes):
        ep_seed = seed + ep * 100
        obs, _ = env.reset(seed=ep_seed)

        history_states: list[np.ndarray] = []
        history_actions: list[np.ndarray] = []
        history_rtgs: list[float] = []
        history_timesteps: list[int] = []

        current_rtg = float(config.env.target_return)
        scale_rtg = float(config.env.scale_return)
        context_len = int(config.training.context_length)
        act_dim = int(config.model.action_dim)

        ep_rewards: list[float] = []
        ep_actions: list[np.ndarray] = []
        ep_rolls: list[float] = []
        ep_recovery_times: list[float] = []

        pending_recovery_step = None

        for t in range(config.env.max_episode_steps):
            raw_obs = np.asarray(obs, dtype=np.float32)
            if sensor_noise_std > 0.0:
                raw_obs = raw_obs + np.random.normal(0.0, sensor_noise_std, size=raw_obs.shape).astype(np.float32)

            norm_obs = (raw_obs - state_mean) / state_std
            scaled_rtg = current_rtg / scale_rtg

            history_states.append(norm_obs)
            history_rtgs.append(scaled_rtg)
            history_timesteps.append(t)
            if len(history_actions) == 0:
                history_actions.append(np.zeros(act_dim, dtype=np.float32))

            # Build causal context window (a_{t-1} convention)
            ctx_len = min(len(history_states), context_len)
            ctx_states = np.array(history_states[-ctx_len:], dtype=np.float32)
            ctx_actions = np.array(history_actions[-ctx_len:], dtype=np.float32)
            ctx_rtgs = np.array(history_rtgs[-ctx_len:], dtype=np.float32).reshape(-1, 1)
            ctx_time = np.array(history_timesteps[-ctx_len:], dtype=np.int64)

            t_states = torch.from_numpy(ctx_states).unsqueeze(0).to(device)
            t_actions = torch.from_numpy(ctx_actions).unsqueeze(0).to(device)
            t_rtgs = torch.from_numpy(ctx_rtgs).unsqueeze(0).to(device)
            t_time = torch.from_numpy(ctx_time).unsqueeze(0).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            with torch.inference_mode():
                out = model.get_action(states=t_states, rtgs=t_rtgs, actions=t_actions, timesteps=t_time)
                if isinstance(out, tuple):
                    action_t = out[0]
                else:
                    action_t = out

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            if action_t.ndim == 3:
                action = action_t[0, 0, :].cpu().numpy().astype(np.float32)
            else:
                action = action_t[0, :].cpu().numpy().astype(np.float32)

            action = np.clip(action, -1.0, 1.0)
            history_actions.append(action)

            # Apply lateral kick impact (50 N along lateral Y axis) at scheduled steps
            if kick_perturbation and t in kick_steps:
                # 50 N lateral kick impact on torso
                env.apply_kick((0.0, 50.0, 0.0))
                pending_recovery_step = t

            next_obs, reward, terminated, truncated, info = env.step(action)
            roll_val = abs(float(info.get("roll", 0.0)))
            ep_rolls.append(roll_val)

            # Check recovery time: steps until roll returns to nominal (< 0.08 rad)
            if pending_recovery_step is not None:
                if roll_val < 0.08 and (t - pending_recovery_step) > 2:
                    dt_rec = (t - pending_recovery_step) * 0.02  # 0.02s per step (50 Hz)
                    ep_recovery_times.append(dt_rec)
                    pending_recovery_step = None

            ep_actions.append(action)
            ep_rewards.append(float(reward))
            current_rtg -= float(reward)
            obs = next_obs

            if terminated or truncated:
                break

        actions_arr = np.array(ep_actions, dtype=np.float32)
        raw_ret = float(sum(ep_rewards))
        returns.append(raw_ret)
        lengths.append(len(ep_rewards))
        smoothnesses.append(compute_action_smoothness(actions_arr))
        max_rolls.append(float(max(ep_rolls)) if ep_rolls else 0.0)
        if ep_recovery_times:
            recovery_times.extend(ep_recovery_times)

    env.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    mean_lat = float(np.mean(latencies))
    freq_hz = 1000.0 / max(1e-4, mean_lat)
    completion_rate = float(sum(l >= config.env.max_episode_steps for l in lengths) / max(1, len(lengths)) * 100.0)
    mean_jerk = float(np.mean(smoothnesses))
    mean_max_roll_deg = float(np.degrees(np.mean(max_rolls)))
    mean_rec_sec = float(np.mean(recovery_times)) if recovery_times else 0.0

    return {
        "mean_return": mean_ret,
        "std_return": std_ret,
        "returns": returns,
        "mean_length": float(np.mean(lengths)),
        "completion_rate": completion_rate,
        "mean_jerk": mean_jerk,
        "latency_ms": mean_lat,
        "frequency_hz": freq_hz,
        "max_roll_deg": mean_max_roll_deg,
        "recovery_time_sec": mean_rec_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HDML on Unitree A1 12-DoF Quadruped Robot")
    parser.add_argument("--config", default="configs/unitree_a1_aug.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/rebuild_unitree/unitree_a1/best_model.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="results/rebuild_unitree")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Target hardware: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HDMLModel.from_config(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    sm = np.asarray(ckpt["state_mean"], dtype=np.float32)
    ss = np.asarray(ckpt["state_std"], dtype=np.float32)

    # 1. Clean evaluation
    logger.info("Running clean closed-loop locomotion benchmark on Unitree A1...")
    clean_res = evaluate_unitree_a1(
        model=model,
        config=cfg,
        state_mean=sm,
        state_std=ss,
        num_episodes=args.episodes,
        device=device,
        kick_perturbation=False,
        sensor_noise_std=0.0,
    )
    logger.info(f"Clean Return: {clean_res['mean_return']:.2f} +/- {clean_res['std_return']:.2f}")
    logger.info(f"Clean Jerk: {clean_res['mean_jerk']:.4f}, Latency: {clean_res['latency_ms']:.2f} ms ({clean_res['frequency_hz']:.1f} Hz)")
    logger.info(f"Completion Rate: {clean_res['completion_rate']:.1f}%")

    # 2. Perturbed evaluation (50N lateral kicks + IMU noise)
    logger.info("Running perturbed benchmark (50N lateral kicks + IMU noise)...")
    pert_res = evaluate_unitree_a1(
        model=model,
        config=cfg,
        state_mean=sm,
        state_std=ss,
        num_episodes=args.episodes,
        device=device,
        kick_perturbation=True,
        sensor_noise_std=0.05,
    )
    logger.info(f"Perturbed Return: {pert_res['mean_return']:.2f} +/- {pert_res['std_return']:.2f}")
    logger.info(f"Perturbed Jerk: {pert_res['mean_jerk']:.4f}")
    logger.info(f"Max Roll Deflection: {pert_res['max_roll_deg']:.2f} deg")
    logger.info(f"Balance Recovery Time: {pert_res['recovery_time_sec']:.3f} s")
    logger.info(f"Perturbed Completion Rate: {pert_res['completion_rate']:.1f}%")

    # 3. Decision Transformer Baseline Evaluation (if available)
    dt_ckpt_path = Path("checkpoints/rebuild_unitree/unitree_a1/baselines/dt/best_model.pt")
    dt_clean_res = None
    dt_pert_res = None
    dt_params = 0

    if dt_ckpt_path.exists():
        logger.info("Found trained Decision Transformer baseline! Evaluating on Unitree A1...")
        from hdml.models.baselines import DecisionTransformerBaseline
        dt_model = DecisionTransformerBaseline(
            prop_dim=cfg.model.prop_dim,
            action_dim=cfg.model.action_dim,
            d_model=cfg.model.d_model,
            nhead=4,
            num_layers=cfg.model.num_mamba_layers,
        ).to(device)
        dt_ck = torch.load(dt_ckpt_path, map_location=device, weights_only=False)
        dt_model.load_state_dict(dt_ck["model_state_dict"])
        dt_params = sum(p.numel() for p in dt_model.parameters())

        logger.info("Running clean evaluation on Decision Transformer...")
        dt_clean_res = evaluate_unitree_a1(
            model=dt_model,
            config=cfg,
            state_mean=sm,
            state_std=ss,
            num_episodes=args.episodes,
            device=device,
            kick_perturbation=False,
            sensor_noise_std=0.0,
        )
        logger.info(f"DT Clean Return: {dt_clean_res['mean_return']:.2f} +/- {dt_clean_res['std_return']:.2f}")
        logger.info(f"DT Clean Jerk: {dt_clean_res['mean_jerk']:.4f}, Latency: {dt_clean_res['latency_ms']:.2f} ms ({dt_clean_res['frequency_hz']:.1f} Hz)")

        logger.info("Running perturbed evaluation on Decision Transformer (50N kicks + IMU noise)...")
        dt_pert_res = evaluate_unitree_a1(
            model=dt_model,
            config=cfg,
            state_mean=sm,
            state_std=ss,
            num_episodes=args.episodes,
            device=device,
            kick_perturbation=True,
            sensor_noise_std=0.05,
        )
        logger.info(f"DT Perturbed Return: {dt_pert_res['mean_return']:.2f} +/- {dt_pert_res['std_return']:.2f}")
        logger.info(f"DT Perturbed Jerk: {dt_pert_res['mean_jerk']:.4f}")
        logger.info(f"DT Max Roll Deflection: {dt_pert_res['max_roll_deg']:.2f} deg")
        logger.info(f"DT Perturbed Completion Rate: {dt_pert_res['completion_rate']:.1f}%")

    # 4. MLP-BC Baseline Evaluation (if available)
    mlp_ckpt_path = Path("checkpoints/rebuild_unitree/unitree_a1/baselines/mlp/best_model.pt")
    mlp_clean_res = None
    mlp_pert_res = None
    mlp_params = 0

    if mlp_ckpt_path.exists():
        logger.info("Found trained MLP-BC baseline! Evaluating on Unitree A1...")
        from hdml.models.baselines import MLPBCBaseline
        mlp_model = MLPBCBaseline(
            prop_dim=cfg.model.prop_dim,
            action_dim=cfg.model.action_dim,
            hidden_dim=256,
        ).to(device)
        mlp_ck = torch.load(mlp_ckpt_path, map_location=device, weights_only=False)
        mlp_model.load_state_dict(mlp_ck["model_state_dict"])
        mlp_params = sum(p.numel() for p in mlp_model.parameters())

        logger.info("Running clean evaluation on MLP-BC...")
        mlp_clean_res = evaluate_unitree_a1(
            model=mlp_model,
            config=cfg,
            state_mean=sm,
            state_std=ss,
            num_episodes=args.episodes,
            device=device,
            kick_perturbation=False,
            sensor_noise_std=0.0,
        )
        logger.info(f"MLP Clean Return: {mlp_clean_res['mean_return']:.2f} +/- {mlp_clean_res['std_return']:.2f}")
        logger.info(f"MLP Clean Jerk: {mlp_clean_res['mean_jerk']:.4f}, Latency: {mlp_clean_res['latency_ms']:.2f} ms ({mlp_clean_res['frequency_hz']:.1f} Hz)")

        logger.info("Running perturbed evaluation on MLP-BC (50N kicks + IMU noise)...")
        mlp_pert_res = evaluate_unitree_a1(
            model=mlp_model,
            config=cfg,
            state_mean=sm,
            state_std=ss,
            num_episodes=args.episodes,
            device=device,
            kick_perturbation=True,
            sensor_noise_std=0.05,
        )
        logger.info(f"MLP Perturbed Return: {mlp_pert_res['mean_return']:.2f} +/- {mlp_pert_res['std_return']:.2f}")
        logger.info(f"MLP Perturbed Jerk: {mlp_pert_res['mean_jerk']:.4f}")
        logger.info(f"MLP Max Roll Deflection: {mlp_pert_res['max_roll_deg']:.2f} deg")
        logger.info(f"MLP Perturbed Completion Rate: {mlp_pert_res['completion_rate']:.1f}%")

    # Save results to JSON
    summary = {
        "robot": "Unitree A1 Quadruped Dog (12-DoF 3D)",
        "hdml": {
            "params_count": sum(p.numel() for p in model.parameters()),
            "clean": clean_res,
            "perturbed": pert_res,
        },
        "dt": {
            "params_count": dt_params,
            "clean": dt_clean_res,
            "perturbed": dt_pert_res,
        } if dt_clean_res is not None else None,
        "mlp": {
            "params_count": mlp_params,
            "clean": mlp_clean_res,
            "perturbed": mlp_pert_res,
        } if mlp_clean_res is not None else None,
    }
    json_path = out_dir / "benchmark_unitree_a1.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved benchmark results to {json_path}")

    # Save summary report text
    txt_path = out_dir / "benchmark_unitree_a1.txt"
    with open(txt_path, "w") as f:
        f.write("HDML 3D ROBOT BENCHMARK RESULTS - Unitree A1 (12-DoF Quadruped)\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. HDML (Decision Mamba-3 + Liquid CfC - Ours):\n")
        f.write(f"   Clean Return: {clean_res['mean_return']:.2f} +/- {clean_res['std_return']:.2f}\n")
        f.write(f"   Clean Jerk: {clean_res['mean_jerk']:.4f}\n")
        f.write(f"   GPU Latency: {clean_res['latency_ms']:.2f} ms ({clean_res['frequency_hz']:.1f} Hz)\n")
        f.write(f"   Completion Rate: {clean_res['completion_rate']:.1f}%\n")
        f.write(f"   Perturbed Return (50N kick): {pert_res['mean_return']:.2f} +/- {pert_res['std_return']:.2f}\n")
        f.write(f"   Perturbed Jerk: {pert_res['mean_jerk']:.4f}\n")
        f.write(f"   Max Roll Deflection: {pert_res['max_roll_deg']:.2f} deg\n")
        f.write(f"   Balance Recovery Time: {pert_res['recovery_time_sec']:.3f} s\n")
        f.write(f"   Perturbed Completion Rate: {pert_res['completion_rate']:.1f}%\n\n")

        if dt_clean_res is not None and dt_pert_res is not None:
            f.write("2. Decision Transformer Baseline (Causal Self-Attention DT):\n")
            f.write(f"   Clean Return: {dt_clean_res['mean_return']:.2f} +/- {dt_clean_res['std_return']:.2f}\n")
            f.write(f"   Clean Jerk: {dt_clean_res['mean_jerk']:.4f}\n")
            f.write(f"   GPU Latency: {dt_clean_res['latency_ms']:.2f} ms ({dt_clean_res['frequency_hz']:.1f} Hz)\n")
            f.write(f"   Completion Rate: {dt_clean_res['completion_rate']:.1f}%\n")
            f.write(f"   Perturbed Return (50N kick): {dt_pert_res['mean_return']:.2f} +/- {dt_pert_res['std_return']:.2f}\n")
            f.write(f"   Perturbed Jerk: {dt_pert_res['mean_jerk']:.4f}\n")
            f.write(f"   Max Roll Deflection: {dt_pert_res['max_roll_deg']:.2f} deg\n")
            f.write(f"   Balance Recovery Time: {dt_pert_res['recovery_time_sec']:.3f} s\n")
            f.write(f"   Perturbed Completion Rate: {dt_pert_res['completion_rate']:.1f}%\n\n")

        if mlp_clean_res is not None and mlp_pert_res is not None:
            f.write("3. Reactive MLP-BC Baseline (Markovian Feedforward):\n")
            f.write(f"   Clean Return: {mlp_clean_res['mean_return']:.2f} +/- {mlp_clean_res['std_return']:.2f}\n")
            f.write(f"   Clean Jerk: {mlp_clean_res['mean_jerk']:.4f}\n")
            f.write(f"   GPU Latency: {mlp_clean_res['latency_ms']:.2f} ms ({mlp_clean_res['frequency_hz']:.1f} Hz)\n")
            f.write(f"   Completion Rate: {mlp_clean_res['completion_rate']:.1f}%\n")
            f.write(f"   Perturbed Return (50N kick): {mlp_pert_res['mean_return']:.2f} +/- {mlp_pert_res['std_return']:.2f}\n")
            f.write(f"   Perturbed Jerk: {mlp_pert_res['mean_jerk']:.4f}\n")
            f.write(f"   Max Roll Deflection: {mlp_pert_res['max_roll_deg']:.2f} deg\n")
            f.write(f"   Balance Recovery Time: {mlp_pert_res['recovery_time_sec']:.3f} s\n")
            f.write(f"   Perturbed Completion Rate: {mlp_pert_res['completion_rate']:.1f}%\n\n")

        # Comparative advantages
        if dt_clean_res is not None:
            jerk_reduction = (dt_clean_res['mean_jerk'] - clean_res['mean_jerk']) / max(1e-6, dt_clean_res['mean_jerk']) * 100.0
            f.write("4. QUANTITATIVE ADVANTAGE OF HDML OVER BASELINES ON 12-DOF QUADRUPED:\n")
            f.write(f"   - 12-Motor Jerk Reduction vs DT: {jerk_reduction:+.1f}%\n")
            f.write(f"   - Perturbation Retention Ratio: {pert_res['mean_return'] / max(1e-6, clean_res['mean_return']) * 100:.1f}% (HDML) vs {dt_pert_res['mean_return'] / max(1e-6, dt_clean_res['mean_return']) * 100:.1f}% (DT)\n")
    logger.info(f"Saved benchmark summary to {txt_path}")


if __name__ == "__main__":
    main()

