#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any
import gymnasium as gym
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
)
from hdml.evaluation.perturbations import SensorNoisePerturbation, ForceImpulsePerturbation
from hdml.utils.metrics import compute_action_smoothness
from hdml.utils.config import HDMLConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Standard D4RL Reference Scores (random / expert) for benchmark environments
D4RL_REF_SCORES: dict[str, tuple[float, float]] = {
    "HalfCheetah-v4": (-281.0, 12135.0),
    "halfcheetah": (-281.0, 12135.0),
    "Ant-v4": (-325.0, 4700.0),
    "ant": (-325.0, 4700.0),
    "Humanoid-v4": (123.0, 6000.0),
    "humanoid": (123.0, 6000.0),
}


def get_d4rl_normalized_score(env_name: str, raw_return: float) -> float:
    """Compute standard D4RL normalized score in range [0, 100+]."""
    for key, (r_score, e_score) in D4RL_REF_SCORES.items():
        if key.lower() in env_name.lower():
            return 100.0 * (raw_return - r_score) / (e_score - r_score)
    return raw_return


def evaluate_policy(
    model: nn.Module,
    model_type: str,
    env_name: str = "HalfCheetah-v4",
    num_episodes: int = 5,
    context_length: int = 20,
    target_return: float = 4000.0,
    scale_return: float = 1000.0,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
    with_perturbations: bool = False,
    macro_interval: int = 5,
    device: torch.device = torch.device("cuda"),
) -> dict[str, Any]:
    """Evaluate a single policy architecture across benchmark episodes."""
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]  # type: ignore
    act_dim = env.action_space.shape[0]        # type: ignore

    st_mean = state_mean if state_mean is not None else np.zeros(obs_dim, dtype=np.float32)
    st_std = state_std if state_std is not None else np.ones(obs_dim, dtype=np.float32)

    sensor_noise = SensorNoisePerturbation(noise_std=0.05, seed=42) if with_perturbations else None
    force_perturb = ForceImpulsePerturbation(impulse_prob=0.05, force_magnitude=0.6, seed=42) if with_perturbations else None

    returns: list[float] = []
    lengths: list[int] = []
    smoothnesses: list[float] = []
    latencies: list[float] = []

    model = model.to(device)
    model.eval()

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=42 + ep)
        history_states: list[np.ndarray] = []
        history_actions: list[np.ndarray] = []
        history_rtgs: list[float] = []
        history_timesteps: list[int] = []

        ep_actions: list[np.ndarray] = []
        ep_rewards: list[float] = []

        current_rtg = target_return
        hx = None
        current_subgoal = None
        prev_action = np.zeros(act_dim, dtype=np.float32)

        for t in range(1000):
            raw_obs = np.asarray(obs, dtype=np.float32)
            if sensor_noise is not None:
                raw_obs = sensor_noise.apply(raw_obs)

            norm_obs = (raw_obs - st_mean) / st_std
            scaled_rtg = current_rtg / scale_return

            history_states.append(norm_obs)
            history_rtgs.append(scaled_rtg)
            history_timesteps.append(t)
            history_actions.append(prev_action.copy())

            # Build context
            ctx_len = min(len(history_states), context_length)
            ctx_states = np.array(history_states[-ctx_len:], dtype=np.float32)
            ctx_actions = np.array(history_actions[-ctx_len:], dtype=np.float32)
            ctx_rtgs = np.array(history_rtgs[-ctx_len:], dtype=np.float32).reshape(-1, 1)
            ctx_time = np.array(history_timesteps[-ctx_len:], dtype=np.int64)

            # Pad context if needed
            if ctx_len < context_length:
                pad_k = context_length - ctx_len
                inp_states = np.vstack([np.zeros((pad_k, obs_dim), dtype=np.float32), ctx_states])
                inp_actions = np.vstack([np.zeros((pad_k, act_dim), dtype=np.float32), ctx_actions])
                inp_rtgs = np.vstack([np.zeros((pad_k, 1), dtype=np.float32), ctx_rtgs])
                inp_time = np.concatenate([np.zeros((pad_k,), dtype=np.int64), ctx_time])
            else:
                inp_states = ctx_states
                inp_actions = ctx_actions
                inp_rtgs = ctx_rtgs
                inp_time = ctx_time

            t_states = torch.from_numpy(inp_states).unsqueeze(0).to(device)
            t_actions = torch.from_numpy(inp_actions).unsqueeze(0).to(device)
            t_rtgs = torch.from_numpy(inp_rtgs).unsqueeze(0).to(device)
            t_time = torch.from_numpy(inp_time).unsqueeze(0).to(device)

            step_t0 = time.perf_counter()
            with torch.inference_mode():
                if model_type == "hdml":
                    if t % macro_interval == 0 or current_subgoal is None:
                        action_t, hx, current_subgoal = model.get_action(
                            states=t_states, rtgs=t_rtgs, actions=t_actions, timesteps=t_time, hx=hx
                        )
                    else:
                        t_prop = torch.from_numpy(norm_obs).unsqueeze(0).to(device)
                        action_t, hx = model.liquid_head(subgoals=current_subgoal, current_prop=t_prop, hx=hx)
                elif model_type == "diffusion":
                    action_t = model.get_action(states=t_states, rtgs=t_rtgs, actions=t_actions, timesteps=t_time)
                elif model_type == "dt":
                    action_t = model.get_action(states=t_states, rtgs=t_rtgs, actions=t_actions, timesteps=t_time)
                elif model_type == "iql":
                    t_cur_state = torch.from_numpy(norm_obs).unsqueeze(0).to(device)
                    action_t = model.get_action(states=t_cur_state)
                elif model_type == "rnn":
                    action_t, hx = model.get_action(states=t_states, rtgs=t_rtgs, actions=t_actions, timesteps=t_time, hx=hx)
                elif model_type == "mlp":
                    t_cur_state = torch.from_numpy(norm_obs).unsqueeze(0).to(device)
                    t_cur_rtg = torch.tensor([[scaled_rtg]], dtype=torch.float32, device=device)
                    action_t = model.get_action(t_cur_state, t_cur_rtg)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

            step_t1 = time.perf_counter()
            latencies.append((step_t1 - step_t0) * 1000.0)

            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
            prev_action = action.copy()
            history_actions[-1] = action

            exec_action = action
            if force_perturb is not None:
                exec_action = force_perturb.apply_action_perturbation(action)

            next_obs, reward, terminated, truncated, _ = env.step(exec_action)
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

    env.close()

    params_count = sum(p.numel() for p in model.parameters())
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    mean_lat = float(np.mean(latencies))
    freq_hz = 1000.0 / max(1e-4, mean_lat)
    d4rl_norm_score = get_d4rl_normalized_score(env_name, mean_ret)

    return {
        "model_type": model_type,
        "mean_return": mean_ret,
        "std_return": std_ret,
        "d4rl_normalized_score": d4rl_norm_score,
        "mean_length": float(np.mean(lengths)),
        "survival_rate": float(sum(l >= 1000 for l in lengths) / max(1, len(lengths)) * 100.0),
        "mean_smoothness_jerk": float(np.mean(smoothnesses)),
        "mean_latency_ms": mean_lat,
        "frequency_hz": freq_hz,
        "params_count": params_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HDML against SOTA baseline paradigms.")
    parser.add_argument("--config", type=str, default="configs/halfcheetah_v4_default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/halfcheetah_v4/best_model.pt", help="Path to trained HDML checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    args = parser.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"Execution hardware: {device}")

    # 1. Instantiate HDML (Trained)
    hdml_model = HDMLModel.from_config(cfg.model).to(device)
    state_mean = None
    state_std = None
    if Path(args.checkpoint).exists():
        logger.info(f"Loading trained HDML checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        hdml_model.load_state_dict(ckpt["model_state_dict"])
        state_mean = ckpt.get("state_mean")
        state_std = ckpt.get("state_std")
    else:
        logger.warning("Checkpoint not found, evaluating with initialized weights.")

    # 2. Instantiate SOTA Baselines
    diffusion_model = DiffusionPolicyBaseline(
        prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, d_model=cfg.model.d_model, denoising_steps=10
    ).to(device)

    dt_model = DecisionTransformerBaseline(
        prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, d_model=cfg.model.d_model
    ).to(device)

    iql_model = IQLBaseline(
        prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, hidden_dim=256
    ).to(device)

    rnn_model = DecisionRNNBaseline(
        prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, d_model=cfg.model.d_model
    ).to(device)

    mlp_model = MLPBCBaseline(
        prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, hidden_dim=256
    ).to(device)

    models_to_test = [
        ("HDML (Decision Mamba + Liquid CfC - Ours)", hdml_model, "hdml"),
        ("Diffusion Policy (DDPM 10-step Denoising)", diffusion_model, "diffusion"),
        ("Decision Transformer (Causal Attention DT)", dt_model, "dt"),
        ("Implicit Q-Learning (IQL / Value-Advantage)", iql_model, "iql"),
        ("Decision RNN (LSTM Recurrent Policy)", rnn_model, "rnn"),
        ("MLP-BC (Standard Feedforward Reactive)", mlp_model, "mlp"),
    ]

    print("\n" + "=" * 115)
    print(f"ACADEMIC SOTA BENCHMARK COMPARISON ON {cfg.env.env_name.upper()} (Standard & Perturbed Robustness)")
    print("=" * 115)

    results_std: list[dict[str, Any]] = []
    results_rob: list[dict[str, Any]] = []

    for name, model, mtype in models_to_test:
        logger.info(f"Running Standard Benchmark on: {name}...")
        res_s = evaluate_policy(
            model=model,
            model_type=mtype,
            env_name=cfg.env.env_name,
            num_episodes=args.episodes,
            context_length=cfg.training.context_length,
            state_mean=state_mean,
            state_std=state_std,
            with_perturbations=False,
            device=device,
        )
        res_s["name"] = name
        results_std.append(res_s)

        logger.info(f"Running Perturbation Benchmark on: {name}...")
        res_r = evaluate_policy(
            model=model,
            model_type=mtype,
            env_name=cfg.env.env_name,
            num_episodes=args.episodes,
            context_length=cfg.training.context_length,
            state_mean=state_mean,
            state_std=state_std,
            with_perturbations=True,
            device=device,
        )
        res_r["name"] = name
        results_rob.append(res_r)

    # Print Publication Comparative Tables
    print("\n" + "-" * 115)
    print(
        f"{'Architecture / Paradigm':<44} | {'Params':<9} | {'Frequency (Hz)':<14} | {'Latency (ms)':<12} | {'Jerk (Smooth)':<14} | {'D4RL Score':<10}"
    )
    print("-" * 115)
    for res in results_std:
        print(
            f"{res['name']:<44} | {res['params_count']:<9,d} | {res['frequency_hz']:<14.1f} | {res['mean_latency_ms']:<12.3f} | {res['mean_smoothness_jerk']:<14.4f} | {res['d4rl_normalized_score']:<10.2f}"
        )
    print("-" * 115)

    print("\n" + "-" * 115)
    print(f"PERTURBATION ROBUSTNESS (Random Force Impulses & Continuous Sensor Noise)")
    print("-" * 115)
    print(
        f"{'Architecture / Paradigm':<44} | {'Raw Return':<20} | {'D4RL Score':<12} | {'Jerk (Smooth)':<14} | {'Survival %':<10}"
    )
    print("-" * 115)
    for res in results_rob:
        ret_str = f"{res['mean_return']:.2f} +/- {res['std_return']:.2f}"
        print(
            f"{res['name']:<44} | {ret_str:<20} | {res['d4rl_normalized_score']:<12.2f} | {res['mean_smoothness_jerk']:<14.4f} | {res['survival_rate']:<10.1f}%"
        )
    print("-" * 115 + "\n")


if __name__ == "__main__":
    main()
