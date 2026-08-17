#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
import gymnasium as gym
import imageio
import numpy as np
import torch

# Enable hardware-accelerated headless EGL rendering for NVIDIA GPUs
os.environ["MUJOCO_GL"] = "egl"

from hdml.models import HDMLModel, DecisionTransformerBaseline
from hdml.utils.config import HDMLConfig
from hdml.evaluation.perturbations import SensorNoisePerturbation, ForceImpulsePerturbation
from hdml.utils.metrics import compute_action_smoothness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def record_episode_video(
    model: torch.nn.Module,
    env_name: str = "Ant-v4",
    checkpoint_path: str | None = None,
    output_path: str = "videos/hdml_ant_v4.gif",
    max_steps: int = 500,
    macro_interval: int = 5,
    with_perturbations: bool = False,
    device: str = "cuda",
) -> dict[str, float]:
    """Run simulation rollout and record video animation frames."""
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(dev)
    model.eval()

    state_mean = None
    state_std = None

    if checkpoint_path and Path(checkpoint_path).exists():
        logger.info(f"Loading weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        state_mean = ckpt.get("state_mean")
        state_std = ckpt.get("state_std")

    env = gym.make(env_name, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]  # type: ignore
    act_dim = env.action_space.shape[0]        # type: ignore

    st_mean = state_mean if state_mean is not None else np.zeros(obs_dim, dtype=np.float32)
    st_std = state_std if state_std is not None else np.ones(obs_dim, dtype=np.float32)

    sensor_noise = SensorNoisePerturbation(noise_std=0.05, seed=42) if with_perturbations else None
    force_perturb = ForceImpulsePerturbation(impulse_prob=0.05, force_magnitude=0.6, seed=42) if with_perturbations else None

    obs, _ = env.reset(seed=42)
    frames: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    rewards_list: list[float] = []

    history_states: list[np.ndarray] = []
    history_actions: list[np.ndarray] = []
    history_rtgs: list[float] = []
    history_timesteps: list[int] = []
    target_rtg = 4000.0
    scale_return = 1000.0
    context_length = 20
    cfc_hx = None
    current_subgoal = None
    prev_action = np.zeros(act_dim, dtype=np.float32)

    logger.info(f"Recording video on {env_name} (Max Steps={max_steps}, Macro Interval={macro_interval})...")

    for t in range(max_steps):
        # Render frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        raw_obs = np.asarray(obs, dtype=np.float32)
        if sensor_noise is not None:
            raw_obs = sensor_noise.apply(raw_obs)

        norm_obs = (raw_obs - st_mean) / st_std
        scaled_rtg = target_rtg / scale_return

        history_states.append(norm_obs)
        history_rtgs.append(scaled_rtg)
        history_timesteps.append(t)
        history_actions.append(prev_action.copy())

        # Two-tier execution
        if t % macro_interval == 0 or current_subgoal is None:
            ctx_len = min(len(history_states), context_length)
            ctx_states = np.array(history_states[-ctx_len:], dtype=np.float32)
            ctx_actions = np.array(history_actions[-ctx_len:], dtype=np.float32)
            ctx_rtgs = np.array(history_rtgs[-ctx_len:], dtype=np.float32).reshape(-1, 1)
            ctx_time = np.array(history_timesteps[-ctx_len:], dtype=np.int64)

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

            t_states = torch.from_numpy(inp_states).unsqueeze(0).to(dev)
            t_actions = torch.from_numpy(inp_actions).unsqueeze(0).to(dev)
            t_rtgs = torch.from_numpy(inp_rtgs).unsqueeze(0).to(dev)
            t_time = torch.from_numpy(inp_time).unsqueeze(0).to(dev)

            with torch.inference_mode():
                action_tensor, cfc_hx, current_subgoal = model.get_action(
                    states=t_states, rtgs=t_rtgs, actions=t_actions, timesteps=t_time, hx=cfc_hx
                )
        else:
            t_prop = torch.from_numpy(norm_obs).unsqueeze(0).to(dev)
            with torch.inference_mode():
                action_tensor, cfc_hx = model.liquid_head(
                    subgoals=current_subgoal, current_prop=t_prop, hx=cfc_hx
                )

        action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        action = np.clip(action, -1.0, 1.0)
        prev_action = action.copy()
        history_actions[-1] = action

        exec_action = action
        if force_perturb is not None:
            exec_action = force_perturb.apply_action_perturbation(action)

        next_obs, reward, terminated, truncated, _ = env.step(exec_action)
        actions_list.append(action)
        rewards_list.append(float(reward))
        target_rtg -= float(reward)
        obs = next_obs

        if terminated or truncated:
            break

    env.close()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save animation (GIF or MP4)
    if out_path.suffix.lower() == ".gif":
        imageio.mimsave(out_path, frames[::2], fps=25, loop=0)
    else:
        imageio.mimsave(out_path, frames, fps=30)

    actions_arr = np.array(actions_list, dtype=np.float32)
    jerk = compute_action_smoothness(actions_arr)
    total_return = float(sum(rewards_list))

    logger.info(
        f"Video saved to: {out_path} ({len(frames)} frames). "
        f"Return: {total_return:.2f} | Steps: {len(rewards_list)} | Jerk: {jerk:.4f}"
    )

    return {
        "return": total_return,
        "steps": float(len(rewards_list)),
        "jerk": jerk,
        "frames_count": float(len(frames)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and record MuJoCo simulation video animation.")
    parser.add_argument("--config", type=str, default="configs/ant_v4_default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ant_v4/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="videos/hdml_ant_v4_rollout.gif", help="Output video or GIF path")
    parser.add_argument("--steps", type=int, default=300, help="Number of steps to record")
    parser.add_argument("--macro-interval", type=int, default=5, help="Macro-planning interval")
    parser.add_argument("--perturbations", action="store_true", default=False, help="Enable physical force perturbations")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    args = parser.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    model = HDMLModel.from_config(cfg.model)

    record_episode_video(
        model=model,
        env_name=cfg.env.env_name,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_steps=args.steps,
        macro_interval=args.macro_interval,
        with_perturbations=args.perturbations,
        device=args.device,
    )


if __name__ == "__main__":
    main()
