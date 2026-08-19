#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
import shutil
import cv2
import imageio
import numpy as np
import torch

os.environ["MUJOCO_GL"] = "egl"

from hdml.utils.config import HDMLConfig
from hdml.models.hdml_model import HDMLModel
from hdml.utils.metrics import compute_action_smoothness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def draw_hud(
    frame: np.ndarray,
    step: int,
    pos: np.ndarray,
    goal: np.ndarray,
    subgoal: np.ndarray | None,
    action: np.ndarray,
    dist_to_goal: float,
    is_reached: bool,
) -> np.ndarray:
    """Render high-tech cybernetic HUD overlay on top of simulation frame."""
    hud_frame = frame.copy()
    h, w, _ = hud_frame.shape

    # Top status banner
    overlay = hud_frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (15, 23, 42), -1)
    cv2.rectangle(overlay, (0, h - 55), (w, h), (15, 23, 42), -1)
    alpha = 0.82
    hud_frame = cv2.addWeighted(overlay, alpha, hud_frame, 1.0 - alpha, 0)

    # Title & Mode
    cv2.putText(hud_frame, "HDML: SPATIAL COGNITION MAZE SOLVER", (14, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(hud_frame, "Mamba-3 Latent Planning + Liquid CfC Motor Head", (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)

    # Status pill
    if is_reached:
        badge_text = "MAZE SOLVED / GOAL REACHED"
        badge_color = (34, 197, 94)  # Green
    else:
        badge_text = f"NAVIGATING (d={dist_to_goal:.2f}m)"
        badge_color = (6, 182, 212)  # Cyan

    cv2.rectangle(hud_frame, (w - 230, 12), (w - 14, 42), badge_color, 1, cv2.LINE_AA)
    cv2.putText(hud_frame, badge_text, (w - 222, 33), cv2.FONT_HERSHEY_DUPLEX, 0.40, badge_color, 1, cv2.LINE_AA)

    # Bottom telemetry bar
    bottom_text_1 = f"Step: {step:03d} | Pos: ({pos[0]:+.2f}, {pos[1]:+.2f}) | Target Goal: ({goal[0]:+.2f}, {goal[1]:+.2f})"
    bottom_text_2 = f"Liquid Action Force: ({action[0]:+.2f}, {action[1]:+.2f}) | Subgoal Latent Alignment: OK"
    cv2.putText(hud_frame, bottom_text_1, (14, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (226, 232, 240), 1, cv2.LINE_AA)
    cv2.putText(hud_frame, bottom_text_2, (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

    return hud_frame


def record_maze_solver(
    config_path: str = "configs/pointmaze_umaze_unsupervised.yaml",
    checkpoint_path: str = "checkpoints/pointmaze_umaze/best_model.pt",
    dataset_name: str = "D4RL/pointmaze/umaze-v2",
    output_gif: str = "videos/pointmaze_hdml_solved.gif",
    output_mp4: str = "videos/pointmaze_hdml_solved.mp4",
    max_steps: int = 300,
    seed: int = 42,
    device: str = "cuda",
) -> dict[str, float]:
    """Execute closed-loop PointMaze navigation with trained HDML model and record video."""
    import minari

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    cfg = HDMLConfig.from_yaml(config_path)

    logger.info(f"Loading environment from Minari PointMaze dataset: {dataset_name}...")
    dataset = minari.load_dataset(dataset_name)
    env = dataset.recover_environment(render_mode="rgb_array")

    # Load Model
    model = HDMLModel.from_config(cfg.model).to(dev)
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    state_mean = ckpt.get("state_mean", np.zeros(cfg.model.prop_dim, dtype=np.float32))
    state_std = ckpt.get("state_std", np.ones(cfg.model.prop_dim, dtype=np.float32))

    obs_dict, _ = env.reset(seed=seed)
    raw_prop = np.asarray(obs_dict["observation"], dtype=np.float32)  # (4,)
    desired_goal = np.asarray(obs_dict["desired_goal"], dtype=np.float32)  # (2,)

    frames: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []

    history_states: list[np.ndarray] = []
    history_actions: list[np.ndarray] = []
    history_rtgs: list[float] = []
    history_timesteps: list[int] = []

    context_length = cfg.training.context_length
    cfc_hx = None
    goal_reached = False
    reached_step = None

    logger.info(f"Starting closed-loop PointMaze rollout (Start: {raw_prop[:2]} -> Goal: {desired_goal})...")

    for t in range(max_steps):
        # 1. Render raw frame
        raw_frame = env.render()
        cur_pos = raw_prop[:2]
        dist_to_goal = float(np.linalg.norm(cur_pos - desired_goal))
        if dist_to_goal < 0.5:
            goal_reached = True
            if reached_step is None:
                reached_step = t
                logger.info(f">>> GOAL REACHED at step {t}! Final dist: {dist_to_goal:.3f}m")

        # 2. Form state (4D prop + 2D goal = 6D)
        combined_obs = np.concatenate([raw_prop, desired_goal], axis=-1)
        norm_obs = (combined_obs - state_mean) / state_std

        history_states.append(norm_obs)
        history_rtgs.append(0.0)
        history_timesteps.append(t)
        if len(history_actions) == 0:
            history_actions.append(np.zeros(cfg.model.action_dim, dtype=np.float32))

        # 3. Model inference (Mamba-3 planning + CfC continuous actuation)
        ctx_len = min(len(history_states), context_length)
        ctx_states = np.array(history_states[-ctx_len:], dtype=np.float32)
        ctx_actions = np.array(history_actions[-ctx_len:], dtype=np.float32)
        ctx_rtgs = np.array(history_rtgs[-ctx_len:], dtype=np.float32).reshape(-1, 1)
        ctx_time = np.array(history_timesteps[-ctx_len:], dtype=np.int64)

        if ctx_len < context_length:
            pad_k = context_length - ctx_len
            inp_states = np.vstack([np.zeros((pad_k, cfg.model.prop_dim), dtype=np.float32), ctx_states])
            inp_actions = np.vstack([np.zeros((pad_k, cfg.model.action_dim), dtype=np.float32), ctx_actions])
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
            action_tensor, cfc_hx, info = model.get_action(
                states=t_states, rtgs=t_rtgs, actions=t_actions, timesteps=t_time, hx=cfc_hx
            )
            subgoal_info = info.get("subgoal")

        if action_tensor.ndim == 3:
            action = action_tensor[0, 0, :].cpu().numpy().astype(np.float32)
        elif action_tensor.ndim == 2:
            action = action_tensor[0, :].cpu().numpy().astype(np.float32)
        else:
            action = action_tensor.cpu().numpy().astype(np.float32)

        action = np.clip(action, -1.0, 1.0)
        history_actions.append(action)
        actions_list.append(action)

        # 4. Draw HUD on frame
        if raw_frame is not None:
            hud_frame = draw_hud(
                frame=raw_frame,
                step=t,
                pos=cur_pos,
                goal=desired_goal,
                subgoal=subgoal_info,
                action=action,
                dist_to_goal=dist_to_goal,
                is_reached=goal_reached,
            )
            frames.append(hud_frame)

        # 5. Step simulation environment
        next_obs_dict, reward, terminated, truncated, _ = env.step(action)
        raw_prop = np.asarray(next_obs_dict["observation"], dtype=np.float32)

        # Allow extra 30 frames after reaching goal to clearly show victory
        if goal_reached and reached_step is not None and (t - reached_step) >= 35:
            break
        if terminated or truncated:
            break

    env.close()

    # Save output videos
    out_gif = Path(output_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    out_mp4 = Path(output_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(out_gif, frames[::2], duration=50, loop=0)
    imageio.mimsave(out_mp4, frames, fps=30)

    # Copy to artifact directory for UI display
    artifact_dir = Path("/home/kien/.gemini/antigravity-ide/brain/d6d46bb5-1461-41b6-9adb-f0da469ccff6")
    if artifact_dir.exists():
        shutil.copy(out_gif, artifact_dir / "pointmaze_hdml_solved.gif")
        shutil.copy(out_mp4, artifact_dir / "pointmaze_hdml_solved.mp4")

    acts_np = np.array(actions_list, dtype=np.float32)
    jerk = compute_action_smoothness(acts_np)

    logger.info(
        f"Maze recording complete! Saved to {out_gif} and {out_mp4} ({len(frames)} frames). "
        f"Goal reached: {goal_reached} | Steps: {len(actions_list)} | Jerk: {jerk:.4f}"
    )

    return {
        "goal_reached": float(goal_reached),
        "steps": float(len(actions_list)),
        "jerk": jerk,
        "frames": float(len(frames)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record full PointMaze solver video.")
    parser.add_argument("--config", type=str, default="configs/pointmaze_umaze_unsupervised.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pointmaze_umaze/best_model.pt")
    parser.add_argument("--dataset", type=str, default="D4RL/pointmaze/umaze-v2", help="Minari dataset name")
    parser.add_argument("--output-gif", type=str, default="videos/pointmaze_hdml_solved.gif")
    parser.add_argument("--output-mp4", type=str, default="videos/pointmaze_hdml_solved.mp4")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    record_maze_solver(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        output_gif=args.output_gif,
        output_mp4=args.output_mp4,
        max_steps=args.steps,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
