#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
import logging
import math
from pathlib import Path
import cv2
import imageio
import numpy as np
import torch

from hdml.evaluation.unitree_a1_maze_env import UnitreeA1MazeEnv
from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import HDMLConfig
from hdml.utils.metrics import compute_action_smoothness

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RecordUnitreeA1Maze")


def draw_lidar_radar(
    canvas: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    lidar_ranges: np.ndarray,
    rel_goal: np.ndarray,
    max_range: float = 4.0,
) -> None:
    """Draw high-contrast 16-beam circular radar screen displaying onboard LiDAR range & goal direction."""
    # Background glass circle
    cv2.circle(canvas, (center_x, center_y), radius, (18, 24, 34), -1)
    cv2.circle(canvas, (center_x, center_y), radius, (0, 220, 255), 2)
    cv2.circle(canvas, (center_x, center_y), int(radius * 0.75), (35, 55, 75), 1)
    cv2.circle(canvas, (center_x, center_y), int(radius * 0.50), (35, 55, 75), 1)
    cv2.circle(canvas, (center_x, center_y), int(radius * 0.25), (35, 55, 75), 1)

    # Crosshairs
    cv2.line(canvas, (center_x - radius, center_y), (center_x + radius, center_y), (45, 65, 85), 1)
    cv2.line(canvas, (center_x, center_y - radius), (center_x, center_y + radius), (45, 65, 85), 1)

    # 16-beam LiDAR points
    num_rays = len(lidar_ranges)
    angles = np.linspace(-math.pi, math.pi, num_rays, endpoint=False)

    for i in range(num_rays):
        dist = float(lidar_ranges[i])
        norm_r = min(1.0, dist / max_range) * (radius - 6)
        # In robot frame: index 8 is forward => map to UP (-Y)
        ang = angles[i] - math.pi / 2.0
        px = int(center_x + norm_r * math.cos(ang))
        py = int(center_y + norm_r * math.sin(ang))

        # Color coding: Red (<0.8m obstacle), Yellow (0.8-1.8m), Green (>1.8m clear)
        if dist < 0.8:
            col = (40, 40, 255)
            pt_r = 4
        elif dist < 1.8:
            col = (0, 215, 255)
            pt_r = 3
        else:
            col = (80, 255, 120)
            pt_r = 3

        cv2.line(canvas, (center_x, center_y), (px, py), (40, 50, 65), 1)
        cv2.circle(canvas, (px, py), pt_r, col, -1)

    # Relative Goal Vector Arrow
    dx, dy = float(rel_goal[0]), float(rel_goal[1])
    goal_dist = math.sqrt(dx**2 + dy**2)
    if goal_dist > 1e-2:
        g_ang = math.atan2(dy, dx) - math.pi / 2.0
        g_len = min(radius - 4, 38)
        gx = int(center_x + g_len * math.cos(g_ang))
        gy = int(center_y + g_len * math.sin(g_ang))
        cv2.arrowedLine(canvas, (center_x, center_y), (gx, gy), (0, 255, 120), 3, tipLength=0.35)

    # Robot Center Marker (Vibrant Orange Core)
    cv2.circle(canvas, (center_x, center_y), 5, (0, 140, 255), -1)
    cv2.putText(canvas, "16-BEAM ONBOARD LIDAR", (center_x - 68, center_y + radius + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)


def render_split_view_frame(
    follow_img: np.ndarray,
    overview_img: np.ndarray,
    step: int,
    pos: np.ndarray,
    goal: np.ndarray,
    dist: float,
    jerk: float,
    lidar: np.ndarray,
    rel_goal: np.ndarray,
    goal_reached: bool,
    path_history: list[np.ndarray],
) -> np.ndarray:
    """Combines 3D follow perspective + top-down radar and HUD into a 1280x720 split video."""
    h_out, w_out = 720, 1280
    canvas = np.zeros((h_out, w_out, 3), dtype=np.uint8)

    # 1. Left pane: 3D third-person follow view (65% width = 830px)
    w_left = 830
    follow_resized = cv2.resize(follow_img, (w_left, h_out))
    canvas[:, :w_left] = follow_resized

    # 2. Right Top: Overview top-down map (360px height)
    w_right = w_out - w_left
    h_top_right = 360
    overview_resized = cv2.resize(overview_img, (w_right, h_top_right))

    # Draw trajectory path line on overview map
    # Maze bounds: [-4.0, 4.0] x [-4.0, 4.0]
    if len(path_history) > 1:
        pts = []
        for p in path_history:
            # Map [-4.0, 4.0] to [0, w_right] and [h_top_right, 0]
            px = int(w_left + ((p[0] + 4.0) / 8.0) * w_right)
            py = int(((4.0 - p[1]) / 8.0) * h_top_right)
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], (0, 255, 255), 2, cv2.LINE_AA)

    canvas[:h_top_right, w_left:] = overview_resized

    # 3. Right Bottom: Modern Cyberpunk Telemetry Box (360px height)
    panel_y0 = h_top_right
    canvas[panel_y0:, w_left:] = (14, 18, 26)

    # Boundary framing lines
    cv2.line(canvas, (w_left, 0), (w_left, h_out), (0, 220, 255), 2)
    cv2.line(canvas, (w_left, panel_y0), (w_out, panel_y0), (0, 220, 255), 2)

    # Draw 16-beam Radar in the center of the telemetry panel
    draw_lidar_radar(
        canvas=canvas,
        center_x=w_left + w_right // 2,
        center_y=panel_y0 + 105,
        radius=72,
        lidar_ranges=lidar,
        rel_goal=rel_goal,
        max_range=4.0,
    )

    # Telemetry Text Card
    text_x = w_left + 24
    ty = panel_y0 + 215

    # Card background
    cv2.rectangle(canvas, (text_x - 10, ty - 18), (w_out - 14, h_out - 14), (20, 28, 40), -1)
    cv2.rectangle(canvas, (text_x - 10, ty - 18), (w_out - 14, h_out - 14), (45, 75, 105), 1)

    cv2.putText(canvas, f"PROGRESS: Step {step:03d} / 500", (text_x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 240, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"TARGET DIST: {dist:.2f} m", (text_x, ty + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"ROBOT POSE: [{pos[0]:+.2f}, {pos[1]:+.2f}]", (text_x, ty + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 210, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"MOTOR JERK: {jerk:.4f} (Smooth)", (text_x, ty + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 210, 240), 1, cv2.LINE_AA)

    status_str = "[GOAL REACHED - SUCCESS]" if goal_reached else "[AUTONOMOUS CORRIDOR PURSUIT]"
    status_col = (0, 255, 120) if goal_reached else (0, 220, 255)
    cv2.putText(canvas, status_str, (text_x, ty + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_col, 2, cv2.LINE_AA)

    # Top Glass Header Banner
    header_overlay = canvas.copy()
    cv2.rectangle(header_overlay, (0, 0), (w_out, 42), (8, 12, 18), -1)
    cv2.addWeighted(header_overlay, 0.88, canvas, 0.12, 0, canvas)
    cv2.line(canvas, (0, 42), (w_out, 42), (0, 200, 255), 2)

    cv2.putText(
        canvas,
        "HDML: Unitree A1 Blind Quadruped Labyrinth Navigation (53D Onboard Sensory State)",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(canvas, "3D ISOMETRIC TRACKING VIEW", (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "GLOBAL MAP BEACON", (w_left + 16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

    if goal_reached:
        rw, rh = 440, 60
        rx, ry = (w_left - rw) // 2, 310
        banner_overlay = canvas.copy()
        cv2.rectangle(banner_overlay, (rx, ry), (rx + rw, ry + rh), (0, 150, 60), -1)
        cv2.addWeighted(banner_overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (0, 255, 120), 2)
        cv2.putText(canvas, "MAZE SOLVED (100% BLIND NAV)", (rx + 28, ry + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def record_maze_navigation(
    config_path: str = "configs/unitree_a1_maze_unsupervised.yaml",
    checkpoint_path: str = "checkpoints/unitree_a1_maze/best_model.pt",
    dataset_path: str = "data/unitree_a1_maze_trajectories.npz",
    output_mp4: str = "videos/unitree_a1_maze_hdml_solved.mp4",
    output_gif: str = "videos/unitree_a1_maze_hdml_solved.gif",
    max_steps: int = 500,
    seed: int = 42,
    device: str = "cuda",
) -> dict[str, float]:
    """Runs closed-loop Unitree A1 maze rollout and records high-fidelity split-view telemetry video."""
    cfg = HDMLConfig.from_yaml(config_path)
    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

    model = HDMLModel.from_config(cfg.model).to(dev)
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded trained HDML checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint {checkpoint_path} not found. Running with initialized policy.")

    model.eval()

    if Path(dataset_path).exists():
        data = np.load(dataset_path)
        st_mean = data["state_mean"]
        st_std = data["state_std"]
    else:
        st_mean = np.zeros(cfg.model.prop_dim, dtype=np.float32)
        st_std = np.ones(cfg.model.prop_dim, dtype=np.float32)

    env = UnitreeA1MazeEnv(render_mode="rgb_array", max_episode_steps=max_steps, goal_pos=(2.8, 2.8), start_pos=(-2.8, -2.8))
    obs, info = env.reset(seed=seed)

    history_states: list[np.ndarray] = []
    history_actions: list[np.ndarray] = []
    history_rtgs: list[float] = []
    history_timesteps: list[int] = []
    path_history: list[np.ndarray] = [info["pos"].copy()]

    actions_list: list[np.ndarray] = []
    frames: list[np.ndarray] = []

    current_rtg = 0.5
    goal_reached = False
    hx = None

    logger.info(f"Recording closed-loop Unitree A1 navigation from {env.start_pos} to {env.goal_pos}...")

    for t in range(max_steps):
        norm_obs = (obs - st_mean) / st_std
        scaled_rtg = current_rtg / cfg.env.scale_return

        history_states.append(norm_obs)
        history_rtgs.append(scaled_rtg)
        history_timesteps.append(t)
        if len(history_actions) == 0:
            history_actions.append(np.zeros(cfg.model.action_dim, dtype=np.float32))

        ctx_len = min(len(history_states), cfg.training.context_length)
        ctx_states = np.array(history_states[-ctx_len:], dtype=np.float32)
        ctx_actions = np.array(history_actions[-ctx_len:], dtype=np.float32)
        ctx_rtgs = np.array(history_rtgs[-ctx_len:], dtype=np.float32).reshape(-1, 1)
        ctx_time = np.array(history_timesteps[-ctx_len:], dtype=np.int64)

        t_states = torch.from_numpy(ctx_states).unsqueeze(0).to(dev)
        t_actions = torch.from_numpy(ctx_actions).unsqueeze(0).to(dev)
        t_rtgs = torch.from_numpy(ctx_rtgs).unsqueeze(0).to(dev)
        t_timesteps = torch.from_numpy(ctx_time).unsqueeze(0).to(dev)

        with torch.inference_mode():
            action_tensor, _, _ = model.get_action(
                states=t_states,
                rtgs=t_rtgs,
                actions=t_actions,
                timesteps=t_timesteps,
                hx=None,
            )

        unclipped_action = action_tensor[0].cpu().numpy().astype(np.float32)
        action = np.clip(unclipped_action, -1.0, 1.0)

        # Step physics
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        actions_list.append(action)
        history_actions.append(unclipped_action)
        path_history.append(step_info["pos"].copy())
        # Keep RTG constant, decrementing it causes it to drop below 0 (OOD)
        current_rtg = 0.5

        # Render dual cameras
        follow_img = env.render(camera_name="follow_cam")
        overview_img = env.render(camera_name="overview_cam")

        if follow_img is not None and overview_img is not None:
            lidar_slice = obs[35:51]
            rel_goal_slice = obs[51:53]
            current_jerk = compute_action_smoothness(np.array(actions_list, dtype=np.float32))

            split_frame = render_split_view_frame(
                follow_img=follow_img,
                overview_img=overview_img,
                step=t,
                pos=step_info["pos"],
                goal=step_info["goal"],
                dist=step_info["dist_to_goal"],
                jerk=current_jerk,
                lidar=lidar_slice,
                rel_goal=rel_goal_slice,
                goal_reached=step_info["goal_reached"],
                path_history=path_history,
            )
            frames.append(split_frame)

        obs = next_obs
        if step_info.get("goal_reached", False):
            goal_reached = True
            logger.info(f"Target destination reached successfully at step {t}! Dist: {step_info['dist_to_goal']:.2f}m")
            for _ in range(20):
                frames.append(split_frame)
            break

        if terminated or truncated:
            break

    env.close()

    out_mp4_path = Path(output_mp4)
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    out_gif_path = Path(output_gif)
    out_gif_path.parent.mkdir(parents=True, exist_ok=True)

    if len(frames) > 0:
        imageio.mimsave(out_mp4_path, frames, fps=30)
        imageio.mimsave(out_gif_path, frames[::2], duration=66, loop=0)

    final_jerk = compute_action_smoothness(np.array(actions_list, dtype=np.float32))
    logger.info(f"Saved video to {out_mp4_path} ({len(frames)} frames). Jerk: {final_jerk:.4f} | Solved: {goal_reached}")

    return {
        "goal_reached": float(goal_reached),
        "steps": float(len(actions_list)),
        "jerk": final_jerk,
        "frames": float(len(frames)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Unitree A1 Labyrinth Navigation split-view video.")
    parser.add_argument("--config", type=str, default="configs/unitree_a1_maze_unsupervised.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/unitree_a1_maze/best_model.pt")
    parser.add_argument("--dataset", type=str, default="data/unitree_a1_maze_trajectories.npz")
    parser.add_argument("--output-mp4", type=str, default="videos/unitree_a1_maze_hdml_solved.mp4")
    parser.add_argument("--output-gif", type=str, default="videos/unitree_a1_maze_hdml_solved.gif")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    record_maze_navigation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        output_mp4=args.output_mp4,
        output_gif=args.output_gif,
        max_steps=args.max_steps,
        seed=args.seed,
        device=args.device,
    )
