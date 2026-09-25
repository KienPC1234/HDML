"""AntBot Multi-View 3D Quadruped Maze Navigation Video Recorder with Clean Scientific HUD.

Features:
- Multi-Camera Layout:
    * Top Full Pane: Global Labyrinth Overview
    * Bottom-Left: Close-up Gait & Joint Tracking
    * Bottom-Right: Dynamic 3D Isometric Perspective
- Minimalist Scientific HUD (Zero clutter, elegant typography, clear telemetry)
"""

from __future__ import annotations

import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
import logging
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import HDMLConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def render_minimalist_scientific_hud(
    canvas: np.ndarray,
    step: int,
    total_steps: int,
    pos: np.ndarray,
    goal: np.ndarray,
    dist: float,
    jerk: float,
    goal_reached: bool,
) -> np.ndarray:
    """Overlays clean, minimalist scientific HUD on the multi-view canvas."""
    frame = canvas.copy()
    h, w, _ = frame.shape

    # 1. Top & Bottom Glassmorphic Banner Bars
    overlay = frame.copy()
    header_h = 42
    footer_h = 32
    cv2.rectangle(overlay, (0, 0), (w, header_h), (12, 16, 24), -1)
    cv2.rectangle(overlay, (0, h - footer_h), (w, h), (12, 16, 24), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Accent divider lines
    cv2.line(frame, (0, header_h), (w, header_h), (0, 200, 255), 1)
    cv2.line(frame, (0, h - footer_h), (w, h - footer_h), (40, 60, 80), 1)
    cv2.line(frame, (w // 2, 448), (w // 2, h - footer_h), (0, 200, 255), 1)

    # Header Left Title
    cv2.putText(
        frame,
        "HDML: AntBot Hierarchical Navigation (Zero-Label Pre-training)",
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Header Right Status Badge
    status_str = "TARGET REACHED" if goal_reached else "NAVIGATING"
    status_col = (0, 255, 120) if goal_reached else (0, 220, 255)
    badge_text = f"Step: {step:03d} | Dist: {dist:.2f}m | {status_str}"
    cv2.putText(frame, badge_text, (w - 320, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_col, 1, cv2.LINE_AA)

    # Subtle view label pills
    def draw_view_pill(img: np.ndarray, text: str, px: int, py: int) -> None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
        pill_overlay = img.copy()
        cv2.rectangle(pill_overlay, (px, py - 16), (px + tw + 12, py + 6), (8, 12, 18), -1)
        cv2.addWeighted(pill_overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (px, py - 16), (px + tw + 12, py + 6), (0, 180, 255), 1)
        cv2.putText(img, text, (px + 6, py - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 240, 255), 1, cv2.LINE_AA)

    draw_view_pill(frame, "GLOBAL LABYRINTH OVERVIEW", 15, 68)
    draw_view_pill(frame, "CLOSE-UP GAIT TRACKING", 15, 472)
    draw_view_pill(frame, "ISOMETRIC PERSPECTIVE", (w // 2) + 15, 472)

    # Footer Information
    footer_text = f"Position: [{pos[0]:+.2f}, {pos[1]:+.2f}]  -->  Target Goal: [{goal[0]:+.2f}, {goal[1]:+.2f}]  |  Smoothness Jerk: {jerk:.4f}"
    cv2.putText(frame, footer_text, (15, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 210, 230), 1, cv2.LINE_AA)

    # Big Centered Goal Banner if completed
    if goal_reached:
        rw, rh = 360, 48
        rx, ry = (w - rw) // 2, 200
        banner_overlay = frame.copy()
        cv2.rectangle(banner_overlay, (rx, ry), (rx + rw, ry + rh), (0, 150, 70), -1)
        cv2.addWeighted(banner_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (150, 255, 180), 2)
        cv2.putText(
            frame,
            "MISSION COMPLETE: GOAL REACHED!",
            (rx + 18, ry + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def record_antmaze_multiview_solver(
    config_path: str = "configs/antmaze_medium_unsupervised.yaml",
    checkpoint_path: str = "checkpoints/antmaze_medium/best_model.pt",
    dataset_name: str = "D4RL/antmaze/medium-play-v2",
    output_gif: str = "videos/antmaze_medium_multiview_solved.gif",
    output_mp4: str = "videos/antmaze_medium_multiview_solved.mp4",
    cam_distance: float = 27.0,
    max_steps: int = 420,
    seed: int = 31,
    device: str = "cuda",
) -> dict[str, float]:
    """Execute closed-loop AntMaze navigation and record tri-camera multi-view video."""
    import minari

    dev = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    cfg = HDMLConfig.from_yaml(config_path)

    logger.info(f"Loading environment from Minari AntMaze dataset: {dataset_name}...")
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

    import mujoco

    renderer = env.unwrapped.ant_env.mujoco_renderer
    renderer.camera_id = -1
    viewer = renderer._get_viewer(render_mode="rgb_array")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = -1

    obs_dict, _ = env.reset(seed=seed)
    raw_prop = np.asarray(obs_dict["observation"], dtype=np.float32)
    pos = np.asarray(obs_dict["achieved_goal"], dtype=np.float32)
    goal = np.asarray(obs_dict["desired_goal"], dtype=np.float32)

    logger.info(f"Starting closed-loop AntMaze multi-view rollout (Start: {pos} -> Goal: {goal})...")

    history_states: list[np.ndarray] = []
    history_actions: list[np.ndarray] = []
    history_rtgs: list[float] = []
    history_timesteps: list[int] = []

    frames: list[np.ndarray] = []
    action_history: list[np.ndarray] = []
    reached_goal = False
    cfc_hx: torch.Tensor | None = None
    step_reached = -1

    for t in range(max_steps):
        # 1. Render View 1: Top Panoramic Overview (Distance 28m, straight down from sky)
        renderer.camera_id = -1
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.fixedcamid = -1
        viewer.cam.distance = cam_distance
        viewer.cam.elevation = -89.0
        viewer.cam.azimuth = 90.0
        viewer.cam.lookat[0] = 0.0
        viewer.cam.lookat[1] = 0.0
        viewer.cam.lookat[2] = 0.0
        frame_top = env.render()

        # 2. Render View 2: Close-Up Gait Tracking (Distance 4.5m follow ant)
        viewer.cam.distance = 4.5
        viewer.cam.elevation = -30.0
        viewer.cam.azimuth = 90.0
        viewer.cam.lookat[0] = float(pos[0])
        viewer.cam.lookat[1] = float(pos[1])
        viewer.cam.lookat[2] = 0.75
        frame_close = env.render()

        # 3. Render View 3: Dynamic Isometric Perspective (Distance 8.0m angled)
        viewer.cam.distance = 8.0
        viewer.cam.elevation = -45.0
        viewer.cam.azimuth = 45.0
        viewer.cam.lookat[0] = float(pos[0])
        viewer.cam.lookat[1] = float(pos[1])
        viewer.cam.lookat[2] = 0.50
        frame_iso = env.render()

        if frame_top is not None and frame_close is not None and frame_iso is not None:
            # Compose Tri-Camera Multi-View Canvas (800 x 768)
            top_resized = cv2.resize(frame_top, (800, 448), interpolation=cv2.INTER_AREA)
            close_resized = cv2.resize(frame_close, (400, 320), interpolation=cv2.INTER_AREA)
            iso_resized = cv2.resize(frame_iso, (400, 320), interpolation=cv2.INTER_AREA)

            bottom_combined = np.hstack([close_resized, iso_resized])
            canvas_rgb = np.vstack([top_resized, bottom_combined])
            canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)

            d_curr = float(np.linalg.norm(pos - goal))
            if len(action_history) > 2:
                cur_jerk = float(np.mean(np.abs(np.diff(np.array(action_history), n=2, axis=0))))
            else:
                cur_jerk = 0.0

            hud_bgr = render_minimalist_scientific_hud(
                canvas=canvas_bgr,
                step=t,
                total_steps=max_steps,
                pos=pos,
                goal=goal,
                dist=d_curr,
                jerk=cur_jerk,
                goal_reached=reached_goal,
            )
            frames.append(cv2.cvtColor(hud_bgr, cv2.COLOR_BGR2RGB))

        # Check goal reaching (tolerance = 1.0m for AntBot)
        d = np.linalg.norm(pos - goal)
        if d < 1.0 and not reached_goal:
            reached_goal = True
            step_reached = t
            logger.info(f">>> GOAL REACHED at step {t}! Final dist: {d:.3f}m")

        if reached_goal and (t - step_reached >= 25):
            break

        # Form 31-dimensional combined state
        combined = np.concatenate([pos, raw_prop, goal], axis=-1)
        norm_obs = (combined - state_mean) / state_std
        history_states.append(norm_obs)
        history_rtgs.append(0.0)
        history_timesteps.append(t)
        if len(history_actions) == 0:
            history_actions.append(np.zeros(8, dtype=np.float32))

        # Sliding context window (K=25)
        ctx_len = min(len(history_states), 25)
        inp_states = np.array(history_states[-ctx_len:], dtype=np.float32)
        inp_actions = np.array(history_actions[-ctx_len:], dtype=np.float32)
        inp_rtgs = np.array(history_rtgs[-ctx_len:], dtype=np.float32).reshape(-1, 1)
        inp_time = np.array(history_timesteps[-ctx_len:], dtype=np.int64)

        if ctx_len < 25:
            pad = 25 - ctx_len
            inp_states = np.vstack([np.zeros((pad, 31), dtype=np.float32), inp_states])
            inp_actions = np.vstack([np.zeros((pad, 8), dtype=np.float32), inp_actions])
            inp_rtgs = np.vstack([np.zeros((pad, 1), dtype=np.float32), inp_rtgs])
            inp_time = np.concatenate([np.zeros((pad,), dtype=np.int64), inp_time])

        t_states = torch.from_numpy(inp_states).unsqueeze(0).to(dev)
        t_actions = torch.from_numpy(inp_actions).unsqueeze(0).to(dev)
        t_rtgs = torch.from_numpy(inp_rtgs).unsqueeze(0).to(dev)
        t_time = torch.from_numpy(inp_time).unsqueeze(0).to(dev)

        with torch.inference_mode():
            act, cfc_hx, _ = model.get_action(t_states, t_rtgs, t_actions, t_time, hx=cfc_hx)

        action = act[0, 0].cpu().numpy() if act.ndim == 3 else (act[0].cpu().numpy() if act.ndim == 2 else act.cpu().numpy())
        action = np.clip(action, -1.0, 1.0)
        history_actions.append(action)
        action_history.append(action)

        next_obs, _, term, trunc, _ = env.step(action)
        raw_prop = np.asarray(next_obs["observation"], dtype=np.float32)
        pos = np.asarray(next_obs["achieved_goal"], dtype=np.float32)
        if term or trunc:
            break

    env.close()

    # Compute Smoothness (Jerk = mean |d^2 a|)
    if len(action_history) > 2:
        acts = np.array(action_history)
        jerk = float(np.mean(np.abs(np.diff(acts, n=2, axis=0))))
    else:
        jerk = 0.0

    Path(output_gif).parent.mkdir(parents=True, exist_ok=True)
    Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)

    # Save GIF & MP4
    fps = 30
    subsample = 2
    gif_frames = frames[::subsample]
    imageio.mimsave(output_gif, gif_frames, fps=fps // subsample, loop=0)
    imageio.mimsave(output_mp4, frames, fps=fps)

    logger.info(
        f"Multi-View AntMaze recording complete! Saved to {output_gif} and {output_mp4} ({len(frames)} frames). "
        f"Goal reached: {reached_goal} | Steps: {len(frames)} | Jerk: {jerk:.4f}"
    )

    return {
        "reached": 1.0 if reached_goal else 0.0,
        "steps": float(len(frames)),
        "jerk": jerk,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record full Multi-View AntMaze quadruped solver video.")
    parser.add_argument("--config", type=str, default="configs/antmaze_medium_unsupervised.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/antmaze_medium/best_model.pt")
    parser.add_argument("--dataset", type=str, default="D4RL/antmaze/medium-play-v2")
    parser.add_argument("--output-gif", type=str, default="videos/antmaze_medium_multiview_solved.gif")
    parser.add_argument("--output-mp4", type=str, default="videos/antmaze_medium_multiview_solved.mp4")
    parser.add_argument("--cam-distance", type=float, default=27.0, help="Overview camera distance")
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    record_antmaze_multiview_solver(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset,
        output_gif=args.output_gif,
        output_mp4=args.output_mp4,
        cam_distance=args.cam_distance,
        max_steps=args.steps,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
