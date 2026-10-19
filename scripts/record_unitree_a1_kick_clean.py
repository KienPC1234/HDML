#!/usr/bin/env python3
"""Record high-fidelity clean video of Unitree A1 lateral kick recovery under HDML.

No artificial HUD, no fake overlays: pure, verified physics rollout in MuJoCo.
Matches the exact camera, physics parameters, and frames published in the report
and poster (force=200N, kick_step=200, seed=42).

Usage:
    .venv/bin/python scripts/record_unitree_a1_kick_clean.py
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import logging
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch

from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.evaluation.rollout import load_hdml_checkpoint, torso_roll
from hdml.utils.config import HDMLConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RecordKickClean")


def record_kick_video(
    cfg_path: str = "configs/unitree_a1_aug.yaml",
    checkpoint_path: str = "checkpoints/rebuild_unitree/unitree_a1/best_model.pt",
    output_video: str = "videos/unitree_a1_robot_dog_kick_recovery.mp4",
    force: float = 200.0,
    kick_step: int = 200,
    seed: int = 42,
    max_steps: int = 400,
    fps: int = 50,
) -> None:
    cfg = HDMLConfig.from_yaml(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model, sm, ss = load_hdml_checkpoint(checkpoint_path, cfg, device)

    env = QuadrupedDogEnv(max_episode_steps=max_steps, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)

    ctx = cfg.training.context_length
    adim = cfg.model.action_dim
    hs: list[np.ndarray] = []
    ha: list[np.ndarray] = []
    hr: list[float] = []
    ht: list[int] = []

    rtg = float(cfg.env.target_return)
    frames: list[np.ndarray] = []

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.60
    cam.elevation = -22.0
    cam.azimuth = 135.0

    logger.info("Starting rollout: force=%.1f, kick_step=%d, max_steps=%d", force, kick_step, max_steps)

    for t in range(max_steps):
        # 1. Update camera to follow robot center of mass
        cam.lookat[:] = env.data.subtree_com[0]
        cam.lookat[2] += 0.06

        frame = env.render(camera=cam)
        if frame is not None:
            frames.append(frame)

        # 2. HDML Policy inference
        n = (np.asarray(obs, np.float32) - sm) / ss
        hs.append(n)
        hr.append(rtg / cfg.env.scale_return)
        ht.append(t)
        if not ha:
            ha.append(np.zeros(adim, np.float32))

        cl = min(len(hs), ctx)
        ts = torch.from_numpy(np.array(hs[-cl:], np.float32)).unsqueeze(0).to(device)
        ta = torch.from_numpy(np.array(ha[-cl:], np.float32)).unsqueeze(0).to(device)
        tr = torch.from_numpy(np.array(hr[-cl:], np.float32).reshape(-1, 1)).unsqueeze(0).to(device)
        tt = torch.from_numpy(np.array(ht[-cl:], np.int64)).unsqueeze(0).to(device)

        with torch.inference_mode():
            out = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)

        if isinstance(out, tuple):
            out = out[0]
        a = out.squeeze(0).cpu().numpy().astype(np.float32)
        if a.ndim > 1:
            a = a[-1]
        a = np.clip(a, -1.0, 1.0)

        # 3. Apply 3-step physical kick impulse
        if kick_step <= t < kick_step + 3:
            env.apply_kick((0.0, float(force), 0.0))

        obs, r, term, trunc, _ = env.step(a)
        ha.append(a)
        rtg -= float(r)

        roll = abs(torso_roll(obs))
        if t % 50 == 0 or t in (185, 200, 208, 218, 300):
            logger.info("Step %3d | Roll: %5.1f deg | RTG: %6.1f | Frames: %d", t, roll, rtg, len(frames))

        if term or trunc:
            logger.warning("Episode ended early at step %d (term=%s, trunc=%s)", t, term, trunc)
            break

    env.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("Rollout finished with %d frames. Writing MP4 to %s...", len(frames), output_video)
    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg writer with H.264 yuv420p for universal playback
    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=9,
        pixelformat="yuv420p",
    )
    for f in frames:
        writer.append_data(f)
    writer.close()

    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.info("Successfully saved clean video: %s (%.2f MB, %d frames @ %d FPS)", out_path, size_mb, len(frames), fps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/unitree_a1_aug.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/rebuild_unitree/unitree_a1/best_model.pt")
    ap.add_argument("--output", default="videos/unitree_a1_robot_dog_kick_recovery.mp4")
    ap.add_argument("--force", type=float, default=200.0)
    ap.add_argument("--kick-step", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--fps", type=int, default=50)
    args = ap.parse_args()

    record_kick_video(
        cfg_path=args.config,
        checkpoint_path=args.checkpoint,
        output_video=args.output,
        force=args.force,
        kick_step=args.kick_step,
        seed=args.seed,
        max_steps=args.steps,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()

