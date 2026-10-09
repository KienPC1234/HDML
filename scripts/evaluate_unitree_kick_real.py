#!/usr/bin/env python3
"""Honest lateral-kick recovery experiment on the 12-DoF Unitree A1.

Runs the *trained* HDML policy and the *trained* Decision-Transformer baseline
closed-loop in the MuJoCo quadruped. A genuine lateral impulse is applied via
``qfrc_applied`` for three control steps. We record the torso-roll trajectory,
peak roll, survival and recovery time for several force magnitudes and seeds,
and render real frames of the HDML rollout at the key physical phases.

This replaces the earlier figure that used the scripted CPG policy, a fabricated
``qvel`` impulse, a hand-drawn DT baseline curve and an unverified PACE label.

Force units: the MuJoCo ``qfrc_applied`` is a generalised force on the floating
base. Values are reported in the same (uncalibrated) units the CPG collector
used; they are not calibrated newtons.

Usage:
    .venv/bin/python scripts/evaluate_unitree_kick_real.py \
        --forces 50 100 200 400 --episodes 3 \
        --output results/unitree_kick_real.json \
        --frames-out KH_thi_KHKT_TP_2026-2027/Bao_cao_du_an_LaTeX/figures/unitree_kick_frames
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.models import DecisionTransformerBaseline, HDMLModel
from hdml.utils.config import HDMLConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("KickReal")


def torso_roll(obs: np.ndarray) -> float:
    w, x, y, z = float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4])
    return float(np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))))


def load_hdml(path: str, cfg: HDMLConfig, device: torch.device) -> tuple[HDMLModel, np.ndarray, np.ndarray]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = HDMLModel.from_config(cfg.model).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    model.deterministic = True
    model.num_flow_steps = 4
    return model, np.asarray(ck["state_mean"], np.float32), np.asarray(ck["state_std"], np.float32)


def load_dt(path: str, cfg: HDMLConfig, device: torch.device) -> tuple[DecisionTransformerBaseline, np.ndarray, np.ndarray]:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = cfg.model
    model = DecisionTransformerBaseline(
        prop_dim=m.prop_dim, action_dim=m.action_dim, d_model=m.d_model,
        nhead=4, num_layers=m.num_mamba_layers,
    ).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    return model, np.asarray(ck["state_mean"], np.float32), np.asarray(ck["state_std"], np.float32)


def _to_action(out) -> np.ndarray:
    a = out[0] if isinstance(out, tuple) else out
    a = a[0, 0, :] if a.ndim == 3 else a[0, :]
    return np.clip(a.detach().cpu().numpy().astype(np.float32), -1.0, 1.0)


def run_episode(
    model, mtype: str, cfg: HDMLConfig, device: torch.device,
    sm: np.ndarray, ss: np.ndarray, force: float, kick_step: int, seed: int,
    max_steps: int, capture_steps: set[int] | None = None,
) -> dict:
    env = QuadrupedDogEnv(max_episode_steps=max_steps, render_mode="rgb_array" if capture_steps else None)
    obs, _ = env.reset(seed=seed)
    ctx = cfg.training.context_length
    adim = cfg.model.action_dim
    hs: list[np.ndarray] = []
    ha: list[np.ndarray] = []
    hr: list[float] = []
    ht: list[int] = []
    ep_r: list[float] = []
    rolls: list[float] = []
    frames: dict[int, np.ndarray] = {}
    rtg = float(cfg.env.target_return)
    peak = 0.0
    recovered_at = None
    alive = 0
    for t in range(max_steps):
        n = (np.asarray(obs, np.float32) - sm) / ss
        hs.append(n); hr.append(rtg / cfg.env.scale_return); ht.append(t)
        if not ha:
            ha.append(np.zeros(adim, np.float32))
        cl = min(len(hs), ctx)
        ts = torch.from_numpy(np.array(hs[-cl:], np.float32)).unsqueeze(0).to(device)
        ta = torch.from_numpy(np.array(ha[-cl:], np.float32)).unsqueeze(0).to(device)
        tr = torch.from_numpy(np.array(hr[-cl:], np.float32).reshape(-1, 1)).unsqueeze(0).to(device)
        tt = torch.from_numpy(np.array(ht[-cl:], np.int64)).unsqueeze(0).to(device)
        with torch.inference_mode():
            if mtype == "hdml":
                out = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)
            else:
                out = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt)
        a = _to_action(out)
        if kick_step <= t < kick_step + 3:
            env.apply_kick((0.0, float(force), 0.0))
        obs, r, term, trunc, _ = env.step(a)
        ha.append(a)
        ep_r.append(float(r))
        rtg -= float(r)
        roll = abs(torso_roll(obs))
        rolls.append(round(roll, 2))
        peak = max(peak, roll)
        alive = t
        if capture_steps and t in capture_steps:
            frame = env.render()
            if frame is not None:
                frames[t] = np.asarray(frame)
        if t > kick_step and recovered_at is None and roll < 10.0:
            recovered_at = t - kick_step
        if term or trunc:
            break
    env.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    survived = bool(alive >= max_steps - 1)
    return {
        "force": float(force), "seed": int(seed), "steps": alive + 1,
        "peak_roll_deg": round(peak, 1),
        "recovery_steps": recovered_at,
        "recovery_s": (recovered_at * 0.02) if recovered_at is not None else None,
        "return": float(np.sum(ep_r)),
        "completion_pct": 100.0 if survived else round(100.0 * (alive + 1) / max_steps, 1),
        "survived_full_episode": survived,
        "rolls": rolls,
        "_frames": frames,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/unitree_a1_aug.yaml")
    ap.add_argument("--hdml-checkpoint", default="checkpoints/rebuild_unitree/unitree_a1/best_model.pt")
    ap.add_argument("--dt-checkpoint", default="checkpoints/rebuild_unitree/unitree_a1/baselines/dt/best_model.pt")
    ap.add_argument("--forces", type=float, nargs="+", default=[50, 100, 200, 400])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--kick-step", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="results/unitree_kick_real.json")
    ap.add_argument("--frames-out", default="KH_thi_KHKT_TP_2026-2027/Bao_cao_du_an_LaTeX/figures/unitree_kick_frames")
    args = ap.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    hdml, hsm, hss = load_hdml(args.hdml_checkpoint, cfg, device)
    dt, dsm, dss = load_dt(args.dt_checkpoint, cfg, device)

    max_steps = int(cfg.env.max_episode_steps)
    # Frames captured around the kick for the first HDML episode at the lowest force.
    capture = {int(args.kick_step) - 4, int(args.kick_step), int(args.kick_step) + 5, int(args.kick_step) + 25}
    frames_dir = Path(args.frames_out)
    frames_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {"hdml": [], "dt": []}
    for model, mtype, sm, ss in ((hdml, "hdml", hsm, hss), (dt, "dt", dsm, dss)):
        for force in args.forces:
            eps = []
            for i in range(args.episodes):
                do_capture = (mtype == "hdml" and force == args.forces[0] and i == 0)
                ep = run_episode(
                    model, mtype, cfg, device, sm, ss, force,
                    args.kick_step, 42 + i, max_steps,
                    capture_steps=capture if do_capture else None,
                )
                if do_capture:
                    for st, img in ep.pop("_frames", {}).items():
                        from PIL import Image
                        Image.fromarray(img).save(frames_dir / f"hdml_f{int(force)}_t{st:04d}.png")
                else:
                    ep.pop("_frames", None)
                eps.append(ep)
            surv = float(np.mean([e["survived_full_episode"] for e in eps]))
            peak = float(np.mean([e["peak_roll_deg"] for e in eps]))
            recs = [e["recovery_s"] for e in eps if e["recovery_s"] is not None]
            ret = float(np.mean([e["return"] for e in eps]))
            logger.info(
                "%-5s force=%6.1f  peak_roll=%5.1f deg  recovery=%s s  survival=%.0f%%  return=%.1f  steps=%s",
                mtype, force, peak, (f"{np.mean(recs):.3f}" if recs else "n/a"), surv * 100, ret,
                [e["steps"] for e in eps],
            )
            results[mtype].append({
                "force": force, "peak_roll_deg": round(peak, 2),
                "recovery_s": (float(np.mean(recs)) if recs else None),
                "survival_rate": surv, "return_mean": ret, "episodes": eps,
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "kick_step": args.kick_step, "episodes": args.episodes,
        "forces": args.forces, "results": results,
        "note": "qfrc_applied generalised force, uncalibrated units; synchronous macro_interval=1",
    }, indent=2), encoding="utf-8")
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()
