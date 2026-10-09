#!/usr/bin/env python3
"""Does the CfC micro-filter actually help under impact? (Unitree A1)

Runs the trained HDML policy closed-loop and applies the same lateral impulse
with the CfC filter ON vs OFF, at several magnitudes. If the CfC tier is doing
its job as a continuous-time damper, removing it should hurt peak roll and/or
recovery — especially at the impact, where a filter matters. Reports peak roll,
post-impact roll oscillation (RMS of roll after the kick) and return.

Usage:
    .venv/bin/python scripts/benchmark_unitree_cfc_kick.py --forces 50 200 400 \
        --episodes 3 --output results/rebuild_unitree/cfc_kick.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.models import HDMLModel
from hdml.utils.config import HDMLConfig


def torso_roll(obs: np.ndarray) -> float:
    w, x, y, z = float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4])
    return float(np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))))


def load_cfc_contribution(cfg, device, ck, sm, ss, steps=300):
    """Mean |CfC-refined action - nominal action| over a clean rollout."""
    h = HDMLModel.from_config(cfg.model).to(device)
    h.load_state_dict(ck["model_state_dict"]); h.eval()
    h.deterministic = True; h.num_flow_steps = 4
    env = QuadrupedDogEnv(max_episode_steps=steps)
    obs, _ = env.reset(seed=42)
    ctx = cfg.training.context_length
    ad = cfg.model.action_dim
    hs, ha, hr, ht = [], [], [], []
    rtg = float(cfg.env.target_return)
    deltas = []
    for t in range(steps):
        n = (np.asarray(obs, np.float32) - sm) / ss
        hs.append(n); hr.append(rtg / cfg.env.scale_return); ht.append(t)
        if not ha:
            ha.append(np.zeros(ad, np.float32))
        cl = min(len(hs), ctx)
        ts = torch.from_numpy(np.array(hs[-cl:], np.float32)).unsqueeze(0).to(device)
        ta = torch.from_numpy(np.array(ha[-cl:], np.float32)).unsqueeze(0).to(device)
        tr = torch.from_numpy(np.array(hr[-cl:], np.float32).reshape(-1, 1)).unsqueeze(0).to(device)
        tt = torch.from_numpy(np.array(ht[-cl:], np.int64)).unsqueeze(0).to(device)
        with torch.inference_mode():
            _sg, lat, _v, _ns, _fc, executed = h.encode(states=ts, rtgs=tr, actions=ta, timesteps=tt)
            nominal = h._sample_chunk(_fc[:, -1:, :])[:, :, 0, :]  # (B,1,ad)
            nominal = nominal[:, -1, :]
        deltas.append(float((executed[:, -1, :] - nominal).abs().mean().item()))
        a = np.clip(executed[:, -1, :].cpu().numpy().astype(np.float32)[0], -1, 1)
        ha.append(a)
        obs, r, term, trunc, _ = env.step(a)
        rtg -= float(r)
        if term or trunc:
            break
    env.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(np.mean(deltas))


def rollout(model, cfg, sm, ss, force, kick_step, seed, max_steps, device):
    env = QuadrupedDogEnv(max_episode_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    ctx = cfg.training.context_length
    ad = cfg.model.action_dim
    hs, ha, hr, ht = [], [], [], []
    ep_r = []
    rolls = []
    rtg = float(cfg.env.target_return)
    for t in range(max_steps):
        n = (np.asarray(obs, np.float32) - sm) / ss
        hs.append(n); hr.append(rtg / cfg.env.scale_return); ht.append(t)
        if not ha:
            ha.append(np.zeros(ad, np.float32))
        cl = min(len(hs), ctx)
        ts = torch.from_numpy(np.array(hs[-cl:], np.float32)).unsqueeze(0).to(device)
        ta = torch.from_numpy(np.array(ha[-cl:], np.float32)).unsqueeze(0).to(device)
        tr = torch.from_numpy(np.array(hr[-cl:], np.float32).reshape(-1, 1)).unsqueeze(0).to(device)
        tt = torch.from_numpy(np.array(ht[-cl:], np.int64)).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)
        a = out[0] if isinstance(out, tuple) else out
        a = a[0, 0, :] if a.ndim == 3 else a[0, :]
        a = np.clip(a.detach().cpu().numpy().astype(np.float32), -1, 1)
        if kick_step <= t < kick_step + 3:
            env.apply_kick((0.0, float(force), 0.0))
        obs, r, term, trunc, _ = env.step(a)
        ha.append(a); ep_r.append(float(r)); rtg -= float(r)
        rolls.append(abs(torso_roll(obs)))
        if term or trunc:
            break
    env.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rolls = np.asarray(rolls)
    peak = float(rolls.max())
    post = rolls[kick_step:]
    osc = float(np.sqrt(np.mean((post - post.mean()) ** 2))) if len(post) > 1 else 0.0
    return {"force": force, "seed": seed, "peak_roll_deg": round(peak, 2),
            "post_roll_rms_deg": round(osc, 2), "return": round(float(np.sum(ep_r)), 1),
            "steps": len(rolls)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/unitree_a1_aug.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/rebuild_unitree/unitree_a1/best_model.pt")
    ap.add_argument("--forces", type=float, nargs="+", default=[50, 200, 400])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--kick-step", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="results/rebuild_unitree/cfc_kick.json")
    args = ap.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sm, ss = np.asarray(ck["state_mean"], np.float32), np.asarray(ck["state_std"], np.float32)
    max_steps = int(cfg.env.max_episode_steps)

    contrib = load_cfc_contribution(cfg, device, ck, sm, ss)
    print(f"CfC contribution (mean |cfc-nominal| on action in [-1,1]): {contrib:.5f}", flush=True)

    def fresh(with_cfc: bool):
        h = HDMLModel.from_config(cfg.model).to(device)
        h.load_state_dict(ck["model_state_dict"]); h.eval()
        h.deterministic = True; h.num_flow_steps = 4
        if not with_cfc:
            h.cfc_filter = None
        return h

    out = {"cfc_action_contribution_abs_mean": contrib, "kick_step": args.kick_step, "results": {}}
    for label, with_cfc in (("with_cfc", True), ("no_cfc", False)):
        model = fresh(with_cfc)
        rows = []
        for force in args.forces:
            peaks, oscs, rets = [], [], []
            for i in range(args.episodes):
                r = rollout(model, cfg, sm, ss, force, args.kick_step, 42 + i, max_steps, device)
                peaks.append(r["peak_roll_deg"]); oscs.append(r["post_roll_rms_deg"]); rets.append(r["return"])
            row = {"force": force, "peak_roll_deg": round(float(np.mean(peaks)), 2),
                   "post_roll_rms_deg": round(float(np.mean(oscs)), 2),
                   "return_mean": round(float(np.mean(rets)), 1)}
            rows.append(row)
            print(f"{label:9s} force={force:5.0f}  peak_roll={row['peak_roll_deg']:6.2f}  "
                  f"post_roll_rms={row['post_roll_rms_deg']:6.2f}  return={row['return_mean']:7.1f}", flush=True)
        out["results"][label] = rows
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", args.output)


if __name__ == "__main__":
    main()
