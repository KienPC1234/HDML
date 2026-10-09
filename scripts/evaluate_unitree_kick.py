#!/usr/bin/env python3
"""Real lateral-kick recovery experiment for the HDML Unitree A1 policy.

Runs the trained HDML policy closed-loop in the 12-DOF MuJoCo quadruped and
applies a genuine lateral force impulse (via ``qfrc_applied``) of a given
magnitude, then measures torso roll, survival and recovery time. This replaces
the earlier hand-drawn "50 N" figure with a recorded measurement.

Force units: the MuJoCo model's ``qfrc_applied`` is a generalised force on the
floating base; the value is reported in the same N units the collector used, so
magnitudes are comparable to the collection kicks (8-10) rather than an
unverified 50 N claim.

Usage:
    python scripts/evaluate_unitree_kick.py --checkpoint <ckpt> \
        --forces 8 12 16 20 --episodes 3 --output results/unitree_kick.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import HDMLConfig


def torso_roll(obs: np.ndarray) -> float:
    w, x, y, z = float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4])
    return float(np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))))


def run_episode(model, cfg, device, sm, ss, force, kick_step, seed, max_steps=1000):
    env = QuadrupedDogEnv(max_episode_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    act_dim = 12
    hs, ha, hr, ht = [], [], [], []
    rolls, heights = [], []
    recovered_at = None
    peak_roll = 0.0
    alive = 0
    for t in range(max_steps):
        norm = (np.asarray(obs, np.float32) - sm) / ss
        hs.append(norm); hr.append(0.0); ht.append(t)
        if not ha:
            ha.append(np.zeros(act_dim, np.float32))
        cl = min(len(hs), cfg.training.context_length)
        ts = torch.from_numpy(np.array(hs[-cl:], np.float32)).unsqueeze(0).to(device)
        ta = torch.from_numpy(np.array(ha[-cl:], np.float32)).unsqueeze(0).to(device)
        tr = torch.from_numpy(np.array(hr[-cl:], np.float32).reshape(-1, 1)).unsqueeze(0).to(device)
        tt = torch.from_numpy(np.array(ht[-cl:], np.int64)).unsqueeze(0).to(device)
        with torch.inference_mode():
            a, _, _ = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)
        a = np.clip(a[0].cpu().numpy().astype(np.float32), -1.0, 1.0)
        # apply real lateral force at the kick window (3 control steps)
        if kick_step <= t < kick_step + 3:
            env.apply_kick((0.0, float(force), 0.0))
        obs, r, term, trunc, _ = env.step(a)
        ha.append(a)
        roll = abs(torso_roll(obs))
        rolls.append(roll); heights.append(float(obs[0]))
        peak_roll = max(peak_roll, roll)
        if t > kick_step and recovered_at is None and roll < 10.0:
            recovered_at = t - kick_step
        alive = t
        if term or trunc:
            break
    env.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    survived = bool((not term) or (alive >= max_steps - 1))
    return {
        "force": force, "seed": seed, "steps": alive + 1,
        "peak_roll_deg": round(peak_roll, 1),
        "recovery_steps": recovered_at, "recovery_s": (recovered_at * 0.02) if recovered_at is not None else None,
        "survived_full_episode": survived,
        "rolls": [round(x, 1) for x in rolls],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/unitree_a1_aug.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--forces", type=float, nargs="+", default=[8, 12, 16, 20])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--kick-step", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="results/unitree_kick.json")
    args = ap.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HDMLModel.from_config(cfg.model).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval(); model.deterministic = True; model.num_flow_steps = 4
    sm = ck.get("state_mean"); ss = ck.get("state_std")
    if sm is None or ss is None:
        raise ValueError("checkpoint missing state normalization")

    results = []
    for force in args.forces:
        eps = [run_episode(model, cfg, device, sm, ss, force, args.kick_step, 42 + i) for i in range(args.episodes)]
        surv = float(np.mean([e["survived_full_episode"] for e in eps]))
        peak = float(np.mean([e["peak_roll_deg"] for e in eps]))
        rec = [e["recovery_s"] for e in eps if e["recovery_s"] is not None]
        rec_mean = float(np.mean(rec)) if rec else None
        print(f"force={force:5.1f}  peak_roll={peak:5.1f} deg  recovery={rec_mean} s  survival={surv*100:.0f}%  steps={[e['steps'] for e in eps]}")
        results.append({"force": force, "peak_roll_deg": peak, "recovery_s": rec_mean, "survival_rate": surv, "episodes": eps})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"kick_step": args.kick_step, "results": results}, indent=2), encoding="utf-8")
    print("saved", args.output)


if __name__ == "__main__":
    main()
