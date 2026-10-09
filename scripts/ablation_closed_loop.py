#!/usr/bin/env python3
"""Closed-loop A/B: what maximizes HDML's return?

Sweeps evaluation-side choices (flow steps, deterministic vs stochastic, CfC
on/off) on the same checkpoint to find the configuration with the highest
closed-loop return. Diagnostic tool, not a leaderboard.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_baselines import evaluate_policy  # noqa: E402
from hdml.models import HDMLModel  # noqa: E402
from hdml.utils.config import HDMLConfig  # noqa: E402


def run(ckpt, cfg, device, seed, episodes, deterministic, steps, cfc_on):
    m = HDMLModel.from_config(cfg.model).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(sd["model_state_dict"])
    m.eval()
    m.deterministic = deterministic
    m.num_flow_steps = steps
    if not cfc_on:
        m.cfc_filter = None
    res = evaluate_policy(
        model=m, model_type="hdml", env_name=cfg.env.env_name,
        num_episodes=episodes, context_length=cfg.training.context_length,
        target_return=cfg.env.target_return, scale_return=cfg.env.scale_return,
        state_mean=sd.get("state_mean"), state_std=sd.get("state_std"),
        with_perturbations=False, device=device, seed=seed,
    )
    return res["mean_return"], float(np.mean(res["norm_scores_array"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/halfcheetah_v5_default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"{'det':>5} {'steps':>5} {'cfc':>4} {'return':>9} {'norm':>7}")
    for determ in (True, False):
        for steps in (1, 4, 8):
            for cfc_on in (True, False):
                ret, norm = run(args.checkpoint, cfg, device, 42, args.episodes, determ, steps, cfc_on)
                print(f"{str(determ):>5} {steps:>5} {str(cfc_on):>4} {ret:>9.0f} {norm:>7.2f}")


if __name__ == "__main__":
    main()
