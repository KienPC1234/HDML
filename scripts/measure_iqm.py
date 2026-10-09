#!/usr/bin/env python3
"""Careful IQM measurement for HDML only, pooled across evaluation seeds.

Reports, for the trained HDML checkpoint on HalfCheetah-v5:
  * per-seed mean normalized score,
  * pooled mean +- std over all episodes,
  * IQM with 95% stratified-bootstrap CI (RLiable protocol).

Usage:
    python scripts/measure_iqm.py --checkpoint <ckpt> --seeds 42 100 2024 --episodes 10
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from hdml.utils.config import HDMLConfig
from hdml.utils.metrics import get_d4rl_normalized_score
from hdml.utils.rliable_metrics import compute_iqm, stratified_bootstrap_ci
from benchmark_baselines import evaluate_policy
from hdml.models import HDMLModel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/halfcheetah_v5_default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = HDMLModel.from_config(cfg.model).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    model.deterministic = True
    model.num_flow_steps = 4
    sm = ck.get("state_mean")
    ss = ck.get("state_std")

    pooled: list[float] = []
    per_seed: dict[int, float] = {}
    for seed in args.seeds:
        res = evaluate_policy(
            model=model, model_type="hdml", env_name=cfg.env.env_name,
            num_episodes=args.episodes, context_length=cfg.training.context_length,
            target_return=cfg.env.target_return, scale_return=cfg.env.scale_return,
            state_mean=sm, state_std=ss, with_perturbations=False, device=device, seed=seed,
        )
        arr = np.asarray(res["norm_scores_array"], dtype=np.float64)
        pooled.extend(arr.tolist())
        per_seed[seed] = float(arr.mean())
        print(f"seed {seed}: mean norm={arr.mean():.2f}  return={res['mean_return']:.0f}")

    pooled_arr = np.asarray(pooled, dtype=np.float64)
    iqm, lo, hi = stratified_bootstrap_ci(pooled_arr, stat_fn=compute_iqm, num_bootstraps=5000)
    record = {
        "checkpoint": args.checkpoint,
        "env": cfg.env.env_name,
        "episodes_per_seed": args.episodes,
        "seeds": args.seeds,
        "per_seed_mean_norm": per_seed,
        "pooled_mean_norm": float(pooled_arr.mean()),
        "pooled_std_norm": float(pooled_arr.std()),
        "pooled_iqm": float(iqm),
        "iqm_ci95": [float(lo), float(hi)],
        "n_episodes": int(pooled_arr.size),
    }
    print(json.dumps(record, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
