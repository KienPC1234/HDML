#!/usr/bin/env python3
"""Fair comparison on the 12-DoF Unitree A1 quadruped: HDML vs common baselines.

All models are trained on the same dataset/config/seed and evaluated closed-loop
in the same MuJoCo quadruped env with the same protocol, context length and
state normalisation. Reports raw episode return (the env has its own reward
scale, so D4RL normalisation does not apply), completion, action jerk and GPU
latency.

Usage:
    python scripts/benchmark_unitree_fair.py --episodes 10 --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hdml.evaluation.rollout import run_rollout
from hdml.models import (
    DecisionRNNBaseline, DecisionTransformerBaseline, DiffusionPolicyBaseline,
    HDMLModel, IQLBaseline, MLPBCBaseline,
)
from hdml.utils.config import HDMLConfig


def rollout(model, mtype, cfg, sm, ss, episodes, seed, device):
    """Closed-loop evaluation; dispatch matches the historical per-script loop."""

    def policy_step(m, ts, ta, tr, tt):
        if mtype == "hdml":
            return m.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)
        if mtype == "iql":
            return m.get_action(states=ts[:, -1, :])
        if mtype == "mlp":
            return m.get_action(ts[:, -1, :], tr[:, -1, :])
        return m.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt)

    result = run_rollout(
        model, cfg, sm, ss, device, episodes, policy_step,
        seed_base=seed, manual_torch_seed=True, measure_latency=True,
    )
    return {
        "return_mean": float(np.mean(result.returns)), "return_std": float(np.std(result.returns)),
        "completion_pct": float(np.mean(result.completions) * 100),
        "jerk": float(np.mean(result.jerks)),
        "latency_ms": float(np.mean(result.latencies_ms)),
        "freq_hz": float(1000.0 / np.mean(result.latencies_ms)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/unitree_a1_aug.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/rebuild_unitree/unitree_a1/best_model.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="results/rebuild_unitree/fair_benchmark.json")
    args = ap.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckp = Path(args.checkpoint)
    bdir = ckp.parent / "baselines"
    ck = torch.load(ckp, map_location=device, weights_only=False)
    sm, ss = ck.get("state_mean"), ck.get("state_std")

    def load(cls, name, **kw):
        m = cls(prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, **kw).to(device)
        m.load_state_dict(torch.load(bdir / f"{name}_best.pt", map_location=device, weights_only=False)["model_state_dict"])
        m.eval()
        if hasattr(m, "deterministic"):
            m.deterministic = True
        return m

    hdml = HDMLModel.from_config(cfg.model).to(device)
    hdml.load_state_dict(ck["model_state_dict"]); hdml.eval(); hdml.deterministic = True; hdml.num_flow_steps = 4

    models = [
        ("HDML", hdml, "hdml"),
        ("Decision Transformer", load(DecisionTransformerBaseline, "dt", d_model=cfg.model.d_model, nhead=4, num_layers=cfg.model.num_mamba_layers), "seq"),
        ("Decision RNN", load(DecisionRNNBaseline, "rnn", d_model=cfg.model.d_model, num_layers=cfg.model.num_mamba_layers), "seq"),
        ("Diffusion Policy", load(DiffusionPolicyBaseline, "diffusion", d_model=cfg.model.d_model, denoising_steps=10), "seq"),
        ("IQL", load(IQLBaseline, "iql", hidden_dim=256), "iql"),
        ("MLP-BC", load(MLPBCBaseline, "mlp", hidden_dim=256), "mlp"),
    ]
    out = {}
    for name, model, mt in models:
        r = rollout(model, mt, cfg, sm, ss, args.episodes, 42, device)
        out[name] = r
        print(f"{name:22s} return={r['return_mean']:8.1f} +- {r['return_std']:6.1f}  comp={r['completion_pct']:5.1f}%  "
              f"jerk={r['jerk']:.4f}  lat={r['latency_ms']:.2f}ms ({r['freq_hz']:.0f}Hz)")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", args.output)


if __name__ == "__main__":
    main()
