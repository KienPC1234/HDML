#!/usr/bin/env python3
"""Component-contribution ablation on Unitree A1.

Decomposes HDML's gain by removing one component at a time, all evaluated
closed-loop on the same 12-DoF quadruped with the same protocol/seed:

  * HDML (full)              Mamba-3 + Flow Matching + CfC
  * HDML w/o CfC             Mamba-3 + Flow Matching
  * HDML w/o Flow            Mamba-3 + CfC, Gaussian action head
  * Mamba + MLP head         Mamba-3 only (no CfC, no generative head)
  * Transformer + CfC        Transformer backbone + CfC (no Mamba-3)

The "w/o CfC" and "w/o Flow" variants are produced by structurally editing the
trained HDML model at inference (removing the CfC filter / swapping the Flow
head for a Gaussian head). The Mamba+MLP and Transformer+CfC variants are the
separately-trained ablation checkpoints. This is an inference-time component
analysis, stated as such; it is not a re-trained ablation for every cell.

Usage:
    .venv/bin/python scripts/benchmark_unitree_ablation.py --episodes 10 \
        --output results/rebuild_unitree/ablation.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hdml.evaluation.rollout import run_rollout
from hdml.models import HDMLModel, MambaMLPHeadAblation, TransformerLiquidHeadAblation
from hdml.utils.config import HDMLConfig


def rollout(model, mtype, cfg, sm, ss, episodes, device):
    """Closed-loop ablation evaluation with the shared rollout bookkeeping."""

    def policy_step(m, ts, ta, tr, tt):
        if mtype in ("hdml", "hdml_nocfc", "hdml_noflow"):
            return m.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)
        if mtype == "mamba_mlp":
            return m.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt)
        if mtype == "transformer_liquid":
            return m.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)
        raise ValueError(mtype)

    result = run_rollout(model, cfg, sm, ss, device, episodes, policy_step, seed_base=42)
    return {
        "return_mean": float(np.mean(result.returns)), "return_std": float(np.std(result.returns)),
        "jerk": float(np.mean(result.jerks)), "completion_pct": float(np.mean(result.completions) * 100),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/unitree_a1_aug.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/rebuild_unitree/unitree_a1/best_model.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="results/rebuild_unitree/ablation.json")
    args = ap.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    m = cfg.model
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sm, ss = np.asarray(ck["state_mean"], np.float32), np.asarray(ck["state_std"], np.float32)
    bdir = Path(args.checkpoint).parent / "baselines"

    def base_hdml():
        h = HDMLModel.from_config(m).to(device)
        h.load_state_dict(ck["model_state_dict"]); h.eval()
        h.deterministic = True; h.num_flow_steps = 4
        return h

    variants: list[tuple[str, object, str]] = []

    h_full = base_hdml()
    variants.append(("HDML (đầy đủ: Mamba-3 + Flow + CfC)", h_full, "hdml"))

    h_nocfc = base_hdml()
    h_nocfc.cfc_filter = None
    variants.append(("HDML bỏ CfC (Mamba-3 + Flow)", h_nocfc, "hdml_nocfc"))

    ck_mm = torch.load(bdir / "mamba_mlp_best.pt", map_location=device, weights_only=False)
    mm = MambaMLPHeadAblation(prop_dim=m.prop_dim, action_dim=m.action_dim, d_model=m.d_model,
                              num_mamba_layers=m.num_mamba_layers).to(device)
    mm.load_state_dict(ck_mm["model_state_dict"]); mm.eval()
    variants.append(("Mamba-3 + MLP (bỏ CfC và Flow)", mm, "mamba_mlp"))

    ck_tl = torch.load(bdir / "transformer_liquid" / "best_model.pt", map_location=device, weights_only=False)
    tl = TransformerLiquidHeadAblation(prop_dim=m.prop_dim, action_dim=m.action_dim, d_model=m.d_model,
                                       nhead=4, num_layers=m.num_mamba_layers, d_subgoal=m.d_subgoal,
                                       cfc_units=m.cfc_units).to(device)
    tl.load_state_dict(ck_tl["model_state_dict"]); tl.eval()
    variants.append(("Transformer + CfC (bỏ Mamba-3)", tl, "transformer_liquid"))

    out: dict[str, dict] = {}
    for label, model, mt in variants:
        r = rollout(model, mt, cfg, sm, ss, args.episodes, device)
        out[label] = r
        print(f"{label:50s} return={r['return_mean']:8.1f}  jerk={r['jerk']:.4f}  comp={r['completion_pct']:.0f}%", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "episodes": args.episodes,
        "note": "Inference-time component ablation on Unitree A1; w/o-CfC and w/o-Flow are structural edits of the trained HDML model; Mamba+MLP and Transformer+CfC are separately-trained checkpoints.",
        "variants": out,
    }, indent=2), encoding="utf-8")
    print("saved", args.output)


if __name__ == "__main__":
    main()
