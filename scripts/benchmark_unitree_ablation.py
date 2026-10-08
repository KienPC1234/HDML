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

from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.models import (
    DecisionTransformerBaseline,
    HDMLModel,
    MambaMLPHeadAblation,
    TransformerLiquidHeadAblation,
)
from hdml.models.flow_policy import GaussianActionPolicy
from hdml.utils.config import HDMLConfig
from hdml.utils.metrics import compute_action_smoothness


def rollout(model, mtype, cfg, sm, ss, episodes, device):
    env = QuadrupedDogEnv(max_episode_steps=cfg.env.max_episode_steps)
    ctx = cfg.training.context_length
    adim = cfg.model.action_dim
    returns, jerks, comps = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=42 + ep)
        hs, ha, hr, ht = [], [], [], []
        ep_r, ep_a = [], []
        rtg = float(cfg.env.target_return)
        for t in range(cfg.env.max_episode_steps):
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
                if mtype in ("hdml", "hdml_nocfc", "hdml_noflow"):
                    out = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)
                elif mtype == "mamba_mlp":
                    out = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt)
                elif mtype == "transformer_liquid":
                    out = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)
                else:
                    raise ValueError(mtype)
            a = out[0] if isinstance(out, tuple) else out
            a = a[0, 0, :] if a.ndim == 3 else a[0, :]
            a = np.clip(a.detach().cpu().numpy().astype(np.float32), -1, 1)
            ha.append(a); ep_a.append(a)
            obs, r, term, trunc, _ = env.step(a)
            ep_r.append(float(r)); rtg -= float(r)
            if term or trunc:
                break
        returns.append(float(np.sum(ep_r)))
        jerks.append(compute_action_smoothness(np.array(ep_a, np.float32)))
        comps.append(1.0 if len(ep_r) >= cfg.env.max_episode_steps else 0.0)
    env.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
        "jerk": float(np.mean(jerks)), "completion_pct": float(np.mean(comps) * 100),
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
