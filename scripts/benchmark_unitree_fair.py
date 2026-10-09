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

from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.models import (
    HDMLModel, DecisionTransformerBaseline, DecisionRNNBaseline,
    DiffusionPolicyBaseline, IQLBaseline, MLPBCBaseline,
)
from hdml.utils.config import HDMLConfig
from hdml.utils.metrics import compute_action_smoothness


def rollout(model, mtype, cfg, sm, ss, episodes, seed, device):
    env = QuadrupedDogEnv(max_episode_steps=cfg.env.max_episode_steps)
    ctx = cfg.training.context_length
    adim = cfg.model.action_dim
    returns, jerks, latencies, comps = [], [], [], []
    for ep in range(episodes):
        torch.manual_seed(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
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
            if device.type == "cuda":
                torch.cuda.synchronize()
            import time as _t
            t0 = _t.perf_counter()
            with torch.inference_mode():
                if mtype == "hdml":
                    out = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)
                elif mtype == "iql":
                    out = model.get_action(states=ts[:, -1, :])
                elif mtype == "mlp":
                    out = model.get_action(ts[:, -1, :], tr[:, -1, :])
                else:
                    out = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt)
            a = out[0] if isinstance(out, tuple) else out
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((_t.perf_counter() - t0) * 1000.0)
            a = a[0, 0, :] if a.ndim == 3 else a[0, :]
            a = np.clip(a.cpu().numpy().astype(np.float32), -1, 1)
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
        "completion_pct": float(np.mean(comps) * 100),
        "jerk": float(np.mean(jerks)),
        "latency_ms": float(np.mean(latencies)), "freq_hz": float(1000.0 / np.mean(latencies)),
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
