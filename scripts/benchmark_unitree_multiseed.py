#!/usr/bin/env python3
"""Pooled multi-seed fair benchmark on Unitree A1 (HDML vs 5 baselines).

Loads one checkpoint per (seed, architecture): seed 42 lives in
``checkpoints/rebuild_unitree/unitree_a1`` and extra seeds in
``checkpoints/rebuild_unitree_multiseed/seed<seed>/unitree_a1``. Evaluates every
architecture over the same 10 deterministic episodes and reports mean +/- std
across the pooled per-seed episode returns, plus mean of per-seed means.

Usage:
    .venv/bin/python scripts/benchmark_unitree_multiseed.py \
        --seeds 42 100 2024 --episodes 10 \
        --output results/rebuild_unitree/multiseed.json
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


def rollout(model, mtype, cfg, sm, ss, episodes, device):
    env = QuadrupedDogEnv(max_episode_steps=cfg.env.max_episode_steps)
    ctx = cfg.training.context_length
    adim = cfg.model.action_dim
    returns, jerks = [], []
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
                if mtype == "hdml":
                    out = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)
                elif mtype == "iql":
                    out = model.get_action(states=ts[:, -1, :])
                elif mtype == "mlp":
                    out = model.get_action(ts[:, -1, :], tr[:, -1, :])
                else:
                    out = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt)
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
    env.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return returns, float(np.mean(jerks))


def seed_dir(base: Path, seed: int) -> Path:
    return (base / f"seed{seed}" / "unitree_a1") if seed != 42 else Path("checkpoints/rebuild_unitree/unitree_a1")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/unitree_a1_aug.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--base-dir", default="checkpoints/rebuild_unitree_multiseed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="results/rebuild_unitree/multiseed.json")
    args = ap.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    base = Path(args.base_dir)

    archs = {
        "HDML": ("hdml", None),
        "Decision Transformer": ("seq", lambda p: DecisionTransformerBaseline(
            prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim,
            d_model=cfg.model.d_model, nhead=4, num_layers=cfg.model.num_mamba_layers)),
        "Decision RNN": ("seq", lambda p: DecisionRNNBaseline(
            prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim,
            d_model=cfg.model.d_model, num_layers=cfg.model.num_mamba_layers)),
        "Diffusion Policy": ("seq", lambda p: DiffusionPolicyBaseline(
            prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim,
            d_model=cfg.model.d_model, denoising_steps=10)),
        "IQL": ("iql", lambda p: IQLBaseline(
            prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, hidden_dim=256)),
        "MLP-BC": ("mlp", lambda p: MLPBCBaseline(
            prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, hidden_dim=256)),
    }
    ckpt_names = {
        "HDML": "best_model.pt", "Decision Transformer": "baselines/dt_best.pt",
        "Decision RNN": "baselines/rnn_best.pt", "Diffusion Policy": "baselines/diffusion_best.pt",
        "IQL": "baselines/iql_best.pt", "MLP-BC": "baselines/mlp_best.pt",
    }

    results: dict[str, dict] = {}
    for name, (mtype, factory) in archs.items():
        per_seed_returns: dict[int, float] = {}
        pooled: list[float] = []
        jerks: list[float] = []
        for seed in args.seeds:
            sd = seed_dir(base, seed)
            ck = torch.load(sd / ckpt_names[name], map_location=device, weights_only=False)
            if name == "HDML":
                model = HDMLModel.from_config(cfg.model).to(device)
                model.load_state_dict(ck["model_state_dict"]); model.eval()
                model.deterministic = True; model.num_flow_steps = 4
            else:
                assert factory is not None
                model = factory(None).to(device)
                model.load_state_dict(ck["model_state_dict"]); model.eval()
                if hasattr(model, "deterministic"):
                    model.deterministic = True
            sm = np.asarray(ck["state_mean"], np.float32)
            ss = np.asarray(ck["state_std"], np.float32)
            rets, jerk = rollout(model, mtype, cfg, sm, ss, args.episodes, device)
            per_seed_returns[seed] = float(np.mean(rets))
            pooled.extend(rets); jerks.append(jerk)
            print(f"{name:22s} seed={seed:5d}  return={per_seed_returns[seed]:8.1f}  jerk={jerk:.4f}", flush=True)
            del model
        arr = np.asarray(pooled, np.float64)
        seed_means = np.asarray(list(per_seed_returns.values()), np.float64)
        results[name] = {
            "per_seed_mean": per_seed_returns,
            "mean": float(arr.mean()), "std": float(arr.std()),
            "mean_of_seed_means": float(seed_means.mean()), "std_of_seed_means": float(seed_means.std()),
            "jerk": float(np.mean(jerks)), "n_episodes": int(arr.size),
        }
        print(f"==> {name:22s} pooled mean={results[name]['mean']:.1f} +/- {results[name]['std']:.1f} | "
              f"seed-mean={results[name]['mean_of_seed_means']:.1f} +/- {results[name]['std_of_seed_means']:.1f}",
              flush=True)

    out = {"seeds": args.seeds, "episodes_per_seed": args.episodes, "models": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", args.output)


if __name__ == "__main__":
    main()
