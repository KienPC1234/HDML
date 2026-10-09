#!/usr/bin/env python3
"""Fair multi-seed benchmark: HDML vs trained baselines, pooled IQM + mean/std.

Every architecture is evaluated over the SAME evaluation seeds and episode count,
then scores are pooled before computing IQM (RLiable protocol). This removes the
single-seed noise that made earlier HDML numbers look better than they replicate.

Usage:
    python scripts/benchmark_multiseed.py --checkpoint <hdml ckpt> \
        --seeds 42 100 2024 --episodes 10 --output results/multiseed.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_baselines import evaluate_policy  # noqa: E402
from hdml.models import (  # noqa: E402
    HDMLModel, DecisionTransformerBaseline, DecisionRNNBaseline,
    DiffusionPolicyBaseline, IQLBaseline, MLPBCBaseline,
)
from hdml.utils.config import HDMLConfig  # noqa: E402
from hdml.utils.rliable_metrics import compute_iqm, stratified_bootstrap_ci  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/halfcheetah_v5_default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="results/multiseed.json")
    args = ap.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    bdir = ckpt_path.parent / "baselines"

    hdml = HDMLModel.from_config(cfg.model).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    hdml.load_state_dict(ck["model_state_dict"])
    hdml.eval()
    hdml.deterministic = True
    hdml.num_flow_steps = 4
    sm, ss = ck.get("state_mean"), ck.get("state_std")

    def load(cls, name, **kw):
        m = cls(prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim, **kw).to(device)
        m.load_state_dict(torch.load(bdir / f"{name}_best.pt", map_location=device, weights_only=False)["model_state_dict"])
        m.eval()
        return m

    models = [
        ("HDML", hdml, "hdml"),
        ("Diffusion Policy", load(DiffusionPolicyBaseline, "diffusion", d_model=cfg.model.d_model, denoising_steps=10), "diffusion"),
        ("Decision Transformer", load(DecisionTransformerBaseline, "dt", d_model=cfg.model.d_model, num_layers=cfg.model.num_mamba_layers), "dt"),
        ("Decision RNN", load(DecisionRNNBaseline, "rnn", d_model=cfg.model.d_model, num_layers=cfg.model.num_mamba_layers), "rnn"),
        ("IQL", load(IQLBaseline, "iql", hidden_dim=256), "iql"),
        ("MLP-BC", load(MLPBCBaseline, "mlp", hidden_dim=256), "mlp"),
    ]

    results: dict[str, dict] = {}
    for name, model, mtype in models:
        pooled: list[float] = []
        per_seed = {}
        for seed in args.seeds:
            torch.manual_seed(seed)
            res = evaluate_policy(
                model=model, model_type=mtype, env_name=cfg.env.env_name,
                num_episodes=args.episodes, context_length=cfg.training.context_length,
                target_return=cfg.env.target_return, scale_return=cfg.env.scale_return,
                state_mean=sm, state_std=ss, with_perturbations=False, device=device, seed=seed,
            )
            arr = np.asarray(res["norm_scores_array"], dtype=np.float64)
            pooled.extend(arr.tolist())
            per_seed[seed] = float(arr.mean())
        arr = np.asarray(pooled, dtype=np.float64)
        iqm, lo, hi = stratified_bootstrap_ci(arr, stat_fn=compute_iqm, num_bootstraps=5000)
        results[name] = {
            "per_seed_mean": per_seed,
            "mean": float(arr.mean()), "std": float(arr.std()),
            "iqm": float(iqm), "iqm_ci95": [float(lo), float(hi)],
            "n_episodes": int(arr.size),
        }
        print(f"{name:22s} mean={arr.mean():6.2f} +- {arr.std():5.2f}  IQM={iqm:6.2f} [{lo:.2f},{hi:.2f}]")

    out = {"seeds": args.seeds, "episodes_per_seed": args.episodes, "models": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", args.output)


if __name__ == "__main__":
    main()
