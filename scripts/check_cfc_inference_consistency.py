#!/usr/bin/env python3
"""Check CfC train/rollout memory consistency without retraining.

The training path applies ``ncps.torch.CfC`` to a whole context window from a
zero hidden state. This script feeds the same real checkpoint, real dataset
windows, and the same Mamba latents through:

1. one full-sequence CfC call (hx=None) and
2. T single-token calls carrying ``next_hx`` step by step.

If both agree, stateful rollout uses exactly the same recurrence as training and
no retraining is needed merely to make the memory usage consistent. The script
also checks that ``HDMLModel.get_action`` accepts a carried state on a
multi-step context (the old code rejected that) and that ``hx=None`` really
resets the filter.

Example:
    .venv/bin/python scripts/check_cfc_inference_consistency.py \
        --config configs/unitree_a1_cfc_strong.yaml \
        --checkpoint checkpoints/rebuild_unitree/unitree_a1/best_model.pt \
        --dataset data/unitree_a1_trajectories.npz \
        --output results/fix_pace_cfc_cudagraph/cfc_consistency.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hdml.data.collector import TrajectoryCollector
from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import HDMLConfig


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build_windows(
    trajectories: list[dict[str, np.ndarray]],
    context_length: int,
    num_windows: int,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    scale_return: float,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample real causal windows: states, RTGs, previous actions, timesteps."""
    rng = np.random.default_rng(seed)
    valid = [traj for traj in trajectories if len(traj["observations"]) >= context_length]
    if not valid:
        raise RuntimeError("No trajectory is long enough for the requested context length")
    states: list[np.ndarray] = []
    rtgs: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    timesteps: list[np.ndarray] = []
    for _ in range(num_windows):
        traj = valid[int(rng.integers(0, len(valid)))]
        n = len(traj["observations"])
        start = int(rng.integers(0, n - context_length + 1))
        sl = slice(start, start + context_length)
        obs = (traj["observations"][sl].astype(np.float32) - state_mean) / state_std
        raw_actions = traj["actions"].astype(np.float32)
        # Causal convention: input at window position 0 is a_{start-1}
        # (zero at episode start); subsequent positions are a_{start}, ...
        prev_actions = np.empty_like(raw_actions[sl])
        prev_actions[0] = raw_actions[start - 1] if start > 0 else 0.0
        if context_length > 1:
            prev_actions[1:] = raw_actions[start : start + context_length - 1]
        states.append(obs)
        rtgs.append(
            (traj["returns_to_go"][sl].astype(np.float32) / scale_return).reshape(-1, 1)
        )
        actions.append(prev_actions)
        if "timesteps" in traj:
            timesteps.append(traj["timesteps"][sl].astype(np.int64))
        else:
            timesteps.append(np.arange(start, start + context_length, dtype=np.int64))
    return (
        torch.from_numpy(np.stack(states)),
        torch.from_numpy(np.stack(rtgs)),
        torch.from_numpy(np.stack(actions)),
        torch.from_numpy(np.stack(timesteps)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unitree_a1_cfc_strong.yaml")
    parser.add_argument(
        "--checkpoint", default="checkpoints/rebuild_unitree/unitree_a1/best_model.pt"
    )
    parser.add_argument("--dataset", default="data/unitree_a1_trajectories.npz")
    parser.add_argument("--output", default="results/fix_pace_cfc_cudagraph/cfc_consistency.json")
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = HDMLConfig.from_yaml(args.config)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stored_cfg = ckpt.get("config", {})
    if "cfc_residual" in stored_cfg and float(stored_cfg["cfc_residual"]) != float(cfg.model.cfc_residual):
        print(
            f"WARNING: config cfc_residual={cfg.model.cfc_residual} but checkpoint "
            f"stored {stored_cfg['cfc_residual']}; using the checkpoint value."
        )
        cfg.model.cfc_residual = float(stored_cfg["cfc_residual"])
    if not cfg.model.use_cfc:
        raise SystemExit("This check requires cfg.model.use_cfc=True")

    model = HDMLModel.from_config(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.deterministic = True
    model.num_flow_steps = 4
    assert model.cfc_filter is not None

    ctx = int(args.context or cfg.training.context_length)
    trajectories = TrajectoryCollector.load_dataset(args.dataset)
    state_mean = np.asarray(ckpt["state_mean"], np.float32)
    state_std = np.asarray(ckpt["state_std"], np.float32)
    states, rtgs, actions, timesteps = _build_windows(
        trajectories,
        context_length=ctx,
        num_windows=args.windows,
        state_mean=state_mean,
        state_std=state_std,
        scale_return=float(cfg.env.scale_return),
        seed=cfg.training.seed,
    )
    states = states.to(device)
    rtgs = rtgs.to(device)
    actions = actions.to(device)
    timesteps = timesteps.to(device)

    with torch.inference_mode():
        _sg, latent, _v, _ns, _fc, _executed, _hx, chunk = model.encode_with_chunk(
            states=states,
            rtgs=rtgs,
            actions=actions,
            timesteps=timesteps,
            hx=None,
            action_last_only=False,
        )
        assert chunk is not None
        nominal = chunk[:, :, 0, :]

        full, full_hx = model.cfc_filter(nominal, latent, hx=None)

        stepwise: list[torch.Tensor] = []
        hx: torch.Tensor | None = None
        for t in range(ctx):
            out_t, hx = model.cfc_filter(nominal[:, t], latent[:, t], hx=hx)
            stepwise.append(out_t)
        stepwise_tensor = torch.stack(stepwise, dim=1)

        max_action_diff = float(
            (full - stepwise_tensor).abs().max().item()
        )
        max_hx_diff = float((full_hx - hx).abs().max().item()) if hx is not None else 0.0

        # get_action must accept carried state on a multi-step context.
        action_reset, hx_reset, _ = model.get_action(
            states=states, rtgs=rtgs, actions=actions, timesteps=timesteps, hx=None
        )
        action_carried, hx_carried, _ = model.get_action(
            states=states, rtgs=rtgs, actions=actions, timesteps=timesteps, hx=hx_reset
        )
        action_reset_again, hx_reset_again, _ = model.get_action(
            states=states, rtgs=rtgs, actions=actions, timesteps=timesteps, hx=None
        )
        assert hx_carried is not None and hx_reset_again is not None
        get_action_carry_effect = float(
            (action_carried - action_reset).abs().max().item()
        )
        get_action_reset_effect = float(
            (action_reset_again - action_reset).abs().max().item()
        )
        hx_finite = bool(torch.isfinite(hx_carried).all().item())

    passed = max_action_diff <= args.tolerance and max_hx_diff <= args.tolerance
    record: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_current_step": ckpt.get("current_step"),
        "dataset": args.dataset,
        "device": str(device),
        "cfc_residual": float(cfg.model.cfc_residual),
        "mode": "full-sequence CfC (training path) vs token-by-token carried hx (rollout path)",
        "context_length": ctx,
        "windows": args.windows,
        "tolerance": args.tolerance,
        "full_vs_stepwise_max_abs_diff": max_action_diff,
        "final_hx_max_abs_diff": max_hx_diff,
        "passed": passed,
        "get_action_with_carried_hx": {
            "carry_vs_reset_max_abs_diff": get_action_carry_effect,
            "reset_is_deterministic_max_abs_diff": get_action_reset_effect,
            "next_hx_finite": hx_finite,
        },
        "time_step": "ncps CfC default timespans=1.0 (normalized unit), unchanged",
        "cfc_input": "Mamba latent (d_model), not the raw sensor vector",
        "requires_retraining": False if passed else "unknown - inspect before retraining",
        "note": (
            "Training resets hx at each sampled context window; rollout resets at "
            "episode start and then carries next_hx across control steps."
        ),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(
        f"full-sequence vs stepwise: max|diff|={max_action_diff:.3e}, "
        f"final hx diff={max_hx_diff:.3e}, tolerance={args.tolerance:.1e} -> "
        f"{'PASS' if passed else 'FAIL'}"
    )
    print(
        "get_action carried hx vs reset: "
        f"{get_action_carry_effect:.3e}; reset repeat diff "
        f"{get_action_reset_effect:.3e}"
    )
    print("saved", out_path)


if __name__ == "__main__":
    main()
