#!/usr/bin/env python3
"""Measure PACE's effect on macro-backbone call counts with a real checkpoint.

This is an operational integration check, not a return benchmark. It runs the
real HDML model and Unitree A1 MuJoCo environment under three settings:

* no PACE, ``macro_interval=10``;
* PACE with an unreachable threshold (normal plan expiry only);
* PACE with threshold 0 (every due prediction triggers an early replan).

The key assertion is that PACE never turns a 10-step macro plan into a
per-step backbone call. Events and call counts are written to JSON.

Example:
    .venv/bin/python scripts/bench_pace_integration.py \
        --config configs/unitree_a1_cfc_strong.yaml \
        --checkpoint checkpoints/rebuild_unitree/unitree_a1/best_model.pt \
        --output results/fix_pace_cfc_cudagraph/pace_integration.json
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
import torch.nn as nn

from hdml.evaluation.evaluator import HDMLEvaluator
from hdml.evaluation.pace_controller import PACEController
from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import HDMLConfig


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class _CountingPolicy(nn.Module):
    """Transparent wrapper counting macro and micro inference calls."""

    def __init__(self, model: HDMLModel) -> None:
        super().__init__()
        self.model = model
        self.macro_calls = 0
        self.micro_calls = 0

    @property
    def chunk_size(self) -> int:
        return int(self.model.chunk_size)

    def get_action(self, *args: Any, **kwargs: Any):
        self.macro_calls += 1
        return self.model.get_action(*args, **kwargs)

    def act_from_subgoal(self, *args: Any, **kwargs: Any):
        self.micro_calls += 1
        return self.model.act_from_subgoal(*args, **kwargs)


def _run(
    policy: _CountingPolicy,
    cfg: HDMLConfig,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
    pace: PACEController | None,
    max_steps: int,
    macro_interval: int,
) -> dict[str, Any]:
    policy.macro_calls = 0
    policy.micro_calls = 0
    evaluator = HDMLEvaluator(
        model=policy,
        env_name="UnitreeA1-v0",
        context_length=cfg.training.context_length,
        target_return=cfg.env.target_return,
        scale_return=cfg.env.scale_return,
        state_mean=state_mean,
        state_std=state_std,
        device=device,
    )
    env = QuadrupedDogEnv(max_episode_steps=max_steps)
    result = evaluator.evaluate_episode(
        env=env,
        seed=cfg.env.seed,
        max_steps=max_steps,
        macro_interval=macro_interval,
        pace_controller=pace,
    )
    env.close()
    return {
        "macro_backbone_calls": policy.macro_calls,
        "micro_action_calls": policy.micro_calls,
        "per_step_backbone": policy.macro_calls >= result["episode_length"],
        "episode_length": result["episode_length"],
        "pace": result["pace"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unitree_a1_cfc_strong.yaml")
    parser.add_argument(
        "--checkpoint", default="checkpoints/rebuild_unitree/unitree_a1/best_model.pt"
    )
    parser.add_argument("--output", default="results/fix_pace_cfc_cudagraph/pace_integration.json")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--macro-interval", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = HDMLConfig.from_yaml(args.config)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    stored = ckpt.get("config", {})
    if "cfc_residual" in stored and float(stored["cfc_residual"]) != float(cfg.model.cfc_residual):
        print(
            f"WARNING: config cfc_residual={cfg.model.cfc_residual} but checkpoint "
            f"stored {stored['cfc_residual']}; using the checkpoint value."
        )
        cfg.model.cfc_residual = float(stored["cfc_residual"])
    model = HDMLModel.from_config(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.deterministic = True
    model.num_flow_steps = 4
    policy = _CountingPolicy(model)
    state_mean = np.asarray(ckpt["state_mean"], np.float32)
    state_std = np.asarray(ckpt["state_std"], np.float32)

    scenarios = {
        "no_pace": _run(
            policy, cfg, state_mean, state_std, device,
            pace=None, max_steps=args.steps, macro_interval=args.macro_interval,
        ),
        "pace_expiry_only": _run(
            policy, cfg, state_mean, state_std, device,
            pace=PACEController(threshold=1e9), max_steps=args.steps,
            macro_interval=args.macro_interval,
        ),
        "pace_frequent_deviation": _run(
            policy, cfg, state_mean, state_std, device,
            pace=PACEController(threshold=0.0), max_steps=args.steps,
            macro_interval=args.macro_interval,
        ),
    }
    for name, record in scenarios.items():
        assert not record["per_step_backbone"], (
            f"{name}: PACE caused a backbone call at every step: {record}"
        )

    out = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_current_step": ckpt.get("current_step"),
        "chunk_size": int(cfg.model.chunk_size),
        "macro_interval": args.macro_interval,
        "mode": (
            "subgoal-hold micro tier; chunk_size is the flow action-chunk length, "
            "macro_interval is the subgoal lifetime"
        ),
        "max_steps": args.steps,
        "scenarios": scenarios,
        "note": (
            "macro_backbone_calls counts HDMLModel.get_action calls; micro_action_calls "
            "counts act_from_subgoal calls. Return values are not benchmark evidence."
        ),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, record in scenarios.items():
        print(
            f"{name}: macro={record['macro_backbone_calls']} micro={record['micro_action_calls']} "
            f"steps={record['episode_length']} events={[e['step'] for e in record['pace']['events']]}"
        )
    print("saved", out_path)


if __name__ == "__main__":
    main()
