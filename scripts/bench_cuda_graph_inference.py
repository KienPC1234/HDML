"""CUDA-graph closed-loop inference benchmark with honest equivalence checks.

The runner now uses exact-length graphs (no zero padding) and copies the real
timesteps into the static buffer before every replay. This script checks:

1. eager vs graph equality on identical observation/action/RTG/timestep
   sequences, for every window length from 1 to ``context_length``;
2. independent eager and graph rollouts from the same environment seed;
3. pure inference and closed-loop latency on the same checkpoint.

Every number is written to a fresh JSON file together with the git commit,
config, checkpoint, and control mode.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hdml.deployment.graph_runner import HDMLGraphRunner
from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.models import HDMLModel
from hdml.utils.config import HDMLConfig


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[1]
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _context(
    history_states: list[np.ndarray],
    history_rtgs: list[float],
    history_actions: list[np.ndarray],
    history_timesteps: list[int],
    context_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the same causal window used by the evaluator (no padding)."""
    n = min(len(history_states), context_length)
    return (
        np.asarray(history_states[-n:], np.float32),
        np.asarray(history_rtgs[-n:], np.float32).reshape(-1, 1),
        np.asarray(history_actions[-n:], np.float32),
        np.asarray(history_timesteps[-n:], np.int64),
    )


def _eager_action(
    model: HDMLModel,
    states: np.ndarray,
    rtgs: np.ndarray,
    actions: np.ndarray,
    timesteps: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    with torch.inference_mode():
        action, _, _ = model.get_action(
            states=torch.from_numpy(states).unsqueeze(0).to(device),
            rtgs=torch.from_numpy(rtgs).unsqueeze(0).to(device),
            actions=torch.from_numpy(actions).unsqueeze(0).to(device),
            timesteps=torch.from_numpy(timesteps).unsqueeze(0).to(device),
            hx=None,
        )
    return action[0].detach().cpu().numpy().astype(np.float32)


def _compare_eager_graph(
    model: HDMLModel,
    runner: HDMLGraphRunner,
    device: torch.device,
    contexts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    per_length: dict[int, float] = {}
    for states, rtgs, actions, timesteps in contexts:
        eager = _eager_action(model, states, rtgs, actions, timesteps, device)
        graph = runner(states, rtgs, actions, timesteps)
        diff = float(np.max(np.abs(eager - graph)))
        n = int(states.shape[0])
        per_length[n] = max(per_length.get(n, 0.0), diff)
    return {
        "max_abs_diff": max(per_length.values()) if per_length else None,
        "per_length_max_abs_diff": {str(k): v for k, v in sorted(per_length.items())},
        "lengths_checked": sorted(per_length),
    }


def _rollout(
    model: HDMLModel,
    runner: HDMLGraphRunner | None,
    cfg: HDMLConfig,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    env = QuadrupedDogEnv(max_episode_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    ctx = int(cfg.training.context_length)
    target_rtg = float(cfg.env.target_return)
    hs: list[np.ndarray] = []
    ha: list[np.ndarray] = []
    hr: list[float] = []
    ht: list[int] = []
    acts: list[np.ndarray] = []
    rewards: list[float] = []
    for t in range(max_steps):
        norm = (np.asarray(obs, np.float32) - state_mean) / state_std
        hs.append(norm)
        hr.append(target_rtg / cfg.env.scale_return)
        ht.append(t)
        if not ha:
            ha.append(np.zeros(int(cfg.model.action_dim), np.float32))
        states, rtgs, actions, timesteps = _context(hs, hr, ha, ht, ctx)
        if runner is None:
            action = _eager_action(model, states, rtgs, actions, timesteps, device)
        else:
            action = runner(states, rtgs, actions, timesteps)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        ha.append(action)
        acts.append(action)
        obs, reward, term, trunc, _ = env.step(action)
        rewards.append(float(reward))
        target_rtg -= float(reward)
        if term or trunc:
            break
    env.close()
    return {
        "return": float(np.sum(rewards)),
        "steps": len(rewards),
        "actions": np.asarray(acts, np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/unitree_a1_cfc_strong.yaml")
    parser.add_argument(
        "--checkpoint", default="checkpoints/rebuild_unitree/unitree_a1/best_model.pt"
    )
    parser.add_argument("--output", default="results/fix_pace_cfc_cudagraph/cuda_graph_bench.json")
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=300)
    parser.add_argument("--samples", type=int, default=500)
    args = parser.parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    cfg = HDMLConfig.from_yaml(config_path)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    if "cfc_residual" in ckpt_cfg and float(ckpt_cfg["cfc_residual"]) != float(cfg.model.cfc_residual):
        print(
            "WARNING: config cfc_residual="
            f"{cfg.model.cfc_residual} but checkpoint stored {ckpt_cfg['cfc_residual']}; "
            "using the checkpoint value so the evaluated architecture matches its weights."
        )
        cfg.model.cfc_residual = float(ckpt_cfg["cfc_residual"])

    model = HDMLModel.from_config(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.deterministic = True
    model.num_flow_steps = 4
    state_mean = np.asarray(ckpt["state_mean"], np.float32)
    state_std = np.asarray(ckpt["state_std"], np.float32)
    ctx = int(args.context or cfg.training.context_length)

    t_capture0 = time.perf_counter()
    runner = HDMLGraphRunner(
        model,
        prop_dim=cfg.model.prop_dim,
        action_dim=cfg.model.action_dim,
        context_length=ctx,
        device=device,
        precapture=True,
    )
    torch.cuda.synchronize(device)
    capture_seconds = time.perf_counter() - t_capture0

    # ------------------------------------------------------------------
    # 1. Equivalence on identical eager sequences and timesteps.
    # ------------------------------------------------------------------
    contexts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    # Real episode contexts, every prefix length seen by the evaluator.
    env = QuadrupedDogEnv(max_episode_steps=ctx)
    obs, _ = env.reset(seed=321)
    hs: list[np.ndarray] = []
    ha: list[np.ndarray] = []
    hr: list[float] = []
    ht: list[int] = []
    target_rtg = float(cfg.env.target_return)
    for t in range(ctx):
        norm = (np.asarray(obs, np.float32) - state_mean) / state_std
        hs.append(norm)
        hr.append(target_rtg / cfg.env.scale_return)
        ht.append(t)
        if not ha:
            ha.append(np.zeros(int(cfg.model.action_dim), np.float32))
        contexts.append(_context(hs, hr, ha, ht, ctx))
        state, rtg, act, tim = contexts[-1]
        action = _eager_action(model, state, rtg, act, tim, device)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        ha.append(action)
        obs, reward, term, trunc, _ = env.step(action)
        target_rtg -= float(reward)
        if term or trunc:
            break
    env.close()
    checked_lengths = {int(c[0].shape[0]) for c in contexts}

    # Synthetic windows cover any lengths the probe missed (e.g. early fall).
    rng = np.random.default_rng(7)
    for n in range(1, ctx + 1):
        if n in checked_lengths:
            continue
        states = rng.normal(size=(n, cfg.model.prop_dim)).astype(np.float32)
        actions = rng.normal(size=(n, cfg.model.action_dim)).astype(np.float32)
        rtgs = rng.normal(size=(n, 1)).astype(np.float32)
        timesteps = (100 + np.arange(n)).astype(np.int64)
        contexts.append((states, rtgs, actions, timesteps))

    equivalence = _compare_eager_graph(model, runner, device, contexts)

    # Timestep sensitivity: same window, different absolute timesteps.
    states = rng.normal(size=(ctx, cfg.model.prop_dim)).astype(np.float32)
    actions = rng.normal(size=(ctx, cfg.model.action_dim)).astype(np.float32)
    rtgs = np.zeros((ctx, 1), np.float32)
    timesteps_a = np.arange(ctx, dtype=np.int64)
    timesteps_b = (1000 + np.arange(ctx)).astype(np.int64)
    eager_a = _eager_action(model, states, rtgs, actions, timesteps_a, device)
    eager_b = _eager_action(model, states, rtgs, actions, timesteps_b, device)
    graph_a = runner(states, rtgs, actions, timesteps_a)
    graph_b = runner(states, rtgs, actions, timesteps_b)
    timestep_check = {
        "eager_change_abs": float(np.max(np.abs(eager_a - eager_b))),
        "graph_change_abs": float(np.max(np.abs(graph_a - graph_b))),
        "graph_vs_eager_a_abs": float(np.max(np.abs(graph_a - eager_a))),
        "graph_vs_eager_b_abs": float(np.max(np.abs(graph_b - eager_b))),
    }

    # ------------------------------------------------------------------
    # 2. Independent rollouts (same seed, same window policy).
    # ------------------------------------------------------------------
    eager_roll = _rollout(
        model, None, cfg, state_mean, state_std, device, args.rollout_steps, seed=42
    )
    graph_roll = _rollout(
        model, runner, cfg, state_mean, state_std, device, args.rollout_steps, seed=42
    )
    n_common = min(eager_roll["actions"].shape[0], graph_roll["actions"].shape[0])
    rollout_diff = float(
        np.max(
            np.abs(
                eager_roll["actions"][:n_common] - graph_roll["actions"][:n_common]
            )
        )
    )

    # ------------------------------------------------------------------
    # 3. Latency: pure inference and closed-loop.
    # ------------------------------------------------------------------
    states = rng.normal(size=(ctx, cfg.model.prop_dim)).astype(np.float32)
    actions = rng.normal(size=(ctx, cfg.model.action_dim)).astype(np.float32)
    rtgs = rng.normal(size=(ctx, 1)).astype(np.float32)
    timesteps = (1000 + np.arange(ctx)).astype(np.int64)

    def _eager() -> torch.Tensor:
        with torch.inference_mode():
            return model.get_action(
                states=torch.from_numpy(states).unsqueeze(0).to(device),
                rtgs=torch.from_numpy(rtgs).unsqueeze(0).to(device),
                actions=torch.from_numpy(actions).unsqueeze(0).to(device),
                timesteps=torch.from_numpy(timesteps).unsqueeze(0).to(device),
                hx=None,
            )[0]

    for _ in range(20):
        _eager()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(args.samples):
        _eager()
    torch.cuda.synchronize(device)
    eager_ms = (time.perf_counter() - t0) * 1000.0 / args.samples

    for _ in range(20):
        runner(states, rtgs, actions, timesteps)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(args.samples):
        runner(states, rtgs, actions, timesteps)
    torch.cuda.synchronize(device)
    graph_ms = (time.perf_counter() - t0) * 1000.0 / args.samples

    # Closed-loop graph/eager timings over a fresh episode.
    def _closed_loop(use_graph: bool) -> tuple[list[float], list[float], list[float], int]:
        env = QuadrupedDogEnv(max_episode_steps=args.rollout_steps)
        obs, _ = env.reset(seed=42)
        hs_c: list[np.ndarray] = []
        ha_c: list[np.ndarray] = []
        hr_c: list[float] = []
        ht_c: list[int] = []
        target = float(cfg.env.target_return)
        replay_ms: list[float] = []
        control_ms: list[float] = []
        e2e_ms: list[float] = []
        for t in range(args.rollout_steps):
            t_e2e = time.perf_counter()
            norm = (np.asarray(obs, np.float32) - state_mean) / state_std
            hs_c.append(norm)
            hr_c.append(target / cfg.env.scale_return)
            ht_c.append(t)
            if not ha_c:
                ha_c.append(np.zeros(int(cfg.model.action_dim), np.float32))
            st, rg, ac, tm = _context(hs_c, hr_c, ha_c, ht_c, ctx)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            if use_graph:
                action = runner(st, rg, ac, tm)
            else:
                action = _eager_action(model, st, rg, ac, tm, device)
            torch.cuda.synchronize(device)
            replay_ms.append((time.perf_counter() - t0) * 1000.0)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            ha_c.append(action)
            control_ms.append((time.perf_counter() - t_e2e) * 1000.0)
            obs, reward, term, trunc, _ = env.step(action)
            target -= float(reward)
            e2e_ms.append((time.perf_counter() - t_e2e) * 1000.0)
            if term or trunc:
                break
        env.close()
        return replay_ms, control_ms, e2e_ms, len(e2e_ms)

    graph_replay, graph_control, graph_e2e, graph_steps = _closed_loop(use_graph=True)
    eager_replay, eager_control, eager_e2e, eager_steps = _closed_loop(use_graph=False)

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_current_step": ckpt.get("current_step"),
        "checkpoint_best_loss": ckpt.get("best_loss"),
        "cfc_residual": float(cfg.model.cfc_residual),
        "mode": "synchronous macro_interval=1, subgoal-hold micro tier, CfC hx=None",
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "context_length": ctx,
        "window_policy": "exact causal length 1..context_length; no zero padding",
        "timesteps": "real absolute control steps copied before every replay",
        "capture_seconds_all_lengths": capture_seconds,
        "equivalence": equivalence,
        "timestep_check": timestep_check,
        "independent_rollouts": {
            "seed": 42,
            "max_steps": args.rollout_steps,
            "eager_return": eager_roll["return"],
            "graph_return": graph_roll["return"],
            "return_abs_diff": abs(eager_roll["return"] - graph_roll["return"]),
            "eager_steps": eager_roll["steps"],
            "graph_steps": graph_roll["steps"],
            "max_abs_action_diff_over_common_steps": rollout_diff,
            "common_steps": n_common,
        },
        "latency": {
            "pure_inference_samples": args.samples,
            "eager_pure_inference_ms": eager_ms,
            "eager_pure_inference_hz": 1000.0 / eager_ms,
            "graph_pure_inference_ms": graph_ms,
            "graph_pure_inference_hz": 1000.0 / graph_ms,
            "speedup_pure_inference": eager_ms / graph_ms,
            "graph_replay_only_ms": float(np.mean(graph_replay)),
            "graph_replay_only_hz": 1000.0 / float(np.mean(graph_replay)),
            "graph_control_step_ms": float(np.mean(graph_control)),
            "graph_control_step_hz": 1000.0 / float(np.mean(graph_control)),
            "graph_end_to_end_ms": float(np.mean(graph_e2e)),
            "graph_end_to_end_hz": 1000.0 / float(np.mean(graph_e2e)),
            "graph_closed_loop_steps": graph_steps,
            "eager_replay_only_ms": float(np.mean(eager_replay)),
            "eager_control_step_ms": float(np.mean(eager_control)),
            "eager_control_step_hz": 1000.0 / float(np.mean(eager_control)),
            "eager_end_to_end_ms": float(np.mean(eager_e2e)),
            "eager_end_to_end_hz": 1000.0 / float(np.mean(eager_e2e)),
            "eager_closed_loop_steps": eager_steps,
        },
        "note": (
            "pure latency excludes environment stepping; control_step includes sensor "
            "normalisation, context construction, inference and action clipping; "
            "end_to_end additionally includes the MuJoCo physics step. The historical "
            "0.536 ms figure was a control_step-style measurement, not end_to_end."
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"equivalence max|diff| = {equivalence['max_abs_diff']:.3e} "
          f"over lengths {equivalence['lengths_checked']}")
    print(f"timestep change: eager={timestep_check['eager_change_abs']:.3e}, "
          f"graph={timestep_check['graph_change_abs']:.3e}")
    print(f"rollout return eager={eager_roll['return']:.2f} graph={graph_roll['return']:.2f} "
          f"action diff={rollout_diff:.3e}")
    print(f"eager pure inference: {eager_ms:.3f} ms")
    print(f"graph pure inference: {graph_ms:.3f} ms")
    print(f"graph control step: {np.mean(graph_control):.3f} ms")
    print(f"graph end-to-end: {np.mean(graph_e2e):.3f} ms")
    print("saved", out_path)


if __name__ == "__main__":
    main()
