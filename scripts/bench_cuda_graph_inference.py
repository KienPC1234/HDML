"""CUDA-graph closed-loop inference benchmark with stateful CfC parity checks.

The runner uses exact-length graphs (no zero padding), copies real timesteps
before every replay, and carries the CfC hidden state exactly like the
evaluator. This script checks:

1. eager vs graph equality on identical observation/action/RTG/timestep
   sequences and identical starting CfC state, for every window length;
2. independent eager and graph rollouts from the same environment seed;
3. closed-loop evaluation of three policies: eager stateless (old behaviour),
   eager stateful (new evaluator behaviour), and graph stateful;
4. pure inference and closed-loop latency.

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
from hdml.utils.metrics import compute_action_smoothness


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
    hx: torch.Tensor | None,
) -> tuple[np.ndarray, torch.Tensor | None]:
    with torch.inference_mode():
        action, next_hx, _ = model.get_action(
            states=torch.from_numpy(states).unsqueeze(0).to(device),
            rtgs=torch.from_numpy(rtgs).unsqueeze(0).to(device),
            actions=torch.from_numpy(actions).unsqueeze(0).to(device),
            timesteps=torch.from_numpy(timesteps).unsqueeze(0).to(device),
            hx=hx,
        )
    return action[0].detach().cpu().numpy().astype(np.float32), next_hx


def _random_hx(
    model: HDMLModel, device: torch.device, rng: np.random.Generator
) -> torch.Tensor | None:
    cfc_filter = getattr(model, "cfc_filter", None)
    if cfc_filter is None:
        return None
    size = int(cfc_filter.cfc.state_size)
    return torch.from_numpy(rng.normal(size=(1, size)).astype(np.float32)).to(device)


def _compare_eager_graph(
    model: HDMLModel,
    runner: HDMLGraphRunner,
    device: torch.device,
    contexts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int = 0,
) -> dict[str, Any]:
    """Compare actions and next_hx for identical windows and starting state."""
    rng = np.random.default_rng(seed)
    per_length_action: dict[int, float] = {}
    per_length_hx: dict[int, float] = {}
    for states, rtgs, actions, timesteps in contexts:
        hx0 = _random_hx(model, device, rng)
        eager, eager_hx = _eager_action(model, states, rtgs, actions, timesteps, device, hx0)
        graph = runner(states, rtgs, actions, timesteps, hx=hx0)
        n = int(states.shape[0])
        per_length_action[n] = max(
            per_length_action.get(n, 0.0), float(np.max(np.abs(eager - graph)))
        )
        if runner.cfc_state_size is not None:
            assert eager_hx is not None
            graph_hx = runner.get_hx()
            assert graph_hx is not None
            hx_diff = float(
                np.max(np.abs(graph_hx - eager_hx.detach().cpu().numpy()))
            )
            per_length_hx[n] = max(per_length_hx.get(n, 0.0), hx_diff)
    return {
        "max_abs_action_diff": max(per_length_action.values()) if per_length_action else None,
        "max_abs_next_hx_diff": max(per_length_hx.values()) if per_length_hx else None,
        "per_length_max_abs_action_diff": {
            str(k): v for k, v in sorted(per_length_action.items())
        },
        "per_length_max_abs_next_hx_diff": {
            str(k): v for k, v in sorted(per_length_hx.items())
        },
        "lengths_checked": sorted(per_length_action),
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
    *,
    stateful: bool,
) -> dict[str, Any]:
    """One episode. Graph runner always carries state; eager can be toggled."""
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
    hx: torch.Tensor | None = None
    if runner is not None:
        runner.reset()
    for t in range(max_steps):
        norm = (np.asarray(obs, np.float32) - state_mean) / state_std
        hs.append(norm)
        hr.append(target_rtg / cfg.env.scale_return)
        ht.append(t)
        if not ha:
            ha.append(np.zeros(int(cfg.model.action_dim), np.float32))
        states, rtgs, actions, timesteps = _context(hs, hr, ha, ht, ctx)
        if runner is not None:
            action = runner(states, rtgs, actions, timesteps)
        else:
            action, next_hx = _eager_action(
                model, states, rtgs, actions, timesteps, device,
                hx if stateful else None,
            )
            if stateful:
                hx = next_hx
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        ha.append(action)
        acts.append(action)
        obs, reward, term, trunc, _ = env.step(action)
        rewards.append(float(reward))
        target_rtg -= float(reward)
        if term or trunc:
            break
    env.close()
    actions_arr = np.asarray(acts, np.float32)
    return {
        "return": float(np.sum(rewards)),
        "steps": len(rewards),
        "jerk": compute_action_smoothness(actions_arr),
        "actions": actions_arr,
    }


def _evaluate_policy(
    model: HDMLModel,
    runner: HDMLGraphRunner | None,
    cfg: HDMLConfig,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    device: torch.device,
    episodes: int,
    max_steps: int,
    *,
    stateful: bool,
) -> dict[str, Any]:
    records = [
        _rollout(
            model, runner, cfg, state_mean, state_std, device,
            max_steps=max_steps, seed=42 + ep, stateful=stateful,
        )
        for ep in range(episodes)
    ]
    returns = np.asarray([r["return"] for r in records], np.float64)
    jerks = np.asarray([r["jerk"] for r in records], np.float64)
    steps = np.asarray([r["steps"] for r in records], np.int64)
    return {
        "episodes": episodes,
        "returns": returns.tolist(),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "jerks": jerks.tolist(),
        "jerk_mean": float(jerks.mean()),
        "steps": steps.tolist(),
        "completion_pct": float(100.0 * np.mean(steps >= max_steps)),
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
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-max-steps", type=int, default=1000)
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
    # 1. Equivalence on identical eager sequences, timesteps, and CfC state.
    # ------------------------------------------------------------------
    contexts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
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
        action, _ = _eager_action(model, state, rtg, act, tim, device, hx=None)
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

    # Reset/carry behaviour must be visible and deterministic.
    states = rng.normal(size=(ctx, cfg.model.prop_dim)).astype(np.float32)
    actions = rng.normal(size=(ctx, cfg.model.action_dim)).astype(np.float32)
    rtgs = np.zeros((ctx, 1), np.float32)
    timesteps = (1000 + np.arange(ctx)).astype(np.int64)
    runner.reset()
    action_reset_1 = runner(states, rtgs, actions, timesteps)
    runner.reset()
    hx_after_reset = runner.get_hx()
    action_reset_2 = runner(states, rtgs, actions, timesteps)
    reset_check = {
        "reset_repeat_max_abs_diff": float(np.max(np.abs(action_reset_1 - action_reset_2))),
        "hx_after_reset_max_abs": (
            float(np.max(np.abs(hx_after_reset))) if hx_after_reset is not None else None
        ),
    }

    # Timestep sensitivity: same window and same initial state, different steps.
    timesteps_a = np.arange(ctx, dtype=np.int64)
    timesteps_b = (1000 + np.arange(ctx)).astype(np.int64)
    hx_a = _random_hx(model, device, rng)
    eager_a, _ = _eager_action(model, states, rtgs, actions, timesteps_a, device, hx_a)
    eager_b, _ = _eager_action(model, states, rtgs, actions, timesteps_b, device, hx_a)
    graph_a = runner(states, rtgs, actions, timesteps_a, hx=hx_a)
    graph_b = runner(states, rtgs, actions, timesteps_b, hx=hx_a)
    timestep_check = {
        "eager_change_abs": float(np.max(np.abs(eager_a - eager_b))),
        "graph_change_abs": float(np.max(np.abs(graph_a - graph_b))),
        "graph_vs_eager_a_abs": float(np.max(np.abs(graph_a - eager_a))),
        "graph_vs_eager_b_abs": float(np.max(np.abs(graph_b - eager_b))),
    }

    # ------------------------------------------------------------------
    # 2. Independent rollouts (same seed, same window and CfC policy).
    # ------------------------------------------------------------------
    eager_roll = _rollout(
        model, None, cfg, state_mean, state_std, device,
        max_steps=args.rollout_steps, seed=42, stateful=True,
    )
    graph_roll = _rollout(
        model, runner, cfg, state_mean, state_std, device,
        max_steps=args.rollout_steps, seed=42, stateful=True,
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
    # 3. Closed-loop evaluation: old stateless CfC vs new stateful policy.
    # ------------------------------------------------------------------
    eager_stateless_eval = _evaluate_policy(
        model, None, cfg, state_mean, state_std, device,
        episodes=args.eval_episodes, max_steps=args.eval_max_steps, stateful=False,
    )
    eager_stateful_eval = _evaluate_policy(
        model, None, cfg, state_mean, state_std, device,
        episodes=args.eval_episodes, max_steps=args.eval_max_steps, stateful=True,
    )
    graph_stateful_eval = _evaluate_policy(
        model, runner, cfg, state_mean, state_std, device,
        episodes=args.eval_episodes, max_steps=args.eval_max_steps, stateful=True,
    )

    # ------------------------------------------------------------------
    # 4. Latency: pure inference and closed-loop.
    # ------------------------------------------------------------------
    hx_latency: torch.Tensor | None = None

    def _eager() -> torch.Tensor:
        nonlocal hx_latency
        with torch.inference_mode():
            action, hx_latency, _ = model.get_action(
                states=torch.from_numpy(states).unsqueeze(0).to(device),
                rtgs=torch.from_numpy(rtgs).unsqueeze(0).to(device),
                actions=torch.from_numpy(actions).unsqueeze(0).to(device),
                timesteps=torch.from_numpy(timesteps).unsqueeze(0).to(device),
                hx=hx_latency,
            )
        return action

    for _ in range(20):
        _eager()
    torch.cuda.synchronize(device)
    hx_latency = None
    t0 = time.perf_counter()
    for _ in range(args.samples):
        _eager()
    torch.cuda.synchronize(device)
    eager_ms = (time.perf_counter() - t0) * 1000.0 / args.samples

    runner.reset()
    for _ in range(20):
        runner(states, rtgs, actions, timesteps)
    torch.cuda.synchronize(device)
    runner.reset()
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
        hx_eager: torch.Tensor | None = None
        if use_graph:
            runner.reset()
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
                action, hx_eager = _eager_action(model, st, rg, ac, tm, device, hx_eager)
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
        "mode": (
            "synchronous macro_interval=1, subgoal-hold micro tier, "
            "CfC state carried across control steps and reset at episode start"
        ),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "context_length": ctx,
        "windowing": "exact causal length 1..context_length; no zero padding",
        "timesteps": "real absolute control steps copied before every replay",
        "cfc_state_size": runner.cfc_state_size,
        "capture_seconds_all_lengths": capture_seconds,
        "equivalence": equivalence,
        "reset_check": reset_check,
        "timestep_check": timestep_check,
        "independent_rollouts": {
            "seed": 42,
            "max_steps": args.rollout_steps,
            "policy": "eager and graph both carry CfC state",
            "eager_return": eager_roll["return"],
            "graph_return": graph_roll["return"],
            "return_abs_diff": abs(eager_roll["return"] - graph_roll["return"]),
            "eager_steps": eager_roll["steps"],
            "graph_steps": graph_roll["steps"],
            "max_abs_action_diff_over_common_steps": rollout_diff,
            "common_steps": n_common,
        },
        "closed_loop_evaluation": {
            "episodes": args.eval_episodes,
            "max_steps": args.eval_max_steps,
            "eager_stateless": eager_stateless_eval,
            "eager_stateful": eager_stateful_eval,
            "graph_stateful": graph_stateful_eval,
            "stateful_minus_stateless_return": (
                eager_stateful_eval["return_mean"] - eager_stateless_eval["return_mean"]
            ),
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

    print(
        f"equivalence action max|diff| = {equivalence['max_abs_action_diff']:.3e}, "
        f"next_hx max|diff| = {equivalence['max_abs_next_hx_diff']:.3e} "
        f"over lengths {equivalence['lengths_checked']}"
    )
    print(f"reset repeat diff = {reset_check['reset_repeat_max_abs_diff']:.3e}")
    print(
        f"timestep change: eager={timestep_check['eager_change_abs']:.3e}, "
        f"graph={timestep_check['graph_change_abs']:.3e}"
    )
    print(
        f"rollout return eager={eager_roll['return']:.2f} "
        f"graph={graph_roll['return']:.2f} action diff={rollout_diff:.3e}"
    )
    print(
        "closed-loop returns: "
        f"stateless={eager_stateless_eval['return_mean']:.2f} "
        f"stateful={eager_stateful_eval['return_mean']:.2f} "
        f"graph={graph_stateful_eval['return_mean']:.2f}"
    )
    print(f"eager pure inference: {eager_ms:.3f} ms")
    print(f"graph pure inference: {graph_ms:.3f} ms")
    print(f"graph control step: {np.mean(graph_control):.3f} ms")
    print(f"graph end-to-end: {np.mean(graph_e2e):.3f} ms")
    print("saved", out_path)


if __name__ == "__main__":
    main()
