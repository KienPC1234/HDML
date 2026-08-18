from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from hdml.models import (
    HDMLModel,
    DecisionTransformerBaseline,
    DecisionRNNBaseline,
    DiffusionPolicyBaseline,
    IQLBaseline,
    MLPBCBaseline,
    MambaMLPHeadAblation,
    TransformerLiquidHeadAblation,
)
from hdml.utils.config import HDMLConfig
from hdml.utils.metrics import compute_action_smoothness, get_d4rl_normalized_score

# Ablation / baseline models that must be trained beforehand via train_baselines.py.
TRAINABLE_MODELS: dict[str, str] = {
    "mamba_mlp": "mamba_mlp",
    "transformer_liquid": "transformer_liquid",
    "dt": "dt",
    "diffusion": "diffusion",
    "iql": "iql",
    "rnn": "rnn",
    "mlp": "mlp",
}


def build_model(model_type: str, cfg: HDMLConfig) -> nn.Module:
    """Instantiate a model for the ablation benchmark."""
    m = cfg.model
    if model_type == "hdml":
        return HDMLModel.from_config(cfg.model)
    if model_type == "mamba_mlp":
        return MambaMLPHeadAblation(
            prop_dim=m.prop_dim,
            action_dim=m.action_dim,
            d_model=m.d_model,
            num_mamba_layers=m.num_mamba_layers,
        )
    if model_type == "transformer_liquid":
        return TransformerLiquidHeadAblation(
            prop_dim=m.prop_dim,
            action_dim=m.action_dim,
            d_model=m.d_model,
            nhead=4,
            num_layers=m.num_mamba_layers,
            d_subgoal=m.d_subgoal,
            cfc_units=m.cfc_units,
        )
    if model_type == "dt":
        return DecisionTransformerBaseline(
            prop_dim=m.prop_dim, action_dim=m.action_dim, d_model=m.d_model, nhead=4, num_layers=m.num_mamba_layers
        )
    if model_type == "diffusion":
        return DiffusionPolicyBaseline(
            prop_dim=m.prop_dim, action_dim=m.action_dim, d_model=m.d_model, denoising_steps=10
        )
    if model_type == "iql":
        return IQLBaseline(prop_dim=m.prop_dim, action_dim=m.action_dim, hidden_dim=256)
    if model_type == "rnn":
        return DecisionRNNBaseline(
            prop_dim=m.prop_dim, action_dim=m.action_dim, d_model=m.d_model, num_layers=m.num_mamba_layers
        )
    if model_type == "mlp":
        return MLPBCBaseline(prop_dim=m.prop_dim, action_dim=m.action_dim, hidden_dim=256)
    raise ValueError(f"Unknown model type: {model_type}")


def evaluate_single_seed(
    model: nn.Module,
    model_type: str,
    env_name: str,
    cfg: HDMLConfig,
    device: torch.device,
    seed: int,
    episodes: int = 3,
    perturbation: bool = False,
    state_mean: np.ndarray | None = None,
    state_std: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate a single seed rollout with a correct, training-consistent protocol.

    The protocol mirrors hdml.evaluation.evaluator.HDMLEvaluator:
    - observations are normalized with the training state_mean / state_std
    - RTG targets are scaled by env.scale_return (as during training)
    - real episode timesteps are fed as positional context
    - the action input at the final context position is a_{t-1} (causal, no leakage)
    - survival requires completing the full 1000-step episode
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]  # type: ignore
    act_dim = env.action_space.shape[0]        # type: ignore
    st_mean = state_mean if state_mean is not None else np.zeros(obs_dim, dtype=np.float32)
    st_std = state_std if state_std is not None else np.ones(obs_dim, dtype=np.float32)

    returns: list[float] = []
    jerks: list[float] = []
    latencies: list[float] = []
    completed_episodes: int = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep * 100)
        history_states: list[np.ndarray] = []
        history_actions: list[np.ndarray] = []
        history_rtgs: list[float] = []
        history_timesteps: list[int] = []
        target_rtg = cfg.env.target_return

        ep_actions: list[np.ndarray] = []
        ep_return = 0.0
        cfc_hx: torch.Tensor | None = None
        rnn_hx: tuple[torch.Tensor, torch.Tensor] | None = None
        done = False
        step_count = 0

        while not done and step_count < 1000:
            step_count += 1
            raw_obs = np.asarray(obs, dtype=np.float32)
            if perturbation:
                raw_obs = raw_obs + np.random.normal(0.0, 0.05, size=raw_obs.shape).astype(np.float32)

            norm_obs = (raw_obs - st_mean) / st_std
            scaled_rtg = target_rtg / cfg.env.scale_return

            history_states.append(norm_obs)
            history_rtgs.append(scaled_rtg)
            history_timesteps.append(step_count - 1)
            if len(history_actions) == 0:
                history_actions.append(np.zeros(act_dim, dtype=np.float32))

            ctx_len = min(len(history_states), cfg.training.context_length)
            ctx_states = np.array(history_states[-ctx_len:], dtype=np.float32)
            ctx_actions = np.array(history_actions[-ctx_len:], dtype=np.float32)
            ctx_rtgs = np.array(history_rtgs[-ctx_len:], dtype=np.float32).reshape(-1, 1)
            ctx_time = np.array(history_timesteps[-ctx_len:], dtype=np.int64)

            if ctx_len < cfg.training.context_length:
                pad_k = cfg.training.context_length - ctx_len
                inp_states = np.vstack([np.zeros((pad_k, obs_dim), dtype=np.float32), ctx_states])
                inp_actions = np.vstack([np.zeros((pad_k, act_dim), dtype=np.float32), ctx_actions])
                inp_rtgs = np.vstack([np.zeros((pad_k, 1), dtype=np.float32), ctx_rtgs])
                inp_time = np.concatenate([np.zeros((pad_k,), dtype=np.int64), ctx_time])
            else:
                inp_states, inp_actions, inp_rtgs, inp_time = ctx_states, ctx_actions, ctx_rtgs, ctx_time

            states_t = torch.from_numpy(inp_states).unsqueeze(0).to(device)
            acts_t = torch.from_numpy(inp_actions).unsqueeze(0).to(device)
            rtgs_t = torch.from_numpy(inp_rtgs).unsqueeze(0).to(device)
            time_t = torch.from_numpy(inp_time).unsqueeze(0).to(device)
            cur_state_t = torch.from_numpy(norm_obs).unsqueeze(0).to(device)
            cur_rtg_t = torch.tensor([[scaled_rtg]], dtype=torch.float32, device=device)

            t0 = time.perf_counter()
            with torch.inference_mode():
                if model_type == "hdml":
                    action_t, cfc_hx, _ = model.get_action(
                        states=states_t, rtgs=rtgs_t, actions=acts_t, timesteps=time_t, hx=cfc_hx
                    )
                elif model_type == "mamba_mlp":
                    action_t = model.get_action(states=states_t, rtgs=rtgs_t, actions=acts_t, timesteps=time_t)
                elif model_type == "transformer_liquid":
                    action_t, cfc_hx, _ = model.get_action(
                        states=states_t, rtgs=rtgs_t, actions=acts_t, timesteps=time_t, hx=cfc_hx
                    )
                elif model_type == "dt":
                    action_t = model.get_action(states=states_t, rtgs=rtgs_t, actions=acts_t, timesteps=time_t)
                elif model_type == "diffusion":
                    action_t = model.get_action(states=states_t, rtgs=rtgs_t)
                elif model_type == "iql":
                    action_t = model.get_action(states=cur_state_t)
                elif model_type == "rnn":
                    action_t, rnn_hx = model.get_action(
                        states=states_t, rtgs=rtgs_t, actions=acts_t, timesteps=time_t, hx=rnn_hx
                    )
                elif model_type == "mlp":
                    action_t = model.get_action(cur_state_t, cur_rtg_t)
                else:
                    raise ValueError(f"Unknown model_type {model_type}")

            dt = time.perf_counter() - t0
            latencies.append(dt)

            act_np = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            act_np = np.clip(act_np, -1.0, 1.0)
            ep_actions.append(act_np)

            next_obs, reward, terminated, truncated, _ = env.step(act_np)
            ep_return += float(reward)
            target_rtg -= float(reward)

            obs = next_obs
            history_actions.append(act_np)
            done = terminated or truncated

        returns.append(ep_return)
        jerks.append(compute_action_smoothness(np.array(ep_actions, dtype=np.float32)))
        if step_count >= 1000:
            completed_episodes += 1

    env.close()

    mean_raw_return = float(np.mean(returns))
    mean_latency_ms = float(np.mean(latencies)) * 1000.0

    return {
        "raw_return": mean_raw_return,
        "d4rl_score": get_d4rl_normalized_score(env_name, mean_raw_return),
        "jerk": float(np.mean(jerks)),
        "latency_ms": mean_latency_ms,
        "frequency_hz": 1000.0 / max(mean_latency_ms, 1e-4),
        "survival_rate": (completed_episodes / max(episodes, 1)) * 100.0,
    }


def run_multi_seed_benchmark(
    config_path: str = "configs/halfcheetah_v4_default.yaml",
    checkpoint_path: str = "checkpoints/halfcheetah_v4/best_model.pt",
    seeds: list[int] | None = None,
    device_str: str = "cuda",
) -> None:
    if seeds is None:
        seeds = [42, 100, 2024, 777, 999]

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    cfg = HDMLConfig.from_yaml(config_path)

    print(f"\n{'=' * 115}")
    print(f"MULTI-SEED STATISTICAL ABLATION & BENCHMARK ON {cfg.env.env_name.upper()}")
    print(f"Random Seeds: {seeds} | Active Hardware: {device}")
    print(f"{'=' * 115}\n")

    ckpt_path = Path(checkpoint_path)
    baseline_dir = ckpt_path.parent / "baselines"
    state_mean = None
    state_std = None

    models: dict[str, tuple[str, str, str]] = {
        "HDML (Decision Mamba + Liquid CfC - Ours)": ("hdml", "O(N)", "O(1)"),
        "Ablation: Mamba + MLP Head (No Liquid)": ("mamba_mlp", "O(N)", "O(1)"),
        "Ablation: Transformer + Liquid Head (No Mamba)": ("transformer_liquid", "O(N^2)", "O(N)"),
        "Decision Transformer (Causal Attention)": ("dt", "O(N^2)", "O(N)"),
        "Decision RNN (LSTM Recurrent Policy)": ("rnn", "O(N)", "O(1)"),
        "Diffusion Policy (DDPM 10-step Denoising)": ("diffusion", "O(K*N)", "O(N)"),
        "Implicit Q-Learning (IQL Advantage Actor)": ("iql", "O(1)", "O(1)"),
        "MLP-BC (Standard Feedforward Reactive)": ("mlp", "O(1)", "O(1)"),
    }

    instantiated: dict[str, nn.Module] = {}
    for name, (model_type, _t, _m) in models.items():
        model = build_model(model_type, cfg).to(device)
        if model_type == "hdml":
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                state_mean = ckpt.get("state_mean")
                state_std = ckpt.get("state_std")
                print(f"[LOADED] HDML checkpoint: {ckpt_path}")
            else:
                print(f"[WARNING] HDML checkpoint not found at {ckpt_path}; evaluating UNTRAINED model.")
        else:
            ckpt_file = baseline_dir / f"{model_type}_best.pt"
            if ckpt_file.exists():
                ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                if state_mean is None:
                    state_mean = ckpt.get("state_mean")
                    state_std = ckpt.get("state_std")
                print(f"[LOADED] Trained checkpoint: {ckpt_file}")
            else:
                print(
                    f"[WARNING] No trained checkpoint at {ckpt_file}; "
                    f"evaluating UNTRAINED model. Run `python scripts/train_baselines.py --model {model_type}` first."
                )
        model.eval()
        instantiated[name] = model

    results: dict[str, dict[str, list[float]]] = {}
    for name, (model_type, _t, _m) in models.items():
        print(f"[EVALUATING] {name} across {len(seeds)} random seeds (synchronous per-step inference)...")
        results[name] = {
            "d4rl_score": [],
            "raw_return": [],
            "jerk": [],
            "latency_ms": [],
            "frequency_hz": [],
            "survival_rate": [],
        }
        for seed in seeds:
            metrics = evaluate_single_seed(
                model=instantiated[name],
                model_type=model_type,
                env_name=cfg.env.env_name,
                cfg=cfg,
                device=device,
                seed=seed,
                episodes=3,
                perturbation=False,
                state_mean=state_mean,
                state_std=state_std,
            )
            for k in results[name]:
                results[name][k].append(metrics[k])

    print("\n" + "=" * 115)
    print(f"{'Architecture / Model Variant':<46} | {'Time/Mem':<10} | {'Freq (Hz)':<16} | {'Jerk':<16} | {'D4RL Score':<18} | {'Surv.':<7}")
    print("-" * 115)

    for name, data in results.items():
        _, time_c, mem_c = models[name]
        complexity = f"{time_c}/{mem_c}"
        freq_m, freq_s = np.mean(data["frequency_hz"]), np.std(data["frequency_hz"])
        jerk_m, jerk_s = np.mean(data["jerk"]), np.std(data["jerk"])
        score_m, score_s = np.mean(data["d4rl_score"]), np.std(data["d4rl_score"])
        surv = float(np.mean(data["survival_rate"]))
        print(
            f"{name:<46} | {complexity:<10} | "
            f"{freq_m:>6.1f} +/- {freq_s:<5.1f} | "
            f"{jerk_m:>7.4f} +/- {jerk_s:<5.4f} | "
            f"{score_m:>6.2f} +/- {score_s:<5.2f} | {surv:>5.1f}%"
        )
    print("=" * 115)

    print("\n% LaTeX Code for Publication Paper Submission (Table: Multi-Seed Statistical Ablation):")
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Multi-seed comparison ($5$ Random Seeds) on " + cfg.env.env_name + r". Trained baselines, synchronous per-step inference. Jerk = mean $|\Delta^2 a_t|$.}")
    print(r"\label{tab:ablation_results}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{lccccr}")
    print(r"\toprule")
    print(r"\textbf{Architecture / Model Variant} & \textbf{Complexity (Time/Mem)} & \textbf{Freq (Hz) $\uparrow$} & \textbf{Latency (ms) $\downarrow$} & \textbf{Jerk $\downarrow$} & \textbf{D4RL Score $\uparrow$} \\")
    print(r"\midrule")

    for name, data in results.items():
        _, time_c, mem_c = models[name]
        freq_m, freq_s = np.mean(data["frequency_hz"]), np.std(data["frequency_hz"])
        lat_m, lat_s = np.mean(data["latency_ms"]), np.std(data["latency_ms"])
        jerk_m, jerk_s = np.mean(data["jerk"]), np.std(data["jerk"])
        score_m, score_s = np.mean(data["d4rl_score"]), np.std(data["d4rl_score"])
        print(
            f"{name} & ${time_c} / {mem_c}$ & "
            f"${freq_m:.1f} \\pm {freq_s:.1f}$ & "
            f"${lat_m:.2f} \\pm {lat_s:.2f}$ & "
            f"${jerk_m:.4f} \\pm {jerk_s:.4f}$ & "
            f"${score_m:.2f} \\pm {score_s:.2f}$ \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table*}")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed statistical ablation benchmark.")
    parser.add_argument("--config", type=str, default="configs/halfcheetah_v5_default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/halfcheetah_v5/best_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg_path = args.config
    ckpt_path = args.checkpoint
    if not Path(cfg_path).exists() and Path("configs/halfcheetah_v4_default.yaml").exists():
        cfg_path = "configs/halfcheetah_v4_default.yaml"
    if not Path(ckpt_path).exists() and Path("checkpoints/halfcheetah_v4/best_model.pt").exists():
        ckpt_path = "checkpoints/halfcheetah_v4/best_model.pt"

    run_multi_seed_benchmark(
        config_path=cfg_path,
        checkpoint_path=ckpt_path,
        device_str=args.device,
    )
