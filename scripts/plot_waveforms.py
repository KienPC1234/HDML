from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from hdml.models import (
    HDMLModel,
    DecisionTransformerBaseline,
    DiffusionPolicyBaseline,
    MambaMLPHeadAblation,
)
from hdml.utils.config import HDMLConfig
from hdml.utils.metrics import compute_action_smoothness


def _rollout_honest(
    model: torch.nn.Module,
    model_type: str,
    cfg: HDMLConfig,
    env: gym.Env,
    device: torch.device,
    num_steps: int,
    state_mean: np.ndarray | None,
    state_std: np.ndarray | None,
    obs_dim: int,
    act_dim: int,
) -> np.ndarray:
    """Roll out a policy in closed loop and return the raw (unmodified) action sequence.

    The protocol matches hdml.evaluation.evaluator.HDMLEvaluator: normalized
    observations, scaled RTG, real timesteps, and the causal no-leakage action-input
    convention. No synthetic signal is added to any model's outputs.
    """
    obs, _ = env.reset(seed=42)
    st_mean = state_mean if state_mean is not None else np.zeros(obs_dim, dtype=np.float32)
    st_std = state_std if state_std is not None else np.ones(obs_dim, dtype=np.float32)

    history_states: list[np.ndarray] = []
    history_actions: list[np.ndarray] = []
    history_rtgs: list[float] = []
    history_timesteps: list[int] = []
    target_rtg = cfg.env.target_return
    cfc_hx: torch.Tensor | None = None

    raw_actions: list[np.ndarray] = []
    for t in range(num_steps):
        raw_obs = np.asarray(obs, dtype=np.float32)
        norm_obs = (raw_obs - st_mean) / st_std
        scaled_rtg = target_rtg / cfg.env.scale_return

        history_states.append(norm_obs)
        history_rtgs.append(scaled_rtg)
        history_timesteps.append(t)
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

        with torch.inference_mode():
            if model_type == "hdml":
                action_t, cfc_hx, _ = model.get_action(
                    states=states_t, rtgs=rtgs_t, actions=acts_t, timesteps=time_t, hx=cfc_hx
                )
            elif model_type == "mamba_mlp":
                action_t = model.get_action(states=states_t, rtgs=rtgs_t, actions=acts_t, timesteps=time_t)
            elif model_type == "dt":
                action_t = model.get_action(states=states_t, rtgs=rtgs_t, actions=acts_t, timesteps=time_t)
            elif model_type == "diffusion":
                action_t = model.get_action(states=states_t, rtgs=rtgs_t)
            elif model_type == "mlp":
                action_t = model.get_action(cur_state_t, cur_rtg_t)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        act_np = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        act_np = np.clip(act_np, -1.0, 1.0)
        raw_actions.append(act_np)

        next_obs, reward, terminated, truncated, _ = env.step(act_np)
        target_rtg -= float(reward)
        history_actions.append(act_np)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(seed=42)
            history_states = []
            history_actions = []
            history_rtgs = []
            history_timesteps = []
            target_rtg = cfg.env.target_return
            cfc_hx = None

    return np.array(raw_actions, dtype=np.float32)


def _load_ckpt(model: torch.nn.Module, path: Path, device: torch.device) -> bool:
    if path.exists():
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        return True
    return False


def generate_waveforms(
    config_path: str = "configs/halfcheetah_v5_default.yaml",
    checkpoint_path: str = "checkpoints/halfcheetah_v5/best_model.pt",
    num_steps: int = 120,
    joint_idx: int = 0,
    device_str: str = "cuda",
    output_png: str = "plots/action_waveforms.png",
    output_pdf: str = "plots/action_waveforms.pdf",
) -> None:
    """Simulate honest closed-loop rollouts and plot action waveforms and jerk profiles.

    All models are evaluated identically (same protocol, raw outputs, no synthetic
    signal injection). Jerk is the mean absolute second-order action difference.
    """
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    if not Path(config_path).exists() and Path("configs/halfcheetah_v4_default.yaml").exists():
        config_path = "configs/halfcheetah_v4_default.yaml"
    if not Path(checkpoint_path).exists() and Path("checkpoints/halfcheetah_v4/best_model.pt").exists():
        checkpoint_path = "checkpoints/halfcheetah_v4/best_model.pt"

    cfg = HDMLConfig.from_yaml(config_path)
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(checkpoint_path).parent / "baselines"

    state_mean: np.ndarray | None = None
    state_std: np.ndarray | None = None

    # 1. Instantiate models and load trained checkpoints
    hdml = HDMLModel.from_config(cfg.model).to(device)
    if not _load_ckpt(hdml, Path(checkpoint_path), device):
        print(f"[WARNING] HDML checkpoint not found at {checkpoint_path}; using random init.")
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_mean = ckpt.get("state_mean")
        state_std = ckpt.get("state_std")
    hdml.eval()

    dt = DecisionTransformerBaseline(
        prop_dim=cfg.model.prop_dim,
        action_dim=cfg.model.action_dim,
        d_model=cfg.model.d_model,
        nhead=4,
        num_layers=cfg.model.num_mamba_layers,
    ).to(device)
    if not _load_ckpt(dt, baseline_dir / "dt_best.pt", device):
        print("[WARNING] DT checkpoint not found; using random init.")
    dt.eval()

    diff = DiffusionPolicyBaseline(
        prop_dim=cfg.model.prop_dim,
        action_dim=cfg.model.action_dim,
        d_model=cfg.model.d_model,
        denoising_steps=10,
    ).to(device)
    if not _load_ckpt(diff, baseline_dir / "diffusion_best.pt", device):
        print("[WARNING] Diffusion checkpoint not found; using random init.")
    diff.eval()

    mamba_mlp = MambaMLPHeadAblation(
        prop_dim=cfg.model.prop_dim,
        action_dim=cfg.model.action_dim,
        d_model=cfg.model.d_model,
        num_mamba_layers=cfg.model.num_mamba_layers,
    ).to(device)
    if not _load_ckpt(mamba_mlp, baseline_dir / "mamba_mlp_best.pt", device):
        print("[WARNING] Mamba+MLP ablation checkpoint not found; using random init.")
    mamba_mlp.eval()

    # 2. Honest rollouts on the same deterministic seed
    env = gym.make(cfg.env.env_name)
    obs_dim = env.observation_space.shape[0]  # type: ignore
    act_dim = env.action_space.shape[0]        # type: ignore

    model_records: dict[str, dict[str, list[float]]] = {
        "HDML (Mamba + Liquid CfC - Ours)": {"actions": [], "jerks": [], "jerk_mean": 0.0},
        "Decision Transformer (Causal DT)": {"actions": [], "jerks": [], "jerk_mean": 0.0},
        "Diffusion Policy (DDPM 10-Step)": {"actions": [], "jerks": [], "jerk_mean": 0.0},
        "Mamba + MLP Head (Ablation)": {"actions": [], "jerks": [], "jerk_mean": 0.0},
    }
    models = [
        ("HDML (Mamba + Liquid CfC - Ours)", hdml, "hdml"),
        ("Decision Transformer (Causal DT)", dt, "dt"),
        ("Diffusion Policy (DDPM 10-Step)", diff, "diffusion"),
        ("Mamba + MLP Head (Ablation)", mamba_mlp, "mamba_mlp"),
    ]

    for model_name, model, mtype in models:
        actions_arr = _rollout_honest(
            model=model,
            model_type=mtype,
            cfg=cfg,
            env=env,
            device=device,
            num_steps=num_steps,
            state_mean=state_mean,
            state_std=state_std,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )
        model_records[model_name]["actions"] = actions_arr[:, joint_idx].tolist()
        model_records[model_name]["jerk_mean"] = compute_action_smoothness(actions_arr)

        raw_jerks = []
        for i in range(len(actions_arr)):
            if i < 2:
                raw_jerks.append(1e-6)
            else:
                j = actions_arr[i] - 2.0 * actions_arr[i - 1] + actions_arr[i - 2]
                raw_jerks.append(float(np.mean(j**2)))
        window_size = 5
        smoothed_jerks = []
        for i in range(len(raw_jerks)):
            start_k = max(0, i - window_size + 1)
            window_vals = raw_jerks[start_k : i + 1]
            smoothed_jerks.append(float(np.mean(window_vals)))
        model_records[model_name]["jerks"] = smoothed_jerks

    env.close()

    # 3. Publication-quality plot with honest measured labels
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, dpi=300)

    time_steps = np.arange(num_steps)
    styles = {
        "HDML (Mamba + Liquid CfC - Ours)": {"color": "#1f77b4", "linewidth": 2.6, "linestyle": "-", "alpha": 1.0, "zorder": 5},
        "Decision Transformer (Causal DT)": {"color": "#d62728", "linewidth": 1.6, "linestyle": "--", "alpha": 0.85, "zorder": 4},
        "Diffusion Policy (DDPM 10-Step)": {"color": "#ff7f0e", "linewidth": 1.5, "linestyle": ":", "alpha": 0.85, "zorder": 3},
        "Mamba + MLP Head (Ablation)": {"color": "#2ca02c", "linewidth": 1.6, "linestyle": "-.", "alpha": 0.85, "zorder": 2},
    }
    for model_name, data in model_records.items():
        label = f"{model_name} (Jerk={data['jerk_mean']:.4f})"
        st = styles[model_name]
        ax1.plot(
            time_steps, data["actions"], color=st["color"], linewidth=st["linewidth"],
            linestyle=st["linestyle"], label=label, alpha=st["alpha"], zorder=st["zorder"],
        )
        ax2.plot(
            time_steps, data["jerks"], color=st["color"], linewidth=st["linewidth"],
            linestyle=st["linestyle"], label=label, alpha=st["alpha"], zorder=st["zorder"],
        )

    ax1.set_ylabel(f"Continuous Torque Command $a_t^{{({joint_idx})}} \in [-1, 1]$", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Mechanical Actuation Waveforms & Torque Comparison ({cfg.env.env_name} Joint {joint_idx})",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax1.set_ylim(-1.08, 1.08)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=9)

    ax2.set_yscale("log")
    ax2.set_xlabel("Time Step $t$ (Control Horizon)", fontsize=12, fontweight="bold")
    ax2.set_ylabel(r"Instantaneous Jerk $\|\Delta^2 a_t\|^2$ (Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_title("Instantaneous Mechanical Jerk & Actuator Stress (Lower is Smoother)", fontsize=13, fontweight="bold")
    ax2.set_ylim(1e-8, 1e1)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()

    print(f"Action waveform figure successfully exported to:\n  - Raster: {output_png}\n  - Vector: {output_pdf}")
    for name, data in model_records.items():
        print(f"  {name}: mean|d2a| jerk = {data['jerk_mean']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot honest action waveforms and jerk profiles.")
    parser.add_argument("--config", type=str, default="configs/halfcheetah_v5_default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/halfcheetah_v5/best_model.pt")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--joint", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-png", type=str, default="plots/action_waveforms.png")
    parser.add_argument("--output-pdf", type=str, default="plots/action_waveforms.pdf")
    args = parser.parse_args()

    generate_waveforms(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_steps=args.steps,
        joint_idx=args.joint,
        device_str=args.device,
        output_png=args.output_png,
        output_pdf=args.output_pdf,
    )