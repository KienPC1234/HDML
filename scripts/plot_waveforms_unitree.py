#!/usr/bin/env python3
"""Real Unitree A1 actuation-waveform figure: HDML vs Decision Transformer.

Rolls out both *trained* policies closed-loop in the 12-DoF MuJoCo quadruped
(no kick, clean locomotion), records the raw joint-0 torque command over time
and computes the instantaneous second-difference (jerk) of the executed
action. Everything is measured; no synthetic signal is added to any model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.models import DecisionTransformerBaseline, HDMLModel
from hdml.utils.config import HDMLConfig
from hdml.utils.metrics import compute_action_smoothness


def rollout(model, mtype, cfg, device, sm, ss, steps, seed=42):
    env = QuadrupedDogEnv(max_episode_steps=steps)
    obs, _ = env.reset(seed=seed)
    ctx = cfg.training.context_length
    adim = cfg.model.action_dim
    hs, ha, hr, ht = [], [], [], []
    act = []
    rtg = float(cfg.env.target_return)
    for t in range(steps):
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
            else:
                out = model.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt)
        a = out[0] if isinstance(out, tuple) else out
        a = a[0, 0, :] if a.ndim == 3 else a[0, :]
        a = np.clip(a.detach().cpu().numpy().astype(np.float32), -1, 1)
        ha.append(a)
        act.append(a)
        obs, r, term, trunc, _ = env.step(a)
        rtg -= float(r)
        if term or trunc:
            break
    env.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    arr = np.asarray(act, np.float32)
    jerk = np.zeros(len(arr), np.float32)
    for i in range(2, len(arr)):
        jerk[i] = np.mean((arr[i] - 2 * arr[i - 1] + arr[i - 2]) ** 2)
    return arr, jerk, float(compute_action_smoothness(arr))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/unitree_a1_aug.yaml")
    ap.add_argument("--hdml-checkpoint", default="checkpoints/rebuild_unitree/unitree_a1/best_model.pt")
    ap.add_argument("--dt-checkpoint", default="checkpoints/rebuild_unitree/unitree_a1/baselines/dt/best_model.pt")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--output-pdf", default="KH_thi_KHKT_TP_2026-2027/Bao_cao_du_an_LaTeX/figures/action_waveforms_unitree.pdf")
    ap.add_argument("--output-png", default="KH_thi_KHKT_TP_2026-2027/Bao_cao_du_an_LaTeX/figures/action_waveforms_unitree.png")
    args = ap.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.hdml_checkpoint, map_location=device, weights_only=False)
    hdml = HDMLModel.from_config(cfg.model).to(device)
    hdml.load_state_dict(ck["model_state_dict"]); hdml.eval()
    hdml.deterministic = True; hdml.num_flow_steps = 4
    sm, ss = ck["state_mean"], ck["state_std"]

    ck_d = torch.load(args.dt_checkpoint, map_location=device, weights_only=False)
    dt = DecisionTransformerBaseline(
        prop_dim=cfg.model.prop_dim, action_dim=cfg.model.action_dim,
        d_model=cfg.model.d_model, nhead=4, num_layers=cfg.model.num_mamba_layers,
    ).to(device)
    dt.load_state_dict(ck_d["model_state_dict"]); dt.eval()

    ha, hja, hjm = rollout(hdml, "hdml", cfg, device, sm, ss, args.steps)
    da, dja, djm = rollout(dt, "dt", cfg, device, sm, ss, args.steps)
    print(f"HDML jerk={hjm:.4f}  DT jerk={djm:.4f}")

    j = args.joint
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "sans-serif"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True, dpi=260)
    t_h = np.arange(len(ha)); t_d = np.arange(len(da))
    ax1.plot(t_h, ha[:, j], color="#1d4ed8", linewidth=2.4, label="HDML (đề xuất)")
    ax1.plot(t_d, da[:, j], color="#dc2626", linewidth=1.4, linestyle="--", label="Decision Transformer")
    ax1.set_ylabel(f"Lệnh mô-men khớp {j}  $a_t \\in [-1,1]$", fontsize=11, fontweight="bold")
    ax1.set_title("Dạng sóng lệnh mô-men trên robot Unitree A1 (đo thật, vòng kín)", fontsize=12.5, fontweight="bold", pad=10)
    ax1.set_ylim(-1.08, 1.08); ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    ax2.plot(t_h, hja + 1e-9, color="#1d4ed8", linewidth=2.2, label="HDML")
    ax2.plot(t_d, dja + 1e-9, color="#dc2626", linewidth=1.3, linestyle="--", label="Decision Transformer")
    ax2.set_yscale("log")
    ax2.set_xlabel("Bước điều khiển $t$ (50 Hz)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Rung giật tức thời $\|\Delta^2 a_t\|^2$", fontsize=11, fontweight="bold")
    ax2.set_title("Rung giật cơ học tức thời (thang log, càng thấp càng mượt)", fontsize=11.5, fontweight="bold", pad=8)
    ax2.grid(True, which="both", linestyle="--", alpha=0.45)
    ax2.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    plt.tight_layout()
    for p in (args.output_png, args.output_pdf):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print("saved", args.output_pdf)


if __name__ == "__main__":
    main()
