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
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--joint", type=int, default=1,
                    help="Joint index to plot in panel (a); 1,4,7,10 are the four thigh joints")
    ap.add_argument("--output-json", default="results/rebuild_unitree/waveform_rollout.json")
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

    def _perstep_jerk(a: np.ndarray) -> np.ndarray:
        """Per-step smoothness: mean over the 12 joints of |Δ²a_t| (matches the report metric)."""
        d2 = a[2:] - 2 * a[1:-1] + a[:-2]
        return np.mean(np.abs(d2), axis=1)

    hh, dd = _perstep_jerk(ha), _perstep_jerk(da)
    import json
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps({
        "config": args.config,
        "steps": int(args.steps),
        "joint": int(args.joint),
        "hdml_jerk_metric": hjm, "dt_jerk_metric": djm,
        "hdml_mean_jerk": float(hh.mean()), "dt_mean_jerk": float(dd.mean()),
        "hdml_median_jerk": float(np.median(hh)), "dt_median_jerk": float(np.median(dd)),
        "hdml_joint0_amp": float(np.ptp(ha[:, 0])), "dt_joint0_amp": float(np.ptp(da[:, 0])),
        "hdml_joint1_amp": float(np.ptp(ha[:, 1])), "dt_joint1_amp": float(np.ptp(da[:, 1])),
        "note": "clean locomotion, no kick; joint indices are 0-based over the 12 actuators",
    }, indent=2), encoding="utf-8")
    print("saved", args.output_json)

    j = args.joint
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "sans-serif"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True, dpi=260)
    t_h = np.arange(len(ha)); t_d = np.arange(len(da))
    ax1.plot(t_h, ha[:, j], color="#1d4ed8", linewidth=2.0, label="HDML (đề xuất)")
    ax1.plot(t_d, da[:, j], color="#dc2626", linewidth=1.2, linestyle="--", label="Decision Transformer")
    # Zoom to the real signal range instead of the full [-1,1] action box, otherwise a
    # low-amplitude joint looks flat and the two policies are indistinguishable.
    lo = float(min(ha[:, j].min(), da[:, j].min()))
    hi = float(max(ha[:, j].max(), da[:, j].max()))
    pad = 0.08 * max(hi - lo, 1e-3)
    ax1.set_ylim(lo - pad, hi + pad)
    ax1.axhline(0.0, color="#94a3b8", linewidth=0.8, linestyle=":")
    joint_names = [
        "FR hip", "FR thigh", "FR calf", "FL hip", "FL thigh", "FL calf",
        "RR hip", "RR thigh", "RR calf", "RL hip", "RL thigh", "RL calf",
    ]
    jlabel = joint_names[j] if 0 <= j < len(joint_names) else f"khớp {j}"
    ax1.set_ylabel(f"Lệnh mô-men {jlabel}  $a_t$", fontsize=11, fontweight="bold")
    ax1.set_title("Dạng sóng lệnh mô-men trên robot Unitree A1 (đo thật, vòng kín)", fontsize=12.5, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9.5, framealpha=0.92)

    ax2.plot(np.arange(len(hh)), hh + 1e-9, color="#1d4ed8", linewidth=0.9, alpha=0.55, label="HDML (tức thời)")
    ax2.plot(np.arange(len(dd)), dd + 1e-9, color="#dc2626", linewidth=0.8, linestyle="--", alpha=0.55, label="Decision Transformer (tức thời)")
    ax2.axhline(hh.mean(), color="#1d4ed8", linewidth=2.2,
                label=f"HDML trung bình = {hh.mean():.4f}".replace(".", ","))
    ax2.axhline(dd.mean(), color="#dc2626", linewidth=2.2, linestyle="--",
                label=f"Decision Transformer trung bình = {dd.mean():.4f}".replace(".", ","))
    ax2.set_yscale("log")
    ax2.set_xlabel("Bước điều khiển $t$ (50 Hz)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Rung giật tức thời $\frac{1}{12}\sum_j|\Delta^2 a_{t,j}|$", fontsize=10.5, fontweight="bold")
    ax2.set_title("Rung giật cơ học trung bình 12 khớp (thang log, càng thấp càng mượt)", fontsize=11.5, fontweight="bold", pad=8)
    ax2.grid(True, which="both", linestyle="--", alpha=0.45)
    ax2.legend(loc="upper right", fontsize=8.4, framealpha=0.92, ncol=1)

    plt.tight_layout()
    for p in (args.output_png, args.output_pdf):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print("saved", args.output_pdf)


if __name__ == "__main__":
    main()
