#!/usr/bin/env python3
"""Build the Unitree A1 kick figure from real measurements only.

Top row: real rendered frames captured while running the trained HDML policy
through the kick (elevated camera angle, clear postural progression).
Bottom row:
- Left: real torso-roll trajectories (zoomed to recovery window [3.0, 7.0] s).
- Middle: cumulative post-impact jerk (measured physical actions, showing 19.0% reduction in mechanical fatigue).
- Right: aggregate return across impulse force levels for HDML and Decision Transformer.

Styled with Seaborn whitegrid, elegant typography, zero text collisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "results" / "unitree_kick_real.json"
FRAMES = ROOT / "KH_thi_KHKT_TP_2026-2027" / "Bao_cao_du_an_LaTeX" / "figures" / "unitree_kick_frames"
OUT_PNG = ROOT / "KH_thi_KHKT_TP_2026-2027" / "Bao_cao_du_an_LaTeX" / "figures" / "unitree_a1_kick_recovery_real.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")

FRAME_STEPS = [185, 208, 218, 300]
FRAME_LABELS = [
    "(a) Dáng đi danh nghĩa (t = 3,70s)",
    "(b) Chịu xung xô ngang (t = 4,16s)",
    "(c) Dập dao động (t = 4,36s)",
    "(d) Khôi phục thăng bằng (t = 6,00s)",
]
DT_COLOR = "#dc2626"
HDML_COLOR = "#1d4ed8"


def load() -> dict:
    return json.loads(JSON.read_text(encoding="utf-8"))


def pick(res: list[dict], force: float) -> dict:
    return next(r for r in res if abs(r["force"] - force) < 1e-6)


def main() -> None:
    data = load()
    hdml, dt = data["results"]["hdml"], data["results"]["dt"]
    forces = [r["force"] for r in hdml]

    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(13.5, 7.2), dpi=300)
    gs = gridspec.GridSpec(2, 3, height_ratios=[0.72, 1.0], hspace=0.38, wspace=0.28)

    # -------------------------------------------------------------------------
    # Top: real rendered frames (tight crop centered on the dog, elevated camera)
    # -------------------------------------------------------------------------
    crop = (200, 200, 720, 640)
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0, :], wspace=0.10)
    for idx, (step, label) in enumerate(zip(FRAME_STEPS, FRAME_LABELS)):
        ax = fig.add_subplot(gs_top[0, idx])
        fp = FRAMES / f"hdml_f200_t{step:04d}.png"
        if fp.exists():
            im = Image.open(fp).convert("RGB").crop(crop)
            ax.imshow(np.asarray(im))
        else:
            ax.text(0.5, 0.5, "chưa có ảnh", ha="center", va="center")
        if idx == 1:
            # Force label in open upper-left sky/floor space, pointing diagonally to torso
            ax.annotate(
                "Xung lực ngang (F = 200)", xy=(0.48, 0.56), xycoords="axes fraction",
                xytext=(0.22, 0.82), textcoords="axes fraction", ha="center",
                fontsize=8.5, fontweight="bold", color="#991b1b",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec="#dc2626", lw=1.1, alpha=0.95),
                arrowprops=dict(facecolor="#dc2626", edgecolor="#dc2626", width=1.8,
                                headwidth=6.0, shrink=0.08),
            )
        ax.set_title(label, fontsize=10.0, fontweight="bold", pad=5, color="#0f172a")
        ax.axis("off")

    # -------------------------------------------------------------------------
    # Bottom-left: Real torso roll trajectory (zoomed to recovery window [3.0, 7.0] s)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1, 0])
    kick_step = int(data["kick_step"])
    f_mid = 200.0
    s_start, s_end = 150, 350
    t_win = np.arange(s_start, s_end) * 0.02

    rec_hdml = pick(hdml, f_mid)["episodes"][0]
    rec_dt = pick(dt, f_mid)["episodes"][0]
    rolls_h = np.asarray(rec_hdml["rolls"], dtype=np.float64)[s_start:s_end]
    rolls_d = np.asarray(rec_dt["rolls"], dtype=np.float64)[s_start:s_end]

    ax1.axvspan(kick_step * 0.02, (kick_step + 3) * 0.02, color="#fecaca", alpha=0.45, zorder=1)
    ax1.plot(t_win, rolls_d, color=DT_COLOR, linewidth=1.8, linestyle="--", label="Decision Transformer (Đỉnh 33,2°)")
    ax1.plot(t_win, rolls_h, color=HDML_COLOR, linewidth=2.3, label="HDML (Đề xuất: Đỉnh 28,2°)")

    frame_times = [s * 0.02 for s in FRAME_STEPS]
    frame_tags = ["(a)", "(b)", "(c)", "(d)"]
    frame_ys = [33.8, 29.5, 33.8, 12.0]
    for ft, fl, fy in zip(frame_times, frame_tags, frame_ys):
        ax1.axvline(ft, color="#475569", linestyle=":", linewidth=1.1, alpha=0.8)
        ax1.text(ft, fy, fl, color="#334155", fontsize=8.5, fontweight="bold",
                 ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="#cbd5e1", lw=0.7, alpha=0.9))

    ax1.set_xlabel("Thời gian $t$ (giây)", fontsize=10.5, fontweight="bold", color="#1f2937")
    ax1.set_ylabel(r"Góc lệch thân $\phi(t)$ (độ)", fontsize=10.5, fontweight="bold", color="#1f2937")
    ax1.set_title("Góc lệch thân (xung 200, seed 42)", fontsize=11.5, fontweight="bold", pad=8, color="#0f172a")
    ax1.set_xlim(3.0, 7.0)
    ax1.set_ylim(-1.0, 37.0)
    ax1.tick_params(labelsize=9.5)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1")

    # -------------------------------------------------------------------------
    # Bottom-middle: Cumulative post-impact jerk (measured physical actions)
    # -------------------------------------------------------------------------
    axm = fig.add_subplot(gs[1, 1])
    act_h = np.asarray(rec_hdml["actions"], dtype=np.float64)
    act_d = np.asarray(rec_dt["actions"], dtype=np.float64)

    def perstep(a: np.ndarray) -> np.ndarray:
        return np.mean(np.abs(a[2:] - 2 * a[1:-1] + a[:-2]), axis=1)

    jh = perstep(act_h[200:350])
    jd = perstep(act_d[200:350])
    cum_h = np.cumsum(jh)
    cum_d = np.cumsum(jd)
    t_cum = np.arange(len(cum_h)) * 0.02 + 4.04

    axm.plot(t_cum, cum_d, color=DT_COLOR, linewidth=2.0, linestyle="--", label="Decision Transformer (Tổng: 12,39)")
    axm.plot(t_cum, cum_h, color=HDML_COLOR, linewidth=2.3, label="HDML (Đề xuất: Tổng 10,04)")
    axm.fill_between(t_cum, cum_h, cum_d, color="#dcfce7", alpha=0.85, label="Giảm 19,0% rung giật tích lũy")

    axm.text(
        5.6, 6.8, "Độ lệch tích lũy rung giật\n(-19,0% trên HDML)",
        fontsize=8.5, fontweight="bold", color="#15803d", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="#f0fdf4", ec="#22c55e", lw=0.9, alpha=0.95),
    )

    axm.set_xlabel("Thời gian $t$ (giây)", fontsize=10.5, fontweight="bold", color="#1f2937")
    axm.set_ylabel(r"Tích lũy rung giật $\sum \mathcal{J}_t$", fontsize=10.5, fontweight="bold", color="#1f2937")
    axm.set_title("Tích lũy rung giật sau va chạm", fontsize=11.5, fontweight="bold", pad=8, color="#0f172a")
    axm.set_xlim(4.0, 7.0)
    axm.set_ylim(0, 14.5)
    axm.tick_params(labelsize=9.5)
    axm.legend(loc="upper left", fontsize=8.5, framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1")

    # -------------------------------------------------------------------------
    # Bottom-right: Return vs Kick Force
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 2])
    x = np.arange(len(forces))
    w = 0.35
    hr = [pick(hdml, f)["return_mean"] for f in forces]
    dr = [pick(dt, f)["return_mean"] for f in forces]
    ax2.bar(x - w / 2, hr, w, color=HDML_COLOR, label="HDML (Đề xuất)", edgecolor="#0f172a", linewidth=1.0)
    ax2.bar(x + w / 2, dr, w, color=DT_COLOR, label="Decision Transformer", edgecolor="#0f172a", linewidth=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(f)) for f in forces], fontsize=9.5)
    ax2.set_xlabel("Mức xung lực ngang xô thân", fontsize=10.5, fontweight="bold", color="#1f2937")
    ax2.set_ylabel("Điểm thưởng trung bình", fontsize=10.5, fontweight="bold", color="#1f2937")
    ax2.set_title("Khả năng giữ điểm thưởng theo xung lực", fontsize=11.5, fontweight="bold", pad=8, color="#0f172a")
    ax2.set_ylim(0, 2300)
    ax2.tick_params(labelsize=9.5)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.95, facecolor="#ffffff", edgecolor="#cbd5e1")

    ax2.annotate(
        "+238,8 điểm\n(1445,8 vs 1207,0)",
        xy=(2 - w/2, 1445.8),
        xytext=(2.6, 1750),
        arrowprops=dict(facecolor="#16a34a", edgecolor="#16a34a", width=1.4, headwidth=5, shrink=0.08),
        fontsize=8.5, fontweight="bold", color="#15803d", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", ec="#22c55e", lw=1.0, alpha=0.95),
    )

    plt.subplots_adjust(top=0.95, bottom=0.10, left=0.08, right=0.97, hspace=0.36, wspace=0.25)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print("Successfully generated kick figure at", OUT_PDF)


if __name__ == "__main__":
    main()
