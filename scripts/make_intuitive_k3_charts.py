#!/usr/bin/env python3
"""Generate ultra-intuitive, highly visual charts for Khung 3 of the poster.
Designed so that ANY viewer (even non-specialists) understands instantly.
Directly maps to the 4 key metrics: Return, Smoothness, Robustness, Real-time Latency.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "KH_thi_KHKT_TP_2026-2027" / "poster_assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. CHART 1: Điểm thưởng vận động (3 Seed độc lập)
    # -------------------------------------------------------------------------
    models = ["HDML", "DT", "Dec.RNN", "IQL", "MLP-BC", "Diffusion"]
    means = [1466, 1443, 1438, 1430, 1311, 987]
    stds = [23, 11, 13, 19, 26, 38]
    colors = ["#059669", "#475569", "#d97706", "#7c3aed", "#e11d48", "#2563eb"]

    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=250)
    x = np.arange(len(models))
    ax.bar(x, means, color=colors, edgecolor="#0f172a", linewidth=1.5, width=0.72)
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="#0f172a", capsize=5, lw=2.0)

    for xi, v in zip(x, means):
        ax.text(xi, v + 35, f"{v}", ha="center", fontsize=13.5, fontweight="bold", color="#0f172a")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=12.5, fontweight="bold", color="#0f172a")
    ax.set_ylabel("Điểm thưởng (càng cao càng tốt)", fontsize=12.5, fontweight="bold", color="#0f172a")
    ax.set_title("1. Điểm thưởng bước đi (3 Seed)", fontsize=14, fontweight="bold", color="#0c384a", pad=10)
    ax.set_ylim(0, 1720)
    ax.grid(axis="y", ls="--", alpha=0.55)
    ax.tick_params(axis="y", labelsize=11.5)

    # Note best
    ax.text(0, 1600, "★ Cao nhất", ha="center", fontsize=11, fontweight="bold", color="#059669")

    fig.tight_layout()
    fig.savefig(out_dir / "chart_k3_returns.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated chart_k3_returns.png")

    # -------------------------------------------------------------------------
    # 2. CHART 2: Độ mượt mô-men & Tỷ lệ xung nguy hiểm (HDML vs Transformer)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=250)
    categories = ["Rung giật Jerk J\n(x100, thấp = tốt)", "Tỷ lệ vượt ngưỡng\nnguy hiểm (>0,2)"]
    hdml_vals = [6.28, 7.4]   # J = 0.0628 scaled x100, and 7.4%
    dt_vals = [7.33, 11.6]    # J = 0.0733 scaled x100, and 11.6%

    x = np.arange(len(categories))
    w = 0.34

    ax.bar(x - w / 2, hdml_vals, width=w, color="#059669", edgecolor="#0f172a", linewidth=1.5, label="HDML (Đề xuất)")
    ax.bar(x + w / 2, dt_vals, width=w, color="#dc2626", edgecolor="#0f172a", linewidth=1.5, label="Transformer (DT)")

    # Values on bars
    ax.text(x[0] - w / 2, 6.28 + 0.35, "0,063", ha="center", fontsize=13.5, fontweight="bold", color="#059669")
    ax.text(x[0] + w / 2, 7.33 + 0.35, "0,073", ha="center", fontsize=13.5, fontweight="bold", color="#dc2626")

    ax.text(x[1] - w / 2, 7.4 + 0.35, "7,4%", ha="center", fontsize=13.5, fontweight="bold", color="#059669")
    ax.text(x[1] + w / 2, 11.6 + 0.35, "11,6%", ha="center", fontsize=13.5, fontweight="bold", color="#dc2626")

    # Callouts
    ax.text(x[0], 9.2, "Giảm 14,4%\nrung giật", ha="center", fontsize=11.5, fontweight="bold", color="#065f46",
            bbox=dict(boxstyle="round,pad=0.25", fc="#d1fae5", ec="#10b981", lw=1.2))
    ax.text(x[1], 13.2, "Giảm 36,2%\nxung nguy hiểm", ha="center", fontsize=11.5, fontweight="bold", color="#065f46",
            bbox=dict(boxstyle="round,pad=0.25", fc="#d1fae5", ec="#10b981", lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12.5, fontweight="bold", color="#0f172a")
    ax.set_ylabel("Mức độ rung động (thấp = mượt)", fontsize=13, fontweight="bold", color="#0f172a")
    ax.set_title("2. Độ mượt & An toàn cơ cấu", fontsize=14.5, fontweight="bold", color="#0c384a", pad=10)
    ax.set_ylim(0, 16.0)
    ax.grid(axis="y", ls="--", alpha=0.55)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_dir / "chart_k3_smoothness.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated chart_k3_smoothness.png")

    # -------------------------------------------------------------------------
    # 3. CHART 3: Kháng ngoại lực xô đẩy (HDML vs Transformer theo các mức xô)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=250)
    forces = ["0 N", "50 N", "100 N", "200 N", "400 N"]
    hdml_kick = [1495, 1489, 1499, 1446, 293]
    dt_kick = [1449, 1445, 1468, 1207, 287]

    x = np.arange(len(forces))
    wb = 0.33
    offset = 0.18

    ax.bar(x - offset, hdml_kick, width=wb, color="#059669", edgecolor="#0f172a", linewidth=1.4, label="HDML (Đề xuất)")
    ax.bar(x + offset, dt_kick, width=wb, color="#dc2626", edgecolor="#0f172a", linewidth=1.4, label="Transformer (DT)")

    # Values on x=3 (Mức 200 N) - smaller font, clean spacing, no collisions
    ax.text(x[3] - offset, 1446 + 22, "1446", ha="center", fontsize=9.5, fontweight="bold", color="#059669")
    ax.text(x[3] + offset, 1207 + 22, "1207", ha="center", fontsize=9.5, fontweight="bold", color="#dc2626")

    # Difference annotation at 200 N - lowered down and centered over 200 N bars
    ax.text(x[3], 1675, "Chênh lệch:\n+238,8 điểm", ha="center", va="center", fontsize=10, fontweight="bold", color="#065f46",
            bbox=dict(boxstyle="round,pad=0.28", fc="#d1fae5", ec="#10b981", lw=1.1))

    # Note at 400 N - smaller text in 2 lines, strictly contained above 400 N bars
    ax.text(x[4], 340, "Mất\nthăng bằng", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#64748b")

    ax.set_xticks(x)
    ax.set_xticklabels(forces, fontsize=12, fontweight="bold", color="#0f172a")
    ax.set_xlabel("Xung lực xô ngang", fontsize=12.5, fontweight="bold", color="#0f172a")
    ax.set_ylabel("Điểm thưởng duy trì", fontsize=12.5, fontweight="bold", color="#0f172a")
    ax.set_title("3. Kháng xung lực xô ngang", fontsize=14.5, fontweight="bold", color="#0c384a", pad=10)
    ax.set_ylim(0, 1950)
    ax.grid(axis="y", ls="--", alpha=0.55)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_dir / "chart_k3_robustness.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated chart_k3_robustness.png")

    # -------------------------------------------------------------------------
    # 4. CHART 4: Độ trễ suy luận & Đáp ứng thời gian thực
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.4), dpi=250)
    modes = ["HDML\nCUDA Graph", "HDML\nCPU máy trạm", "DT\n(GPU)", "HDML\nGPU Eager"]
    lats = [0.38, 2.41, 2.44, 8.60]
    mcolors = ["#059669", "#0284c7", "#475569", "#94a3b8"]

    x = np.arange(len(modes))
    ax.bar(x, lats, color=mcolors, edgecolor="#0f172a", linewidth=1.5, width=0.68)

    for xi, v in zip(x, lats):
        ax.text(xi, v + 0.45, f"{v:.2f}".replace(".", ",") + " ms", ha="center", fontsize=13.5, fontweight="bold", color="#0f172a")

    # Real-time threshold line
    ax.axhline(20.0, color="#dc2626", ls="--", lw=2.2)
    ax.text(1.5, 20.7, "Giới hạn độ trễ: 20 ms (50 Hz)", ha="center", fontsize=11.5, fontweight="bold", color="#dc2626")

    # Neutral callout on CUDA Graph (replace 'Siêu tốc' with factual frequency)
    ax.text(0, 3.6, "Tần số suy luận\n2649 Hz", ha="center", fontsize=10.5, fontweight="bold", color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.25", fc="#059669", ec="#047857", lw=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=10.0, fontweight="bold", color="#0f172a")
    ax.set_ylabel("Độ trễ suy luận (ms, thấp = tốt)", fontsize=12.5, fontweight="bold", color="#0f172a")
    ax.set_title("4. Suy luận thời gian thực", fontsize=14.5, fontweight="bold", color="#0c384a", pad=10)
    ax.set_ylim(0, 24.0)
    ax.grid(axis="y", ls="--", alpha=0.55)
    ax.tick_params(axis="y", labelsize=11.5)

    fig.tight_layout()
    fig.savefig(out_dir / "chart_k3_latency.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated chart_k3_latency.png")


if __name__ == "__main__":
    main()
