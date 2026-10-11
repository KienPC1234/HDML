#!/usr/bin/env python3
"""Generate ultra-high-legibility charts for Khung 3 of the poster.
Massive fonts, high contrast, zero whitespace waste, readable from 3 meters away.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ORDER = ["HDML", "Dec.RNN", "IQL", "DT", "MLP-BC", "Diffusion"]
FULL_NAMES = ["HDML", "Decision RNN", "IQL", "Decision Transformer", "MLP-BC", "Diffusion Policy"]
COLORS = ["#059669", "#d97706", "#7c3aed", "#475569", "#e11d48", "#2563eb"]


def main() -> None:
    repo_root = Path("/data/HDML_Model")
    out_dir = repo_root / "KH_thi_KHKT_TP_2026-2027" / "poster_assets"
    fair_path = repo_root / "results/rebuild_unitree/fair_benchmark.json"
    multi_path = repo_root / "results/rebuild_unitree/multiseed.json"

    fair = json.loads(fair_path.read_text(encoding="utf-8"))
    multi = json.loads(multi_path.read_text(encoding="utf-8"))["models"]

    # -------------------------------------------------------------
    # 1. CHART: Điểm thưởng 3 seed (Trung bình ± Độ lệch chuẩn)
    # -------------------------------------------------------------
    mmean = [multi[n]["mean"] for n in FULL_NAMES]
    mstd = [multi[n]["std"] for n in FULL_NAMES]

    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=250)
    x = np.arange(len(ORDER))
    bars = ax.bar(x, mmean, color=COLORS, edgecolor="#0f172a", linewidth=1.5, width=0.74)
    ax.errorbar(x, mmean, yerr=mstd, fmt="none", ecolor="#0f172a", capsize=5, lw=2.0)

    for xi, v in zip(x, mmean):
        ax.text(xi, v + 35, f"{v:.0f}", ha="center", fontsize=14, fontweight="bold", color="#0f172a")

    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=22, ha="right", fontsize=12.5, fontweight="bold", color="#0f172a")
    ax.set_ylabel("Điểm thưởng", fontsize=14, fontweight="bold", color="#0f172a")
    ax.set_title("Điểm thưởng (3 Seed độc lập)", fontsize=15, fontweight="bold", color="#0c384a", pad=10)
    ax.set_ylim(0, 1720)
    ax.grid(axis="y", ls="--", alpha=0.55)
    ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout()
    fig.savefig(out_dir / "chart_unitree_multiseed.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated high-legibility chart_unitree_multiseed.png")

    # -------------------------------------------------------------
    # 2. CHART: Độ mượt mô-men khớp J (Thang log, thấp = mượt)
    # -------------------------------------------------------------
    jerk = [fair[n]["jerk"] for n in FULL_NAMES]

    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=250)
    bars = ax.bar(x, jerk, color=COLORS, edgecolor="#0f172a", linewidth=1.5, width=0.74)

    for xi, v in zip(x, jerk):
        ax.text(xi, v * 1.14, f"{v:.3f}".replace(".", ","), ha="center", fontsize=13.5, fontweight="bold", color="#0f172a")

    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=22, ha="right", fontsize=12.5, fontweight="bold", color="#0f172a")
    ax.set_ylabel("Chỉ số jerk J (thấp = mượt)", fontsize=14, fontweight="bold", color="#0f172a")
    ax.set_title("Độ mượt mô-men khớp J", fontsize=15, fontweight="bold", color="#0c384a", pad=10)
    ax.set_yscale("log")
    ax.set_ylim(0.045, 0.45)
    ax.yaxis.set_major_locator(ticker.FixedLocator([0.05, 0.1, 0.2, 0.4]))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(["0,05", "0,10", "0,20", "0,40"]))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(axis="y", ls="--", which="major", alpha=0.55)
    ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout()
    fig.savefig(out_dir / "chart_unitree_jerk.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated high-legibility chart_unitree_jerk.png")

    # -------------------------------------------------------------
    # 3. CHART: Điểm thưởng Seed 42 (10 episode)
    # -------------------------------------------------------------
    ret42 = [fair[n]["return_mean"] for n in FULL_NAMES]

    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=250)
    bars = ax.bar(x, ret42, color=COLORS, edgecolor="#0f172a", linewidth=1.5, width=0.74)

    for xi, v in zip(x, ret42):
        ax.text(xi, v + 35, f"{v:.0f}", ha="center", fontsize=14, fontweight="bold", color="#0f172a")

    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=22, ha="right", fontsize=12.5, fontweight="bold", color="#0f172a")
    ax.set_ylabel("Điểm thưởng", fontsize=14, fontweight="bold", color="#0f172a")
    ax.set_title("Điểm thưởng (Seed 42)", fontsize=15, fontweight="bold", color="#0c384a", pad=10)
    ax.set_ylim(0, 1750)
    ax.grid(axis="y", ls="--", alpha=0.55)
    ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout()
    fig.savefig(out_dir / "chart_unitree_returns.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated high-legibility chart_unitree_returns.png")

    # -------------------------------------------------------------
    # 4. CHART: Độ trễ suy luận GPU (RTX 4070 SUPER)
    # -------------------------------------------------------------
    lat = [fair[n]["latency_ms"] for n in FULL_NAMES]

    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=250)
    bars = ax.bar(x, lat, color=COLORS, edgecolor="#0f172a", linewidth=1.5, width=0.74)

    for xi, v in zip(x, lat):
        if xi == 0:  # HDML
            ax.text(xi, v + 0.35, f"{v:.2f}".replace(".", ","), ha="center", fontsize=13.5, fontweight="bold", color="#0f172a")
            # Clean white text directly inside the green bar
            ax.text(xi, 4.3, "CUDA\nGraph\n0,38 ms", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#ffffff")
        else:
            ax.text(xi, v + 0.35, f"{v:.2f}".replace(".", ","), ha="center", fontsize=13.5, fontweight="bold", color="#0f172a")

    ax.axhline(20, color="#dc2626", ls="--", lw=2.2)
    ax.text(len(ORDER) - 0.5, 20.7, "Chuẩn 20 ms (50 Hz)", ha="right", fontsize=12, fontweight="bold", color="#dc2626")

    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=22, ha="right", fontsize=12.5, fontweight="bold", color="#0f172a")
    ax.set_ylabel("Độ trễ suy luận (ms)", fontsize=14, fontweight="bold", color="#0f172a")
    ax.set_title("Độ trễ suy luận GPU (RTX 4070S)", fontsize=15, fontweight="bold", color="#0c384a", pad=10)
    ax.set_ylim(0, 24)
    ax.grid(axis="y", ls="--", alpha=0.55)
    ax.tick_params(axis="y", labelsize=12)

    fig.tight_layout()
    fig.savefig(out_dir / "chart_unitree_latency.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated high-legibility chart_unitree_latency.png")


if __name__ == "__main__":
    main()
