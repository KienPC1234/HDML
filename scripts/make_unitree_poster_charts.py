#!/usr/bin/env python3
"""Re-draw the poster charts for the Unitree A1 3D results.

Every bar is read from a real benchmark JSON produced by the unitree scripts,
so the poster cannot drift from the report again:

    results/rebuild_unitree/fair_benchmark.json     (seed 42, 6 architectures)
    results/rebuild_unitree/multiseed.json          (3 seeds, 6 architectures)

Outputs (into --out-dir, default the competition poster_assets folder):
    chart_unitree_returns.png
    chart_unitree_jerk.png
    chart_unitree_multiseed.png
    chart_unitree_latency.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

ORDER = ["HDML", "Decision RNN", "IQL", "Decision Transformer", "MLP-BC", "Diffusion Policy"]
LABEL = {
    "HDML": "HDML", "Decision RNN": "Dec. RNN", "IQL": "IQL",
    "Decision Transformer": "Dec. Trans.", "MLP-BC": "MLP-BC", "Diffusion Policy": "Diffusion",
}
COLORS = ["#0E9F6E", "#F59E0B", "#8B5CF6", "#6B7280", "#F43F5E", "#2563EB"]


def _bar(ax, vals, colors, ylabel, title, fmt="{:.0f}", highlight_idx=0, yerr=None):
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8)
    if yerr is not None:
        ax.errorbar(x, vals, yerr=yerr, fmt="none", ecolor="black", capsize=5, lw=1.4)
    top = max(v + (e if yerr is not None else 0) for v, e in zip(vals, yerr or [0] * len(vals)))
    for xi, v in zip(x, vals):
        ax.text(xi, v + top * 0.02, fmt.format(v), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[n] for n in ORDER], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_ylim(0, top * 1.18)
    ax.grid(axis="y", ls="--", alpha=0.45)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fair", default="results/rebuild_unitree/fair_benchmark.json")
    ap.add_argument("--multiseed", default="results/rebuild_unitree/multiseed.json")
    ap.add_argument("--out-dir",
                    default="KH_thi_KHKT_TP_2026-2027/poster_assets")
    args = ap.parse_args()

    fair = json.loads(Path(args.fair).read_text(encoding="utf-8"))
    multi = json.loads(Path(args.multiseed).read_text(encoding="utf-8"))["models"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. returns, seed 42
    ret = [fair[n]["return_mean"] for n in ORDER]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    _bar(ax, ret, COLORS, "Điểm thưởng vòng kín",
         "Điểm thưởng trên robot Unitree A1 (seed 42, 10 episode)")
    fig.tight_layout(); fig.savefig(out / "chart_unitree_returns.png", bbox_inches="tight"); plt.close(fig)

    # 2. jerk, seed 42
    jerk = [fair[n]["jerk"] for n in ORDER]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    _bar(ax, jerk, COLORS, r"Chỉ số rung giật $\mathcal{J}$ (thấp = mượt)",
         "Độ mượt mô-men khớp trên robot Unitree A1", fmt="{:.4f}")
    fig.tight_layout(); fig.savefig(out / "chart_unitree_jerk.png", bbox_inches="tight"); plt.close(fig)

    # 3. multi-seed returns with std
    mmean = [multi[n]["mean"] for n in ORDER]
    mstd = [multi[n]["std"] for n in ORDER]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    _bar(ax, mmean, COLORS, "Điểm thưởng vòng kín",
         "Điểm thưởng 3 seed (42/100/2024), trung bình ± độ lệch chuẩn",
         yerr=mstd)
    fig.tight_layout(); fig.savefig(out / "chart_unitree_multiseed.png", bbox_inches="tight"); plt.close(fig)

    # 4. eager latency, seed 42
    lat = [fair[n]["latency_ms"] for n in ORDER]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    _bar(ax, lat, COLORS, "Độ trễ suy luận (ms, càng thấp càng nhanh)",
         "Độ trễ suy luận eager trên GPU RTX 4070 SUPER", fmt="{:.2f}")
    ax.axhline(20, color="#E11D48", ls="--", lw=1.4)
    ax.text(len(ORDER) - 0.5, 20.6, "ngưỡng 20 ms (50 Hz)", ha="right", fontsize=8.5, color="#E11D48")
    fig.tight_layout(); fig.savefig(out / "chart_unitree_latency.png", bbox_inches="tight"); plt.close(fig)

    for name in ("chart_unitree_returns", "chart_unitree_jerk",
                 "chart_unitree_multiseed", "chart_unitree_latency"):
        print("wrote", out / f"{name}.png")


if __name__ == "__main__":
    main()
