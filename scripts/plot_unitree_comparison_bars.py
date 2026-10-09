#!/usr/bin/env python3
"""Bar chart of the fair 6-model Unitree A1 comparison (returns + jerk)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "rebuild_unitree" / "fair_benchmark.json"
OUT_PNG = ROOT / "KH_thi_KHKT_TP_2026-2027" / "Bao_cao_du_an_LaTeX" / "figures" / "unitree_comparison_bars.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")

ORDER = ["HDML", "Decision RNN", "IQL", "Decision Transformer", "MLP-BC", "Diffusion Policy"]
LABEL = {
    "HDML": "HDML", "Decision RNN": "Dec. RNN", "IQL": "IQL",
    "Decision Transformer": "Dec. Transformer", "MLP-BC": "MLP-BC", "Diffusion Policy": "Diffusion",
}


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    names = ORDER
    ret = [d[n]["return_mean"] for n in names]
    jerk = [d[n]["jerk"] for n in names]
    colors = ["#1d4ed8"] + ["#94a3b8"] * (len(names) - 1)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6), dpi=240)

    x = np.arange(len(names))
    ax1.bar(x, ret, color=colors, width=0.62)
    ax1.set_xticks(x); ax1.set_xticklabels([LABEL[n] for n in names], rotation=18, ha="right", fontsize=9)
    ax1.set_ylabel("Điểm thưởng vòng kín", fontsize=10.5, fontweight="bold")
    ax1.set_title("Điểm thưởng trên robot Unitree A1 (10 episode)", fontsize=11.5, fontweight="bold")
    ax1.set_ylim(0, max(ret) * 1.15)
    for xi, v in zip(x, ret):
        ax1.text(xi, v + max(ret) * 0.02, f"{v:.0f}", ha="center", fontsize=8.6)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.45)

    ax2.bar(x, jerk, color=["#16a34a"] + ["#cbd5e1"] * (len(names) - 1), width=0.62)
    ax2.set_xticks(x); ax2.set_xticklabels([LABEL[n] for n in names], rotation=18, ha="right", fontsize=9)
    ax2.set_ylabel(r"Chỉ số rung giật $\mathcal{J}$ (thấp = mượt)", fontsize=10.5, fontweight="bold")
    ax2.set_title("Độ mượt mô-men khớp (10 episode)", fontsize=11.5, fontweight="bold")
    ax2.set_ylim(0, max(jerk) * 1.18)
    for xi, v in zip(x, jerk):
        ax2.text(xi, v + max(jerk) * 0.02, f"{v:.3f}".replace(".", ","), ha="center", fontsize=8.6)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.45)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    for p in (OUT_PNG, OUT_PDF):
        fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT_PDF)


if __name__ == "__main__":
    main()
