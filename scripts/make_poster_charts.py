#!/usr/bin/env python3
"""Re-draw the poster charts from a real benchmark JSON.

The submitted poster charts (`poster_assets/chart_*.png`) could not be traced to
any run: their numbers (IQM 82.4 for HDML, CPU 5.38 ms, P(HDML>...) 96.8%) do not
appear in any results log, and they contradict the main report. This script rebuilds
the four charts directly from a benchmark JSON produced by
`scripts/benchmark_baselines.py`, so every bar is traceable to a logged run.

Outputs (into --out-dir):
    chart_iqm.png      IQM with 95% stratified-bootstrap CI
    chart_jitter.png   action smoothness (mean |Δ²a|), lower is smoother
    chart_latency.png  per-model inference latency, 40 Hz threshold
    chart_rliable.png  probability of improvement P(HDML > baseline)

Usage:
    python scripts/make_poster_charts.py \
        --benchmark results/rebuild_final/benchmark_halfcheetah-v5.json \
        --out-dir   poster_assets_from_logs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hdml.utils.rliable_metrics import compute_iqm, stratified_bootstrap_ci, compute_probability_of_improvement

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

SHORT = {
    "HDML (Decision Mamba + Liquid CfC - Ours)": "HDML",
    "Diffusion Policy (DDPM 10-step Denoising)": "Diff.",
    "Decision Transformer (Causal Attention DT)": "DT",
    "Implicit Q-Learning (IQL / Value-Advantage)": "IQL",
    "Decision RNN (LSTM Recurrent Policy)": "D-RNN",
    "MLP-BC (Standard Feedforward Reactive)": "BC",
}


def _iqm(scores: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(scores, dtype=np.float64)
    point, lo, hi = stratified_bootstrap_ci(arr, stat_fn=compute_iqm)
    return float(point), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out-dir", default="poster_assets_from_logs")
    ap.add_argument("--env-label", default=None)
    args = ap.parse_args()

    data = json.load(open(args.benchmark, encoding="utf-8"))
    std = data["standard"]
    env = args.env_label or Path(args.benchmark).stem.replace("benchmark_", "")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = [SHORT.get(s["name"], s["name"]) for s in std]
    colors = ["#0E9F6E", "#2563EB", "#6B7280", "#8B5CF6", "#F59E0B", "#F43F5E"]

    # 1. IQM with 95% CI -----------------------------------------------------
    iqm, lo, hi = zip(*[_iqm(s["norm_scores_array"]) for s in std])
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    bars = ax.bar(names, iqm, color=colors[: len(names)], edgecolor="black", linewidth=0.8)
    ax.errorbar(range(len(names)), iqm, yerr=[np.array(iqm) - np.array(lo), np.array(hi) - np.array(iqm)],
                fmt="none", ecolor="black", capsize=6, lw=1.6)
    for i, v in enumerate(iqm):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontweight="bold")
    ax.set_title(f"Điểm chuẩn IQM ({env}) — từ log thật", fontweight="bold", fontsize=15)
    ax.set_ylabel("IQM (thang chuẩn hoá)", fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "chart_iqm.png"); plt.close(fig)

    # 2. Jitter --------------------------------------------------------------
    jerk = [s["mean_smoothness_jerk"] for s in std]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.bar(names, jerk, color=colors[: len(names)], edgecolor="black", linewidth=0.8)
    for i, v in enumerate(jerk):
        ax.text(i, v + max(jerk) * 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_title(f"Độ trơn mô-men (Jerk, càng thấp càng mịn) — {env}", fontweight="bold", fontsize=15)
    ax.set_ylabel("J = mean |Δ²a|", fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "chart_jitter.png"); plt.close(fig)

    # 3. Latency -------------------------------------------------------------
    lat = [s["mean_latency_ms"] for s in std]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.bar(names, lat, color=colors[: len(names)], edgecolor="black", linewidth=0.8)
    ax.axhline(25.0, ls="--", color="red", label="Hạn mức 40 Hz (25 ms)")
    for i, v in enumerate(lat):
        ax.text(i, v + max(lat) * 0.01, f"{v:.2f} ms\n({1000.0 / v:.0f} Hz)", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_title(f"Độ trễ suy luận mô hình trên GPU — {env}", fontweight="bold", fontsize=15)
    ax.set_ylabel("Thời gian (ms)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "chart_latency.png"); plt.close(fig)

    # 4. Probability of improvement (HDML vs each baseline) ------------------
    hdml = np.asarray(std[0]["norm_scores_array"], dtype=np.float64)
    labels, pois = [], []
    for s in std[1:]:
        p, _lo, _hi = compute_probability_of_improvement(hdml, np.asarray(s["norm_scores_array"], dtype=np.float64))
        labels.append(f"vs {SHORT.get(s['name'], s['name'])}")
        pois.append(float(p))
    order = np.argsort(pois)[::-1]
    labels = [labels[i] for i in order]
    pois = [pois[i] for i in order]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.bar(labels, [p * 100 for p in pois], color="#0E9F6E", edgecolor="black", linewidth=0.8)
    for i, v in enumerate(pois):
        ax.text(i, v * 100 + 1.5, f"{v * 100:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_title(f"Xác suất cải thiện P(HDML > baseline) — {env}", fontweight="bold", fontsize=15)
    ax.set_ylabel("Xác suất (%)", fontweight="bold")
    ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "chart_rliable.png"); plt.close(fig)

    print(f"Wrote 4 charts to {out}")
    for i, s in enumerate(std):
        print(f"  {names[i]:6s} IQM={iqm[i]:.2f} [{lo[i]:.2f},{hi[i]:.2f}]  jerk={jerk[i]:.4f}  lat={lat[i]:.2f}ms")


if __name__ == "__main__":
    main()
