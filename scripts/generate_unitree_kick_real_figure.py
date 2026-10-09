#!/usr/bin/env python3
"""Build the Unitree A1 kick figure from *real* measurements only.

Top row: real rendered frames captured while running the trained HDML policy
through the kick (script ``evaluate_unitree_kick_real.py``).
Bottom row: real torso-roll trajectories and aggregate return / peak-roll
curves for HDML and the trained Decision-Transformer baseline, all read from
``results/unitree_kick_real.json``. No hand-drawn baselines, no fabricated
impulses, no unverified mechanism labels.
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

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
JSON = ROOT / "results" / "unitree_kick_real.json"
FRAMES = ROOT / "KH_thi_KHKT_TP_2026-2027" / "Bao_cao_du_an_LaTeX" / "figures" / "unitree_kick_frames"
OUT_PNG = ROOT / "KH_thi_KHKT_TP_2026-2027" / "Bao_cao_du_an_LaTeX" / "figures" / "unitree_a1_kick_recovery_real.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")

FRAME_STEPS = [196, 200, 205, 225]
FRAME_LABELS = [
    "(a) Trước va chạm\n(dáng đi định mức)",
    "(b) Thời điểm va chạm\n(xung lực ngang)",
    "(c) $+0{,}1$ giây\n(hồi phục)",
    "(d) $+0{,}5$ giây\n(chạy ổn định)",
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

    fig = plt.figure(figsize=(15.0, 6.6), dpi=250)
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.05, 0.95], hspace=0.32, wspace=0.24)

    # ---- Top: real frames (4 columns) ----
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0, :], wspace=0.04)
    for idx, (step, label) in enumerate(zip(FRAME_STEPS, FRAME_LABELS)):
        ax = fig.add_subplot(gs_top[0, idx])
        fp = FRAMES / f"hdml_f50_t{step:04d}.png"
        if fp.exists():
            im = np.asarray(Image.open(fp))
            ax.imshow(im)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        ax.set_title(label, fontsize=9.6, fontweight="bold", pad=5, color="#111827")
        ax.axis("off")

    # ---- Bottom-left: real roll trajectories at the highest surviving force ----
    ax1 = fig.add_subplot(gs[1, 0])
    kick_step = int(data["kick_step"])
    f_mid = 200.0
    for res, color, name, ls in ((hdml, HDML_COLOR, "HDML", "-"), (dt, DT_COLOR, "Decision Transformer", "--")):
        rec = pick(res, f_mid)["episodes"][0]
        rolls = np.asarray(rec["rolls"], dtype=np.float64)
        t = np.arange(len(rolls)) * 0.02
        ax1.plot(t, rolls, color=color, linewidth=2.4, linestyle=ls, label=name)
    ax1.axvline(kick_step * 0.02, color="#6b7280", linestyle=":", linewidth=1.2)
    ax1.text(kick_step * 0.02, ax1.get_ylim()[1] * 0.94, " va chạm", fontsize=8, color="#374151")
    ax1.set_xlabel("Thời gian $t$ (giây)", fontsize=9.3, fontweight="bold", color="#1f2937")
    ax1.set_ylabel("Góc lệch thân $\\phi(t)$ (độ)", fontsize=9.3, fontweight="bold", color="#1f2937")
    ax1.set_title(f"Góc lệch thân thật (xung lực {int(f_mid)})", fontsize=10.3, fontweight="bold", pad=5)
    ax1.legend(loc="upper left", fontsize=8.2, frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ---- Bottom-middle: real return vs force ----
    ax2 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(forces))
    w = 0.38
    hr = [pick(hdml, f)["return_mean"] for f in forces]
    dr = [pick(dt, f)["return_mean"] for f in forces]
    ax2.bar(x - w / 2, hr, w, color=HDML_COLOR, label="HDML")
    ax2.bar(x + w / 2, dr, w, color=DT_COLOR, label="Decision Transformer")
    for xi, v in zip(x, hr):
        ax2.text(xi - w / 2, v + 25, f"{v:.0f}", ha="center", fontsize=7.6, color="#1e3a8a")
    for xi, v in zip(x, dr):
        ax2.text(xi + w / 2, v + 25, f"{v:.0f}", ha="center", fontsize=7.6, color="#7f1d1d")
    ax2.set_xticks(x); ax2.set_xticklabels([str(int(f)) for f in forces])
    ax2.set_xlabel("Xung lực ngang (đơn vị chưa hiệu chuẩn)", fontsize=9.0, fontweight="bold", color="#1f2937")
    ax2.set_ylabel("Điểm thưởng trung bình", fontsize=9.3, fontweight="bold", color="#1f2937")
    ax2.set_title("Điểm thưởng theo mức xung lực (3 episode)", fontsize=10.3, fontweight="bold", pad=5)
    ax2.legend(loc="upper right", fontsize=8.2)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    # ---- Bottom-right: real peak roll vs force with survival marker ----
    ax3 = fig.add_subplot(gs[1, 2])
    hp = [pick(hdml, f)["peak_roll_deg"] for f in forces]
    dp = [pick(dt, f)["peak_roll_deg"] for f in forces]
    hs = [pick(hdml, f)["survival_rate"] for f in forces]
    ds = [pick(dt, f)["survival_rate"] for f in forces]
    ax3.plot(x, hp, "o-", color=HDML_COLOR, linewidth=2.2, markersize=6, label="HDML")
    ax3.plot(x, dp, "s--", color=DT_COLOR, linewidth=2.2, markersize=6, label="Decision Transformer")
    for xi, (sh, sd) in zip(x, zip(hs, ds)):
        if sh < 0.5:
            ax3.text(xi, hp[list(x).index(xi)] + 2.5, "ngã", ha="center", fontsize=8, color=HDML_COLOR, fontweight="bold")
        if sd < 0.5:
            ax3.text(xi, dp[list(x).index(xi)] - 4.5, "ngã", ha="center", fontsize=8, color=DT_COLOR, fontweight="bold")
    ax3.axhline(30.0, color="#9ca3af", linestyle=":", linewidth=1.0)
    ax3.set_xticks(x); ax3.set_xticklabels([str(int(f)) for f in forces])
    ax3.set_xlabel("Xung lực ngang (đơn vị chưa hiệu chuẩn)", fontsize=9.0, fontweight="bold", color="#1f2937")
    ax3.set_ylabel("Góc lệch đỉnh $\\phi_{\\max}$ (độ)", fontsize=9.3, fontweight="bold", color="#1f2937")
    ax3.set_title("Góc lệch đỉnh và ngưỡng ngã", fontsize=10.3, fontweight="bold", pad=5)
    ax3.legend(loc="upper left", fontsize=8.2)
    ax3.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Phản ứng thân robot Unitree A1 dưới xung lực xô ngang — đo trực tiếp trong MuJoCo",
        fontsize=12.5, fontweight="bold", y=0.99,
    )
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=250, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT_PNG)
    print("saved", OUT_PDF)


if __name__ == "__main__":
    main()
