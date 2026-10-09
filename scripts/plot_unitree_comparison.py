import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "figure.titlesize": 12,
})

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=300)

models = ["HDML (Ours)", "Decision\nTransformer", "Reactive\nMLP-BC"]
x = np.arange(len(models))
width = 0.32

# (a) Returns: Clean vs Perturbed (50N kick)
clean_returns = [1494.93, 1415.24, 997.72]
pert_returns = [1362.00, 1400.46, 1023.45]
pert_errors = [21.96, 40.10, 38.16]

ax0 = axes[0]
rects1 = ax0.bar(x - width/2, clean_returns, width, label="Dieu kien chuan", color="#1f77b4", alpha=0.9, edgecolor="black", linewidth=0.8)
rects2 = ax0.bar(x + width/2, pert_returns, width, yerr=pert_errors, capsize=4, label="Va dap 50N + IMU", color="#ff7f0e", alpha=0.9, edgecolor="black", linewidth=0.8)
ax0.set_ylabel("Diem thuong (Cumulative Return)")
ax0.set_title("(a) Diem thuong van hanh vong kin")
ax0.set_xticks(x)
ax0.set_xticklabels(models)
ax0.set_ylim(800, 1600)
ax0.legend(loc="lower left", framealpha=0.9)
ax0.grid(axis="y", linestyle="--", alpha=0.5)

# (b) Jerk: Clean vs Perturbed
clean_jerk = [0.0628, 0.0756, 0.0022]
pert_jerk = [0.2343, 0.1363, 0.3310]

ax1 = axes[1]
ax1.bar(x - width/2, clean_jerk, width, label="Dieu kien chuan", color="#2ca02c", alpha=0.9, edgecolor="black", linewidth=0.8)
ax1.bar(x + width/2, pert_jerk, width, label="Va dap 50N + IMU", color="#d62728", alpha=0.9, edgecolor="black", linewidth=0.8)
ax1.set_ylabel(r"Chi so rung giat 12 dong co $\mathcal{J}$")
ax1.set_title(r"(b) Do rung giat luc xoan (Jerk)")
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend(loc="upper left", framealpha=0.9)
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# (c) High Impact Limits (Forces: 50N, 100N, 200N, 400N)
forces = [50, 100, 200, 400]
peak_rolls = [7.0, 8.0, 25.0, 51.1]
survivals = [100, 100, 100, 0]

ax2 = axes[2]
line1 = ax2.plot(forces, peak_rolls, "o-", color="#1f77b4", linewidth=1.8, markersize=6, label=r"Goc nghieng cuc dai $\Delta \phi_{\max}$ (do)")
ax2.set_xlabel("Xung luc va dap ngang (N)")
ax2.set_ylabel(r"Goc lech than cuc dai ($^{\circ}$)", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax2.set_ylim(0, 60)
ax2.grid(True, linestyle="--", alpha=0.5)

ax2_twin = ax2.twinx()
line2 = ax2_twin.plot(forces, survivals, "s--", color="#d62728", linewidth=1.5, markersize=5, label="Ty le song sot (%)")
ax2_twin.set_ylabel("Ty le song sot (%)", color="#d62728")
ax2_twin.tick_params(axis="y", labelcolor="#d62728")
ax2_twin.set_ylim(-10, 115)

# Annotation for tip-over at 400N
ax2.annotate("Lat do tai 400N\n(qua tai dong hoc)", xy=(400, 51.1), xytext=(240, 45),
             arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
             fontsize=8.5, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff2f2", edgecolor="#d62728", alpha=0.9))

ax2.set_title("(c) Nguong chiu luc va dap ngang HDML")

plt.tight_layout()
plt.savefig("KH_thi_KHKT_TP_2026-2027/Bao_cao_du_an_LaTeX/figures/unitree_a1_benchmark_comparison.pdf", bbox_inches="tight")
plt.savefig("KH_thi_KHKT_TP_2026-2027/Bao_cao_du_an_LaTeX/figures/unitree_a1_benchmark_comparison.png", bbox_inches="tight")
print("Saved comparison figure successfully!")
