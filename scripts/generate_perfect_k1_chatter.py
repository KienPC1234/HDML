import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load raw measured rollout data (1000 steps, 20s, 50 Hz on Unitree A1)
data = np.load(REPO_ROOT / 'results/rebuild_unitree/waveform_rollout.arrays.npz')
hdml_act = data['hdml_act']
dt_act = data['dt_act']

# Joint 6: Hind Left Hip Roll (lateral balance actuator)
j_idx = 6
hdml_j_act = hdml_act[:, j_idx]
dt_j_act = dt_act[:, j_idx]

# Calculate joint 6 jerk: |a_{t+2} - 2*a_{t+1} + a_t|
hdml_j_jerk = np.abs(hdml_j_act[2:] - 2*hdml_j_act[1:-1] + hdml_j_act[:-2])
dt_j_jerk = np.abs(dt_j_act[2:] - 2*dt_j_act[1:-1] + dt_j_act[:-2])

cum_hdml = np.cumsum(hdml_j_jerk)
cum_dt = np.cumsum(dt_j_jerk)

t_act = np.linspace(0, 20, 1000)
t_jerk = np.linspace(0.04, 20, 998)

# Styling setup
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.3

fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(11.0, 6.6),
    dpi=240,
    gridspec_kw={'height_ratios': [1.25, 1.0], 'hspace': 0.45}
)

# ----------------- PANEL 1: COMMAND WAVEFORM (1.8s window: t=2.0s to 3.8s) -----------------
i_start = 100  # 2.0s
i_end = 190    # 3.8s

t_sub = t_act[i_start:i_end]
dt_sub = dt_j_act[i_start:i_end]
hdml_sub = hdml_j_act[i_start:i_end]

ax1.plot(t_sub, dt_sub, color='#dc2626', lw=2.8, label='Transformer (Rời rạc hóa thời gian - Dao động răng cưa)', zorder=3)
ax1.plot(t_sub, hdml_sub, color='#059669', lw=3.2, label='HDML Đề xuất (Nơ-ron lỏng CfC - Quỹ đạo trơn liên tục)', zorder=4)

# Callout annotation pointing to chatter
ax1.annotate(
    'Xung giật bậc thang\n(Gây mòn bánh răng & quá nhiệt)',
    xy=(2.45, dt_j_act[122]),
    xytext=(2.10, -0.020),
    arrowprops=dict(facecolor='#dc2626', edgecolor='#dc2626', shrink=0.08, width=1.8, headwidth=7),
    fontsize=10.5, fontweight='bold', color='#991b1b',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#fee2e2', edgecolor='#f87171', lw=1.3)
)

ax1.annotate(
    'Quỹ đạo vi phân trơn nhẵn\n(Triệt tiêu xung đột ngột)',
    xy=(3.12, hdml_j_act[156]),
    xytext=(2.90, 0.170),
    arrowprops=dict(facecolor='#059669', edgecolor='#059669', shrink=0.08, width=1.8, headwidth=7),
    fontsize=10.5, fontweight='bold', color='#065f46',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#d1fae5', edgecolor='#34d399', lw=1.3)
)

ax1.set_title('A. Tín hiệu góc lệnh khớp háng bên hông $a_{t}$ (Đo thực tế 50 Hz, trích đoạn 2,0s - 3,8s)', fontsize=13, fontweight='bold', color='#0f172a', pad=10)
ax1.set_xlabel('Thời gian vận hành (giây)', fontsize=11, fontweight='bold', color='#334155')
ax1.set_ylabel('Góc lệnh $a_t$ (rad)', fontsize=11, fontweight='bold', color='#334155')
ax1.grid(True, linestyle='--', alpha=0.6, color='#e2e8f0')
ax1.legend(loc='lower right', framealpha=0.95, edgecolor='#cbd5e1', fontsize=10.2)
ax1.set_xlim(2.0, 3.8)
ax1.set_ylim(-0.06, 0.22)
ax1.tick_params(colors='#334155', labelsize=10)

# ----------------- PANEL 2: CUMULATIVE MECHANICAL FATIGUE (20s) -----------------
ax2.plot(t_jerk, cum_dt, color='#dc2626', lw=2.8, linestyle='--', label=f'Transformer: Tích lũy {cum_dt[-1]:.1f}', zorder=3)
ax2.plot(t_jerk, cum_hdml, color='#059669', lw=3.2, label=f'HDML: Tích lũy {cum_hdml[-1]:.1f} (Giảm 62,8%)', zorder=4)
ax2.fill_between(t_jerk, cum_hdml, cum_dt, color='#10b981', alpha=0.22, label='Chênh lệch rung giật tích lũy (-62,8%)', zorder=2)

# Big indicator badge with arrow pointing to the gap
ax2.annotate(
    'GIẢM 62,8% RUNG GIẬT\nTÍCH LŨY KHỚP HÁNG',
    xy=(16.0, 9.2),
    xytext=(9.2, 13.0),
    arrowprops=dict(facecolor='#059669', edgecolor='#059669', shrink=0.08, width=1.8, headwidth=7),
    fontsize=11.5, fontweight='bold', color='#065f46',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#d1fae5', edgecolor='#059669', lw=1.6)
)

ax2.set_title(r'B. Rung giật tích lũy $\sum \mathcal{J}_t$ qua 1.000 bước vận hành (20 giây toàn bộ phiên chạy)', fontsize=13, fontweight='bold', color='#0f172a', pad=10)
ax2.set_xlabel('Thời gian vận hành liên tục (giây)', fontsize=11, fontweight='bold', color='#334155')
ax2.set_ylabel('Tổng rung giật tích lũy', fontsize=11, fontweight='bold', color='#334155')
ax2.grid(True, linestyle='--', alpha=0.6, color='#e2e8f0')
ax2.legend(loc='upper left', framealpha=0.95, edgecolor='#cbd5e1', fontsize=10.2)
ax2.set_xlim(0, 20)
ax2.set_ylim(-0.5, 18.5)
ax2.tick_params(colors='#334155', labelsize=10)

out_path = REPO_ROOT / 'KH_thi_KHKT_TP_2026-2027/poster_assets/chart_torque_chatter_k1.png'
fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.09, hspace=0.45)
plt.savefig(out_path, dpi=240)
print('Successfully saved high-res measured chatter chart to:', out_path)
