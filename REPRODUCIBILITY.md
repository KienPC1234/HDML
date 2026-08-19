# HDML: Hướng Dẫn Tái Lập Kết Quả Toàn Diện (Full Reproducibility Guide)

Tài liệu này cung cấp hướng dẫn chi tiết từng bước (step-by-step) để bất kỳ nhà nghiên cứu, kỹ sư hay sinh viên nào cũng có thể **tái lập 100% tất cả các kết quả thực nghiệm, đồ thị benchmark, mô hình ONNX và video mô phỏng** trong bài báo/dự án HDML.

---

## 1. Yêu Cầu Hệ Thống & Cài Đặt Môi Trường

### 1.1. Yêu cầu phần cứng
- **Hệ điều hành:** Linux (Ubuntu 20.04/22.04/24.04 khuyến nghị) hoặc WSL2.
- **GPU:** NVIDIA GPU (khuyến nghị RTX 3060/3070/4070/4090 với VRAM $\ge 8\text{ GB}$). Có hỗ trợ chế độ CPU.
- **RAM:** Tối thiểu 16 GB.

### 1.2. Các bước cài đặt từ đầu
```bash
# 1. Clone repository
git clone https://github.com/KienPC1234/HDML.git
cd HDML

# 2. Khởi tạo môi trường ảo Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Cài đặt PyTorch với CUDA (ví dụ CUDA 12.1 hoặc 13.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Cài đặt các thư viện Mamba SSM & Causal-Conv1D
export CUDA_HOME="/usr/local/cuda-13.2"  # Hoặc đường dẫn CUDA của bạn
export TORCH_CUDA_ARCH_LIST="8.9"       # 8.9 cho RTX 40-series, 8.6 cho RTX 30-series
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm>=2.0.0 --no-build-isolation

# 5. Cài đặt toàn bộ dependencies còn lại & package ở chế độ editable
pip install -r requirements.txt
pip install -e .
```

---

## 2. Kiểm Thử Hệ Thống Tự Động (Smoke Test)

Trước khi chạy train, chạy toàn bộ 45 unit & integration tests để đảm bảo phần cứng và thư viện hoạt động 100%:
```bash
pytest tests/ -v
# Kết quả mong đợi: 45 passed, 0 failed (100% PASS)
```

---

## 3. Tái Lập Thực Nghiệm 1: Học Tự Giám Sát Giải Mê Cung (Zero-Label Cognition)

### 3.1. AntBot 4 chân 8 khớp trên Mê Cung Đa Phòng Khổng Lồ (`AntMaze Medium`)
Tập dữ liệu tự động tải từ Farama Minari (`D4RL/antmaze/medium-play-v2`).

```bash
# Bước 1: Huấn luyện tự giám sát (Zero-Label World Dynamics + Hindsight Subgoal HER)
python scripts/train_offline.py \
    --config configs/antmaze_medium_unsupervised.yaml \
    --dataset D4RL/antmaze/medium-play-v2 \
    --max-episodes 2000 \
    --her-prob 0.8 \
    --epochs 15 \
    --batch-size 256 \
    --stride 2 \
    --device cuda

# Bước 2: Chạy closed-loop rollout và xuất Video 3 Camera Đồng Bộ (Tri-Camera Multi-View)
python scripts/record_antmaze_navigation_video.py \
    --config configs/antmaze_medium_unsupervised.yaml \
    --checkpoint checkpoints/antmaze_medium/best_model.pt \
    --dataset D4RL/antmaze/medium-play-v2 \
    --output-gif videos/antmaze_medium_multiview_solved.gif \
    --output-mp4 videos/antmaze_medium_multiview_solved.mp4 \
    --cam-distance 27.0 \
    --steps 420 \
    --seed 31 \
    --device cuda
```
- **Kết quả thu được:**
  - File video: `videos/antmaze_medium_multiview_solved.gif` (gồm 3 góc: Toàn cảnh trên cao, Cận cảnh chân robot, Phối cảnh 3D nghiêng).
  - Số bước chạm đích: $\sim 212$ steps ($\text{Dist} < 1.0\text{ m}$).
  - Độ mượt điều khiển (Jerk): $\approx 0.3070$.

---

### 3.2. PointMaze (Mê Cung 2D: U-Maze & Medium)
```bash
# Huấn luyện và xuất video PointMaze Medium
python scripts/train_offline.py \
    --config configs/pointmaze_medium_unsupervised.yaml \
    --dataset D4RL/pointmaze/medium-v2 \
    --max-episodes 1000 \
    --her-prob 0.8 \
    --epochs 10 \
    --device cuda

python scripts/record_maze_navigation_video.py \
    --config configs/pointmaze_medium_unsupervised.yaml \
    --checkpoint checkpoints/pointmaze_medium/best_model.pt \
    --dataset D4RL/pointmaze/medium-v2 \
    --output-gif videos/pointmaze_medium_hdml_solved.gif \
    --output-mp4 videos/pointmaze_medium_hdml_solved.mp4 \
    --steps 300 \
    --seed 7 \
    --device cuda
```

---

## 4. Tái Lập Thực Nghiệm 2: Benchmark So Sánh Với Các Baseline SOTA (NeurIPS `rliable`)

Thực nghiệm đánh giá thống kê nghiêm ngặt theo giao thức NeurIPS 2021 `rliable` trên `HalfCheetah-v5` với 2.000 bootstrap resamples.

```bash
# Bước 1: Thu thập tập dữ liệu chuyên gia
python scripts/collect_data.py \
    --env HalfCheetah-v5 \
    --num-episodes 50 \
    --output data/halfcheetah_v5_expert.npz

# Bước 2: Huấn luyện HDML và toàn bộ 5 baseline (DT, Decision RNN, IQL, Diffusion, MLP-BC)
python scripts/train_baselines.py \
    --config configs/halfcheetah_v5_default.yaml \
    --dataset data/halfcheetah_v5_expert.npz \
    --model all \
    --epochs 20

# Bước 3: Đánh giá phân tầng, tính IQM, xác suất vượt trội và xuất đồ thị
python scripts/benchmark_ablations.py \
    --dataset data/halfcheetah_v5_expert.npz \
    --episodes 10 \
    --device cuda
```
- **Kết quả thu được:**
  - Bảng số liệu: `results/benchmark_halfcheetah-v5.txt`
  - Đồ thị IQM & Performance Profile: `plots/rliable_halfcheetah-v5_benchmark.png`
  - Biểu đồ dạng sóng dao động động cơ: `plots/action_waveforms.png`

---

## 5. Tái Lập Thực Nghiệm 3: Mô Phỏng Robot Chó 12-DOF Bị Tác Động Ngoại Lực (Unitree A1)

Đánh giá khả năng dập tắt dao động và phục hồi thăng bằng của tầng Liquid CfC khi chịu 3 cú đá ngoại lực liên tiếp ($+8\text{ N}, -8\text{ N}, +10\text{ N}$):

```bash
# Thu thập dữ liệu dáng đi A1 (trotting)
python scripts/collect_unitree_a1.py

# Huấn luyện HDML
python scripts/train_offline.py \
    --config configs/unitree_a1_default.yaml \
    --dataset data/unitree_a1_trajectories.npz \
    --stride 4 \
    --epochs 40 \
    --batch-size 256 \
    --device cuda

# Chạy mô phỏng kiểm tra va chạm và xuất video HUD
python scripts/record_robot_dog_kick_hud.py \
    --config configs/unitree_a1_default.yaml \
    --checkpoint checkpoints/unitree_a1/best_model.pt \
    --output-mp4 videos/unitree_a1_robot_dog_kick_recovery.mp4 \
    --device cuda
```
- **Kết quả thu được:** Video `videos/unitree_a1_robot_dog_kick_recovery.mp4` hiển thị chi tiết phản ứng mô-men xoắn bù trừ của 12 khớp chân và biểu đồ lực đá theo thời gian thực.

---

## 6. Tái Lập Thực Nghiệm 4: Xuất Bản ONNX & Kiểm Tra Hiệu Năng Trên CPU

Xuất mô hình PyTorch sang định dạng ONNX tiêu chuẩn và kiểm tra sai số số học:

```bash
python scripts/export_onnx.py \
    --config configs/antmaze_medium_unsupervised.yaml \
    --checkpoint checkpoints/antmaze_medium/best_model.pt \
    --output deployment/hdml_antmaze_medium_policy.onnx
```

- **Tiêu chuẩn kiểm định thành công:**
  - File ONNX sinh ra: `deployment/hdml_antmaze_medium_policy.onnx` ($\approx 4.70\text{ MB}$).
  - Sai số số học tối đa giữa PyTorch và ONNX Runtime: $\|\mathbf{y}_{\text{torch}} - \mathbf{y}_{\text{onnx}}\|_\infty \le 1.0 \times 10^{-6}$.
  - Tần số suy luận trên 1 nhân CPU: $\ge 150\text{ Hz}$ ($\le 6.5\text{ ms}$/step).

---

## 7. Khắc Phục Lỗi Thường Gặp (Troubleshooting)

1. **Lỗi `MUJOCO_GL` trên máy chủ không có màn hình (Headless Server):**
   - Đảm bảo đã khai báo `export MUJOCO_GL="egl"` trước khi chạy các script sinh video.
2. **Lỗi GPU Out of Memory (OOM):**
   - Giảm `--batch-size` từ 256 xuống 128 hoặc 64 trong lệnh train (không ảnh hưởng tới độ hội tụ của mô hình).
3. **Cài đặt Mamba SSM thất bại:**
   - Đảm bảo biến môi trường `CUDA_HOME` trỏ chính xác đến thư mục cài đặt CUDA Toolkit và cờ `--no-build-isolation` được sử dụng khi chạy `pip install mamba-ssm`.
