# HDML: Complete Step-by-Step Reproducibility Guide

This document provides a comprehensive, deterministic, and fully verifiable step-by-step protocol to reproduce all empirical benchmark results, unsupervised cognitive labyrinth navigation experiments, ONNX edge deployments, and simulation artifacts described in the HDML paper and repository.

---

## 1. System Requirements & Environment Setup

### 1.1. Hardware & Operating System Specifications
- **Operating System:** Linux (Ubuntu 20.04 / 22.04 / 24.04 LTS or WSL2 with NVIDIA CUDA support).
- **Target GPU:** NVIDIA GeForce RTX 4070 SUPER / RTX 3080 / RTX 4090 (Compute Capability $\ge 8.0$, VRAM $\ge 8\text{ GB}$). A CPU-only mode is also supported for evaluation and ONNX inference.
- **RAM:** Minimum 16 GB system memory.

### 1.2. Environment Installation from Scratch
```bash
# 1. Clone the repository
git clone https://github.com/KienPC1234/HDML.git
cd HDML

# 2. Create and activate Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch with CUDA Toolkit support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Compile and install Mamba-SSM and Causal-Conv1D with native CUDA headers
export CUDA_HOME="/usr/local/cuda-13.2"  # Update to your local CUDA installation path if different
export TORCH_CUDA_ARCH_LIST="8.9"       # Use 8.9 for Ada Lovelace (RTX 40-series), 8.6 for Ampere (RTX 30-series)
export MAX_JOBS=8

pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm>=2.0.0 --no-build-isolation

# 5. Install all remaining dependencies and install hdml in editable mode
pip install -r requirements.txt
pip install -e .
```

---

## 2. Automated Smoke Test & System Verification

Verify environment correctness, GPU kernel compilation, and cross-modal gradient flow across all 45 automated unit and integration tests:

```bash
pytest tests/ -v
```
**Expected Verification Output:**
```text
======================= 45 passed in 20.19s =======================
```

---

## 3. Experiment 1: Unsupervised Spatial Navigation & Labyrinth Solving (Zero-Label Pre-training)

HDML learns latent cognitive spatial representations and closed-loop motor control on **PointMaze** and multi-articulated **AntBot (8-DOF)** without human demonstrations or external reward engineering.

### 3.1. AntBot 8-DOF Multi-Room Quadruped Labyrinth (`AntMaze Medium`)
The dataset is automatically fetched and cached from the official Farama Minari repository (`D4RL/antmaze/medium-play-v2`, 2,000 trajectories across an 8x8 interconnected labyrinth).

```bash
# Step 1: Self-Supervised Offline Training (Forward Dynamics Modeling + Hindsight Goal Relabeling)
python scripts/train_offline.py \
    --config configs/antmaze_medium_unsupervised.yaml \
    --dataset D4RL/antmaze/medium-play-v2 \
    --max-episodes 2000 \
    --her-prob 0.8 \
    --epochs 15 \
    --batch-size 256 \
    --stride 2 \
    --device cuda

# Step 2: Evaluate closed-loop goal reaching rate across seeds
python scripts/evaluate_maze_cognition.py \
    --config configs/antmaze_medium_unsupervised.yaml \
    --checkpoint checkpoints/antmaze_medium/best_model.pt \
    --dataset D4RL/antmaze/medium-play-v2 \
    --episodes 10 \
    --device cuda

# Step 3: Record Tri-Camera Multi-View Telemetry Video (Top Overview + Close-up Gait + Isometric 3D)
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

**Key Verified Metrics:**
- **Validation Loss:** $\le 0.2070$
- **Goal Reached:** Reached target destination ($< 1.0\text{ m}$) at step **212**
- **Kinematic Smoothness (Jerk $\text{mean}|\Delta^2 a_t|$):** $0.3070$
- **Video Artifacts:** `videos/antmaze_medium_multiview_solved.gif` and `.mp4`

---

### 3.2. PointMaze 2D Cognition (`PointMaze Medium & U-Maze`)

```bash
# Train on PointMaze Medium (1,000 unlabelled trajectories)
python scripts/train_offline.py \
    --config configs/pointmaze_medium_unsupervised.yaml \
    --dataset D4RL/pointmaze/medium-v2 \
    --max-episodes 1000 \
    --her-prob 0.8 \
    --epochs 10 \
    --device cuda

# Record closed-loop navigation rollout
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

## 4. Experiment 2: Baseline Comparison & NeurIPS 2021 `rliable` Benchmark

Rigorous evaluation across 2,000 stratified bootstrap resamples with 95% confidence intervals against 5 baseline architectures: **Decision Transformer (DT)**, **Decision RNN**, **Implicit Q-Learning (IQL)**, **Diffusion Policy**, and **MLP-BC**.

```bash
# Step 1: Collect standard expert dataset
python scripts/collect_data.py \
    --env HalfCheetah-v5 \
    --num-episodes 50 \
    --output data/halfcheetah_v5_expert.npz

# Step 2: Train all baselines under identical offline dataset & optimization budgets
python scripts/train_baselines.py \
    --config configs/halfcheetah_v5_default.yaml \
    --dataset data/halfcheetah_v5_expert.npz \
    --model all \
    --epochs 20

# Step 3: Run comprehensive ablation and benchmark evaluation with rliable statistical metrics
python scripts/benchmark_ablations.py \
    --dataset data/halfcheetah_v5_expert.npz \
    --episodes 10 \
    --device cuda
```

**Generated Verification Artifacts:**
- Numerical Table: `results/benchmark_halfcheetah-v5.txt`
- Statistical Metric Plots: `plots/rliable_halfcheetah-v5_benchmark.png`
- Motor Chattering Waveforms: `plots/action_waveforms.png`

---

## 5. Experiment 3: Unitree A1 12-DOF Quadruped Perturbation Recovery

Evaluate the closed-loop recovery capabilities of the Liquid CfC ODE layer when subjected to 3 consecutive lateral impulse kicks ($+8\text{ N}, -8\text{ N}, +10\text{ N}$):

```bash
# Step 1: Collect Unitree A1 locomotion trajectories
python scripts/collect_unitree_a1.py

# Step 2: Train HDML on Unitree A1 quadruped dataset
python scripts/train_offline.py \
    --config configs/unitree_a1_default.yaml \
    --dataset data/unitree_a1_trajectories.npz \
    --stride 4 \
    --epochs 40 \
    --batch-size 256 \
    --device cuda

# Step 3: Run closed-loop perturbation simulation and record HUD video
python scripts/record_robot_dog_kick_hud.py \
    --config configs/unitree_a1_default.yaml \
    --checkpoint checkpoints/unitree_a1/best_model.pt \
    --output-mp4 videos/unitree_a1_robot_dog_kick_recovery.mp4 \
    --device cuda
```

---

## 6. Experiment 4: High-Throughput ONNX Edge Deployment & CPU Parity Benchmark

Export the trained PyTorch policy to portable ONNX format and verify numerical parity against ONNX Runtime:

```bash
python scripts/export_onnx.py \
    --config configs/antmaze_medium_unsupervised.yaml \
    --checkpoint checkpoints/antmaze_medium/best_model.pt \
    --output deployment/hdml_antmaze_medium_policy.onnx
```

**Verification Acceptance Criteria:**
- Output Model File: `deployment/hdml_antmaze_medium_policy.onnx` ($4.70\text{ MB}$)
- Numerical Parity: $\max |\mathbf{y}_{\text{torch}} - \mathbf{y}_{\text{onnx}}| \le 1.0 \times 10^{-6}$ (**Verified: $5.66 \times 10^{-7}$**)
- CPU Inference Frequency: $\ge 150\text{ Hz}$ on single CPU core (**Verified: $186\text{ Hz}$ / $5.39\text{ ms}$**)

---

## 7. Troubleshooting & FAQ

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **`MUJOCO_GL` Headless Error** | Headless server lacks X11 display context | Set `export MUJOCO_GL="egl"` before invoking video rendering scripts. |
| **CUDA Out of Memory (OOM)** | Large batch size on lower-VRAM GPUs | Decrease `--batch-size 256` to `--batch-size 128` or `64`. |
| **Mamba Compilation Failure** | Missing native CUDA compiler headers | Ensure `CUDA_HOME` is exported and use `--no-build-isolation`. |
