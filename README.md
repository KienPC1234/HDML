# Hierarchical Decision Mamba-Liquid (HDML)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%7C%20CUDA%2012%2B%20%2F%2013%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Mamba](https://img.shields.io/badge/Mamba-S6%20%2F%20Mamba2-008080?style=flat)](https://github.com/state-spaces/mamba)
[![Liquid LNN](https://img.shields.io/badge/MIT-Liquid%20Neural%20Networks%20(CfC)-6f42c1?style=flat)](https://github.com/mlech26l/ncps)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3D%20Physics%20Engine-F58220?style=flat)](https://mujoco.org/)
[![WandB](https://img.shields.io/badge/WandB-Experiment%20Tracking-FFBE00?style=flat&logo=weightsandbiases&logoColor=white)](https://wandb.ai/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

HDML is a hybrid neural control architecture engineered for high-dimensional 3D robotic systems, such as bipedal humanoids, quadrupedal robots, and continuous locomotion agents.

By unifying **Selective State Space Models (Mamba / S6)** with **Closed-Form Continuous-Time Liquid Neural Networks (CfC / LTC)**, HDML decouples long-horizon cognitive planning from high-frequency reactive motor control.

> **Honest-benchmark notice**: all comparisons in this repository use **trained baselines** (same data, budget and seeds) and a **leakage-free causal action-input convention** (see [Benchmarking Methodology](#benchmarking-methodology)). The results below are reproducible on an NVIDIA RTX 4070 SUPER with the commands in [Quickstart](#quickstart--execution).

---

## Simulation Rollouts (Hardware EGL Accelerated)

| Ant-v4 (Decoupled 10Hz/100Hz Macro Planning) | HalfCheetah-v5 (50,000 Step Dataset Rollout) |
| :---: | :---: |
| ![Ant-v4 HDML Rollout](videos/hdml_ant_v4_rollout.gif) | ![HalfCheetah-v5 HDML Rollout](videos/hdml_halfcheetah_v5_rollout.gif) |

---

## Action Waveforms & Mechanical Chatter Comparison

![Mechanical Actuation Waveforms](plots/action_waveforms.png)
*Figure 1: Closed-loop continuous joint torque commands $a_t \in [-1, 1]$ (top) and instantaneous mechanical jerk $\|\Delta^2 a_t\|^2$ on log scale (bottom) on HalfCheetah-v5. All models are evaluated identically on raw outputs (no synthetic signal injection). Measured mean-jerk: Diffusion Policy 0.22, Mamba+MLP 0.69, Decision Transformer 0.71, HDML 0.91.*

---

## Overview

Complex 3D robotic tasks require reasoning across extended temporal horizons while concurrently executing continuous, low-latency motor corrections in the presence of noise, variable latency, and physical perturbations. Traditional architectures face significant trade-offs:

- **Decision Transformers & Attention Models**: Incur quadratic computational cost $\mathcal{O}(N^2)$ and an unbounded key-value cache memory $\mathcal{O}(N)$, rendering high-frequency real-time execution intractable at long horizons.
- **Recurrent Architectures (LSTM / GRU)**: Exhibit constant inference memory $\mathcal{O}(1)$, but struggle with long-horizon gradient degradation and lack continuous-time differential adaptation.

HDML addresses these limitations through a **two-tier hierarchical decoupling**:

1. **Macro-Planning Layer (Mamba S6 SSM)**: Operates at low frequency (10–20 Hz) across tens of thousands of tokens, managing long-horizon intent and task routing with linear complexity $\mathcal{O}(N)$ and constant memory state $\mathcal{O}(1)$.
2. **Micro-Actuation Layer (Liquid CfC / LTC)**: Operates at high frequency (100–500 Hz), solving an explicit closed-form approximation of ordinary differential equations (ODEs) to provide continuous-time adaptation and rapid disturbance rejection.

```
                                  HDML Topology

   Sensory Stream:
   [Vision: Depth/RGB] ---> [Patch Encoder] ----+
   [Proprioception   ] ---> [MLP Kinematics] ---+---> [Cross-Modal Fusion] U_t
   [Return-to-Go R_t ] ---> [Linear Project] ---+
   [Action History   ] ---> [Linear Project] ---+
                                                                 |
     +----------------------------------------------------------+
     |
     v (Low-Frequency Macro-Planning: 10-20 Hz)
+--------------------------------------------------------------+
|             MAMBA COGNITIVE PLANNER (SSM S6)                 |
|  - Selective Discretized State Update: h_t = A_t h_{t-1} + B_t U_t |
|  - Linear Sequence Complexity O(N) | Invariant State Size O(1)|
|  - Generates Latent Subgoal Intent Vector: c_t in R^{d_subgoal}|
+--------------------------------------------------------------+
     |
     | Intent Vector / Latent Subgoal (c_t)
     v (High-Frequency Micro-Actuation: 100-500 Hz)
+--------------------------------------------------------------+
|         LIQUID REACTIVE MOTOR HEAD (MIT CfC / LTC)           |
|  - Closed-Form ODE Solution with dynamic time constants τ_i  |
|  - Continuous-Time Dynamics & Sub-millisecond Execution      |
|  - Rejects physical force impulses & high-frequency noise    |
+--------------------------------------------------------------+
     |
     +---> Target Joint Torques / Continuous Control Commands a_t (MuJoCo)
```

---

## Empirical Benchmark Results

All experiments were executed on active hardware (**NVIDIA GeForce RTX 4070 SUPER 12GB**, CUDA 13.2, PyTorch 2.x with AMP BFloat16). Baselines are trained offline on the identical dataset/budget as HDML. D4RL-normalized scores use the official D4RL reference bounds; note the datasets are synthetically generated by the built-in CPG policy (not the original D4RL datasets), so raw returns are not comparable to published D4RL results.

### 1. Ant-v4 (30-Episode Dataset Benchmark, 5 Evaluation Episodes)

| Architecture / Paradigm | Parameters | Control Frequency (Hz) | Step Latency (ms) | Jerk $\Delta^2 a_t$ (Lower = Smoother) | D4RL Normalized Score | Perturbation Survival % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HDML (Decision Mamba + Liquid CfC - Ours)** | **997,609** | **308.6 Hz** | **3.241 ms** | **0.0227** | **3.65** | **100.0%** |
| Diffusion Policy (DDPM 10-step Denoising) | 155,912 | 141.8 Hz | 7.052 ms | 0.5282 | 3.54 | 80.0% |
| Decision Transformer (Causal Attention DT) | 1,208,712 | 405.8 Hz | 2.464 ms | 0.0601 | 4.38 | 100.0% |
| Implicit Q-Learning (IQL / Value-Advantage) | 298,763 | 3,119.0 Hz | 0.321 ms | 0.0041 | 5.50 | 40.0% |
| Decision RNN (LSTM Recurrent Policy) | 1,010,184 | 863.9 Hz | 1.158 ms | 0.0169 | 4.16 | 80.0% |
| MLP-BC (Standard Feedforward Reactive) | 75,272 | 2,335.9 Hz | 0.428 ms | 0.0047 | -0.25 | 100.0% |

### 2. HalfCheetah-v5 (50,000 Step Dataset Benchmark, 5 Evaluation Episodes)

| Architecture / Paradigm | Parameters | Control Frequency (Hz) | Step Latency (ms) | Jerk $\Delta^2 a_t$ (Lower = Smoother) | D4RL Normalized Score | Perturbation Survival % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HDML (Decision Mamba + Liquid CfC - Ours)** | **995,367** | **310.8 Hz** | **3.217 ms** | **0.6962** | **1.61** | **100.0%** |
| Diffusion Policy (DDPM 10-step Denoising) | 153,606 | 138.9 Hz | 7.198 ms | 0.1950 | 2.26 | 100.0% |
| Decision Transformer (Causal Attention DT) | 1,206,918 | 407.3 Hz | 2.455 ms | 0.6942 | 1.14 | 100.0% |
| Implicit Q-Learning (IQL / Value-Advantage) | 286,985 | 3,466.2 Hz | 0.288 ms | 0.6847 | 1.47 | 100.0% |
| Decision RNN (LSTM Recurrent Policy) | 1,008,390 | 867.5 Hz | 1.153 ms | 0.6943 | 1.54 | 100.0% |
| MLP-BC (Standard Feedforward Reactive) | 72,198 | 2,648.7 Hz | 0.378 ms | 0.5796 | -0.88 | 100.0% |

### 3. Perturbation Robustness (Random Force Impulses & Continuous Sensor Noise)

Ant-v4 perturbation table (raw return / D4RL score / jerk / survival):

| Architecture / Paradigm | Raw Return | D4RL Score | Jerk | Survival % |
| :--- | :---: | :---: | :---: | :---: |
| **HDML (Decision Mamba + Liquid CfC - Ours)** | **+62.75 +/- 32.00** | **6.32** | 0.0843 | **100.0%** |
| Diffusion Policy (DDPM 10-step Denoising) | -117.39 +/- 26.46 | 3.39 | 0.5576 | 80.0% |
| Decision Transformer (Causal Attention DT) | -86.96 +/- 38.06 | 3.88 | 0.0773 | 100.0% |
| Implicit Q-Learning (IQL / Value-Advantage) | -33.11 +/- 33.13 | 4.76 | 0.2466 | 40.0% |
| Decision RNN (LSTM Recurrent Policy) | -84.27 +/- 30.18 | 3.93 | 0.0179 | 80.0% |
| MLP-BC (Standard Feedforward Reactive) | -431.82 +/- 33.04 | -1.73 | 0.2207 | 100.0% |

HalfCheetah-v5 perturbation table:

| Architecture / Paradigm | Raw Return | D4RL Score | Jerk | Survival % |
| :--- | :---: | :---: | :---: | :---: |
| **HDML (Decision Mamba + Liquid CfC - Ours)** | -91.52 +/- 54.50 | 1.52 | 0.6952 | **100.0%** |
| Diffusion Policy (DDPM 10-step Denoising) | +0.92 +/- 10.64 | **2.26** | 0.2284 | 100.0% |
| Decision Transformer (Causal Attention DT) | -74.48 +/- 42.06 | 1.66 | 0.6926 | 100.0% |
| Implicit Q-Learning (IQL / Value-Advantage) | -172.11 +/- 76.47 | 0.87 | 0.6747 | 100.0% |
| Decision RNN (LSTM Recurrent Policy) | -94.90 +/- 26.72 | 1.49 | 0.6925 | 100.0% |
| MLP-BC (Standard Feedforward Reactive) | -128.43 +/- 52.50 | 1.22 | 0.5902 | 100.0% |

### 4. Multi-Seed Statistical Ablation Study (5 Random Seeds on HalfCheetah-v5)

Synchronous per-step inference (`macro_interval=1`) for an equal-compute comparison. All models trained offline on the identical dataset.

| Architecture / Model Variant | Complexity (Time/Mem) | Frequency (Hz) $\uparrow$ | Step Latency (ms) $\downarrow$ | Jerk $\Delta^2 a_t \downarrow$ | D4RL Score $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HDML (Decision Mamba + Liquid CfC - Ours)** | **$\mathcal{O}(N) / \mathcal{O}(1)$** | **$81.1 \pm 0.5$** | **$12.33 \pm 0.08$** | **$0.7143 \pm 0.0024$** | **$1.68 \pm 0.31$** |
| Ablation: Mamba + MLP Head (No Liquid) | $\mathcal{O}(N) / \mathcal{O}(1)$ | $280.9 \pm 1.0$ | $3.56 \pm 0.01$ | $0.6899 \pm 0.0004$ | $1.35 \pm 0.29$ |
| Ablation: Transformer + Liquid Head (No Mamba) | $\mathcal{O}(N^2) / \mathcal{O}(N)$ | $87.1 \pm 0.2$ | $11.48 \pm 0.03$ | $0.6910 \pm 0.0007$ | $1.51 \pm 0.27$ |
| Decision Transformer (Causal Attention) | $\mathcal{O}(N^2) / \mathcal{O}(N)$ | $407.1 \pm 1.8$ | $2.46 \pm 0.01$ | $0.6931 \pm 0.0006$ | $1.49 \pm 0.33$ |
| Decision RNN (LSTM Recurrent Policy) | $\mathcal{O}(N) / \mathcal{O}(1)$ | $868.1 \pm 1.5$ | $1.15 \pm 0.00$ | $0.6939 \pm 0.0013$ | $1.64 \pm 0.29$ |
| Diffusion Policy (DDPM 10-step Denoising) | $\mathcal{O}(K \cdot N) / \mathcal{O}(N)$ | $142.9 \pm 0.4$ | $7.00 \pm 0.02$ | $0.1993 \pm 0.0020$ | $2.24 \pm 0.03$ |
| Implicit Q-Learning (IQL Advantage Actor) | $\mathcal{O}(1) / \mathcal{O}(1)$ | $4,321.2 \pm 24.9$ | $0.23 \pm 0.00$ | $0.6834 \pm 0.0005$ | $1.61 \pm 0.24$ |
| MLP-BC (Standard Feedforward Reactive) | $\mathcal{O}(1) / \mathcal{O}(1)$ | $3,910.8 \pm 25.3$ | $0.26 \pm 0.00$ | $0.5970 \pm 0.0044$ | $0.88 \pm 1.02$ |

> **Key Findings (honest reading)**:
> 1. **Perturbation robustness (main strength)**: on Ant-v4, HDML is the only architecture that both maximizes its perturbed score (+62.75 raw return, D4RL 6.32) **and** maintains 100% survival under stochastic force impulses and sensor noise. The strongest standard scorer (IQL, D4RL 5.50) collapses to 40% survival under the same perturbations; Diffusion drops to 80% and its perturbed score to 3.39.
> 2. **Smoothness is NOT an advantage of HDML**: with trained baselines and no leakage, HDML's jerk (0.0227 on Ant, 0.6962 on HalfCheetah) is comparable to or worse than most baselines; Diffusion Policy is actually the smoothest on HalfCheetah (0.1950). Earlier claims of 60x smoother output were artifacts of untrained baselines and a trivially learnable action-copy shortcut.
> 3. **Ablation verdict**: the Liquid (CfC) head does not reduce jerk versus a simple MLP head (0.714 vs 0.690) and costs ~3.5x latency (12.3 ms vs 3.6 ms). HDML's advantage over its own ablations is a modest score gain (+0.33 over Mamba+MLP, +0.17 over Transformer+Liquid).
> 4. **Complexity profile**: HDML keeps constant $\mathcal{O}(1)$ state memory during rollouts, and with macro-decoupling (`macro_interval=5`) sustains ~310 Hz control frequency with sub-3.3 ms latency. Its synchronous per-step latency (12.3 ms) is the highest among the compared architectures.
> 5. **Standard-score position**: HDML is mid-pack on standard D4RL-normalized scores (3.65 on Ant, 1.61 on HalfCheetah) — it does not exceed IQL, DT, or Diffusion on clean (non-perturbed) rollouts on these synthetic datasets.

---

## Repository Structure

```
HDML_Model/
├── LICENSE                   # Apache License 2.0 open-source protection
├── pyproject.toml            # Editable packaging configuration (pip install -e .)
├── requirements.txt          # Python dependency specifications
├── research.md               # Academic research paper & mathematical formulations
├── SETUP_AND_DOCS.md         # Technical architecture & deployment documentation
├── README.md                 # Project overview and quickstart guide
├── hdml/                     # Core HDML Python package
│   ├── models/               # Fusion, Mamba S6 Backbone, Liquid Head, Baselines (DT, RNN, MLP, Diffusion, IQL), Ablations
│   ├── data/                 # Collector, FastTensorTrajectoryDataset, Minari Adapter
│   ├── training/             # HDMLLoss, HDMLTrainer, BaselineTrainer (AMP BFloat16, WandB, TB)
│   ├── evaluation/           # Closed-loop MuJoCo Evaluator & Perturbation Generators
│   └── utils/                # Config loaders, Kinematic Metrics, D4RL refs, ONNX Exporters
├── configs/                  # Benchmark configurations (Ant-v4, HalfCheetah-v4/v5, Humanoid, Hopper, Walker2d)
├── scripts/                  # CLI tools (collection, training, baseline training, benchmark, video, ONNX)
├── results/                  # Persisted benchmark & ablation tables (raw script output)
├── videos/                   # Rendered simulation videos (EGL hardware accelerated)
└── tests/                    # 33 automated unit & integration tests (12 test files)
```

---

## Quickstart & Execution

### 1. Data Collection
```bash
python scripts/collect_data.py --env HalfCheetah-v4 --num-episodes 50 --output data/halfcheetah_v4_trajectories.npz
```

### 2. High-Throughput Offline Training (AMP BFloat16 + WandB / TensorBoard)
```bash
python scripts/train_offline.py \
    --config configs/halfcheetah_v5_default.yaml \
    --dataset data/halfcheetah_v4_trajectories.npz \
    --batch-size 64 \
    --epochs 15 \
    --amp \
    --fast-data \
    --tensorboard \
    --wandb
```

### 3. Train Baselines (Fair Comparison)
```bash
python scripts/train_baselines.py \
    --config configs/halfcheetah_v5_default.yaml \
    --dataset data/halfcheetah_v4_trajectories.npz \
    --model all \
    --epochs 15
```
Checkpoints are written to `checkpoints/<env>/baselines/`.

### 4. Comprehensive Baselines Benchmark
```bash
python scripts/benchmark_baselines.py \
    --config configs/halfcheetah_v5_default.yaml \
    --checkpoint checkpoints/halfcheetah_v5/best_model.pt \
    --episodes 5 \
    --device cuda
```

### 5. Multi-Seed Ablation Benchmark
```bash
python scripts/benchmark_ablations.py \
    --config configs/halfcheetah_v5_default.yaml \
    --checkpoint checkpoints/halfcheetah_v5/best_model.pt \
    --device cuda
```

### 6. 3D Simulation Video Recording (Headless EGL Accelerated)
```bash
MUJOCO_GL=egl python scripts/record_video.py \
    --config configs/halfcheetah_v5_default.yaml \
    --checkpoint checkpoints/halfcheetah_v5/best_model.pt \
    --output videos/hdml_halfcheetah_v5_rollout.gif \
    --steps 500 \
    --macro-interval 5 \
    --device cuda
```

### 7. Automated Tests (33 Tests)
```bash
pytest tests/ -v
```

---

## Benchmarking Methodology

- **Baselines are trained**: all baseline architectures (DT, Diffusion, IQL, Decision RNN, MLP-BC) and the two ablation variants are trained offline on the identical dataset, epochs, batch size and optimizer settings via `scripts/train_baselines.py`. The benchmark scripts load these trained checkpoints and **warn** if a checkpoint is missing (falling back to random init would invalidate the comparison).
- **No action leakage**: sequence models use the standard causal Decision-Transformer convention — the model input action at position `t` is `a_{t-1}` (the previously executed action) and the prediction target is `a_t` (`hdml/data/dataset.py`). Rollout evaluation follows the same convention, so training and inference distributions match.
- **D4RL-normalized scores** use the official D4RL library reference bounds (`hdml/utils/metrics.py`). The benchmark datasets are *synthetically generated* by the built-in CPG locomotion policy (`hdml/data/collector.py`), not the original D4RL datasets — scores are therefore only meaningful as relative comparisons within this repository.
- **Jerk / smoothness metric** is consistently defined as the mean absolute second-order action difference `mean|Δ²a_t|` (`hdml/utils/metrics.py`).
- **Macro decoupling**: HDML benchmark runs use `macro_interval=5` (the Mamba planner is invoked every 5 steps; the Liquid head runs at full rate) — this is the deployment mode. Ablation runs use synchronous per-step inference (`macro_interval=1`) for an equal-compute architectural comparison.

---

## Experiment Tracking (TensorBoard & WandB)

HDML integrates real-time dashboard tracking for loss convergence, action reconstruction error, subgoal prediction loss, and runtime throughput FPS:

- **TensorBoard**: Launch the local dashboard via `tensorboard --logdir logs/` and navigate to `http://localhost:6006`.
- **Weights & Biases**: Enable live cloud telemetry by passing `--wandb --wandb-project hdml-robotics`.

---

## License

This project is open-sourced under the [Apache License 2.0](LICENSE).

---

## Academic Citation

```bibtex
@article{hdml2026,
  title={Hierarchical Decision Mamba-Liquid Architecture for 3D Robotic Continuous Control},
  author={HDML Research Team},
  journal={arXiv preprint},
  year={2026}
}
```