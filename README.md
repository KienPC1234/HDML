# Hierarchical Decision Mamba-Liquid (HDML)

HDML is a continuous-control architecture designed for high-dimensional 3D robotic systems, including multi-articulated quadrupeds, humanoids, and continuous locomotion platforms.

The framework couples **Selective State Space Models (Mamba-3)** with **Closed-Form Continuous-Time Liquid Neural Networks (CfC / LTC)** to bridge the gap between long-horizon cognitive planning and high-frequency, perturbation-resilient micro-actuation.

## Core Architecture

High-dimensional physical control involves two distinct operational timescales:
- **Macro-Planning**: Operating across extended temporal sequences, tracking periodic phase and intent without quadratic memory overhead.
- **Micro-Actuation**: Operating at high frequency, generating smooth, chattering-free joint torques that respond dynamically to external forces and perturbations.

```
                             HDML System Topology

   Sensory Inputs:
   [Vision: RGB-D  ] ---> [Multimodal Encoder] ---+
   [Proprioception ] ---> [Kinematics MLP   ] ---+---> [Cross-Modal Fusion] U_t
   [Return-to-Go   ] ---> [Linear Projection] ---+
   [Action History ] ---> [Linear Projection] ---+
                                                        |
     +--------------------------------------------------+
     |
     v (Macro-Planning: 10–20 Hz)
+--------------------------------------------------------------+
|             MAMBA-3 COGNITIVE BACKBONE (SSM)                 |
|  - Exponential-Trapezoidal Discretization (2nd Order)        |
|  - Rotary Position Embeddings (RoPE) for SO(3) Phase Tracking |
|  - Rank-R MIMO State Space Updates                           |
|  - Generates Latent Subgoal / Intent Vector c_t              |
+--------------------------------------------------------------+
     |
     v (Micro-Actuation: 100–500 Hz)
+--------------------------------------------------------------+
|            GENERATIVE ACTION & PHYSICAL LAYER                |
|  - HiQC Action Chunking with Q-Weighted Flow Matching         |
|  - PAVE (Mixed Hessian) & Grad-CAPS Smoothness Regularization|
|  - Closed-Form Continuous-Time (CfC) ODE Dynamic Filter      |
|  - PACE (Phase-Aware Chunk Execution) Truncation             |
+--------------------------------------------------------------+
     |
     v
Target Continuous Joint Torques a_t (Physical Actuators / MuJoCo)
```

### Key Technical Components

1. **Mamba-3 Backbone with RoPE Discretization**: Formulates state-space dynamics using Rotary Position Embeddings on input/output projections, enabling phase tracking in 3D angular dynamics with 2nd-order exponential-trapezoidal accuracy and linear $\mathcal{O}(N)$ sequence complexity.
2. **Hierarchical Implicit Q-Chunking (HiQC)**: Compresses temporal decision complexity via $k$-step action chunking, reducing Bellman backup depth from $T$ to $T/k$.
3. **Q-Weighted Flow Matching Policy**: Replaces unimodal Gaussian assumptions with Optimal Transport Flow Matching, routing action trajectories directly toward high-value regions of the action space.
4. **Policy-Aware Value-Field Equalization (PAVE) & Grad-CAPS**: Penalizes the mixed Hessian $\nabla_{sa}^2 Q$ of the critic and second-order action variations, eliminating high-frequency mechanical chattering at the optimization source.
5. **Continuous ODE Filtering via CfC & PACE**: Processes multimodal states through dynamic time constants $\tau(x)$, smoothing sensor noise during steady-state motion while contracting instantaneously to resolve sudden collisions.

## Empirical Benchmark Results

Experiments are conducted on local GPU hardware (**NVIDIA GeForce RTX 4070 SUPER**, CUDA 13.2, PyTorch with AMP BFloat16). Baselines are trained under identical conditions (identical dataset, budget, seeds, and causal action-input conventions).

### 1. NeurIPS 2021 `rliable` Protocol (HalfCheetah-v5 Benchmark)

Evaluated across 2,000 stratified bootstrap resamples with 95% confidence intervals:

| Architecture / Paradigm | Parameters | Control Freq (Hz) | Step Latency (ms) | Jerk $\Delta^2 a_t \downarrow$ | Clean IQM [95% CI] | Perturbed IQM [95% CI] $\uparrow$ | Survival Rate % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HDML (Decision Mamba + Liquid CfC - Ours)** | **1,441,713** | **77.3 Hz** | **12.93 ms** | **0.7862** | **88.09 [41.43, 98.51]** | **18.88 [9.85, 26.92]** | **100.0%** |
| Decision Transformer (Causal Attention DT) | 1,206,918 | 444.9 Hz | 2.25 ms | 0.8109 | 95.60 [25.36, 107.79] | 1.17 [0.32, 4.19] | 100.0% |
| Decision RNN (LSTM Recurrent Policy) | 1,008,390 | 953.5 Hz | 1.05 ms | 0.9689 | 113.91 [89.56, 115.45] | 7.14 [4.22, 10.71] | 100.0% |
| Implicit Q-Learning (IQL / Value-Advantage) | 286,985 | 4,051.6 Hz | 0.25 ms | 0.0080 | 1.84 [1.79, 2.26] | 2.11 [2.03, 2.32] | 100.0% |
| Flow Matching (Standard Denoising) | 153,606 | 148.9 Hz | 6.71 ms | 1.2983 | -0.11 [-0.46, 0.53] | -0.44 [-1.05, -0.05] | 100.0% |
| MLP-BC (Standard Feedforward Reactive) | 72,198 | 3,294.2 Hz | 0.30 ms | 0.0028 | -0.26 [-0.26, -0.24] | -0.34 [-0.40, -0.30] | 100.0% |

### 2. Probability of Improvement ($P(\text{HDML} > \text{Baseline})$)

- $P(\text{HDML} > \text{Flow Matching}) = \mathbf{100.00\%}$ [95% CI: 100.00% - 100.00%]
- $P(\text{HDML} > \text{MLP-BC}) = \mathbf{100.00\%}$ [95% CI: 100.00% - 100.00%]
- $P(\text{HDML} > \text{Implicit Q-Learning}) = \mathbf{100.00\%}$ [95% CI: 100.00% - 100.00%]
- $P(\text{HDML} > \text{Decision Transformer}) = 36.00\%$ [95% CI: 0.00% - 80.00%]
- $P(\text{HDML} > \text{Decision RNN}) = 16.00\%$ [95% CI: 0.00% - 60.00%]

### 3. Unitree A1 12-DOF Quadruped Physical Simulation

Under closed-loop MuJoCo simulation with 3 external lateral kicks ($+8\text{ N}, -8\text{ N}, +10\text{ N}$), HDML maintains balance and stable forward trotting ($v_x \approx -1.81\text{ m/s}$, torso height $z \approx 0.248\text{ m}$):

![Unitree A1 Kick Recovery Telemetry](videos/unitree_a1_kick_hud_snapshot.png)

Rendered telemetry video: [`videos/unitree_a1_robot_dog_kick_recovery.mp4`](videos/unitree_a1_robot_dog_kick_recovery.mp4) (1.3 MB, 720p 30fps).

### 4. Zero-Label Unsupervised Pre-training & Tri-Camera Labyrinth Solving

HDML learns latent cognitive spatial representations and closed-loop motor control on **PointMaze** and multi-articulated **AntBot (8-DOF)** without human demonstrations or external reward labels:
- **Unsupervised World Dynamics**: Self-predicts next states $s_t + a_t \to \hat{s}_{t+1}$ to form a continuous latent topological graph.
- **Hindsight Subgoal Stitching (HER)**: Automatically chains disparate exploration segments into globally optimal shortest paths across interconnected rooms.
- **Tri-Camera Multi-View Compositor**: Top pane displays a global top-down overview ($22\text{m} \times 22\text{m}$), while bottom panes display real-time close-up leg tracking and 3D isometric corridor perspective.

| Environment / Robot | State Dim | Action Dim | Unsupervised Dataset | Reached Step | Jerk $\Delta^2 a_t$ | Multi-View Video |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PointMaze U-Maze** | 4D | 2D | `D4RL/pointmaze/umaze-v2` | 76 | 0.0520 | [`videos/pointmaze_hdml_solved.gif`](videos/pointmaze_hdml_solved.gif) |
| **PointMaze Medium** | 4D | 2D | `D4RL/pointmaze/medium-v2` | 138 | 0.0384 | [`videos/pointmaze_medium_hdml_solved.gif`](videos/pointmaze_medium_hdml_solved.gif) |
| **AntBot U-Maze (8-DOF)** | 31D | 8D | `D4RL/antmaze/umaze-v2` | 274 | 0.3174 | [`videos/antmaze_hdml_solved.gif`](videos/antmaze_hdml_solved.gif) |
| **AntBot Medium (8-DOF Multi-Room)** | 31D | 8D | `D4RL/antmaze/medium-play-v2` | **212** | **0.3070** | [`videos/antmaze_medium_multiview_solved.gif`](videos/antmaze_medium_multiview_solved.gif) |

### 5. High-Throughput ONNX Export & Edge CPU Deployment

HDML models export cleanly to standard ONNX format with 100% numerical parity verification against PyTorch:

```bash
# Export trained HDML policy to ONNX with parity verification
python scripts/export_onnx.py \
    --config configs/antmaze_medium_unsupervised.yaml \
    --checkpoint checkpoints/antmaze_medium/best_model.pt \
    --output deployment/hdml_antmaze_medium_policy.onnx
```

- **Model Size:** 4.70 MB
- **Max Absolute Error ($\|\text{PyTorch} - \text{ONNX}\|_\infty$):** $5.66 \times 10^{-7}$
- **Inference Latency (Single CPU Core):** $5.39\text{ ms}$ / step (**$186\text{ Hz}$**)

## Quickstart & Reproducibility

> [!TIP]
> For a comprehensive, step-by-step reproduction guide covering all benchmarks, multi-room AntBot labyrinths, baseline comparisons, and ONNX deployment, see **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**.

### Environment Setup

```bash
# 1. Create and activate Python environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Dataset Collection

```bash
# Collect Unitree A1 quadruped locomotion trajectories
python scripts/collect_unitree_a1.py

# Collect standard continuous control benchmark trajectories
python scripts/collect_data.py --env HalfCheetah-v4 --num-episodes 50 --output data/halfcheetah_v4_trajectories.npz
```

### Model Training

```bash
# Train HDML offline on collected trajectories
python scripts/train_offline.py \
    --config configs/unitree_a1_default.yaml \
### 2. Multi-Embodiment Foundation Model (Zero-Shot & Few-Shot Transfer)

Pre-trained across 11 distinct physics morphologies (1,735,673 physical transitions). Adapting to novel robot embodiments with over 93% (11.6M parameters) of the core backbone frozen:

| Target Robot | Morphology | State / Act Dim | Frozen % | Adapt Time | Action Loss | Jerk ($\|\Delta^2 a\|$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Unitree A1** | 12-DoF Quadruped | $53\text{D} \to 12\text{D}$ | 93.7% | 30.0s | **0.00602** | 0.1401 |
| **Humanoid 3D** | 17-DoF 3D Biped | $348\text{D} \to 17\text{D}$ | 92.8% | 29.4s | **0.02579** | **0.0689** |
| **Swimmer** | 2-DoF Serpentine | $8\text{D} \to 2\text{D}$ | 93.9% | 29.3s | 0.16664 | **0.0149** |
| **Ant** | 8-DoF Quadruped | $105\text{D} \to 8\text{D}$ | 93.5% | 29.4s | 0.16161 | 0.1677 |
| **Hopper** | 3-DoF Monopod | $11\text{D} \to 3\text{D}$ | 93.8% | 29.0s | 0.15356 | 0.1798 |
| **Walker2d** | 6-DoF Bipedal | $17\text{D} \to 6\text{D}$ | 93.8% | 29.2s | 0.10298 | 0.7174 |

---

## Unified Command-Line Interface (`hdml_cli.py`)

All core operations are accessible via the unified multi-functional CLI:

```bash
# 1. Evaluate Multi-Embodiment Transfer on Frozen Foundation Backbone
python scripts/hdml_cli.py benchmark-foundation

# 2. Few-Shot Fine-Tuning on a New Custom Robot Embodiment
python scripts/hdml_cli.py evaluate-transfer \
    --checkpoint checkpoints/hdml_foundation/hdml_foundation_best.pt \
    --target-embodiment my_custom_robot \
    --target-dataset data/my_robot_data.npz \
    --prop-dim 24 --action-dim 8 --epochs 10

# 3. Benchmark HDML against all baselines (DT, RNN, Diffusion, IQL, MLP)
python scripts/hdml_cli.py benchmark-baselines \
    --config configs/halfcheetah_v5_default.yaml \
    --checkpoint checkpoints/halfcheetah_v5/best_model.pt

# 4. Export to ONNX for High-Frequency Edge Deployment
python scripts/hdml_cli.py export-onnx \
    --config configs/halfcheetah_v5_default.yaml \
    --checkpoint checkpoints/halfcheetah_v5/best_model.pt \
    --output-dir deployment/onnx
```

---

## 📜 Scientific Preprint

The complete publication manuscript is available in [`paper/`](paper/):
- **Manuscript PDF**: [`paper/main.pdf`](paper/main.pdf)
- **LaTeX Source**: [`paper/main.tex`](paper/main.tex)
- **BibTeX References**: [`paper/references.bib`](paper/references.bib)

---

## License

This project and its pre-trained model weights (`hdml_foundation_best.pt`) are open-sourced under the Apache License 2.0. See [LICENSE](LICENSE) for details.