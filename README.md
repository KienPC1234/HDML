# Hierarchical Decision Mamba-Liquid (HDML)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%7C%20CUDA%2012%2B%20%2F%2013%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Mamba](https://img.shields.io/badge/Mamba-S6%20%2F%20Mamba2-008080?style=flat)](https://github.com/state-spaces/mamba)
[![Liquid LNN](https://img.shields.io/badge/MIT-Liquid%20Neural%20Networks%20(CfC)-6f42c1?style=flat)](https://github.com/mlech26l/ncps)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3D%20Physics%20Engine-F58220?style=flat)](https://mujoco.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

HDML is a hybrid neural control architecture engineered for high-dimensional 3D robotic systems, such as bipedal humanoids, quadrupedal robots, and multi-fingered dexterous hands. 

By unifying **Selective State Space Models (Mamba / S6)** with **Closed-Form Continuous-Time Liquid Neural Networks (CfC / LTC)**, HDML decouples long-horizon cognitive planning from high-frequency reactive motor control.

---

## Overview

Complex 3D robotic tasks require reasoning across extended temporal horizons while concurrently executing continuous, low-latency motor corrections in the presence of noise, variable latency, and physical perturbations. Traditional architectures face significant trade-offs:

- **Decision Transformers & Attention Models**: Incur quadratic computational cost $\mathcal{O}(N^2)$ and an unbounded key-value cache memory $\mathcal{O}(N)$, rendering high-frequency real-time execution intractable.
- **Recurrent Architectures (LSTM / GRU)**: Exhibit constant inference memory $\mathcal{O}(1)$, but struggle with long-horizon gradient degradation and inability to model continuous-time dynamics.

HDML addresses these limitations through a **two-tier hierarchical decoupling**:

1. **Macro-Planning Layer (Mamba S6 SSM)**: Operates at low frequency (10–20 Hz) across tens of thousands of tokens, managing long-horizon intent and task routing with linear complexity $\mathcal{O}(N)$ and constant memory state $\mathcal{O}(1)$.
2. **Micro-Actuation Layer (Liquid CfC / LTC)**: Operates at high frequency (100–500 Hz), solving an explicit closed-form approximation of ordinary differential equations (ODEs) to provide continuous-time adaptation, rapid disturbance rejection, and smooth joint actuation.

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

## Comparative Analysis

| Feature | Decision Transformer (DT) | Conventional RNN / LSTM | HDML (Mamba + Liquid) |
| :--- | :--- | :--- | :--- |
| **Sequence Complexity** | Quadratic $\mathcal{O}(N^2)$ | Linear $\mathcal{O}(N)$ | **Linear $\mathcal{O}(N)$** |
| **Inference Step Memory** | Growing $\mathcal{O}(N)$ KV Cache | Constant $\mathcal{O}(1)$ | **Constant $\mathcal{O}(1)$ SSM State** |
| **Continuous-Time Modeling** | Discretized steps only | Discretized steps only | **Native Continuous ODEs** |
| **Perturbation Robustness** | Sensitive to visual/force noise | Moderate | **High (Contractive Dynamics)** |
| **Control Latency** | High (> 50 ms) | Low (< 5 ms) | **Ultra-Low (< 2 ms on Edge GPU)** |

---

## Repository Structure

```
HDML_Model/
├── LICENSE                   # Apache License 2.0 open-source protection
├── pyproject.toml            # Editable packaging configuration (pip install -e .)
├── requirements.txt          # Python dependency specifications
├── research.md               # Rigorous academic research paper & mathematical formulations
├── SETUP_AND_DOCS.md         # Technical architecture & deployment documentation
├── README.md                 # Project overview and quickstart guide
├── hdml/                     # Core HDML Python package
│   ├── models/               # Fusion, Mamba S6 Backbone, Liquid Head, HDMLModel
│   ├── data/                 # Collector, FastTensorTrajectoryDataset, Minari Adapter
│   ├── training/             # HDMLLoss (Advantage-weighted), HDMLTrainer (AMP BFloat16)
│   ├── evaluation/           # Closed-loop MuJoCo Evaluator & Perturbation Generators
│   └── utils/                # Config loaders, Kinematic Metrics, ONNX Exporters
├── configs/                  # Benchmark configurations (Ant-v4, Humanoid-v4)
├── scripts/                  # CLI commands for collection, training, evaluation, ONNX export
└── tests/                    # 16 automated unit & integration test suites
```

---

## Quickstart & Execution

### 1. Data Collection
```bash
python scripts/collect_data.py --env Ant-v4 --num-episodes 50 --output data/ant_v4_trajectories.npz
```

### 2. High-Throughput Offline Training (AMP BFloat16)
```bash
python scripts/train_offline.py \
    --config configs/ant_v4_default.yaml \
    --dataset data/ant_v4_trajectories.npz \
    --batch-size 128 \
    --epochs 20 \
    --amp \
    --num-workers 4 \
    --fast-data
```

### 3. Closed-Loop Evaluation & Decoupled Hierarchical Inference
```bash
python scripts/evaluate.py \
    --config configs/ant_v4_default.yaml \
    --checkpoint checkpoints/ant_v4/best_model.pt \
    --macro-interval 5 \
    --episodes 10 \
    --device cuda
```

### 4. Edge Deployment (ONNX Export & Verification)
```bash
python scripts/export_onnx.py \
    --config configs/ant_v4_default.yaml \
    --checkpoint checkpoints/ant_v4/best_model.pt \
    --output checkpoints/ant_v4/hdml_liquid_head.onnx
```

### 5. Automated Tests
```bash
pytest tests/ -v
```

---

## License

This project is open-sourced under the [Apache License 2.0](LICENSE).

---

## Academic Citation

```bibtex
@article{hdml2026,
  title={Hierarchical Decision Mamba-Liquid Architecture for 3D Robotic Continuous Control},
  author={HDML Project Contributors},
  journal={arXiv preprint},
  year={2026}
}

@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}

@article{hasani2022closed,
  title={Closed-form continuous-time neural networks},
  author={Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Rus, Daniela and Grosu, Radu},
  journal={Nature Machine Intelligence},
  volume={4},
  number={11},
  pages={992--1003},
  year={2022}
}
```
