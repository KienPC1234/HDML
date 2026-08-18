# HDML Technical Deployment & System Architecture Guide
## (Hierarchical Decision Mamba-Liquid Architecture)

---

## 1. System Ecosystem & Library Architecture

To realize the **HDML (Hierarchical Decision Mamba-Liquid)** architecture formulated in [research.md](file:///data/HDML_Model/research.md), the codebase integrates four foundational technological pillars:

```
+---------------------------------------------------------------------------------------+
|                                    HDML SYSTEM STACK                                  |
+---------------------------------------------------------------------------------------+
|  1. Vision & Multimodal  |  timm, OpenCV, TorchVision (Patch & Cross-Modal Embedding) |
|  2. Cognitive Backbone   |  mamba-ssm, causal-conv1d (S6 / Mamba2 Long-term Planner)  |
|  3. Reactive Policy Head |  ncps (MIT Neural Circuit Policies: Closed-form CfC / LTC) |
|  4. RL & Simulation      |  Gymnasium, MuJoCo, Minari / D4RL, TorchRL                 |
|  5. Edge & Optimization  |  ONNX, TensorRT, TorchScript (ARM Cortex / Jetson Deploy)  |
+---------------------------------------------------------------------------------------+
```

---

## 2. Core Python & PyTorch Engineering Standards

All modules within this repository strictly adhere to modern software design patterns and deep learning best practices:

- **Strict Type Annotations**: Every function, method, and tensor contract includes explicit PEP 484/585/604 annotations with `from __future__ import annotations`.
- **Zero Hallucination & Empirical GPU Execution**: All models and training loops are verified via real-time forward/backward passes on NVIDIA Ada Lovelace hardware (`cuda:0`).
- **Defensive Error Handling**: No silent `except:` blocks or placeholder stubs. Complete exception tracebacks are logged and handled.
- **Explicit Tensor Shape Contracts**: Documented in module docstrings with assertion verification at critical projection boundaries.

---

## 3. Environment Installation & CUDA Compilation Guide

### Step 1: Initialize Python 3.11 Virtual Environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install PyTorch with CUDA Support
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Compile Mamba & Causal-Conv1D with Native CUDA Headers
```bash
export CUDA_HOME="/usr/local/cuda-13.2"
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=8

# Install causal-conv1d first
pip install causal-conv1d>=1.4.0 --no-build-isolation

# Install mamba-ssm
pip install mamba-ssm>=2.0.0 --no-build-isolation
```

### Step 4: Install Remaining Dependencies & Package in Editable Mode
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 4. Complete Code Architecture & Module Reference

The codebase is organized as a modular product package `hdml` (version `0.1.0`):

```python
from __future__ import annotations
import torch
from hdml.models import HDMLModel
from hdml.utils.config import ModelConfig
from hdml.data import FastTensorTrajectoryDataset, TrajectoryCollector, MinariDatasetAdapter
from hdml.training import HDMLTrainer
from hdml.evaluation import HDMLEvaluator

# Initialize Hierarchical Mamba-Liquid Model
cfg = ModelConfig(
    prop_dim=27,            # Proprioceptive kinematic dimension (e.g. Ant-v4)
    action_dim=8,           # Continuous action dimension
    d_model=128,            # Unified multimodal token dimension U_t
    num_mamba_layers=3,     # Depth of Mamba S6 Cognitive Backbone
    d_subgoal=64,           # Dimension of Latent Subgoal vector c_t
    cfc_units=32,           # Units in Closed-Form Continuous-Time Liquid Head
)
model = HDMLModel.from_config(cfg).to("cuda")

# 1. Full Sequence Training Pass (O(N) Time Complexity)
# states: (B, T, prop_dim), rtgs: (B, T, 1), actions: (B, T, action_dim)
actions_pred, subgoals_pred, values_pred, next_hx = model(states, rtgs, actions)

# 2. Closed-Loop Real-Time Rollout (Decoupled Hierarchical Inference)
action, cfc_hx, subgoal = model.get_action(states_ctx, rtgs_ctx, actions_ctx)
```

---

## 5. Command-Line Interface (CLI) Workflow

### 5.1. Trajectory Data Collection
```bash
python scripts/collect_data.py --env Ant-v5 --num-episodes 50 --output data/ant_v5_trajectories.npz
```

### 5.2. Offline Decision Mamba Training (High-Throughput AMP BFloat16)
```bash
python scripts/train_offline.py \
    --config configs/ant_v5_default.yaml \
    --dataset data/ant_v5_trajectories.npz \
    --batch-size 128 \
    --epochs 20 \
    --amp \
    --num-workers 4 \
    --fast-data
```

### 5.3. Closed-Loop Simulation & Two-Tier Hierarchical Decoupling
```bash
python scripts/evaluate.py \
    --config configs/ant_v5_default.yaml \
    --checkpoint checkpoints/ant_v5/best_model.pt \
    --macro-interval 5 \
    --episodes 10 \
    --device cuda
```

### 5.4. Record Rollout Video (GIF / MP4)
```bash
python scripts/record_video.py \
    --config configs/ant_v5_default.yaml \
    --checkpoint checkpoints/ant_v5/best_model.pt \
    --output videos/hdml_ant_v5_rollout.gif \
    --macro-interval 5
```

### 5.5. Execute Automated Test Suite (Unit & Integration Tests)
```bash
python -c "import pytest; pytest.main(['tests/', '-v'])"
```

---

## 6. Multi-User & Benchmark Dataset Scaling

### 6.1. Integrating Minari / D4RL Datasets
The package includes native support for official Farama Minari datasets:
```python
from hdml.data import MinariDatasetAdapter

trajectories = MinariDatasetAdapter.load_minari_dataset("door-human-v0")
```

### 6.2. Multi-GPU Training Support
HDML seamlessly scales across multiple GPUs using PyTorch Distributed Data Parallel (DDP) or standard PyTorch Lightning wrappers.
