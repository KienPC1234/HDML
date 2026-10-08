# Setup and Documentation

This document specifies the software architecture, execution environment, and documentation map for the Hierarchical Decision Mamba-Liquid (HDML) repository.

## 1. System Environment

- **Hardware Platform**: NVIDIA GeForce RTX 4070 SUPER (12GB VRAM, Compute Capability 8.9 / Ada Lovelace).
- **CUDA Toolkit**: CUDA 13.2 (`/usr/local/cuda-13.2`).
- **Python Runtime**: Python 3.11.15 in local virtual environment (`/data/HDML_Model/.venv`).
- **Core Dependencies**:
  - `torch==2.13.0+cu130`
  - `mamba-ssm==2.3.2.post1` (compiled via `--no-build-isolation` with native CUDA/Triton kernels)
  - `causal-conv1d==1.6.1`
  - `ncps==0.0.2`
  - `gymnasium==1.3.0`
  - `mujoco==3.4.0`

## 2. Architecture Status & Verification

The codebase implements the verified HDML architecture:
1. **Backbone**: Native Mamba-3 selective state-space model (`mamba_ssm.modules.mamba3.Mamba3`) with built-in rotary state-space embedding and trapezoidal discretisation (`hdml/models/mamba3_native.py`).
2. **Action Policy**: Flow-Matching continuous-time velocity field over an action chunk ($k=5$), solved via deterministic Euler integration.
3. **Micro-Tier Filtering**: Closed-Form Continuous-time neural network (CfC, `ncps.torch.CfC`) smoothing actuator control signals at the micro rate.
4. **Regularization**: HiQC critic with PAVE (Path-Action Value Equalization) and Grad-CAPS (action gradient penalty) integrated into `HDMLTrainer`.
5. **Two-Tier Control**: Macro planning emits subgoals at low frequency ($1-5\text{ Hz}$); micro tier samples and tracks via continuous filtering ($50-100\text{ Hz}$).

### Verified Performance Summary (HalfCheetah-v5, RTX 4070 SUPER, Seed 42)

| Metric | Measured Value | Baseline Comparison |
| :--- | :--- | :--- |
| **Model Parameters** | 1,396,240 (~1.40M) | Decision Transformer: 1.21M, Decision RNN: 1.01M |
| **Pure GPU Inference** | 8.14 ms (122.9 Hz) | Forward pass on context length $T=20$ |
| **Closed-Loop Cycle** | 9.02 ms (110.8 Hz) | Includes sensor read, normalisation, inference, and physics step |
| **Portable CPU Inference** | 11.22 ms (89.1 Hz) | Portable Mamba-1/RoPE architecture on CPU |
| **Normalized IQM** | 65.99 [7.81, 101.14] | Outperforms Decision RNN (45.27); DT: 95.60 [25.36, 107.79] |
| **Mechanical Jerk** | 0.5724 | Lowest among sequence models (29.4% lower than DT's 0.8109) |
| **Perturbed IQM** | 10.24 [8.44, 13.72] | 8.7x retention over DT (1.17); 100% completion rate |

## 3. Quick Start & Execution

### Environment Activation
```bash
source /data/HDML_Model/.venv/bin/activate
```

### Test Suite Execution
```bash
# Full test suite (CUDA and portable unit tests)
pytest tests/ tests_cpu/ -v
```

### Benchmark Replication
```bash
# Run RLiable benchmark on HalfCheetah-v5
python scripts/benchmark_baselines.py \
  --config configs/halfcheetah_v5_default.yaml \
  --checkpoint checkpoints/rebuild_long/halfcheetah_v5/best_model.pt \
  --episodes 5 --seed 42 --device cuda \
  --output-dir results/rebuild_det
```

### Latency Measurement
```bash
# Measure pure inference and end-to-end closed-loop frequency
python scripts/measure_closed_loop.py \
  --checkpoint checkpoints/rebuild_long/halfcheetah_v5/best_model.pt \
  --samples 3000
```

## 4. Documentation Map

- [README.md](README.md): Main project overview, installation, and usage.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): Protocol for verifying and reproducing benchmark results.
- [docs/REBUILD_2026-10-09.md](docs/REBUILD_2026-10-09.md): Detailed rebuild log, architectural audit, and run artifacts.
- [docs/AI_ASSISTANCE.md](docs/AI_ASSISTANCE.md): Disclosure of AI-assisted engineering and administrative tasks.
- [paper/](paper/): Academic manuscript source (`main.tex`), compiled paper (`main.pdf`), and figures.
- [submission/](submission/): Complete anonymous competition submission package.
