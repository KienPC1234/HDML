# Setup and Documentation

This document specifies the software architecture, execution environment, and documentation map for the Hierarchical Decision Mamba-Liquid (HDML) repository.

## 1. System Environment

- **Hardware Platform**: NVIDIA GeForce RTX 4070 SUPER (12GB VRAM, Compute Capability 8.9 / Ada Lovelace).
- **CUDA Toolkit**: CUDA 13.2 (`/usr/local/cuda-13.2`).
- **Python Runtime**: Python 3.11.15 in a local virtual environment (`.venv`).
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
2. **Action Policy**: Flow-Matching continuous-time velocity field over an action chunk ($k=5$), solved via deterministic Euler integration. `chunk_size=5` is the number of emitted flow actions; it is not the macro-plan lifetime.
3. **Micro-Tier Filtering**: Closed-Form Continuous-time neural network (CfC, `ncps.torch.CfC`) smoothing actuator control signals at the micro rate. The filter input is `[first flow action, Mamba latent]`, not the raw sensor vector. Its hidden state is carried across control steps and reset at episode start; the default normalized time step is 1.0.
4. **Regularization**: HiQC critic with PAVE (Path-Action Value Equalization) and Grad-CAPS (action gradient penalty) integrated into `HDMLTrainer`.
5. **Two-Tier Control**: Macro planning emits subgoals every `macro_interval` control steps; the micro tier holds the subgoal and regenerates an action every step (`act_from_subgoal`). The flow chunk is not executed action-by-action. PACE monitors the forward-dynamics prediction $s_{t+\text{chunk\_size}}$ at its due step and distinguishes deviation from normal plan expiry.
6. **Execution Semantics**: Unitree A1 actions are normalized target joint angles for position servos (`data.ctrl`), not direct motor torques. Action-jerk numbers are therefore command-smoothness proxies.

### Verified Performance Summary (HalfCheetah-v5, RTX 4070 SUPER, Seed 42)

| Metric | Measured Value | Baseline Comparison |
| :--- | :--- | :--- |
| **Model Parameters** | 1,396,240 (~1.40M) | Decision Transformer: 1.21M, Decision RNN: 1.01M |
| **Pure GPU Inference** | 8.14 ms (122.9 Hz) | Forward pass on context length $T=20$ |
| **Closed-Loop Cycle** | 9.02 ms (110.8 Hz) | Includes sensor read, normalisation, inference, and physics step |
| **Portable CPU Inference** | 11.22 ms (89.1 Hz) | Portable pure-PyTorch Mamba-3 architecture on CPU |
| **Normalized IQM** | 65.99 [7.81, 101.14] | Outperforms Decision RNN (45.27); DT: 95.60 [25.36, 107.79] |
| **Mechanical Jerk** | 0.5724 | Lowest among sequence models (29.4% lower than DT's 0.8109) |
| **Perturbed IQM** | 10.24 [8.44, 13.72] | 8.7x retention over DT (1.17); 100% completion rate |

## 3. Quick Start & Execution

### Environment Activation
```bash
source .venv/bin/activate
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
- [docs/ONNX_DEPLOYMENT.md](docs/ONNX_DEPLOYMENT.md): Production ONNX export reference, C#/C++/Python game engine runtime integration, and latency benchmarks.
- [docs/REBUILD_2026-10-09.md](docs/REBUILD_2026-10-09.md): Detailed rebuild log, architectural audit, and run artifacts.
- [docs/OPERATIONAL_FIXES_2026-10-20.md](docs/OPERATIONAL_FIXES_2026-10-20.md): PACE prediction timing, stateful CfC rollout, CUDA-graph timesteps/short windows, and the re-run evidence.
- [docs/AI_ASSISTANCE.md](docs/AI_ASSISTANCE.md): Disclosure of AI-assisted engineering and administrative tasks.
