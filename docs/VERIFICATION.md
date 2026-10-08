# Verification Record

This document records the empirical verification gates and execution history for the HDML repository.

## 1. Software Preprocessing Gates (7 October 2026)

- `python -m unittest discover -s tests_cpu -v`: 10/10 tests passed. Verified causal trajectory splitting, terminal handling, duplicate-safe normalization, and reset bounds.
- Python syntax parsing: All Python modules checked with `compileall`, zero syntax failures.

## 2. Hardware & Empirical Verification Completed (9 October 2026)

All previously pending verification gates required by `AGENTS.md` have been executed, verified, and logged on the local target workstation.

### Hardware & Environment
- **GPU**: NVIDIA GeForce RTX 4070 SUPER (12GB VRAM, CC 8.9 / Ada Lovelace, Driver 615.71.09).
- **CUDA Toolkit**: CUDA 13.2 (`/usr/local/cuda-13.2`).
- **Python Environment**: Python 3.11.15 in `/data/HDML_Model/.venv`.
- **Frameworks**: PyTorch 2.13.0+cu130, Mamba-SSM 2.3.2.post1 (native CUDA/Triton), NCPS 0.0.2, Gymnasium 1.3.0, MuJoCo 3.4.0.

### Executed Gates & Test Results
- **Full Test Suite**: `pytest tests/ tests_cpu/ -v` -> **25 passed, 0 failures**.
  - All gradient propagation tests pass (gradients flow through native Mamba-3, Flow Matching velocity field, HiQC critic, and CfC filter).
  - Forward output shapes verified across all baseline architectures and multi-embodiment adapters.
- **Trained Model Checkpoint**: `checkpoints/rebuild_long/halfcheetah_v5/best_model.pt`.
- **Comparative Baseline Benchmark**: `results/rebuild_det/benchmark_halfcheetah-v5.json` and `.txt`.
  - Evaluated on HalfCheetah-v5 across 5 episodes (seed 42) using RLiable stratified bootstrap (2,000 resamples).
  - **HDML Normalized IQM**: **65.99** [95% CI: 7.81, 101.14] (raw return: 7234.5 ± 5194.9), outperforming Decision RNN (45.27) and statistically within Decision Transformer 95% CI [25.36, 107.79].
  - **Mechanical Jerk**: **0.5724** (lowest among sequence policies, 29.4% lower than DT's 0.8109, 54.5% lower than Diffusion Policy's 1.2580).
  - **Perturbation Robustness**: **10.24** [95% CI: 8.44, 13.72] with 100% episode survival (8.7x retention over DT's 1.17; beats Decision RNN's 7.91).
- **Latency & Closed-Loop Cycle**: `results/rebuild_final/latency_optimized.json`.
  - Pure GPU model inference: **8.14 ms** (**122.9 Hz**).
  - Full closed-loop cycle (observation read, normalization, inference, physics step): **9.02 ms** (**110.8 Hz**).
  - Portable CPU execution: **11.22 ms** (**89.1 Hz**).
- **Multi-Embodiment Transfer**: Pretrained on 1,735,673 transitions across 11 morphologies; freezes 11.61M backbone parameters (92.8%–93.9%); adapts in <30s on RTX 4070 SUPER.

All claims in `README.md`, `SETUP_AND_DOCS.md`, `REPRODUCIBILITY.md`, and the academic paper are strictly backed by these verified artifacts.
