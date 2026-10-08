# HDML simulation research prototype

HDML combines a Mamba-1 backbone with input rotary position encoding, a learned output gate, a direct action predictor, and optional CfC refinement. It is a research codebase for simulated continuous control. It is not a validated physical robot controller.

## Current verification status

The October 2026 correction branch addresses data boundaries, evaluation leakage, missing-checkpoint handling, recurrent context replay, timing, and unsupported architecture claims. No replacement GPU benchmark scores have been produced for this revision. See [corrections](docs/CORRECTIONS.md) and [verification](docs/VERIFICATION.md).

The v1.0.0 release, `paper/main.pdf`, figures, and existing result text files are historical artifacts. They do not validate the corrected implementation. In particular, earlier claims of unseen-robot transfer in 30 seconds, 50 N robustness, Mamba-3 discretization, 100–500 Hz dual-rate execution, and survival are not supported by the associated public implementation. Do not reuse those claims as results of this revision.

## Implemented path

- State, return-to-go, previous action, and timestep are fused into sequence features.
- `Mamba3Block` is a legacy API name for **Mamba-1 + input RoPE + an output gate**, not native Mamba-3. Its state dictionary layout is retained for existing checkpoints.
- The standalone policy predicts actions with a direct MLP and optional residual CfC filter.
- The foundation model shares a backbone and CfC feature refinement across embodiment-specific adapters.
- Inference recomputes the context window; no streaming Mamba state cache is implemented. Recomputed windows must use a fresh recurrent state.
- Legacy flow/critic modules are kept for checkpoint compatibility but are inactive in the supported standalone trainer. Unsupported dual-rate execution is rejected. Use `macro_interval=1`.
- CfC is a learned refinement; it does not by itself prove actuator smoothness, stability, or disturbance rejection.

## Installation

Use an isolated environment with an explicitly compatible PyTorch, NVIDIA driver, CUDA toolkit, and Mamba build. The repository's lower-bound dependency lists are not an exact reproduction lockfile. Record the versions actually used with each run.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# Install a PyTorch/CUDA build compatible with the local driver and toolkit first.
pip install 'causal-conv1d>=1.4.0' --no-build-isolation
pip install 'mamba-ssm>=2.0.0' --no-build-isolation
pip install -r requirements.txt
pip install -e .
```

## Checks

Data preprocessing regressions can run with Python and NumPy only:

```bash
python -m unittest discover -s tests_cpu -v
```

The complete model checks require the installed model dependencies and a working CUDA Mamba environment:

```bash
python -m pytest tests/ -v
```

Passing preprocessing checks does not establish model correctness or benchmark performance.

## Reproduce an existing single-task experiment

Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The following commands consume an existing, provenance-checked dataset. They do not create or certify expert demonstrations.

```bash
python scripts/train_offline.py --config configs/halfcheetah_v5_default.yaml \
  --dataset data/halfcheetah_v5_expert.npz --epochs 3 --stride 1 --seed 42 \
  --output-dir checkpoints/corrected/seed42
python scripts/train_baselines.py --config configs/halfcheetah_v5_default.yaml \
  --dataset data/halfcheetah_v5_expert.npz --model all --epochs 3 --seed 42 \
  --output-dir checkpoints/corrected/seed42/baselines
python scripts/benchmark_baselines.py --config configs/halfcheetah_v5_default.yaml \
  --checkpoint checkpoints/corrected/seed42/best_model.pt --episodes 10 --seed 42 \
  --output-dir results/corrected/seed42
```

Both training entry points now use stride 1 by default. Equal data and optimizer-step settings do not imply equal FLOPs or wall-clock compute. Each benchmark consumes trained checkpoints and records individual episode returns, configuration, perturbation settings, and checkpoint hashes in JSON. Bootstrap intervals describe evaluation episodes of one checkpoint per model; they are not training-seed uncertainty.

The perturbation benchmark adds random noise to normalized actions (probability 0.05, magnitude 0.6) and observation noise/dropout. It does not apply a measured 50 N body force. Episode completion is reported separately from task return and is not a physical safety metric.

## Foundation adaptation diagnostics

Buffers split whole episodes before normalization and never sample across reset boundaries. Adapter validation uses held-out target episodes. A model with prior exposure to those episodes or the target embodiment must not be called unseen transfer.

```bash
python scripts/hdml_cli.py benchmark-foundation \
  --checkpoint checkpoints/hdml_foundation/hdml_foundation_best.pt \
  --allow-pretraining-overlap --output results/corrected/seen_adapter_diagnostics.json
```

The explicit overlap flag allows **diagnostics only**, labeled `seen_or_unverified`. Strict evaluation without that flag requires a new checkpoint with recorded data provenance and an excluded target. Held-out action loss is not closed-loop robot performance. Several foundation collection scripts use reward-filtered random actions, not expert controllers.

## ONNX

```bash
python scripts/hdml_cli.py export-onnx \
  --config configs/halfcheetah_v5_default.yaml \
  --checkpoint checkpoints/corrected/seed42/best_model.pt --output deployment/model.onnx
```

The portable exporter supports a fixed context length and dynamic batch size. Its parity check compares portable PyTorch to ONNX; CUDA Mamba versus the portable implementation still needs separate numerical verification.

## AI assistance and license

Changes on the correction branch were AI-assisted; see [AI assistance record](docs/AI_ASSISTANCE.md). The code remains under the repository's [Apache 2.0 license notice](LICENSE). Existing third-party assets retain their own licenses.
