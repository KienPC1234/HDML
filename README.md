# HDML simulation research prototype

HDML combines a native Mamba-3 selective state-space backbone (built-in rotary
state-space embedding and trapezoidal discretisation) with a Flow-Matching
action-chunk policy and a continuous-time CfC output filter. It is a research
codebase for simulated continuous control. It is not a validated physical robot
controller.

## Current verification status

The 9 October 2026 rebuild implements the architecture the report describes and
retrains it: native Mamba-3, Flow-Matching action chunking (k=5), HiQC critic with
PAVE and Grad-CAPS regularisation, and a two-tier macro/micro controller. See
[docs/REBUILD_2026-10-09.md](docs/REBUILD_2026-10-09.md) for the exact changes and
the raw run that backs each number.

On HalfCheetah-v5 (5 episodes, seed 42, RTX 4070 SUPER, deterministic flow solve)
the validated model achieves an IQM score of **65.99** [7.81, 101.14] (raw return
7234.5 +/- 5194.9), outperforming Decision RNN (45.27) and statistically within
the Decision Transformer 95% confidence interval [25.36, 107.79]. It maintains
the lowest mechanical jerk among sequence models (**0.5724**, a 29.4% reduction
from DT's 0.8109). Under continuous sensor noise and impulse perturbations,
HDML preserves an IQM of **10.24** [8.44, 13.72] with 100% episode survival
(an 8.7x retention margin over DT's 1.17).

Pure model inference latency is **8.14 ms** (**122.9 Hz**), closed-loop control
cycle is **9.02 ms** (**110.8 Hz**), and portable CPU deployment achieves
**11.22 ms** (**89.1 Hz**). Data boundaries, no-leakage causal action inputs,
and exact artifact verification conform to AGENTS.md.

## Implemented path

- State, return-to-go, previous action, and timestep are fused into sequence features.
- The backbone is native `mamba_ssm.Mamba3` (Triton/CUDA). `Mamba3Block` in
  `hdml/models/mamba3_backbone.py` is the legacy Mamba-1 + input-RoPE + gate block,
  retained for old checkpoints and for CPU/ONNX use.
- The policy is a Flow-Matching velocity field over an action chunk, refined on the
  executed action by the CfC filter.
- The foundation model shares a backbone and CfC feature refinement across
  embodiment-specific adapters.
- Two-tier control is implemented: the macro tier plans a subgoal, the micro tier
  (`act_from_subgoal`) resamples a flow chunk from it. `macro_interval=1` is the
  synchronous mode.
- Native Mamba-3 recomputes the supplied context window; a single-step streaming
  cache is not wired into the evaluator.
- CfC is a learned refinement; it does not by itself prove actuator smoothness,
  stability, or disturbance rejection.

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
