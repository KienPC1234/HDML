# ONNX Deployment

Export trained HDML checkpoints to ONNX and run them from Python, C#, or a game
engine runtime (Unity Sentis, Unreal NNE, ONNX Runtime).

## 1. Export

`scripts/export_onnx.py` loads a checkpoint, bakes the training-set
normalization (`state_mean` / `state_std` from the checkpoint) into the graph,
replaces the native Mamba-3 CUDA kernel with `PortableMamba3`, exports the
graph, and verifies numerical parity against the portable PyTorch model.

```bash
python scripts/export_onnx.py \
  --config configs/unitree_a1_cfc_strong.yaml \
  --checkpoint checkpoints/rebuild_unitree/unitree_a1/best_model.pt \
  --output deployment/unitree_a1_action_only.onnx \
  --action-only \
  --opset 17
```

The exporter infers the backbone variant (`native Mamba-3` vs legacy Mamba-1 +
RoPE) from the checkpoint's parameter names, so a config that omits
`use_native_mamba3` cannot silently instantiate the wrong architecture.

Options:

| Flag | Meaning |
| :--- | :--- |
| `--action-only` | Lean graph with a single `action` output. Omit for `action`, `subgoal`, `next_state`. |
| `--context-length` | Override the sequence length. Must match the value used at inference. |
| `--opset` | ONNX opset (default 17). |
| `--quant {none,static-int8,int4}` | Optional post-export quantization. |
| `--calib-iters` | Calibration batches for `static-int8` (synthetic inputs). |

Each export writes `<output>.meta.json` with size, parity, CPU latency, opset,
and checkpoint provenance.

### Graph contract

- Inputs: `states (B, T, prop_dim)`, `rtgs (B, T, 1)`, `actions (B, T, action_dim)`, `timesteps (B, T)` (`int64`).
- Outputs: `action (B, action_dim)`; with `--action-only` disabled, also `subgoal` and `next_state`.
- Batch is dynamic; `T` is fixed at export time. Feeding a different `T` fails at runtime by design.

### Parity scope

The exporter compares the portable PyTorch path against ONNX Runtime. It does
not claim CUDA-kernel parity: `portable PyTorch vs ONNX` and
`CUDA vs portable` are separate checks (`hdml/deployment/onnx_exporter.py`).

### Quantization caveats

- `static-int8` calibrates on synthetic random inputs, matching the normalized
  training distribution approximately. Re-calibrate with real rollouts for
  production.
- `int4` is weight-only (`MatMulNBitsQuantizer`, block size 128).
- Quantized parity is measured and reported, not asserted. Check
  `quant_parity_vs_fp32` in the metadata before deploying.
- Dynamic INT8 is intentionally unsupported: ORT's dynamic quantizer
  mis-handles transB Gemm weights shared across unrolled steps.

## 2. C# / ONNX Runtime

`tests_csharp/` is a .NET 8 console benchmark using
`Microsoft.ML.OnnxRuntime`. It prints input/output metadata, latency
statistics, and action outputs, and can dump results as JSON.

```bash
dotnet run --project tests_csharp -- \
  --model deployment/unitree_a1_action_only.onnx \
  --prop-dim 35 --action-dim 12 --context-length 20 --iters 100
```

`--input <json>` feeds a deterministic input vector and `--dump-output <json>`
writes the action tensor, which is how `tests_cpu/test_onnx_cpu_and_csharp.py`
checks Python/C# parity.

## 3. Python / ONNX Runtime

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "deployment/unitree_a1_action_only.onnx",
    providers=["CPUExecutionProvider"],
)
outputs = session.run(None, {
    "states": np.zeros((1, 20, 35), dtype=np.float32),
    "rtgs": np.zeros((1, 20, 1), dtype=np.float32),
    "actions": np.zeros((1, 20, 12), dtype=np.float32),
    "timesteps": np.zeros((1, 20), dtype=np.int64),
})
action = outputs[0]  # (1, 12)
```

## 4. Verification

```bash
# CPU ONNX + C# parity tests. Missing deployment/*.onnx files are skipped.
pytest tests_cpu/ -q

# Fail instead of skipping when artifacts are missing (CI/release gate).
HDML_REQUIRE_DEPLOYMENT_MODELS=1 pytest tests_cpu/ -q
```

Deployment artifacts are generated from checkpoints and are not committed
(`*.onnx` is ignored). Run the export command above first on a fresh clone.
