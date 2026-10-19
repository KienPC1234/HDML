# HDML: Hierarchical Decision Mamba-Liquid

Hierarchical sequence model for real-time robot locomotion and adaptive game AI:
- **Macro Tier**: Mamba-3 linear-time selective state-space model with RoPE positional embeddings ($10 - 20\text{ Hz}$).
- **Micro Tier**: Closed-form Continuous-time (CfC) liquid neural ODE filter ($50 - 300\text{ Hz}$).
- **Inference Runtime**: Exportable to lean ONNX for Unity (`Sentis` / `Microsoft.ML.OnnxRuntime`), Unreal Engine (NNE), and C# / C++ game engines ($< 4\text{ ms}$ on CPU).

## Benchmark (Unitree A1, 12-DoF, RTX 4070 SUPER)

Same 100k-step dataset, same hyper-params, 10 episodes (`results/rebuild_unitree/fair_benchmark.json`):

| Model | Params | Latency | Return | Jerk |
| :--- | :---: | :---: | :---: | :---: |
| **HDML** | 1.42M | 8.60 ms | **1495.3** | 0.0628 |
| Decision RNN | 1.01M | 1.18 ms | 1454.0 | 0.0640 |
| IQL | 0.31M | 0.28 ms | 1452.3 | 0.0623 |
| Decision Transformer | 1.21M | 2.44 ms | 1449.1 | 0.0733 |
| MLP-BC | 0.08M | 0.31 ms | 1343.9 | 0.0663 |
| Diffusion | 0.16M | 6.89 ms | 960.3 | 0.2118 |

## Production ONNX & C# Game Runtime (`tests_csharp/`)

Inference latency and memory footprint measured on CPU with `Microsoft.ML.OnnxRuntime 1.31.0` under .NET 8:

| Model Target | Batch / Context | P50 Latency | Rate | Working Set (RAM) | Parity vs PyTorch |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unitree A1 Locomotion** (`unitree_a1_action_only.onnx`) | 1 / 20 | **1.80 ms** | 556 Hz | 85.0 MB | $3.28 \times 10^{-7}$ |

## Quickstart

```bash
# Environment setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install causal-conv1d --no-build-isolation && pip install mamba-ssm --no-build-isolation
pip install -r requirements.txt && pip install -e .

# Test suite (Python + C# ONNX parity)
pytest tests/ tests_cpu/ -q

# Export a checkpoint to ONNX (deployment/*.onnx is generated, not committed)
python scripts/export_onnx.py --config configs/unitree_a1_cfc_strong.yaml \
  --checkpoint checkpoints/rebuild_unitree/unitree_a1/best_model.pt \
  --output deployment/unitree_a1_action_only.onnx --action-only

# Run C# ONNX benchmark on CPU
dotnet run --project tests_csharp -- --model deployment/unitree_a1_action_only.onnx --prop-dim 35 --action-dim 12 --iters 100

# Benchmark Unitree A1 robot
python scripts/benchmark_unitree_a1.py
```

## Documentation

- [SETUP_AND_DOCS.md](SETUP_AND_DOCS.md): Full installation, hardware setup, and reproducible benchmarks.
- [docs/ONNX_DEPLOYMENT.md](docs/ONNX_DEPLOYMENT.md): ONNX export pipeline, C# Unity Sentis integration, and C++ game engine runtime.

## License

[Apache 2.0](LICENSE).
