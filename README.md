# HDML: Hierarchical Decision Mamba-Liquid

Hierarchical sequence model for real-time robot locomotion: **Mamba-3** macro planner (linear-time SSM + RoPE) → **CfC ODE** micro actuator filter → **PACE** safety watchdog (replan on impact) → **morphology adapters** for cross-robot transfer.

## Benchmark (Unitree A1, 12-DoF, RTX 4070 SUPER)

Same 100k-step dataset, same hyper-params, 10 episodes. Source: `results/rebuild_unitree/fair_benchmark.json`.

| Model | Params | Latency | Return | Jerk |
| :--- | :---: | :---: | :---: | :---: |
| **HDML** | 1.42M | 8.60 ms | **1495.3** | 0.0628 |
| Decision RNN | 1.01M | 1.18 ms | 1454.0 | 0.0640 |
| IQL | 0.31M | 0.28 ms | 1452.3 | 0.0623 |
| Decision Transformer | 1.21M | 2.44 ms | 1449.1 | 0.0733 |
| MLP-BC | 0.08M | 0.31 ms | 1343.9 | 0.0663 |
| Diffusion | 0.16M | 6.89 ms | 960.3 | 0.2118 |

Notes: 3-seed spread HDML 1465.7 ± 23.0 (`multiseed.json`); CUDA-graph inference 0.38 ms bit-exact (`cuda_graph_bench.json`); kick robustness + transfer + ONNX (parity 1.1e-6, 2.41 ms CPU, 4.98 MB) in `results/unitree_kick_real.json`, `logs/benchmark_foundation_results.json`, `results/onnx_unitree.log`. Details: `SETUP_AND_DOCS.md`, `REPRODUCIBILITY.md`, `docs/`.

## Run

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install causal-conv1d --no-build-isolation && pip install mamba-ssm --no-build-isolation
pip install -r requirements.txt && pip install -e .
pytest tests/ tests_cpu/ -q
python scripts/benchmark_unitree_a1.py
```

## License

[Apache 2.0](LICENSE).
