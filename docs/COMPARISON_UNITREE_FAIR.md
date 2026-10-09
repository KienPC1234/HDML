# Fair baseline comparison — Unitree A1 (2026-10-09)

All architectures are trained and evaluated under the same protocol, so the
comparison is fair (Phụ lục 9, mục II.3 and III.2).

## Protocol

- Same dataset: `data/unitree_a1_trajectories.npz` (200 trajs, obs 35D, action 12D).
- Same config knobs: AdamW, lr 3e-4, weight decay 1e-4, batch 512, context 20,
  40 epochs, seed 42, warmup + cosine schedule.
- Same model class `HDMLModel` for HDML; baselines via `scripts/train_baselines.py`
  with each model's native objective (DT/RNN/MLP: BC; Diffusion: DDPM eps; IQL:
  expectile value + advantage-weighted BC).
- Same closed-loop evaluation: `scripts/benchmark_unitree_fair.py`, 10 episodes,
  seed 42, deterministic action sampling, identical state normalisation and
  causal `a_{t-1}` convention.

## Result (raw return; the env has its own reward scale, D4RL normalisation does not apply)

| Model | Return | Completion | Jerk J (lower=better) | Latency |
| :-- | --: | --: | --: | --: |
| **HDML (proposed)** | **1495.3** | 100% | 0.0628 | 8.60 ms (116 Hz) |
| Decision RNN | 1454.0 | 100% | 0.0640 | 1.18 ms (848 Hz) |
| IQL | 1452.3 | 100% | 0.0623 | 0.28 ms (3603 Hz) |
| Decision Transformer | 1449.1 | 100% | 0.0733 | 2.44 ms (410 Hz) |
| MLP-BC | 1343.9 | 100% | 0.0663 | 0.31 ms (3263 Hz) |
| Diffusion Policy | 960.3 | 100% | 0.2118 | 6.89 ms (145 Hz) |

Log/JSON: `results/rebuild_unitree/fair_benchmark.json`.

### Reading

- HDML has the **highest return** on the 3D quadruped, above DT, D-RNN and IQL,
  with the joint-lowest jerk outside IQL (which is a near-constant policy).
- Its latency is higher than the small baselines (8.60 ms vs 0.28-2.44 ms) but
  still far above the 40 Hz closed-loop target.
- The single-seed spreads are 0.0 for most policies because the deterministic
  evaluation reproduces the same trajectory across the 10 episode seeds; run
  more training seeds before quoting confidence intervals.

## Real lateral-kick recovery (measured)

`results/unitree_kick.json`, `results/unitree_kick_high.json`:

| Lateral force | Peak torso roll | Survival |
| --: | --: | --: |
| 50 | 7.0 deg | 100% |
| 100 | 8.0 deg | 100% |
| 200 | 25.0 deg | 100% |
| 400 | 51.1 deg | 0% (falls) |

The quadruped holds a measured lateral kick up to 200 (model force units) and
falls at 400. The old, hand-drawn "30 N recovers at 1.0 s" figure is removed.

## Honest caveats

- Force values are in the same model force units the collector used; they are not
  load-cell calibrated newtons.
- The dataset is a scripted CPG collector with noise, not expert RL.
- One training seed so far; add seeds before quoting mean ± std.
