# Operational fixes — PACE, CfC memory, CUDA graph (2026-10-20)

Scope: fix three operational integration bugs without changing the architecture,
the submitted datasets, or the submitted checkpoints.

- Base commit reviewed: `7b45d20`
- Fix commits: `e2402be` (PACE/CfC/CUDA graph), `9ae4275` (stateful CfC inside CUDA graph)
- Branch: `fix/pace-cfc-cudagraph-operational`, fast-forwarded into `master`
- GPU: NVIDIA GeForce RTX 4070 SUPER, PyTorch `2.13.0+cu130`
- New artifacts: `results/fix_pace_cfc_cudagraph/` (all currently reference `9ae4275`)

Every number below comes from a run saved under that directory. Old benchmark
artifacts were left untouched.

---

## 1. PACE: which execution mode, which prediction, which trigger

### Execution mode is subgoal-hold, not chunk playback

`HDMLEvaluator` freezes a macro subgoal every `macro_interval` control steps and
then regenerates an action **every step** with `HDMLModel.act_from_subgoal`.
The flow policy still emits an action chunk, but only its first action is used;
the chunk is not executed action-by-action.

Two separate quantities therefore coexist:

| Quantity | Meaning |
| :-- | :-- |
| `chunk_size: 5` | Number of actions emitted by the flow head; used for the flow target and the `s_{t+5}` dynamics target. |
| `macro_interval: 1` | Synchronous mode: replan the subgoal every control step. |
| `macro_interval: 5` | Deployment-style mode: hold one subgoal for 5 control steps. |

`chunk_size=5` and `macro_interval=1` are a valid combination.

### Prediction timing

The forward-dynamics head is trained on `s_{t+c}` with `c = chunk_size`
(`hdml/data/dataset.py`). It is **not** an `s_{t+1}` predictor. The evaluator now
registers the prediction with its absolute due step:

```text
macro plan starts at t  ->  predicted state is s_{t + chunk_size}
                            PACE compares it only at control step t + chunk_size
```

Predictions are kept in a pending queue when `chunk_size > macro_interval`, so a
short macro plan does not silently disable the deviation check. A deviation
trigger clears the queue; a normal expiry keeps it.

### Trigger reasons and instrumentation

`PACEController.should_replan()` returns `(needs_replanning, reason, deviation)`
with `reason` in `{none, deviation, expired, no_plan}`. Every trigger stores:

- absolute control step,
- reason (`deviation` or `expired`),
- observed deviation and threshold,
- plan start step and horizon.

The evaluator adds per-episode counters and the full event list under
`result["pace"]`, and logs each trigger at `INFO`.

With `macro_interval=1`, PACE monitoring is intentionally inactive: the backbone
already replans every step, so there is no held plan to protect.

### Real-model integration measurement

Command:

```bash
.venv/bin/python scripts/bench_pace_integration.py \
  --config configs/unitree_a1_cfc_strong.yaml \
  --checkpoint checkpoints/rebuild_unitree/unitree_a1/best_model.pt \
  --output results/fix_pace_cfc_cudagraph/pace_integration.json \
  --steps 40 --macro-interval 10
```

Result (`pace_integration.json`, checkpoint step 6513, `chunk_size=5`):

| Scenario | Macro backbone calls | Micro action calls | Trigger steps |
| :-- | --: | --: | :-- |
| No PACE | 4 / 40 steps | 36 | — |
| PACE, unreachable threshold | 4 / 40 steps | 36 | 10, 20, 30 (`expired`) |
| PACE, threshold 0 | 8 / 40 steps | 32 | 5, 10, 15, 20, 25, 30, 35 (`deviation`) |

The threshold-0 run replans as often as a deviation can be due (every
`chunk_size` steps), **not** every step. The `s_{t+5}` prediction is therefore
never mistaken for `s_{t+1}`.

### Limitation that must not be overstated

With `chunk_size=5` and `macro_interval=5`, the first possible deviation check is
at step `t+5`, which is exactly the step at which the plan must be refreshed
anyway. PACE therefore cannot produce an **early** reaction inside that 5-step
plan; it can only classify the trigger as deviation rather than expiry at the
same boundary.

Early replanning is possible only when a prediction becomes due before the
current plan expires, for example `macro_interval=10` (trigger at step 5), or a
pending prediction from an older plan that outlives a shorter plan. The
threshold-0 test above demonstrates the mechanism, **not** a noise-rejection
benefit. Claiming a robustness gain would require a perturbation sweep at a
realistic threshold against periodic replanning; that has not been run.

---

## 2. CfC: one memory rule for training and rollout

### Chosen rule: carry `next_hx`, reset at episode start

The evaluator now carries the CfC hidden state across control steps:

- macro replan: `get_action(..., hx=cfc_hx)`
- micro step: `act_from_subgoal(..., hx=cfc_hx)`
- episode start: `cfc_hx = None`

The old guard that rejected `hx` when the Mamba context had more than one step
was removed. The backbone still consumes the full causal window, while the CfC
advances by exactly one token (the window's last position). The context is not
replayed through the CfC at the same time.

`HDMLGraphRunner` now uses the same policy. A shared static `hx` buffer is an
input to every captured graph, `next_hx` is copied back after each replay, and
`reset()` zeroes the state at episode start. `set_hx()` / `get_hx()` expose the
state for parity checks. The previous runner passed `hx=None` and discarded
`next_hx`, so it only represented the old stateless policy.

### Inference check before any retraining

Command:

```bash
.venv/bin/python scripts/check_cfc_inference_consistency.py \
  --config configs/unitree_a1_cfc_strong.yaml \
  --checkpoint checkpoints/rebuild_unitree/unitree_a1/best_model.pt \
  --dataset data/unitree_a1_trajectories.npz \
  --output results/fix_pace_cfc_cudagraph/cfc_consistency.json
```

Result (`cfc_consistency.json`, 32 real windows, context length 20):

| Check | Value |
| :-- | --: |
| Full-sequence CfC vs token-by-token carried state, max abs diff | `1.192e-07` |
| Final hidden-state max abs diff | `0.0` |
| Tolerance | `1e-05` |
| Verdict | **PASS** |
| `get_action` carried state vs reset, max abs diff | `0.1918` |
| `get_action` reset repeat, max abs diff | `0.0` |

The full sequence and the stateful stepwise path agree within float noise, so
**no retraining is required** merely to unify CfC memory usage. The reset and
carry paths are demonstrably different (`0.1918`), and reset is deterministic.

### Clarifications kept explicit

- The CfC input is `[first flow action, Mamba latent]`. It receives the backbone
  latent (`d_model`), not the raw sensor vector described in the conceptual
  report equation.
- `ncps.torch.CfC` uses its default `timespans=1.0` per token. This is a
  normalized unit step, not a physical 20 ms. It was **not** changed to `0.02`,
  because that would alter the ODE dynamics of existing checkpoints.
- Training resets `hx` at each sampled context window; rollout resets at episode
  start and then carries state. This is the normal truncated-sequence training
  convention and is now documented rather than contradicted by the rollout.

---

## 3. CUDA graph: real timesteps, exact short windows, and carried CfC state

### Changes

- `HDMLGraphRunner` accepts real `timesteps` and copies them into the static
  buffer before every replay.
- Zero padding was removed. The runner captures one graph per sequence length
  used (`1 .. context_length`) and fills a same-shape buffer. A 3-step window
  replays a 3-step graph; it is not compared against a padded 20-step graph.
- Inputs longer than `context_length` use the last `context_length` steps, which
  matches the evaluator's rolling window.
- The runner now carries CfC state: a shared static hidden-state buffer is fed
  to every graph, `next_hx` is copied back after each replay, and `reset()`
  zeroes it at episode start. The old `hx=None` / discard-`next_hx` behaviour is
  gone. This is the same policy as the evaluator.

### Equivalence

Command:

```bash
.venv/bin/python scripts/bench_cuda_graph_inference.py \
  --config configs/unitree_a1_cfc_strong.yaml \
  --checkpoint checkpoints/rebuild_unitree/unitree_a1/best_model.pt \
  --output results/fix_pace_cfc_cudagraph/cuda_graph_bench.json \
  --rollout-steps 300 --eval-episodes 10 --eval-max-steps 1000 --samples 500
```

Result (`cuda_graph_bench.json`):

- Eager and graph were compared on identical observation, action, RTG, timestep,
  and **starting CfC state** sequences for **every length 1–20**.
- Max absolute action difference: `0.0` at every length.
- Max absolute `next_hx` difference: `0.0` at every length.
- Reset repeat difference: `0.0`; `hx` after reset: `0.0`.
- Changing absolute timesteps by +1000 changed both eager and graph output by
  `0.4408`; graph-vs-eager difference was `0.0` for both timestep sets. The
  runner is using the supplied timesteps, not a fixed `arange`.
- Independent 300-step rollouts from seed 42, both carrying CfC state: return
  `415.4320940709106` in both modes, max action difference `0.0`.

### Closed-loop evaluation after unifying CfC memory

10 deterministic episodes of 1000 steps, seed base 42, checkpoint step 6513:

| Policy | Return mean | Return std | Jerk | Completion |
| :-- | --: | --: | --: | --: |
| Eager, stateless CfC (old) | `1207.1505` | `0.0` | `0.09708` | `100%` |
| Eager, stateful CfC (new evaluator) | `1206.4399` | `0.0` | `0.11401` | `100%` |
| Graph, stateful CfC (new runner) | `1206.4399` | `0.0` | `0.11401` | `100%` |

Stateful minus stateless return: `-0.7106` (`-0.059%`). The action-jerk proxy
rises from `0.09708` to `0.11401` (`+17.4%`). Graph and eager stateful results
are identical, so the change comes from the CfC policy itself, not from the
CUDA graph.

Caveat: `QuadrupedDogEnv` resets to a fixed keyframe and the policy is
deterministic, so these 10 episodes are repeated runs of the same trajectory
(std `0.0`). This is an operational change check, not a multi-seed benchmark.

### Latency after the fix

| Measurement | Result |
| :-- | --: |
| Eager pure inference | `8.077 ms` (123.8 Hz) |
| Graph pure inference | `0.492 ms` (2033 Hz) |
| Graph speed-up, pure inference | `~16.4x` |
| Graph replay only | `0.542 ms` |
| Graph control step (normalise + context + replay + clip) | `0.625 ms` (1600 Hz) |
| Graph end-to-end (control step + MuJoCo physics) | `1.896 ms` |
| Eager end-to-end | `9.604 ms` (104.1 Hz) |
| Capture time for all 20 lengths | `6.12 s` |

The historical `0.536 ms` graph figure was a **control-step-style** measurement,
not end-to-end including physics. The new record labels both quantities
separately.

---

## 4. Tests added

PACE (`tests/test_evaluation_and_deployment.py`):

- prediction is checked at its own horizon, not `t+1`;
- deviation and normal expiry produce different reasons and counters;
- evaluator integration with a 10-step macro plan does not call the backbone
  every step (3 calls in 12 steps; 9 micro actions).

CfC (`tests/test_core_models.py`):

- full-sequence CfC equals token-by-token calls carrying `next_hx`;
- `hx=None` really starts from the zero state;
- `get_action` accepts a carried state with a multi-step Mamba window and uses it.

CUDA graph (`tests/test_evaluation_and_deployment.py`):

- graph equals eager for short windows (1, 2, 4, 6 steps) with no padding;
- graph equals eager for both action and `next_hx` when CfC state is carried;
- `reset()` returns the state to zero and repeated resets are deterministic;
- changing timesteps changes the output and still matches eager.

Verification run:

```text
pytest tests/ -q       -> 31 passed
pytest tests_cpu/ -q   -> 16 passed
```

---

## 5. Description synchronization

- `chunk_size` (flow actions emitted) is documented separately from
  `macro_interval` (subgoal lifetime).
- The Unitree A1 environments map actions to **target joint angles** for
  position servos (`data.ctrl`), not direct motor torques. The jerk metric is a
  command-smoothness proxy; comments in `quadruped_dog_env.py`, `metrics.py`,
  and the waveform plotting script were corrected.
- New measurements are tied to commit `e2402be`, config
  `configs/unitree_a1_cfc_strong.yaml`, checkpoint
  `checkpoints/rebuild_unitree/unitree_a1/best_model.pt` (step 6513), and the
  stated control mode.
- No dataset or checkpoint was modified. No model was retrained.

### What was not re-run

The existing Unitree/HalfCheetah benchmark numbers were not regenerated. They
remain historical results under their original commits and are not claimed as
measurements of this fix. Only the affected operational checks above were
re-run.

PACE was not tested under sensor noise or force perturbations at a realistic
threshold. The current evidence establishes correct timing and trigger
classification, not a robustness benefit. With `chunk_size=5` and
`macro_interval=5`, PACE cannot react before the scheduled replan step.
