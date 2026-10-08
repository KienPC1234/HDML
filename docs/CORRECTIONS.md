# Engineering corrections

This is a software review record, not a competition report or a set of new experimental findings.

## Changes in the correction branch

1. Foundation NPZ preprocessing preserves explicit trajectory ends, terminals, timeouts, and truncations. Return-to-go and timesteps restart per episode; previous actions are correctly preserved inside windows.
2. Whole-episode splitting groups exact duplicates before normalization. Validation uses training statistics. Windows include the last valid start and cannot cross an episode boundary.
3. Foundation training records split hashes, normalization, and embodiment IDs; best checkpoints are selected by held-out action loss. Missing required datasets fail explicitly.
4. Adapter evaluation records training loss separately from held-out prediction loss, before and after fitting. Seen or unverified pretraining exposure requires explicit opt-in and is labeled. It does not claim closed-loop success or few-shot sample efficiency.
5. Every shared foundation parameter is frozen during adapter fitting. CUDA autocast and synchronization are conditional on the selected device.
6. Comparative benchmark aborts on missing checkpoints and missing HDML normalization; GPU timing synchronizes on both boundaries. Output JSON retains raw episode returns and checkpoint hashes. Completion replaces the misleading survival label.
7. Windowed rollout passes fresh recurrent state instead of replaying overlapping history with a prior final state. Unsupported dual-rate inference raises a descriptive error.
8. Model documentation describes the actual Mamba-1/RoPE/output-gate computation. Unused standalone flow/critic/value modules are excluded from optimization, while their parameter names remain for old checkpoint loading.
9. Standalone training no longer regularizes a dimension-mismatched subgoal toward zero. The first-difference smoothness penalty is described as such, rather than second-order Grad-CAPS. This change requires retraining before claiming comparable results.
10. Baseline and HDML training defaults align on stride 1, and explicit training seed options support independently executed runs. A baseline checkpoint named best is now copied from the checkpoint selected on validation, not overwritten by the final epoch.
11. Random-policy foundation collection seeds its action space, records truncation, and is labeled accurately. It remains random-policy data.
12. ONNX declares a fixed context length, reports verification honestly, and the CLI passes a supported config argument. Numerical equivalence to the CUDA implementation remains unverified.

## Historical artifacts

Existing `results/*.txt`, plots, videos, the preprint, DOI record, and v1.0.0 weights were not regenerated. They must not be cited as evidence for this revision. The original artifacts remain available for provenance; README and the paper directory identify their historical status.

## Outstanding evidence

- Real CUDA forward/backward, configured-model tests, and training smoke tests.
- Independent training runs with retained datasets, logs, seeds, versions, checkpoints, and raw rollout results.
- Strict target exclusion during pretraining if unseen-embodiment transfer is investigated.
- Closed-loop task metrics after adaptation; action prediction error alone is insufficient.
- Actual physical robot experiments before any hardware deployment or safety claims.
- Independent review of the mathematical formulations and the remaining experimental modules. This patch is not a proof that every repository path is correct.

## 8 October continuation

- Comparative benchmark now rejects a missing requested config/checkpoint without silently loading v4 artifacts. It honors the requested torch device and fails if CUDA was requested but unavailable.
- Foundation latency helper synchronizes the selected device immediately before measurement.
- These changes are source-reviewed and syntax-checked only, pending GPU execution.
