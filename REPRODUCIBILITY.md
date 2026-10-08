# Reproduction status and execution checks

No replacement GPU scores are available for the October 2026 correction branch. Historical results are not comparable without re-running the corrected evaluation. Read `docs/CORRECTIONS.md` before using a checkpoint or figure.

## Local preprocessing checks

```bash
python -m unittest discover -s tests_cpu -v
python -m compileall -q hdml scripts tests tests_cpu
```

These checks require only Python and NumPy and do not evaluate a robot or neural policy.

## Model environment

Use the GPU environment documented by the project or another explicitly verified compatible environment. Record the interpreter, GPU/driver, toolkit, PyTorch, Mamba, CfC, Gymnasium and MuJoCo versions actually used. The prior documentation's combination of different CUDA generations is not a tested installation recipe.

```bash
python -m pip freeze > environment.txt
python -m pytest tests/ -v
```

Do not count absent dependencies or CPU checks as successful CUDA model verification. Do not silently substitute random policies for missing weights.

## Existing-data training and evaluation

Use the single-task commands in README. Keep the data fixed, set an explicit seed and a new output directory for each independent training run, and retain all runs. Equal gradient-step budgets are not equal FLOP budgets. The provided benchmark evaluates one trained checkpoint per architecture, with multiple environment resets. Its confidence intervals concern those evaluation episodes only.

The dataset named `expert` must have its provenance checked by the researcher. A filename does not establish expert quality. The default heuristic/random collectors are not equivalent to an official expert dataset. For the competition, the student must execute and record the research process personally under the applicable rules.

## Foundation path

The corrected buffers split complete episodes and normalize with training-only statistics. Pretraining writes `data_manifest` and `embodiment_indices`. Adapter diagnostics independently hold out target episodes, but those may already have appeared during older pretraining. Strict mode therefore rejects seen targets, matching episode hashes, and missing provenance. `--allow-pretraining-overlap` permits a clearly labeled diagnostic, not a claim of unseen transfer.

The correction changes preprocessing, embeddings, recurrent evaluation and model supervision. Old scores and old weights must not be represented as freshly trained results of the new code.

## Reporting

Save raw episode returns, the exact config, data/checkpoint hashes, versions, and complete logs. State the perturbation in action units, including sensor dropout; do not call it a 50 N external body impulse. Report episode completion as completion, not survival. The unchanged preprint and figures remain historical and require author-led revision based on new evidence.
