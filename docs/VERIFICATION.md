# Verification record

Date: 7 October 2026. This record distinguishes executed software checks from pending experiments.

## Executed

- `python -m unittest discover -s tests_cpu -v`: 10 tests passed. These execute the real NumPy preprocessing routines, including reset boundaries, timeout handling, causal previous actions, duplicate-safe splits, train-only normalization and invalid input rejection. Fixtures are synthetic unit-test inputs, not research measurements.
- Python syntax parsing and compileall: 63 Python modules checked, no syntax failures.
- `python scripts/hdml_cli.py --help` and `export-onnx --help`: argument parsers executed successfully; no ONNX export claimed.

The preprocessing test output is retained in `docs/verification_preprocessing.txt`.

## Not executed

- PyTorch model forward/backward, full pytest suite, CUDA compilation, training, model benchmarks, ONNX export/runtime parity, or physical robot tests.
- The assistant runtime does not contain torch, mamba_ssm, ncps, gymnasium or the project-specific interpreter `/data/HDML_Model/.venv/bin/python`.
- An isolated attempt to install CPU torch from the PyTorch CPU wheel index returned `No matching distribution found for torch`. No dependency versions were changed in the repository.

The project AGENTS.md requires actual CUDA model execution before empirical verification. That gate remains unmet. The proposed branch must remain under review until a compatible environment executes the model checks. Passing CPU preprocessing tests is not evidence of policy reward, adaptation speed or hardware safety.

## Changes that require fresh results

Episode splitting and normalization, timestep reset behavior, foundation embedding IDs, standalone subgoal supervision, recurrent history handling, synchronized timing and checkpoint selection differ from the historical protocol. Old tables must not be reused as measurements of this revision. No scores have been invented to replace them.

## Continuation on 8 October 2026

The 10 NumPy preprocessing tests were rerun successfully in the assistant runtime before re-reading the repository runtime rules. This is explicitly not the required project-interpreter/CUDA gate. The required interpreter `/data/HDML_Model/.venv/bin/python` is absent. Static AST parsing after the final edits checked 63 Python files with no syntax errors.

Removed benchmark config/checkpoint v4 fallback and implicit CPU fallback; added pre-timing device synchronization in the foundation latency helper. These final edits have not been executed on the required GPU environment.
