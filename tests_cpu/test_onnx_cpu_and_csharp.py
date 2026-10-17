"""CPU and C# ONNX Runtime integration tests for HDML deployment models."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest


# (model_path, prop_dim, action_dim, context_length).
# context_length MUST match the value used at export time: the sequence axis
# is baked into the graph (only batch is dynamic), so feeding any other
# length fails with an ORT Gather out-of-bounds error.
DEPLOYMENT_MODELS: list[tuple[str, int, int, int]] = [
    ("deployment/unitree_a1_action_only.onnx", 35, 12, 20),
    ("deployment/combat_npc_action_only.onnx", 34, 5, 20),
    ("deployment/unitree_a1_full.onnx", 35, 12, 20),
    ("deployment/combat_npc_full.onnx", 34, 5, 20),
    ("deployment/hdml_halfcheetah.onnx", 17, 6, 20),
    ("deployment/combat_npc_action_only.static-int8.onnx", 34, 5, 20),
    ("deployment/combat_npc_action_only.int4.onnx", 34, 5, 20),
    ("deployment/fps_arena_npc_action_only.onnx", 48, 7, 20),
]


@pytest.mark.parametrize(
    "model_path,prop_dim,action_dim,context_length",
    DEPLOYMENT_MODELS,
)
def test_onnx_cpu_inference_parity(model_path: str, prop_dim: int, action_dim: int, context_length: int) -> None:
    path = Path(model_path)
    if not path.exists():
        pytest.skip(f"Model {model_path} not found")

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    rng = np.random.RandomState(42)
    states = rng.randn(1, context_length, prop_dim).astype(np.float32)
    rtgs = np.zeros((1, context_length, 1), dtype=np.float32)
    actions = rng.randn(1, context_length, action_dim).astype(np.float32)
    timesteps = np.zeros((1, context_length), dtype=np.int64)

    outputs = session.run(
        None,
        {
            "states": states,
            "rtgs": rtgs,
            "actions": actions,
            "timesteps": timesteps,
        },
    )

    action_out = outputs[0]
    assert action_out.shape == (1, action_dim)
    for out in outputs:
        assert np.isfinite(out).all(), "ONNX CPU output contains NaN or Inf"


@pytest.mark.parametrize(
    "model_path,prop_dim,action_dim,context_length",
    DEPLOYMENT_MODELS,
)
def test_csharp_onnx_runtime_parity(
    tmp_path: Path, model_path: str, prop_dim: int, action_dim: int, context_length: int
) -> None:
    path = Path(model_path)
    if not path.exists():
        pytest.skip(f"Model {model_path} not found")

    dotnet_bin = shutil.which("dotnet")
    if not dotnet_bin:
        pytest.skip("dotnet SDK is not installed on this system")

    csharp_proj = Path("tests_csharp")
    if not (csharp_proj / "CSharpOnnxBenchmark.csproj").exists():
        pytest.skip("tests_csharp project not found")

    rng = np.random.RandomState(2026)
    states = rng.randn(1, context_length, prop_dim).astype(np.float32)
    rtgs = np.zeros((1, context_length, 1), dtype=np.float32)
    actions = rng.randn(1, context_length, action_dim).astype(np.float32)
    timesteps = np.zeros((1, context_length), dtype=np.int64)

    # 1. Run Python ONNX Runtime
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    py_out = session.run(
        ["action"],
        {
            "states": states,
            "rtgs": rtgs,
            "actions": actions,
            "timesteps": timesteps,
        },
    )[0][0]

    # 2. Dump input for C# runner
    input_file = tmp_path / "input.json"
    dump_out_file = tmp_path / "csharp_out.json"

    input_data = {
        "states": states.flatten().tolist(),
        "rtgs": rtgs.flatten().tolist(),
        "actions": actions.flatten().tolist(),
        "timesteps": timesteps.flatten().tolist(),
    }
    with open(input_file, "w") as f:
        json.dump(input_data, f)

    # 3. Invoke C# Runner
    cmd = [
        "dotnet",
        "run",
        "--project",
        str(csharp_proj),
        "--",
        "--model",
        str(path),
        "--context-length",
        str(context_length),
        "--prop-dim",
        str(prop_dim),
        "--action-dim",
        str(action_dim),
        "--input",
        str(input_file),
        "--dump-output",
        str(dump_out_file),
        "--iters",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"C# execution failed:\n{proc.stdout}\n{proc.stderr}"
    assert dump_out_file.exists(), "C# runner failed to write output file"

    # 4. Compare outputs
    with open(dump_out_file) as f:
        cs_result = json.load(f)
    cs_out = np.array(cs_result["action"], dtype=np.float32)

    assert cs_out.shape == (action_dim,)
    assert np.isfinite(cs_out).all(), "C# output contains NaN or Inf"

    max_diff = float(np.max(np.abs(py_out - cs_out)))
    assert max_diff < 1e-5, f"Discrepancy between Python ORT and C# ORT ({max_diff}) exceeds 1e-5"

