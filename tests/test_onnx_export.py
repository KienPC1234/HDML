from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import ModelConfig
from hdml.deployment.onnx_exporter import export_hdml_to_onnx


def test_hdml_onnx_export_and_numerical_parity():
    """Verify that HDML with Mamba-3 and CfC exports to ONNX with tight numerical parity."""
    cfg = ModelConfig(
        prop_dim=6,
        action_dim=2,
        d_model=32,
        d_state=8,
        d_conv=2,
        expand=2,
        num_mamba_layers=2,
        d_subgoal=16,
        cfc_units=16,
    )
    model = HDMLModel.from_config(cfg).eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "test_hdml.onnx"

        res = export_hdml_to_onnx(
            model=model,
            output_path=onnx_file,
            prop_dim=6,
            action_dim=2,
            context_length=10,
            opset_version=17,
            verify=True,
        )

        assert onnx_file.exists()
        assert res["verified"] is True
        assert res["max_numerical_diff"] < 1e-3
        assert res["file_size_mb"] > 0.0
