"""Tests for Evaluation, PACE Controller, Metrics, and ONNX Deployment."""

from __future__ import annotations

import pytest
import numpy as np
import torch

from hdml.evaluation.pace_controller import PACEController
from hdml.utils.metrics import get_d4rl_normalized_score
from hdml.utils.rliable_metrics import compute_iqm, stratified_bootstrap_ci, compute_probability_of_improvement
from hdml.deployment.onnx_exporter import export_hdml_to_onnx, infer_native_mamba3


def test_pace_controller() -> None:
    pace = PACEController(threshold=0.5, norm_p=2)
    # Set a plan
    action_chunk = torch.randn(4, 6)
    predicted_states = torch.zeros(4, 10)
    pace.set_new_plan(action_chunk, predicted_states)

    # First action under threshold
    act, trunc = pace.get_next_action(torch.zeros(10))
    assert act is not None
    assert not trunc

    # Reset
    pace.reset()
    assert pace.active_chunk is None


def test_rliable_metrics() -> None:
    scores = np.array([1.0, 1.2, 0.8, 1.5, 0.9, 1.1, 1.3])
    iqm = compute_iqm(scores)
    assert 0.8 <= iqm <= 1.5

    point_est, ci_low, ci_high = stratified_bootstrap_ci(scores, num_bootstraps=100)
    assert np.isfinite(point_est) and np.isfinite(ci_low) and np.isfinite(ci_high)
    assert ci_low <= point_est <= ci_high

    scores_b = scores + 0.2
    p_pt, p_lo, p_hi = compute_probability_of_improvement(scores_b.reshape(1, -1), scores.reshape(1, -1))
    assert all(np.isfinite(v) for v in (p_pt, p_lo, p_hi))
    assert 0.0 <= p_lo <= p_pt <= p_hi <= 1.0


def test_onnx_export(tmp_path) -> None:
    from hdml.models.hdml_model import HDMLModel
    model = HDMLModel(
        prop_dim=17, action_dim=6, d_model=64, num_mamba_layers=2, cfc_units=16,
        use_native_mamba3=False,
    )
    onnx_file = tmp_path / "model.onnx"
    export_hdml_to_onnx(model, str(onnx_file), prop_dim=17, action_dim=6, context_length=10)
    assert onnx_file.exists()


def test_onnx_export_native_mamba3(tmp_path) -> None:
    import torch
    from hdml.models.hdml_model import HDMLModel
    if not torch.cuda.is_available():
        pytest.skip("native Mamba-3 ONNX export requires CUDA")
    model = HDMLModel(
        prop_dim=17, action_dim=6, d_model=64, num_mamba_layers=2, cfc_units=16,
        use_native_mamba3=True, mamba3_headdim=32, d_state=16,
    ).cuda()
    onnx_file = tmp_path / "model_native.onnx"
    export_hdml_to_onnx(model, str(onnx_file), prop_dim=17, action_dim=6, context_length=10)
    assert onnx_file.exists()
    assert onnx_file.stat().st_size > 0


def test_onnx_export_action_only_and_without_cfc(tmp_path) -> None:
    from hdml.models.hdml_model import HDMLModel
    model = HDMLModel(
        prop_dim=17, action_dim=6, d_model=64, num_mamba_layers=2,
        use_cfc=False, action_policy="gaussian", use_critic=False,
        use_native_mamba3=False,
    )
    onnx_file = tmp_path / "model_action_only.onnx"
    res = export_hdml_to_onnx(
        model, str(onnx_file), prop_dim=17, action_dim=6, context_length=10,
        action_only=True, verify=True,
    )
    assert onnx_file.exists()
    assert res["action_only"] is True
    assert res["max_numerical_diff"] < 1e-3


def test_d4rl_normalization_matches_env_family_token() -> None:
    assert get_d4rl_normalized_score("Ant-v4", 0.0) != 0.0
    # "antmaze" must not inherit Ant's reference bounds.
    assert get_d4rl_normalized_score("antmaze-medium-v2", 0.0) == 0.0
    assert get_d4rl_normalized_score("HalfCheetah-v5", 0.0) != 0.0


def test_infer_native_mamba3_from_state_dict() -> None:
    native = {"mamba_backbone.layers.0.mamba.B_norm.weight": torch.zeros(1)}
    legacy = {"mamba_backbone.layers.0.mamba.x_proj.weight": torch.zeros(1)}
    other = {"fusion.prop_encoder.0.weight": torch.zeros(1)}
    assert infer_native_mamba3(native) is True
    assert infer_native_mamba3(legacy) is False
    assert infer_native_mamba3(other) is None

