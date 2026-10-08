"""Tests for Evaluation, PACE Controller, Metrics, and ONNX Deployment."""

from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn as nn

from hdml.evaluation.pace_controller import PACEController
from hdml.utils.rliable_metrics import compute_iqm, stratified_bootstrap_ci, compute_probability_of_improvement
from hdml.deployment.onnx_exporter import export_hdml_to_onnx


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
    assert ci_low <= ci_high

    scores_b = scores + 0.2
    poi = compute_probability_of_improvement(scores_b.reshape(1, -1), scores.reshape(1, -1))
    assert poi is not None


def test_onnx_export(tmp_path) -> None:
    from hdml.models.hdml_model import HDMLModel
    model = HDMLModel(prop_dim=17, action_dim=6, d_model=64, num_mamba_layers=2, cfc_units=16)
    onnx_file = tmp_path / "model.onnx"
    export_hdml_to_onnx(model, str(onnx_file), prop_dim=17, action_dim=6, context_length=10)
    assert onnx_file.exists()
