from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from hdml.utils.rliable_metrics import (
    compute_iqm,
    stratified_bootstrap_ci,
    compute_probability_of_improvement,
    compute_performance_profile,
    generate_rliable_summary_plot,
)


def test_compute_iqm():
    scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    iqm = compute_iqm(scores)
    # Middle 50% of [10..80] is [30, 40, 50, 60], mean = 45.0
    assert abs(iqm - 45.0) < 1.0


def test_stratified_bootstrap_ci():
    task_scores = np.array([
        [80.0, 82.0, 85.0, 78.0, 83.0],
        [90.0, 92.0, 88.0, 95.0, 91.0],
    ])
    pt, lo, hi = stratified_bootstrap_ci(task_scores, stat_fn=compute_iqm, num_bootstraps=500, seed=42)
    assert lo <= pt <= hi
    assert 75.0 <= lo <= 95.0
    assert 80.0 <= hi <= 100.0


def test_probability_of_improvement():
    a = np.array([[90.0, 95.0, 92.0, 94.0]])
    b = np.array([[60.0, 65.0, 62.0, 68.0]])
    prob, lo, hi = compute_probability_of_improvement(a, b, num_bootstraps=500, seed=42)
    assert prob == 1.0
    assert lo >= 0.95

    # Equal distributions should have prob around 0.5
    rng = np.random.default_rng(42)
    eq_a = rng.standard_normal((1, 200))
    eq_b = rng.standard_normal((1, 200))
    prob_eq, _, _ = compute_probability_of_improvement(eq_a, eq_b, num_bootstraps=200, seed=42)
    assert 0.35 <= prob_eq <= 0.65


def test_performance_profile():
    scores = np.array([[0.2, 0.4, 0.6, 0.8, 1.0]])
    taus = np.array([0.0, 0.5, 1.0])
    prof, lo, hi = compute_performance_profile(scores, taus, num_bootstraps=200, seed=42)
    assert prof[0] == 1.0       # All scores >= 0.0
    assert prof[1] == 0.6       # 3 out of 5 scores (0.6, 0.8, 1.0) >= 0.5
    assert prof[2] == 0.2       # 1 out of 5 scores (1.0) >= 1.0
    assert lo[0] <= prof[0] <= hi[0]


def test_generate_rliable_summary_plot(tmp_path: Path):
    res_dict = {
        "HDML (Ours)": np.array([92.0, 95.0, 91.0, 96.0, 94.0, 93.0]),
        "Decision Transformer": np.array([85.0, 82.0, 89.0, 86.0, 84.0, 88.0]),
        "Diffusion Policy": np.array([78.0, 80.0, 75.0, 82.0, 79.0, 81.0]),
    }
    plot_file = tmp_path / "test_rliable.png"
    out = generate_rliable_summary_plot(res_dict, env_name="HalfCheetah-v5", save_path=str(plot_file))
    assert Path(out).exists()
    assert Path(out).stat().st_size > 1000
