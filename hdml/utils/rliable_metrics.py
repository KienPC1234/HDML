from __future__ import annotations

import logging
from typing import Callable, Any
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def compute_iqm(scores: np.ndarray | list[float]) -> float:
    """Compute Interquartile Mean (IQM) across a 1D or flattened array of scores.
    
    Trims the lowest 25% and highest 25% of values, computing the mean of the middle 50%.
    Robust to outliers and failure cases in deep reinforcement learning benchmarks.
    
    Args:
        scores: 1D array of normalized evaluation scores.
        
    Returns:
        Scalar IQM score.
    """
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if len(arr) == 0:
        return 0.0
    if len(arr) < 4:
        return float(np.mean(arr))
    return float(stats.trim_mean(arr, proportiontocut=0.25))


def stratified_bootstrap_ci(
    task_scores: np.ndarray,
    stat_fn: Callable[[np.ndarray], float] = compute_iqm,
    num_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute Stratified Bootstrap Point Estimate and Confidence Interval.
    
    Resamples runs with replacement independently for each task (stratification),
    then computes the aggregate statistic across all resampled scores.
    
    Args:
        task_scores: 2D array of shape (num_tasks, num_runs) or 1D array (num_runs).
        stat_fn: Summary statistic function (e.g. compute_iqm, np.mean, np.median).
        num_bootstraps: Number of bootstrap resamples (standard: 2000).
        confidence_level: Confidence level (standard: 0.95 for 95% CI).
        seed: Random seed for bootstrap reproducibility.
        
    Returns:
        tuple (point_estimate, ci_lower, ci_upper).
    """
    scores = np.asarray(task_scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
        
    num_tasks, num_runs = scores.shape
    point_est = float(stat_fn(scores.ravel()))
    
    if num_runs <= 1:
        return point_est, point_est, point_est
        
    rng = np.random.default_rng(seed)
    bootstrap_stats = np.empty(num_bootstraps, dtype=np.float64)
    
    for b in range(num_bootstraps):
        # Stratified resampling: resample runs with replacement per task
        idx = rng.integers(0, num_runs, size=(num_tasks, num_runs))
        resampled = np.take_along_axis(scores, idx, axis=1)
        bootstrap_stats[b] = stat_fn(resampled.ravel())
        
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(bootstrap_stats, 100.0 * alpha))
    ci_upper = float(np.percentile(bootstrap_stats, 100.0 * (1.0 - alpha)))
    
    return point_est, ci_lower, ci_upper


def compute_probability_of_improvement(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    num_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute Probability of Improvement P(A > B) with Stratified Bootstrap 95% CI.
    
    Uses the Mann-Whitney U formulation:
    P(A > B) = 1/(M * K_A * K_B) sum_{m} sum_{i,j} ( 1[A_{m,i} > B_{m,j}] + 0.5 * 1[A_{m,i} == B_{m,j}] )
    
    Args:
        scores_a: Scores of algorithm A, shape (num_tasks, num_runs) or (num_runs,).
        scores_b: Scores of baseline B, shape (num_tasks, num_runs) or (num_runs,).
        num_bootstraps: Number of bootstrap iterations.
        confidence_level: Confidence level (default 0.95).
        seed: Random seed.
        
    Returns:
        tuple (prob_point, ci_lower, ci_upper) in [0, 1].
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
        
    num_tasks = min(a.shape[0], b.shape[0])
    k_a = a.shape[1]
    k_b = b.shape[1]
    
    def _pairwise_p(x: np.ndarray, y: np.ndarray) -> float:
        total = 0.0
        for m in range(num_tasks):
            xm = x[m]
            ym = y[m]
            greater = np.sum(xm[:, None] > ym[None, :])
            equal = np.sum(xm[:, None] == ym[None, :])
            total += (greater + 0.5 * equal) / (k_a * k_b)
        return float(total / num_tasks)
        
    point_est = _pairwise_p(a[:num_tasks], b[:num_tasks])
    
    rng = np.random.default_rng(seed)
    boot_probs = np.empty(num_bootstraps, dtype=np.float64)
    
    for step in range(num_bootstraps):
        idx_a = rng.integers(0, k_a, size=(num_tasks, k_a))
        idx_b = rng.integers(0, k_b, size=(num_tasks, k_b))
        res_a = np.take_along_axis(a[:num_tasks], idx_a, axis=1)
        res_b = np.take_along_axis(b[:num_tasks], idx_b, axis=1)
        boot_probs[step] = _pairwise_p(res_a, res_b)
        
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_probs, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_probs, 100.0 * (1.0 - alpha)))
    
    return point_est, ci_lower, ci_upper


def compute_performance_profile(
    task_scores: np.ndarray,
    tau_thresholds: np.ndarray,
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Performance Profile curve rho(tau) with Stratified Bootstrap Bands.
    
    rho(tau) = fraction of runs across all tasks achieving normalized score >= tau.
    
    Args:
        task_scores: Scores of shape (num_tasks, num_runs) or (num_runs,).
        tau_thresholds: Array of normalized performance thresholds (e.g. 0.0 to 1.2).
        num_bootstraps: Number of bootstrap resamples.
        confidence_level: Confidence level for error bands.
        seed: Random seed.
        
    Returns:
        tuple (profile_mean, profile_ci_lower, profile_ci_upper) each of shape (len(tau_thresholds),).
    """
    scores = np.asarray(task_scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
        
    num_tasks, num_runs = scores.shape
    taus = np.asarray(tau_thresholds, dtype=np.float64)
    
    # Point estimate
    profile_mean = np.mean(scores.ravel()[:, None] >= taus[None, :], axis=0)
    
    rng = np.random.default_rng(seed)
    boot_profiles = np.empty((num_bootstraps, len(taus)), dtype=np.float64)
    
    for b in range(num_bootstraps):
        idx = rng.integers(0, num_runs, size=(num_tasks, num_runs))
        resampled = np.take_along_axis(scores, idx, axis=1)
        boot_profiles[b] = np.mean(resampled.ravel()[:, None] >= taus[None, :], axis=0)
        
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = np.percentile(boot_profiles, 100.0 * alpha, axis=0)
    ci_upper = np.percentile(boot_profiles, 100.0 * (1.0 - alpha), axis=0)
    
    return profile_mean, ci_lower, ci_upper


def generate_rliable_summary_plot(
    results_dict: dict[str, np.ndarray],
    env_name: str = "HalfCheetah-v5",
    save_path: str = "plots/rliable_benchmark_profile.png",
) -> str:
    """Generate publication-standard 3-panel statistical figure.
    
    Panel 1: Interquartile Mean (IQM) with 95% Stratified Bootstrap CIs.
    Panel 2: Performance Profiles rho(tau) with shaded 95% confidence bands.
    Panel 3: Probability of Improvement P(HDML > Baseline).
    
    Args:
        results_dict: Dictionary mapping model names to normalized score arrays of shape (num_runs,) or (tasks, runs).
        env_name: Environment title for the plot.
        save_path: Filepath to save the PNG figure.
        
    Returns:
        Path to the saved figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(results_dict))))
    
    # 1. Panel 1: IQM Bar Chart with 95% Bootstrap CIs
    ax1 = axes[0]
    names = list(results_dict.keys())
    iqm_vals = []
    ci_lowers = []
    ci_uppers = []
    
    for name in names:
        pt, lo, hi = stratified_bootstrap_ci(results_dict[name], stat_fn=compute_iqm)
        iqm_vals.append(pt)
        ci_lowers.append(lo)
        ci_uppers.append(hi)
        
    y_pos = np.arange(len(names))
    err_low = np.array(iqm_vals) - np.array(ci_lowers)
    err_high = np.array(ci_uppers) - np.array(iqm_vals)
    
    ax1.barh(y_pos, iqm_vals, xerr=[err_low, err_high], align="center", color=colors[:len(names)], alpha=0.85, capsize=5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("Interquartile Mean (IQM) Normalized Score", fontsize=10, fontweight="bold")
    ax1.set_title(f"Aggregate IQM (95% Stratified Bootstrap CI)\n{env_name}", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    
    # 2. Panel 2: Performance Profiles rho(tau)
    ax2 = axes[1]
    taus = np.linspace(0.0, 1.2, 101)
    
    for i, (name, scores) in enumerate(results_dict.items()):
        prof, lo, hi = compute_performance_profile(scores, taus)
        ax2.plot(taus, prof, label=name, color=colors[i], linewidth=2.0)
        ax2.fill_between(taus, lo, hi, color=colors[i], alpha=0.15)
        
    ax2.set_xlabel("Normalized Score Threshold (tau)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Fraction of Runs >= tau", fontsize=10, fontweight="bold")
    ax2.set_title("Performance Profiles with 95% CI Bands", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, loc="lower left")
    
    # 3. Panel 3: Probability of Improvement P(HDML > Baseline)
    ax3 = axes[2]
    hdml_key = next((k for k in names if "hdml" in k.lower()), names[0])
    hdml_scores = results_dict[hdml_key]
    
    other_names = [k for k in names if k != hdml_key]
    prob_pts = []
    p_err_low = []
    p_err_high = []
    
    for oname in other_names:
        p_pt, p_lo, p_hi = compute_probability_of_improvement(hdml_scores, results_dict[oname])
        prob_pts.append(p_pt)
        p_err_low.append(p_pt - p_lo)
        p_err_high.append(p_hi - p_pt)
        
    if len(other_names) > 0:
        y_pos3 = np.arange(len(other_names))
        ax3.barh(y_pos3, prob_pts, xerr=[p_err_low, p_err_high], align="center", color="teal", alpha=0.85, capsize=5)
        ax3.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="Equivalence (P=0.5)")
        ax3.set_yticks(y_pos3)
        ax3.set_yticklabels(other_names, fontsize=9)
        ax3.invert_yaxis()
        ax3.set_xlabel(f"P({hdml_key} > Baseline)", fontsize=10, fontweight="bold")
        ax3.set_title("Probability of Improvement (Mann-Whitney CI)", fontsize=11, fontweight="bold")
        ax3.set_xlim(0.0, 1.0)
        ax3.grid(axis="x", linestyle="--", alpha=0.5)
        ax3.legend(fontsize=8, loc="lower right")
        
    plt.tight_layout()
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved rliable statistical benchmark plot to: {save_path}")
    return save_path
