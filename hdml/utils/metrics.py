from __future__ import annotations

import time
from typing import Callable, Any
import numpy as np
import torch


def compute_action_smoothness(actions: np.ndarray | torch.Tensor) -> float:
    """Compute mean second-order difference (jerk metric / smoothness) of an action sequence.

    Lower values indicate smoother joint actuation.

    Args:
        actions: Array of actions with shape (T, D_a) or (B, T, D_a).

    Returns:
        Mean absolute jerk value across time and action dimensions.
    """
    if isinstance(actions, torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
    else:
        actions_np = np.asarray(actions)

    if actions_np.shape[0] < 3:
        return 0.0

    # First difference: acceleration / velocity change
    first_diff = np.diff(actions_np, axis=0)
    # Second difference: jerk / rate of acceleration change
    second_diff = np.diff(first_diff, axis=0)

    return float(np.mean(np.abs(second_diff)))


def compute_action_rate_of_change(actions: np.ndarray | torch.Tensor) -> float:
    """Compute mean first-order difference (rate of change) of an action sequence.

    Args:
        actions: Array of actions with shape (T, D_a) or (B, T, D_a).

    Returns:
        Mean absolute action change per step.
    """
    if isinstance(actions, torch.Tensor):
        actions_np = actions.detach().cpu().numpy()
    else:
        actions_np = np.asarray(actions)

    if actions_np.shape[0] < 2:
        return 0.0

    first_diff = np.diff(actions_np, axis=0)
    return float(np.mean(np.abs(first_diff)))


def benchmark_inference_latency(
    model_fn: Callable[..., Any],
    sample_inputs: tuple[torch.Tensor, ...],
    num_warmup: int = 20,
    num_iterations: int = 100,
    device: torch.device | str = "cuda",
) -> dict[str, float]:
    """Benchmark inference latency (mean, std, min, max, throughput) on the target device.

    Args:
        model_fn: Forward or step function to evaluate.
        sample_inputs: Tuple of input tensors on the target device.
        num_warmup: Number of warmup executions to prime CUDA kernels and caches.
        num_iterations: Number of timed executions for measurement.
        device: Execution device ('cuda' or 'cpu').

    Returns:
        Dictionary containing latency statistics in milliseconds and throughput (Hz).
    """
    dev = torch.device(device)
    is_cuda = dev.type == "cuda" and torch.cuda.is_available()

    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model_fn(*sample_inputs)
        if is_cuda:
            torch.cuda.synchronize(dev)

    latencies_ms: list[float] = []

    # Measurement
    with torch.inference_mode():
        for _ in range(num_iterations):
            if is_cuda:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = model_fn(*sample_inputs)
                end_event.record()
                torch.cuda.synchronize(dev)
                latencies_ms.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                _ = model_fn(*sample_inputs)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies_arr))
    std_lat = float(np.std(latencies_arr))
    min_lat = float(np.min(latencies_arr))
    max_lat = float(np.max(latencies_arr))
    throughput_hz = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "mean_latency_ms": mean_lat,
        "std_latency_ms": std_lat,
        "min_latency_ms": min_lat,
        "max_latency_ms": max_lat,
        "throughput_hz": throughput_hz,
    }
