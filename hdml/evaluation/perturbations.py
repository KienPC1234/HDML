from __future__ import annotations

from typing import Any
import numpy as np


class SensorNoisePerturbation:
    """Injects Gaussian noise and random sensor dropout into state observations."""

    def __init__(
        self,
        noise_std: float = 0.05,
        dropout_prob: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.rng = np.random.default_rng(seed)

    def apply(self, obs: np.ndarray) -> np.ndarray:
        """Apply noise and dropouts to observation vector."""
        noisy_obs = obs + self.rng.normal(0.0, self.noise_std, size=obs.shape).astype(np.float32)
        if self.dropout_prob > 0:
            mask = (self.rng.uniform(size=obs.shape) > self.dropout_prob).astype(np.float32)
            noisy_obs = noisy_obs * mask
        return noisy_obs


class ForceImpulsePerturbation:
    """Applies sudden random physical impulse force perturbations during simulation."""

    def __init__(
        self,
        impulse_prob: float = 0.05,
        force_magnitude: float = 2.0,
        seed: int = 42,
    ) -> None:
        self.impulse_prob = impulse_prob
        self.force_magnitude = force_magnitude
        self.rng = np.random.default_rng(seed)

    def apply_action_perturbation(self, action: np.ndarray) -> np.ndarray:
        """Simulate external force perturbation on joint actuation."""
        if self.rng.uniform() < self.impulse_prob:
            impulse = self.rng.uniform(-self.force_magnitude, self.force_magnitude, size=action.shape)
            perturbed_act = np.clip(action + impulse, -1.0, 1.0).astype(np.float32)
            return perturbed_act
        return action
