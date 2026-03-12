from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import register

from hdml.evaluation.evaluator import HDMLEvaluator
from hdml.evaluation.perturbations import SensorNoisePerturbation, ForceImpulsePerturbation
from hdml.evaluation.pace_controller import PACEController
from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.evaluation.unitree_a1_maze_env import UnitreeA1MazeEnv

# Standard Gymnasium Environment Registrations
try:
    register(
        id="UnitreeA1-v0",
        entry_point="hdml.evaluation.quadruped_dog_env:QuadrupedDogEnv",
        max_episode_steps=1000,
    )
except Exception:
    pass

try:
    register(
        id="UnitreeA1Maze-v0",
        entry_point="hdml.evaluation.unitree_a1_maze_env:UnitreeA1MazeEnv",
        max_episode_steps=600,
    )
except Exception:
    pass

__all__ = [
    "HDMLEvaluator",
    "PACEController",
    "QuadrupedDogEnv",
    "UnitreeA1MazeEnv",
    "SensorNoisePerturbation",
    "ForceImpulsePerturbation",
]
