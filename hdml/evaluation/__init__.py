from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import register

from hdml.evaluation.evaluator import HDMLEvaluator
from hdml.evaluation.perturbations import SensorNoisePerturbation, ForceImpulsePerturbation
from hdml.evaluation.pace_controller import PACEController
from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.evaluation.unitree_a1_maze_env import UnitreeA1MazeEnv

# Register only when absent; errors must remain visible.
for env_id, entry_point, limit in (
    ("UnitreeA1-v0", "hdml.evaluation.quadruped_dog_env:QuadrupedDogEnv", 1000),
    ("UnitreeA1Maze-v0", "hdml.evaluation.unitree_a1_maze_env:UnitreeA1MazeEnv", 600),
):
    if env_id not in gym.registry:
        register(id=env_id, entry_point=entry_point, max_episode_steps=limit)

__all__ = [
    "HDMLEvaluator",
    "PACEController",
    "QuadrupedDogEnv",
    "UnitreeA1MazeEnv",
    "SensorNoisePerturbation",
    "ForceImpulsePerturbation",
]

