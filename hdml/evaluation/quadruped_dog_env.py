from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class QuadrupedDogEnv(gym.Env):
    """Realistic 12-DOF Quadruped Robot Dog environment modeled after Unitree A1/Go1."""

    metadata = {"render_modes": ["rgb_array", "depth_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = "models/unitree_a1/scene.xml",
        frame_skip: int = 10,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.xml_path = str(Path(xml_path).resolve())
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # Action space: 12 motor torques in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Observation space: 35-dim continuous kinematic state
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32)

        self._renderer: mujoco.Renderer | None = None
        if self.render_mode is not None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=960)

    def _get_obs(self) -> np.ndarray:
        """Extract proprioceptive kinematic state from MuJoCo physics."""
        # 1. Torso pos (z only) & quat
        torso_z = self.data.qpos[2:3]
        torso_quat = self.data.qpos[3:7]
        # 2. 12 joint positions
        joint_pos = self.data.qpos[7:19]
        # 3. Torso linear & angular velocity
        torso_vel = self.data.qvel[0:6]
        # 4. 12 joint velocities
        joint_vel = self.data.qvel[6:18]

        obs = np.concatenate([torso_z, torso_quat, joint_pos, torso_vel, joint_vel], dtype=np.float32)
        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        if self.model.nkey > 0:
            self.data.qpos[:] = self.model.key_qpos[0]
        else:
            self.data.qpos[:] = np.array([
                0.0, 0.0, 0.27, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8
            ], dtype=np.float64)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        act = np.clip(action, -1.0, 1.0)

        # Map normalized action [-1, 1] to target joint positions (PD position servos)
        q_target = np.zeros(12, dtype=np.float64)
        for i in range(4):
            q_target[i * 3 + 0] = 0.0 + 0.35 * float(act[i * 3 + 0])
            q_target[i * 3 + 1] = 0.85 + 0.45 * float(act[i * 3 + 1])
            q_target[i * 3 + 2] = -1.75 + 0.50 * float(act[i * 3 + 2])
        self.data.ctrl[:] = q_target

        pos_before = self.data.qpos[0]
        # Step MuJoCo physics frame_skip times (dt = 0.002 * 10 = 0.02s => 50 Hz control loop)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        pos_after = self.data.qpos[0]
        obs = self._get_obs()

        # Rewards:
        # 1. Forward velocity reward
        forward_reward = (pos_after - pos_before) / (self.frame_skip * 0.002)
        # 2. Stability / upright posture reward (torso z in [0.22, 0.30])
        torso_z = float(obs[0])
        healthy_reward = 1.0 if (0.22 <= torso_z <= 0.32) else -1.5
        # 3. Energy efficiency penalty
        ctrl_cost = 0.002 * float(np.sum(np.square(act)))

        reward = float(1.5 * forward_reward + healthy_reward - ctrl_cost)

        terminated = not (0.15 <= torso_z <= 0.36)
        truncated = self.step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, {"forward_vel": forward_reward, "height": torso_z}

    def apply_kick(self, force_xyz: tuple[float, float, float]) -> None:
        """Apply an external physical kick/impact force to the quadruped dog's torso."""
        self.data.qfrc_applied[0:3] += np.array(force_xyz, dtype=np.float64)

    def render(self) -> np.ndarray | None:
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data, camera="track")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
