from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


class TopologicalMazeRouter:
    """BFS shortest-path router on the labyrinth topological corridor graph."""
    CORRIDOR_NODES: dict[str, tuple[float, float]] = {
        "SW": (-2.8, -2.8),
        "MW": (-2.8, 0.0),
        "NW": (-2.8, 2.8),
        "NC": (0.0, 2.8),
        "NE": (2.8, 2.8),
        "ME": (2.8, 0.0),
        "SE": (2.8, -2.8),
        "SC": (0.0, -2.8),
    }
    GRAPH: dict[str, list[str]] = {
        "SW": ["MW", "SC"],
        "MW": ["SW", "NW"],
        "NW": ["MW", "NC"],
        "NC": ["NW", "NE"],
        "NE": ["NC", "ME"],
        "ME": ["NE", "SE"],
        "SE": ["ME", "SC"],
        "SC": ["SE", "SW"],
    }

    @classmethod
    def get_closest_node(cls, pos: tuple[float, float]) -> str:
        best_node = "SW"
        best_dist = float("inf")
        for name, p in cls.CORRIDOR_NODES.items():
            d = (pos[0] - p[0])**2 + (pos[1] - p[1])**2
            if d < best_dist:
                best_dist = d
                best_node = name
        return best_node

    @classmethod
    def plan_path(cls, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
        start_node = cls.get_closest_node(start)
        goal_node = cls.get_closest_node(goal)

        queue = [[start_node]]
        visited = set()
        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == goal_node:
                waypoints = [cls.CORRIDOR_NODES[n] for n in path]
                waypoints.append((float(goal[0]), float(goal[1])))
                return waypoints
            if node not in visited:
                visited.add(node)
                for neighbor in cls.GRAPH.get(node, []):
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        return [(float(goal[0]), float(goal[1]))]


class UnitreeA1MazeEnv(gym.Env):
    """12-DOF Unitree A1 Quadruped Labyrinth Navigation Environment.

    Strict Sensory Constraint:
      - NO global overhead tracking / NO bird's eye view camera.
      - ONLY standard onboard sensors:
          * 35D Proprioception: Joint angles (12), Joint vels (12), Torso z (1),
                                IMU quat (4), Torso lin/ang vel (6).
          * 16D Onboard Horizontal LiDAR: 16 raycast distances to maze walls [0.1m - 4.0m].
          * 2D Relative Goal Vector: [dx_rel, dy_rel] in the robot's local heading frame.
      - Total Observation Dimension: 35 + 16 + 2 = 53D.
      - Action Dimension: 12 continuous motor controls in [-1, 1] (mapped to PD joint targets).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    WALL_SEGMENTS = [
        # Outer Bounding Box
        (-4.0, 3.9, 4.0, 4.1),    # North
        (-4.0, -4.1, 4.0, -3.9),  # South
        (3.9, -4.0, 4.1, 4.0),    # East
        (-4.1, -4.0, -3.9, 4.0),  # West
        # Hard Multi-Room Inner Dividers & U-Turn Chokepoints
        (-1.6, -1.4, -1.4, 3.0),  # Left Divider
        (1.4, -3.0, 1.6, 1.4),    # Right Divider
        (-1.5, -1.6, 0.0, -1.4),  # Bottom Baffle (Chokepoint)
        (0.0, 1.4, 1.5, 1.6),     # Top Baffle (Chokepoint)
    ]

    def __init__(
        self,
        xml_path: str = "models/unitree_a1/maze_scene.xml",
        frame_skip: int = 10,
        render_mode: str | None = None,
        max_episode_steps: int = 600,
        goal_pos: tuple[float, float] = (2.8, 2.8),
        start_pos: tuple[float, float] = (-2.8, -2.8),
        sensor_noise: float = 0.02,
    ) -> None:
        super().__init__()
        self.xml_path = str(Path(xml_path).resolve())
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.sensor_noise = sensor_noise
        self.step_count = 0

        # Predefined Free-Space Labyrinth Corridor Zones for Dynamic Goal Repositioning
        self.FREE_SPACE_WAYPOINTS: list[tuple[float, float]] = [
            (-2.8, -2.8),  # SW Chamber
            (-2.8, 0.0),   # MW Corridor
            (-2.8, 2.8),   # NW Corner
            (0.0, 2.8),    # North Top Passage
            (2.8, 2.8),    # NE Chamber
            (2.8, 0.0),    # ME Corridor
            (2.8, -2.8),   # SE Chamber
            (0.0, -2.8),   # South Bottom Passage
            (0.0, 0.0),    # Central Chamber
        ]

        # Action space: 12 PD joint actuators in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Observation space: 53-dim onboard sensory state
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)

        self._renderer: mujoco.Renderer | None = None
        if self.render_mode is not None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=960)

    def _compute_raycast_lidar(self, rx: float, ry: float, yaw: float, num_rays: int = 16, max_range: float = 4.0) -> np.ndarray:
        """Compute 16-beam horizontal LiDAR distances from robot torso to maze walls."""
        distances = np.full(num_rays, max_range, dtype=np.float32)
        angles = yaw + np.linspace(-math.pi, math.pi, num_rays, endpoint=False)

        for i, angle in enumerate(angles):
            dx = math.cos(angle)
            dy = math.sin(angle)
            min_dist = max_range

            for (x1, y1, x2, y2) in self.WALL_SEGMENTS:
                # Check 4 bounding line segments of each rectangular wall
                segments = [
                    ((x1, y1), (x2, y1)),
                    ((x2, y1), (x2, y2)),
                    ((x2, y2), (x1, y2)),
                    ((x1, y2), (x1, y1)),
                ]
                for (sx1, sy1), (sx2, sy2) in segments:
                    # Ray-line intersection in 2D
                    denom = dx * (sy2 - sy1) - dy * (sx2 - sx1)
                    if abs(denom) < 1e-6:
                        continue
                    t_val = ((sx1 - rx) * (sy2 - sy1) - (sy1 - ry) * (sx2 - sx1)) / denom
                    u_val = ((sx1 - rx) * dy - (sy1 - ry) * dx) / denom
                    if t_val > 0.05 and 0.0 <= u_val <= 1.0:
                        if t_val < min_dist:
                            min_dist = t_val

            distances[i] = min_dist

        return distances

    def _get_obs(self) -> np.ndarray:
        """Extract 53D onboard sensory state: 35D Proprio + 16D LiDAR + 2D Relative Goal."""
        # 1. Proprioceptive State (35D)
        torso_z = self.data.qpos[2:3]
        torso_quat = self.data.qpos[3:7]
        joint_pos = self.data.qpos[7:19]
        torso_vel = self.data.qvel[0:6]
        joint_vel = self.data.qvel[6:18]
        proprio = np.concatenate([torso_z, torso_quat, joint_pos, torso_vel, joint_vel], dtype=np.float32)

        # Robot global pose
        rx, ry = float(self.data.qpos[0]), float(self.data.qpos[1])
        w, x_q, y_q, z_q = float(torso_quat[0]), float(torso_quat[1]), float(torso_quat[2]), float(torso_quat[3])
        yaw = float(math.atan2(2 * (w * z_q + x_q * y_q), 1 - 2 * (y_q * y_q + z_q * z_q)))

        # 2. 16-beam Onboard LiDAR (16D)
        lidar_ranges = self._compute_raycast_lidar(rx, ry, yaw, num_rays=16, max_range=4.0)

        # 3. Active Subgoal / Waypoint Vector in robot body frame (2D) - Hierarchical Topological Navigation
        if not hasattr(self, "waypoints") or self.waypoints is None:
            self.waypoints = [tuple(self.goal_pos)]
            self.wp_idx = 0

        target_wp = self.waypoints[min(self.wp_idx, len(self.waypoints) - 1)]
        dist_wp = math.sqrt((target_wp[0] - rx)**2 + (target_wp[1] - ry)**2)
        if dist_wp < 0.75 and self.wp_idx < len(self.waypoints) - 1:
            self.wp_idx += 1
            target_wp = self.waypoints[self.wp_idx]

        dx_world = target_wp[0] - rx
        dy_world = target_wp[1] - ry
        # Rotate into robot's local heading frame
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        dx_rel = dx_world * cos_yaw + dy_world * sin_yaw
        dy_rel = -dx_world * sin_yaw + dy_world * cos_yaw

        if self.sensor_noise > 0.0:
            dx_rel += float(self.np_random.normal(0.0, self.sensor_noise))
            dy_rel += float(self.np_random.normal(0.0, self.sensor_noise))

        rel_goal = np.array([dx_rel, dy_rel], dtype=np.float32)

        obs = np.concatenate([proprio, lidar_ranges, rel_goal], dtype=np.float32)
        assert obs.shape == (53,), f"Expected 53D observation, got {obs.shape}"
        return obs

    def set_goal(self, new_goal: tuple[float, float] | np.ndarray) -> None:
        """Dynamically reposition the target goal pillar beacon in the 3D scene."""
        self.goal_pos = np.array(new_goal, dtype=np.float32)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
        if site_id >= 0:
            self.model.site_pos[site_id] = np.array([self.goal_pos[0], self.goal_pos[1], 0.3], dtype=np.float64)

    def sample_random_goal(self) -> np.ndarray:
        """Sample a valid goal position within the labyrinth corridors."""
        idx = int(self.np_random.integers(0, len(self.FREE_SPACE_WAYPOINTS)))
        base_wp = self.FREE_SPACE_WAYPOINTS[idx]
        jitter_x = float(self.np_random.uniform(-0.25, 0.25))
        jitter_y = float(self.np_random.uniform(-0.25, 0.25))
        return np.array([base_wp[0] + jitter_x, base_wp[1] + jitter_y], dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        # Handle dynamic goal options
        if options is not None and "goal" in options:
            self.set_goal(options["goal"])
        elif options is not None and options.get("random_goal", False):
            self.set_goal(self.sample_random_goal())
        else:
            self.set_goal(self.goal_pos)

        # Set initial robot spawn position at start of labyrinth
        spawn_x = self.start_pos[0] + (self.np_random.uniform(-0.1, 0.1) if seed is not None else 0.0)
        spawn_y = self.start_pos[1] + (self.np_random.uniform(-0.1, 0.1) if seed is not None else 0.0)
        self.data.qpos[0] = spawn_x
        self.data.qpos[1] = spawn_y
        self.data.qpos[2] = 0.27
        self.data.qpos[3:7] = np.array([0.7071068, 0.0, 0.0, 0.7071068])  # Face North (+Y) along open corridor

        # Default compliant standing joint angles
        self.data.qpos[7:19] = np.array([
            0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8
        ], dtype=np.float64)
        # Compute hierarchical topological corridor waypoints for active guidance
        self.waypoints = TopologicalMazeRouter.plan_path((spawn_x, spawn_y), tuple(self.goal_pos))
        self.wp_idx = 0

        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {"pos": np.array([spawn_x, spawn_y]), "goal": self.goal_pos}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        act = np.clip(action, -1.0, 1.0)

        # Active LiDAR Collision Avoidance Shield
        w, x_q, y_q, z_q = float(self.data.qpos[3]), float(self.data.qpos[4]), float(self.data.qpos[5]), float(self.data.qpos[6])
        yaw = float(math.atan2(2 * (w * z_q + x_q * y_q), 1 - 2 * (y_q * y_q + z_q * z_q)))
        lidar = self._compute_raycast_lidar(float(self.data.qpos[0]), float(self.data.qpos[1]), yaw, num_rays=16, max_range=4.0)
        left_dist = float(np.min(lidar[11:14]))
        right_dist = float(np.min(lidar[2:5]))
        front_dist = float(np.min(lidar[7:10]))

        shield_steer = 0.0
        # Trust HDML's learned policy to navigate using LiDAR obs and Waypoints
        # Removed aggressive reactive shield that destabilized roll

        # Map 12D action to target joint positions (PD servo control)
        q_target = np.zeros(12, dtype=np.float64)
        for i in range(4):
            hip_mod = float(act[i * 3 + 0]) + (shield_steer if i in [0, 2] else -shield_steer)
            q_target[i * 3 + 0] = 0.25 * float(np.clip(hip_mod, -1.0, 1.0))
            thigh_target = 0.85 + 0.35 * float(act[i * 3 + 1])
            if front_dist < 0.55:
                thigh_target *= 0.70
            q_target[i * 3 + 1] = thigh_target
            q_target[i * 3 + 2] = -1.80 + 0.35 * float(act[i * 3 + 2])
        self.data.ctrl[:] = q_target

        pos_before = np.array([self.data.qpos[0], self.data.qpos[1]], dtype=np.float32)
        dist_before = float(np.linalg.norm(pos_before - self.goal_pos))

        # Physics step (50 Hz control loop)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        pos_after = np.array([self.data.qpos[0], self.data.qpos[1]], dtype=np.float32)
        dist_after = float(np.linalg.norm(pos_after - self.goal_pos))

        obs = self._get_obs()

        # Goal reached condition (< 0.75m distance)
        goal_reached = dist_after < 0.75

        # Rewards:
        # 1. Progress reward towards goal
        progress_reward = 15.0 * (dist_before - dist_after)
        # 2. Goal reach bonus
        goal_bonus = 100.0 if goal_reached else 0.0
        # 3. Upright posture reward
        torso_z = float(obs[0])
        w, x_q, y_q, z_q = float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4])
        roll = float(math.atan2(2 * (w * x_q + y_q * z_q), 1 - 2 * (x_q * x_q + y_q * y_q)))
        pitch = float(math.asin(max(-1.0, min(1.0, 2 * (w * y_q - z_q * x_q)))))
        healthy = (0.12 <= torso_z <= 0.55) and (abs(roll) < 1.40) and (abs(pitch) < 1.40)
        alive_reward = 0.5 if healthy else -5.0
        # 4. Action regularization cost
        ctrl_cost = 0.001 * float(np.sum(np.square(act)))

        reward = float(progress_reward + goal_bonus + alive_reward - ctrl_cost)

        terminated = goal_reached or (not healthy)
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "pos": pos_after,
            "goal": self.goal_pos,
            "dist_to_goal": dist_after,
            "goal_reached": goal_reached,
            "height": torso_z,
            "roll": roll,
            "pitch": pitch,
        }

        return obs, reward, terminated, truncated, info

    def render(self, camera_name: str = "follow_cam") -> np.ndarray | None:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=720, width=960)
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
