from __future__ import annotations

import logging
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset

from hdml.data.collector import discount_cumsum

logger = logging.getLogger(__name__)


class FastTensorTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Pre-vectorized, contiguous tensor dataset for maximum GPU throughput.

    Pre-materializes all context sequences into contiguous tensor buffers,
    eliminating per-step slicing and Python dictionary overhead.
    """

    def __init__(
        self,
        trajectories: Sequence[dict[str, np.ndarray]],
        context_length: int = 20,
        scale_return: float = 1000.0,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.scale_return = scale_return
        self.trajectories = list(trajectories)

        if len(self.trajectories) == 0:
            raise ValueError("FastTensorTrajectoryDataset cannot be empty.")

        all_states = np.concatenate([traj["observations"] for traj in self.trajectories], axis=0)
        self.prop_dim = all_states.shape[-1]
        self.action_dim = self.trajectories[0]["actions"].shape[-1]

        if state_mean is None:
            self.state_mean = np.mean(all_states, axis=0, dtype=np.float32)
        else:
            self.state_mean = np.asarray(state_mean, dtype=np.float32)

        if state_std is None:
            self.state_std = np.std(all_states, axis=0, dtype=np.float32) + 1e-6
        else:
            self.state_std = np.asarray(state_std, dtype=np.float32) + 1e-6

        # Calculate total valid frames
        total_samples = sum(len(traj["observations"]) for traj in self.trajectories)
        k = self.context_length

        # Preallocate contiguous buffers
        states_buf = np.zeros((total_samples, k, self.prop_dim), dtype=np.float32)
        actions_buf = np.zeros((total_samples, k, self.action_dim), dtype=np.float32)
        rtgs_buf = np.zeros((total_samples, k, 1), dtype=np.float32)
        timesteps_buf = np.zeros((total_samples, k), dtype=np.int64)
        mask_buf = np.zeros((total_samples, k), dtype=np.float32)

        cursor = 0
        for traj in self.trajectories:
            raw_states = traj["observations"]
            raw_actions = traj["actions"]
            raw_rtgs = traj["returns_to_go"]
            raw_timesteps = traj["timesteps"]
            traj_len = len(raw_states)

            norm_states = (raw_states - self.state_mean) / self.state_std
            scaled_rtgs = raw_rtgs / self.scale_return

            for t in range(traj_len):
                end_t = min(t + k, traj_len)
                actual_len = end_t - t

                states_buf[cursor, :actual_len] = norm_states[t:end_t]
                actions_buf[cursor, :actual_len] = raw_actions[t:end_t]
                rtgs_buf[cursor, :actual_len, 0] = scaled_rtgs[t:end_t]
                timesteps_buf[cursor, :actual_len] = raw_timesteps[t:end_t]
                mask_buf[cursor, :actual_len] = 1.0

                cursor += 1

        self.states = torch.from_numpy(states_buf)
        self.actions = torch.from_numpy(actions_buf)
        self.rtgs = torch.from_numpy(rtgs_buf)
        self.timesteps = torch.from_numpy(timesteps_buf)
        self.mask = torch.from_numpy(mask_buf)
        self.target_actions = self.actions.clone()
        self.target_rtgs = self.rtgs.clone()

        logger.info(
            f"FastTensorTrajectoryDataset initialized with {total_samples} pre-vectorized frames "
            f"in contiguous memory (context_length={context_length})."
        )

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": self.states[index],
            "actions": self.actions[index],
            "rtgs": self.rtgs[index],
            "timesteps": self.timesteps[index],
            "mask": self.mask[index],
            "target_actions": self.target_actions[index],
            "target_rtgs": self.target_rtgs[index],
        }


class TrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Standard dynamic PyTorch Dataset for offline RL and sequence modeling."""

    def __init__(
        self,
        trajectories: Sequence[dict[str, np.ndarray]],
        context_length: int = 20,
        scale_return: float = 1000.0,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.scale_return = scale_return
        self.trajectories = list(trajectories)

        if len(self.trajectories) == 0:
            raise ValueError("TrajectoryDataset cannot be initialized with empty trajectories.")

        all_states = np.concatenate([traj["observations"] for traj in self.trajectories], axis=0)
        self.prop_dim = all_states.shape[-1]
        self.action_dim = self.trajectories[0]["actions"].shape[-1]

        if state_mean is None:
            self.state_mean = np.mean(all_states, axis=0, dtype=np.float32)
        else:
            self.state_mean = np.asarray(state_mean, dtype=np.float32)

        if state_std is None:
            self.state_std = np.std(all_states, axis=0, dtype=np.float32) + 1e-6
        else:
            self.state_std = np.asarray(state_std, dtype=np.float32) + 1e-6

        self.indices: list[tuple[int, int]] = []
        for traj_idx, traj in enumerate(self.trajectories):
            traj_len = len(traj["observations"])
            for t in range(traj_len):
                self.indices.append((traj_idx, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        traj_idx, start_t = self.indices[index]
        traj = self.trajectories[traj_idx]
        traj_len = len(traj["observations"])

        end_t = min(start_t + self.context_length, traj_len)
        actual_len = end_t - start_t

        raw_states = traj["observations"][start_t:end_t]
        raw_actions = traj["actions"][start_t:end_t]
        raw_rtgs = traj["returns_to_go"][start_t:end_t]
        raw_timesteps = traj["timesteps"][start_t:end_t]

        norm_states = (raw_states - self.state_mean) / self.state_std
        scaled_rtgs = raw_rtgs / self.scale_return

        k = self.context_length
        padded_states = np.zeros((k, self.prop_dim), dtype=np.float32)
        padded_actions = np.zeros((k, self.action_dim), dtype=np.float32)
        padded_rtgs = np.zeros((k, 1), dtype=np.float32)
        padded_timesteps = np.zeros((k,), dtype=np.int64)
        mask = np.zeros((k,), dtype=np.float32)

        padded_states[:actual_len] = norm_states
        padded_actions[:actual_len] = raw_actions
        padded_rtgs[:actual_len, 0] = scaled_rtgs
        padded_timesteps[:actual_len] = raw_timesteps
        mask[:actual_len] = 1.0

        return {
            "states": torch.from_numpy(padded_states),
            "actions": torch.from_numpy(padded_actions),
            "rtgs": torch.from_numpy(padded_rtgs),
            "timesteps": torch.from_numpy(padded_timesteps),
            "mask": torch.from_numpy(mask),
            "target_actions": torch.from_numpy(padded_actions.copy()),
            "target_rtgs": torch.from_numpy(padded_rtgs.copy()),
        }


class MinariDatasetAdapter:
    """Converts official Minari / D4RL dataset episodes into HDML trajectory format."""

    @staticmethod
    def load_minari_dataset(dataset_name: str, gamma: float = 0.99) -> list[dict[str, np.ndarray]]:
        """Load a Minari dataset and convert it into HDML format.

        Args:
            dataset_name: Name of Minari dataset (e.g. 'antmaze-umaze-v0', 'door-human-v0').
            gamma: Discount factor for RTG computation.

        Returns:
            List of HDML trajectory dictionaries.
        """
        try:
            import minari
        except ImportError as e:
            logger.error(f"minari is required: {e}")
            raise

        logger.info(f"Loading Minari dataset: {dataset_name}")
        minari_data = minari.load_dataset(dataset_name)

        trajectories: list[dict[str, np.ndarray]] = []
        for ep in minari_data.iterate_episodes():
            obs = np.asarray(ep.observations[:-1] if len(ep.observations) > len(ep.actions) else ep.observations, dtype=np.float32)
            act = np.asarray(ep.actions, dtype=np.float32)
            rew = np.asarray(ep.rewards, dtype=np.float32)
            dones = np.asarray(ep.terminations, dtype=bool) | np.asarray(ep.truncations, dtype=bool)

            rtg = discount_cumsum(rew, gamma)
            timesteps = np.arange(len(obs), dtype=np.int64)

            traj = {
                "observations": obs,
                "actions": act,
                "rewards": rew,
                "returns_to_go": rtg,
                "dones": dones,
                "timesteps": timesteps,
                "total_return": np.float32(np.sum(rew)),
            }
            trajectories.append(traj)

        logger.info(f"Loaded {len(trajectories)} trajectories from Minari dataset: {dataset_name}")
        return trajectories
