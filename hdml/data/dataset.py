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
        chunk_size: int = 4,
        scale_return: float = 1000.0,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        gamma: float = 0.99,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.chunk_size = chunk_size
        self.scale_return = scale_return
        self.gamma = gamma
        self.stride = stride
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

        # Fast vectorized sliding window buffer creation with configurable stride.
        k = self.context_length
        c = self.chunk_size
        all_states_list: list[np.ndarray] = []
        all_actions_list: list[np.ndarray] = []
        all_target_actions_list: list[np.ndarray] = []
        all_target_chunks_list: list[np.ndarray] = []
        all_rtgs_list: list[np.ndarray] = []
        all_time_list: list[np.ndarray] = []
        all_mask_list: list[np.ndarray] = []
        all_reward_chunks_list: list[np.ndarray] = []
        all_next_states_list: list[np.ndarray] = []

        for traj in self.trajectories:
            raw_states = (traj["observations"].astype(np.float32) - self.state_mean) / self.state_std
            raw_actions = traj["actions"].astype(np.float32)
            raw_rtgs = (traj["returns_to_go"].astype(np.float32) / self.scale_return).reshape(-1, 1)
            raw_time = traj["timesteps"].astype(np.int64)
            traj_len = len(raw_states)

            # Pad states, actions, and RTGs for sliding windows
            pad_states = np.pad(raw_states, ((0, k), (0, 0)), mode="constant")
            # Causal action input padding: a_{-1} = 0
            pad_actions = np.pad(raw_actions, ((1, k), (0, 0)), mode="constant")
            tgt_pad_actions = np.pad(raw_actions, ((0, k), (0, 0)), mode="constant")
            pad_rtgs = np.pad(raw_rtgs, ((0, k), (0, 0)), mode="constant")
            pad_time = np.pad(raw_time, (0, k), mode="constant")

            s_windows = np.lib.stride_tricks.sliding_window_view(pad_states, (k, self.prop_dim))[:traj_len, 0, :, :]
            a_windows = np.lib.stride_tricks.sliding_window_view(pad_actions, (k, self.action_dim))[:traj_len, 0, :, :]
            tgt_a_windows = np.lib.stride_tricks.sliding_window_view(tgt_pad_actions, (k, self.action_dim))[:traj_len, 0, :, :]
            
            # Action Chunking (HiQC): Extract a chunk of size `c` for each timestep in the context window
            tgt_chunk_pad = np.pad(raw_actions, ((0, k + c), (0, 0)), mode="constant")
            chunk_strides = tgt_chunk_pad.strides
            chunk_shape = (traj_len, k, c, self.action_dim)
            chunk_strd = (chunk_strides[0], chunk_strides[0], chunk_strides[0], chunk_strides[1])
            tgt_chunks = np.lib.stride_tricks.as_strided(tgt_chunk_pad, shape=chunk_shape, strides=chunk_strd)
            
            r_windows = np.lib.stride_tricks.sliding_window_view(pad_rtgs, (k, 1))[:traj_len, 0, :, :]
            t_windows = np.lib.stride_tricks.sliding_window_view(pad_time, k)[:traj_len]

            # HiQC Bellman target: discounted c-step reward sum R_c[t] = sum_m gamma^m r_{t+m}
            if "rewards" in traj:
                scaled_rewards = traj["rewards"].astype(np.float32) / self.scale_return
            else:
                rtg = traj["returns_to_go"].astype(np.float32)
                raw_rewards = rtg.copy()
                raw_rewards[:-1] = rtg[:-1] - self.gamma * rtg[1:]
                raw_rewards[-1] = rtg[-1]
                scaled_rewards = raw_rewards / self.scale_return
            r_c = np.zeros(traj_len, dtype=np.float32)
            for m in range(c):
                shifted = np.zeros(traj_len, dtype=np.float32)
                if m < traj_len:
                    shifted[: traj_len - m] = scaled_rewards[m:]
                r_c += (self.gamma ** m) * shifted
            pad_rc = np.pad(r_c, (0, k), mode="constant")
            reward_chunks = np.lib.stride_tricks.sliding_window_view(pad_rc, k)[:traj_len].reshape(traj_len, k, 1)

            # Next state after the action chunk: s_{t+c}
            next_state_arr = np.zeros_like(raw_states, dtype=np.float32)
            if traj_len > c:
                next_state_arr[: traj_len - c] = raw_states[c:]
            pad_ns = np.pad(next_state_arr, ((0, k), (0, 0)), mode="constant")
            next_states = np.lib.stride_tricks.sliding_window_view(pad_ns, (k, self.prop_dim))[:traj_len, 0, :, :]

            # Vectorized mask construction
            steps_remaining = np.maximum(0, np.minimum(k, traj_len - np.arange(traj_len)))
            col_indices = np.arange(k)
            masks = (col_indices < steps_remaining[:, None]).astype(np.float32)

            st = self.stride
            all_states_list.append(s_windows[::st])
            all_actions_list.append(a_windows[::st])
            all_target_actions_list.append(tgt_a_windows[::st])
            all_target_chunks_list.append(tgt_chunks[::st])
            all_rtgs_list.append(r_windows[::st])
            all_time_list.append(t_windows[::st])
            all_mask_list.append(masks[::st])
            all_reward_chunks_list.append(reward_chunks[::st])
            all_next_states_list.append(next_states[::st])

        states_buf = np.ascontiguousarray(np.concatenate(all_states_list, axis=0), dtype=np.float32)
        actions_buf = np.ascontiguousarray(np.concatenate(all_actions_list, axis=0), dtype=np.float32)
        target_actions_buf = np.ascontiguousarray(np.concatenate(all_target_actions_list, axis=0), dtype=np.float32)
        target_chunks_buf = np.ascontiguousarray(np.concatenate(all_target_chunks_list, axis=0), dtype=np.float32)
        rtgs_buf = np.ascontiguousarray(np.concatenate(all_rtgs_list, axis=0), dtype=np.float32)
        timesteps_buf = np.ascontiguousarray(np.concatenate(all_time_list, axis=0), dtype=np.int64)
        mask_buf = np.ascontiguousarray(np.concatenate(all_mask_list, axis=0), dtype=np.float32)
        reward_chunks_buf = np.ascontiguousarray(np.concatenate(all_reward_chunks_list, axis=0), dtype=np.float32)
        next_states_buf = np.ascontiguousarray(np.concatenate(all_next_states_list, axis=0), dtype=np.float32)

        self.states = torch.from_numpy(states_buf)
        self.actions = torch.from_numpy(actions_buf)
        self.rtgs = torch.from_numpy(rtgs_buf)
        self.timesteps = torch.from_numpy(timesteps_buf)
        self.mask = torch.from_numpy(mask_buf)
        # Prediction targets are the un-shifted actions of the same window positions;
        # the shifted input actions are stored in self.actions.
        self.target_actions = torch.from_numpy(target_actions_buf)
        self.target_chunks = torch.from_numpy(target_chunks_buf)
        self.target_rtgs = self.rtgs.clone()
        self.reward_chunks = torch.from_numpy(reward_chunks_buf)
        self.next_states = torch.from_numpy(next_states_buf)

        total_samples = self.states.shape[0]
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
            "target_chunks": self.target_chunks[index],
            "target_rtgs": self.target_rtgs[index],
            "reward_chunks": self.reward_chunks[index],
            "next_states": self.next_states[index],
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

        # Causal action-input convention: input action at position j is the action of
        # the *previous* step (a_{start+j-1}), target is a_{start+j}. No leakage.
        if actual_len > 0:
            if start_t > 0:
                prev_action = traj["actions"][start_t - 1 : start_t]
            else:
                prev_action = np.zeros((1, self.action_dim), dtype=np.float32)
            input_actions = np.concatenate([prev_action, raw_actions[:-1]], axis=0)
        else:
            input_actions = np.zeros((0, self.action_dim), dtype=np.float32)

        k = self.context_length
        padded_states = np.zeros((k, self.prop_dim), dtype=np.float32)
        padded_actions = np.zeros((k, self.action_dim), dtype=np.float32)
        padded_rtgs = np.zeros((k, 1), dtype=np.float32)
        padded_timesteps = np.zeros((k,), dtype=np.int64)
        mask = np.zeros((k,), dtype=np.float32)

        padded_states[:actual_len] = norm_states
        padded_actions[:actual_len] = input_actions
        padded_rtgs[:actual_len, 0] = scaled_rtgs
        padded_timesteps[:actual_len] = raw_timesteps
        mask[:actual_len] = 1.0

        target_actions = np.zeros((k, self.action_dim), dtype=np.float32)
        target_actions[:actual_len] = raw_actions

        return {
            "states": torch.from_numpy(padded_states),
            "actions": torch.from_numpy(padded_actions),
            "rtgs": torch.from_numpy(padded_rtgs),
            "timesteps": torch.from_numpy(padded_timesteps),
            "mask": torch.from_numpy(mask),
            "target_actions": torch.from_numpy(target_actions),
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
