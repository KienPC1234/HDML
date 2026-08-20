"""High-Performance Multi-Embodiment Dataset for HDML-Foundation.

Optimized for 100% GPU saturation on NVIDIA RTX 4070 SUPER using
pre-tensorized pinned buffers, zero-copy indexing, and monolithic embodiment batching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class FastEmbodimentBuffer:
    """Pre-tensorized GPU-friendly buffer for a single robot morphology."""

    def __init__(
        self,
        name: str,
        path: str | Path,
        context_length: int = 30,
        gamma: float = 0.99,
        scale_return: float = 1.0,
        embodiment_idx: int = 0,
    ) -> None:
        self.name = name
        self.context_length = context_length
        self.embodiment_idx = embodiment_idx

        ds = np.load(path, allow_pickle=True)
        if "observations" in ds:
            obs_list = [ds["observations"]]
            act_list = [ds["actions"]]
            rew_list = [ds["rewards"]]
            term_list = [ds["terminals"]]
        else:
            traj_indices = sorted(list(set(int(k.split("_")[1]) for k in ds.keys() if k.startswith("traj_"))))
            obs_list = [ds[f"traj_{i}_observations"] for i in traj_indices if f"traj_{i}_observations" in ds]
            act_list = [ds[f"traj_{i}_actions"] for i in traj_indices if f"traj_{i}_actions" in ds]
            rew_list = [ds[f"traj_{i}_rewards"] for i in traj_indices if f"traj_{i}_rewards" in ds]
            term_list = []
            for i in traj_indices:
                if f"traj_{i}_terminals" in ds:
                    term_list.append(ds[f"traj_{i}_terminals"])
                elif f"traj_{i}_dones" in ds:
                    term_list.append(ds[f"traj_{i}_dones"])
                elif f"traj_{i}_observations" in ds:
                    term_list.append(np.zeros(len(ds[f"traj_{i}_observations"]), dtype=bool))

        all_obs = np.concatenate(obs_list, axis=0)
        all_acts = np.concatenate(act_list, axis=0)
        all_rews = np.concatenate(rew_list, axis=0)
        all_terms = np.concatenate(term_list, axis=0)

        # Compute Returns-to-go
        rtgs = np.zeros_like(all_rews, dtype=np.float32)
        cur_rtg = 0.0
        for t in reversed(range(len(all_rews))):
            if all_terms[t]:
                cur_rtg = 0.0
            cur_rtg = all_rews[t] + gamma * cur_rtg
            rtgs[t] = cur_rtg / scale_return

        st_mean = all_obs.mean(axis=0, keepdims=True)
        st_std = all_obs.std(axis=0, keepdims=True) + 1e-6
        norm_obs = (all_obs - st_mean) / st_std

        self.prop_dim = all_obs.shape[1]
        self.action_dim = all_acts.shape[1]
        self.num_steps = len(norm_obs)
        self.num_valid_starts = max(1, self.num_steps - context_length)

        # Pre-convert to contiguous Float32 Tensors (Pinned for fast CUDA async transfer)
        self.states = torch.from_numpy(norm_obs.astype(np.float32)).pin_memory()
        self.actions = torch.from_numpy(all_acts.astype(np.float32)).pin_memory()
        self.rtgs = torch.from_numpy(rtgs.astype(np.float32)).unsqueeze(-1).pin_memory()

    def sample_batch(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor | str | int]:
        """Lightning-fast vectorized batch sampling directly via stride indexing."""
        starts = torch.randint(0, self.num_valid_starts, (batch_size,))
        T = self.context_length

        # Vectorized slice gathering
        idx_grid = starts.unsqueeze(1) + torch.arange(T).unsqueeze(0)  # (B, T)

        batch_states = self.states[idx_grid].to(device, non_blocking=True)
        batch_acts_target = self.actions[idx_grid].to(device, non_blocking=True)
        batch_rtgs = self.rtgs[idx_grid].to(device, non_blocking=True)

        # Causal action shift (a_{t-1})
        batch_acts_in = torch.zeros_like(batch_acts_target)
        batch_acts_in[:, 1:] = batch_acts_target[:, :-1]

        batch_timesteps = idx_grid.to(device, non_blocking=True)

        return {
            "states": batch_states,
            "actions_in": batch_acts_in,
            "actions_target": batch_acts_target,
            "rtgs": batch_rtgs,
            "timesteps": batch_timesteps,
            "embodiment_name": self.name,
            "embodiment_idx": self.embodiment_idx,
        }


class FastMultiEmbodimentManager:
    """Manages all robotic embodiment buffers with round-robin or balanced sampling."""

    def __init__(
        self,
        embodiment_paths: dict[str, str | Path],
        context_length: int = 30,
        gamma: float = 0.99,
        scale_return: float = 1.0,
    ) -> None:
        self.buffers: dict[str, FastEmbodimentBuffer] = {}
        self.embodiment_names = sorted(list(embodiment_paths.keys()))

        for idx, (name, path) in enumerate(embodiment_paths.items()):
            p = Path(path)
            if p.exists():
                self.buffers[name] = FastEmbodimentBuffer(
                    name=name,
                    path=p,
                    context_length=context_length,
                    gamma=gamma,
                    scale_return=scale_return,
                    embodiment_idx=idx,
                )

        self.total_steps = sum(b.num_steps for b in self.buffers.values())

    def get_buffer(self, name: str) -> FastEmbodimentBuffer:
        return self.buffers[name]

    def sample_embodiment_batch(self, embodiment_name: str, batch_size: int, device: torch.device) -> dict[str, Any]:
        return self.buffers[embodiment_name].sample_batch(batch_size, device)
