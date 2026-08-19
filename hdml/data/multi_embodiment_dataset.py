"""Multi-Embodiment Dataset Loader for HDML-Foundation.

Handles heterogeneous state and action dimensions across multiple robot morphologies
with balanced batching and dynamic embodiment indexing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MultiEmbodimentDataset(Dataset):
    """Multi-embodiment trajectory dataset for foundation pre-training."""

    def __init__(
        self,
        embodiment_paths: dict[str, str | Path],
        context_length: int = 30,
        gamma: float = 0.99,
        scale_return: float = 1.0,
    ) -> None:
        """Args:

        embodiment_paths: Dict mapping embodiment_name -> npz_path
        context_length: Sequence window length
        gamma: Discount factor
        scale_return: Scaling factor for returns
        """
        super().__init__()
        self.context_length = context_length
        self.gamma = gamma
        self.scale_return = scale_return
        self.embodiment_names = sorted(list(embodiment_paths.keys()))
        self.embodiment_to_idx = {name: idx for idx, name in enumerate(self.embodiment_names)}

        self.embodiment_data: dict[str, dict[str, Any]] = {}
        self.samples_per_embodiment: dict[str, int] = {}
        self.indices_map: list[tuple[str, int]] = []

        for name, path in embodiment_paths.items():
            p = Path(path)
            if not p.exists():
                continue
            ds = np.load(p, allow_pickle=True)
            
            # Check format: flat or trajectory-based
            if "observations" in ds:
                obs_list = [ds["observations"]]
                act_list = [ds["actions"]]
                rew_list = [ds["rewards"]]
                term_list = [ds["terminals"]]
            else:
                # traj_i_* format
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

            # Compute RTGs
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

            n_samples = max(0, len(norm_obs) - context_length + 1)
            self.embodiment_data[name] = {
                "states": norm_obs.astype(np.float32),
                "actions": all_acts.astype(np.float32),
                "rtgs": rtgs.astype(np.float32),
                "prop_dim": all_obs.shape[1],
                "action_dim": all_acts.shape[1],
                "num_samples": n_samples,
            }
            self.samples_per_embodiment[name] = n_samples

            for i in range(n_samples):
                self.indices_map.append((name, i))

    def __len__(self) -> int:
        return len(self.indices_map)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        name, start_idx = self.indices_map[idx]
        data = self.embodiment_data[name]
        T = self.context_length

        st_slice = data["states"][start_idx : start_idx + T]
        act_slice = data["actions"][start_idx : start_idx + T]
        rtg_slice = data["rtgs"][start_idx : start_idx + T]

        # Causal action shift: input is a_{t-1}, target is a_t
        act_in = np.zeros_like(act_slice)
        act_in[1:] = act_slice[:-1]

        timesteps = np.arange(start_idx, start_idx + T, dtype=np.int64)

        return {
            "states": torch.from_numpy(st_slice),
            "actions_in": torch.from_numpy(act_in),
            "actions_target": torch.from_numpy(act_slice),
            "rtgs": torch.from_numpy(rtg_slice).unsqueeze(-1),
            "timesteps": torch.from_numpy(timesteps),
            "embodiment_name": name,
            "embodiment_idx": self.embodiment_to_idx[name],
        }


def collate_multi_embodiment(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates samples grouping them by embodiment name for efficient adapter routing."""
    by_embodiment: dict[str, list[dict[str, Any]]] = {}
    for item in batch:
        name = item["embodiment_name"]
        if name not in by_embodiment:
            by_embodiment[name] = []
        by_embodiment[name].append(item)

    batched_by_embodiment: dict[str, dict[str, torch.Tensor | int | str]] = {}
    for name, items in by_embodiment.items():
        batched_by_embodiment[name] = {
            "states": torch.stack([x["states"] for x in items], dim=0),
            "actions_in": torch.stack([x["actions_in"] for x in items], dim=0),
            "actions_target": torch.stack([x["actions_target"] for x in items], dim=0),
            "rtgs": torch.stack([x["rtgs"] for x in items], dim=0),
            "timesteps": torch.stack([x["timesteps"] for x in items], dim=0),
            "embodiment_name": name,
            "embodiment_idx": items[0]["embodiment_idx"],
        }
    return batched_by_embodiment
