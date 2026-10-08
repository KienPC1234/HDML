"""Episode-safe, split-aware buffers for multi-embodiment experiments."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from hdml.data.episodes import load_episodes, split_episodes, pack_episodes, episode_fingerprint


class FastEmbodimentBuffer:
    def __init__(self, name: str, path: str | Path, context_length: int = 30,
                 gamma: float = 0.99, scale_return: float = 1.0,
                 embodiment_idx: int = 0, split: str = 'train',
                 validation_fraction: float = 0.2, split_seed: int = 42) -> None:
        if split not in ('train', 'validation'):
            raise ValueError('split must be train or validation')
        self.name, self.context_length, self.embodiment_idx = name, context_length, embodiment_idx
        train, validation = split_episodes(load_episodes(path), validation_fraction, split_seed)
        train_obs = np.concatenate([ep['observations'] for ep in train])
        selected = train if split == 'train' else validation
        packed = pack_episodes(selected, context_length, gamma, scale_return,
                               train_obs.mean(axis=0), train_obs.std(axis=0) + 1e-6)
        self.state_mean, self.state_std = packed.pop('state_mean'), packed.pop('state_std')
        self.episode_hashes = [episode_fingerprint(ep) for ep in selected]
        self.num_episodes = len(selected)
        self.num_steps, self.prop_dim = packed['states'].shape
        self.action_dim = packed['actions'].shape[1]
        for key, value in packed.items():
            tensor = torch.from_numpy(np.ascontiguousarray(value))
            if torch.cuda.is_available():
                tensor = tensor.pin_memory()
            setattr(self, key, tensor)
        self.num_valid_starts = len(self.valid_starts)

    def sample_batch(self, batch_size: int, device: torch.device,
                     generator: torch.Generator | None = None) -> dict[str, Any]:
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        choices = torch.randint(self.num_valid_starts, (batch_size,), generator=generator)
        idx = self.valid_starts[choices, None] + torch.arange(self.context_length)[None, :]
        return {'states': self.states[idx].to(device),
                'actions_in': self.previous_actions[idx].to(device),
                'actions_target': self.actions[idx].to(device),
                'rtgs': self.rtgs[idx].to(device),
                'timesteps': self.timesteps[idx].to(device),
                'embodiment_name': self.name, 'embodiment_idx': self.embodiment_idx}


class FastMultiEmbodimentManager:
    def __init__(self, embodiment_paths: dict[str, str | Path], context_length: int = 30,
                 gamma: float = 0.99, scale_return: float = 1.0,
                 split: str = 'train', split_seed: int = 42) -> None:
        self.embodiment_names = sorted(embodiment_paths)
        self.buffers = {name: FastEmbodimentBuffer(
            name, embodiment_paths[name], context_length, gamma, scale_return, idx,
            split=split, split_seed=split_seed)
            for idx, name in enumerate(self.embodiment_names)}
        self.total_steps = sum(b.num_steps for b in self.buffers.values())

    def get_buffer(self, name: str) -> FastEmbodimentBuffer:
        return self.buffers[name]

    def sample_embodiment_batch(self, embodiment_name: str, batch_size: int,
                                device: torch.device) -> dict[str, Any]:
        return self.buffers[embodiment_name].sample_batch(batch_size, device)
