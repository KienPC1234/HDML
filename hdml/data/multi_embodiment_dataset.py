"""Episode-safe, split-aware buffers for multi-embodiment experiments."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import numpy as np
import torch
from hdml.data.episodes import load_episodes, split_episodes, pack_episodes, episode_fingerprint

logger = logging.getLogger(__name__)


class FastEmbodimentBuffer:
    def __init__(self, name: str, path: str | Path, context_length: int = 30,
                 gamma: float = 0.99, scale_return: float = 1.0,
                 embodiment_idx: int = 0, split: str = 'train',
                 validation_fraction: float = 0.2, split_seed: int = 42) -> None:
        if split not in ('train', 'validation'):
            raise ValueError('split must be train or validation')
        self.name, self.embodiment_idx = name, embodiment_idx
        train, validation = split_episodes(load_episodes(path), validation_fraction, split_seed)
        train_obs = np.concatenate([ep['observations'] for ep in train])
        selected = train if split == 'train' else validation
        # Quality gate: drop degenerate episodes too short for any temporal
        # learning (a 1-step episode would drag the whole split to context 1).
        # Dropped episodes are logged with counts for the data report.
        MIN_EP_LEN = 8
        kept = [ep for ep in selected if len(ep['rewards']) >= MIN_EP_LEN]
        n_dropped = len(selected) - len(kept)
        if n_dropped:
            logger.warning(
                f"Embodiment '{name}' ({split}): dropped {n_dropped}/{len(selected)} "
                f"degenerate episodes shorter than {MIN_EP_LEN} steps.")
        if not kept:
            raise ValueError(f"Embodiment '{name}' ({split}): no episodes >= {MIN_EP_LEN} steps")
        selected = kept
        # Per-embodiment effective context: some embodiments (e.g. pendulum,
        # hopper) contain episodes shorter than the requested context. Shrinking
        # the window to the split's minimum episode length keeps every window
        # on real transitions (no zero-padding fake data). Batches are sampled
        # per-embodiment, so variable T across embodiments is safe.
        min_len = min(len(ep['rewards']) for ep in selected)
        self.context_length = max(1, min(context_length, min_len))
        if self.context_length < context_length:
            logger.warning(
                f"Embodiment '{name}' ({split}): {sum(1 for ep in selected if len(ep['rewards']) < context_length)}/"
                f"{len(selected)} episodes shorter than context_length={context_length}; "
                f"using effective context_length={self.context_length} on real data only.")
        packed = pack_episodes(selected, self.context_length, gamma, scale_return,
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
        # Buffers are pinned (pin_memory) so non_blocking H2D copies overlap GPU work.
        return {'states': self.states[idx].to(device, non_blocking=True),
                'actions_in': self.previous_actions[idx].to(device, non_blocking=True),
                'actions_target': self.actions[idx].to(device, non_blocking=True),
                'rtgs': self.rtgs[idx].to(device, non_blocking=True),
                'timesteps': self.timesteps[idx].to(device, non_blocking=True),
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
