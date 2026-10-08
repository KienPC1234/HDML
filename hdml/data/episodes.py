"""Episode-safe NumPy preprocessing shared by training and regression checks."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np


def load_episodes(path: str | Path) -> list[dict[str, np.ndarray]]:
    """Read numeric NPZ trajectories; split flat arrays at termination OR truncation."""
    episodes: list[dict[str, np.ndarray]] = []
    with np.load(path, allow_pickle=False) as ds:
        prefixes = [''] if 'observations' in ds else [
            f'traj_{i}_' for i in sorted({int(k.split('_')[1]) for k in ds.files
                                        if k.startswith('traj_') and k.endswith('_observations')})]
        for prefix in prefixes:
            obs = np.asarray(ds[prefix + 'observations'], dtype=np.float32)
            acts = np.asarray(ds[prefix + 'actions'], dtype=np.float32)
            rews = np.asarray(ds[prefix + 'rewards'], dtype=np.float32).reshape(-1)
            n = len(rews)
            if not n or obs.ndim != 2 or acts.ndim != 2 or len(obs) != n or len(acts) != n:
                raise ValueError(f'Invalid trajectory shapes in {path}: {prefix}')
            if not all(np.isfinite(x).all() for x in (obs, acts, rews)):
                raise ValueError(f'Nonfinite trajectory data in {path}: {prefix}')
            ends = np.zeros(n, dtype=bool)
            for suffix in ('terminals', 'dones', 'truncations', 'timeouts'):
                key = prefix + suffix
                if key in ds:
                    flags = np.asarray(ds[key], dtype=bool).reshape(-1)
                    if flags.shape != (n,):
                        raise ValueError(f'Invalid boundary array: {key}')
                    ends |= flags
            # A per-trajectory archive has an explicit boundary even at time limit.
            ends[-1] = True
            start = 0
            for end in np.flatnonzero(ends) + 1:
                episodes.append({'observations': obs[start:end].copy(),
                                 'actions': acts[start:end].copy(),
                                 'rewards': rews[start:end].copy()})
                start = int(end)
    if not episodes:
        raise ValueError(f'No numeric trajectories found in {path}')
    return episodes


def episode_fingerprint(episode: dict[str, np.ndarray]) -> str:
    """Hash observations/actions/rewards, independent of file names or ordering."""
    digest = hashlib.sha256()
    for key in ('observations', 'actions', 'rewards'):
        value = np.asarray(episode[key], dtype='<f4', order='C')
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def split_episodes(episodes: list[dict[str, np.ndarray]], validation_fraction: float = 0.2,
                   seed: int = 42) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
    """Split whole episodes, grouping exact duplicates to prevent cross-split leakage."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must lie strictly between 0 and 1')
    groups: dict[str, list[dict[str, np.ndarray]]] = {}
    for ep in episodes:
        groups.setdefault(episode_fingerprint(ep), []).append(ep)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError('At least two distinct episodes are required for a held-out split')
    order = np.random.default_rng(seed).permutation(len(keys))
    n_val = min(len(keys) - 1, max(1, int(np.ceil(len(keys) * validation_fraction))))
    val_keys = {keys[i] for i in order[:n_val]}
    train = [ep for k in keys if k not in val_keys for ep in groups[k]]
    val = [ep for k in keys if k in val_keys for ep in groups[k]]
    return train, val


def pack_episodes(episodes: list[dict[str, np.ndarray]], context_length: int,
                  gamma: float = 0.99, scale_return: float = 1.0,
                  state_mean: np.ndarray | None = None,
                  state_std: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Prepare arrays and valid window starts without crossing any episode boundary."""
    if context_length < 1 or scale_return <= 0 or not 0 <= gamma <= 1:
        raise ValueError('Invalid context length, return scale, or discount')
    obs = np.concatenate([ep['observations'] for ep in episodes])
    mean = obs.mean(axis=0) if state_mean is None else np.asarray(state_mean)
    std = obs.std(axis=0) + 1e-6 if state_std is None else np.asarray(state_std)
    if mean.shape != obs.shape[1:] or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError('Invalid state normalization statistics')
    actions, previous, returns, times, starts = [], [], [], [], []
    offset = 0
    for ep in episodes:
        acts, rews = ep['actions'], ep['rewards']
        n = len(rews)
        actions.append(acts)
        previous.append(np.concatenate([np.zeros_like(acts[:1]), acts[:-1]]))
        rtg = np.empty(n, dtype=np.float32)
        accumulator = 0.0
        for t in range(n - 1, -1, -1):
            accumulator = float(rews[t]) + gamma * accumulator
            rtg[t] = accumulator / scale_return
        returns.append(rtg[:, None])
        times.append(np.arange(n, dtype=np.int64))
        starts.extend(range(offset, offset + max(0, n - context_length + 1)))
        offset += n
    if not starts:
        raise ValueError('No episode is long enough for the requested context length')
    return {'states': ((obs - mean) / std).astype(np.float32),
            'actions': np.concatenate(actions), 'previous_actions': np.concatenate(previous),
            'rtgs': np.concatenate(returns), 'timesteps': np.concatenate(times),
            'valid_starts': np.asarray(starts, dtype=np.int64),
            'state_mean': mean.astype(np.float32), 'state_std': std.astype(np.float32)}
