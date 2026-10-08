#!/usr/bin/env python3
"""Closed-loop rollout with optional action-chunk execution.

Compares replanning every step (chunk_k=1) against executing a k-step action
chunk before replanning (temporal action chunking, as used by ACT / Diffusion
Policy). This is the report's macro/micro two-tier control: the Mamba tier plans
a chunk, the micro tier executes several steps before the next macro replan.

Usage:
    python scripts/rollout_chunk.py --checkpoint <ckpt> --chunks 1 3 5 --episodes 5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import HDMLConfig
from hdml.utils.metrics import get_d4rl_normalized_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rollout_chunk")


@torch.inference_mode()
def rollout(model: HDMLModel, cfg: HDMLConfig, chunk_k: int, episodes: int, seed: int,
            device: torch.device, state_mean, state_std) -> tuple[float, float]:
    env = gym.make(cfg.env.env_name)
    act_dim = env.action_space.shape[0]  # type: ignore
    ctx_len = cfg.training.context_length
    returns: list[float] = []
    for ep in range(episodes):
        torch.manual_seed(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        target_rtg = float(cfg.env.target_return)
        scale_rtg = float(cfg.env.scale_return)
        hs: list[np.ndarray] = [(np.asarray(obs, dtype=np.float32) - state_mean) / state_std]
        ha: list[np.ndarray] = [np.zeros(act_dim, dtype=np.float32)]
        hr: list[float] = [target_rtg / scale_rtg]
        ht: list[int] = [0]
        total = 0.0
        t = 0
        done = False
        while not done and t < cfg.env.max_episode_steps:
            cl = min(len(hs), ctx_len)
            ts = torch.from_numpy(np.array(hs[-cl:], dtype=np.float32)).unsqueeze(0).to(device)
            ta = torch.from_numpy(np.array(ha[-cl:], dtype=np.float32)).unsqueeze(0).to(device)
            tr = torch.from_numpy(np.array(hr[-cl:], dtype=np.float32).reshape(-1, 1)).unsqueeze(0).to(device)
            tt = torch.from_numpy(np.array(ht[-cl:], dtype=np.int64)).unsqueeze(0).to(device)
            with torch.inference_mode():
                _, lat, _, _, flow_ctx, _ = model.encode(states=ts, rtgs=tr, actions=ta, timesteps=tt)
                chunk = model._sample_chunk(flow_ctx)[0, -1]  # (k, action_dim)
            k = min(chunk_k, chunk.shape[0])
            for j in range(k):
                nominal = chunk[j:j+1]
                if model.cfc_filter is not None:
                    filtered, _ = model.cfc_filter(nominal, lat[:, -1, :])
                    a = np.clip(filtered[0].cpu().numpy().astype(np.float32), -1.0, 1.0)
                else:
                    a = np.clip(nominal[0].cpu().numpy().astype(np.float32), -1.0, 1.0)
                obs, r, term, trunc, _ = env.step(a)
                total += float(r)
                target_rtg -= float(r)
                t += 1
                hs.append((np.asarray(obs, dtype=np.float32) - state_mean) / state_std)
                ha.append(a)
                hr.append(target_rtg / scale_rtg)
                ht.append(t)
                if term or trunc:
                    done = True
                    break
        returns.append(total)
    env.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mean = float(np.mean(returns))
    return mean, float(get_d4rl_normalized_score(cfg.env.env_name, mean))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/halfcheetah_v5_default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--chunks", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = HDMLConfig.from_yaml(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HDMLModel.from_config(cfg.model).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    model.deterministic = True
    model.num_flow_steps = 4
    sm = ck.get("state_mean")
    ss = ck.get("state_std")

    for k in args.chunks:
        ret, norm = rollout(model, cfg, k, args.episodes, args.seed, device, sm, ss)
        logger.info("chunk_k=%d: return=%.1f  norm=%.1f", k, ret, norm)


if __name__ == "__main__":
    main()
