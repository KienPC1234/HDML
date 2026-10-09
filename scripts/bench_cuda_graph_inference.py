"""Prototype: CUDA-graph-accelerated closed-loop inference for HDML.

Verifies (1) numerical equivalence with eager inference and (2) latency on the
Unitree A1 policy. Static input buffers are reused and filled with copy_ each
control step, then the captured graph is replayed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from hdml.evaluation.quadruped_dog_env import QuadrupedDogEnv
from hdml.models import HDMLModel
from hdml.utils.config import HDMLConfig


class GraphRunner:
    def __init__(self, model: HDMLModel, prop_dim: int, action_dim: int, ctx: int, device):
        self.m = model
        self.device = device
        self.ts = torch.zeros(1, ctx, prop_dim, device=device)
        self.tr = torch.zeros(1, ctx, 1, device=device)
        self.ta = torch.zeros(1, ctx, action_dim, device=device)
        self.tt = torch.arange(ctx, device=device, dtype=torch.int64).unsqueeze(0)
        self.ctx = ctx
        self._capture()

    def _capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                with torch.inference_mode():
                    self.m.get_action(states=self.ts, rtgs=self.tr, actions=self.ta, timesteps=self.tt, hx=None)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(self.graph):
                self.out = self.m.get_action(
                    states=self.ts, rtgs=self.tr, actions=self.ta, timesteps=self.tt, hx=None)

    def __call__(self, ts: np.ndarray, tr: np.ndarray, ta: np.ndarray):
        # copy fresh context into the static buffers, then replay
        self.ts.copy_(torch.from_numpy(ts).to(self.device))
        self.tr.copy_(torch.from_numpy(tr).to(self.device))
        self.ta.copy_(torch.from_numpy(ta).to(self.device))
        self.graph.replay()
        return self.out[0]


def main() -> None:
    cfg = HDMLConfig.from_yaml("configs/unitree_a1_aug.yaml")
    dev = torch.device("cuda")
    ck = torch.load("checkpoints/rebuild_unitree/unitree_a1/best_model.pt", map_location=dev, weights_only=False)
    m = HDMLModel.from_config(cfg.model).to(dev)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    m.deterministic = True; m.num_flow_steps = 4
    sm, ss = np.asarray(ck["state_mean"], np.float32), np.asarray(ck["state_std"], np.float32)
    ctx = cfg.training.context_length
    ad = cfg.model.action_dim

    # 1. equivalence on one random context
    rng = np.random.default_rng(0)
    hs = rng.normal(size=(1, ctx, cfg.model.prop_dim)).astype(np.float32)
    ha = rng.normal(size=(1, ctx, ad)).astype(np.float32)
    hr = np.zeros((1, ctx, 1), np.float32)
    ts = torch.from_numpy(hs).to(dev); ta = torch.from_numpy(ha).to(dev)
    tr = torch.from_numpy(hr).to(dev); tt = torch.arange(ctx, device=dev).unsqueeze(0)
    with torch.inference_mode():
        eager = m.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)[0]
    gr = GraphRunner(m, cfg.model.prop_dim, ad, ctx, dev)
    graph = gr(hs, hr, ha)
    diff = float((eager - graph).abs().max().item())
    print(f"equivalence max|diff| = {diff:.3e}")

    # 1b. pure inference latency: eager vs graph replay (no env step, no sensor read)
    def _eager():
        with torch.inference_mode():
            return m.get_action(states=ts, rtgs=tr, actions=ta, timesteps=tt, hx=None)[0]

    for _ in range(20):
        _eager()
    torch.cuda.synchronize()
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        _eager()
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) * 1000.0 / N
    print(f"eager pure inference: {eager_ms:.3f} ms ({1000.0/eager_ms:.0f} Hz), n={N}")

    for _ in range(20):
        gr(hs, hr, ha)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        gr(hs, hr, ha)
    torch.cuda.synchronize()
    pure_ms = (time.perf_counter() - t0) * 1000.0 / N
    print(f"graph pure inference: {pure_ms:.3f} ms ({1000.0/pure_ms:.0f} Hz), n={N}")

    # 2. closed-loop latency (graph) over a real episode. Two boundaries:
    #    - replay_only: just graph.replay() + static-buffer copies
    #    - end_to_end : normalise sensor + build context + replay + clip action
    env = QuadrupedDogEnv(max_episode_steps=cfg.env.max_episode_steps)
    obs, _ = env.reset(seed=42)
    HS, HA, HR, HT = [], [], [], []
    rtg = float(cfg.env.target_return)
    lat = []
    lat_e2e = []
    for t in range(cfg.env.max_episode_steps):
        t_e2e = time.perf_counter()
        n = (np.asarray(obs, np.float32) - sm) / ss
        HS.append(n); HR.append(rtg / cfg.env.scale_return); HT.append(t)
        if not HA:
            HA.append(np.zeros(ad, np.float32))
        cl = min(len(HS), ctx)
        cts = np.array(HS[-cl:], np.float32)
        cta = np.array(HA[-cl:], np.float32)
        ctr = np.array(HR[-cl:], np.float32).reshape(-1, 1)
        if cl < ctx:
            pad = ctx - cl
            cts = np.vstack([np.zeros((pad, cfg.model.prop_dim), np.float32), cts])
            cta = np.vstack([np.zeros((pad, ad), np.float32), cta])
            ctr = np.vstack([np.zeros((pad, 1), np.float32), ctr])
        torch.cuda.synchronize(); t0 = time.perf_counter()
        a = gr(cts[None], ctr[None], cta[None])
        torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) * 1000)
        a = np.clip(a.cpu().numpy()[0].astype(np.float32), -1, 1)
        lat_e2e.append((time.perf_counter() - t_e2e) * 1000)
        HA.append(a); obs, r, term, trunc, _ = env.step(a); rtg -= float(r)
        if term or trunc:
            break
    env.close()
    loop_ms = float(np.mean(lat))
    e2e_ms = float(np.mean(lat_e2e))
    print(f"graph replay only: {loop_ms:.2f} ms ({1000/loop_ms:.0f} Hz), n={len(lat)}")
    print(f"graph end-to-end : {e2e_ms:.2f} ms ({1000/e2e_ms:.0f} Hz), n={len(lat_e2e)}")

    out_path = "results/rebuild_unitree/cuda_graph_bench.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({
        "checkpoint": "checkpoints/rebuild_unitree/unitree_a1/best_model.pt",
        "device": "cuda",
        "context_length": ctx,
        "equivalence_max_abs_diff": diff,
        "eager_pure_inference_ms": eager_ms,
        "eager_pure_inference_hz": 1000.0 / eager_ms,
        "graph_pure_inference_ms": pure_ms,
        "graph_pure_inference_hz": 1000.0 / pure_ms,
        "speedup_pure_inference": eager_ms / pure_ms,
        "graph_replay_only_ms": loop_ms,
        "graph_replay_only_hz": 1000.0 / loop_ms,
        "graph_end_to_end_ms": e2e_ms,
        "graph_end_to_end_hz": 1000.0 / e2e_ms,
        "pure_inference_samples": N,
        "closed_loop_samples": len(lat),
    }, indent=2), encoding="utf-8")
    print("saved", out_path)


if __name__ == "__main__":
    main()
