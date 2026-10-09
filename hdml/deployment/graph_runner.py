"""CUDA-graph accelerated closed-loop inference for HDML.

The HDML policy is a small network (about 1.4M parameters) run one step at a
time. Its GPU kernels execute in roughly 0.3 ms, but eager inference costs about
8-9 ms per step because hundreds of tiny CUDA kernels are launched from the CPU
each step. Capturing one control step into a CUDA graph and replaying it removes
the launch overhead almost entirely.

This module captures the inference path for a fixed context length (the context
is zero-padded on the left until it is full) and replays it every step. Numerical
results are bit-exact with eager inference because the same kernels run on the
same inputs.

Usage:
    runner = HDMLGraphRunner(model, prop_dim=35, action_dim=12, context_length=20)
    action = runner(states_np, rtgs_np, actions_np)   # (1, T, d) numpy arrays
"""
from __future__ import annotations

import numpy as np
import torch

from hdml.models.hdml_model import HDMLModel


class HDMLGraphRunner:
    """Fixed-shape CUDA-graph wrapper around ``HDMLModel.get_action``."""

    def __init__(
        self,
        model: HDMLModel,
        prop_dim: int,
        action_dim: int,
        context_length: int = 20,
        device: torch.device | str = "cuda",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("HDMLGraphRunner requires a CUDA device")
        self.model = model
        self.device = torch.device(device)
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.context_length = context_length
        model.eval()
        if hasattr(model, "deterministic"):
            model.deterministic = True

        self._states = torch.zeros(1, context_length, prop_dim, device=self.device)
        self._rtgs = torch.zeros(1, context_length, 1, device=self.device)
        self._actions = torch.zeros(1, context_length, action_dim, device=self.device)
        self._timesteps = torch.arange(context_length, device=self.device, dtype=torch.int64).unsqueeze(0)
        self._capture()

    def _capture(self) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                with torch.inference_mode():
                    self.model.get_action(
                        states=self._states, rtgs=self._rtgs, actions=self._actions,
                        timesteps=self._timesteps, hx=None,
                    )
        torch.cuda.current_stream().wait_stream(stream)
        self._graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(self._graph):
                self._action_out, _, self._info = self.model.get_action(
                    states=self._states, rtgs=self._rtgs, actions=self._actions,
                    timesteps=self._timesteps, hx=None,
                )

    @staticmethod
    def _pad_fill(buf: torch.Tensor, arr: np.ndarray) -> None:
        """Left-pad ``arr`` to the buffer length and copy into ``buf`` (in place)."""
        T = buf.shape[1]
        n = arr.shape[0]
        if n >= T:
            buf.copy_(torch.from_numpy(np.ascontiguousarray(arr[-T:])).to(buf.device))
        else:
            buf[:, T - n:].copy_(torch.from_numpy(np.ascontiguousarray(arr)).to(buf.device))
            buf[:, :T - n].zero_()

    def __call__(self, states: np.ndarray, rtgs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Run one control step. Inputs are (T_i, d) arrays; returns (action_dim,)."""
        self._pad_fill(self._states, np.asarray(states, np.float32))
        self._pad_fill(self._rtgs, np.asarray(rtgs, np.float32).reshape(-1, 1))
        self._pad_fill(self._actions, np.asarray(actions, np.float32))
        self._graph.replay()
        return self._action_out[0].detach().cpu().numpy().astype(np.float32)
