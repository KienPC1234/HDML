"""CUDA-graph accelerated closed-loop inference for HDML.

The HDML policy is a small network (about 1.4M parameters) run one step at a
time. Its GPU kernels execute in roughly 0.3 ms, but eager inference costs about
8-9 ms per step because hundreds of tiny CUDA kernels are launched from the CPU
each step. Capturing one control step into a CUDA graph and replaying it removes
the launch overhead almost entirely.

Window policy
-------------
CUDA graphs require static tensor shapes, but the evaluator deliberately builds
short causal windows at the start of an episode (no zero padding). Zero-padding
a short window is not equivalent to running the short window: the causal SSM
still consumes the padded tokens. This runner therefore captures one graph per
sequence length actually used (``1 .. context_length``) and copies the real
context into a same-shape buffer. Eager and graph inference receive the same
observations, actions, RTGs, and **real timesteps**, with no padding. For inputs
longer than ``context_length`` the last ``context_length`` steps are used, which
matches the evaluator's rolling-window policy.

CfC memory
----------
The evaluator carries the CfC hidden state ``next_hx`` across control steps and
resets it at episode start. The runner captures that same policy: a shared static
hidden-state buffer is fed to every graph, and ``next_hx`` is copied back into it
after each replay. Call :meth:`HDMLGraphRunner.reset` at the start of an episode.
Unlike the old stateless path, graph and evaluator now use identical CfC memory.

Usage:
    runner = HDMLGraphRunner(model, prop_dim=35, action_dim=12, context_length=20)
    runner.reset()
    action = runner(states_np, rtgs_np, actions_np, timesteps_np)  # (T, d) arrays
"""
from __future__ import annotations

import numpy as np
import torch

from hdml.models.hdml_model import HDMLModel


class HDMLGraphRunner:
    """Fixed-shape CUDA-graph wrapper around ``HDMLModel.get_action``.

    Each sequence length gets its own static input buffers and captured graph.
    Timesteps are copied from the caller before every replay, so the fusion
    timestep embedding sees the true control step rather than a fixed
    ``arange`` buffer.
    """

    def __init__(
        self,
        model: HDMLModel,
        prop_dim: int,
        action_dim: int,
        context_length: int = 20,
        device: torch.device | str = "cuda",
        precapture: bool = True,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("HDMLGraphRunner requires a CUDA device")
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        self.model = model
        self.device = torch.device(device)
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.context_length = context_length
        model.eval()
        if hasattr(model, "deterministic"):
            model.deterministic = True

        # The evaluator carries CfC state across control steps. Capture that
        # same policy: a shared static hidden-state buffer is an input to every
        # graph, and each graph exposes its next_hx output.
        cfc_filter = getattr(model, "cfc_filter", None)
        self.cfc_state_size: int | None = None
        if cfc_filter is not None:
            cfc_core = getattr(cfc_filter, "cfc", None)
            state_size = getattr(cfc_core, "state_size", None)
            if state_size is None:
                raise ValueError("CfC filter does not expose its hidden-state size")
            self.cfc_state_size = int(state_size)
        self._hx: torch.Tensor | None = (
            torch.zeros(1, self.cfc_state_size, device=self.device)
            if self.cfc_state_size is not None
            else None
        )

        self._states: dict[int, torch.Tensor] = {}
        self._rtgs: dict[int, torch.Tensor] = {}
        self._actions: dict[int, torch.Tensor] = {}
        self._timesteps: dict[int, torch.Tensor] = {}
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._outputs: dict[int, torch.Tensor] = {}
        self._next_hx: dict[int, torch.Tensor] = {}
        if precapture:
            for seq_len in range(1, context_length + 1):
                self._capture(seq_len)

    def reset(self) -> None:
        """Reset the carried CfC state at the start of an episode."""
        if self._hx is not None:
            self._hx.zero_()

    def set_hx(self, hx: torch.Tensor | np.ndarray | None) -> None:
        """Set the carried CfC state explicitly (``None`` resets to zeros)."""
        if self._hx is None:
            if hx is not None:
                raise ValueError("This model has no CfC filter; hx must be None")
            return
        if hx is None:
            self._hx.zero_()
            return
        hx_t = torch.as_tensor(hx, dtype=self._hx.dtype, device=self.device)
        if hx_t.ndim == 1:
            hx_t = hx_t.unsqueeze(0)
        if hx_t.shape != self._hx.shape:
            raise ValueError(
                f"hx must have shape {tuple(self._hx.shape)}, got {tuple(hx_t.shape)}"
            )
        self._hx.copy_(hx_t)

    def get_hx(self) -> np.ndarray | None:
        """Return a CPU copy of the currently carried CfC state."""
        if self._hx is None:
            return None
        return self._hx.detach().cpu().numpy().astype(np.float32)

    def _capture(self, seq_len: int) -> None:
        """Capture one graph for a static window of ``seq_len`` control steps."""
        if seq_len in self._graphs:
            return
        if not 1 <= seq_len <= self.context_length:
            raise ValueError(
                f"seq_len must be in [1, {self.context_length}], got {seq_len}"
            )
        states = torch.zeros(1, seq_len, self.prop_dim, device=self.device)
        rtgs = torch.zeros(1, seq_len, 1, device=self.device)
        actions = torch.zeros(1, seq_len, self.action_dim, device=self.device)
        timesteps = torch.zeros(1, seq_len, device=self.device, dtype=torch.int64)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                with torch.inference_mode():
                    self.model.get_action(
                        states=states,
                        rtgs=rtgs,
                        actions=actions,
                        timesteps=timesteps,
                        hx=self._hx,
                    )
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph):
                action_out, next_hx, _ = self.model.get_action(
                    states=states,
                    rtgs=rtgs,
                    actions=actions,
                    timesteps=timesteps,
                    hx=self._hx,
                )
        if self.cfc_state_size is not None:
            if next_hx is None:
                raise RuntimeError("CfC is enabled but get_action returned no next_hx")
            self._next_hx[seq_len] = next_hx
        self._states[seq_len] = states
        self._rtgs[seq_len] = rtgs
        self._actions[seq_len] = actions
        self._timesteps[seq_len] = timesteps
        self._graphs[seq_len] = graph
        self._outputs[seq_len] = action_out

    def _select_inputs(
        self,
        states: np.ndarray,
        rtgs: np.ndarray,
        actions: np.ndarray,
        timesteps: np.ndarray,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float32)
        rtgs = np.asarray(rtgs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        timesteps = np.asarray(timesteps, dtype=np.int64)

        if states.ndim != 2 or states.shape[-1] != self.prop_dim:
            raise ValueError(
                f"states must be (T, {self.prop_dim}), got {states.shape}"
            )
        if rtgs.ndim == 1:
            rtgs = rtgs.reshape(-1, 1)
        if rtgs.ndim != 2 or rtgs.shape[-1] != 1:
            raise ValueError(f"rtgs must be (T,) or (T, 1), got {rtgs.shape}")
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must be (T, {self.action_dim}), got {actions.shape}"
            )
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must be (T,), got {timesteps.shape}")

        seq_len = states.shape[0]
        if not (states.shape[0] == rtgs.shape[0] == actions.shape[0] == timesteps.shape[0]):
            raise ValueError(
                "states, rtgs, actions and timesteps must share the same length: "
                f"{states.shape[0]}, {rtgs.shape[0]}, {actions.shape[0]}, {timesteps.shape[0]}"
            )
        if seq_len < 1:
            raise ValueError("context window must contain at least one step")

        keep = min(seq_len, self.context_length)
        return (
            keep,
            np.ascontiguousarray(states[-keep:]),
            np.ascontiguousarray(rtgs[-keep:]),
            np.ascontiguousarray(actions[-keep:]),
            np.ascontiguousarray(timesteps[-keep:]),
        )

    def __call__(
        self,
        states: np.ndarray,
        rtgs: np.ndarray,
        actions: np.ndarray,
        timesteps: np.ndarray,
        *,
        hx: torch.Tensor | np.ndarray | None = None,
        carry_state: bool = True,
    ) -> np.ndarray:
        """Run one control step and return ``(action_dim,)``.

        By default the runner advances the CfC state exactly like the evaluator:
        the hidden state starts at zero after :meth:`reset`, is fed into every
        replay, and is replaced by ``next_hx`` afterwards.

        Args:
            states: ``(T, prop_dim)`` normalized observations.
            rtgs: ``(T,)`` or ``(T, 1)`` normalized return-to-go values.
            actions: ``(T, action_dim)`` causal action history.
            timesteps: ``(T,)`` integer control steps, copied before replay.
            hx: optional explicit CfC state to use for this call.
            carry_state: when True, update the internal state to ``next_hx``.
        """
        if hx is not None:
            self.set_hx(hx)
        seq_len, states_np, rtgs_np, actions_np, timesteps_np = self._select_inputs(
            states, rtgs, actions, timesteps
        )
        if seq_len not in self._graphs:
            self._capture(seq_len)

        self._states[seq_len].copy_(torch.from_numpy(states_np).to(self.device))
        self._rtgs[seq_len].copy_(torch.from_numpy(rtgs_np.reshape(-1, 1)).to(self.device))
        self._actions[seq_len].copy_(torch.from_numpy(actions_np).to(self.device))
        self._timesteps[seq_len].copy_(torch.from_numpy(timesteps_np).to(self.device))
        self._graphs[seq_len].replay()
        action = self._outputs[seq_len][0].detach().cpu().numpy().astype(np.float32)
        if carry_state and self._hx is not None:
            self._hx.copy_(self._next_hx[seq_len])
        return action
