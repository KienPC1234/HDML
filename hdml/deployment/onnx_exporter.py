from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, cast
import warnings
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HDMLDeploymentWrapper(nn.Module):
    """Clean self-contained inference wrapper for ONNX & TensorRT deployment.

    Optionally bakes dataset input normalization into the graph: when
    ``state_mean``/``state_std`` are given, raw (un-normalized) game
    observations are normalized inside ``forward`` as
    ``(states - mean) / std`` — the same transform
    :class:`FastTensorTrajectoryDataset` applied at training. Deploy callers
    then feed RAW observations (SPEC §5) with no JS-side math.
    """

    def __init__(
        self,
        model: nn.Module,
        action_only: bool = False,
        state_mean: torch.Tensor | np.ndarray | None = None,
        state_std: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.action_only = bool(action_only)
        if (state_mean is None) != (state_std is None):
            raise ValueError("state_mean and state_std must be given together or both be None.")
        if state_mean is not None:
            mean = torch.as_tensor(np.asarray(state_mean, dtype=np.float32)).reshape(1, 1, -1)
            std = torch.as_tensor(np.asarray(state_std, dtype=np.float32)).reshape(1, 1, -1)
            if bool((std <= 0).any()):
                raise ValueError("state_std must be strictly positive.")
            self.register_buffer("state_mean", mean)
            self.register_buffer("state_std", std)
            self.normalizes_inputs = True
        else:
            self.normalizes_inputs = False
        # Clone model to CPU
        self.model = copy.deepcopy(model).to("cpu").eval()
        model_any = cast(Any, self.model)
        # Deterministic flow solve for tracing/parity.
        if hasattr(model_any, "deterministic"):
            model_any.deterministic = True

        # Replace the SSM kernels with pure-PyTorch portable versions so the graph
        # is traceable by torch.onnx and runnable on CPU.
        if hasattr(model_any, "mamba_backbone"):
            from hdml.models.mamba3_native import Mamba3NativeBlock
            from hdml.models.mamba3_portable import PortableMamba3
            for block in model_any.mamba_backbone.layers:
                if isinstance(block, Mamba3NativeBlock) and hasattr(block, "mamba"):
                    if isinstance(block.mamba, PortableMamba3):
                        continue
                    nat = block.mamba
                    rope_fraction = getattr(block, "rope_fraction", 0.5)
                    dev = next(nat.parameters()).device
                    dt = next(nat.parameters()).dtype
                    port = PortableMamba3(
                        d_model=nat.d_model,
                        d_state=nat.d_state,
                        expand=nat.expand,
                        headdim=nat.headdim,
                        rope_fraction=rope_fraction,
                    ).to(device=dev, dtype=dt)
                    port.load_state_dict(nat.state_dict())
                    block.mamba = port

    def forward(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference forward pass for deployment.

        Args:
            states: (Batch, Seq_Len, Prop_Dim)
            rtgs: (Batch, Seq_Len, 1)
            actions: (Batch, Seq_Len, Action_Dim)
            timesteps: (Batch, Seq_Len)

        Returns:
            If action_only is True:
                action: (Batch, Action_Dim)
            Else:
                action: (Batch, Action_Dim)
                subgoal: (Batch, Subgoal_Dim)
                next_state: (Batch, Prop_Dim)
        """
        if self.normalizes_inputs:
            state_mean = cast(torch.Tensor, self.state_mean)
            state_std = cast(torch.Tensor, self.state_std)
            states = (states - state_mean) / state_std
        model_any = cast(Any, self.model)
        subgoals, latent, values, next_states, _, executed, _ = model_any.encode(
            states=states, rtgs=rtgs, actions=actions, timesteps=timesteps,
            action_last_only=True,
        )
        assert executed is not None, "executed action is None"
        curr_action = executed[:, -1, :]
        if self.action_only:
            return curr_action

        curr_subgoal = subgoals[:, -1, :]
        curr_next_state = next_states[:, -1, :]
        return curr_action, curr_subgoal, curr_next_state


def infer_native_mamba3(state_dict: dict[str, torch.Tensor]) -> bool | None:
    """Infer the backbone variant from checkpoint parameter names.

    Native ``mamba_ssm.Mamba3`` blocks expose ``B_norm``/``C_norm``/``dt_bias``
    and ``B_bias``/``C_bias``; the legacy Mamba-1 + RoPE wrapper exposes
    ``x_proj``/``conv1d``/``dt_proj`` instead. Returns ``None`` when the
    checkpoint has neither (e.g. a non-HDML state dict).
    """
    keys = [k for k in state_dict if "mamba_backbone.layers." in k]
    if not keys:
        return None
    is_native = any(k.endswith(".mamba.B_norm.weight") or k.endswith(".mamba.dt_bias") for k in keys)
    is_legacy = any(k.endswith(".mamba.x_proj.weight") for k in keys)
    if is_native == is_legacy:
        return None
    return is_native


def export_hdml_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    prop_dim: int,
    action_dim: int,
    context_length: int = 20,
    action_only: bool = False,
    opset_version: int = 17,
    verify: bool = True,
    state_mean: torch.Tensor | np.ndarray | None = None,
    state_std: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """Export an HDML model to ONNX format with dynamic batch size and a fixed context length.

    When ``state_mean``/``state_std`` are provided they are baked into the
    graph (raw-obs in, normalized inside); ``verify`` then checks RAW-input
    parity end-to-end. The flags are echoed in the result dict for provenance.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "prop_dim") and model.prop_dim != prop_dim:
        logger.warning(
            "export_hdml_to_onnx: argument prop_dim=%d != model.prop_dim=%d; using model.prop_dim",
            prop_dim,
            model.prop_dim,
        )
        prop_dim = model.prop_dim
    if hasattr(model, "action_dim") and model.action_dim != action_dim:
        logger.warning(
            "export_hdml_to_onnx: argument action_dim=%d != model.action_dim=%d; using model.action_dim",
            action_dim,
            model.action_dim,
        )
        action_dim = model.action_dim

    wrapper = HDMLDeploymentWrapper(
        model, action_only=action_only, state_mean=state_mean, state_std=state_std
    ).eval()

    batch_size = 1
    dummy_states = torch.randn(batch_size, context_length, prop_dim, dtype=torch.float32)
    dummy_rtgs = torch.zeros(batch_size, context_length, 1, dtype=torch.float32)
    dummy_actions = torch.randn(batch_size, context_length, action_dim, dtype=torch.float32)
    dummy_timesteps = torch.arange(context_length, dtype=torch.int64).unsqueeze(0)

    input_names = ["states", "rtgs", "actions", "timesteps"]
    if action_only:
        output_names = ["action"]
        dynamic_axes = {
            "states": {0: "batch_size"},
            "rtgs": {0: "batch_size"},
            "actions": {0: "batch_size"},
            "timesteps": {0: "batch_size"},
            "action": {0: "batch_size"},
        }
    else:
        output_names = ["action", "subgoal", "next_state"]
        dynamic_axes = {
            "states": {0: "batch_size"},
            "rtgs": {0: "batch_size"},
            "actions": {0: "batch_size"},
            "timesteps": {0: "batch_size"},
            "action": {0: "batch_size"},
            "subgoal": {0: "batch_size"},
            "next_state": {0: "batch_size"},
        }

    logger.info(f"Exporting HDML model to ONNX: {out_file} (opset={opset_version}, action_only={action_only})...")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            wrapper,
            (dummy_states, dummy_rtgs, dummy_actions, dummy_timesteps),
            str(out_file),
            dynamo=False,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX model saved successfully: {out_file} ({file_size_mb:.2f} MB)")

    max_diff = 0.0
    if verify:
        try:
            import onnx
            import onnxruntime as ort

            onnx_model = onnx.load(str(out_file))
            onnx.checker.check_model(onnx_model)

            ort_session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])

            with torch.inference_mode():
                torch_out = wrapper(dummy_states, dummy_rtgs, dummy_actions, dummy_timesteps)

            ort_inputs = {
                "states": dummy_states.numpy(),
                "rtgs": dummy_rtgs.numpy(),
                "actions": dummy_actions.numpy(),
                "timesteps": dummy_timesteps.numpy(),
            }
            ort_out = ort_session.run(None, ort_inputs)

            if action_only:
                torch_act = torch_out.numpy()
                ort_act = ort_out[0]
            else:
                torch_act = torch_out[0].numpy()
                ort_act = ort_out[0]
            max_diff = float(np.max(np.abs(torch_act - ort_act)))

            logger.info(f"ONNX numerical parity verified! Max absolute difference: {max_diff:.6e}")
            assert max_diff < 1e-3, f"ONNX parity failed: max diff {max_diff:.6e} exceeds tolerance 1e-3"
        except Exception as e:
            logger.error(f"ONNX validation error: {e}")
            raise

    return {
        "onnx_path": str(out_file),
        "file_size_mb": file_size_mb,
        "max_numerical_diff": max_diff,
        "verified": bool(verify),
        "action_only": action_only,
        "context_length": context_length,
        "normalizes_inputs": wrapper.normalizes_inputs,
        "parity_scope": "portable PyTorch vs ONNX; CUDA vs portable parity requires a separate test",
    }

