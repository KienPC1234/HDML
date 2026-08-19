from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hdml.models.mamba3_backbone import Mamba3Block

logger = logging.getLogger(__name__)


class PurePyTorchMamba(nn.Module):
    """Pure PyTorch portable implementation of Mamba SSM block for ONNX & CPU execution."""

    def __init__(self, mamba_module: nn.Module) -> None:
        super().__init__()
        self.d_model = mamba_module.d_model
        self.d_state = mamba_module.d_state
        self.d_conv = mamba_module.d_conv
        self.expand = mamba_module.expand
        self.d_inner = mamba_module.d_inner
        self.dt_rank = mamba_module.dt_rank

        self.in_proj = copy.deepcopy(mamba_module.in_proj).to("cpu")
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=mamba_module.conv1d.bias is not None,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        ).to("cpu")
        self.conv1d.weight.data.copy_(mamba_module.conv1d.weight.data.to("cpu"))
        if mamba_module.conv1d.bias is not None:
            self.conv1d.bias.data.copy_(mamba_module.conv1d.bias.data.to("cpu"))

        self.x_proj = copy.deepcopy(mamba_module.x_proj).to("cpu")
        self.dt_proj = copy.deepcopy(mamba_module.dt_proj).to("cpu")
        self.out_proj = copy.deepcopy(mamba_module.out_proj).to("cpu")
        self.A_log = nn.Parameter(mamba_module.A_log.data.clone().to("cpu"))
        self.D = nn.Parameter(mamba_module.D.data.clone().to("cpu"))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = u.shape
        xz = self.in_proj(u)
        x, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :seqlen].transpose(1, 2)
        x_act = F.silu(x_conv)

        x_dbl = self.x_proj(x_act)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        y_list = []
        h = torch.zeros(batch, self.d_inner, self.d_state, device=u.device, dtype=u.dtype)
        for t in range(seqlen):
            dt_t = dt[:, t, :, None]
            dA_t = torch.exp(dt_t * A[None, :, :])
            dB_t = dt_t * B_ssm[:, t, None, :]
            x_t = x_act[:, t, :, None]
            h = h * dA_t + dB_t * x_t
            y_t = (h * C_ssm[:, t, None, :]).sum(dim=-1) + x_act[:, t, :] * D[None, :]
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out


class HDMLDeploymentWrapper(nn.Module):
    """Clean self-contained inference wrapper for ONNX & TensorRT deployment."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        # Clone model to CPU
        self.model = copy.deepcopy(model).to("cpu").eval()

        # Replace CUDA Mamba with portable PurePyTorchMamba
        if hasattr(self.model, "mamba_backbone"):
            for block in self.model.mamba_backbone.layers:
                if isinstance(block, Mamba3Block) and hasattr(block, "mamba"):
                    block.mamba = PurePyTorchMamba(block.mamba)

    def forward(
        self,
        states: torch.Tensor,
        rtgs: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference forward pass for deployment.

        Args:
            states: (Batch, Seq_Len, Prop_Dim)
            rtgs: (Batch, Seq_Len, 1)
            actions: (Batch, Seq_Len, Action_Dim)
            timesteps: (Batch, Seq_Len)

        Returns:
            action: (Batch, Action_Dim)
            subgoal: (Batch, Subgoal_Dim)
            next_state: (Batch, Prop_Dim)
        """
        actions_pred, subgoals_pred, _, next_states_pred, _ = self.model(
            states=states,
            rtgs=rtgs,
            actions=actions,
            timesteps=timesteps,
        )

        if actions_pred.ndim == 3:
            curr_action = actions_pred[:, -1, :]
        else:
            curr_action = actions_pred

        if subgoals_pred.ndim == 3:
            curr_subgoal = subgoals_pred[:, -1, :]
        else:
            curr_subgoal = subgoals_pred

        if next_states_pred.ndim == 3:
            curr_next_state = next_states_pred[:, -1, :]
        else:
            curr_next_state = next_states_pred

        return curr_action, curr_subgoal, curr_next_state


def export_hdml_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    prop_dim: int,
    action_dim: int,
    context_length: int = 20,
    opset_version: int = 17,
    verify: bool = True,
) -> dict[str, Any]:
    """Export an HDML model to ONNX format with dynamic batch and sequence dimensions."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    wrapper = HDMLDeploymentWrapper(model).eval()

    batch_size = 1
    dummy_states = torch.randn(batch_size, context_length, prop_dim, dtype=torch.float32)
    dummy_rtgs = torch.zeros(batch_size, context_length, 1, dtype=torch.float32)
    dummy_actions = torch.randn(batch_size, context_length, action_dim, dtype=torch.float32)
    dummy_timesteps = torch.arange(context_length, dtype=torch.int64).unsqueeze(0)

    input_names = ["states", "rtgs", "actions", "timesteps"]
    output_names = ["action", "subgoal", "next_state"]

    dynamic_axes = {
        "states": {0: "batch_size", 1: "sequence_length"},
        "rtgs": {0: "batch_size", 1: "sequence_length"},
        "actions": {0: "batch_size", 1: "sequence_length"},
        "timesteps": {0: "batch_size", 1: "sequence_length"},
        "action": {0: "batch_size"},
        "subgoal": {0: "batch_size"},
        "next_state": {0: "batch_size"},
    }

    logger.info(f"Exporting HDML model to ONNX: {out_file} (opset={opset_version})...")

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
        "verified": True,
    }
