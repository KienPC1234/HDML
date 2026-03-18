from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_liquid_head_to_onnx(
    liquid_head: nn.Module,
    subgoal_dim: int,
    prop_dim: int,
    output_path: str | Path,
    device: str = "cpu",
    opset_version: int = 17,
) -> Path:
    """Export the Liquid Reactive Control Head to ONNX format.

    Args:
        liquid_head: PyTorch Liquid Reactive Control Head module.
        subgoal_dim: Dimension of latent subgoal vector c_t.
        prop_dim: Dimension of current proprioceptive state vector.
        output_path: Target path for the exported .onnx file.
        device: Device to run the export on ('cpu' or 'cuda').
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX model.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    liquid_head.eval()
    liquid_head.to(device)

    dummy_subgoal = torch.randn(1, subgoal_dim, device=device)
    dummy_prop = torch.randn(1, prop_dim, device=device)

    logger.info(f"Exporting Liquid Head to ONNX at: {out_file}")

    torch.onnx.export(
        liquid_head,
        (dummy_subgoal, dummy_prop),
        str(out_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["subgoal", "current_prop"],
        output_names=["action", "next_hx"],
        dynamic_axes={
            "subgoal": {0: "batch_size"},
            "current_prop": {0: "batch_size"},
            "action": {0: "batch_size"},
            "next_hx": {0: "batch_size"},
        },
    )

    logger.info("ONNX export complete.")
    return out_file


def verify_onnx_equivalence(
    torch_model: nn.Module,
    onnx_path: str | Path,
    sample_subgoal: torch.Tensor,
    sample_prop: torch.Tensor,
    atol: float = 1e-4,
) -> bool:
    """Verify numeric equivalence between PyTorch model and ONNX Runtime execution.

    Args:
        torch_model: PyTorch Liquid Head module.
        onnx_path: Path to the exported .onnx file.
        sample_subgoal: Sample subgoal tensor.
        sample_prop: Sample proprioception tensor.
        atol: Absolute tolerance for difference checking.

    Returns:
        True if outputs are within tolerance, False otherwise.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        logger.error(f"onnxruntime is required for verification: {e}")
        raise

    torch_model.eval()
    with torch.inference_mode():
        torch_act, _ = torch_model(sample_subgoal, sample_prop)
        torch_act_np = torch_act.detach().cpu().numpy()

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3  # Error only
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_options, providers=["CPUExecutionProvider"])

    ort_inputs = {
        "subgoal": sample_subgoal.detach().cpu().numpy().astype("float32"),
        "current_prop": sample_prop.detach().cpu().numpy().astype("float32"),
    }
    ort_outs = session.run(None, ort_inputs)
    onnx_act_np = ort_outs[0]

    max_diff = float((torch_act_np - onnx_act_np).__abs__().max())
    is_close = bool(max_diff <= atol)

    logger.info(f"ONNX Equivalence verification: Max Diff={max_diff:.6f}, Passed={is_close}")
    return is_close
