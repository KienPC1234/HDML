#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import torch

from hdml.utils.config import HDMLConfig
from hdml.models.hdml_model import HDMLModel
from hdml.utils.export import export_liquid_head_to_onnx, verify_onnx_equivalence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HDML Liquid Head to ONNX for edge deployment.")
    parser.add_argument("--config", type=str, default="configs/ant_v4_default.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default="checkpoints/hdml_liquid_head.onnx", help="Output .onnx file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pt (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = HDMLConfig.from_yaml(args.config)

    model = HDMLModel.from_config(cfg.model)
    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Loading weights from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    liquid_head = model.liquid_head
    out_path = export_liquid_head_to_onnx(
        liquid_head=liquid_head,
        subgoal_dim=cfg.model.d_subgoal,
        prop_dim=cfg.model.prop_dim,
        output_path=args.output,
        device="cpu",
    )

    # Verification
    dummy_subgoal = torch.randn(2, cfg.model.d_subgoal)
    dummy_prop = torch.randn(2, cfg.model.prop_dim)
    is_valid = verify_onnx_equivalence(
        torch_model=liquid_head,
        onnx_path=out_path,
        sample_subgoal=dummy_subgoal,
        sample_prop=dummy_prop,
    )

    if is_valid:
        logger.info(f"SUCCESS: ONNX model verified at {out_path}")
    else:
        logger.error(f"FAILURE: ONNX output mismatch above tolerance.")


if __name__ == "__main__":
    main()
