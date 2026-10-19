#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import time
import numpy as np
import torch

from hdml.utils.config import HDMLConfig
from hdml.models.hdml_model import HDMLModel
from hdml.deployment.onnx_exporter import export_hdml_to_onnx, infer_native_mamba3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained HDML model to optimized ONNX.")
    parser.add_argument("--config", type=str, default="configs/halfcheetah_v5_default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/rebuild_long/halfcheetah_v5/best_model.pt", help="Path to checkpoint .pt")
    parser.add_argument("--output", type=str, default="deployment/hdml_halfcheetah.onnx", help="Output .onnx file path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--action-only", action="store_true", help="Export lean ONNX graph with single 'action' output")
    parser.add_argument("--context-length", type=int, default=0, help="Override sequence context length")
    parser.add_argument(
        "--quant",
        type=str,
        default="none",
        choices=["none", "static-int8", "int4"],
        help="Post-export quantization: 'static-int8' (QDQ INT8, needs calibration) or 'int4' (weight-only 4-bit). "
        "Quantized output is approximate: parity vs FP32 is measured and recorded, not asserted. "
        "Dynamic INT8 is intentionally unsupported: ORT's dynamic quantizer mis-handles transB Gemm "
        "weights shared across unrolled steps (missing transpose on weight_quantized) and emits graphs "
        "that fail ORT shape inference.",
    )
    parser.add_argument("--calib-iters", type=int, default=50, help="Calibration batches for static-int8 (synthetic inputs)")
    return parser.parse_args()


def quantize_onnx_model(
    fp32_path: str | Path,
    quant: str,
    prop_dim: int,
    action_dim: int,
    context_length: int = 20,
    calib_iters: int = 50,
) -> dict[str, float | str]:
    """Quantize an exported FP32 ONNX model.

    Returns a report dict with quantized path, size, parity vs FP32 and latency.
    The quantized model is approximate by construction: this function verifies
    it loads, runs finite, and REPORTS parity (no strict assert).
    """
    import onnx
    import onnxruntime as ort

    fp32_path = Path(fp32_path)
    quant_path = fp32_path.with_name(f"{fp32_path.stem}.{quant}.onnx")

    if quant == "static-int8":
        from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static

        class _SyntheticReader(CalibrationDataReader):
            """Synthetic calibration inputs (same RNG scheme as deployment tests).

            Limitation: random inputs, not real gameplay observations. Ranges
            match the normalized training distribution approximately; re-calibrate
            with real rollouts for production use.
            """

            def __init__(self, n: int) -> None:
                self.n = n
                self.i = 0
                self.rng = np.random.RandomState(0)

            def get_next(self) -> dict[str, np.ndarray] | None:
                if self.i >= self.n:
                    return None
                self.i += 1
                return {
                    "states": self.rng.randn(1, context_length, prop_dim).astype(np.float32),
                    "rtgs": np.zeros((1, context_length, 1), dtype=np.float32),
                    "actions": self.rng.randn(1, context_length, action_dim).astype(np.float32),
                    "timesteps": np.zeros((1, context_length), dtype=np.int64),
                }

            def rewind(self) -> None:
                self.i = 0

        quantize_static(
            str(fp32_path),
            str(quant_path),
            _SyntheticReader(calib_iters),
            quant_format=QuantFormat.QDQ,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
    elif quant == "int4":
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

        model = onnx.load(str(fp32_path))
        quantizer = MatMulNBitsQuantizer(model, bits=4, block_size=128)
        quantizer.process()
        quantizer.model.save_model_to_file(str(quant_path), use_external_data_format=False)
    else:
        raise ValueError(f"Unknown quant mode: {quant}")

    onnx.checker.check_model(onnx.load(str(quant_path)))

    rng = np.random.RandomState(0)
    feed = {
        "states": rng.randn(1, context_length, prop_dim).astype(np.float32),
        "rtgs": np.zeros((1, context_length, 1), dtype=np.float32),
        "actions": rng.randn(1, context_length, action_dim).astype(np.float32),
        "timesteps": np.zeros((1, context_length), dtype=np.int64),
    }
    fp32_out = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    quant_out = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    assert np.isfinite(quant_out).all(), "Quantized output contains NaN or Inf"
    parity = float(np.max(np.abs(fp32_out - quant_out)))
    latency_ms = benchmark_onnx_latency(quant_path, prop_dim, action_dim, context_length)
    size_mb = quant_path.stat().st_size / (1024 * 1024)
    if parity > 0.1:
        logger.warning(f"Quantized parity {parity:.3e} exceeds 0.1 — review before deploying {quant_path}")
    return {"quant_path": str(quant_path), "file_size_mb": size_mb, "parity_vs_fp32": parity, "latency_ms_cpu": latency_ms}


def benchmark_onnx_latency(onnx_path: str | Path, prop_dim: int, action_dim: int, context_length: int = 20, num_iters: int = 200) -> float:
    """Benchmark raw ONNX Runtime inference latency."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy_states = np.random.randn(1, context_length, prop_dim).astype(np.float32)
    dummy_rtgs = np.zeros((1, context_length, 1), dtype=np.float32)
    dummy_actions = np.random.randn(1, context_length, action_dim).astype(np.float32)
    dummy_timesteps = np.arange(context_length, dtype=np.int64).reshape(1, -1)

    inputs = {
        "states": dummy_states,
        "rtgs": dummy_rtgs,
        "actions": dummy_actions,
        "timesteps": dummy_timesteps,
    }

    # Warmup
    for _ in range(20):
        _ = session.run(None, inputs)

    t0 = time.perf_counter()
    for _ in range(num_iters):
        _ = session.run(None, inputs)
    t1 = time.perf_counter()

    avg_latency_ms = ((t1 - t0) / num_iters) * 1000.0
    return avg_latency_ms


def main() -> None:
    args = parse_args()
    cfg = HDMLConfig.from_yaml(args.config)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # The YAML may not describe the architecture the checkpoint was trained
    # with (e.g. configs/unitree_a1_default.yaml omits use_native_mamba3). The
    # state_dict is authoritative; override instead of failing load_state_dict.
    inferred_native = infer_native_mamba3(ckpt["model_state_dict"])
    if inferred_native is not None and inferred_native != cfg.model.use_native_mamba3:
        logger.warning(
            "Config says use_native_mamba3=%s but checkpoint keys indicate %s; "
            "overriding for export.",
            cfg.model.use_native_mamba3,
            inferred_native,
        )
        cfg.model.use_native_mamba3 = inferred_native

    model = HDMLModel.from_config(cfg.model).cpu()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # SPEC §5: dataset normalization lives in the checkpoint (saved by
    # HDMLTrainer). Bake it into the ONNX graph so deploy feeds RAW obs.
    # Older checkpoints without norm stats export un-normalized (warn loudly).
    norm_mean = ckpt.get("state_mean", None)
    norm_std = ckpt.get("state_std", None)
    if norm_mean is None or norm_std is None:
        logger.warning(
            "Checkpoint has no state_mean/state_std — exporting WITHOUT input "
            "normalization. Deploy input distribution will NOT match training."
        )

    ctx_len = args.context_length if args.context_length > 0 else cfg.training.context_length
    res = export_hdml_to_onnx(
        model=model,
        output_path=args.output,
        prop_dim=cfg.model.prop_dim,
        action_dim=cfg.model.action_dim,
        context_length=ctx_len,
        action_only=args.action_only,
        opset_version=args.opset,
        verify=True,
        state_mean=norm_mean,
        state_std=norm_std,
    )

    latency_ms = benchmark_onnx_latency(
        onnx_path=args.output,
        prop_dim=cfg.model.prop_dim,
        action_dim=cfg.model.action_dim,
        context_length=ctx_len,
    )

    meta = {
        "onnx_path": res["onnx_path"],
        "config": args.config,
        "checkpoint": args.checkpoint,
        "prop_dim": cfg.model.prop_dim,
        "action_dim": cfg.model.action_dim,
        "context_length": ctx_len,
        "action_only": args.action_only,
        "opset": args.opset,
        "file_size_mb": res["file_size_mb"],
        "max_numerical_diff_torch_vs_ort": res["max_numerical_diff"],
        "latency_ms_cpu": latency_ms,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = Path(str(args.output) + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Wrote export provenance sidecar: {meta_path}")

    if args.quant != "none":
        qrep = quantize_onnx_model(
            args.output,
            args.quant,
            prop_dim=cfg.model.prop_dim,
            action_dim=cfg.model.action_dim,
            context_length=ctx_len,
            calib_iters=args.calib_iters,
        )
        meta.update(
            {
                "quant": args.quant,
                "quant_path": qrep["quant_path"],
                "quant_file_size_mb": qrep["file_size_mb"],
                "quant_parity_vs_fp32": qrep["parity_vs_fp32"],
                "quant_latency_ms_cpu": qrep["latency_ms_cpu"],
            }
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Quantized Model   : {qrep['quant_path']} ({qrep['file_size_mb']:.2f} MB)")
        print(f"Quant Parity      : {qrep['parity_vs_fp32']:.6e} (vs FP32 ORT, approximate)")
        print(f"Quant Latency     : {qrep['latency_ms_cpu']:.2f} ms / step on CPU")

    print("\n" + "=" * 80)
    print("HDML ONNX EXPORT & BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"ONNX Model File   : {res['onnx_path']}")
    print(f"Model File Size   : {res['file_size_mb']:.2f} MB")
    print(f"Max Absolute Diff : {res['max_numerical_diff']:.6e} (PyTorch vs ONNX Runtime)")
    print(f"Inference Latency : {latency_ms:.2f} ms / step on CPU ({1000.0 / latency_ms:.0f} Hz)")
    print("Verification      : PASSED (portable PyTorch vs ONNX within configured tolerance)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

