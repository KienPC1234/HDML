"""HDML Unified Command-Line Interface (CLI).

Consolidates all repository operations into a single multi-functional tool:
- train-foundation: Pre-train multi-embodiment HDML-Foundation model
- train-baseline: Train baseline models (DT, RNN, Diffusion, IQL, MLP) with equal compute
- benchmark-foundation: Multi-embodiment transfer and hardware latency benchmark
- benchmark-baselines: Comparative benchmark against baseline architectures
- evaluate-transfer: Few-shot adaptation to a target robot morphology
- collect-data: Trajectory collection from diverse physics environments
- export-onnx: Export models to ONNX format for deployment
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def cmd_train_foundation(args: argparse.Namespace) -> None:
    from scripts.train_foundation import train_foundation
    train_foundation(args.config)


def cmd_benchmark_foundation(args: argparse.Namespace) -> None:
    from scripts.benchmark_foundation import run_comprehensive_benchmark
    run_comprehensive_benchmark(checkpoint_path=args.checkpoint, output_json=args.output)


def cmd_evaluate_transfer(args: argparse.Namespace) -> None:
    from scripts.evaluate_foundation_transfer import evaluate_transfer
    evaluate_transfer(
        checkpoint_path=args.checkpoint,
        target_embodiment=args.target_embodiment,
        target_dataset_path=args.target_dataset,
        prop_dim=args.prop_dim,
        action_dim=args.action_dim,
        epochs=args.epochs,
    )


def cmd_collect_data(args: argparse.Namespace) -> None:
    from scripts.collect_foundation_data import main as collect_main
    sys.argv = ["collect_foundation_data.py", "--output-dir", args.output_dir]
    if args.envs:
        sys.argv.extend(["--envs", *args.envs])
    collect_main()


def cmd_train_baseline(args: argparse.Namespace) -> None:
    from scripts.train_baselines import main as train_baselines_main
    sys.argv = [
        "train_baselines.py",
        "--config", args.config,
        "--dataset", args.dataset,
        "--model", args.model,
        "--device", args.device,
    ]
    if args.epochs:
        sys.argv.extend(["--epochs", str(args.epochs)])
    if args.batch_size:
        sys.argv.extend(["--batch-size", str(args.batch_size)])
    train_baselines_main()


def cmd_benchmark_baselines(args: argparse.Namespace) -> None:
    from scripts.benchmark_baselines import main as bench_baselines_main
    sys.argv = [
        "benchmark_baselines.py",
        "--config", args.config,
        "--checkpoint", args.checkpoint,
        "--episodes", str(args.episodes),
        "--device", args.device,
    ]
    bench_baselines_main()


def cmd_export_onnx(args: argparse.Namespace) -> None:
    from scripts.export_onnx import main as export_main
    sys.argv = [
        "export_onnx.py",
        "--checkpoint", args.checkpoint,
        "--output", args.output,
        "--device", args.device,
    ]
    export_main()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hdml",
        description="HDML (Hierarchical Decision Mamba-Liquid) Unified CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="HDML Subcommands")

    # 1. train-foundation
    p_tf = subparsers.add_parser("train-foundation", help="Pre-train HDML-Foundation Model")
    p_tf.add_argument("--config", type=str, default="configs/hdml_foundation_full.yaml", help="Path to config YAML")
    p_tf.set_defaults(func=cmd_train_foundation)

    # 2. benchmark-foundation
    p_bf = subparsers.add_parser("benchmark-foundation", help="Benchmark HDML-Foundation Model")
    p_bf.add_argument("--checkpoint", type=str, default="checkpoints/hdml_foundation/hdml_foundation_best.pt")
    p_bf.add_argument("--output", type=str, default="logs/benchmark_foundation_results.json")
    p_bf.set_defaults(func=cmd_benchmark_foundation)

    # 3. evaluate-transfer
    p_et = subparsers.add_parser("evaluate-transfer", help="Few-shot transfer to target embodiment")
    p_et.add_argument("--checkpoint", type=str, default="checkpoints/hdml_foundation/hdml_foundation_best.pt")
    p_et.add_argument("--target-embodiment", type=str, default="unitree_a1_maze")
    p_et.add_argument("--target-dataset", type=str, default="data/unitree_a1_maze_trajectories.npz")
    p_et.add_argument("--prop-dim", type=int, default=53)
    p_et.add_argument("--action-dim", type=int, default=12)
    p_et.add_argument("--epochs", type=int, default=3)
    p_et.set_defaults(func=cmd_evaluate_transfer)

    # 4. collect-data
    p_cd = subparsers.add_parser("collect-data", help="Collect offline trajectory datasets")
    p_cd.add_argument("--output-dir", type=str, default="data")
    p_cd.add_argument("--envs", type=str, nargs="*", default=None)
    p_cd.set_defaults(func=cmd_collect_data)

    # 5. train-baseline
    p_tb = subparsers.add_parser("train-baseline", help="Train baseline architectures for fair comparison")
    p_tb.add_argument("--config", type=str, default="configs/halfcheetah_v5_default.yaml")
    p_tb.add_argument("--dataset", type=str, default="data/halfcheetah_v5_expert.npz")
    p_tb.add_argument("--model", type=str, default="all", choices=["all", "dt", "rnn", "mlp", "diffusion", "iql", "mamba_mlp", "transformer_liquid"])
    p_tb.add_argument("--device", type=str, default="cuda")
    p_tb.add_argument("--epochs", type=int, default=10)
    p_tb.add_argument("--batch-size", type=int, default=256)
    p_tb.set_defaults(func=cmd_train_baseline)

    # 6. benchmark-baselines
    p_bb = subparsers.add_parser("benchmark-baselines", help="Benchmark HDML against SOTA baselines")
    p_bb.add_argument("--config", type=str, default="configs/halfcheetah_v5_default.yaml")
    p_bb.add_argument("--checkpoint", type=str, default="checkpoints/halfcheetah_v5/best_model.pt")
    p_bb.add_argument("--episodes", type=int, default=5)
    p_bb.add_argument("--device", type=str, default="cuda")
    p_bb.set_defaults(func=cmd_benchmark_baselines)

    # 7. export-onnx
    p_eo = subparsers.add_parser("export-onnx", help="Export HDML model to ONNX format")
    p_eo.add_argument("--checkpoint", type=str, default="checkpoints/halfcheetah_v5/best_model.pt")
    p_eo.add_argument("--output", type=str, default="deployment/model.onnx")
    p_eo.add_argument("--device", type=str, default="cpu")
    p_eo.set_defaults(func=cmd_export_onnx)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
