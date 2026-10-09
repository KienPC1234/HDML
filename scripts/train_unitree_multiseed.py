#!/usr/bin/env python3
"""Train HDML + all fair baselines for several extra seeds on Unitree A1.

Each seed gets its own checkpoint tree so that the spread across seeds is a
*training* spread (the environment is deterministic, so evaluation seeds would
not add variance). After training, ``scripts/benchmark_unitree_multiseed.py``
pools all seeds for a mean/std/IQM table.

Usage:
    .venv/bin/python scripts/train_unitree_multiseed.py --seeds 100 2024 \
        --base-dir checkpoints/rebuild_unitree_multiseed \
        --epochs 40 --dataset data/unitree_a1_trajectories.npz
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\n\n$ " + " ".join(cmd) + "\n")
        f.flush()
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\n[exit={proc.returncode} elapsed={time.time() - t0:.1f}s]\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 2024])
    ap.add_argument("--config", default="configs/unitree_a1_aug.yaml")
    ap.add_argument("--dataset", default="data/unitree_a1_trajectories.npz")
    ap.add_argument("--base-dir", default="checkpoints/rebuild_unitree_multiseed")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    for seed in args.seeds:
        out = Path(args.base_dir) / f"seed{seed}" / "unitree_a1"
        log = Path("results") / "multiseed_train" / f"seed{seed}.log"
        print(f"=== seed {seed} -> {out} ===", flush=True)
        run([
            PY, "scripts/train_offline.py",
            "--config", args.config, "--dataset", args.dataset,
            "--seed", str(seed), "--epochs", str(args.epochs),
            "--output-dir", str(out),
        ], log)
        run([
            PY, "scripts/train_baselines.py",
            "--config", args.config, "--dataset", args.dataset,
            "--seed", str(seed), "--epochs", str(args.epochs), "--model", "all",
            "--output-dir", str(out / "baselines"),
        ], log)
        print(f"=== seed {seed} done ===", flush=True)


if __name__ == "__main__":
    main()
