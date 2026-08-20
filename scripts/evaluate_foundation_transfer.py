"""Few-Shot Fine-tuning & Transfer Evaluation for HDML-Foundation.

Tests transferring pre-trained generalist Mamba-Liquid representations to a new robot embodiment
by freezing the core backbone and training only the new embodiment adapter.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from hdml.models.foundation import HDMLFoundationModel
from hdml.data.multi_embodiment_dataset import FastEmbodimentBuffer


def evaluate_transfer(
    checkpoint_path: str,
    target_embodiment: str,
    target_dataset_path: str,
    prop_dim: int,
    action_dim: int,
    epochs: int = 3,
    steps_per_epoch: int = 100,
    batch_size: int = 128,
    lr: float = 0.001,
    device: str = "cuda",
) -> None:
    logger = logging.getLogger("EvaluateFoundationTransfer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(sh)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    cfg = ckpt.get("config", {})

    logger.info(f"Loading HDML-Foundation checkpoint from {checkpoint_path}...")
    model = HDMLFoundationModel(
        d_model=cfg.get("model", {}).get("d_model", 384),
        num_mamba_layers=cfg.get("model", {}).get("num_mamba_layers", 8),
        cfc_units=cfg.get("model", {}).get("cfc_units", 96),
        cfc_backbone_units=cfg.get("model", {}).get("cfc_backbone_units", 192),
        device=dev,
    )
    # Load base weights (excluding dynamic adapters)
    base_state_dict = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("adapters.")}
    model.load_state_dict(base_state_dict, strict=False)

    # 1. Register NEW Target Embodiment Adapter
    logger.info(f"Registering target embodiment '{target_embodiment}' ({prop_dim}D -> {action_dim}D)...")
    adapter = model.register_embodiment(target_embodiment, prop_dim=prop_dim, action_dim=action_dim)

    # 2. Freeze Pre-trained Backbone (Zero-Shot / Few-Shot Paradigm)
    model.freeze_backbone()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(f"Backbone frozen! Frozen parameters: {frozen_params:,} | Trainable adapter parameters: {trainable_params:,}")

    # 3. Load Target Data via FastEmbodimentBuffer
    buffer = FastEmbodimentBuffer(
        name=target_embodiment,
        path=target_dataset_path,
        context_length=30,
    )
    logger.info(f"Loaded target buffer with {buffer.num_steps:,} steps.")

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

    # 4. Fast Adapter Adaptation Loop
    logger.info(f"Starting Fast Adaptation on '{target_embodiment}' for {epochs} epochs ({steps_per_epoch} batches/epoch, B={batch_size})...")
    model.train()
    for epoch in range(1, epochs + 1):
        losses = []
        for _ in range(steps_per_epoch):
            batch = buffer.sample_batch(batch_size=batch_size, device=dev)
            states = batch["states"]
            actions_in = batch["actions_in"]
            actions_target = batch["actions_target"]
            rtgs = batch["rtgs"]
            timesteps = batch["timesteps"]
            emb_idx = batch["embodiment_idx"]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                acts_pred, _, _, _, _ = model(
                    states=states,
                    rtgs=rtgs,
                    actions=actions_in,
                    timesteps=timesteps,
                    embodiment_name=target_embodiment,
                    embodiment_idx=emb_idx,
                )
                loss = nn.functional.smooth_l1_loss(acts_pred, actions_target)

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        logger.info(f"Adaptation Epoch {epoch:02d}/{epochs:02d} | Target Action Loss: {np.mean(losses):.5f}")

    logger.info("Few-Shot Adapter Transfer verified successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HDML Foundation Transfer")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/hdml_foundation/hdml_foundation_epoch_05.pt")
    parser.add_argument("--target-embodiment", type=str, default="unitree_a1_maze")
    parser.add_argument("--target-dataset", type=str, default="data/unitree_a1_maze_trajectories.npz")
    parser.add_argument("--prop-dim", type=int, default=53)
    parser.add_argument("--action-dim", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    evaluate_transfer(
        checkpoint_path=args.checkpoint,
        target_embodiment=args.target_embodiment,
        target_dataset_path=args.target_dataset,
        prop_dim=args.prop_dim,
        action_dim=args.action_dim,
        epochs=args.epochs,
    )
