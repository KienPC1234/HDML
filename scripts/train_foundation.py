"""Multi-Embodiment Pre-training Script for HDML-Foundation.

Trains a unified Mamba-Liquid sequence backbone across multiple robotics datasets.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hdml.models.foundation import HDMLFoundationModel
from hdml.data.multi_embodiment_dataset import MultiEmbodimentDataset, collate_multi_embodiment


def setup_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("TrainHDMLFoundation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(Path(log_dir) / "train_foundation.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def train_foundation(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger = setup_logger(cfg["training"]["log_dir"])
    logger.info(f"Loaded HDML-Foundation config from {config_path}")

    device = torch.device(cfg["model"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    # 1. Initialize Foundation Model
    model = HDMLFoundationModel(
        d_model=cfg["model"]["d_model"],
        num_mamba_layers=cfg["model"]["num_mamba_layers"],
        d_state=cfg["model"]["d_state"],
        d_conv=cfg["model"]["d_conv"],
        expand=cfg["model"]["expand"],
        cfc_units=cfg["model"]["cfc_units"],
        cfc_backbone_units=cfg["model"]["cfc_backbone_units"],
        cfc_residual=cfg["model"]["cfc_residual"],
        max_timesteps=cfg["model"]["max_timesteps"],
        max_embodiments=cfg["model"]["max_embodiments"],
        dropout=cfg["model"]["dropout"],
        device=device,
    )

    # 2. Register Embodiments from Config
    embodiment_paths: dict[str, str] = {}
    for name, spec in cfg["embodiments"].items():
        model.register_embodiment(
            name=name,
            prop_dim=spec["prop_dim"],
            action_dim=spec["action_dim"],
        )
        if Path(spec["path"]).exists():
            embodiment_paths[name] = spec["path"]
            logger.info(f"Registered embodiment '{name}': {spec['prop_dim']}D state -> {spec['action_dim']}D action (Found dataset: {spec['path']})")
        else:
            logger.warning(f"Dataset for embodiment '{name}' not found at {spec['path']}, skipping from active pretraining.")

    if len(embodiment_paths) == 0:
        logger.error("No valid embodiment datasets found. Please check paths in config.")
        return

    # 3. Create Multi-Embodiment Dataset
    dataset = MultiEmbodimentDataset(
        embodiment_paths=embodiment_paths,
        context_length=cfg["training"]["context_length"],
        gamma=cfg["training"]["gamma"],
    )
    logger.info(f"MultiEmbodimentDataset created with {len(dataset)} total multi-robot trajectories.")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_multi_embodiment,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
    )

    # 4. Optimizer & Scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["use_amp"])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"HDML-Foundation Model initialized with {total_params:,} trainable parameters.")

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. Training Loop
    max_epochs = cfg["training"]["max_epochs"]
    logger.info(f"Starting Multi-Embodiment Pre-training for {max_epochs} epochs...")

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        action_losses: list[float] = []

        for batch_dict in dataloader:
            optimizer.zero_grad()
            batch_total_loss = torch.tensor(0.0, device=device)

            for emb_name, batch in batch_dict.items():
                states = batch["states"].to(device)
                actions_in = batch["actions_in"].to(device)
                actions_target = batch["actions_target"].to(device)
                rtgs = batch["rtgs"].to(device)
                timesteps = batch["timesteps"].to(device)
                emb_idx = batch["embodiment_idx"]

                with torch.amp.autocast("cuda", enabled=cfg["training"]["use_amp"], dtype=torch.bfloat16):
                    acts_pred, intents_pred, values_pred, rtg_pred, _ = model(
                        states=states,
                        rtgs=rtgs,
                        actions=actions_in,
                        timesteps=timesteps,
                        embodiment_name=emb_name,
                        embodiment_idx=emb_idx,
                    )

                    # Smooth-L1 Loss on actions
                    l_act = nn.functional.smooth_l1_loss(acts_pred, actions_target)
                    l_rtg = nn.functional.mse_loss(rtg_pred, rtgs)
                    l_intent = 0.01 * (intents_pred**2).mean()

                    loss = (
                        cfg["training"]["action_weight"] * l_act
                        + cfg["training"]["rtg_weight"] * l_rtg
                        + cfg["training"]["intent_weight"] * l_intent
                    )

                batch_total_loss = batch_total_loss + loss

            scaler.scale(batch_total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(batch_total_loss.item())

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        logger.info(f"Epoch {epoch:03d}/{max_epochs:03d} | Multi-Embodiment Loss: {mean_loss:.5f}")

        if epoch % cfg["training"]["save_interval"] == 0 or epoch == max_epochs:
            ckpt_path = output_dir / f"hdml_foundation_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": mean_loss,
                    "config": cfg,
                },
                ckpt_path,
            )
            logger.info(f"Saved foundation checkpoint to {ckpt_path}")

    # Save final best foundation checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
        },
        output_dir / "hdml_foundation_best.pt",
    )
    logger.info(f"Pre-training complete! Final model saved to {output_dir / 'hdml_foundation_best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HDML-Foundation Model")
    parser.add_argument("--config", type=str, default="configs/hdml_foundation_base.yaml")
    args = parser.parse_args()
    train_foundation(args.config)
