"""High-Performance Multi-Embodiment Pre-training Script for HDML-Foundation.

Utilizes FastMultiEmbodimentManager for zero-copy GPU transfers, large batch sizing (B=256),
and 100% GPU saturation on NVIDIA RTX 4070 SUPER.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from hdml.models.foundation import HDMLFoundationModel
from hdml.data.multi_embodiment_dataset import FastMultiEmbodimentManager
from hdml.utils.foundation_config import load_foundation_config


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
    cfg = load_foundation_config(config_path)

    # CLI overrides via environment (set by __main__ below; keeps the
    # function signature stable for programmatic callers).
    if os.environ.get("HDML_FM_EPOCHS"):
        cfg["training"]["max_epochs"] = int(os.environ["HDML_FM_EPOCHS"])
    if os.environ.get("HDML_FM_BATCH"):
        cfg["training"]["batch_size"] = int(os.environ["HDML_FM_BATCH"])
    if os.environ.get("HDML_FM_OUTDIR"):
        cfg["training"]["output_dir"] = os.environ["HDML_FM_OUTDIR"]
    if os.environ.get("HDML_FM_LOGDIR"):
        cfg["training"]["log_dir"] = os.environ["HDML_FM_LOGDIR"]
    if os.environ.get("HDML_FM_SCALERET"):
        cfg["training"]["scale_return"] = float(os.environ["HDML_FM_SCALERET"])

    logger = setup_logger(cfg["training"]["log_dir"])
    logger.info(f"Loaded HDML-Foundation config from {config_path}")

    device = torch.device(cfg["model"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])
    # GPU throughput knobs that do not change numerics: TF32 + cudnn autotune
    # for fixed input shapes (same policy as HDMLTrainer).
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        except Exception as e:  # noqa: BLE001 - backend flags must not crash training
            logger.warning(f"Could not enable TF32/high-precision matmul: {e}")

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
        use_native_mamba3=cfg["model"].get("use_native_mamba3", True),
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
            logger.info(f"Registered embodiment '{name}': {spec['prop_dim']}D state -> {spec['action_dim']}D action (Dataset: {spec['path']})")
        else:
            raise FileNotFoundError(f"Missing required embodiment dataset: {spec['path']}")

    if len(embodiment_paths) == 0:
        raise ValueError("No embodiment datasets configured")

    # 3. Initialize High-Performance FastMultiEmbodimentManager
    # scale_return keeps RTG targets O(1) so the RTG-MSE term cannot drown the
    # action-imitation loss (raw returns are O(1e2)-O(1e3) on locomotion data).
    scale_return = float(cfg["training"].get("scale_return", 1000.0))
    manager = FastMultiEmbodimentManager(
        embodiment_paths=embodiment_paths,
        context_length=cfg["training"]["context_length"],
        gamma=cfg["training"]["gamma"], scale_return=scale_return,
        split="train", split_seed=cfg["training"]["seed"],
    )
    logger.info(f"FastMultiEmbodimentManager initialized with {manager.total_steps:,} total multi-robot steps.")

    validation_manager = FastMultiEmbodimentManager(
        embodiment_paths, context_length=cfg["training"]["context_length"],
        gamma=cfg["training"]["gamma"], scale_return=scale_return,
        split="validation", split_seed=cfg["training"]["seed"])
    data_manifest = {
        name: {"train_episode_hashes": manager.buffers[name].episode_hashes,
               "validation_episode_hashes": validation_manager.buffers[name].episode_hashes,
               "state_mean": manager.buffers[name].state_mean.tolist(),
               "state_std": manager.buffers[name].state_std.tolist()}
        for name in manager.embodiment_names}
    embodiment_indices = {name: manager.buffers[name].embodiment_idx for name in manager.embodiment_names}

    # 4. Optimizer & Scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bfloat16 does not need scaling

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"HDML-Foundation Model initialized with {total_params:,} trainable parameters on {device}.")

    output_dir = Path(cfg["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 5. High-Speed Training Loop
    max_epochs = cfg["training"]["max_epochs"]
    batch_size = cfg["training"].get("batch_size", 256)
    steps_per_epoch = max(100, len(embodiment_paths) * 50)  # 50 large batches per embodiment per epoch

    active_embodiment_names = list(embodiment_paths.keys())
    # Embodiment sampling policy (config `sampling`):
    # - round_robin: equal share per embodiment (default, legacy behaviour).
    # - proportional_sqrt: sample probability ~ sqrt(train steps). Tiny
    #   embodiments (pen/door ~5k steps) would otherwise be recycled hundreds
    #   of times per epoch while 1M-step ones are undersampled -> overfit.
    sampling = str(cfg["training"].get("sampling", "round_robin"))
    sample_weights: list[float] | None = None
    sample_gen: torch.Generator | None = None
    sample_prob: torch.Tensor | None = None
    if sampling == "proportional_sqrt":
        sample_weights = [math.sqrt(max(1, manager.buffers[n].num_steps))
                          for n in active_embodiment_names]
        sample_prob = torch.tensor(sample_weights) / sum(sample_weights)
        sample_gen = torch.Generator().manual_seed(cfg["training"]["seed"] + 777)
        logger.info("Proportional-sqrt sampling weights: "
                    + ", ".join(f"{n}={w:.0f}" for n, w in zip(active_embodiment_names, sample_weights)))
    elif sampling != "round_robin":
        raise ValueError(f"Unknown sampling policy: {sampling}")

    # LR schedule (config `lr_schedule`): constant | cosine with linear warmup.
    lr_schedule = str(cfg["training"].get("lr_schedule", "constant"))
    base_lr = float(cfg["training"]["learning_rate"])
    warmup = int(cfg["training"].get("warmup_steps", 500))
    total_steps = max(1, max_epochs * steps_per_epoch)
    global_step = 0

    def _adjust_lr() -> float:
        if lr_schedule == "constant":
            return base_lr
        if lr_schedule == "cosine":
            if global_step < warmup and warmup > 0:
                lr = base_lr * (float(global_step + 1) / float(warmup))
            else:
                progress = float(global_step - warmup) / float(max(1, total_steps - warmup))
                lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                lr = max(lr, base_lr * 0.01)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            return lr
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

    patience = int(cfg["training"].get("early_stopping_patience", 0))
    logger.info(f"Starting High-Speed Foundation Pre-training ({max_epochs} epochs, batch_size={batch_size}, steps_per_epoch={steps_per_epoch}, "
                f"sampling={sampling}, lr_schedule={lr_schedule}, patience={patience})...")

    best_validation_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for step in range(steps_per_epoch):
            if sample_weights is None:
                emb_name = active_embodiment_names[step % len(active_embodiment_names)]
            else:
                assert sample_gen is not None and sample_prob is not None
                pick = int(torch.multinomial(sample_prob, 1, generator=sample_gen).item())
                emb_name = active_embodiment_names[pick]
            batch = manager.sample_embodiment_batch(emb_name, batch_size=batch_size, device=device)

            states = batch["states"]
            actions_in = batch["actions_in"]
            actions_target = batch["actions_target"]
            rtgs = batch["rtgs"]
            timesteps = batch["timesteps"]
            emb_idx = batch["embodiment_idx"]

            optimizer.zero_grad(set_to_none=True)
            _adjust_lr()
            global_step += 1

            with torch.amp.autocast("cuda", enabled=cfg["training"]["use_amp"] and device.type == "cuda", dtype=torch.bfloat16):
                acts_pred, intents_pred, values_pred, rtg_pred, _ = model(
                    states=states,
                    rtgs=rtgs,
                    actions=actions_in,
                    timesteps=timesteps,
                    embodiment_name=emb_name,
                    embodiment_idx=emb_idx,
                )

                l_act = nn.functional.smooth_l1_loss(acts_pred, actions_target)
                l_rtg = nn.functional.mse_loss(rtg_pred, rtgs)
                # Intent scale comes from intent_weight alone (a second 0.01
                # factor here previously discounted it twice to ~1e-4).
                l_intent = (intents_pred**2).mean()

                loss = (
                    cfg["training"]["action_weight"] * l_act
                    + cfg["training"]["rtg_weight"] * l_rtg
                    + cfg["training"]["intent_weight"] * l_intent
                )

            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite foundation training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        logger.info(f"Epoch {epoch:02d}/{max_epochs:02d} | Foundation Multi-Embodiment Loss: {mean_loss:.5f}")

        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for name in manager.embodiment_names:
                generator = torch.Generator().manual_seed(cfg["training"]["seed"] + 10000)
                for _ in range(10):
                    batch = validation_manager.buffers[name].sample_batch(batch_size, device, generator)
                    pred, _, _, _, _ = model(batch["states"], batch["rtgs"], batch["actions_in"],
                                             batch["timesteps"], name, batch["embodiment_idx"])
                    validation_losses.append(nn.functional.smooth_l1_loss(pred, batch["actions_target"]).item())
        validation_loss = float(np.mean(validation_losses))
        if not np.isfinite(validation_loss):
            raise FloatingPointError("Nonfinite foundation validation loss")
        logger.info(f"Epoch {epoch}: held-out prediction loss={validation_loss:.6f}")
        checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "config": cfg,
                      "train_loss": mean_loss, "validation_loss": validation_loss,
                      "data_manifest": data_manifest, "embodiment_indices": embodiment_indices,
                      "protocol_version": 2}
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_no_improve = 0
            torch.save(checkpoint, output_dir / "hdml_foundation_best.pt")
        else:
            epochs_no_improve += 1
        if epoch % cfg["training"]["save_interval"] == 0 or epoch == max_epochs:
            torch.save(checkpoint, output_dir / f"hdml_foundation_epoch_{epoch:02d}.pt")
        if patience > 0 and epochs_no_improve >= patience:
            logger.info(f"Early stopping: no validation improvement for {epochs_no_improve} epochs "
                        f"(best={best_validation_loss:.6f}).")
            break
    logger.info("Training completed; best checkpoint selected by held-out action prediction loss")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HDML-Foundation Model")
    parser.add_argument("--config", type=str, default="configs/hdml_foundation_full.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output_dir (checkpoints)")
    parser.add_argument("--log-dir", type=str, default=None, help="Override log_dir")
    parser.add_argument("--scale-return", type=float, default=None, help="Override RTG scale")
    args = parser.parse_args()
    if args.epochs is not None:
        os.environ["HDML_FM_EPOCHS"] = str(args.epochs)
    if args.batch_size is not None:
        os.environ["HDML_FM_BATCH"] = str(args.batch_size)
    if args.output_dir is not None:
        os.environ["HDML_FM_OUTDIR"] = args.output_dir
    if args.log_dir is not None:
        os.environ["HDML_FM_LOGDIR"] = args.log_dir
    if args.scale_return is not None:
        os.environ["HDML_FM_SCALERET"] = str(args.scale_return)
    train_foundation(args.config)

