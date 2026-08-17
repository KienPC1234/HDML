from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from hdml.models.hdml_model import HDMLModel
from hdml.training.losses import HDMLLoss
from hdml.utils.config import TrainingConfig

logger = logging.getLogger(__name__)


class HDMLTrainer:
    """High-Performance Trainer engine for Hierarchical Decision Mamba-Liquid models.

    Implements:
      - Automatic Mixed Precision (AMP: BFloat16 / FP16) with GradScaler
      - High-throughput multi-worker DataLoaders with pinned memory and prefetching
      - Linear Warmup + Cosine Annealing learning rate schedule
      - Gradient clipping and NaN/Inf safeguards
      - Model checkpointing and loss logging
    """

    def __init__(
        self,
        model: HDMLModel,
        train_dataset: Dataset[dict[str, torch.Tensor]],
        val_dataset: Dataset[dict[str, torch.Tensor]] | None = None,
        config: TrainingConfig | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.config = config if config is not None else TrainingConfig()
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # DataLoader configuration
        use_cuda = self.device.type == "cuda"
        is_fast_dataset = hasattr(self.train_dataset, "states")
        num_workers = 0 if is_fast_dataset else (self.config.num_workers if use_cuda else 0)
        pin_memory = self.config.pin_memory if use_cuda else False
        persistent_workers = (num_workers > 0) and self.config.persistent_workers
        prefetch_factor = self.config.prefetch_factor if num_workers > 0 else None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

        if self.val_dataset is not None:
            is_fast_val = hasattr(self.val_dataset, "states")
            val_workers = 0 if is_fast_val else (self.config.num_workers if use_cuda else 0)
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=val_workers,
                pin_memory=pin_memory,
                persistent_workers=(val_workers > 0) and self.config.persistent_workers,
                prefetch_factor=self.config.prefetch_factor if val_workers > 0 else None,
            )
        else:
            self.val_loader = None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Automatic Mixed Precision (AMP)
        self.use_amp = self.config.use_amp and use_cuda
        if self.config.amp_dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)  # bfloat16 doesn't require scaling
        else:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Loss Function
        self.criterion = HDMLLoss(
            action_weight=self.config.action_loss_weight,
            subgoal_weight=self.config.subgoal_loss_weight,
            value_weight=self.config.reg_loss_weight,
            use_advantage_weighting=self.config.use_advantage_weighting,
            advantage_temperature=self.config.advantage_temperature,
        )

        # Learning Rate Schedule
        total_steps = max(1, len(self.train_loader) * self.config.max_epochs)
        self.warmup_steps = min(self.config.warmup_steps, total_steps // 4)
        self.total_steps = total_steps
        self.current_step = 0

        # Checkpoints setup
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

        # TensorBoard & WandB Web Tracking
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer: SummaryWriter | None = SummaryWriter(log_dir=str(self.log_dir))
                logger.info(f"TensorBoard logging enabled. Log dir: {self.log_dir}")
            except ImportError:
                logger.warning("torch.utils.tensorboard not available. Disabling TensorBoard.")
                self.writer = None
        else:
            self.writer = None

        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=self.config.__dict__,
                )
                self.wandb_active = True
                logger.info(f"WandB logging active. Project: {self.config.wandb_project}")
            except Exception as e:
                logger.warning(f"Could not initialize WandB: {e}")
                self.wandb_active = False
        else:
            self.wandb_active = False

    def _adjust_lr(self) -> float:
        """Calculate and set the learning rate using Warmup + Cosine Decay."""
        if self.current_step < self.warmup_steps and self.warmup_steps > 0:
            lr = self.config.learning_rate * (float(self.current_step + 1) / float(self.warmup_steps))
        else:
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr = self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = max(lr, self.config.learning_rate * 0.01)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train the model for one epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Dictionary of averaged epoch training metrics.
        """
        self.model.train()
        total_losses: list[float] = []
        action_losses: list[float] = []
        subgoal_losses: list[float] = []
        value_losses: list[float] = []

        total_samples = 0
        t0 = time.perf_counter()

        for batch in self.train_loader:
            states = batch["states"].to(self.device, non_blocking=True)
            actions = batch["actions"].to(self.device, non_blocking=True)
            rtgs = batch["rtgs"].to(self.device, non_blocking=True)
            timesteps = batch["timesteps"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            target_actions = batch["target_actions"].to(self.device, non_blocking=True)
            target_rtgs = batch["target_rtgs"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            self._adjust_lr()

            # Sequence forward pass with AMP
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                actions_pred, subgoals_pred, values_pred, _ = self.model(
                    states=states,
                    rtgs=rtgs,
                    actions=actions,
                    timesteps=timesteps,
                )

                loss, loss_dict = self.criterion(
                    actions_pred=actions_pred,
                    target_actions=target_actions,
                    subgoals=subgoals_pred,
                    values_pred=values_pred,
                    target_rtgs=target_rtgs,
                    mask=mask,
                )

            # Defensive NaN/Inf Safeguard
            if not torch.isfinite(loss).all():
                logger.error(f"Non-finite loss detected: {loss.item()} at epoch {epoch}, step {self.current_step}")
                raise RuntimeError(f"NaN or Inf loss encountered during training: {loss.item()}")

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()

            self.current_step += 1
            total_samples += states.shape[0]

            total_losses.append(loss_dict["total_loss"])
            action_losses.append(loss_dict["action_loss"])
            subgoal_losses.append(loss_dict["subgoal_loss"])
            value_losses.append(loss_dict["value_loss"])

        t1 = time.perf_counter()
        throughput = float(total_samples / max(1e-5, (t1 - t0)))

        metrics = {
            "epoch": epoch,
            "train_loss": float(sum(total_losses) / max(1, len(total_losses))),
            "train_action_loss": float(sum(action_losses) / max(1, len(action_losses))),
            "train_subgoal_loss": float(sum(subgoal_losses) / max(1, len(subgoal_losses))),
            "train_value_loss": float(sum(value_losses) / max(1, len(value_losses))),
            "throughput_fps": throughput,
        }

        # Log training metrics to TensorBoard & WandB
        if self.writer is not None:
            self.writer.add_scalar("Train/TotalLoss", metrics["train_loss"], epoch)
            self.writer.add_scalar("Train/ActionLoss", metrics["train_action_loss"], epoch)
            self.writer.add_scalar("Train/SubgoalLoss", metrics["train_subgoal_loss"], epoch)
            self.writer.add_scalar("Train/ValueLoss", metrics["train_value_loss"], epoch)
            self.writer.add_scalar("Train/Throughput_FPS", metrics["throughput_fps"], epoch)

        if self.wandb_active:
            try:
                import wandb
                wandb.log({
                    "train/epoch": epoch,
                    "train/loss": metrics["train_loss"],
                    "train/action_loss": metrics["train_action_loss"],
                    "train/subgoal_loss": metrics["train_subgoal_loss"],
                    "train/value_loss": metrics["train_value_loss"],
                    "train/throughput_fps": metrics["throughput_fps"],
                })
            except Exception as e:
                logger.warning(f"WandB logging error: {e}")

        return metrics

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        """Evaluate the model on the validation dataset."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_losses: list[float] = []
        action_losses: list[float] = []

        for batch in self.val_loader:
            states = batch["states"].to(self.device, non_blocking=True)
            actions = batch["actions"].to(self.device, non_blocking=True)
            rtgs = batch["rtgs"].to(self.device, non_blocking=True)
            timesteps = batch["timesteps"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            target_actions = batch["target_actions"].to(self.device, non_blocking=True)
            target_rtgs = batch["target_rtgs"].to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                actions_pred, subgoals_pred, values_pred, _ = self.model(
                    states=states,
                    rtgs=rtgs,
                    actions=actions,
                    timesteps=timesteps,
                )

                _, loss_dict = self.criterion(
                    actions_pred=actions_pred,
                    target_actions=target_actions,
                    subgoals=subgoals_pred,
                    values_pred=values_pred,
                    target_rtgs=target_rtgs,
                    mask=mask,
                )

            total_losses.append(loss_dict["total_loss"])
            action_losses.append(loss_dict["action_loss"])

        val_metrics = {
            "val_loss": float(sum(total_losses) / max(1, len(total_losses))),
            "val_action_loss": float(sum(action_losses) / max(1, len(action_losses))),
        }
        return val_metrics

    def save_checkpoint(self, path: Path | str, is_best: bool = False) -> Path:
        """Save current training state and weights."""
        save_file = Path(path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        state_mean = getattr(self.train_dataset, "state_mean", None)
        state_std = getattr(self.train_dataset, "state_std", None)

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_step": self.current_step,
            "best_loss": self.best_loss,
            "config": self.config.__dict__,
            "state_mean": state_mean,
            "state_std": state_std,
        }
        torch.save(state, save_file)
        if is_best:
            best_file = save_file.parent / "best_model.pt"
            torch.save(state, best_file)
            logger.info(f"Saved new best model checkpoint to: {best_file}")
        return save_file

    def train(self) -> list[dict[str, float]]:
        """Run full training loop across all configured epochs."""
        amp_info = f" | AMP: {self.config.amp_dtype}" if self.use_amp else " | AMP: Disabled"
        logger.info(f"Starting HDML training for {self.config.max_epochs} epochs on {self.device}{amp_info}...")
        history: list[dict[str, float]] = []

        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            combined = {**train_metrics, **val_metrics}
            history.append(combined)

            current_val = val_metrics.get("val_loss", train_metrics["train_loss"])
            is_best = current_val < self.best_loss
            if is_best:
                self.best_loss = current_val

            if "val_loss" in val_metrics and self.writer is not None:
                self.writer.add_scalar("Val/TotalLoss", val_metrics["val_loss"], epoch)
                self.writer.add_scalar("Val/ActionLoss", val_metrics["val_action_loss"], epoch)

            if "val_loss" in val_metrics and self.wandb_active:
                try:
                    import wandb
                    wandb.log({
                        "val/epoch": epoch,
                        "val/loss": val_metrics["val_loss"],
                        "val/action_loss": val_metrics["val_action_loss"],
                    })
                except Exception:
                    pass

            val_str = f" | Val Loss: {val_metrics['val_loss']:.4f}" if "val_loss" in val_metrics else ""
            logger.info(
                f"Epoch [{epoch:02d}/{self.config.max_epochs:02d}] "
                f"Train: {train_metrics['train_loss']:.4f} (Act: {train_metrics['train_action_loss']:.4f})"
                f"{val_str} | Speed: {train_metrics['throughput_fps']:.1f} frames/s | Best: {self.best_loss:.4f}"
            )

            # Periodic saving
            if epoch % self.config.save_interval == 0 or epoch == self.config.max_epochs:
                ckpt_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(ckpt_path, is_best=is_best)

        # Clear VRAM cache at end of training
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if self.writer is not None:
            self.writer.close()
            logger.info(f"TensorBoard session closed. Launch dashboard with: tensorboard --logdir {self.log_dir}")

        if self.wandb_active:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        logger.info("HDML training completed successfully.")
        return history
