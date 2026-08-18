from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
      - Flow Matching and PAVE computation for HDML-V2
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

        # Loss Function for HDML-V2
        self.criterion = HDMLLoss(
            flow_weight=self.config.flow_weight,
            q_weight=self.config.q_weight,
            value_weight=self.config.value_weight,
            pave_weight=self.config.pave_weight,
            grad_caps_weight=self.config.grad_caps_weight,
            dynamics_weight=self.config.dynamics_weight,
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
        value_losses: list[float] = []
        subgoal_losses: list[float] = []
        dynamics_losses: list[float] = []
        grad_caps_losses: list[float] = []

        total_samples = 0
        t0 = time.perf_counter()

        batch_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:02d}",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in batch_bar:
            states = batch["states"].to(self.device, non_blocking=True)
            actions = batch["actions"].to(self.device, non_blocking=True)
            rtgs = batch["rtgs"].to(self.device, non_blocking=True)
            timesteps = batch["timesteps"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            
            target_actions = batch["target_actions"].to(self.device, non_blocking=True)
            target_rtgs = batch["target_rtgs"].to(self.device, non_blocking=True)
            next_states = batch["next_states"].to(self.device, non_blocking=True)

            valid_tokens = torch.clamp(mask.sum(), min=1.0)
            mask_exp_act = mask.unsqueeze(-1)
            mask_exp_state = mask.unsqueeze(-1)

            self.optimizer.zero_grad(set_to_none=True)
            lr = self._adjust_lr()

            # Sequence forward pass with AMP
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                actions_pred, subgoals_pred, values_pred, next_states_pred, _ = self.model(
                    states=states,
                    rtgs=rtgs,
                    actions=actions,
                    timesteps=timesteps,
                )
                
                # 1. Action imitation loss (Smooth L1)
                raw_act = F.smooth_l1_loss(actions_pred, target_actions, reduction="none")
                action_loss = (raw_act * mask_exp_act).sum() / (valid_tokens * actions_pred.shape[-1])
                
                # 2. Future state subgoal representation loss (eliminates latent collapse)
                if subgoals_pred.shape[-1] == next_states.shape[-1]:
                    raw_subgoal = F.smooth_l1_loss(subgoals_pred, next_states, reduction="none")
                    subgoal_loss = (raw_subgoal * mask_exp_state).sum() / (valid_tokens * subgoals_pred.shape[-1])
                else:
                    subgoal_loss = (subgoals_pred**2).sum() / (valid_tokens * subgoals_pred.shape[-1])

                # 3. Value loss
                raw_val = F.mse_loss(values_pred, target_rtgs, reduction="none")
                value_loss = (raw_val * mask_exp_act).sum() / valid_tokens

                # 4. Dynamics loss
                raw_dyn = F.smooth_l1_loss(next_states_pred, next_states, reduction="none")
                dynamics_loss = (raw_dyn * mask_exp_state).sum() / (valid_tokens * next_states_pred.shape[-1])

                # 5. Grad-CAPS Smoothness loss (penalty on consecutive action changes)
                if actions_pred.shape[1] > 1:
                    diff = actions_pred[:, 1:, :] - actions_pred[:, :-1, :]
                    mask_diff = mask[:, 1:].unsqueeze(-1)
                    grad_caps_loss = ((diff ** 2) * mask_diff).sum() / (torch.clamp(mask_diff.sum(), min=1.0) * actions_pred.shape[-1])
                else:
                    grad_caps_loss = torch.tensor(0.0, device=self.device)

                loss = (
                    action_loss
                    + self.config.reg_loss_weight * value_loss
                    + self.config.subgoal_loss_weight * subgoal_loss
                    + self.config.dynamics_weight * dynamics_loss
                    + self.config.grad_caps_weight * grad_caps_loss
                )
                loss_dict = {
                    "total_loss": float(loss.item()),
                    "action_loss": float(action_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "subgoal_loss": float(subgoal_loss.item()),
                    "dynamics_loss": float(dynamics_loss.item()),
                    "grad_caps_loss": float(grad_caps_loss.item()),
                }

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
            value_losses.append(loss_dict["value_loss"])
            subgoal_losses.append(loss_dict["subgoal_loss"])
            dynamics_losses.append(loss_dict["dynamics_loss"])
            grad_caps_losses.append(loss_dict["grad_caps_loss"])

            batch_bar.set_postfix(
                loss=f"{loss_dict['total_loss']:.3f}",
                act=f"{loss_dict['action_loss']:.3f}",
                lr=f"{lr:.1e}",
            )

        t1 = time.perf_counter()
        throughput = float(total_samples / max(1e-5, (t1 - t0)))

        metrics = {
            "epoch": epoch,
            "train_loss": float(sum(total_losses) / max(1, len(total_losses))),
            "train_action_loss": float(sum(action_losses) / max(1, len(action_losses))),
            "train_value_loss": float(sum(value_losses) / max(1, len(value_losses))),
            "train_subgoal_loss": float(sum(subgoal_losses) / max(1, len(subgoal_losses))),
            "train_dynamics_loss": float(sum(dynamics_losses) / max(1, len(dynamics_losses))),
            "train_grad_caps_loss": float(sum(grad_caps_losses) / max(1, len(grad_caps_losses))),
            "throughput_fps": throughput,
        }

        # Log training metrics to TensorBoard & WandB
        if self.writer is not None:
            for k, v in metrics.items():
                if k != "epoch":
                    self.writer.add_scalar(f"Train/{k}", v, epoch)

        if self.wandb_active:
            try:
                import wandb
                wandb_metrics = {f"train/{k}": v for k, v in metrics.items() if k != "epoch"}
                wandb_metrics["train/epoch"] = epoch
                wandb.log(wandb_metrics)
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
        value_losses: list[float] = []
        subgoal_losses: list[float] = []
        dynamics_losses: list[float] = []

        for batch in self.val_loader:
            states = batch["states"].to(self.device, non_blocking=True)
            actions = batch["actions"].to(self.device, non_blocking=True)
            rtgs = batch["rtgs"].to(self.device, non_blocking=True)
            timesteps = batch["timesteps"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            
            target_actions = batch["target_actions"].to(self.device, non_blocking=True)
            target_rtgs = batch["target_rtgs"].to(self.device, non_blocking=True)
            next_states = batch["next_states"].to(self.device, non_blocking=True)

            valid_tokens = torch.clamp(mask.sum(), min=1.0)
            mask_exp_act = mask.unsqueeze(-1)
            mask_exp_state = mask.unsqueeze(-1)

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                actions_pred, subgoals_pred, values_pred, next_states_pred, _ = self.model(
                    states=states,
                    rtgs=rtgs,
                    actions=actions,
                    timesteps=timesteps,
                )
                
                raw_act = F.smooth_l1_loss(actions_pred, target_actions, reduction="none")
                action_loss = (raw_act * mask_exp_act).sum() / (valid_tokens * actions_pred.shape[-1])
                
                if subgoals_pred.shape[-1] == next_states.shape[-1]:
                    raw_subgoal = F.smooth_l1_loss(subgoals_pred, next_states, reduction="none")
                    subgoal_loss = (raw_subgoal * mask_exp_state).sum() / (valid_tokens * subgoals_pred.shape[-1])
                else:
                    subgoal_loss = (subgoals_pred**2).sum() / (valid_tokens * subgoals_pred.shape[-1])

                raw_val = F.mse_loss(values_pred, target_rtgs, reduction="none")
                value_loss = (raw_val * mask_exp_act).sum() / valid_tokens

                raw_dyn = F.smooth_l1_loss(next_states_pred, next_states, reduction="none")
                dynamics_loss = (raw_dyn * mask_exp_state).sum() / (valid_tokens * next_states_pred.shape[-1])

                if actions_pred.shape[1] > 1:
                    diff = actions_pred[:, 1:, :] - actions_pred[:, :-1, :]
                    mask_diff = mask[:, 1:].unsqueeze(-1)
                    grad_caps_loss = ((diff ** 2) * mask_diff).sum() / (torch.clamp(mask_diff.sum(), min=1.0) * actions_pred.shape[-1])
                else:
                    grad_caps_loss = torch.tensor(0.0, device=self.device)

                loss = (
                    action_loss
                    + self.config.reg_loss_weight * value_loss
                    + self.config.subgoal_loss_weight * subgoal_loss
                    + self.config.dynamics_weight * dynamics_loss
                    + self.config.grad_caps_weight * grad_caps_loss
                )

            total_losses.append(float(loss.item()))
            action_losses.append(float(action_loss.item()))
            value_losses.append(float(value_loss.item()))
            subgoal_losses.append(float(subgoal_loss.item()))
            dynamics_losses.append(float(dynamics_loss.item()))

        val_metrics = {
            "val_loss": float(sum(total_losses) / max(1, len(total_losses))),
            "val_action_loss": float(sum(action_losses) / max(1, len(action_losses))),
            "val_value_loss": float(sum(value_losses) / max(1, len(value_losses))),
            "val_subgoal_loss": float(sum(subgoal_losses) / max(1, len(subgoal_losses))),
            "val_dynamics_loss": float(sum(dynamics_losses) / max(1, len(dynamics_losses))),
        }

        if self.writer is not None:
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"Val/{k}", v, self.current_step)

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

        epoch_bar = tqdm(
            range(1, self.config.max_epochs + 1),
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_bar:
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            combined = {**train_metrics, **val_metrics}
            history.append(combined)

            current_val = val_metrics.get("val_loss", train_metrics["train_loss"])
            is_best = current_val < self.best_loss
            if is_best:
                self.best_loss = current_val
                # Persist the best model immediately, independent of the periodic interval.
                self.save_checkpoint(self.output_dir / "best_model.pt")

            if "val_loss" in val_metrics and self.wandb_active:
                try:
                    import wandb
                    wandb_val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                    wandb_val_metrics["val/epoch"] = epoch
                    wandb.log(wandb_val_metrics)
                except Exception as e:  # noqa: BLE001 - logging must not crash training
                    logger.warning(f"WandB validation logging error: {e}")

            val_str = f" | Val Loss: {val_metrics['val_loss']:.4f}" if "val_loss" in val_metrics else ""
            logger.info(
                f"Epoch [{epoch:02d}/{self.config.max_epochs:02d}] "
                f"Train: {train_metrics['train_loss']:.4f} (Act: {train_metrics['train_action_loss']:.4f}, Val: {train_metrics['train_value_loss']:.4f})"
                f"{val_str} | Speed: {train_metrics['throughput_fps']:.1f} frames/s | Best: {self.best_loss:.4f}"
            )
            epoch_bar.set_postfix(
                train=f"{train_metrics['train_loss']:.3f}",
                val=f"{val_metrics.get('val_loss', float('nan')):.3f}",
                best=f"{self.best_loss:.3f}",
            )

            # Periodic checkpointing
            if epoch % self.config.save_interval == 0 or epoch == self.config.max_epochs:
                chkpt_path = self.output_dir / f"checkpoint_ep{epoch:03d}.pt"
                self.save_checkpoint(chkpt_path)

        logger.info("Training complete.")
        return history
