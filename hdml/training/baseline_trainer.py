from __future__ import annotations

import logging
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hdml.utils.config import TrainingConfig

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES: set[str] = {
    "dt",            # DecisionTransformerBaseline
    "rnn",           # DecisionRNNBaseline
    "mlp",           # MLPBCBaseline
    "diffusion",     # DiffusionPolicyBaseline
    "iql",           # IQLBaseline
    "mamba_mlp",     # MambaMLPHeadAblation
    "transformer_liquid",  # TransformerLiquidHeadAblation
}

IQL_EXPECTILE: float = 0.7
IQL_ADV_TEMP: float = 1.0


def _expectile_loss(errors: torch.Tensor, tau: float) -> torch.Tensor:
    """Asymmetric squared (expectile) loss: weights positive errors by tau, negatives by 1-tau."""
    weights = torch.where(errors > 0.0, tau, 1.0 - tau)
    return (weights * errors**2).mean()


class BaselineTrainer:
    """Generic offline trainer for baseline architectures and ablation variants.

    Trains the following model families with their native objectives:

    - Sequence behavior-cloning models (``dt``, ``rnn``, ``mamba_mlp``,
      ``transformer_liquid``): masked Smooth-L1 action loss against the causal
      no-leakage targets; ``mamba_mlp`` / ``transformer_liquid`` additionally use
      the auxiliary value / subgoal regularization terms for parity with HDML.
    - ``mlp``: Markovian behavior cloning on the last valid position of each window.
    - ``diffusion``: standard DDPM epsilon-prediction denoising loss.
    - ``iql``: expectile value regression + advantage-weighted behavior cloning
      (offline value anchoring; the twin Q-networks are not used in this setting).

    All models consume the same FastTensorTrajectoryDataset batches (states,
    actions, rtgs, timesteps, mask, target_actions, target_rtgs) used by HDML.
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        train_dataset: Dataset[dict[str, torch.Tensor]],
        val_dataset: Dataset[dict[str, torch.Tensor]] | None = None,
        config: TrainingConfig | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unsupported baseline model type: {model_type}. Choose from {sorted(SUPPORTED_MODEL_TYPES)}")
        self.model_type = model_type
        self.config = config if config is not None else TrainingConfig()
        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        use_cuda = self.device.type == "cuda"
        is_fast_dataset = hasattr(self.train_dataset, "states")
        num_workers = 0 if is_fast_dataset else (self.config.num_workers if use_cuda else 0)
        pin_memory = self.config.pin_memory if use_cuda else False

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
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
            )
        else:
            self.val_loader = None

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.use_amp = self.config.use_amp and use_cuda
        if self.config.amp_dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
            self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        else:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        total_steps = max(1, len(self.train_loader) * self.config.max_epochs)
        self.warmup_steps = min(self.config.warmup_steps, total_steps // 4)
        self.total_steps = total_steps
        self.current_step = 0

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    def _adjust_lr(self) -> float:
        """Warmup + cosine annealing learning rate schedule."""
        if self.current_step < self.warmup_steps and self.warmup_steps > 0:
            lr = self.config.learning_rate * (float(self.current_step + 1) / float(self.warmup_steps))
        else:
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            lr = self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = max(lr, self.config.learning_rate * 0.01)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _extract_last_valid(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather the last valid (masked) position of each window.

        Returns:
            states_last: (B, prop_dim)
            rtgs_last: (B, 1)
            target_actions_last: (B, action_dim)
        """
        mask = batch["mask"].to(self.device)
        j = mask.sum(dim=1).long().clamp(min=0) - 1
        j = j.clamp(min=0)
        batch_idx = torch.arange(mask.shape[0], device=self.device)
        states_last = batch["states"].to(self.device)[batch_idx, j]
        rtgs_last = batch["rtgs"].to(self.device)[batch_idx, j]
        actions_last = batch["target_actions"].to(self.device)[batch_idx, j]
        return states_last, rtgs_last, actions_last

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the native training objective for the configured model type."""
        states = batch["states"].to(self.device, non_blocking=True)
        actions = batch["actions"].to(self.device, non_blocking=True)
        rtgs = batch["rtgs"].to(self.device, non_blocking=True)
        timesteps = batch["timesteps"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)
        target_actions = batch["target_actions"].to(self.device, non_blocking=True)
        target_rtgs = batch["target_rtgs"].to(self.device, non_blocking=True)

        valid_tokens = torch.clamp(mask.sum(), min=1.0)
        mask_exp_act = mask.unsqueeze(-1)

        if self.model_type in ("dt", "rnn", "mamba_mlp", "transformer_liquid"):
            if self.model_type == "dt":
                actions_pred = self.model(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)
            elif self.model_type == "rnn":
                actions_pred, _ = self.model(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)
            elif self.model_type == "mamba_mlp":
                actions_pred, values_pred = self.model(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)
            else:  # transformer_liquid
                actions_pred, subgoals_pred, values_pred, _ = self.model(
                    states=states, rtgs=rtgs, actions=actions, timesteps=timesteps
                )

            raw_act = F.smooth_l1_loss(actions_pred, target_actions, reduction="none")
            action_loss = (raw_act * mask_exp_act).sum() / (valid_tokens * actions_pred.shape[-1])
            losses: dict[str, float] = {"action_loss": float(action_loss.item())}
            total_loss = action_loss

            if self.model_type == "mamba_mlp":
                raw_val = F.mse_loss(values_pred, target_rtgs, reduction="none")
                value_loss = (raw_val * mask_exp_act).sum() / valid_tokens
                total_loss = total_loss + self.config.reg_loss_weight * value_loss
                losses["value_loss"] = float(value_loss.item())

            if self.model_type == "transformer_liquid":
                raw_val = F.mse_loss(values_pred, target_rtgs, reduction="none")
                value_loss = (raw_val * mask_exp_act).sum() / valid_tokens
                subgoal_l2 = (subgoals_pred**2).sum(dim=-1)
                subgoal_loss = (subgoal_l2 * mask).sum() / valid_tokens
                total_loss = total_loss + self.config.reg_loss_weight * value_loss + self.config.subgoal_loss_weight * subgoal_loss
                losses["value_loss"] = float(value_loss.item())
                losses["subgoal_loss"] = float(subgoal_loss.item())

        elif self.model_type == "mlp":
            states_last, rtgs_last, actions_last = self._extract_last_valid(batch)
            # Use forward (not get_action) so gradients flow during training.
            actions_pred = self.model.forward(states_last, rtgs_last)
            raw_act = F.smooth_l1_loss(actions_pred, actions_last, reduction="none")
            total_loss = raw_act.mean()
            losses = {"action_loss": float(total_loss.item())}

        elif self.model_type == "diffusion":
            states_last, rtgs_last, actions_last = self._extract_last_valid(batch)
            noise = torch.randn_like(actions_last)
            k = torch.randint(0, self.model.denoising_steps, (states_last.shape[0],), device=self.device)
            alpha_cumprod = self.model.alphas_cumprod[k]  # (B,)
            sqrt_acp = torch.sqrt(alpha_cumprod).unsqueeze(-1)
            sqrt_1m_acp = torch.sqrt(1.0 - alpha_cumprod).unsqueeze(-1)
            noisy = sqrt_acp * actions_last + sqrt_1m_acp * noise
            eps_pred = self.model.forward(noisy, k, states_last.unsqueeze(1), rtgs_last.unsqueeze(1))
            total_loss = F.mse_loss(eps_pred, noise)
            losses = {"diffusion_loss": float(total_loss.item())}

        elif self.model_type == "iql":
            states_last, rtgs_last, actions_last = self._extract_last_valid(batch)
            v_pred = self.model.v(states_last)  # (B, 1)
            errors = rtgs_last - v_pred
            value_loss = _expectile_loss(errors, IQL_EXPECTILE)
            with torch.no_grad():
                adv = rtgs_last - v_pred.detach()
                adv_weights = torch.clamp(torch.exp(adv / IQL_ADV_TEMP), min=0.1, max=10.0)
            act_pred = self.model.actor(states_last)
            policy_loss = (adv_weights * F.smooth_l1_loss(act_pred, actions_last, reduction="none")).mean()
            total_loss = policy_loss + value_loss
            losses = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
            }
        else:
            raise ValueError(f"Unsupported baseline model type: {self.model_type}")

        losses["total_loss"] = float(total_loss.item())
        return total_loss, losses

    @torch.inference_mode()
    def _evaluate(self) -> float:
        """Compute mean total loss over the validation split."""
        if self.val_loader is None:
            return 0.0
        self.model.eval()
        total: list[float] = []
        for batch in self.val_loader:
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                loss, _ = self._compute_loss(batch)
            total.append(float(loss.item()))
        self.model.train()
        return float(sum(total) / max(1, len(total)))

    def train(self) -> list[dict[str, float]]:
        """Run the full training loop and return per-epoch metrics."""
        logger.info(
            f"Training baseline '{self.model_type}' for {self.config.max_epochs} epochs "
            f"({sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} params) on {self.device}."
        )
        history: list[dict[str, float]] = []
        for epoch in range(1, self.config.max_epochs + 1):
            self.model.train()
            epoch_losses: dict[str, float] = {}
            t0 = time.perf_counter()
            batch_bar = tqdm(
                self.train_loader,
                desc=f"{self.model_type} Epoch {epoch:02d}",
                leave=False,
                unit="batch",
                dynamic_ncols=True,
            )
            for batch in batch_bar:
                self.optimizer.zero_grad(set_to_none=True)
                self._adjust_lr()
                with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    loss, loss_dict = self._compute_loss(batch)
                if not torch.isfinite(loss).all():
                    raise RuntimeError(f"Non-finite loss in baseline '{self.model_type}': {loss.item()}")
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
                for k, v in loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                batch_bar.set_postfix(loss=f"{loss_dict.get('total_loss', float(loss.item())):.3f}")

            dt = time.perf_counter() - t0
            metrics = {
                "epoch": epoch,
                "loss": epoch_losses.get("total_loss", 0.0) / max(1, len(self.train_loader)),
                "throughput_fps": float(len(self.train_loader) * self.config.batch_size / max(1e-5, dt)),
            }
            for k, v in epoch_losses.items():
                metrics[f"train_{k}"] = v / max(1, len(self.train_loader))

            val_loss = self._evaluate()
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(self.output_dir / "best_model.pt")
            metrics["val_loss"] = val_loss

            history.append(metrics)
            logger.info(
                f"Epoch [{epoch:02d}/{self.config.max_epochs:02d}] "
                f"Train Loss: {metrics['loss']:.4f} | Val Loss: {val_loss:.4f} | "
                f"Speed: {metrics['throughput_fps']:.1f} samples/s"
            )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return history

    def save_checkpoint(self, path: Path | str) -> Path:
        """Save the trained model and dataset statistics."""
        save_file = Path(path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        state_mean = getattr(self.train_dataset, "state_mean", None)
        state_std = getattr(self.train_dataset, "state_std", None)
        state = {
            "model_type": self.model_type,
            "model_state_dict": self.model.state_dict(),
            "current_step": self.current_step,
            "best_loss": self.best_loss,
            "config": self.config.__dict__,
            "state_mean": state_mean,
            "state_std": state_std,
        }
        torch.save(state, save_file)
        logger.info(f"Saved baseline '{self.model_type}' checkpoint to: {save_file}")
        return save_file
