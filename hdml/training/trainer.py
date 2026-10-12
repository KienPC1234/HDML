from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hdml.models.flow_policy import GaussianActionPolicy
from hdml.models.hdml_model import HDMLModel
from hdml.training.losses import advantage_weights
from hdml.utils.config import TrainingConfig

logger = logging.getLogger(__name__)

_VALID_BEST_METRICS = (
    "val_loss",
    "val_action_loss",
    "val_flow_loss",
    "val_value_loss",
    "val_subgoal_loss",
    "val_dynamics_loss",
)


class TensorBatchLoader:
    """Batch iterator over pre-vectorised contiguous tensor buffers.

    Replaces ``DataLoader`` for :class:`FastTensorTrajectoryDataset`: one
    ``index_select`` per field per batch instead of per-sample ``__getitem__``
    plus ``default_collate``. Buffers live on the training device when they fit
    the configured budget, otherwise on (pinned) host memory.

    Shuffling reproduces ``RandomSampler(generator=g)`` exactly: per epoch it
    draws ``randperm(n)`` and then a second ``randperm(n)`` whose (empty) slice
    ``RandomSampler`` consumes, so batch order and generator state match the
    previous ``DataLoader(shuffle=True, generator=g)`` pipeline.
    """

    def __init__(
        self,
        tensors: dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool,
        device: torch.device,
        generator: torch.Generator | None = None,
        max_device_bytes: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if not tensors:
            raise ValueError("TensorBatchLoader requires at least one tensor field.")
        sizes = {key: t.shape[0] for key, t in tensors.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"All tensor fields need the same leading dimension, got {sizes}")
        if shuffle and generator is None:
            raise ValueError("A seeded generator is required when shuffle=True.")

        self.num_samples = next(iter(sizes.values()))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.generator = generator

        total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        self.on_device = device.type == "cuda" and total_bytes <= max_device_bytes
        if self.on_device:
            self.tensors = {k: t.to(device) for k, t in tensors.items()}
        elif device.type == "cuda" and pin_memory:
            self.tensors = {k: t.pin_memory() for k, t in tensors.items()}
        else:
            self.tensors = dict(tensors)
        logger.info(
            f"TensorBatchLoader: {self.num_samples} samples, {total_bytes / 2**30:.2f} GiB, "
            f"storage={'device' if self.on_device else 'host'}"
        )

    def __len__(self) -> int:
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self.shuffle:
            order = torch.randperm(self.num_samples, generator=self.generator)
            # RandomSampler parity: it also draws (and discards) a second permutation.
            torch.randperm(self.num_samples, generator=self.generator)
        else:
            order = torch.arange(self.num_samples)
        if self.on_device:
            order = order.to(self.device)
        for start in range(0, self.num_samples, self.batch_size):
            idx = order[start : start + self.batch_size]
            batch = {k: t.index_select(0, idx) for k, t in self.tensors.items()}
            if not self.on_device and self.device.type == "cuda":
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            yield batch


class HDMLTrainer:
    """High-Performance Trainer engine for Hierarchical Decision Mamba-Liquid models.

    Implements:
      - Automatic Mixed Precision (AMP: BFloat16 / FP16) with GradScaler
      - Batch-gather loader for pre-vectorised datasets (GPU-resident when it fits),
        multi-worker DataLoader otherwise
      - Linear Warmup + Cosine Annealing learning rate schedule
      - Gradient clipping and NaN/Inf safeguards
      - Model checkpointing (full resume state) and loss logging
      - Advantage-weighted action imitation and auxiliary critic/value/state losses
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
        if self.config.best_metric not in _VALID_BEST_METRICS:
            raise ValueError(f"best_metric must be one of {_VALID_BEST_METRICS}, got {self.config.best_metric!r}")
        # Deterministic seeding: torch/cuda/numpy/python. Shuffling uses an
        # explicit generator below so runs are reproducible.
        _seed = int(self.config.seed)
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        # GPU throughput knobs that do not change numerics: TF32 + high-precision
        # matmul + cudnn autotune for fixed input shapes.
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        self.model = model.to(self.device)
        if self.config.compile_cfc and self.model.cfc_filter is not None:
            # In-place compile keeps state_dict keys unchanged (checkpoint compatible).
            try:
                self.model.cfc_filter = torch.compile(self.model.cfc_filter)
                logger.info("torch.compile enabled for the CfC filter.")
            except Exception as e:
                logger.warning(f"Could not torch.compile CfC filter: {e}", exc_info=True)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        use_cuda = self.device.type == "cuda"
        self._loader_gen = torch.Generator().manual_seed(_seed)
        self.train_loader = self._build_loader(train_dataset, shuffle=True)
        self.val_loader = self._build_loader(val_dataset, shuffle=False) if val_dataset is not None else None

        # Optimizer. The action flow policy and the HiQC chunk critic are trained.
        self.optimizer = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
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

        # Learning Rate Schedule
        total_steps = max(1, len(self.train_loader) * self.config.max_epochs)
        self.warmup_steps = min(self.config.warmup_steps, total_steps // 4)
        self.total_steps = total_steps
        self.current_step = 0
        self.start_epoch = 0  # number of completed epochs (non-zero after resume)

        # Checkpoints setup
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")  # best value of config.best_metric

        # TensorBoard & WandB Web Tracking
        self.log_dir = Path(self.config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer: Any = None
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                logger.info(f"TensorBoard logging enabled. Log dir: {self.log_dir}")
            except ImportError:
                logger.warning("torch.utils.tensorboard not available. Disabling TensorBoard.")

        self.wandb_active = False
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
                logger.warning(f"Could not initialize WandB: {e}", exc_info=True)

    def _build_loader(
        self, dataset: Dataset[dict[str, torch.Tensor]], shuffle: bool
    ) -> TensorBatchLoader | DataLoader[dict[str, torch.Tensor]]:
        """Batch-gather loader for pre-vectorised datasets, DataLoader otherwise."""
        use_cuda = self.device.type == "cuda"
        pin_memory = self.config.pin_memory if use_cuda else False
        as_tensor_dict = getattr(dataset, "as_tensor_dict", None)
        if callable(as_tensor_dict):
            return TensorBatchLoader(
                tensors=as_tensor_dict(),
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                device=self.device,
                generator=self._loader_gen if shuffle else None,
                max_device_bytes=int(self.config.gpu_dataset_max_gb * 2**30),
                pin_memory=pin_memory,
            )
        num_workers = self.config.num_workers if use_cuda else 0
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0) and self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor if num_workers > 0 else None,
            generator=self._loader_gen if shuffle else None,
        )

    def _adjust_lr(self) -> float:
        """Calculate and set the learning rate using Warmup + Cosine Decay."""
        if self.current_step < self.warmup_steps and self.warmup_steps > 0:
            lr = self.config.learning_rate * (float(self.current_step + 1) / float(self.warmup_steps))
        else:
            progress = float(self.current_step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            # Clamp: past the planned horizon the LR must stay at the floor, not
            # climb back up the cosine.
            progress = min(1.0, progress)
            lr = self.config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = max(lr, self.config.learning_rate * 0.01)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the composite HDML loss for training and evaluation.

        Implements the report's objective:
          0. Action imitation on the executed (CfC-refined) action.
          1. Flow-Matching velocity loss over the action chunk (generative policy).
          2. HiQC chunk critic loss (c-step reward + discounted RTG[t+c] target).
          3. Value loss (IQL expectile regression toward the critic lower bound).
          4. PAVE regularisation of the state-action value Hessian (Hutchinson).
          5. Grad-CAPS second-order action-smoothness penalty.
          6. Forward-dynamics prediction loss.
          7. Subgoal representation loss.
        Terms 0 and 1 are weighted per token by the detached AWR advantage weight
        exp(norm(Q_min - V)/temperature) when ``use_advantage_weighting`` is set;
        this is the only path through which the critic shapes the policy.
        All weights come from the training config.

        Args:
            batch: dict with states, actions, rtgs, timesteps, mask,
                   target_actions, target_chunks, target_rtgs, reward_chunks,
                   next_states (and target_next_rtgs).

        Returns:
            Tuple of (scalar loss tensor, dict of detached scalar loss tensors).
            ``action_loss`` / ``flow_loss`` in the dict are the *unweighted*
            errors so they stay comparable across runs and with validation.
        """
        states = batch["states"].to(self.device, non_blocking=True)
        actions = batch["actions"].to(self.device, non_blocking=True)
        rtgs = batch["rtgs"].to(self.device, non_blocking=True)
        timesteps = batch["timesteps"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)

        target_actions = batch["target_actions"].to(self.device, non_blocking=True)
        target_rtgs = batch["target_rtgs"].to(self.device, non_blocking=True)
        next_states = batch["next_states"].to(self.device, non_blocking=True)
        target_chunks = batch["target_chunks"].to(self.device, non_blocking=True)
        reward_chunks = batch["reward_chunks"].to(self.device, non_blocking=True)
        # RTG at t+c for the k-step target. Older checkpoints/datasets may lack
        # this field; fall back to RTG[t] only to avoid a crash (biased target,
        # logged once). New datasets must provide "target_next_rtgs".
        if "target_next_rtgs" in batch:
            target_next_rtgs = batch["target_next_rtgs"].to(self.device, non_blocking=True)
        else:
            if self.current_step == 0:
                logger.warning("Batch lacks 'target_next_rtgs'; Q-target falls back to RTG[t] (biased). Regenerate dataset.")
            target_next_rtgs = target_rtgs

        valid_tokens = torch.clamp(mask.sum(), min=1.0)
        mask_exp = mask.unsqueeze(-1)  # (B, T, 1)

        # Backbone features, the nominal action chunk and the executed action.
        (
            subgoals_pred, _latent, values_pred, next_states_pred,
            flow_context, actions_pred, _hx, nominal_chunk,
        ) = self.model.encode_with_chunk(states=states, rtgs=rtgs, actions=actions, timesteps=timesteps)

        # 2. HiQC chunk critic loss toward the discounted c-step reward + RTG[t+c].
        #    k-step identity: RTG[t] = R_c[t] + gamma^c * RTG[t+c], so the target
        #    bootstraps from the dataset future return (Monte-Carlo, no target net).
        has_critic = getattr(self.model, "hiqc_critic", None) is not None
        if has_critic and (self.config.q_weight > 0.0 or self.config.use_advantage_weighting):
            critic_context = torch.cat([subgoals_pred.detach(), states], dim=-1)
            q1_pred, q2_pred = self.model.hiqc_critic(critic_context, target_chunks)
            q_target = reward_chunks + (self.config.gamma ** self.model.chunk_size) * target_next_rtgs
            q1_loss = F.mse_loss(q1_pred, q_target, reduction="none") * mask_exp
            q2_loss = F.mse_loss(q2_pred, q_target, reduction="none") * mask_exp
            q_loss = (q1_loss.sum() + q2_loss.sum()) / torch.clamp(mask_exp.sum() * 2, min=1.0)
            q_chunk_value = torch.minimum(q1_pred, q2_pred)

            # 3. Value loss: expectile regression toward the conservative chunk value.
            raw_val = self._expectile_loss(q_chunk_value.detach() - values_pred, tau=0.7)
            value_loss = (raw_val * mask_exp).sum() / torch.clamp(mask_exp.sum(), min=1.0)

            # Per-token policy weights (B, T, 1): AWR advantage weights (unit mean over
            # valid tokens, zero on padding) or the plain mask.
            if self.config.use_advantage_weighting:
                token_w = advantage_weights(q1_pred, q2_pred, values_pred, mask, self.config.advantage_temperature)
            else:
                token_w = mask_exp.float()
        else:
            q_loss = states.new_zeros(())
            value_loss = states.new_zeros(())
            token_w = mask_exp.float()

        # 0. Action imitation loss on the executed action (first chunk step, CfC-refined).
        raw_act = F.smooth_l1_loss(actions_pred, target_actions, reduction="none")
        act_denom = valid_tokens * actions_pred.shape[-1]
        action_loss = (raw_act * token_w).sum() / act_denom
        action_loss_plain = (raw_act * mask_exp).sum() / act_denom

        # 1. Flow-Matching loss over the action chunk (B, T, chunk, action_dim).
        #    The Gaussian head's forward_train output equals the deterministic
        #    chunk already computed in encode (same mu_net, same context), so it
        #    is reused instead of running mu_net twice.
        if isinstance(self.model.flow_policy, GaussianActionPolicy):
            target_velocity, pred_velocity, noise = target_chunks, nominal_chunk, None
        else:
            target_velocity, pred_velocity, noise = self.model.flow_policy.forward_train(
                target_chunks, flow_context
            )
        flow_denom = valid_tokens * target_velocity.shape[-1] * target_velocity.shape[-2]
        raw_flow = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        flow_loss = (raw_flow * token_w.unsqueeze(-1)).sum() / flow_denom
        flow_loss_plain = (raw_flow * mask_exp.unsqueeze(-1)).sum() / flow_denom
        # Predicted action chunk (deterministic reconstruction a_0 + v) for Grad-CAPS.
        pred_chunk = pred_velocity if noise is None else pred_velocity + noise

        # 4. PAVE regularisation (Hutchinson trace of the mixed Hessian).
        #    Double backward is expensive; compute it every ``pave_interval`` steps.
        #    Never computed under inference_mode (validation), so it is excluded
        #    from val_loss.
        pave_active = (
            has_critic
            and self.config.pave_weight > 0.0
            and torch.is_grad_enabled()
            and self.config.pave_interval > 0
            and (self.current_step % self.config.pave_interval == 0)
        )
        if pave_active:
            pave_loss = self._pave_loss(self.model.hiqc_critic, critic_context, target_chunks, mask)
        else:
            pave_loss = states.new_zeros(())

        # 5. Grad-CAPS second-order action smoothness *within* the predicted chunk.
        #    pred_chunk is (B, T, c, D); differentiate along the chunk axis c
        #    (dim=-2), not the context-time axis. Mask weights per context step.
        if self.config.grad_caps_weight > 0.0 and pred_chunk.shape[-2] > 2:
            accel = pred_chunk[:, :, 2:, :] - 2 * pred_chunk[:, :, 1:-1, :] + pred_chunk[:, :, :-2, :]
            accel_norm = (accel ** 2).sum(dim=-1).sum(dim=-1)  # (B, T)
            grad_caps_loss = (accel_norm * mask).sum() / valid_tokens
        else:
            grad_caps_loss = states.new_zeros(())

        # 6. Forward-dynamics prediction loss.
        raw_dyn = F.smooth_l1_loss(next_states_pred, next_states, reduction="none")
        dynamics_loss = (raw_dyn * mask_exp).sum() / (valid_tokens * next_states_pred.shape[-1])

        # 7. Subgoal representation loss against future states (when dims match).
        if subgoals_pred.shape[-1] == next_states.shape[-1]:
            raw_subgoal = F.smooth_l1_loss(subgoals_pred, next_states, reduction="none")
            subgoal_loss = (raw_subgoal * mask_exp).sum() / (valid_tokens * subgoals_pred.shape[-1])
        else:
            subgoal_loss = states.new_zeros(())

        loss = (
            self.config.action_weight * action_loss
            + self.config.flow_weight * flow_loss
            + self.config.q_weight * q_loss
            + self.config.reg_loss_weight * value_loss
            + self.config.pave_weight * pave_loss
            + self.config.grad_caps_weight * grad_caps_loss
            + self.config.dynamics_weight * dynamics_loss
            + self.config.subgoal_loss_weight * subgoal_loss
        )
        loss_dict = {
            "total_loss": loss.detach(),
            "action_loss": action_loss_plain.detach(),
            "flow_loss": flow_loss_plain.detach(),
            "q_loss": q_loss.detach(),
            "value_loss": value_loss.detach(),
            "grad_caps_loss": grad_caps_loss.detach(),
            "dynamics_loss": dynamics_loss.detach(),
            "subgoal_loss": subgoal_loss.detach(),
            # Padding weights are 0 and valid weights >= 0.1, so max() needs no masking.
            "adv_weight_max": token_w.max().detach(),
        }
        if pave_active:
            loss_dict["pave_loss"] = pave_loss.detach()
        return loss, loss_dict

    @staticmethod
    def _expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
        """Asymmetric expectile squared loss L2^tau(x) = |tau - 1[x<0]| x^2."""
        weight = torch.where(diff > 0, tau, 1.0 - tau)
        return weight * (diff ** 2)

    def _pave_loss(
        self,
        critic: nn.Module,
        state_repr: torch.Tensor,
        action_chunk: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """PAVE: Hutchinson estimate of ||grad_s (grad_a Q^T v)||^2."""
        s = state_repr.detach().requires_grad_(True)
        a = action_chunk.detach().requires_grad_(True)
        q1, _ = critic(s, a)
        q = q1.squeeze(-1)
        q_sum = (q * mask).sum()
        grad_a = torch.autograd.grad(
            q_sum, a, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        v = torch.randn_like(a)
        grad_s = torch.autograd.grad(
            (grad_a * v).sum(), s, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        penalty = (grad_s ** 2).sum(dim=-1)
        return (penalty * mask).sum() / torch.clamp(mask.sum(), min=1.0)

    @staticmethod
    def _accumulate(
        sums: dict[str, torch.Tensor], counts: dict[str, int], loss_dict: dict[str, torch.Tensor], weight: int
    ) -> None:
        """Accumulate detached GPU scalars without host sync (sample-weighted)."""
        for k, v in loss_dict.items():
            sums[k] = sums.get(k, torch.zeros((), device=v.device)) + v.float() * weight
            counts[k] = counts.get(k, 0) + weight

    @staticmethod
    def _means(sums: dict[str, torch.Tensor], counts: dict[str, int], prefix: str) -> dict[str, float]:
        """Sample-weighted means; keys without samples are omitted."""
        out: dict[str, float] = {}
        for k, v in sums.items():
            name = "loss" if k == "total_loss" else k
            out[f"{prefix}_{name}"] = float(v / max(1, counts[k]))
        return out

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Train the model for one epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Dictionary of averaged epoch training metrics (sample-weighted). The
            ``train_pave_loss`` average only covers steps where PAVE was computed.
        """
        self.model.train()
        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}

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
            self.optimizer.zero_grad(set_to_none=True)
            lr = self._adjust_lr()

            # Sequence forward pass with AMP
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                loss, loss_dict = self._compute_loss(batch)

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
            batch_size = int(batch["states"].shape[0])
            total_samples += batch_size
            self._accumulate(sums, counts, loss_dict, batch_size)

            # Throttle host-syncing postfix updates: .item() forces a CUDA sync.
            if (self.current_step % 10 == 0) or (self.current_step < 5):
                batch_bar.set_postfix(
                    loss=f"{loss_dict['total_loss'].item():.3f}",
                    act=f"{loss_dict['action_loss'].item():.3f}",
                    lr=f"{lr:.1e}",
                )

        t1 = time.perf_counter()
        metrics: dict[str, float] = {"epoch": float(epoch)}
        metrics.update(self._means(sums, counts, "train"))
        metrics["throughput_fps"] = float(total_samples / max(1e-5, (t1 - t0)))

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
                logger.warning(f"WandB logging error: {e}", exc_info=True)

        return metrics

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        """Evaluate the model on the validation dataset (sample-weighted means)."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}

        for batch in self.val_loader:
            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                _, loss_dict = self._compute_loss(batch)
            self._accumulate(sums, counts, loss_dict, int(batch["states"].shape[0]))

        val_metrics = self._means(sums, counts, "val")

        if self.writer is not None:
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"Val/{k}", v, self.current_step)

        return val_metrics

    def save_checkpoint(self, path: Path | str, epoch: int) -> Path:
        """Save weights plus the full state needed for an exact resume.

        Args:
            path: Target file.
            epoch: Number of completed epochs at save time.
        """
        save_file = Path(path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        rng_state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "loader": self._loader_gen.get_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()

        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "current_step": self.current_step,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_metric": self.config.best_metric,
            "config": self.config.__dict__,
            "state_mean": getattr(self.train_dataset, "state_mean", None),
            "state_std": getattr(self.train_dataset, "state_std", None),
            "rng_state": rng_state,
        }
        torch.save(state, save_file)
        return save_file

    def load_checkpoint(self, path: Path | str) -> None:
        """Restore model/optimizer/scaler/RNG state and the epoch counter.

        Checkpoints written before ``epoch`` was stored derive it from
        ``current_step // len(train_loader)``. ``best_loss`` is only restored
        when the checkpoint tracked the same ``best_metric``.
        """
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        else:
            logger.warning(f"{path}: no optimizer state; AdamW moments restart from zero.")
        if "scaler_state_dict" in ck:
            self.scaler.load_state_dict(ck["scaler_state_dict"])
        self.current_step = int(ck.get("current_step", 0))
        if "epoch" in ck:
            self.start_epoch = int(ck["epoch"])
        else:
            self.start_epoch = self.current_step // max(1, len(self.train_loader))
            logger.warning(f"{path}: no 'epoch' field; derived completed epochs = {self.start_epoch} from step count.")

        ck_metric = ck.get("best_metric", "val_loss")
        if ck_metric == self.config.best_metric:
            self.best_loss = float(ck.get("best_loss", float("inf")))
        else:
            logger.warning(
                f"Checkpoint tracked best '{ck_metric}', config uses '{self.config.best_metric}'; best value reset."
            )
            self.best_loss = float("inf")

        rng = ck.get("rng_state")
        if rng is not None:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            self._loader_gen.set_state(rng["loader"].cpu())
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        else:
            logger.warning(f"{path}: no RNG state; resumed run is not bit-reproducible.")
        logger.info(f"Resumed from {path}: epoch {self.start_epoch}, step {self.current_step}.")

    def train(self) -> list[dict[str, float]]:
        """Run the training loop from ``start_epoch + 1`` to ``max_epochs``."""
        amp_info = f" | AMP: {self.config.amp_dtype}" if self.use_amp else " | AMP: Disabled"
        first_epoch = self.start_epoch + 1
        if first_epoch > self.config.max_epochs:
            logger.warning(
                f"Checkpoint already completed {self.start_epoch} epochs >= max_epochs={self.config.max_epochs}; nothing to train."
            )
            return []
        logger.info(
            f"Starting HDML training epochs {first_epoch}..{self.config.max_epochs} on {self.device}{amp_info} "
            f"| advantage weighting: {self.config.use_advantage_weighting} | best metric: {self.config.best_metric}"
        )
        history: list[dict[str, float]] = []

        epoch_bar = tqdm(
            range(first_epoch, self.config.max_epochs + 1),
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
        )
        best_key = self.config.best_metric
        for epoch in epoch_bar:
            train_metrics = self.train_epoch(epoch)
            # Respect eval_interval: validation every epoch is 2x work for no gain.
            # On skipped epochs val_metrics={} and best-tracking is SKIPPED so
            # best_model.pt always reflects a validation optimum, never a
            # train-loss optimum.
            if self.val_loader is not None and (epoch % max(1, self.config.eval_interval) == 0 or epoch == self.config.max_epochs):
                val_metrics = self.evaluate()
            else:
                val_metrics = {}

            combined = {**train_metrics, **val_metrics}
            history.append(combined)

            if best_key in val_metrics and val_metrics[best_key] < self.best_loss:
                self.best_loss = val_metrics[best_key]
                # Persist the best model immediately, independent of the periodic interval.
                best_file = self.save_checkpoint(self.output_dir / "best_model.pt", epoch)
                logger.info(f"New best {best_key}={self.best_loss:.4f}; saved {best_file}")

            if val_metrics and self.wandb_active:
                try:
                    import wandb
                    wandb_val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                    wandb_val_metrics["val/epoch"] = epoch
                    wandb.log(wandb_val_metrics)
                except Exception as e:
                    logger.warning(f"WandB validation logging error: {e}", exc_info=True)

            val_str = (
                f" | Val: {val_metrics['val_loss']:.4f} (Act: {val_metrics['val_action_loss']:.4f})"
                if "val_loss" in val_metrics else ""
            )
            logger.info(
                f"Epoch [{epoch:02d}/{self.config.max_epochs:02d}] "
                f"Train: {train_metrics['train_loss']:.4f} (Act: {train_metrics['train_action_loss']:.4f}, "
                f"Q: {train_metrics['train_q_loss']:.4f}, AdvWmax: {train_metrics['train_adv_weight_max']:.2f})"
                f"{val_str} | Speed: {train_metrics['throughput_fps']:.1f} samples/s | Best {best_key}: {self.best_loss:.4f}"
            )
            epoch_bar.set_postfix(
                train=f"{train_metrics['train_loss']:.3f}",
                val=f"{val_metrics.get('val_loss', float('nan')):.3f}",
                best=f"{self.best_loss:.3f}",
            )

            # Periodic checkpointing
            if epoch % self.config.save_interval == 0 or epoch == self.config.max_epochs:
                self.save_checkpoint(self.output_dir / f"checkpoint_ep{epoch:03d}.pt", epoch)

        logger.info("Training complete.")
        return history
