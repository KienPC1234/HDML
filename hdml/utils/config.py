from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class ModelConfig:
    """Configuration parameters for the HDML model architecture."""
    prop_dim: int = 27
    action_dim: int = 8
    chunk_size: int = 4  # HDML-V2 Action Chunking size
    d_model: int = 128
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    num_mamba_layers: int = 3
    d_subgoal: int = 64
    cfc_units: int = 32
    cfc_backbone_units: int = 64
    cfc_backbone_layers: int = 1
    cfc_residual: float = 0.5
    use_visual: bool = False
    visual_channels: int = 1
    visual_image_size: int = 64
    dropout: float = 0.1
    action_policy: str = "gaussian"  # "gaussian" (BC regressor) or "flow" (flow matching)
    device: str = "cuda"


@dataclass
class TrainingConfig:
    """Configuration parameters for HDML offline RL & algorithm distillation training."""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_epochs: int = 20
    batch_size: int = 64
    context_length: int = 20
    gamma: float = 0.99
    grad_clip_norm: float = 1.0
    flow_weight: float = 1.0
    q_weight: float = 1.0
    value_weight: float = 1.0
    pave_weight: float = 0.1
    grad_caps_weight: float = 0.05
    dynamics_weight: float = 0.1
    reg_loss_weight: float = 0.01
    subgoal_loss_weight: float = 0.1
    use_advantage_weighting: bool = True
    advantage_temperature: float = 1.0
    cfc_residual: float = 0.5
    eval_interval: int = 5
    save_interval: int = 5
    seed: int = 42
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "bfloat16" or "float16"
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    use_fast_dataset: bool = True
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "hdml-robotics"
    wandb_run_name: str | None = None
    output_dir: str = "checkpoints/ant_v4"
    log_dir: str = "logs/ant_v4"


@dataclass
class EnvConfig:
    """Configuration parameters for the simulation environment."""
    env_name: str = "Ant-v4"
    max_episode_steps: int = 1000
    target_return: float = 5000.0
    scale_return: float = 1000.0
    num_eval_episodes: int = 10
    render_mode: str | None = None
    seed: int = 42


@dataclass
class HDMLConfig:
    """Master configuration class bundling Model, Training, and Environment parameters."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    env: EnvConfig = field(default_factory=EnvConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> HDMLConfig:
        """Load configuration from a YAML file.

        Args:
            yaml_path: Absolute or relative path to the YAML file.

        Returns:
            HDMLConfig instance populated with YAML data.
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {path}")

        with open(path, mode="r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        model_cfg = ModelConfig(**data.get("model", {}))
        training_cfg = TrainingConfig(**data.get("training", {}))
        env_cfg = EnvConfig(**data.get("env", {}))

        return cls(model=model_cfg, training=training_cfg, env=env_cfg)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to a YAML file.

        Args:
            yaml_path: Target path to save the YAML file.
        """
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "env": self.env.__dict__,
        }

        with open(path, mode="w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
