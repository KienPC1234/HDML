"""Schema validation for HDML-Foundation YAML configs.

The foundation trainer uses a different config schema from ``HDMLConfig``:
it adds ``max_timesteps``/``max_embodiments`` to the model section, extra loss
weights to training, and an ``embodiments`` mapping. This module validates the
schema explicitly instead of letting a typo surface later as a ``KeyError``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

MODEL_REQUIRED: frozenset[str] = frozenset({
    "d_model",
    "num_mamba_layers",
    "d_state",
    "d_conv",
    "expand",
    "cfc_units",
    "cfc_backbone_units",
    "cfc_residual",
    "max_timesteps",
    "max_embodiments",
    "dropout",
    "device",
})
MODEL_OPTIONAL: frozenset[str] = frozenset({"use_native_mamba3"})

TRAINING_REQUIRED: frozenset[str] = frozenset({
    "learning_rate",
    "weight_decay",
    "warmup_steps",
    "max_epochs",
    "batch_size",
    "context_length",
    "gamma",
    "grad_clip_norm",
    "eval_interval",
    "save_interval",
    "seed",
    "use_amp",
    "amp_dtype",
    "num_workers",
    "pin_memory",
    "output_dir",
    "log_dir",
    "action_weight",
    "rtg_weight",
    "intent_weight",
})
TRAINING_OPTIONAL: frozenset[str] = frozenset({
    "scale_return",
    "sampling",
    "lr_schedule",
    "early_stopping_patience",
})

EMBODIMENT_REQUIRED: frozenset[str] = frozenset({"prop_dim", "action_dim", "path"})


def _check_section(
    name: str,
    section: dict[str, Any],
    required: frozenset[str],
    optional: frozenset[str],
) -> None:
    missing = required - section.keys()
    if missing:
        raise ValueError(f"Foundation config section '{name}' is missing keys: {sorted(missing)}")
    unknown = section.keys() - required - optional
    if unknown:
        raise ValueError(f"Foundation config section '{name}' has unknown keys: {sorted(unknown)}")


def load_foundation_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a HDML-Foundation YAML config.

    Args:
        config_path: Path to the foundation YAML file.

    Returns:
        The validated config as a plain dictionary, ready for the trainer.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If a required section/key is missing or an unknown key
            would otherwise be silently ignored.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Foundation config not found at: {path}")

    with open(path, mode="r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Foundation config must be a mapping, got {type(cfg).__name__}")

    unknown_sections = cfg.keys() - {"model", "training", "embodiments"}
    if unknown_sections:
        raise ValueError(f"Unknown top-level config sections: {sorted(unknown_sections)}")

    _check_section("model", cfg.get("model", {}), MODEL_REQUIRED, MODEL_OPTIONAL)
    _check_section("training", cfg.get("training", {}), TRAINING_REQUIRED, TRAINING_OPTIONAL)

    embodiments = cfg.get("embodiments", {})
    if not isinstance(embodiments, dict) or not embodiments:
        raise ValueError("Foundation config must define a non-empty 'embodiments' mapping")
    for name, spec in embodiments.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Embodiment '{name}' must be a mapping")
        missing = EMBODIMENT_REQUIRED - spec.keys()
        if missing:
            raise ValueError(f"Embodiment '{name}' is missing keys: {sorted(missing)}")
        unknown = spec.keys() - EMBODIMENT_REQUIRED
        if unknown:
            raise ValueError(f"Embodiment '{name}' has unknown keys: {sorted(unknown)}")

    return cfg
