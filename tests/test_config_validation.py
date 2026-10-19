"""Schema validation tests for HDML-Foundation configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hdml.utils.foundation_config import load_foundation_config


def test_all_foundation_configs_load() -> None:
    paths = sorted(Path("configs").glob("hdml_foundation*.yaml"))
    assert paths, "No foundation configs found"
    for path in paths:
        config = load_foundation_config(path)
        assert config["embodiments"]


def test_unknown_training_key_is_rejected(tmp_path: Path) -> None:
    source = Path("configs/hdml_foundation_full.yaml")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["training"]["rtg_wieght"] = 0.1  # deliberate typo
    bad = tmp_path / "bad_foundation.yaml"
    bad.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown keys"):
        load_foundation_config(bad)
