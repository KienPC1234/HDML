from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "egl"

import pytest
from pathlib import Path
from hdml.models import HDMLModel
from hdml.utils.config import HDMLConfig
from scripts.record_video import record_episode_video


def test_record_halfcheetah_video() -> None:
    cfg = HDMLConfig.from_yaml("configs/halfcheetah_v4_default.yaml")
    model = HDMLModel.from_config(cfg.model)
    
    ckpt_path = "checkpoints/halfcheetah_v4/best_model.pt"
    out_video = "videos/hdml_halfcheetah_v4_rollout.gif"
    
    res = record_episode_video(
        model=model,
        env_name="HalfCheetah-v4",
        checkpoint_path=ckpt_path if Path(ckpt_path).exists() else None,
        output_path=out_video,
        max_steps=500,
        macro_interval=5,
        device="cuda",
    )
    
    assert Path(out_video).exists()
    assert Path(out_video).stat().st_size > 50000
    assert res["frames_count"] > 0
    print(f"Recorded HalfCheetah video: {res}")
