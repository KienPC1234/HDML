from __future__ import annotations

import torch
import numpy as np
import pytest

from hdml.evaluation.unitree_a1_maze_env import UnitreeA1MazeEnv
from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import HDMLConfig


def test_unitree_a1_maze_env_contract() -> None:
    """Verify Unitree A1 Maze environment shape contracts and sensory signals."""
    env = UnitreeA1MazeEnv(max_episode_steps=50)
    obs, info = env.reset(seed=42)

    assert obs.shape == (53,), f"Expected 53D observation, got {obs.shape}"
    assert "pos" in info
    assert "goal" in info

    # Check 16D LiDAR distance ranges (index 35 to 51)
    lidar = obs[35:51]
    assert len(lidar) == 16
    assert np.all(lidar >= 0.05) and np.all(lidar <= 4.0), f"LiDAR out of valid range: {lidar}"

    # Check 2D Relative Goal Vector (index 51 to 53)
    rel_goal = obs[51:53]
    assert len(rel_goal) == 2

    # Step test
    dummy_action = np.zeros(12, dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(dummy_action)

    assert next_obs.shape == (53,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "dist_to_goal" in step_info
    assert "goal_reached" in step_info

    env.close()


def test_hdml_model_53d_forward_backward() -> None:
    """Verify HDML forward and backward passes with 53D observation and 12D action on CUDA."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = HDMLConfig.from_yaml("configs/unitree_a1_maze_unsupervised.yaml")

    model = HDMLModel.from_config(cfg.model).to(device)

    batch_size = 4
    seq_len = 10
    states = torch.randn(batch_size, seq_len, 53, device=device)
    rtgs = torch.ones(batch_size, seq_len, 1, device=device)
    actions = torch.randn(batch_size, seq_len, 12, device=device)
    timesteps = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

    # Sequence forward pass
    actions_pred, subgoals_pred, values_pred, next_states_pred, _ = model(
        states=states,
        rtgs=rtgs,
        actions=actions,
        timesteps=timesteps,
    )

    assert actions_pred.shape == (batch_size, seq_len, 12)
    assert subgoals_pred.shape == (batch_size, seq_len, 53)
    assert values_pred.shape == (batch_size, seq_len, 1)
    assert next_states_pred.shape == (batch_size, seq_len, 53)

    # Backward pass & gradient verification
    loss = (
        actions_pred.sum()
        + subgoals_pred.sum()
        + values_pred.sum()
        + next_states_pred.sum()
    )
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad and not any(k in name for k in ["flow_policy", "hiqc_critic", "value_net"]):
            assert p.grad is not None, f"Parameter {name} has None gradient"
            assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient"

    # Step inference verification
    with torch.inference_mode():
        act, hx, extra = model.get_action(states, rtgs, actions, timesteps)
        assert act.shape == (batch_size, 12)
        assert "subgoal" in extra
