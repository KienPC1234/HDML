import numpy as np
import pytest
import torch
from hdml.data.dataset import FastTensorTrajectoryDataset

def test_fast_tensor_trajectory_dataset_chunking():
    # Create dummy trajectories
    T = 50
    prop_dim = 5
    action_dim = 2
    
    # 2 trajectories
    trajectories = [
        {
            "observations": np.random.randn(T, prop_dim).astype(np.float32),
            "actions": np.arange(1, T * action_dim + 1).reshape(T, action_dim).astype(np.float32),
            "returns_to_go": np.linspace(100, 0, T).astype(np.float32),
            "timesteps": np.arange(T).astype(np.int64)
        },
        {
            "observations": np.random.randn(T, prop_dim).astype(np.float32),
            "actions": np.arange(1, T * action_dim + 1).reshape(T, action_dim).astype(np.float32),
            "returns_to_go": np.linspace(100, 0, T).astype(np.float32),
            "timesteps": np.arange(T).astype(np.int64)
        }
    ]
    
    context_length = 5
    chunk_size = 4
    
    dataset = FastTensorTrajectoryDataset(
        trajectories=trajectories,
        context_length=context_length,
        chunk_size=chunk_size,
    )
    
    # Check length
    assert len(dataset) == T * 2
    
    # Check shape of items
    item = dataset[0] # first step of first trajectory
    
    assert item["target_chunks"].shape == (context_length, chunk_size, action_dim)
    
    # The first step in the window is t=0
    # For t=0, the target chunk should be actions from t=0 to t=3
    first_target_chunk = item["target_chunks"][0]
    
    expected_chunk = trajectories[0]["actions"][0:4]
    np.testing.assert_allclose(first_target_chunk.numpy(), expected_chunk)
    
    # Check the second step in the window (t=1)
    # Note: mask is applied during training, but the tensor contains padding or zeros if beyond sequence
    # For window starting at t=0, the sequence is right-padded
    # So item["target_chunks"][10] is a window starting at t=10 and ending at t=10+context_length-1 = 14.
    # Let's look at index=10
    
    item_10 = dataset[10]
    # The last element in the context window (index context_length - 1 = 4) corresponds to t=14
    t14_chunk = item_10["target_chunks"][4]
    
    expected_chunk_t14 = trajectories[0]["actions"][14:18]
    np.testing.assert_allclose(t14_chunk.numpy(), expected_chunk_t14)
