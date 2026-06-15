import pytest
import torch
from hdml.models.liquid_head import CfCActionFilter

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_cfc_action_filter(device: torch.device):
    B, T, chunk_size, action_dim = 2, 8, 4, 6
    state_dim = 16
    
    filter = CfCActionFilter(
        action_dim=action_dim, 
        chunk_size=chunk_size, 
        state_dim=state_dim, 
        units=16
    ).to(device)
    
    # 4D input
    action_chunk = torch.randn(B, T, chunk_size, action_dim, device=device)
    state_repr = torch.randn(B, T, state_dim, device=device)
    
    filtered, next_hx = filter(action_chunk, state_repr)
    
    assert filtered.shape == (B, T, chunk_size, action_dim)
    assert next_hx is not None
    
    # 3D input
    action_chunk_3d = torch.randn(B, chunk_size, action_dim, device=device)
    state_repr_3d = torch.randn(B, state_dim, device=device)
    
    filtered_3d, next_hx_3d = filter(action_chunk_3d, state_repr_3d, hx=next_hx)
    assert filtered_3d.shape == (B, chunk_size, action_dim)
    
    # Test gradient flow
    loss = filtered.sum()
    loss.backward()
    
    for param in filter.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
