import pytest
import torch
from hdml.models.hiqc_critic import HiQCCritic

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_hiqc_critic_forward(device: torch.device):
    B, T, chunk_size, action_dim = 2, 8, 4, 8
    state_dim = 128
    
    critic = HiQCCritic(state_dim=state_dim, action_dim=action_dim, chunk_size=chunk_size, hidden_dim=64).to(device)
    
    # Test 4D action chunk (Batch, Time, chunk, action_dim)
    state_repr = torch.randn(B, T, state_dim, device=device)
    action_chunk = torch.randn(B, T, chunk_size, action_dim, device=device)
    
    q1, q2 = critic(state_repr, action_chunk)
    
    assert q1.shape == (B, T, 1)
    assert q2.shape == (B, T, 1)
    
    # Test backward pass
    loss = q1.mean() + q2.mean()
    loss.backward()
    
    for param in critic.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
        
def test_hiqc_critic_forward_3d(device: torch.device):
    B, chunk_size, action_dim = 4, 4, 8
    state_dim = 128
    
    critic = HiQCCritic(state_dim=state_dim, action_dim=action_dim, chunk_size=chunk_size, hidden_dim=64).to(device)
    
    # Test 3D action chunk (Batch, chunk, action_dim)
    state_repr = torch.randn(B, state_dim, device=device)
    action_chunk = torch.randn(B, chunk_size, action_dim, device=device)
    
    q1, q2 = critic(state_repr, action_chunk)
    
    assert q1.shape == (B, 1)
    assert q2.shape == (B, 1)
