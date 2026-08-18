import pytest
import torch
from hdml.models.flow_policy import FlowPolicy

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_flow_policy_train(device: torch.device):
    B, T, chunk_size, action_dim = 2, 8, 4, 8
    context_dim = 64
    
    policy = FlowPolicy(action_dim=action_dim, chunk_size=chunk_size, context_dim=context_dim, hidden_dim=64).to(device)
    
    a_1 = torch.randn(B, T, chunk_size, action_dim, device=device)
    context = torch.randn(B, T, context_dim, device=device)
    
    target_v, pred_v, noise = policy.forward_train(a_1, context)
    
    assert target_v.shape == (B, T, chunk_size, action_dim)
    assert pred_v.shape == (B, T, chunk_size, action_dim)
    assert noise.shape == (B, T, chunk_size, action_dim)
    
    loss = torch.nn.functional.mse_loss(pred_v, target_v)
    loss.backward()
    
    for param in policy.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
        
def test_flow_policy_sample(device: torch.device):
    B, chunk_size, action_dim = 4, 4, 8
    context_dim = 64
    
    policy = FlowPolicy(action_dim=action_dim, chunk_size=chunk_size, context_dim=context_dim, hidden_dim=64).to(device)
    
    context = torch.randn(B, context_dim, device=device)
    
    sampled_actions = policy.sample(context, num_steps=3)
    
    assert sampled_actions.shape == (B, chunk_size, action_dim)
    
