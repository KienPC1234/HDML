import pytest
import torch
import torch.nn as nn
from hdml.training.losses import HDMLLoss
from hdml.models.hiqc_critic import HiQCCritic

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_hdml_v2_loss(device: torch.device):
    B, T, chunk_size, action_dim = 2, 8, 4, 8
    state_dim = 128
    
    loss_fn = HDMLLoss(
        flow_weight=1.0,
        q_weight=1.0,
        pave_weight=0.1,
        grad_caps_weight=0.1,
        dynamics_weight=0.1
    ).to(device)
    
    critic = HiQCCritic(state_dim=state_dim, action_dim=action_dim, chunk_size=chunk_size, hidden_dim=64).to(device)
    
    target_velocity = torch.randn(B, T, chunk_size, action_dim, device=device)
    pred_velocity = torch.randn(B, T, chunk_size, action_dim, device=device, requires_grad=True)
    noise = torch.randn(B, T, chunk_size, action_dim, device=device)
    
    q1_pred = torch.randn(B, T, 1, device=device, requires_grad=True)
    q2_pred = torch.randn(B, T, 1, device=device, requires_grad=True)
    q_target = torch.randn(B, T, 1, device=device)
    
    values_pred = torch.randn(B, T, 1, device=device, requires_grad=True)
    
    state_repr = torch.randn(B, T, state_dim, device=device, requires_grad=True)
    action_chunk = torch.randn(B, T, chunk_size, action_dim, device=device, requires_grad=True)
    
    next_states_pred = torch.randn(B, T, state_dim, device=device, requires_grad=True)
    target_states = torch.randn(B, T, state_dim, device=device)
    
    mask = torch.ones(B, T, device=device)
    
    total_loss, loss_dict = loss_fn(
        target_velocity=target_velocity,
        pred_velocity=pred_velocity,
        noise=noise,
        q1_pred=q1_pred,
        q2_pred=q2_pred,
        q_target=q_target,
        values_pred=values_pred,
        state_repr=state_repr,
        action_chunk=action_chunk,
        critic=critic,
        next_states_pred=next_states_pred,
        target_states=target_states,
        mask=mask
    )
    
    assert total_loss.ndim == 0
    assert total_loss.item() > 0
    
    # Test backward pass for Flow and Q losses (we don't want PAVE to backprop to backbone during training)
    # Actually PAVE is computed with detaching inside compute_pave_loss, so it shouldn't affect state_repr's grad.
    total_loss.backward()
    
    assert pred_velocity.grad is not None
    assert q1_pred.grad is not None
    assert q2_pred.grad is not None
    assert values_pred.grad is not None
    assert next_states_pred.grad is not None
    
    # Grad-CAPS now regularizes the reconstructed predicted chunk (pred_velocity + noise),
    # so it no longer touches the ground-truth action_chunk. PAVE detaches its inputs.
    assert action_chunk.grad is None
    
    # Verify that the critic gets gradients from PAVE loss
    critic_has_grad = False
    for param in critic.parameters():
        if param.grad is not None and not torch.all(param.grad == 0):
            critic_has_grad = True
            break
    assert critic_has_grad, "Critic should receive gradients from PAVE loss"

