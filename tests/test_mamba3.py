import pytest
import torch
from hdml.models.mamba3_backbone import apply_rope, Mamba3Block, Mamba3CognitiveBackbone

def test_apply_rope():
    # Test tensor of shape (Batch, Seq_Len, D)
    B, T, D = 2, 10, 64
    x = torch.randn(B, T, D)
    
    # Apply RoPE
    x_rope = apply_rope(x, dim=-1)
    
    # Check shape
    assert x_rope.shape == (B, T, D)
    
    # Output should not be identical to input if positions > 0
    assert not torch.allclose(x, x_rope)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba requires CUDA")
def test_mamba3_block():
    B, T, D = 2, 20, 128
    device = torch.device("cuda")
    
    block = Mamba3Block(d_model=D).to(device)
    x = torch.randn(B, T, D, device=device)
    
    # Forward pass
    out = block(x)
    
    # Check shape
    assert out.shape == (B, T, D)
    
    # Check gradients
    out.sum().backward()
    
    # Check if parameters have gradients
    has_grad = any(p.grad is not None for p in block.parameters())
    assert has_grad, "Mamba3Block parameters did not receive gradients"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba requires CUDA")
def test_mamba3_backbone():
    B, T, D = 2, 20, 128
    device = torch.device("cuda")
    
    backbone = Mamba3CognitiveBackbone(
        d_model=D,
        d_state=16,
        d_conv=4,
        expand=2,
        num_layers=2,
        d_subgoal=64,
        prop_dim=27
    ).to(device)
    
    u_t = torch.randn(B, T, D, device=device)
    
    subgoals, latent_features, values, next_states_pred = backbone(u_t)
    
    assert subgoals.shape == (B, T, 64)
    assert latent_features.shape == (B, T, D)
    assert values.shape == (B, T, 1)
    assert next_states_pred.shape == (B, T, 27)
    
    # Test backward pass
    loss = subgoals.sum() + values.sum() + next_states_pred.sum()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in backbone.parameters())
    assert has_grad, "Mamba3CognitiveBackbone parameters did not receive gradients"
