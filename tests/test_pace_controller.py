import pytest
import torch
from hdml.evaluation.pace_controller import PACEController

def test_pace_controller_basic():
    pace = PACEController(threshold=0.5)
    
    # 4-step action chunk
    action_chunk = torch.tensor([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0]
    ])
    
    pace.set_new_plan(action_chunk)
    
    act1, replan = pace.get_next_action(torch.zeros(2))
    assert torch.allclose(act1, torch.tensor([1.0, 1.0]))
    assert not replan
    
    act2, replan = pace.get_next_action(torch.zeros(2))
    assert torch.allclose(act2, torch.tensor([2.0, 2.0]))
    assert not replan
    
    act3, replan = pace.get_next_action(torch.zeros(2))
    assert torch.allclose(act3, torch.tensor([3.0, 3.0]))
    assert not replan
    
    act4, replan = pace.get_next_action(torch.zeros(2))
    assert torch.allclose(act4, torch.tensor([4.0, 4.0]))
    assert replan  # Chunk is finished, need replanning
    
    act5, replan = pace.get_next_action(torch.zeros(2))
    assert act5 is None
    assert replan

def test_pace_controller_truncation():
    pace = PACEController(threshold=1.0)
    
    action_chunk = torch.tensor([
        [1.0],
        [2.0],
        [3.0]
    ])
    predicted_states = torch.tensor([
        [10.0],  # expected at step 1
        [11.0],  # expected at step 2
        [12.0]   # expected at step 3
    ])
    
    pace.set_new_plan(action_chunk, predicted_states)
    
    # Step 0: no deviation check
    act0, replan = pace.get_next_action(torch.tensor([0.0]))
    assert torch.allclose(act0, torch.tensor([1.0]))
    
    # Step 1: observed state is close to predicted_states[0] (10.0)
    act1, replan = pace.get_next_action(torch.tensor([10.5])) # Deviation 0.5 <= 1.0
    assert torch.allclose(act1, torch.tensor([2.0]))
    assert not replan
    
    # Step 2: observed state deviates significantly from predicted_states[1] (11.0)
    act2, replan = pace.get_next_action(torch.tensor([5.0])) # Deviation 6.0 > 1.0
    assert act2 is None
    assert replan
