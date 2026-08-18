from __future__ import annotations

import torch

class PACEController:
    """
    Phase-Aware Chunk Execution (PACE) Controller.
    
    Dynamically truncates action chunk execution if the physical state 
    deviates significantly from the predicted state (Sim-to-Real gap).
    """
    def __init__(self, threshold: float = 0.5, norm_p: int = 2):
        self.threshold = threshold
        self.norm_p = norm_p
        
        # Internal state
        self.active_chunk = None
        self.predicted_states = None
        self.step_idx = 0
        
    def reset(self):
        """Reset the active chunk and state predictions."""
        self.active_chunk = None
        self.predicted_states = None
        self.step_idx = 0
        
    def set_new_plan(self, action_chunk: torch.Tensor, predicted_states: torch.Tensor | None = None):
        """
        Set a new action chunk and predicted future states.
        
        Args:
            action_chunk: (chunk_size, action_dim)
            predicted_states: (chunk_size, state_dim) or None
        """
        self.active_chunk = action_chunk
        self.predicted_states = predicted_states
        self.step_idx = 0
        
    def get_next_action(self, current_state: torch.Tensor) -> tuple[torch.Tensor | None, bool]:
        """
        Retrieve the next action in the chunk, monitoring deviation.
        
        Args:
            current_state: The actual observed state, shape (state_dim)
            
        Returns:
            action: (action_dim) or None if truncation occurred
            needs_replanning: True if chunk is empty or deviation threshold exceeded
        """
        if self.active_chunk is None or self.step_idx >= self.active_chunk.shape[0]:
            # Chunk is fully executed or none exists
            return None, True
            
        # Check deviation if we have predicted states
        if self.predicted_states is not None and self.step_idx < self.predicted_states.shape[0]:
            # We skip deviation check on the very first step of the chunk (step_idx == 0)
            # because the state prediction usually applies to t+1, t+2, etc.
            if self.step_idx > 0:
                expected_state = self.predicted_states[self.step_idx - 1]
                deviation = torch.norm(current_state - expected_state, p=self.norm_p)
                
                if deviation > self.threshold:
                    # Deviation too large, trigger replanning
                    self.reset()
                    return None, True
                    
        # Extract action
        action = self.active_chunk[self.step_idx]
        self.step_idx += 1
        
        # Check if we just finished the chunk
        needs_replanning = (self.step_idx >= self.active_chunk.shape[0])
        
        return action, needs_replanning
