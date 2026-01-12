from typing import Tuple
import numpy as np
import torch
from buffers.base_buffer import BaseBuffer


class ReplayBuffer(BaseBuffer):
    def __init__(self, 
                 buffer_size: int, 
                 state_shape: Tuple[int,...], 
                 action_shape: Tuple[int,...],
                 device: str = "cpu"):
        super().__init__(buffer_size, state_shape, action_shape)

        self.device = device
        self.next_states = torch.zeros((buffer_size, *state_shape), dtype=torch.float32)
        self.size = 0 


    def add(self, 
            state: np.ndarray, 
            action: np.ndarray, 
            reward: float, 
            done: bool,
            next_state: np.ndarray) -> None:
        
        self.states[self.position] = torch.as_tensor(state).float()
        self.actions[self.position] = torch.as_tensor(action).float()
        self.rewards[self.position] = torch.as_tensor(reward).float()
        self.dones[self.position] = torch.as_tensor(done).float()
        self.next_states[self.position] = torch.as_tensor(next_state).float()

        self.position = (self.position + 1) % self.buffer_size
        
        self.size = min(self.size + 1, self.buffer_size)
        self.is_full = (self.size == self.buffer_size)


    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )