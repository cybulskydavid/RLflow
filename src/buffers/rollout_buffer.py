from typing import Generator, Tuple

import numpy as np
import torch
from buffers.base_buffer import BaseBuffer


class RolloutBuffer(BaseBuffer):
    def __init__(self, 
                 buffer_size: int, 
                 state_shape: Tuple[int,...], 
                 action_shape: Tuple[int,...],
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        super().__init__(buffer_size, state_shape, action_shape)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.log_probs = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.values = torch.zeros((buffer_size, 1), dtype=torch.float32)
        
        self.advantages = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.returns = torch.zeros((buffer_size, 1), dtype=torch.float32)   


    def add(self, 
            state: np.ndarray, 
            action :np.ndarray, 
            reward: float, 
            done: bool, 
            log_prob: float, 
            value: float) -> None:
        
        if self.is_full:
            raise RuntimeError
        
        self.log_probs[self.position] = log_prob
        self.values[self.position] = value
        super().add(state, action, reward, done)


    def compute_gae(self, last_value: torch.Tensor, next_done: bool):
        last_gae_lam = 0
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - float(next_done)
                next_val = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_val = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_val * next_non_terminal - self.values[step]
            
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
            self.advantages[step] = last_gae_lam
            
        self.returns = self.advantages + self.values


    def get_generator(self, batch_size: int) -> Generator:
        def reset():
            self.position = 0
            self.is_full = False

        indices = np.random.permutation(self.buffer_size)

        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            batch_inds = indices[start:end]
            
            yield (
                self.states[batch_inds],
                self.actions[batch_inds],
                self.log_probs[batch_inds],
                self.advantages[batch_inds],
                self.returns[batch_inds],
                self.values[batch_inds]
            )

        reset()

