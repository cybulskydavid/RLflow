from typing import Generator, Tuple

import numpy as np
import torch
from buffers.base_buffer import BaseBuffer


class RolloutBuffer(BaseBuffer):
    def __init__(self, 
                 buffer_size: int, 
                 obs_shape: Tuple[int,...], 
                 act_shape: Tuple[int,...],
                 device: str = "cpu",
                 gamma: float = 0.95,
                 gae_lambda: float = 0.95):
        super().__init__(buffer_size, obs_shape, act_shape, device)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.log_probs = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.values = torch.zeros((buffer_size, 1), dtype=torch.float32)
        
        self.advantages = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.returns = torch.zeros((buffer_size, 1), dtype=torch.float32)


    def add(self, 
            obs: np.ndarray, 
            action :np.ndarray, 
            reward: float, 
            done: bool, 
            log_prob: float, 
            value: float) -> None:
        
        if self.full:
            return
        else:
            self.log_probs[self.pos] = torch.tensor(log_prob)
            self.values[self.pos] = torch.tensor(value)

            super().add(obs, action, reward, done)


    def compute_gae(self, last_value: float, next_done: bool):
        
        last_value = torch.tensor(last_value)
        last_gae_lam = 0
        
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - float(next_done)
                next_val = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_val = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_val * next_non_terminal - self.values[step]
            
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
            self.advantages[step] = last_gae_lam
            
        self.returns = self.advantages + self.values


    def get_generator(self, batch_size: int) -> Generator:
        indices = np.random.permutation(self.buffer_size)
        
        for start in range(0, self.buffer_size, batch_size):
            end = start + batch_size
            batch_inds = indices[start:end]
            
            yield (
                self.observations[batch_inds].to(self.device),
                self.actions[batch_inds].to(self.device),
                self.log_probs[batch_inds].to(self.device),
                self.advantages[batch_inds].to(self.device),
                self.returns[batch_inds].to(self.device),
                self.values[batch_inds].to(self.device)
            )

        self.reset()


    def reset(self):
        self.pos = 0
        self.full = False