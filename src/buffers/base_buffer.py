from abc import ABC, abstractmethod
from typing import Any, Generator, Tuple

import numpy as np
import torch


class BaseBuffer(ABC):
    def __init__(self, 
                 buffer_size: int, 
                 obs_shape: Tuple[int,...], 
                 act_shape: Tuple[int,...], 
                 device: str = "cpu"):
        super().__init__()
        self.buffer_size = buffer_size
        self.device = torch.device(device)
        self.pos = 0
        self.full = False

        self.obs_shape = obs_shape
        self.act_shape = act_shape

        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, *act_shape), dtype=torch.float32)
        
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32)

    def add(self, 
            obs: np.ndarray, 
            action: np.ndarray, 
            reward: float, 
            done: bool) -> None:
        
        assert obs.shape == self.obs_shape
        assert action.shape == self.act_shape
        assert self.full is False
        
        self.observations[self.pos] = torch.as_tensor(obs).float()
        self.actions[self.pos] = torch.as_tensor(action).float()
        self.rewards[self.pos] = torch.as_tensor(reward).float()
        self.dones[self.pos] = torch.as_tensor(done).float()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    @abstractmethod
    def get_generator(self, batch_size: int) -> Generator:
        pass