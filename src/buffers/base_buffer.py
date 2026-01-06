from abc import ABC, abstractmethod
from typing import Any, Generator, Tuple

import numpy as np
import torch


class BaseBuffer(ABC):
    def __init__(self, 
                 buffer_size: int, 
                 state_shape: Tuple[int,...], 
                 action_shape: Tuple[int,...], 
                 device: str = "cpu"):
        super().__init__()
        self.buffer_size = buffer_size
        self.device = torch.device(device)
        self.position = 0
        self.is_full = False

        self.state_shape = state_shape
        self.action_shape = action_shape

        self.states = torch.zeros((buffer_size, *state_shape), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, *action_shape), dtype=torch.float32)
        
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32)

    def add(self, 
            state: np.ndarray, 
            action: np.ndarray, 
            reward: float, 
            done: bool) -> None:
        
        assert state.shape == self.state_shape
        assert action.shape == self.action_shape
        assert self.is_full is False
        
        self.states[self.position] = torch.as_tensor(state).float()
        self.actions[self.position] = torch.as_tensor(action).float()
        self.rewards[self.position] = torch.as_tensor(reward).float()
        self.dones[self.position] = torch.as_tensor(done).float()

        self.position += 1
        if self.position == self.buffer_size:
            self.is_full = True

    @abstractmethod
    def get_generator(self, batch_size: int) -> Generator:
        pass