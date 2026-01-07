import torch
from torch import nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Union, Any

from networks.architectures import BaseArchitecture

class BaseAgent(ABC):

    def __init__(self, architecture: BaseArchitecture):
        super().__init__()
        self.architecture = architecture


    @property
    def device(self) -> torch.device:
        try:
            return next(self.architecture.parameters()).device
        except StopIteration:
            return torch.device("cpu")


    def _state_to_tensor(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
    
        if obs.device != self.device:
            obs = obs.to(self.device)
    
        return obs
    

    @abstractmethod
    def get_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


    @abstractmethod
    def evaluate_actions(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass