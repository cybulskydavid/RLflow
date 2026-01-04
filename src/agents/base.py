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


    def _obs_to_tensor(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
    
        if obs.device != self.device:
            obs = obs.to(self.device)
    
        return obs
    

    @abstractmethod
    def get_action(
        self, 
        obs: Union[np.ndarray, torch.Tensor], 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


    @abstractmethod
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


    def get_value(self, obs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        obs = self._obs_to_tensor(obs)

        with torch.no_grad():
            _, value = self.architecture(obs)

        return value.cpu().numpy()