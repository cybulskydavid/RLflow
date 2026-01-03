from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass
class EnvSpec:
    obs_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]
    action_type: str


class BaseEnv(ABC):
    @property
    @abstractmethod
    def spec(self) -> EnvSpec:
        pass

    @abstractmethod
    def reset(self, seed: int = None)-> Tuple[Any, Dict[str, Any]]:
        pass


    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        pass


    @abstractmethod
    def close(self) -> None:
        pass