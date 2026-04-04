from abc import ABC, abstractmethod

from agents.base import BaseAgent
from buffers.base_buffer import BaseBuffer


class BaseAlgorithm(ABC):
    @abstractmethod
    def update(agent: BaseAgent, buffer: BaseBuffer):
        pass