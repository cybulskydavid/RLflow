from abc import ABC, abstractmethod

from agents.base import BaseAgent
from buffers.base_buffer import BaseBuffer
from envs.base_env import BaseEnv


class BaseRunner(ABC):
    def __init__(self, 
                 env: BaseEnv, 
                 agent: BaseAgent, 
                 buffer: BaseBuffer):
        self.env = env
        self.agent = agent
        self.buffer = buffer


    @abstractmethod
    def run():
        pass


    