from abc import ABC, abstractmethod

from agents.base import BaseAgent
from envs.base_env import BaseEnv


class BaseRunner(ABC):
    def __init__(self, env: BaseEnv, agent: BaseAgent):
        self.env = env
        self.agent = agent


    @abstractmethod
    def run():
        pass


    