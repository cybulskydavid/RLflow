import numpy as np
from agents.base import BaseAgent
from runners.base_runner import BaseRunner
from buffers.base_buffer import BaseBuffer
from configs.runner import OnPolicyRunnerConfig
from envs.base_env import BaseEnv


class OnPolicyRunner(BaseRunner):
    def __init__(self, 
                 env: BaseEnv, 
                 agent: BaseAgent, 
                 buffer: BaseBuffer,
                 cfg: OnPolicyRunnerConfig):
        super().__init__(env, agent)
        self.buffer = buffer

    
    def run(self):
        obs, _ = self.env.reset()

        while self.buffer.full is False:
            action, log_action_prob, value = self.agent.get_action(obs)
            print(action)
            observation, reward, terminated, truncated, info = self.env.step(action)

            self.buffer.add(observation, action, reward, truncated or terminated, log_action_prob, value)

            if terminated:
                self.env.reset()