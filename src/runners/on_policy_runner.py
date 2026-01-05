import numpy as np
from agents.base import BaseAgent
from buffers.rollout_buffer import RolloutBuffer
from runners.base_runner import BaseRunner
from buffers.base_buffer import BaseBuffer
from configs.runner import OnPolicyRunnerConfig
from envs.base_env import BaseEnv


class OnPolicyRunner(BaseRunner):
    def __init__(self, 
                 env: BaseEnv, 
                 agent: BaseAgent, 
                 buffer: RolloutBuffer,
                 cfg: OnPolicyRunnerConfig):
        super().__init__(env, agent, buffer)
        self.config = cfg

    
    def run(self):
        obs, _ = self.env.reset()

        while self.buffer.full is False:
            action, log_action_prob, value = self.agent.get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = truncated or terminated

            self.buffer.add(obs, action, reward, done, log_action_prob, value)

            if terminated:
                self.env.reset()

        action, log_action_prob, value = self.agent.get_action(obs)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = truncated or terminated

        self.buffer.compute_gae(value, done)