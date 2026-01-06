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
        obs, _ = self.env.reset()
        self.state = obs
        self.current_ep_reward = 0
        self.time_step = 0
        self.episode = 1

    
    def run(self):
        while self.buffer.full is False:
            action, log_action_prob, value = self.agent.get_action(self.state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = truncated or terminated

            self.current_ep_reward += reward
            self.time_step += 1

            self.buffer.add(self.state, action, reward, done, log_action_prob, value)
            self.state = next_state

            if done:
                obs, _ = self.env.reset()
                self.state = obs
                
                print(f"Epizod: {self.episode} | Wynik: {self.current_ep_reward:.2f} | Kroki: {self.time_step}")
                self.current_ep_reward = 0
                self.time_step = 0
                self.episode += 1


        action, log_action_prob, value = self.agent.get_action(self.state)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = truncated or terminated

        # self.obs = obs

        self.buffer.compute_gae(value, done)