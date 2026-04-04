import numpy as np
import torch
from agents.base import BaseAgent
from buffers.replay_buffer import ReplayBuffer
from runners.base_runner import BaseRunner
from configs.runner import OffPolicyRunnerConfig 
from envs.base_env import BaseEnv
from utils.tensor_board_logger import TensorBoardLogger


class OffPolicyRunner(BaseRunner):
    def __init__(self, 
                 env: BaseEnv, 
                 agent: BaseAgent, 
                 buffer: ReplayBuffer,
                 logger: TensorBoardLogger,
                 seed: int,
                 start_steps: int,
                 steps_per_run: int):
        super().__init__(env, agent, buffer)
        self.logger = logger

        state, _ = self.env.reset(seed=seed)
        self.state = state
        
        self.steps_per_run = steps_per_run
        self.start_steps = start_steps

        self.current_ep_reward = 0
        self.current_ep_length = 0
        self.episode = 1
        self.global_step = 0


    def run(self):
        for _ in range(self.steps_per_run):
            
            if self.global_step < self.start_steps:
                action = self.env.env.action_space.sample()
            else:
                action = self.agent.get_action(self.state, deterministic=False)

            action = torch.as_tensor(action).float()

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = truncated or terminated

            self.current_ep_reward += reward
            self.current_ep_length += 1
            self.global_step += 1

            self.buffer.add(
                state=self.state, 
                action=action, 
                reward=reward, 
                done=done, 
                next_state=next_state
            )

            self.state = next_state

            if done:
                next_state, _ = self.env.reset()
                self.state = next_state
                
                self.logger.log_metrics({
                    "episode/reward": self.current_ep_reward,
                    "episode/length": self.current_ep_length
                }, self.episode)

                self.current_ep_reward = 0
                self.current_ep_length = 0
                self.episode += 1

    def run_inference(self):
        done = False
        state, _ = self.env.reset(np.random.randint(0, 10000))
        total_reward = 0
        while not done:
            action = torch.tensor(self.agent.get_action(state, deterministic=True)).float()
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = truncated or terminated
            total_reward += reward
            state = next_state
        print(f"Total Reward during Inference: {total_reward}")