import numpy as np
from agents.base import BaseAgent
from buffers.rollout_buffer import RolloutBuffer
from runners.base_runner import BaseRunner
from buffers.base_buffer import BaseBuffer
from configs.runner import OnPolicyRunnerConfig
from envs.base_env import BaseEnv
from utils.tensor_board_logger import TensorBoardLogger


class OnPolicyRunner(BaseRunner):
    def __init__(self, 
                 env: BaseEnv, 
                 agent: BaseAgent, 
                 buffer: RolloutBuffer, 
                 logger: TensorBoardLogger):
        super().__init__(env, agent, buffer)
        state, _ = self.env.reset(42)
        self.state = state
        self.current_ep_reward = 0
        self.current_ep_length = 0
        self.time_step = 0
        self.episode = 1
        self.logger = logger

    
    def run(self):
        done = False
        while True:
            if self.buffer.is_full:
                _, _, last_value = self.agent.get_action(self.state)
                self.buffer.compute_gae(last_value, done)
                break
            action, log_prob, value = self.agent.get_action(self.state)
            action_clipped = np.clip(action.numpy(), self.env.spec.action_min, self.env.spec.action_max)
            next_state, reward, terminated, truncated, info = self.env.step(action_clipped)
            done = truncated or terminated

            self.current_ep_reward += reward
            self.current_ep_length += 1
            self.time_step += 1

            self.buffer.add(self.state, action, reward, done, log_prob, value)
            self.state = next_state

            if done:
                next_state, _ = self.env.reset()
                self.state = next_state

                metrics = {
                    "episode/reward": self.current_ep_reward,
                    "episode/length": self.current_ep_length
                }

                self.logger.log_metrics(metrics, self.episode)
                self.current_ep_reward = 0
                self.current_ep_length = 0
                self.episode += 1

    def run_inference(self):
        done = False
        state, _ = self.env.reset(np.random.randint(0, 10000))
        total_reward = 0
        while not done:
            action, _, _ = self.agent.get_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = truncated or terminated
            total_reward += reward
            state = next_state
        print(f"Total Reward during Inference: {total_reward}")