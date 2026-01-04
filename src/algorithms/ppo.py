import torch
from agents.base import BaseAgent
from algorithms.base_algorithm import BaseAlgorithm
from buffers.rollout_buffer import RolloutBuffer


class PPO(BaseAlgorithm):
    def __init__(self, optimizer):
        super().__init__()
        self.eps_clip = 0.1
        self.c_value = 0.5
        self.c_enthropy = 0.1


    def update(self, agent: BaseAgent, buffer: RolloutBuffer, optimizer):
        agent.architecture.train()

        for old_states, old_actions, old_log_probs, old_values, advantages, returns in buffer.get_generator():
            values, log_probs, dist_entropy = agent.evaluate_actions(old_states, old_actions)
                
            values = values.squeeze()
            ratios = torch.exp(log_probs - old_log_probs.detach())
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.mean(torch.square(values - returns))
            entropy_loss = -dist_entropy.mean()

            loss = policy_loss + (self.c_value * value_loss) + (self.c_entropy * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()