import torch
from agents.base import BaseAgent
from algorithms.base_algorithm import BaseAlgorithm
from buffers.rollout_buffer import RolloutBuffer


class PPO(BaseAlgorithm):
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.eps_clip = 0.1
        self.c_value = 0.5
        self.c_entropy = 0.1
        self.batch_size = 64
        self.k_epochs = 80


    def update(self, agent: BaseAgent, buffer: RolloutBuffer):
        agent.architecture.train()

        avg_loss = 0
        avg_p_loss = 0
        avg_v_loss = 0
        avg_entropy = 0

        for epoch in range(self.k_epochs):
            for old_states, old_actions, old_log_probs, old_values, advantages, returns in buffer.get_generator(self.batch_size):
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

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.item()
                avg_p_loss += policy_loss.item()
                avg_v_loss += value_loss.item()
                avg_entropy += dist_entropy.mean().item()

        total_updates = self.k_epochs * (buffer.buffer_size / self.batch_size)
        
        return {
            "loss": avg_loss / total_updates,
            "policy_loss": avg_p_loss / total_updates,
            "value_loss": avg_v_loss / total_updates,
            "entropy": avg_entropy / total_updates
        }