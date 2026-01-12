from typing import Any
import torch
from agents.base import BaseAgent
from algorithms.base_algorithm import BaseAlgorithm
from algorithms.builders import build_optimizers_parameters
from buffers.rollout_buffer import RolloutBuffer


class PPO(BaseAlgorithm):
    def __init__(self, 
                 eps_clip: float,
                 c_value: float,
                 c_entropy: float,
                 batch_size: int,
                 n_epochs: int,
                 agent: BaseAgent,
                 optimizer_cls: Any,
                 optimizer_params: Any):
        super().__init__()
        self.eps_clip = eps_clip
        self.c_value = c_value
        self.c_entropy = c_entropy
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        optimizer_params = build_optimizers_parameters(agent.architecture, optimizer_params)
        self.optimizer = optimizer_cls(params=optimizer_params)

        self.mse_loss = torch.nn.MSELoss()


    def update(self, agent: BaseAgent, buffer: RolloutBuffer):
        agent.architecture.train()

        avg_loss = 0
        avg_p_loss = 0
        avg_v_loss = 0
        avg_entropy = 0
        avg_kl = 0
        avg_clip_frac = 0

        for epoch in range(self.n_epochs):
            for old_states, old_actions, old_log_probs, advantages, returns, old_values in buffer.get_generator(self.batch_size):
                values, log_probs, dist_entropy = agent.evaluate_actions(old_states, old_actions)
                values = values.squeeze()
                returns = returns.squeeze()   

                log_ratio = log_probs - old_log_probs.detach()
                ratios = torch.exp(log_probs - old_log_probs.detach())
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

                with torch.no_grad():
                    approx_kl = ((ratios - 1) - log_ratio).mean()
                    clipped = ratios.gt(1 + self.eps_clip) | ratios.lt(1 - self.eps_clip)
                    clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean()
                
                avg_kl += approx_kl.item()
                avg_clip_frac += clip_fraction.item()

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.mse_loss(values, returns)
                entropy_loss = -dist_entropy.mean()

                loss = policy_loss + (self.c_value * value_loss) + (self.c_entropy * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.item()
                avg_p_loss += policy_loss.item()
                avg_v_loss += value_loss.item()
                avg_entropy += dist_entropy.mean().item()

        total_updates = self.n_epochs * (buffer.buffer_size / self.batch_size)
        
        return {
            "loss/total": avg_loss / total_updates,
            "loss/policy": avg_p_loss / total_updates,
            "loss/value": avg_v_loss / total_updates,
            "entropy": avg_entropy / total_updates,
            "policy/kl_divergence": avg_kl / total_updates,
            "policy/clip_fraction": avg_clip_frac / total_updates,
        }