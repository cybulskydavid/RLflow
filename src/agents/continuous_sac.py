from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from agents.base import BaseAgent
from torch import Tensor

class SACAgent(BaseAgent):
    def get_action(self, state: np.ndarray, deterministic=False) -> np.ndarray:
        state = torch.as_tensor(state, dtype=torch.float32)

        self.architecture.eval()

        with torch.no_grad():
            mu, std = self.architecture.actor(state)
            
            dist = torch.distributions.Normal(mu, std)

            if deterministic:
                action = torch.tanh(mu)
            else:
                action = torch.tanh(dist.sample())

            return action.detach().cpu().numpy()

    def get_action_and_log_prob(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        self.architecture.train()
        
        mu, std = self.architecture.actor(state)
        dist = torch.distributions.Normal(mu, std)

        action_raw = dist.rsample()
        action = torch.tanh(action_raw)

        log_prob = dist.log_prob(action_raw).sum(axis=-1, keepdim=True)
        correction = 2 * (np.log(2) - action_raw - F.softplus(-2 * action_raw))
        log_prob -= correction.sum(axis=-1, keepdim=True)

        return action, log_prob

    def get_q_values(self, state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        self.architecture.train()
        q1 = self.architecture.critic1(torch.cat((state, action), dim=-1))
        q2 = self.architecture.critic2(torch.cat((state, action), dim=-1))
        return q1, q2