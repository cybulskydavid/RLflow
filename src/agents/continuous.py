from typing import Tuple
import numpy as np
from agents.base import BaseAgent
import torch
from torch import Tensor


class ContinuousAgent(BaseAgent):
    def get_action(self, state: np.ndarray, deterministic=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = torch.as_tensor(state, dtype=torch.float32)

        self.architecture.eval()

        with torch.no_grad():
            (mu, std), value = self.architecture(state)
            
            dist = torch.distributions.Normal(mu, std)

            if deterministic:
                action = mu
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(axis=-1).unsqueeze(-1)

            return (
                action.detach(),
                log_prob.detach(),
                value.detach()
            )


    def evaluate_actions(self, states: np.ndarray, actions: np.ndarray) -> Tuple[Tensor, Tensor, Tensor]:
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32) 

        self.architecture.train()

        (mu, std), values = self.architecture(states)
        
        dist = torch.distributions.Normal(mu, std)
        
        log_probs = dist.log_prob(actions).sum(axis=-1).unsqueeze(-1)
        
        entropy = dist.entropy().sum(axis=-1).unsqueeze(-1)
        
        return (
            values, 
            log_probs, 
            entropy
        )