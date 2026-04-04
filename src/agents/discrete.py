from agents.base import BaseAgent
import torch
from networks import architectures

class DiscreteAgent(BaseAgent):
    def __init__(self, network: architectures.BaseArchitecture):
        self.network = network


    def get_action_and_value(self, obs, action=None, deterministic=False):
        
        logits, value = self.network(obs)
        
        dist = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
                
        return action, dist.log_prob(action), dist.entropy()