from agents.base import BaseAgent
import torch


class ContinuousAgent(BaseAgent):
    def get_action(self, obs, deterministic=False):
        obs = self._obs_to_tensor(obs)
        
        with torch.no_grad():
            (mu, std), value = self.architecture(obs)
            
            dist = torch.distributions.Normal(mu, std)
            
            if deterministic:
                action = mu
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)

        return (
            action.cpu().numpy(), 
            log_prob.cpu().numpy(), 
            value.cpu().numpy()
        )

    def evaluate_actions(self, obs, actions):
        (mu, std), values = self.architecture(obs)
        
        dist = torch.distributions.Normal(mu, std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        
        return values, log_probs, entropy