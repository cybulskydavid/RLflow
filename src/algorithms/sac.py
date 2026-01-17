import torch
import torch.nn.functional as F
from copy import deepcopy
from agents.base import BaseAgent
from algorithms.base_algorithm import BaseAlgorithm
from buffers.replay_buffer import ReplayBuffer
from configs import agent

class SAC(BaseAlgorithm):
    def __init__(self, 
                 actor_optimizer: torch.optim.Optimizer, 
                 critic_optimizer: torch.optim.Optimizer,
                 alpha_optimizer: torch.optim.Optimizer,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 autotune: bool = True,
                 target_entropy: float = -2.0,
                 batch_size: int = 256):
        
        super().__init__()
        
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.alpha_optimizer = alpha_optimizer
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        self.autotune = autotune
        if self.autotune:
            self.target_entropy = target_entropy
            self.log_alpha = torch.tensor(torch.log(torch.tensor(alpha)), requires_grad=True, device='cpu') 
        else:
            self.alpha = alpha

        self.target_critic1 = None
        self.target_critic2 = None


    def update(self, agent: BaseAgent, buffer: ReplayBuffer):
        if self.target_critic1 is None:
            self.target_critic1 = deepcopy(agent.architecture.critic1)
            self.target_critic2 = deepcopy(agent.architecture.critic2)
            for p in self.target_critic1.parameters():
                p.requires_grad = False
            for p in self.target_critic2.parameters():
                p.requires_grad = False

        states, actions, rewards, next_states, dones = buffer.sample(self.batch_size)

        with torch.no_grad():
            next_state_actions, next_state_log_pi = agent.get_action_and_log_prob(next_states)

            q1_next, q2_next = self.target_critic1(torch.cat((next_states, next_state_actions), dim=-1)), self.target_critic2(torch.cat((next_states, next_state_actions), dim=-1))
            min_q_next = torch.min(q1_next, q2_next)
            
            alpha = self.log_alpha.exp().item() if self.autotune else self.alpha
            
            target_q = rewards + (1 - dones) * self.gamma * (min_q_next - alpha * next_state_log_pi)

        q1, q2 = agent.get_q_values(states, actions)
        
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi = agent.get_action_and_log_prob(states)
        q1_pi, q2_pi = agent.get_q_values(states, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((alpha * log_pi) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = 0
        if self.autotune:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        for param, target_param in zip(agent.architecture.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(agent.architecture.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        with torch.no_grad():
            entropy = -log_pi.mean().item()
            
            q_min = q1.min().item()
            q_max = q1.max().item()
            q_std = q1.std().item()

            action_abs_mean = pi.abs().mean().item()

        return {
            "loss/critic": critic_loss.item(),
            "loss/actor": actor_loss.item(),
            "loss/alpha": alpha_loss.item() if self.autotune else 0.0,
            "train/alpha": alpha,
            "train/entropy": entropy,
            "train/target_entropy": self.target_entropy,
            "critic/q_mean": q1.mean().item(),
            "critic/q_min": q_min,
            "critic/q_max": q_max,
            "critic/q_std": q_std,
            "actor/action_abs_mean": action_abs_mean
        }
        
    
    def get_state_dict(self, agent):
        state = {
            "actor": agent.architecture.actor.state_dict(),
            "critic1": agent.architecture.critic1.state_dict(),
            "critic2": agent.architecture.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(), 
            "alpha_optim": self.alpha_optimizer.state_dict() if self.autotune else None
        }
        
        return state


    def load_state_dict(self, agent, state_dict):
        if self.target_critic1 is None:
            self.target_critic1 = deepcopy(agent.architecture.critic1)
        if self.target_critic2 is None:
            self.target_critic2 = deepcopy(agent.architecture.critic2)
        
        agent.architecture.actor.load_state_dict(state_dict["actor"])
        agent.architecture.critic1.load_state_dict(state_dict["critic1"])
        agent.architecture.critic2.load_state_dict(state_dict["critic2"])
        self.target_critic1.load_state_dict(state_dict["target_critic1"])
        self.target_critic2.load_state_dict(state_dict["target_critic2"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optim"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optim"])
    
        if "log_alpha" in state_dict:
            with torch.no_grad():
                self.log_alpha.copy_(state_dict["log_alpha"])
        if self.autotune and state_dict["alpha_optim"] is not None:
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optim"])