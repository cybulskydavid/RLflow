from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from networks import extractors, heads
from torch import nn, Tensor


class BaseArchitecture(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, observations:Tensor) -> Tuple[Any, Tensor]:
        pass


class SharedArchitecture(BaseArchitecture):
    def __init__(self, 
                 shared_extractor: extractors.Extractor, 
                 actor_head: heads.VectorHead, 
                 critic_head: heads.ScalarHead):
        super().__init__()
        self.shared_extractor = shared_extractor
        self.actor_head = actor_head
        self.critic_head = critic_head

    
    def forward(self, observations: Tensor) -> Tuple[Any, Tensor]:
        features = self.shared_extractor(observations)
        return self.actor_head(features), self.critic_head(features)
    

class SeparatedArchitecture(BaseArchitecture):
    def __init__(self, 
                 state_dim : int,
                 action_dim : int,
                 actor_extractor: extractors.Extractor, 
                 actor_head: heads.VectorHead,
                 critic_extractor: extractors.Extractor,
                 critic_head: heads.ScalarHead):
        super().__init__()
        self.actor_extractor = actor_extractor(state_dim)
        self.actor_head = actor_head(self.actor_extractor.feature_dim, action_dim)
        self.critic_extractor = critic_extractor(state_dim)
        self.critic_head = critic_head(self.critic_extractor.feature_dim)
    

    def forward(self, observations: Tensor) -> Tuple[Any, Tensor]:
        actor_features = self.actor_extractor(observations)
        critic_features = self.critic_extractor(observations)
        return self.actor_head(actor_features), self.critic_head(critic_features)
        

class SACArchitecture(BaseArchitecture):
    def __init__(self, 
                 state_dim : int,
                 action_dim : int,
                 actor_extractor: extractors.Extractor, 
                 actor_head: heads.SACHead, 
                 critic_extractor1: extractors.Extractor,
                 critic_extractor2: extractors.Extractor,
                 critic_head1: heads.ScalarHead,
                 critic_head2: heads.ScalarHead):
        super().__init__()
        self.actor_extractor = actor_extractor(state_dim)
        self.actor_head = actor_head(self.actor_extractor.feature_dim, action_dim)
        self.critic_extractor1 = critic_extractor1(state_dim + action_dim)
        self.critic_head1 = critic_head1(self.critic_extractor1.feature_dim)
        self.critic_extractor2 = critic_extractor2(state_dim + action_dim)
        self.critic_head2 = critic_head2(self.critic_extractor2.feature_dim)
        self.actor = nn.Sequential(self.actor_extractor, self.actor_head)
        self.critic1 =  nn.Sequential(self.critic_extractor1, self.critic_head1)
        self.critic2 =  nn.Sequential(self.critic_extractor2, self.critic_head2)


    def forward(self, states: Tensor, actions: Tensor) -> Tuple[Any, Tensor, Tensor]:
        critic_input = torch.cat([states, actions], dim=-1)
        return self.actor(states), self.critic1(critic_input), self.critic2(critic_input)